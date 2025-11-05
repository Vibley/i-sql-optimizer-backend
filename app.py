import os, json, re
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlparse

# ---------------- Config ----------------
ALLOW_ORIGIN = os.getenv("ALLOW_ORIGIN", "*")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------- App ----------------
app = FastAPI(title="AI SQL Optimizer Backend", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOW_ORIGIN] if ALLOW_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Models ----------------
class AnalyzeRequest(BaseModel):
    dbms: str = "sqlserver"
    sql_text: str
    plan_xml: Optional[str] = None
    context: Optional[str] = None
    version: Optional[str] = None

class AnalyzeResponse(BaseModel):
    summary: str
    findings: List[str]
    rewrite_sql: Optional[str] = None
    index_recommendations: List[str] = []
    risks: List[str] = []
    test_steps: List[str] = []

# ---------------- Static rules (fallback & pre-checks) ----------------
def static_rules(sql: str):
    findings, rewrites, index_recs, risks = [], [], [], []
    sql_norm = sql.strip()
    sql_compact = re.sub(r"\s+", " ", sql_norm, flags=re.MULTILINE).upper()

    # SELECT *
    if re.search(r"\bSELECT\s+\*\b", sql_compact):
        findings.append("Avoid SELECT *. Project only required columns.")
        risks.append("Extra I/O and wider rows reduce buffer cache efficiency.")

    # leading wildcard LIKE
    if re.search(r"LIKE\s+['\"]%[^'\"]+['\"]", sql_compact):
        findings.append("Leading wildcard LIKE prevents index seeks.")
        rewrites.append("-- Consider full-text index or trigram/contains search.")
        risks.append("Full scans on large tables can be expensive.")

    # non-sargable function on column
    if re.search(r"WHERE\s+.*\b(YEAR|MONTH|DAY|DATEADD|DATEDIFF|SUBSTRING|CAST|CONVERT)\s*\(", sql_compact):
        findings.append("Non-sargable predicate (function on column) blocks index seeks.")
        rewrites.append("-- Rewrite to range predicate on the raw column when possible.")

    # OR conditions
    if re.search(r"\bWHERE\b.*\bOR\b", sql_compact):
        findings.append("OR conditions may reduce index usage; consider UNION ALL or indexed computed columns.")

    # ORDER BY without JOIN
    if "ORDER BY" in sql_compact and "JOIN" not in sql_compact:
        findings.append("ORDER BY detected; ensure index supports ORDER BY key(s).")

    # JOIN without WHERE
    if "WHERE" not in sql_compact and "JOIN" in sql_compact:
        findings.append("JOIN without WHERE may explode rows; verify join predicates and filters.")

    # Guess composite index keys from equality predicates like t.Col = @p
    m = re.findall(r"\b([A-Z_][A-Z0-9_\.]+)\s*=\s*[@:\w'\-]+", sql_compact)
    if m:
        cols = []
        for col in m:
            cols.append(col.split(".")[-1])
        cols = list(dict.fromkeys(cols))[:3]
        if cols:
            index_recs.append(f"CREATE INDEX IX_Suggested ON <YourTable> ({', '.join(cols)});")

    return findings, ("\n".join(rewrites) if rewrites else None), index_recs, risks

# ---------------- Health ----------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------- Analyze ----------------
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    sql = req.sql_text or ""
    try:
        sql_fmt = sqlparse.format(sql, keyword_case="upper", reindent=True)
    except Exception:
        sql_fmt = sql

    base_findings, base_rewrite, base_indexes, base_risks = static_rules(sql_fmt)

    # If no key, return static analysis only
    if not OPENAI_API_KEY:
        return AnalyzeResponse(
            summary="Static analysis completed (OpenAI not configured).",
            findings=base_findings or ["No obvious issues detected by static rules."],
            rewrite_sql=base_rewrite,
            index_recommendations=base_indexes,
            risks=base_risks,
            test_steps=[
                "Capture current plan & metrics (duration, CPU, reads).",
                "Apply one change at a time (index or rewrite).",
                "Compare estimated vs actual plans; validate row estimates.",
                "Benchmark on prod-like data; check regressions."
            ],
        )

    # With OpenAI: Chat Completions (JSON mode)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        system_msg = (
            f"You are a veteran {req.dbms} performance engineer. "
            f"Return safe, actionable tuning advice. Use <YourTable> placeholders; never invent schema names."
        )
        plan = (req.plan_xml or "")[:20000]  # keep request bounded
        user_msg = f"""SQL (formatted):
