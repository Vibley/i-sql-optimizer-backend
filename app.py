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
app = FastAPI(title="AI SQL Optimizer Backend", version="1.1.1")

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

        # Build prompt without triple-quoted f-strings to avoid syntax pitfalls
        user_msg = (
            "SQL (formatted):\n"
            "```\n"
            f"{sql_fmt}\n"
            "```\n\n"
            "Context:\n"
            f"{req.context or 'n/a'}\n\n"
            "Execution plan XML (optional, truncated):\n"
            f"{plan if plan else 'n/a'}\n"
        )

        json_instructions = (
            "Return a JSON object with keys: "
            "summary (string), findings (array of strings), rewrite_sql (string), "
            "index_recommendations (array of strings), risks (array of strings), "
            "test_steps (array of strings). No extra keys or text."
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
                {"role": "user", "content": json_instructions},
            ],
        )

        llm = json.loads(resp.choices[0].message.content)

        # Merge static & LLM results (dedupe)
        def dedupe(seq):
            seen, out = set(), []
            for s in seq or []:
                if s not in seen:
                    seen.add(s); out.append(s)
            return out

        findings = dedupe((base_findings or []) + (llm.get("findings") or []))
        indexes = dedupe((base_indexes or []) + (llm.get("index_recommendations") or []))
        risks = dedupe((base_risks or []) + (llm.get("risks") or []))
        rewrite = llm.get("rewrite_sql") or base_rewrite
        summary = llm.get("summary") or "Analysis completed."
        steps = llm.get("test_steps") or [
            "Capture current plan & metrics (duration, CPU, reads).",
            "Apply one change at a time (index or rewrite).",
            "Compare estimated vs actual plans; validate row estimates.",
            "Benchmark on prod-like data; check regressions."
        ]

        return AnalyzeResponse(
            summary=summary,
            findings=findings or ["No obvious issues found."],
            rewrite_sql=rewrite,
            index_recommendations=indexes,
            risks=risks,
            test_steps=steps,
        )

    except Exception as e:
        # Fail-soft: return static analysis with reason
        base_findings.append(f"AI enhancer unavailable: {e}")
        return AnalyzeResponse(
            summary="Static analysis completed (LLM call failed).",
            findings=base_findings,
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
