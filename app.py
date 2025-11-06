import os, json, re, datetime
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sqlparse

# --- Proxy cleanup ---
for k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "OPENAI_PROXY"]:
    os.environ.pop(k, None)

ALLOW_ORIGIN = os.getenv("ALLOW_ORIGIN", "*")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="AI SQL Optimizer Backend", version="1.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOW_ORIGIN] if ALLOW_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class AnalyzeRequest(BaseModel):
    dbms: str = "sqlserver"
    sql_text: str
    plan_xml: Optional[str] = None
    context: Optional[str] = None
    version: Optional[str] = None


class AnalyzeResponse(BaseModel):
    summary: str
    findings: List[str] = Field(default_factory=list)
    rewrite_sql: Optional[str] = None
    index_recommendations: List[str] = Field(default_factory=list)
    index_script: Optional[str] = None
    risks: List[str] = Field(default_factory=list)
    test_steps: List[str] = Field(default_factory=list)


# ---------- Date helpers ----------
def _iso_next_day(d: str) -> str:
    y, m, day = map(int, d.split("-"))
    dt = datetime.date(y, m, day) + datetime.timedelta(days=1)
    return dt.strftime("%Y-%m-%d")


def _month_range(yyyy: int, mm: int):
    start = datetime.date(yyyy, mm, 1)
    if mm == 12:
        nxt = datetime.date(yyyy + 1, 1, 1)
    else:
        nxt = datetime.date(yyyy, mm + 1, 1)
    return (start.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d"))


# ---------- Static rules ----------
def static_rules(sql: str):
    findings, guidance_lines, index_recs, risks = [], [], [], []
    sql_norm = sql.strip()
    sql_compact = re.sub(r"\s+", " ", sql_norm, flags=re.MULTILINE).upper()

    # Simple patterns
    if re.search(r"\bSELECT\s+\*\b", sql_compact):
        findings.append("Avoid SELECT *. Project only required columns.")
        risks.append("Extra I/O and wider rows reduce buffer cache efficiency.")
    if re.search(r"LIKE\s+['\"]%[^'\"]+['\"]", sql_compact):
        findings.append("Leading wildcard LIKE prevents index seeks.")
    if re.search(r"\bWHERE\s+.*(YEAR|MONTH|DAY|DATEADD|DATEDIFF|SUBSTRING|CAST|CONVERT)\s*\(", sql_compact):
        findings.append("Non-sargable predicate (function/cast on column) can block index seeks.")
    if re.search(r"\bWHERE\b.*\bOR\b", sql_compact):
        findings.append("OR conditions may reduce index usage.")
    if "ORDER BY" in sql_compact and "JOIN" not in sql_compact:
        findings.append("ORDER BY detected; ensure index supports ORDER BY key(s).")

    # Extract equality columns
    m = re.findall(r"\b([A-Z_][A-Z0-9_\.]+)\s*=\s*[@:\w'\-]+", sql_compact)
    if m:
        cols = [col.split(".")[-1].lower() for col in m]
        cols = list(dict.fromkeys(cols))[:3]

        # Extract table
        tbl_match = re.search(r"\bFROM\s+([A-Za-z0-9_\.\[\]\"`]+)", sql, flags=re.IGNORECASE)
        if not tbl_match:
            tbl_match = re.search(r"\bJOIN\s+([A-Za-z0-9_\.\[\]\"`]+)", sql, flags=re.IGNORECASE)
        table_name = None
        if tbl_match:
            table_name = tbl_match.group(1)
            table_name = re.sub(r'[\[\]"`]', "", table_name).lower()

        if table_name and cols:
            index_recs.append(f"create index ix_{cols[0]}_suggested on {table_name} ({', '.join(cols)});")

    # Build full index script
    index_script = None
    if index_recs:
        index_script = "-- Suggested indexes\n" + "\n".join(
            f"CREATE INDEX {re.search(r'ix_\\w+', rec).group(0)} "
            f"ON {re.search(r'on\\s+(\\S+)', rec).group(1)} "
            f"({', '.join(re.findall(r'\\((.*?)\\)', rec)[0].split(','))});"
            for rec in index_recs
        ) + "\nGO"

    return findings, None, index_recs, index_script, risks


# ---------- Health ----------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- Analyze ----------
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    sql = req.sql_text or ""
    try:
        sql_fmt = sqlparse.format(sql, keyword_case="upper", reindent=True)
    except Exception:
        sql_fmt = sql

    base_findings, base_rewrite, base_indexes, base_script, base_risks = static_rules(sql_fmt)

    return AnalyzeResponse(
        summary="Static analysis completed (OpenAI disabled for this demo).",
        findings=base_findings or ["No obvious issues detected."],
        rewrite_sql=base_rewrite or "No query rewrite suggestions were identified.",
        index_recommendations=base_indexes,
        index_script=base_script,
        risks=base_risks,
        test_steps=[
            "Capture current plan & metrics (duration, CPU, reads).",
            "Apply one change at a time (index or rewrite).",
            "Compare estimated vs actual plans; validate row estimates.",
            "Benchmark on prod-like data; check regressions.",
        ],
    )
