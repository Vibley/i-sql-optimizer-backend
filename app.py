import os, json, re, datetime
from typing import List, Optional, Tuple
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sqlparse

# --- Proxy cleanup ---
for k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "OPENAI_PROXY"]:
    os.environ.pop(k, None)

# Front-end origin(s) — set this env var on Render to your GitHub Pages site
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "https://vibley.github.io")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # unused in this static-only build

app = FastAPI(title="AI SQL Optimizer Backend", version="1.4.1")

# CORS: allow exact origins (add localhost for dev)
allowed_origins = [FRONTEND_ORIGIN]
for local in ("http://localhost:3000", "http://localhost:5173"):
    allowed_origins.append(local)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,               # not using cookies/auth
    allow_methods=["GET", "POST", "OPTIONS"],
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

def _month_range(yyyy: int, mm: int) -> Tuple[str, str]:
    start = datetime.date(yyyy, mm, 1)
    nxt = datetime.date(yyyy + (1 if mm == 12 else 0), (1 if mm == 12 else mm + 1), 1)
    return (start.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d"))

# ---------- Static rules + concrete rewrites ----------
def static_rules(sql: str):
    """
    Returns:
      findings: List[str]
      rewrite_out: Optional[str]
      index_recs: List[str]           # short, readable suggestions
      index_script: Optional[str]     # full executable T-SQL block
      risks: List[str]
    """
    findings, guidance_lines, risks = [], [], []
    index_recs: List[str] = []
    sql_norm = sql.strip()
    sql_compact = re.sub(r"\s+", " ", sql_norm, flags=re.MULTILINE).upper()

    # Simple patterns
    if re.search(r"\bSELECT\s+\*\b", sql_compact):
        findings.append("Avoid SELECT *. Project only required columns.")
        risks.append("Extra I/O and wider rows reduce buffer cache efficiency.")
        guidance_lines.append("-- Replace SELECT * with only required columns.")

    if re.search(r"LIKE\s+['\"]%[^'\"]+['\"]", sql_compact):
        findings.append("Leading wildcard LIKE prevents index seeks.")
        guidance_lines.append("-- Consider full-text index (CONTAINS) or trigram search.")
        risks.append("Full scans on large tables can be expensive.")

    if re.search(r"WHERE\s+.*\b(YEAR|MONTH|DAY|DATE|DATEADD|DATEDIFF|SUBSTRING|CAST|CONVERT)\s*\(", sql_compact) \
       or re.search(r"::\s*DATE\b", sql_norm, flags=re.IGNORECASE):
        findings.append("Non-sargable predicate (function/cast on column) can block index seeks.")
        guidance_lines.append("-- Prefer sargable range predicates over functions/casts on columns.")

    if re.search(r"\bWHERE\b.*\bOR\b", sql_compact):
        findings.append("OR conditions may reduce index usage; consider UNION ALL or indexed computed columns.")

    if "ORDER BY" in sql_compact and "JOIN" not in sql_compact:
        findings.append("ORDER BY detected; ensure index supports ORDER BY key(s).")

    if "WHERE" not in sql_compact and "JOIN" in sql_compact:
        findings.append("JOIN without WHERE may explode rows; verify join predicates and filters.")

    # ----- Concrete rewrites (apply in place where safe) -----
    working = sql_norm
    concrete_changes = 0

    # YEAR(col) = YYYY  --> date range
    for m in re.finditer(r"\bYEAR\s*\(\s*(?P<col>[A-Za-z0-9_\.\[\]]+)\s*\)\s*=\s*(?P<yyyy>19\d{2}|20\d{2})", working, flags=re.IGNORECASE):
        col = m.group("col"); yyyy = int(m.group("yyyy"))
        start = f"{yyyy:04d}-01-01"; end = f"{(yyyy+1):04d}-01-01"
        rng = f"{col} >= '{start}' AND {col} < '{end}'"
        working = re.sub(r"\bYEAR\s*\(\s*"+re.escape(col)+r"\s*\)\s*=\s*"+str(yyyy), rng, working, count=1, flags=re.IGNORECASE)
        concrete_changes += 1

    # DATE(col) = 'YYYY-MM-DD'  --> day range
    for m in re.finditer(r"\bDATE\s*\(\s*(?P<col>[A-Za-z0-9_\.\[\]]+)\s*\)\s*=\s*'(?P<d>\d{4}-\d{2}-\d{2})'", working, flags=re.IGNORECASE):
        col = m.group("col"); d = m.group("d"); d2 = _iso_next_day(d)
        rng = f"{col} >= '{d}' AND {col} < '{d2}'"
        working = re.sub(r"\bDATE\s*\(\s*"+re.escape(col)+r"\s*\)\s*=\s*'"+re.escape(d)+r"'", rng, working, count=1, flags=re.IGNORECASE)
        concrete_changes += 1

    # CAST(col AS DATE) = 'YYYY-MM-DD'  --> day range
    for m in re.finditer(r"\bCAST\s*\(\s*(?P<col>[A-Za-z0-9_\.\[\]]+)\s+AS\s+DATE\s*\)\s*=\s*'(?P<d>\d{4}-\d{2}-\d{2})'", working, flags=re.IGNORECASE):
        col = m.group("col"); d = m.group("d"); d2 = _iso_next_day(d)
        rng = f"{col} >= '{d}' AND {col} < '{d2}'"
        working = re.sub(r"\bCAST\s*\(\s*"+re.escape(col)+r"\s+AS\s+DATE\s*\)\s*=\s*'"+re.escape(d)+r"'", rng, working, count=1, flags=re.IGNORECASE)
        concrete_changes += 1

    # Postgres: col::DATE = 'YYYY-MM-DD'  --> day range
    for m in re.finditer(r"\b(?P<col>[A-Za-z0-9_\.\[\]]+)\s*::\s*DATE\s*=\s*'(?P<d>\d{4}-\d{2}-\d{2})'", working, flags=re.IGNORECASE):
        col = m.group("col"); d = m.group("d"); d2 = _iso_next_day(d)
        rng = f"{col} >= '{d}' AND {col} < '{d2}'"
        working = re.sub(r"\b"+re.escape(col)+r"\s*::\s*DATE\s*=\s*'"+re.escape(d)+r"'", rng, working, count=1, flags=re.IGNORECASE)
        concrete_changes += 1

    # MONTH(col)=M + YEAR(col)=YYYY  --> guidance only
    mon = re.search(r"\bMONTH\s*\(\s*(?P<c1>[A-Za-z0-9_\.\[\]]+)\s*\)\s*=\s*(?P<mm>1[0-2]|0?[1-9])", sql_norm, flags=re.IGNORECASE)
    yr  = re.search(r"\bYEAR\s*\(\s*(?P<c2>[A-Za-z0-9_\.\[\]]+)\s*\)\s*=\s*(?P<yyyy>19\d{2}|20\d{2})", sql_norm, flags=re.IGNORECASE)
    if mon and yr and mon.group("c1").lower() == yr.group("c2").lower():
        c = mon.group("c1"); mm = int(mon.group("mm")); yyyy = int(yr.group("yyyy"))
        start, end = _month_range(yyyy, mm)
        guidance_lines.append(
            f"-- Replace MONTH({c})={mm} AND YEAR({c})={yyyy} with range:\n"
            f"-- {c} >= '{start}' AND {c} < '{end}'"
        )

    # ----- Index key guess (equality predicates) -----
    # Collect components so we can build both short recommendations AND a robust script.
    suggested_scripts: List[Tuple[str, List[str]]] = []  # (table, cols)

    m = re.findall(r"\b([A-Z_][A-Z0-9_\.]+)\s*=\s*[@:\w'\-]+", sql_compact)
    if m:
        cols = [col.split(".")[-1].lower() for col in m]
        cols = list(dict.fromkeys(cols))[:3]

        # Extract table: prefer FROM, fallback to JOIN; clean quotes/brackets; lower-case
        tbl_match = re.search(r"\bFROM\s+([A-Za-z0-9_\.\[\]\"`]+)", sql, flags=re.IGNORECASE) \
                 or re.search(r"\bJOIN\s+([A-Za-z0-9_\.\[\]\"`]+)", sql, flags=re.IGNORECASE)
        table_name = None
        if tbl_match:
            table_name = re.sub(r'[\[\]"`]', "", tbl_match.group(1)).lower()

        if table_name and cols:
            index_recs.append(f"create index ix_{cols[0]}_suggested on {table_name} ({', '.join(cols)});")
            suggested_scripts.append((table_name, cols))

    # ----- Build full index script from components (no regex reverse-parsing) -----
    index_script = None
    if suggested_scripts:
        lines = ["-- Suggested indexes"]
        for table_name, cols in suggested_scripts:
            idx_name = f"ix_{cols[0]}_suggested"
            col_list = ", ".join(cols)
            lines.append(f"IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = '{idx_name}' AND object_id = OBJECT_ID('{table_name}'))")
            lines.append("BEGIN")
            lines.append(f"    CREATE INDEX {idx_name} ON {table_name} ({col_list});")
            lines.append("END")
            lines.append("GO")
        index_script = "\n".join(lines)

    # ----- Choose rewrite output -----
    if concrete_changes > 0:
        rewrite_out = working
    else:
        rewrite_out = "\n".join(guidance_lines) if guidance_lines else None

    return findings, rewrite_out, index_recs, index_script, risks

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

    findings, rewrite_sql, index_recs, index_script, risks = static_rules(sql_fmt)

    return AnalyzeResponse(
        summary="Static analysis completed.",
        findings=findings or ["No obvious issues detected."],
        rewrite_sql=rewrite_sql or "No query rewrite suggestions were identified.",
        index_recommendations=index_recs,
        index_script=index_script,
        risks=risks,
        test_steps=[
            "Capture current plan & metrics (duration, CPU, reads).",
            "Apply one change at a time (index or rewrite).",
            "Compare estimated vs actual plans; validate row estimates.",
            "Benchmark on prod-like data; check regressions.",
        ],
    )
