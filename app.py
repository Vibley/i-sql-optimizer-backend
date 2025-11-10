import os, json, re, datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sqlparse

# -------- Proxy cleanup --------
for k in ["HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy","OPENAI_PROXY"]:
    os.environ.pop(k, None)

ALLOW_ORIGIN   = os.getenv("ALLOW_ORIGIN", "*")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REQUIRE_OPENAI = os.getenv("REQUIRE_OPENAI", "1") == "1"

app = FastAPI(title="AI SQL Optimizer Backend", version="1.7.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOW_ORIGIN] if ALLOW_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Models --------
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

# -------- Helpers --------
def _iso_next_day(d: str) -> str:
    y, m, day = map(int, d.split("-"))
    return (datetime.date(y, m, day) + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

def _month_range(yyyy: int, mm: int):
    start = datetime.date(yyyy, mm, 1)
    nxt = datetime.date(yyyy + (1 if mm == 12 else 0), 1 if mm == 12 else mm + 1, 1)
    return start.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")

def _canon_sql(s: str) -> str:
    s = (s or "")
    s = re.sub(r"```(?:sql)?", "", s, flags=re.IGNORECASE)
    s = s.replace("`", "").strip()
    if s.endswith(";"):
        s = s[:-1]
    return re.sub(r"\s+", " ", s).strip().lower()

# -------- Static analysis rules --------
def static_rules(sql: str):
    findings, guidance_lines, index_recs, risks = [], [], [], []
    sql_norm = sql.strip()
    sql_compact = re.sub(r"\s+", " ", sql_norm, flags=re.MULTILINE).upper()

    if re.search(r"\bSELECT\s+\*\b", sql_compact):
        findings.append("Avoid SELECT *. Project only required columns.")
        risks.append("Extra I/O and wider rows reduce buffer cache efficiency.")
        guidance_lines.append("-- Replace SELECT * with only required columns.")

    if re.search(r"LIKE\s+['\"]%[^'\"]+['\"]", sql_compact):
        findings.append("Leading wildcard LIKE prevents index seeks.")
        guidance_lines.append("-- Consider full-text index (CONTAINS) or trigram search.")
        risks.append("Full scans on large tables can be expensive.")

    if re.search(r"WHERE\s+.*\b(YEAR|MONTH|DAY|DATE|DATEADD|DATEDIFF|SUBSTRING|CAST|CONVERT)\s*\(", sql_compact) or re.search(r"::\s*DATE\b", sql_norm, flags=re.IGNORECASE):
        findings.append("Non-sargable predicate (function/cast on column) can block index seeks.")
        guidance_lines.append("-- Prefer sargable range predicates over functions/casts on columns.")

    if re.search(r"\bWHERE\b.*\bOR\b", sql_compact):
        findings.append("OR conditions may reduce index usage; consider UNION ALL or indexed computed columns.")

    if "ORDER BY" in sql_compact and "JOIN" not in sql_compact:
        findings.append("ORDER BY detected; ensure index supports ORDER BY key(s).")

    if "WHERE" not in sql_compact and "JOIN" in sql_compact:
        findings.append("JOIN without WHERE may explode rows; verify join predicates and filters.")

    # --- Transformations ---
    working = sql_norm
    concrete_changes = 0

    # YEAR(col)=YYYY -> range
    for m in re.finditer(r"\bYEAR\s*\(\s*(?P<col>[A-Za-z0-9_\.\[\]]+)\s*\)\s*=\s*(?P<yyyy>19\d{2}|20\d{2})", working, flags=re.IGNORECASE):
        col, yyyy = m.group("col"), int(m.group("yyyy"))
        start, end = f"{yyyy:04d}-01-01", f"{(yyyy+1):04d}-01-01"
        rng = f"{col} >= '{start}' AND {col} < '{end}'"
        working = re.sub(r"\bYEAR\s*\(\s*"+re.escape(col)+r"\s*\)\s*=\s*"+str(yyyy), rng, working, count=1, flags=re.IGNORECASE)
        concrete_changes += 1
        if "Non-sargable predicate" not in " ".join(findings):
            findings.append("Non-sargable predicate (function on column) blocks index seeks.")

    # DATE(col)='YYYY-MM-DD' -> day range
    for m in re.finditer(r"\bDATE\s*\(\s*(?P<col>[A-Za-z0-9_\.\[\]]+)\s*\)\s*=\s*'(?P<d>\d{4}-\d{2}-\d{2})'", working, flags=re.IGNORECASE):
        col, d = m.group("col"), m.group("d")
        d2 = _iso_next_day(d)
        rng = f"{col} >= '{d}' AND {col} < '{d2}'"
        working = re.sub(r"\bDATE\s*\(\s*"+re.escape(col)+r"\s*\)\s*=\s*'"+re.escape(d)+r"'", rng, working, count=1, flags=re.IGNORECASE)
        concrete_changes += 1

    # CAST(col AS DATE)='YYYY-MM-DD' -> day range
    for m in re.finditer(r"\bCAST\s*\(\s*(?P<col>[A-Za-z0-9_\.\[\]]+)\s+AS\s+DATE\s*\)\s*=\s*'(?P<d>\d{4}-\d{2}-\d{2})'", working, flags=re.IGNORECASE):
        col, d = m.group("col"), m.group("d")
        d2 = _iso_next_day(d)
        rng = f"{col} >= '{d}' AND {col} < '{d2}'"
        working = re.sub(r"\bCAST\s*\(\s*"+re.escape(col)+r"\s+AS\s+DATE\s*\)\s*=\s*'"+re.escape(d)+r"'", rng, working, count=1, flags=re.IGNORECASE)
        concrete_changes += 1

    # MONTH()+YEAR() combination guidance
    mon = re.search(r"\bMONTH\s*\(\s*(?P<c1>[A-Za-z0-9_\.\[\]]+)\s*\)\s*=\s*(?P<mm>1[0-2]|0?[1-9])", sql_norm, flags=re.IGNORECASE)
    yr  = re.search(r"\bYEAR\s*\(\s*(?P<c2>[A-Za-z0-9_\.\[\]]+)\s*\)\s*=\s*(?P<yyyy>19\d{2}|20\d{2})", sql_norm, flags=re.IGNORECASE)
    if mon and yr and mon.group("c1").lower() == yr.group("c2").lower():
        mm, yyyy = int(mon.group("mm")), int(yr.group("yyyy"))
        start, end = _month_range(yyyy, mm)
        c = mon.group("c1")
        guidance_lines.append(f"-- Replace MONTH({c})={mm} AND YEAR({c})={yyyy} with:\n-- {c} >= '{start}' AND {c} < '{end}'")
        findings.append("Non-sargable month/year predicates detected; prefer a single range on the date column.")

    # Index guess
    m = re.findall(r"\b([A-Z_][A-Z0-9_\.]+)\s*=\s*[@:\w'\-]+", sql_compact)
    table_name = None
    if m:
        cols = [col.split(".")[-1].lower() for col in m]
        cols = list(dict.fromkeys(cols))[:3]
        tbl_match = re.search(r"\bFROM\s+([A-Za-z0-9_\.\[\]\"`]+)", sql, flags=re.IGNORECASE) or \
                    re.search(r"\bJOIN\s+([A-Za-z0-9_\.\[\]\"`]+)", sql, flags=re.IGNORECASE)
        if tbl_match:
            table_name = re.sub(r'[\[\]"`]', "", tbl_match.group(1)).lower()
        if table_name and cols:
            index_recs.append(f"create index ix_{cols[0]}_suggested on {table_name} ({', '.join(cols)});")

    index_script = None
    if index_recs:
        lines = ["-- Suggested indexes"]
        for rec in index_recs:
            try:
                ix_name = re.search(r"ix_[a-z0-9_]+", rec, flags=re.IGNORECASE).group(0)
                on_tbl  = re.search(r"on\s+([^\s(]+)", rec, flags=re.IGNORECASE).group(1)
                cols_in = re.search(r"\(([^)]+)\)", rec).group(1)
                cols_clean = ", ".join([c.strip() for c in cols_in.split(",")])
                lines.append(
                    f"IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = '{ix_name}' AND object_id = OBJECT_ID('{on_tbl}'))\n"
                    f"BEGIN\n    CREATE INDEX {ix_name} ON {on_tbl} ({cols_clean});\nEND\nGO"
                )
            except Exception:
                lines.append(rec.strip() + (";" if not rec.strip().endswith(";") else ""))
                lines.append("GO")
        index_script = "\n".join(lines)

    rewrite_out = working if concrete_changes > 0 else ("\n".join(guidance_lines) if guidance_lines else None)
    return findings, rewrite_out, index_recs, index_script, risks

# -------- Health --------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/diag/openai")
def diag_openai():
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not set.")
    try:
        import httpx
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client(timeout=10.0))
        _ = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "ping"}], max_tokens=1)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI check failed: {e}")

# -------- Analyze --------
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    sql = req.sql_text or ""
    try:
        sql_fmt = sqlparse.format(sql, keyword_case="upper", reindent=True)
    except Exception:
        sql_fmt = sql

    base_findings, base_rewrite, base_indexes, base_script, base_risks = static_rules(sql_fmt)

    if not OPENAI_API_KEY:
        if REQUIRE_OPENAI:
            raise HTTPException(status_code=503, detail="OpenAI key missing.")
        return AnalyzeResponse(
            summary="Static analysis only (no LLM configured).",
            findings=base_findings or ["No major issues detected."],
            rewrite_sql=base_rewrite or "No rewrite suggestions available.",
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

    try:
        import httpx
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client(timeout=40.0))

        system_msg = f"You are an expert {req.dbms} query optimizer. Provide safe, detailed, structured recommendations."
        plan = (req.plan_xml or "")[:20000]
        user_msg = (
            "SQL Query (formatted):\n```\n"
            f"{sql_fmt}\n```\n\nContext:\n"
            f"{req.context or 'n/a'}\n\nExecution Plan XML:\n"
            f"{plan if plan else 'n/a'}"
        )
        json_instructions = (
            "Respond ONLY with a JSON object having keys: "
            "summary (string), findings (array of strings), rewrite_sql (string), "
            "index_recommendations (array of strings), risks (array of strings), test_steps (array of strings). "
            "Each key must exist, even if empty."
        )

        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.25,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
                {"role": "user", "content": json_instructions},
            ],
        )

        llm_raw = resp.choices[0].message.content.strip()
        try:
            llm = json.loads(llm_raw)
        except Exception:
            llm = {"summary": llm_raw, "findings": [], "rewrite_sql": "", "index_recommendations": [], "risks": [], "test_steps": []}

        defaults = {
            "summary": "LLM analysis completed.",
            "findings": [],
            "rewrite_sql": "",
            "index_recommendations": [],
            "risks": [],
            "test_steps": [],
        }
        for k, v in defaults.items():
            llm.setdefault(k, v)

        def dedupe(seq): return list(dict.fromkeys(seq or []))
        rewrite_raw = llm.get("rewrite_sql") or ""
        same_as_input = _canon_sql(rewrite_raw) == _canon_sql(sql_fmt)
        rewrite_final = None if same_as_input else (rewrite_raw or None)
        if not rewrite_final:
            rewrite_final = base_rewrite or "No rewrite suggestions identified."

        return AnalyzeResponse(
            summary=llm["summary"],
            findings=dedupe((base_findings or []) + (llm["findings"] or [])),
            rewrite_sql=rewrite_final,
            index_recommendations=dedupe((base_indexes or []) + (llm["index_recommendations"] or [])),
            index_script=base_script,
            risks=dedupe((base_risks or []) + (llm["risks"] or [])),
            test_steps=llm["test_steps"] or [
                "Capture current plan & metrics (duration, CPU, reads).",
                "Apply one change at a time (index or rewrite).",
                "Compare estimated vs actual plans; validate row estimates.",
                "Benchmark on prod-like data; check regressions.",
            ],
        )

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI call failed: {e}")
