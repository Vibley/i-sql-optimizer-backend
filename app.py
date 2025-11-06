import os, json, re, datetime
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sqlparse

# ---------------- Proxy cleanup (prevents httpx/OpenAI from reading proxies) ----------------
for k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "OPENAI_PROXY"]:
    os.environ.pop(k, None)

ALLOW_ORIGIN = os.getenv("ALLOW_ORIGIN", "*")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="AI SQL Optimizer Backend", version="1.5.0")

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
    findings: List[str] = Field(default_factory=list)
    rewrite_sql: Optional[str] = None
    index_recommendations: List[str] = Field(default_factory=list)
    index_script: Optional[str] = None
    risks: List[str] = Field(default_factory=list)
    test_steps: List[str] = Field(default_factory=list)

# ---------------- Date helpers ----------------
def _iso_next_day(d: str) -> str:
    """Return ISO date string for day after d (YYYY-MM-DD)."""
    y, m, day = map(int, d.split("-"))
    dt = datetime.date(y, m, day) + datetime.timedelta(days=1)
    return dt.strftime("%Y-%m-%d")

def _month_range(yyyy: int, mm: int):
    """Return ('YYYY-MM-01', 'YYYY-MM-next-01')."""
    start = datetime.date(yyyy, mm, 1)
    nxt = datetime.date(yyyy + (1 if mm == 12 else 0), 1 if mm == 12 else mm + 1, 1)
    return (start.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d"))

# ---------------- Static rules + simple rewrites ----------------
def static_rules(sql: str):
    """
    Returns:
      findings: List[str]
      rewrite_out: Optional[str]  (concrete rewrite OR guidance comments)
      index_recs: List[str]       (human-readable suggestions)
      index_script: Optional[str] (complete executable T-SQL)
      risks: List[str]
    """
    findings, guidance_lines, index_recs, risks = [], [], [], []
    sql_norm = sql.strip()
    sql_compact = re.sub(r"\s+", " ", sql_norm, flags=re.MULTILINE).upper()

    # Heuristic checks
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

    # -------- Concrete rewrites (apply in-place where safe) --------
    working = sql_norm
    concrete_changes = 0

    # YEAR(col) = YYYY -> [col >= 'YYYY-01-01' AND col < 'YYYY+1-01-01']
    for m in re.finditer(r"\bYEAR\s*\(\s*(?P<col>[A-Za-z0-9_\.\[\]]+)\s*\)\s*=\s*(?P<yyyy>19\d{2}|20\d{2})", working, flags=re.IGNORECASE):
        col = m.group("col")
        yyyy = int(m.group("yyyy"))
        start = f"{yyyy:04d}-01-01"
        end   = f"{(yyyy+1):04d}-01-01"
        rng = f"{col} >= '{start}' AND {col} < '{end}'"
        working = re.sub(
            r"\bYEAR\s*\(\s*"+re.escape(col)+r"\s*\)\s*=\s*"+str(yyyy),
            rng,
            working,
            count=1,
            flags=re.IGNORECASE,
        )
        concrete_changes += 1
        if "Non-sargable predicate" not in " ".join(findings):
            findings.append("Non-sargable predicate (function on column) blocks index seeks.")

    # DATE(col) = 'YYYY-MM-DD' -> day range
    for m in re.finditer(r"\bDATE\s*\(\s*(?P<col>[A-Za-z0-9_\.\[\]]+)\s*\)\s*=\s*'(?P<d>\d{4}-\d{2}-\d{2})'", working, flags=re.IGNORECASE):
        col = m.group("col"); d = m.group("d"); d2 = _iso_next_day(d)
        rng = f"{col} >= '{d}' AND {col} < '{d2}'"
        working = re.sub(
            r"\bDATE\s*\(\s*"+re.escape(col)+r"\s*\)\s*=\s*'"+re.escape(d)+r"'",
            rng,
            working,
            count=1,
            flags=re.IGNORECASE,
        )
        concrete_changes += 1

    # CAST(col AS DATE) = 'YYYY-MM-DD' -> day range
    for m in re.finditer(r"\bCAST\s*\(\s*(?P<col>[A-Za-z0-9_\.\[\]]+)\s+AS\s+DATE\s*\)\s*=\s*'(?P<d>\d{4}-\d{2}-\d{2})'", working, flags=re.IGNORECASE):
        col = m.group("col"); d = m.group("d"); d2 = _iso_next_day(d)
        rng = f"{col} >= '{d}' AND {col} < '{d2}'"
        working = re.sub(
            r"\bCAST\s*\(\s*"+re.escape(col)+r"\s+AS\s+DATE\s*\)\s*=\s*'"+re.escape(d)+r"'",
            rng,
            working,
            count=1,
            flags=re.IGNORECASE,
        )
        concrete_changes += 1

    # Postgres: col::DATE = 'YYYY-MM-DD' -> day range
    for m in re.finditer(r"\b(?P<col>[A-Za-z0-9_\.\[\]]+)\s*::\s*DATE\s*=\s*'(?P<d>\d{4}-\d{2}-\d{2})'", working, flags=re.IGNORECASE):
        col = m.group("col"); d = m.group("d"); d2 = _iso_next_day(d)
        rng = f"{col} >= '{d}' AND {col} < '{d2}'"
        working = re.sub(
            r"\b"+re.escape(col)+r"\s*::\s*DATE\s*=\s*'"+re.escape(d)+r"'",
            rng,
            working,
            count=1,
            flags=re.IGNORECASE,
        )
        concrete_changes += 1

    # MONTH(col)=M + YEAR(col)=YYYY -> guidance month range
    mon = re.search(r"\bMONTH\s*\(\s*(?P<c1>[A-Za-z0-9_\.\[\]]+)\s*\)\s*=\s*(?P<mm>1[0-2]|0?[1-9])", sql_norm, flags=re.IGNORECASE)
    yr  = re.search(r"\bYEAR\s*\(\s*(?P<c2>[A-Za-z0-9_\.\[\]]+)\s*\)\s*=\s*(?P<yyyy>19\d{2}|20\d{2})", sql_norm, flags=re.IGNORECASE)
    if mon and yr:
        c1 = mon.group("c1"); c2 = yr.group("c2")
        mm = int(mon.group("mm")); yyyy = int(yr.group("yyyy"))
        if c1.lower() == c2.lower():
            start, end = _month_range(yyyy, mm)
            guidance_lines.append(
                f"-- Replace MONTH({c1})={mm} AND YEAR({c1})={yyyy} with range:\n"
                f"-- {c1} >= '{start}' AND {c1} < '{end}'"
            )
            findings.append("Non-sargable month/year predicates detected; prefer a single range on the date column.")

    # -------- Index key guess from equality predicates --------
    # Upper-cased copy for simpler pattern matching of tokens
    m = re.findall(r"\b([A-Z_][A-Z0-9_\.]+)\s*=\s*[@:\w'\-]+", sql_compact)
    if m:
        cols = [col.split(".")[-1].lower() for col in m]
        cols = list(dict.fromkeys(cols))[:3]  # de-dupe, keep first 3

        # Extract table (FROM first, else JOIN) from the original (case-insensitive)
        tbl_match = re.search(r"\bFROM\s+([A-Za-z0-9_\.\[\]\"`]+)", sql, flags=re.IGNORECASE)
        if not tbl_match:
            tbl_match = re.search(r"\bJOIN\s+([A-Za-z0-9_\.\[\]\"`]+)", sql, flags=re.IGNORECASE)

        table_name = None
        if tbl_match:
            table_name = tbl_match.group(1)
            table_name = re.sub(r'[\[\]"`]', "", table_name).lower()  # clean & lowercase

        if table_name and cols:
            # Human-readable suggestion
            index_recs.append(f"create index ix_{cols[0]}_suggested on {table_name} ({', '.join(cols)});")

    # -------- Build Index Script (full T-SQL) --------
    index_script = None
    if index_recs:
        lines = ["-- Suggested indexes"]
        for rec in index_recs:
            # rec looks like: "create index ix_col_suggested on schema.table (col1, col2);"
            try:
                ix_name = re.search(r"ix_[a-z0-9_]+", rec, flags=re.IGNORECASE).group(0)
                on_tbl  = re.search(r"on\s+([^\s(]+)", rec, flags=re.IGNORECASE).group(1)
                cols_in_paren = re.search(r"\(([^)]+)\)", rec).group(1)
                cols_clean = ", ".join([c.strip() for c in cols_in_paren.split(",")])
                lines.append(
                    "IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = '{ix}' AND object_id = OBJECT_ID('{tbl}'))\n"
                    "BEGIN\n"
                    "    CREATE INDEX {ix} ON {tbl} ({cols});\n"
                    "END\nGO".format(ix=ix_name, tbl=on_tbl, cols=cols_clean)
                )
            except Exception:
                # Fallback: echo the raw suggestion
                lines.append(rec if rec.strip().endswith(";") else rec.strip() + ";")
                lines.append("GO")
        index_script = "\n".join(lines)

    # -------- Choose rewrite output --------
    rewrite_out = working if concrete_changes > 0 else ("\n".join(guidance_lines) if guidance_lines else None)

    return findings, rewrite_out, index_recs, index_script, risks

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

    base_findings, base_rewrite, base_indexes, base_script, base_risks = static_rules(sql_fmt)

    # -------- Static-only path if no OpenAI key --------
    if not OPENAI_API_KEY:
        return AnalyzeResponse(
            summary="Static analysis completed (OpenAI not configured).",
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

    # -------- OpenAI-enhanced path --------
    try:
        import httpx
        from openai import OpenAI

        http_client = httpx.Client(trust_env=False, timeout=30.0)
        client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)

        system_msg = (
            f"You are a veteran {req.dbms} performance engineer. "
            f"Return safe, actionable tuning advice. Use <YourTable> placeholders; never invent schema names."
        )
        plan = (req.plan_xml or "")[:20000]  # avoid huge requests

        user_msg = (
            "SQL (formatted):\n```\n"
            f"{sql_fmt}\n```\n\nContext:\n"
            f"{req.context or 'n/a'}\n\nExecution plan XML (optional):\n"
            f"{plan if plan else 'n/a'}\n"
        )

        json_instructions = (
            "Return a JSON object with keys: summary (string), findings (array of strings), "
            "rewrite_sql (string), index_recommendations (array of strings), risks (array of strings), "
            "test_steps (array of strings). "
            "If your rewrite would be identical to the provided SQL after formatting "
            "(ignoring case/whitespace/semicolon), set rewrite_sql to an empty string. "
            "Prefer sargable range predicates over functions on columns. No extra keys or text."
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

        def dedupe(seq): return list(dict.fromkeys(seq or []))

        # Guardrail to suppress echo rewrites
        def _canon_sql(s: str) -> str:
            s = (s or "")
            s = re.sub(r"```(?:sql)?", "", s, flags=re.IGNORECASE)  # strip code fences
            s = s.replace("`", "").strip()
            if s.endswith(";"):
                s = s[:-1]
            return re.sub(r"\s+", " ", s).strip().lower()

        rewrite_raw   = llm.get("rewrite_sql") or ""
        same_as_input = _canon_sql(rewrite_raw) == _canon_sql(sql_fmt)
        rewrite_final = None if same_as_input else (rewrite_raw or None)

        if not rewrite_final:
            rewrite_final = base_rewrite or "No query rewrite suggestions were identified."

        return AnalyzeResponse(
            summary=llm.get("summary") or "Analysis completed.",
            findings=dedupe((base_findings or []) + (llm.get("findings") or [])),
            rewrite_sql=rewrite_final,
            index_recommendations=dedupe((base_indexes or []) + (llm.get("index_recommendations") or [])),
            index_script=base_script,  # static block we built above
            risks=dedupe((base_risks or []) + (llm.get("risks") or [])),
            test_steps=llm.get("test_steps") or [
                "Capture current plan & metrics (duration, CPU, reads).",
                "Apply one change at a time (index or rewrite).",
                "Compare estimated vs actual plans; validate row estimates.",
                "Benchmark on prod-like data; check regressions.",
            ],
        )

    except Exception as e:
        base_findings.append(f"AI enhancer unavailable: {e}")
        return AnalyzeResponse(
            summary="Static analysis completed (LLM call failed).",
            findings=base_findings,
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
