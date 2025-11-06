import os, json, re, datetime, hashlib
from typing import List, Optional, Tuple, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
import sqlparse

# ---------- Env / proxy hygiene ----------
for k in ["HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy","OPENAI_PROXY"]:
    os.environ.pop(k, None)

ALLOW_ORIGIN   = os.getenv("ALLOW_ORIGIN", "*")
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()

# ---------- App ----------
app = FastAPI(title="AI SQL Optimizer Backend", version="2.0.0")
app.add_middleware(GZipMiddleware, minimum_size=512)
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

# ---------- Utils ----------
def _sha_key(*parts: str) -> str:
    import hashlib
    h = hashlib.sha1()
    for p in parts:
        h.update((p or "").encode("utf-8")); h.update(b"\x00")
    return h.hexdigest()

def _canon_sql(s: str) -> str:
    s = (s or "")
    s = re.sub(r"```(?:sql)?", "", s, flags=re.IGNORECASE)
    s = s.replace("`","").strip()
    if s.endswith(";"): s = s[:-1]
    return re.sub(r"\s+", " ", s).strip().lower()

def _iso_next_day(d: str) -> str:
    y,m,day = map(int, d.split("-"))
    dt = datetime.date(y,m,day) + datetime.timedelta(days=1)
    return dt.strftime("%Y-%m-%d")

def _month_range(yyyy: int, mm: int) -> Tuple[str,str]:
    start = datetime.date(yyyy, mm, 1)
    nxt = datetime.date(yyyy + (1 if mm==12 else 0), (1 if mm==12 else mm+1), 1)
    return start.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")

# ---------- SQL compression for LLM (preserves structure, drops noise) ----------
_STR_LIT = re.compile(r"('(?:''|[^'])*')")
_HEX     = re.compile(r"0x[0-9A-Fa-f]+")
_NUM     = re.compile(r"\b\d{4,}\b")
_COM1    = re.compile(r"--[^\n]*")
_COM2    = re.compile(r"/\*.*?\*/", re.S)
_IN_LIST = re.compile(r"\bIN\s*\(\s*(?:[^()]*?)(,[^()]*?){10,}\s*\)", re.I|re.S)  # long IN (...)

def compress_sql_for_llm(sql: str, max_len: int = 15000) -> str:
    s = sql
    # remove comments
    s = _COM1.sub("", s)
    s = _COM2.sub("", s)
    # mask long literals / blobs / massive numbers to reduce tokens (keep shape)
    s = _HEX.sub("0x…", s)
    s = _STR_LIT.sub("'…'", s)
    s = _NUM.sub("N", s)
    # collapse huge IN lists
    s = _IN_LIST.sub("IN (…)", s)
    # normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len]

# ---------- Static rules + targeted rewrites ----------
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
        guidance_lines.append("-- Consider full-text / CONTAINS (SQL Server) or trigram / FTS.")
        risks.append("Full scans on large tables can be expensive.")

    if re.search(r"WHERE\s+.*\b(YEAR|MONTH|DAY|DATE|DATEADD|DATEDIFF|SUBSTRING|CAST|CONVERT)\s*\(", sql_compact) \
       or re.search(r"::\s*DATE\b", sql_norm, flags=re.IGNORECASE):
        findings.append("Non-sargable predicate (function/cast on column) can block index seeks.")
        guidance_lines.append("-- Prefer range predicates over functions/casts on columns.")

    if re.search(r"\bWHERE\b.*\bOR\b", sql_compact):
        findings.append("OR conditions may reduce index usage; consider UNION ALL or indexed computed columns.")

    if "ORDER BY" in sql_compact and "JOIN" not in sql_compact:
        findings.append("ORDER BY detected; ensure index supports ORDER BY key(s).")

    if "WHERE" not in sql_compact and "JOIN" in sql_compact:
        findings.append("JOIN without WHERE may explode rows; verify join predicates and filters.")

    # concrete sargability rewrites
    working = sql_norm
    changes = 0

    for m in re.finditer(r"\bYEAR\s*\(\s*(?P<col>[A-Za-z0-9_\.\[\]]+)\s*\)\s*=\s*(?P<yyyy>19\d{2}|20\d{2})", working, flags=re.IGNORECASE):
        col = m.group("col"); yyyy = int(m.group("yyyy"))
        start, end = f"{yyyy:04d}-01-01", f"{(yyyy+1):04d}-01-01"
        pattern = r"\bYEAR\s*\(\s*"+re.escape(col)+r"\s*\)\s*=\s*"+str(yyyy)
        working = re.sub(pattern, f"{col} >= '{start}' AND {col} < '{end}'", working, count=1, flags=re.IGNORECASE)
        changes += 1

    for m in re.finditer(r"\bDATE\s*\(\s*(?P<col>[A-Za-z0-9_\.\[\]]+)\s*\)\s*=\s*'(?P<d>\d{4}-\d{2}-\d{2})'", working, flags=re.IGNORECASE):
        col, d = m.group("col"), m.group("d")
        working = re.sub(
            r"\bDATE\s*\(\s*"+re.escape(col)+r"\s*\)\s*=\s*'"+re.escape(d)+r"'",
            f"{col} >= '{d}' AND {col} < '{_iso_next_day(d)}'",
            working, count=1, flags=re.IGNORECASE
        )
        changes += 1

    for m in re.finditer(r"\bCAST\s*\(\s*(?P<col>[A-Za-z0-9_\.\[\]]+)\s+AS\s+DATE\s*\)\s*=\s*'(?P<d>\d{4}-\d{2}-\d{2})'", working, flags=re.IGNORECASE):
        col, d = m.group("col"), m.group("d")
        working = re.sub(
            r"\bCAST\s*\(\s*"+re.escape(col)+r"\s+AS\s+DATE\s*\)\s*=\s*'"+re.escape(d)+r"'",
            f"{col} >= '{d}' AND {col} < '{_iso_next_day(d)}'",
            working, count=1, flags=re.IGNORECASE
        )
        changes += 1

    for m in re.finditer(r"\b(?P<col>[A-Za-z0-9_\.\[\]]+)\s*::\s*DATE\s*=\s*'(?P<d>\d{4}-\d{2}-\d{2})'", working, flags=re.IGNORECASE):
        col, d = m.group("col"), m.group("d")
        working = re.sub(
            r"\b"+re.escape(col)+r"\s*::\s*DATE\s*=\s*'"+re.escape(d)+r"'",
            f"{col} >= '{d}' AND {col} < '{_iso_next_day(d)}'",
            working, count=1, flags=re.IGNORECASE
        )
        changes += 1

    mon = re.search(r"\bMONTH\s*\(\s*(?P<c1>[A-Za-z0-9_\.\[\]]+)\s*\)\s*=\s*(?P<mm>1[0-2]|0?[1-9])", sql_norm, flags=re.IGNORECASE)
    yr  = re.search(r"\bYEAR\s*\(\s*(?P<c2>[A-Za-z0-9_\.\[\]]+)\s*\)\s*=\s*(?P<yyyy>19\d{2}|20\d{2})", sql_norm, flags=re.IGNORECASE)
    if mon and yr and mon.group("c1").lower() == yr.group("c2").lower():
        c, mm, yyyy = mon.group("c1"), int(mon.group("mm")), int(yr.group("yyyy"))
        start, end = _month_range(yyyy, mm)
        guidance_lines.append(f"-- Replace MONTH({c})={mm} AND YEAR({c})={yyyy} with:\n-- {c} >= '{start}' AND {c} < '{end}'")

    # index guess
    mcols = re.findall(r"\b([A-Z_][A-Z0-9_\.]+)\s*=\s*[@:\w'\-]+", sql_compact)
    index_recs, index_script = [], None
    if mcols:
        cols = [c.split(".")[-1].lower() for c in mcols]
        cols = list(dict.fromkeys(cols))[:3]
        tbl = re.search(r"\bFROM\s+([A-Za-z0-9_\.\[\]\"`]+)", sql_norm, flags=re.IGNORECASE) \
           or re.search(r"\bJOIN\s+([A-Za-z0-9_\.\[\]\"`]+)", sql_norm, flags=re.IGNORECASE)
        table = re.sub(r'[\[\]"`]', "", tbl.group(1)).lower() if tbl else "<yourtable>"
        if cols:
            rec = f"create index ix_{cols[0]}_suggested on {table} ({', '.join(cols)});"
            index_recs.append(rec)
            index_script = f"""-- Suggested indexes
CREATE INDEX ix_{cols[0]}_suggested ON {table} ({', '.join(cols)});
GO"""

    rewrite_out = working if changes > 0 else ("\n".join(guidance_lines) if guidance_lines else None)
    return findings, rewrite_out, index_recs, index_script, risks

# ---------- OpenAI global (async) ----------
openai_client = None
httpx_client  = None

@app.on_event("startup")
async def _startup():
    global httpx_client, openai_client
    try:
        import httpx
        from openai import AsyncOpenAI
        httpx_client = httpx.AsyncClient(trust_env=False, timeout=20.0)
        if OPENAI_API_KEY:
            openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY, http_client=httpx_client)
    except Exception:
        pass

# ---------- Health ----------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- LLM helpers (two-pass) ----------
async def llm_skeletonize(dbms: str, sql_orig: str, context: str, plan_xml: str) -> Dict[str, Any]:
    """
    Pass A: compress long SQL and ask the model for a structured skeleton.
    """
    compressed = compress_sql_for_llm(sql_orig, max_len=15000)
    system = (
        f"You are a veteran {dbms} performance engineer. "
        "Extract a *concise* structural summary of the SQL without rewriting it."
    )
    user = (
        "Compressed SQL (comments/literals trimmed):\n"
        "```\n" + compressed + "\n```\n\n"
        "Context (optional):\n" + (context or "n/a") + "\n\n"
        "Plan XML (optional):\n" + (plan_xml or "n/a") + "\n"
    )
    schema = {
        "type": "object",
        "properties": {
            "tables": {"type":"array","items":{"type":"string"}},
            "joins": {"type":"array","items":{"type":"string"}},
            "predicates": {"type":"array","items":{"type":"string"}},
            "group_order": {"type":"array","items":{"type":"string"}},
            "sargability_risks": {"type":"array","items":{"type":"string"}},
            "rewrite_opportunities": {"type":"array","items":{"type":"string"}}
        },
        "required": ["tables","joins","predicates"]
    }
    res = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=700,
        response_format={"type":"json_object"},
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":user},
            {"role":"user","content":"Return strictly valid JSON matching the schema above. No extra keys or prose."}
        ]
    )
    return json.loads(res.choices[0].message.content)

async def llm_rewrite(dbms: str, sql_fmt: str, skeleton: Dict[str,Any], context: str, plan_xml: str) -> Dict[str,Any]:
    """
    Pass B: provide original SQL + skeleton; ask for precise rewrite & findings.
    """
    system = (
        f"You are a veteran {dbms} performance engineer. "
        "Produce a safer, faster equivalent query when possible; otherwise return guidance. "
        "Do NOT echo the original query; only output your rewrite."
    )
    user = (
        "Original SQL (formatted):\n```\n" + sql_fmt + "\n```\n\n"
        "Extracted skeleton (JSON):\n```\n" + json.dumps(skeleton, indent=2) + "\n```\n\n"
        "Context:\n" + (context or "n/a") + "\n\n"
        "Plan XML (optional):\n" + (plan_xml or "n/a") + "\n"
    )
    json_instr = (
        "Return a JSON object with keys: "
        "summary (string), findings (array of strings), rewrite_sql (string), "
        "index_recommendations (array of strings), risks (array of strings), test_steps (array of strings). "
        "If your rewrite would be identical to the original after formatting (ignore case/whitespace/semicolon), set rewrite_sql to an empty string."
    )
    res = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=900,
        response_format={"type":"json_object"},
        messages=[
            {"role":"system","content":system},
            {"role":"user","content":user},
            {"role":"user","content":json_instr}
        ]
    )
    return json.loads(res.choices[0].message.content)

# ---------- Analyze ----------
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    sql = req.sql_text or ""
    try:
        # avoid heavy reindent on huge payloads
        sql_fmt = sqlparse.format(sql, keyword_case="upper", reindent=True) if len(sql) < 200_000 else sql
    except Exception:
        sql_fmt = sql

    base_findings, base_rewrite, base_indexes, base_script, base_risks = static_rules(sql_fmt)

    # If OpenAI not configured, return static
    if not OPENAI_API_KEY or openai_client is None:
        return AnalyzeResponse(
            summary="Static analysis completed.",
            findings=base_findings or ["No obvious issues detected by static rules."],
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

    # Trim gargantuan fields to keep latency down
    ctx  = (req.context or "")[:6000]
    plan = (req.plan_xml or "")[:15000]

    # Two-pass prompting for better fidelity on long SQL
    try:
        skeleton = await llm_skeletonize(req.dbms, sql_fmt, ctx, plan)
        llm_out  = await llm_rewrite(req.dbms, sql_fmt, skeleton, ctx, plan)

        # Guardrails vs echo
        rewrite_raw   = llm_out.get("rewrite_sql") or ""
        orig_canon    = _canon_sql(sql_fmt)
        rewrite_canon = _canon_sql(rewrite_raw)
        rewrite = None if (rewrite_canon and rewrite_canon == orig_canon) else (rewrite_raw or None)
        if not rewrite:
            rewrite = base_rewrite or "No query rewrite suggestions were identified."

        # Merge + dedupe
        def dedupe(seq):
            seen, out = set(), []
            for s in seq or []:
                if s not in seen:
                    seen.add(s); out.append(s)
            return out

        findings = dedupe((base_findings or []) + (llm_out.get("findings") or []))
        indexes  = dedupe((base_indexes  or []) + (llm_out.get("index_recommendations") or []))
        risks    = dedupe((base_risks    or []) + (llm_out.get("risks") or []))
        steps    = llm_out.get("test_steps") or [
            "Capture current plan & metrics (duration, CPU, reads).",
            "Apply one change at a time (index or rewrite).",
            "Compare estimated vs actual plans; validate row estimates.",
            "Benchmark on prod-like data; check regressions.",
        ]
        summary  = llm_out.get("summary") or "Analysis completed."

        return AnalyzeResponse(
            summary=summary,
            findings=findings or ["No obvious issues found."],
            rewrite_sql=rewrite,
            index_recommendations=indexes,
            index_script=base_script,  # static script stays deterministic
            risks=risks,
            test_steps=steps,
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
