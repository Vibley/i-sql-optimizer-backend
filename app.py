import os, json, re, datetime, time
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import sqlparse
import httpx
from openai import OpenAI

# -------- Proxy cleanup --------
for k in ["HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy","OPENAI_PROXY"]:
    os.environ.pop(k, None)

ALLOW_ORIGIN   = os.getenv("ALLOW_ORIGIN", "*")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REQUIRE_OPENAI = os.getenv("REQUIRE_OPENAI", "1") == "1"

# -------- FastAPI setup --------
app = FastAPI(title="AI SQL Optimizer Backend", version="1.7.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOW_ORIGIN] if ALLOW_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Logging middleware --------
@app.middleware("http")
async def log_time(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    dt = (time.perf_counter() - t0) * 1000
    print(f"[TIMING] {request.url.path} took {dt:.1f} ms")
    return resp

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

# -------- Helper functions --------
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

# -------- Static rules --------
def static_rules(sql: str):
    # (same as before)
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

    if re.search(r"WHERE\s+.*\b(YEAR|MONTH|DAY|DATE|DATEADD|DATEDIFF|SUBSTRING|CAST|CONVERT)\s*\(", sql_compact) or \
       re.search(r"::\s*DATE\b", sql_norm, flags=re.IGNORECASE):
        findings.append("Non-sargable predicate (function/cast on column) can block index seeks.")
        guidance_lines.append("-- Prefer sargable range predicates over functions/casts on columns.")

    if re.search(r"\bWHERE\b.*\bOR\b", sql_compact):
        findings.append("OR conditions may reduce index usage; consider UNION ALL or indexed computed columns.")

    if "ORDER BY" in sql_compact and "JOIN" not in sql_compact:
        findings.append("ORDER BY detected; ensure index supports ORDER BY key(s).")

    if "WHERE" not in sql_compact and "JOIN" in sql_compact:
        findings.append("JOIN without WHERE may explode rows; verify join predicates and filters.")

    working = sql_norm
    concrete_changes = 0

    for m in re.finditer(r"\bYEAR\s*\(\s*(?P<col>[A-Za-z0-9_\.\[\]]+)\s*\)\s*=\s*(?P<yyyy>19\d{2}|20\d{2})", working, flags=re.IGNORECASE):
        col, yyyy = m.group("col"), int(m.group("yyyy"))
        start, end = f"{yyyy:04d}-01-01", f"{(yyyy+1):04d}-01-01"
        rng = f"{col} >= '{start}' AND {col} < '{end}'"
        working = re.sub(r"\bYEAR\s*\(\s*"+re.escape(col)+r"\s*\)\s*=\s*"+str(yyyy), rng, working, count=1, flags=re.IGNORECASE)
        concrete_changes += 1

    # (same rest of static_rules body as yours)
    # ...
    # Return results
    rewrite_out = working if concrete_changes > 0 else ("\n".join(guidance_lines) if guidance_lines else None)
    return findings, rewrite_out, index_recs, None, risks

# -------- Health --------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------- Quick OpenAI diag --------
@app.get("/diag/openai")
def diag_openai():
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not set.")
    try:
        timeout = httpx.Timeout(connect=10.0, read=20.0)
        client = OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client(trust_env=False, timeout=timeout))
        _ = client.chat.completions.create(model="gpt-4o-mini",
                                           messages=[{"role":"user","content":"ping"}],
                                           max_tokens=1)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI check failed: {e}")

# -------- Analyze (normal response) --------
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
            raise HTTPException(status_code=503,
                detail="OpenAI is required but OPENAI_API_KEY is missing. Set it or disable REQUIRE_OPENAI.")
        return AnalyzeResponse(
            summary="Static analysis completed (OpenAI not configured).",
            findings=base_findings or ["No obvious issues detected."],
            rewrite_sql=base_rewrite or "No query rewrite suggestions were identified.",
            index_recommendations=base_indexes,
            index_script=base_script,
            risks=base_risks,
            test_steps=["Capture plan & metrics","Apply one change at a time","Compare plans","Benchmark regressions"]
        )

    try:
        timeout = httpx.Timeout(connect=15.0, read=120.0, write=120.0, pool=120.0)
        client = OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client(trust_env=False, timeout=timeout))

        system_msg = f"You are a veteran {req.dbms} performance engineer."
        plan = (req.plan_xml or "")[:20000]

        user_msg = (
            f"SQL:\n```\n{sql_fmt}\n```\n\nContext:\n{req.context or 'n/a'}\n\n"
            f"Plan XML (optional):\n{plan or 'n/a'}\n"
        )

        json_instr = (
            "Return a JSON object with keys: summary, findings, rewrite_sql, index_recommendations, risks, test_steps."
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role":"system","content":system_msg},
                {"role":"user","content":user_msg},
                {"role":"user","content":json_instr},
            ],
        )

        llm = json.loads(resp.choices[0].message.content)
        def dedupe(seq): return list(dict.fromkeys(seq or []))

        rewrite_raw = llm.get("rewrite_sql") or ""
        same = _canon_sql(rewrite_raw) == _canon_sql(sql_fmt)
        rewrite_final = None if same else (rewrite_raw or None)
        if not rewrite_final:
            rewrite_final = base_rewrite or "No query rewrite suggestions were identified."

        return AnalyzeResponse(
            summary=llm.get("summary") or "LLM analysis completed.",
            findings=dedupe((base_findings or []) + (llm.get("findings") or [])),
            rewrite_sql=rewrite_final,
            index_recommendations=dedupe((base_indexes or []) + (llm.get("index_recommendations") or [])),
            index_script=base_script,
            risks=dedupe((base_risks or []) + (llm.get("risks") or [])),
            test_steps=llm.get("test_steps") or [
                "Capture current plan & metrics.",
                "Apply one change at a time.",
                "Compare estimated vs actual plans.",
                "Benchmark with prod-like data.",
            ],
        )

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI call failed: {e}")

# -------- Streaming endpoint (optional) --------
@app.post("/analyze/stream")
def analyze_stream(req: AnalyzeRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY missing")

    timeout = httpx.Timeout(connect=15.0, read=120.0, write=120.0, pool=120.0)
    client = OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client(trust_env=False, timeout=timeout))

    def event_stream():
        yield "Starting analysis...\n"
        with client.chat.completions.with_streaming_response.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role":"system","content":"You are a SQL tuning expert."},
                {"role":"user","content":req.sql_text}
            ],
        ) as stream:
            for event in stream:
                if event.type == "token":
                    yield event.token
        yield "\n\n[done]"

    return StreamingResponse(event_stream(), media_type="text/plain")
