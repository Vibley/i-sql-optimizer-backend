import os, json, re, datetime, threading
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# -------- Shared transport only (not full client) --------
TRANSPORT = httpx.HTTPTransport(retries=1, verify=True)
TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=60.0, pool=30.0)

# -------- FastAPI setup --------
app = FastAPI(title="AI SQL Optimizer Backend", version="1.6.2")

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

# -------- Static rules (unchanged) --------
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

    working = sql_norm
    concrete_changes = 0

    # ... rest of static_rules logic unchanged ...
    # (not repeated for brevity — your original static_rules stays exactly the same)
    # Paste the unchanged body of static_rules here.
    # ---------------------------------------------
    # [Omitted: same as your uploaded code block]
    # ---------------------------------------------

    rewrite_out = working if concrete_changes > 0 else ("\n".join(guidance_lines) if guidance_lines else None)
    return findings, rewrite_out, index_recs, None, risks

# -------- Health --------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------- OpenAI diag --------
@app.get("/diag/openai")
def diag_openai():
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not set.")
    try:
        client = OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client(transport=TRANSPORT, timeout=TIMEOUT))
        _ = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":"ping"}], max_tokens=1)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI check failed: {e}")

# -------- Warmup thread (to reduce cold-start delay) --------
def _warmup_openai():
    try:
        client = OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client(transport=TRANSPORT, timeout=TIMEOUT))
        client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":"warmup"}], max_tokens=1)
        print("[Warmup] OpenAI client warmed up successfully.")
    except Exception as e:
        print(f"[Warmup] Skipped: {e}")

threading.Thread(target=_warmup_openai, daemon=True).start()

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
            raise HTTPException(status_code=503, detail="OpenAI is required but missing.")
        return AnalyzeResponse(
            summary="Static analysis completed (OpenAI not configured).",
            findings=base_findings or ["No obvious issues detected."],
            rewrite_sql=base_rewrite or "No rewrite suggestions.",
            index_recommendations=base_indexes,
            index_script=base_script,
            risks=base_risks,
            test_steps=[
                "Capture plan & metrics.",
                "Apply one change at a time.",
                "Compare estimated vs actual plans.",
                "Benchmark on prod-like data.",
            ],
        )

    try:
        client = OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client(transport=TRANSPORT, timeout=TIMEOUT))
        system_msg = f"You are a veteran {req.dbms} performance engineer. Return safe, actionable tuning advice."
        plan = (req.plan_xml or "")[:20000]
        user_msg = (
            f"SQL (formatted):\n```\n{sql_fmt}\n```\n\nContext:\n{req.context or 'n/a'}\n\n"
            f"Execution plan XML:\n{plan if plan else 'n/a'}\n"
        )

        json_instructions = (
            "Return a JSON object with keys: summary (string), findings (array of strings), "
            "rewrite_sql (string), index_recommendations (array of strings), risks (array of strings), "
            "test_steps (array of strings). No extra text."
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
        rewrite_raw = llm.get("rewrite_sql") or ""
        same = _canon_sql(rewrite_raw) == _canon_sql(sql_fmt)
        rewrite_final = None if same else (rewrite_raw or None)
        if not rewrite_final:
            rewrite_final = base_rewrite or "No rewrite suggestions."

        return AnalyzeResponse(
            summary=llm.get("summary") or "LLM analysis completed.",
            findings=dedupe((base_findings or []) + (llm.get("findings") or [])),
            rewrite_sql=rewrite_final,
            index_recommendations=dedupe((base_indexes or []) + (llm.get("index_recommendations") or [])),
            index_script=base_script,
            risks=dedupe((base_risks or []) + (llm.get("risks") or [])),
            test_steps=llm.get("test_steps") or [
                "Capture plan & metrics.",
                "Apply one change at a time.",
                "Compare estimated vs actual plans.",
                "Benchmark with prod-like data.",
            ],
        )

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI call failed: {e}")
