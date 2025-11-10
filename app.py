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
REQUIRE_OPENAI = os.getenv("REQUIRE_OPENAI", "1") == "1"   # <-- force LLM by default

app = FastAPI(title="AI SQL Optimizer Backend", version="1.7.0")

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

# -------- Static analysis (unchanged) --------
# (your full static_rules() function remains the same)
# -------- Health & Diag --------
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
        _ = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
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

    from sqlparse import format as fmt
    base_findings, base_rewrite, base_indexes, base_script, base_risks = static_rules(sql_fmt)

    # No OpenAI key
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

    # --- GPT-Enhanced Analysis ---
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
            # Fallback if GPT didn’t strictly output JSON
            llm = {"summary": llm_raw, "findings": [], "rewrite_sql": "", "index_recommendations": [], "risks": [], "test_steps": []}

        # --- Add missing fields to ensure all components exist ---
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
