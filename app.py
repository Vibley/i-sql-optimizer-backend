import os, json, re, datetime, time
from hashlib import md5
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
app = FastAPI(title="AI SQL Optimizer Backend", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOW_ORIGIN] if ALLOW_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Global shared HTTP client & OpenAI client --------
transport = httpx.HTTPTransport(retries=2, verify=True)
timeout = httpx.Timeout(connect=10.0, read=120.0, write=120.0, pool=120.0)
HTTP_CLIENT = httpx.Client(
    transport=transport,
    timeout=timeout,
    limits=httpx.Limits(max_connections=20)
)
OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY, http_client=HTTP_CLIENT)

# -------- In-memory cache --------
CACHE = {}

# -------- Middleware logging --------
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

# -------- Helpers --------
def summarize_sql(sql_text: str, max_len: int = 3000) -> str:
    """Compress or truncate long SQL to reduce LLM tokens."""
    sql_clean = re.sub(r"\s+", " ", sql_text.strip())
    if len(sql_clean) <= max_len:
        return sql_clean
    return sql_clean[:1500] + "\n-- ... truncated for brevity ...\n" + sql_clean[-1000:]

def _canon_sql(s: str) -> str:
    s = (s or "")
    s = re.sub(r"```(?:sql)?", "", s, flags=re.IGNORECASE)
    s = s.replace("`", "").strip()
    if s.endswith(";"):
        s = s[:-1]
    return re.sub(r"\s+", " ", s).strip().lower()

def to_list(x):
    if x is None: return []
    if isinstance(x, list): return x
    if isinstance(x, dict): return [str(v) for v in x.values()]
    return [str(x)]

def dedupe(seq): 
    return list(dict.fromkeys(seq or []))

# -------- Health --------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------- Analyze (optimized + streaming) --------
@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Optimized streaming analyzer with caching & pre-summarization."""
    if not OPENAI_API_KEY:
        if REQUIRE_OPENAI:
            raise HTTPException(status_code=503,
                detail="OpenAI is required but OPENAI_API_KEY is missing.")
        return StreamingResponse(iter(["data: {\"summary\":\"Static analysis only\"}\n\n"]),
                                 media_type="text/event-stream")

    sql = summarize_sql(req.sql_text or "")
    key = md5((req.dbms + sql).encode()).hexdigest()
    if key in CACHE:
        print(f"[CACHE HIT] {key}")
        return StreamingResponse(
            iter(["event: done\ndata: " + json.dumps(CACHE[key]) + "\n\n"]),
            media_type="text/event-stream"
        )

    def event_stream():
        yield "event: message\ndata: {\"status\":\"starting\"}\n\n"

        try:
            # --- Quick mini model for first scan ---
            quick = OPENAI_CLIENT.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": "List any obvious SQL anti-patterns (SELECT *, functions on columns, missing WHERE, etc.)"},
                    {"role": "user", "content": sql}
                ],
                max_tokens=300
            )
            quick_findings = quick.choices[0].message.content
            yield f"event: quick\ndata: {json.dumps({'quick_findings': quick_findings})}\n\n"

            # --- Main deep analysis ---
            system_msg = f"You are a veteran {req.dbms} performance engineer."
            plan = (req.plan_xml or "")[:20000]
            user_msg = (
                f"Quick scan findings:\n{quick_findings}\n\n"
                f"SQL:\n```\n{sql}\n```\n\nContext:\n{req.context or 'n/a'}\n\n"
                f"Plan XML (optional):\n{plan or 'n/a'}\n"
            )
            json_instr = (
                "Return a JSON object with exactly these keys: "
                "summary (string), findings (array of strings), rewrite_sql (string), "
                "index_recommendations (array of strings), risks (array of strings), "
                "test_steps (array of strings). No nested objects or extra keys."
            )

            partial = ""
            with OPENAI_CLIENT.chat.completions.with_streaming_response.create(
                model="gpt-4o-mini",
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                    {"role": "user", "content": json_instr},
                ],
            ) as stream:
                for event in stream:
                    if event.type == "token":
                        partial += event.token
                        if len(partial) > 500 and partial.count("{") == partial.count("}"):
                            yield f"event: chunk\ndata: {json.dumps({'partial': partial})}\n\n"

            try:
                llm = json.loads(partial)
            except Exception:
                llm = {"summary": "Model returned incomplete JSON.", "findings": [partial]}

            result = {
                "summary": str(llm.get("summary") or "LLM analysis completed."),
                "findings": dedupe(to_list(llm.get("findings"))),
                "rewrite_sql": llm.get("rewrite_sql") or "",
                "index_recommendations": dedupe(to_list(llm.get("index_recommendations"))),
                "risks": dedupe(to_list(llm.get("risks"))),
                "test_steps": to_list(llm.get("test_steps")),
            }

            CACHE[key] = result
            yield f"event: done\ndata: {json.dumps(result)}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# -------- Quick OpenAI diag --------
@app.get("/diag/openai")
def diag_openai():
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not set.")
    try:
        _ = OPENAI_CLIENT.chat.completions.create(model="gpt-4o-mini",
                                                  messages=[{"role":"user","content":"ping"}],
                                                  max_tokens=1)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI check failed: {e}")
