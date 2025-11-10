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
    version: Optional[str] = None   # ✅ added version support

class AnalyzeResponse(BaseModel):
    summary: str
    findings: List[str] = Field(default_factory=list)
    rewrite_sql: Optional[str] = None
    index_recommendations: List[str] = Field(default_factory=list)
    index_script: Optional[str] = None
    risks: List[str] = Field(default_factory=list)
    test_steps: List[str] = Field(default_factory=list)

# -------- Helpers --------
def _canon_sql(s: str) -> str:
    s = (s or "")
    s = re.sub(r"```(?:sql)?", "", s, flags=re.IGNORECASE)
    s = s.replace("`", "").strip()
    if s.endswith(";"):
        s = s[:-1]
    return re.sub(r"\s+", " ", s).strip().lower()

# -------- Static rules --------
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

    if re.search(r"WHERE\s+.*\b(YEAR|MONTH|DAY|DATE|DATEADD|DATEDIFF|SUBSTRING|CAST|CONVERT)\s*\(", sql_compact):
        findings.append("Non-sargable predicate (function/cast on column) can block index seeks.")
        guidance_lines.append("-- Prefer sargable range predicates over functions/casts on columns.")

    if re.search(r"\bWHERE\b.*\bOR\b", sql_compact):
        findings.append("OR conditions may reduce index usage; consider UNION ALL or indexed computed columns.")

    if "ORDER BY" in sql_compact and "JOIN" not in sql_compact:
        findings.append("ORDER BY detected; ensure index supports ORDER BY key(s).")

    if "WHERE" not in sql_compact and "JOIN" in sql_compact:
        findings.append("JOIN without WHERE may explode rows; verify join predicates and filters.")

    working = sql_norm
    for m in re.finditer(r"\bYEAR\s*\(\s*(?P<col>[A-Za-z0-9_\.\[\]]+)\s*\)\s*=\s*(?P<yyyy>19\d{2}|20\d{2})", working, flags=re.IGNORECASE):
        col, yyyy = m.group("col"), int(m.group("yyyy"))
        start, end = f"{yyyy:04d}-01-01", f"{(yyyy+1):04d}-01-01"
        rng = f"{col} >= '{start}' AND {col} < '{end}'"
        working = re.sub(r"\bYEAR\s*\(\s*"+re.escape(col)+r"\s*\)\s*=\s*"+str(yyyy), rng, working, count=1, flags=re.IGNORECASE)

    rewrite_out = working if working != sql_norm else ("\n".join(guidance_lines) if guidance_lines else None)
    return findings, rewrite_out, index_recs, None, risks

# -------- Health --------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------- Analyze --------
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    sql = req.sql_text or ""
    try:
        sql_fmt = sqlparse.format(sql, keyword_case="upper", reindent=True)
    except Exception:
        sql_fmt = sql

    base_findings, base_rewrite, base_indexes, base_script, base_risks = static_rules(sql_fmt)

    # --- No OpenAI key fallback ---
    if not OPENAI_API_KEY:
        if REQUIRE_OPENAI:
            raise HTTPException(status_code=503, detail="OpenAI_API_KEY missing.")
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

    # --- OpenAI call ---
    try:
        import httpx
        from openai import OpenAI

        http_client = httpx.Client(trust_env=False, timeout=30.0)
        client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)

        # 🧠 Version-aware system message
        version_label = req.version or "latest"
        system_msg = (
            f"You are a veteran {req.dbms} performance engineer specializing in SQL Server {version_label}. "
            f"Only suggest features and syntax compatible with SQL Server {version_label}. "
            "If a feature was introduced after that version, explain the alternative. "
            "Return safe, actionable tuning advice using only supported syntax. "
            "Use <YourTable> placeholders; never invent schema names."
        )

        plan = (req.plan_xml or "")[:20000]
        user_msg = (
            f"SQL (formatted):\n```\n{sql_fmt}\n```\n\n"
            f"Context:\n{req.context or 'n/a'}\n\n"
            f"Execution plan XML (optional):\n{plan if plan else 'n/a'}\n"
        )

        json_instructions = (
            "Return a JSON object with keys: summary, findings, rewrite_sql, "
            "index_recommendations, risks, and test_steps. "
            "Always include all keys even if empty."
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

        # 🧩 Merge nested JSON if returned in summary
        if isinstance(llm.get("summary"), str) and llm["summary"].strip().startswith("```json"):
            try:
                inner_json = re.search(r"\{.*\}", llm["summary"], re.DOTALL)
                if inner_json:
                    nested = json.loads(inner_json.group(0))
                    for k, v in nested.items():
                        if k not in llm or not llm[k]:
                            llm[k] = v
            except Exception as e:
                print("⚠️ Nested JSON parse failed:", e)

        defaults = {
            "summary": f"LLM analysis completed for SQL Server {version_label}.",
            "findings": [],
            "rewrite_sql": "",
            "index_recommendations": [],
            "risks": [],
            "test_steps": []
        }
        for k, v in defaults.items():
            llm.setdefault(k, v)

        def dedupe(seq): return list(dict.fromkeys(seq or []))
        rewrite_raw = llm.get("rewrite_sql") or ""
        same_as_input = _canon_sql(rewrite_raw) == _canon_sql(sql_fmt)
        rewrite_final = None if same_as_input else (rewrite_raw or None)
        if not rewrite_final:
            rewrite_final = base_rewrite or "No query rewrite suggestions were identified."

        return AnalyzeResponse(
            summary=f"({req.dbms.upper()} {version_label}) - {llm.get('summary')}",
            findings=dedupe((base_findings or []) + (llm.get("findings") or [])),
            rewrite_sql=rewrite_final,
            index_recommendations=dedupe((base_indexes or []) + (llm.get("index_recommendations") or [])),
            index_script=base_script,
            risks=dedupe((base_risks or []) + (llm.get("risks") or [])),
            test_steps=llm.get("test_steps") or [
                "Capture current plan & metrics (duration, CPU, reads).",
                "Apply one change at a time (index or rewrite).",
                "Compare estimated vs actual plans; validate row estimates.",
                "Benchmark on prod-like data; check regressions.",
            ],
        )

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI call failed: {e}")
