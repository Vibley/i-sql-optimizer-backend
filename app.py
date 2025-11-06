import os, json, re
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sqlparse

# --- Remove proxy envs that could confuse OpenAI/httpx ---
for k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "OPENAI_PROXY"]:
    os.environ.pop(k, None)

# Optional: log OpenAI SDK version at startup (helps verify the pinned SDK on Render)
try:
    import openai  # noqa: F401
    import logging
    logging.getLogger("uvicorn.error").info(f"OpenAI SDK version: {openai.__version__}")
except Exception:
    pass

ALLOW_ORIGIN = os.getenv("ALLOW_ORIGIN", "*")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="AI SQL Optimizer Backend", version="1.2.2")

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
    risks: List[str] = Field(default_factory=list)
    test_steps: List[str] = Field(default_factory=list)

# ---------- Static rules + simple rewrites ----------
def static_rules(sql: str):
    """
    Returns:
      findings: List[str]
      rewrite_out: Optional[str]  (either a concrete rewrite or guidance comments)
      index_recs: List[str]
      risks: List[str]
    """
    findings, rewrites, index_recs, risks = [], [], [], []
    sql_norm = sql.strip()
    sql_compact = re.sub(r"\s+", " ", sql_norm, flags=re.MULTILINE).upper()

    # SELECT *
    if re.search(r"\bSELECT\s+\*\b", sql_compact):
        findings.append("Avoid SELECT *. Project only required columns.")
        risks.append("Extra I/O and wider rows reduce buffer cache efficiency.")
        rewrites.append("-- Replace SELECT * with only required columns.")

    # Leading wildcard LIKE
    if re.search(r"LIKE\s+['\"]%[^'\"]+['\"]", sql_compact):
        findings.append("Leading wildcard LIKE prevents index seeks.")
        rewrites.append("-- Consider full-text index (CONTAINS) or trigram search.")
        risks.append("Full scans on large tables can be expensive.")

    # Non-sargable function on column
    if re.search(r"WHERE\s+.*\b(YEAR|MONTH|DAY|DATEADD|DATEDIFF|SUBSTRING|CAST|CONVERT)\s*\(", sql_compact):
        findings.append("Non-sargable predicate (function on column) blocks index seeks.")
        rewrites.append("-- Rewrite to a range predicate on the raw column when possible.")

    # OR conditions
    if re.search(r"\bWHERE\b.*\bOR\b", sql_compact):
        findings.append("OR conditions may reduce index usage; consider UNION ALL or indexed computed columns.")

    # ORDER BY without JOIN
    if "ORDER BY" in sql_compact and "JOIN" not in sql_compact:
        findings.append("ORDER BY detected; ensure index supports ORDER BY key(s).")

    # JOIN without WHERE
    if "WHERE" not in sql_compact and "JOIN" in sql_compact:
        findings.append("JOIN without WHERE may explode rows; verify join predicates and filters.")

    # ---------- Simple static rewrite candidates ----------
    base_rewrite_sql = None

    # Heuristic: WHERE YEAR(col) = 2024  -->  col >= '2024-01-01' AND col < '2025-01-01'
    year_eq = re.search(
        r"\bYEAR\s*\(\s*(?P<col>[A-Za-z0-9_\.\[\]]+)\s*\)\s*=\s*(?P<yyyy>20\d{2}|19\d{2})",
        sql_norm,
        flags=re.IGNORECASE,
    )
    if year_eq:
        col = year_eq.group("col")
        yyyy = int(year_eq.group("yyyy"))
        start = f"'{yyyy:04d}-01-01'"
        next_year = f"'{(yyyy+1):04d}-01-01'"
        range_pred = f"{col} >= {start} AND {col} < {next_year}"
        base_rewrite_sql = re.sub(
            r"\bYEAR\s*\(\s*"+re.escape(col)+r"\s*\)\s*=\s*"+str(yyyy),
            range_pred,
            sql_norm,
            count=1,
            flags=re.IGNORECASE,
        )
        if "Non-sargable predicate" not in " ".join(findings):
            findings.append("Non-sargable predicate (function on column) blocks index seeks.")
        rewrites.append("-- Replaced YEAR(col)=YYYY with a sargable date range.")

    # ---------- Index key guess based on equality predicates ----------
    m = re.findall(r"\b([A-Z_][A-Z0-9_\.]+)\s*=\s*[@:\w'\-]+", sql_compact)
    if m:
        cols = [col.split(".")[-1].lower() for col in m]
        cols = list(dict.fromkeys(cols))[:3]  # de-dupe, keep first 3

        # Try to extract table name from query (FROM first, then JOIN), then lowercase
        tbl_match = re.search(r"\bFROM\s+([A-Z0-9_\.\[\]]+)", sql_compact)
        if not tbl_match:
            tbl_match = re.search(r"\bJOIN\s+([A-Z0-9_\.\[\]]+)", sql_compact)
        table_name = tbl_match.group(1).lower() if tbl_match else "<yourtable>"

        if cols:
            index_recs.append(
                f"create index ix_{cols[0]}_suggested on {table_name} ({', '.join(cols)});"
            )

    guidance = "\n".join(rewrites) if rewrites else None
    rewrite_out = base_rewrite_sql or guidance  # prefer concrete rewrite, fallback to comments
    return findings, rewrite_out, index_recs, risks

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

    base_findings, base_rewrite, base_indexes, base_risks = static_rules(sql_fmt)

    # If no API key, return static analysis only
    if not OPENAI_API_KEY:
        rewrite_text = base_rewrite or "No query rewrite suggestions were identified."
        return AnalyzeResponse(
            summary="Static analysis completed (OpenAI not configured).",
            findings=base_findings or ["No obvious issues detected by static rules."],
            rewrite_sql=rewrite_text,
            index_recommendations=base_indexes,
            risks=base_risks,
            test_steps=[
                "Capture current plan & metrics (duration, CPU, reads).",
                "Apply one change at a time (index or rewrite).",
                "Compare estimated vs actual plans; validate row estimates.",
                "Benchmark on prod-like data; check regressions.",
            ],
        )

    # With OpenAI: Chat Completions (JSON mode)
    try:
        import httpx
        from openai import OpenAI

        # httpx client that ignores any *_PROXY env vars
        http_client = httpx.Client(trust_env=False, timeout=30.0)
        client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)

        system_msg = (
            f"You are a veteran {req.dbms} performance engineer. "
            f"Return safe, actionable tuning advice. Use <YourTable> placeholders; never invent schema names."
        )
        plan = (req.plan_xml or "")[:20000]  # truncate to keep request bounded

        # Build prompt (avoid triple-quoted f-strings)
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

        # Merge + dedupe helpers
        def dedupe(seq):
            seen, out = set(), []
            for s in seq or []:
                if s not in seen:
                    seen.add(s); out.append(s)
            return out

        # Strict guardrail to suppress echo rewrites
        def _canon_sql(s: str) -> str:
            s = (s or "")
            # strip code fences like ```sql ... ``` or ``` ...
            s = re.sub(r"```(?:sql)?", "", s, flags=re.IGNORECASE)
            s = s.replace("`", "")
            s = s.strip()
            if s.endswith(";"):
                s = s[:-1]
            s = re.sub(r"\s+", " ", s).strip().lower()
            return s

        rewrite_raw   = llm.get("rewrite_sql") or ""
        orig_canon    = _canon_sql(sql_fmt)
        rewrite_canon = _canon_sql(rewrite_raw)

        # If identical after canonicalization, discard it
        if rewrite_canon and rewrite_canon == orig_canon:
            rewrite = None
        else:
            rewrite = rewrite_raw or None

        # Prefer our static rewrite if present; otherwise show an explicit message
        if not rewrite:
            rewrite = base_rewrite or "No query rewrite suggestions were identified."

        findings = dedupe((base_findings or []) + (llm.get("findings") or []))
        indexes  = dedupe((base_indexes  or []) + (llm.get("index_recommendations") or []))
        risks    = dedupe((base_risks    or []) + (llm.get("risks") or []))
        summary  = llm.get("summary") or "Analysis completed."
        steps    = llm.get("test_steps") or [
            "Capture current plan & metrics (duration, CPU, reads).",
            "Apply one change at a time (index or rewrite).",
            "Compare estimated vs actual plans; validate row estimates.",
            "Benchmark on prod-like data; check regressions.",
        ]

        return AnalyzeResponse(
            summary=summary,
            findings=findings or ["No obvious issues found."],
            rewrite_sql=rewrite,
            index_recommendations=indexes,
            risks=risks,
            test_steps=steps,
        )

    except Exception as e:
        base_findings.append(f"AI enhancer unavailable: {e}")
        rewrite_text = base_rewrite or "No query rewrite suggestions were identified."
        return AnalyzeResponse(
            summary="Static analysis completed (LLM call failed).",
            findings=base_findings,
            rewrite_sql=rewrite_text,
            index_recommendations=base_indexes,
            risks=base_risks,
            test_steps=[
                "Capture current plan & metrics (duration, CPU, reads).",
                "Apply one change at a time (index or rewrite).",
                "Compare estimated vs actual plans; validate row estimates.",
                "Benchmark on prod-like data; check regressions.",
            ],
        )
