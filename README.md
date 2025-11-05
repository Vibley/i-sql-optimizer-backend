# AI SQL Optimizer Backend (FastAPI + OpenAI)

## Local run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
set OPENAI_API_KEY=sk-...   # Windows (PowerShell: $env:OPENAI_API_KEY='sk-...')
export ALLOW_ORIGIN=https://<your-username>.github.io
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Open http://localhost:8000/docs

## Deploy on Render (free)
- Build: `pip install -r requirements.txt`
- Start: `uvicorn app:app --host 0.0.0.0 --port 10000`
- Env Vars:
  - `OPENAI_API_KEY` = your key
  - `ALLOW_ORIGIN`  = `https://<your-username>.github.io`

## API
- `GET /health` → `{"status":"ok"}`
- `POST /analyze` → returns JSON: summary, findings, rewrite_sql, index_recommendations, risks, test_steps
