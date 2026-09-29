# Portfolio API (Phase 2)

Read-only FastAPI service that serves Phase-1 JSON facts under `data/`.

## Setup

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## Run

From the `backend/` directory:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Health: http://localhost:8000/health
- OpenAPI docs: http://localhost:8000/docs
- Projects: http://localhost:8000/api/projects

## Configuration

| Env var | Default | Notes |
|---|---|---|
| `DATA_DIR` | `backend/data` | Absolute path or path relative to `backend/` |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated browser origins |

Example:

```bash
set CORS_ORIGINS=http://localhost:3000,https://your-portfolio.vercel.app
```

## Docker

```bash
cd backend
docker build -t portfolio-api .
docker run --rm -p 8000:8000 portfolio-api
```

## Notes

- Nested `backend/.git` (if present) is left alone for this phase; the monorepo root is the source of truth for application work.
- Markdown under `data/knowledge/` is not served here (RAG comes in a later phase).
