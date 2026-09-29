# Portfolio API + RAG (Phases 2–4)

Read-only FastAPI service that serves Phase-1 JSON facts under `data/`, plus a RAG ingestion/retrieve pipeline over Markdown knowledge (Qdrant + hosted embeddings via LiteLLM).

## Setup

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# Edit .env — set GEMINI_API_KEY for embeddings (and later LLM). Never commit .env.
```

## Run API

From the `backend/` directory:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Health: http://localhost:8000/health
- OpenAPI docs: http://localhost:8000/docs
- Projects: http://localhost:8000/api/projects

`/health` and portfolio JSON routes do **not** require Qdrant or Gemini.

## Configuration

| Env var | Default | Notes |
|---|---|---|
| `DATA_DIR` | `backend/data` | Absolute path or path relative to `backend/` |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated browser origins |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant HTTP API |
| `QDRANT_API_KEY` | _(empty)_ | Optional; local Docker usually needs none |
| `QDRANT_COLLECTION` | `portfolio_knowledge` | Single shared collection |
| `GEMINI_API_KEY` | _(required for embed)_ | Used by LiteLLM for Gemini models; keep in `.env` only |
| `EMBEDDING_MODEL` | `gemini/text-embedding-004` | Any LiteLLM embedding model id |
| `EMBEDDING_DIMENSIONS` | `768` | Must match the embedding model; omit/empty to probe on create |
| `LLM_MODEL` | `gemini/gemini-2.0-flash` | Placeholder for Phase 5 chat |

### Swapping embedding / LLM models

Change `EMBEDDING_MODEL` (and credentials if the new vendor differs). Ingest and retrieve both read the same setting.

**If you change `EMBEDDING_MODEL` or `EMBEDDING_DIMENSIONS`**, recreate the Qdrant collection and re-run ingest — old vectors are not comparable across models/sizes:

```bash
# Example: delete collection via Qdrant API, then re-ingest
curl -X DELETE "http://localhost:6333/collections/portfolio_knowledge"
python scripts/ingest.py
```

## RAG (Phase 4)

### 1. Start Qdrant locally

```bash
docker run --rm -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 2. Ingest knowledge Markdown

```bash
cd backend
python scripts/ingest.py
```

Reads `data/knowledge/**/*.md` → heading-aware chunks → embeds via the configured provider → upserts into `portfolio_knowledge` with deterministic IDs (re-runs are idempotent).

### 3. Smoke retrieve

```bash
python -c "from app.rag.retrieve import retrieve; print(retrieve('How does the OMS mobile app work?', top_k=5))"
python -c "from app.rag.retrieve import retrieve; print(retrieve('architecture', top_k=5, filters={'project': 'oms'}))"
```

## Docker (API only)

```bash
cd backend
docker build -t portfolio-api .
docker run --rm -p 8000:8000 portfolio-api
```

Qdrant is typically a separate container/service; point `QDRANT_URL` at it when running ingest or Phase 5 chat.

## Notes

- Nested `backend/.git` (if present) is left alone; the monorepo root is the source of truth for application work.
- Markdown under `data/knowledge/` is ingested into Qdrant for retrieval; it is not exposed via the portfolio JSON routes.
- Chat HTTP (`POST /api/chat`) arrives in Phase 5; this phase only provides providers + ingest + `retrieve()`.
