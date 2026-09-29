# Portfolio API + RAG + Chat + Voice (Phases 2–7)

FastAPI service that serves Phase-1 JSON facts under `data/`, a RAG ingestion/retrieve pipeline over Markdown knowledge (Qdrant + hosted embeddings via LiteLLM), `POST /api/chat` (JSON + SSE), and `POST /api/voice/transcribe` (speech-to-text only).

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
# Edit .env — set Gemini + NVIDIA NIM keys. Never commit .env.
```

## Run API

From the `backend/` directory:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Health: http://localhost:8000/health
- OpenAPI docs: http://localhost:8000/docs
- Projects: http://localhost:8000/api/projects
- Chat: `POST http://localhost:8000/api/chat`
- Voice: `POST http://localhost:8000/api/voice/transcribe` (multipart `file`)

`/health` and portfolio JSON routes do **not** require Qdrant, Gemini, or NVIDIA STT.

## Configuration

| Env var | Default | Notes |
|---|---|---|
| `DATA_DIR` | `backend/data` | Absolute path or path relative to `backend/` |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated browser origins |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant HTTP API |
| `QDRANT_API_KEY` | _(empty)_ | Optional; local Docker usually needs none |
| `QDRANT_COLLECTION` | `portfolio_knowledge` | Single shared collection |
| `GEMINI_API_KEY` | _(required for embed + chat)_ | Also accepts `GEMINAI_API_KEY` / `GENAI_API_KEY` |
| `EMBEDDING_MODEL` | `gemini/gemini-embedding-001` | Any LiteLLM embedding model id |
| `EMBEDDING_DIMENSIONS` | `768` | Passed to the provider when set; must match the collection |
| `GEMINI_LLM_API_MODEL` | `gemini/gemini-2.0-flash` | Free-tier Gemini chat default; `LLM_MODEL` is an alias |
| `NVIDIA_NIM_API_KEY` | _(required for voice)_ | NVIDIA build.nvidia.com / NIM key (`nvapi-…`) |
| `NVIDIA_NIM_API_BASE` | `https://integrate.api.nvidia.com/v1` | Chat/models catalog base (not used for Parakeet STT) |
| `STT_MODEL` | `nvidia/parakeet-ctc-1.1b-asr` | Label for the configured STT model |
| `STT_API_BASE` | Parakeet NVCF `…/v1` URL | OpenAI-compatible `/audio/transcriptions` base |
| `STT_LANGUAGE` | `en-US` | Passed to the ASR endpoint |
| `STT_MAX_UPLOAD_BYTES` | `5242880` (5 MiB) | Reject larger uploads with 413 |

### Swapping embedding / LLM / STT models

- **Chat only:** change `GEMINI_LLM_API_MODEL` (or `LLM_MODEL`) and credentials if the vendor differs.
- **Embeddings:** change `EMBEDDING_MODEL` (and credentials if needed).
- **Voice STT:** change `STT_API_BASE` / `STT_MODEL` / `NVIDIA_NIM_API_KEY` — callers stay on `TranscriptionProvider`.

**If you change `EMBEDDING_MODEL` or `EMBEDDING_DIMENSIONS`**, recreate the Qdrant collection and re-run ingest — old vectors are not comparable across models/sizes:

```bash
curl -X DELETE "http://localhost:6333/collections/portfolio_knowledge"
python scripts/ingest.py
```

## RAG (Phase 4) — prerequisite for useful chat

### 1. Start Qdrant locally

```bash
docker run --rm -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 2. Ingest knowledge Markdown

```bash
cd backend
python scripts/ingest.py
```

### 3. Smoke retrieve

```bash
python -c "from app.rag.retrieve import retrieve; print(retrieve('How does the OMS mobile app work?', top_k=5))"
```

## Chat (Phase 5)

Ingest knowledge first. Then:

### JSON (non-streamed)

```bash
curl -s http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"How does the OMS mobile app work?"}'
```

### SSE (streamed)

```bash
curl -N "http://localhost:8000/api/chat?stream=true" \
  -H "Content-Type: application/json" \
  -d '{"query":"How does the OMS mobile app work?"}'
```

Events: `token`, `sources`, `done`. Empty retrieval short-circuits without calling the LLM.

## Voice transcription (Phase 7)

Answers still go through `/api/chat`. This endpoint only returns `{ "text": "..." }`.

```bash
curl -s http://localhost:8000/api/voice/transcribe \
  -F "file=@sample.wav"
```

Requires `NVIDIA_NIM_API_KEY` and a reachable `STT_API_BASE` (default: NVIDIA Parakeet NVCF). Upload size is capped by `STT_MAX_UPLOAD_BYTES`. No TTS.

## Docker (API only)

```bash
cd backend
docker build -t portfolio-api .
docker run --rm -p 8000:8000 portfolio-api
```

## Notes

- Nested `backend/.git` (if present) is left alone; the monorepo root is the source of truth.
- LangGraph / agent routing is still a later blueprint phase.
