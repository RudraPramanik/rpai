# Portfolio answer-trust evals

Manual Phase-9 checklist for `POST /api/chat`. There is no automated CI harness in V1.

## Files

| File | Purpose |
|---|---|
| `portfolio_questions.json` | 20–30 questions with `expected_tool` / `expected_sources` / `expect_insufficient` |
| `CHECKLIST.md` | Blank pass/fail table keyed by question `id` |

## Prerequisites

1. Backend running (`uvicorn` / your usual `backend` start).
2. Qdrant populated (`python scripts/ingest.py` if needed).
3. LLM env configured (same `.env` as normal chat).
4. Logs visible at INFO so you can see `agent route question_class=…`, `agent tool name=…`, `knowledge_chunks=…`, and `agent synthesize complete…`.

## How to run one question

```bash
curl -s -X POST "http://localhost:8000/api/chat" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\": \"How does the OMS mobile app work?\"}"
```

On Unix/macOS use `\` line continuations instead of `^`.

Optional batch helper (still manual grading — not CI): from `backend/`, with the API up:

```bash
python evals/run_once.py
```

That writes `evals/_last_run.json` (gitignored) with answers for checklist fill-in.

Inspect:

1. **Answer** — grounded, or insufficient-information phrasing when `expect_insufficient` is true.
2. **`sources`** — non-empty only when knowledge search returned chunks; empty for profile-only turns; never fabricated paths.
3. **Server logs** — `question_class` and tool names match `expected_tool` / `expected_tools` when those fields are set.
4. **Citation spot-check** (sample of knowledge-backed answers) — open the cited Markdown section and confirm it supports the claim.

## Recording results

1. Copy `CHECKLIST.md` (or clear the Result / Notes columns).
2. For each `id` in `portfolio_questions.json`, mark **pass** or **fail**.
3. Re-run the full set after any material change to synthesis/system prompts, routing, retrieval, or citation assembly.

## Pass criteria (quick)

| Category | Pass when |
|---|---|
| `profile` / `engineering` | Uses `get_profile` (no unnecessary knowledge search); facts match JSON; null fields → not documented |
| `project` | Tools cover that project; `sources` reference that project’s knowledge when chunks exist |
| `comparative` | Tools cover both projects; answer compares them |
| `unanswerable` | Insufficient-documented phrasing; no invented salary, food, awards, compensation, or tech |

## Tracing

Structured INFO logs are the default operator trace (query → route → tools → synthesize). Langfuse is optional and not required for this eval workflow or for `/health`.
