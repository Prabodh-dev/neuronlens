# Server

Owner: Mirdul. Branch: `mirdul`.

FastAPI server. Mock mode (default) reads `contracts/fixtures/sample.db` read-only.

Rule: build against `contracts/schema.sql`; mock mode first.

Text limit: `/inspect` text and `/steer` prompt are capped at 256 tokens (`TEXT_TOO_LONG`). Mock mode counts words, real mode counts GPT-2 tokens, so a text near the limit can pass in mock and fail in real.

Phase 4 steering checks (fields in `docs/phase4-ml-api.md`; the code is `ml/neuronlens/steering_checks.py`):

- `POST /steer/compare` (X3): seed-matched baseline vs steered, compared token by token.
- `POST /steer/sweep` (X2): amplify at several strengths (default 0 to 2, each at most 3), with a degeneration flag.
- `POST /steer/prompt-only` (X4): the prompt alone, same seed, no steering.

Each check holds the model for all its generations (same lock as `/steer`). Measured on Prabodh's laptop: about 3 to 4 s for a check, 12 to 15 s for a 7-point sweep; clients should allow 30 s (60 s for a sweep). In mock mode, inputs that match `demo/phase4_examples.json` return those real outputs, still marked `[simulated]`; other inputs get a filler stand-in. Mock tokens are a word split, so counts and divergence indices differ from real mode.

## Run (from the repo root)

```
pip install -r server/requirements.txt
cd server
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs. API base path: `/api/v1`.

## Test

```
cd server
python -m pytest
```

## Regenerate the API contract

```
python server/scripts/export_openapi.py   # writes contracts/openapi.json
```

## Environment (see `.env.example`)

- `NEURONLENS_MODE`: `mock` (default) or `real`
- `NEURONLENS_DB`: default `contracts/fixtures/sample.db`
- `NEURONLENS_CKPT`: real mode only
- `NEURONLENS_CORS_ORIGINS`: comma-separated, default `http://localhost:3000`
