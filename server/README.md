# Server

Owner: Mirdul. Branch: `mirdul`.

FastAPI server. Mock mode (default) reads `contracts/fixtures/sample.db` read-only.

Rule: build against `contracts/schema.sql`; mock mode first.

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
