# Server

Owner: Mirdul. Branch: `mirdul`.

FastAPI server. Mock mode (default) reads `contracts/fixtures/sample.db` read-only.

Rule: build against `contracts/schema.sql`; mock mode first.

Text limit: `/inspect` text and `/steer` prompt are capped at 256 tokens (`TEXT_TOO_LONG`). Mock mode counts words, real mode counts GPT-2 tokens, so a text near the limit can pass in mock and fail in real.

Phase 4 steering checks (fields in `docs/phase4-ml-api.md`; the code is `ml/neuronlens/steering_checks.py`):

- `POST /steer/check` (X1): baseline vs steered plus the target feature's activation on each finished text (re-encoded without steering) and a verdict: higher, lower or about the same. It says whether the feature fires on the result, not whether the output is better.
- `POST /steer/compare` (X3): seed-matched baseline vs steered, compared token by token.
- `POST /steer/sweep` (X2): amplify at several strengths (default 0 to 2, each at most 3), with a degeneration flag.
- `POST /steer/prompt-only` (X4): the prompt alone, same seed, no steering.

Inspect analysis (aggregate `/inspect`; no new ML):

- `POST /inspect/report` (X6): per sentence, the features with the highest activation on its tokens, with `tokens_active`, `peak_token` and `relative_to_max` (activation / the feature's dictionary max). Split after `.` `!` `?` followed by whitespace or the end, and after newlines. Known wrong splits: "Mr. Smith", "U.S. troops", "e.g. this" split early; "end.Next" is not split.
- `POST /inspect/compare` (X7): features ranked by |max activation in text_b - in text_a|; 0 where a feature is not active.

Both call inspect with the SAE's k (every active feature), then rank and trim to `top_n`. Optional `max_density` (0 to 1, off by default) hides features whose `density` (share of tokens they fire on) is above it before ranking, so very common features such as "the – noun phrase start" do not crowd out specific ones. Optional `sort`: `activation` (default, raw max activation) or `relative` (by `relative_to_max`, how strongly each feature fired for that feature; for compare, by |relative_to_max in b - in a|). Features with no dictionary max go last.

Search (W8):

- `POST /features/search`: the features that fired most on the words of the query (1 to 32 query tokens), with `max_activation`, `relative_to_max`, `tokens_active`, `peak_token` and `density`. Same `top_n`, `max_density` and `sort` options as X6/X7. It shows which features fired on the query, not what the query means. Keyword search on labels is `GET /features?query=`.
- A newline is put before the query so its first word is not token 0 (the outlier position, which has no features). The prefix is never returned in `tokens` and does not count toward the 32-token limit. Without a leading space, the first word is a different GPT-2 token than the same word mid-sentence.
- Real-mode check: `python server/scripts/w8_check.py` (needs `NEURONLENS_MODE=real`, `NEURONLENS_DB`, `NEURONLENS_CKPT`).

Each check holds the model for all its generations (same lock as `/steer`). Rough timing (real mode, warm server, Prabodh's laptop; estimates from two measurements, not benchmarks): about 0.65 s per generation, so a check takes about 1.5 s and a 7-point sweep about 5 s. A call right after server start may be slower. Clients allow 30 s (60 s for a sweep) as generous limits; see Timing in `docs/phase4-ml-api.md`. In mock mode, inputs that match `demo/phase4_examples.json` return those real outputs, still marked `[simulated]`; other inputs get a filler stand-in. Mock tokens are a word split, so counts and divergence indices differ from real mode.

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
- `NEURONLENS_COFIRE`: co-fire file for `GET /features/{id}/cofire` (W4). Default `data/cofire.json` in real mode, `server/app/fixtures/mock_cofire.json` (features 7 and 13 only) in mock mode. If it is missing or built for another run, the server still starts, logs one warning, and the endpoint returns `available: false`.
