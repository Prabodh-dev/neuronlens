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
- For search, send `max_density: 0.01`. Without it, very common features dominate the results: in the real-mode check, four dense features (end of text, gast-, exxon, were) filled the top 10 of every query. With 0.01, "music", "court ruling" and "recipe for pasta" all found sensible features, and a leading space on the first word made no difference.
- A newline is put before the query so its first word is not token 0 (the outlier position, which has no features). The prefix is never returned in `tokens` and does not count toward the 32-token limit. Without a leading space, the first word is a different GPT-2 token than the same word mid-sentence.
- Real-mode check: `python server/scripts/w8_check.py` (needs `NEURONLENS_MODE=real`, `NEURONLENS_DB`, `NEURONLENS_CKPT`).

Multi-concept steering (W5):

- `POST /steer/multi`: amplify 2 or 3 different features together (same seed as the baseline), each with `strength` 0 to 3, **default 0.5**. Returns both texts, `differs`, `first_divergence_index`, `degenerate` and `reason`, and per feature its activation on the finished texts and a verdict. It shows which concepts were active; steering is a test of what features do. `with_single_controls: true` also runs each feature alone (one more generation each).
- Higher strengths often loop: several features start repeating at 1.0 and above, and the steers add together. Check `degenerate`.
- The `single_feature_runs` texts can contain harsh content, so a UI should show only their activation numbers, not the texts.
- Mock: a request that matches the recorded example in `demo/phase4_examples.json` (`multi`) returns it, read straight from that file; any other request gets a stand-in. Both are `simulated: true`. Real-mode check: `python server/scripts/w5_check.py`.

Auto-Lab (the UI runs this chain client-side; the server adds nothing but `with_activations`):

- `POST /steer/sweep` takes optional `with_activations: true` (default false): each result also gets `target_activation_baseline` and `target_activation_steered`, the same numbers as `/steer/check`. No extra generations. Off, the response is unchanged.
- The chain, seed 42: `/features/search` with the goal words (`top_n` 5, `max_density` 0.01), then `/steer/sweep` with `with_activations` for each of the 5 features at strengths 0, 0.25, 0.5, 0.75, 1, 1.5, 2 (0 is the unsteered control).
- Scoring per feature and strength. Rise = steered activation - unsteered activation. **Worked (✓)**: rise ≥ 3.0 and steered ≥ 2× unsteered, and the text does not loop (when unsteered is near 0, the 2× part holds by itself). **Breaks down (⚠️)**: the text loops (`degenerate`). **No effect (✗)**: anything else.
- Recommendation (rule v2), strength 0 (the control) left out. A ✓ cell is **stable** if the strength next to it (before or after) is also ✓. Only features with at least one stable ✓ can be recommended; among them, the one with the most ✓ cells (tie: the higher rise at its best cell), at its lowest stable ✓ strength. ✓ cells but none stable: "Worked at one strength only, not reliable", no recommendation. No ✓ at all: "Nothing worked". A goal **ships** if it has a recommendation on at least 2 of its 3 prompts. These thresholds are tuned on a few prompts and provisional until the laptop test; they are named constants at the top of `server/scripts/autolab_check.py`, and the UI uses the same values.
- What "worked" means: the feature fired more on the finished text at that strength without the text looping. Steering is a test, not a fix; it does not mean the text is better.
- Check: `python server/scripts/autolab_check.py --goal "music" --prompt "The band walked on stage and" -v` prints the features × strengths table, the recommendation and the time. `--goals-file goals.txt` runs several goals (one per line, `goal words | prompt`), `--repeat 5` runs them back to back for timing (target under 35 s a chain; it warns over 40 s and lists any failed or timed-out call), `--json` saves the results. `--replay demo/autolab_examples.json` re-scores a recorded file with the current rule, with no server (the file keeps the raw search and sweep responses). `--fillers` lists features in the top 5 for 2 or more goals and the query token each peaked on (token 0 is the first query word, right after the newline prefix). `--search-only`, `--sort relative` and `--leading-space` are search experiments, not the UI chain; they cannot be recorded. The default URL is `http://127.0.0.1:8000`: on Windows, `localhost` adds about 2 s to every call.
- Mock: search and sweep requests that match a goal in `demo/autolab_examples.json` return the recorded real responses, marked simulated; anything else gets a stand-in. That file is written on the real server with `autolab_check.py --goals-file goals.txt --record demo/autolab_examples.json` (it refuses a mock server), and a test fails if the mock and the file differ.
- `demo/autolab_examples.json` was recorded on Prabodh's laptop (real server, run gpt2-l8-k32, layer 8, seed 42): 6 goals × 3 prompts.

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
