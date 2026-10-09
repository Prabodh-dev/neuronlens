HACKERING 2.0 Round 2 Project Repository for Team Yvl (Open Innovation Track)

# NeuronLens

NeuronLens is a web app that shows which concepts a language model's learned features were active on in any text, and lets users test those concepts by steering (turning a feature up or down and comparing outputs). Model: GPT-2 small, layer 8, TopK sparse autoencoder (k=32, 12,288 features).

## Folder layout

- `contracts/`: `schema.sql`, `openapi.json`, and `fixtures/` (synthetic `sample.db` and its generator)
- `ml/`: ML code (SAE training, interpretation, runtime), added by Prabodh
- `server/`: FastAPI backend (Mirdul)
- `web/`: Next.js frontend (Anand)
- `demo/`: demo script, screenshots, backup recording (Anand)
- `docs/`: handbook, phases, PRD, app build guide, feature cards
- `data/`: local-only data and checkpoints (git-ignored)

## Run in mock mode

Mock mode reads the synthetic fixture database (`contracts/fixtures/sample.db`) and simulates Inspect and Steer. It needs no model, no checkpoint and no GPU, only `server/requirements.txt` for the server and Node for the web app.

```
python contracts/fixtures/make_fixture.py   # regenerate sample.db (optional)
pip install -r server/requirements.txt
uvicorn app.main:app --app-dir server --port 8000
```

Then, in a second terminal:

```
cd web
npm install
npm run dev
```

The web app opens on http://localhost:3000 and calls the server at `http://localhost:8000/api/v1` by default. See `server/README.md` and `web/README.md` for the rest.

## Running in real mode

Mock mode shows simulated data. Real mode serves the trained GPT-2 small layer 8 SAE: Features, Metrics and Runs come from the real database, and Inspect and Steer run the model on the GPU through `ml/neuronlens/runtime.py`.

**Not in git.** The real database and the checkpoint are not in the repository (`data/` is git-ignored, and so are `*.db` except the fixture, `*.pt` and `.env`). They are shared separately; put them under `data/`, for example `data/neuronlens.db` and `data/gpt2-l8-k32.pt` (the example path used in `.env.example`).

**Environment variables.** The server reads them from the process environment only; it does not load a `.env` file, so `export` them in the shell that starts the server.

| Variable                  | Meaning                                                  |
| ------------------------- | -------------------------------------------------------- |
| `NEURONLENS_MODE`         | `mock` (default) or `real`                               |
| `NEURONLENS_DB`           | database path; default `contracts/fixtures/sample.db`    |
| `NEURONLENS_CKPT`         | checkpoint path; real mode only                          |
| `NEURONLENS_CORS_ORIGINS` | comma-separated origins; default `http://localhost:3000` |

Relative paths are tried against the current directory, then the repository root. `~` is expanded.

**Python environment (WSL).** Real mode imports the `ml/` package inside the server process, so the server and the ML requirements go into the same environment. The ML code was developed in WSL2 with an NVIDIA GPU (see `ml/README.md`); `ml/pyproject.toml` requires Python 3.11 or newer (we used 3.12).

```
python3 -m venv ~/venvs/neuronlens
source ~/venvs/neuronlens/bin/activate
# install a CUDA build of PyTorch for your GPU (see pytorch.org), then:
pip install -r ml/requirements.txt     # torch, transformers, datasets, numpy, pyyaml, tensorboard
pip install -r server/requirements.txt
```

**Start the server (from the repository root).**

```
export NEURONLENS_MODE=real
export NEURONLENS_DB=data/neuronlens.db
export NEURONLENS_CKPT=data/gpt2-l8-k32.pt
uvicorn app.main:app --app-dir server --port 8000
```

**Start the web app** (second terminal):

```
cd web
npm install
NEXT_PUBLIC_API_BASE=http://localhost:8000/api/v1 npm run dev
```

**Check which mode is running.** Open http://localhost:8000/api/v1/health (or `curl` it). The response includes `mode` (`mock` or `real`), `is_synthetic` (`true` for the fixture, `false` for a real database), `run_id`, `d_sae` and `k`. `mode` is the configured mode, not proof that the model loaded: if the checkpoint or GPU memory is a problem, the server still starts, the database pages work, and Inspect and Steer return `MODEL_NOT_LOADED` with the reason (also written to the server log). A real-mode start with a good checkpoint logs `neuronlens ready: mode=real`.

**Example-token repair.** Example sentences in `feature_examples.tokens_json` used to be decoded one token at a time, which turns a character split across tokens (for example a curly apostrophe) into replacement characters (U+FFFD). New builds with `ml/scripts/make_index.py` decode them correctly. A database built before that change needs `ml/scripts/repair_example_tokens.py`, which rebuilds only the `feature_examples.tokens_json` column, by re-scanning the same data with the same checkpoint. It leaves every other table and column unchanged, and it checks each rebuilt example against the stored activations before writing.

```
cd ml
PYTHONPATH=. python scripts/repair_example_tokens.py --db ../data/neuronlens.db      # only reports the U+FFFD row count
PYTHONPATH=. python scripts/repair_example_tokens.py --rebuild-examples \
    --ckpt <checkpoint> --db ../data/neuronlens.db --data-dir ~/data/fineweb/sample/10BT --n-seqs 3000
```

Flags: `--db` (default `../data/neuronlens.db`), `--rebuild-examples` (required to change anything), `--ckpt` (the checkpoint the database was built from), `--data-dir` (default `~/data/fineweb/sample/10BT`), `--n-seqs` (use the value given to `make_index.py`; default 3000) and `--seed` (default 123, the seed `make_index.py` uses). Before writing, it copies the database to `<db>.bak2` and stops if that file already exists. It runs in one transaction and prints the number of rows containing U+FFFD before and after. Stop the server before running it, so nothing reads the file while it is rewritten.

## Claims policy

- Say "shows which concepts were active".
- Steering is a test, not a fix.
- Never write "proves", "detects", "fixes", "explains why", "first" or "novel".

## Honest limits

- GPT-2 small, one layer only.
- The strongest features are word-piece features.
- Some features are ambiguous.
- Steering too hard breaks text.

## Disclosures

**Prior work.** Before the event: the SAE training, evaluation, interpretation and steering code under `ml/`. Built during the event: the contracts, the API server, the web app, the real-mode runtime wiring, the example-token decoding fix and repair script, the evaluation and published-SAE comparison scripts, and the feature labelling. [Choose one and delete the other: The trained checkpoint and the real database were produced during the event. / The trained checkpoint existed before the event; the real database was built during it.] The git history only records when files were committed to this repository, so it cannot show what existed earlier.

**AI assistance.** AI tools used: Claude (coding sessions and drafting text and documentation) and Gemini (automatic feature labelling). [Add any tool Anand or Mirdul used for the web app and server.] The team reviewed the results and ran the code and tests.
