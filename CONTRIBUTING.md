# Contributing

Bug reports, fixes, tests and documentation are welcome. For a larger change, open an issue first so the approach can be agreed before you write it.

## Layout

- `server/`: FastAPI back end (mock mode and real mode)
- `web/`: Next.js front end
- `ml/`: SAE training, evaluation, indexing, runtime and steering code
- `contracts/`: `schema.sql`, `openapi.json` and a synthetic sample database. The server and the web app are built against these.
- `docs/`: feature cards, `decisions.md`, data and model notes

## Set up

You do not need a GPU, a checkpoint or a real database to work on the server or the web app. Mock mode reads `contracts/fixtures/sample.db`.

**Back end** (Python 3.11 or newer):

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r server/requirements.txt
uvicorn app.main:app --app-dir server --port 8000
```

**Front end** (Node 22 and npm; CI uses Node 22):

```
cd web
npm install
npm run dev
```

The app opens on http://localhost:3000 and calls `http://localhost:8000/api/v1` by default. Set `NEXT_PUBLIC_API_BASE` to change that.

**ML** (needed only for `ml/`, and for real mode, which imports it):

```
pip install torch                  # a CPU build is enough for the tests; use a CUDA build to train
pip install -r ml/requirements.txt
```

Training, indexing and real mode need a GPU, the FineWeb-Edu text and a checkpoint. See [ml/README.md](ml/README.md), [docs/demo-startup.md](docs/demo-startup.md) and [docs/data-and-models.md](docs/data-and-models.md).

## Run the tests

From the repository root:

```
# server
python -m pytest server/tests

# ml, on CPU, without the slow evaluation test
cd ml && python -m pytest tests --ignore=tests/test_evals.py

# web
cd web && npm run typecheck && npm run build
```

`ml/tests/test_evals.py` runs the evaluation code against a real model and is slow, so it is left out of CI and of the routine run. Run it when you change `ml/neuronlens/evals.py`.

All of these must pass before you open a pull request. CI runs the same commands (`.github/workflows/ci.yml`).

## Branches and pull requests

1. Fork the repository, or create a branch if you have write access. Branch from `main`.
2. Keep one change per branch.
3. Run the tests above. If you changed anything under `web/`, run the build too.
4. Open a pull request into `main`, fill in the template, and link the issue if there is one.
5. The maintainer reviews and merges. Expect questions; reply on the thread and push follow-up commits to the same branch.

A good pull request:

- does one thing, and says in the description what changed and why;
- says how you tested it (mock mode, real mode, or tests only);
- adds or updates a test when it changes behaviour;
- leaves the contracts alone unless the change is about them. A change to `contracts/` (`schema.sql`, `openapi.json`) needs an issue first, because the server, the web types (`npm run gen:types`) and the ML build all depend on it;
- keeps the diff small, with no unrelated formatting or renames;
- updates the docs it touches.

## Commit messages

The history uses `type(scope): message` in lower case, for example `feat(web): model selector with a turning sphere per model` or `docs: repo-ready pass`.

- type: `feat`, `fix`, `docs`, `test`, `chore`
- scope: `ml`, `server`, `web`, `contract`, `demo`; leave it out when a change spans several

Write the message about what the commit does, in the present tense.

## Never commit

- `.db` files, except `contracts/fixtures/sample.db`
- `.pt` checkpoints
- `.env` files (use `.env.example` as the template) and API keys
- `node_modules/`, `.next/`, `data/` contents

Databases and checkpoints are large and are built, not stored; see [docs/data-and-models.md](docs/data-and-models.md).

## Wording rules

The app and the docs make claims about a model's internals, so the wording is checked in review.

- Say a feature "shows which concepts were active", not that it "detects" or "explains why".
- Steering is a test of what a feature does, not a fix.
- Say "tuned", not "improved". Do not write "proves", "fixes", "first" or "novel".
- Do not say one model is better than another. Each model's numbers are its own measurements and are not compared.
- State limits next to results: one prompt, one seed, auto-generated labels, one layer.
- Do not use dates such as "this week" or "yesterday" in docs.
- Write plainly and use the project's own terms. No marketing words, no emoji, no sentence that only repeats its heading.

## Code style

Match the surrounding code. There is no formatter configured; keep the diff to the lines you mean to change. New Python code needs tests under `server/tests/` or `ml/tests/`.
