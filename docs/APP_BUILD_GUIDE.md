# NeuronLens — App Track Build Guide

| | |
|---|---|
| **Audience** | The App track: server, web and QA. The ML track does not need this file. |
| **Version** | 1.0 |
| **Lives at** | `docs/APP_BUILD_GUIDE.md` |
| **Companion** | `docs/PRD.md` (the product requirements) |
| **Authority** | For app work, this guide wins over the PRD. If you find a conflict, follow this guide and record it in `docs/decisions.md`. |

---

## 0. Read this first

This guide contains everything the App track needs to build the server and the web app **completely**, without asking the ML Lead anything. It tells you:

- what to do **before** the template repository is pushed (nothing, except install tools),
- what to check **after** it is pushed,
- the exact order to build things in, and when each stage is finished,
- the exact data contract, API, algorithms, design and wording,
- what to do when the real data arrives from the ML track.

How to use it: read sections 1 to 4 completely now. Then, for each stage in section 4, read the sections it points to and build exactly that stage.

There are no dates in this guide on purpose. Work is ordered by **stages and gates**. A gate is a condition that must be true before you start the next stage.

---

## 1. What you are building

### 1.1 In plain words

A language model keeps lists of numbers inside it. Nobody can read them. The ML track trains a small helper network that translates those numbers into a **dictionary of features**. A feature is one concept the helper found, for example "text about cooking".

You are building the tool people use to look at that dictionary. It is two programs:

- **server** (Python, FastAPI): reads one SQLite file and answers HTTP requests.
- **web** (Next.js): a browser app with five screens.

| Screen | Route | What the user does |
|---|---|---|
| Home | `/` | Reads what the tool is; sees which model and run are loaded |
| Feature Explorer | `/features` | Searches, sorts and filters the feature dictionary |
| Feature Detail | `/features/[id]` | Reads the example texts that make one feature fire |
| Text Inspector | `/inspect` | Pastes text and sees which features fire on each word |
| Steering Panel | `/steer` | Turns one feature up or down and compares two outputs |
| Metrics | `/metrics` | Reads how faithful the helper network is |

### 1.2 Words you need

| Word | Meaning |
|---|---|
| **Token** | A chunk of text, roughly a word or part of a word. The model reads tokens, not letters. |
| **Activation** | How strongly a feature fires on a token. A number, 0 or more. Bigger means stronger. |
| **Feature** | One entry in the dictionary, identified by `feature_id`. May have a human-written label. |
| **Label** | A short human-written name for a feature, for example "cooking". Some features have none. |
| **Density** | The fraction of tokens on which a feature fires. `0.01` means 1% of tokens. |
| **Dead feature** | A feature that never fires. It exists in the dictionary but is useless. |
| **Steering** | Forcing a feature up (`amplify`) or off (`ablate`) while the model writes text, then comparing with normal output. |
| **Run** | One training run of the helper network. The database describes one primary run. |
| **FVU, CE-recovered** | Quality numbers for the helper network. You only display them. |

### 1.3 Your boundaries

You build and own:

- `server/`
- `web/`
- `contracts/fixtures/` (the fake-data generator and its output)
- `Makefile`, `.github/workflows/ci.yml`, `scripts/`
- `docs/decisions.md`

You must **not** change:

- `ml/` (the ML track's code)
- `contracts/schema.sql` (the database contract; changing it needs both leads, see section 3)

You never need a GPU, CUDA or PyTorch. You must never commit a real database, a checkpoint or any model file.

---

## 2. The gate: before and after the push

### 2.1 Before the push

**You cannot start building until the ML Lead posts the message "template pushed" in the team chat.** Until then:

Allowed:
- Read this guide and `docs/PRD.md` if you have a copy.
- Install the tools in section 2.2 and confirm the version commands work.

Not allowed:
- Creating a repository, branches or code.
- Guessing at the schema, the API or the design and building ahead of the template.

### 2.2 Tools to install

| Tool | Version | Check |
|---|---|---|
| Git | any recent | `git --version` |
| Python | **3.11** | `python3 --version` (Windows: `py -3.11 --version`) |
| Node.js | **20 LTS** | `node --version` should print `v20.x` |
| npm | comes with Node 20 | `npm --version` |
| A Chromium-based browser | any recent | for manual checks |
| `make` | optional | `make --version`; if missing use the raw commands in section 5.3 |
| SQLite CLI | optional | useful for looking at the database |

### 2.3 Readiness check after the push

Run this once, right after "template pushed":

```bash
git clone <repo-url> neuronlens && cd neuronlens
git log --oneline -1
ls README.md docs/PRD.md docs/APP_BUILD_GUIDE.md contracts/schema.sql \
   ml/neuronlens/runtime.py .gitignore .env.example server/README.md web/README.md
python3 -c "import sqlite3; c=sqlite3.connect(':memory:'); c.executescript(open('contracts/schema.sql').read()); print('schema ok')"
python3 --version
node --version
```

You should see `schema ok`, Python 3.11.x and Node v20.x.

**The template is expected to contain exactly these files and folders:**

| Path | Purpose |
|---|---|
| `README.md` | Project overview |
| `docs/PRD.md` | Product requirements |
| `docs/APP_BUILD_GUIDE.md` | This guide |
| `contracts/schema.sql` | The database contract (section 6) |
| `ml/` skeleton, including `ml/neuronlens/runtime.py` | ML track code; `runtime.py` is a stub (section 11) |
| `server/README.md`, `web/README.md` | Empty placeholders: you build here |
| `.gitignore`, `.env.example` | Ignore rules and environment variable examples |

**Not in the template (you build these):** `Makefile`, `.github/`, `contracts/fixtures/`, `scripts/`, all server code, all web code, `docs/decisions.md`.

### 2.4 If the readiness check fails

Do not guess. Post one message in the team chat using the blocker template in Appendix B, and name the missing file or failing command. This is the **only** kind of question you should need to ask before the real data arrives. While waiting, do nothing in the repo.

---

## 3. Decision rules: so you never need to ask

### 3.1 Order of authority

1. `contracts/schema.sql`
2. This guide
3. `docs/PRD.md`

### 3.2 When the spec is silent

| Situation | What to do |
|---|---|
| A visual detail or wording is not specified | Use the design system (section 10.1) and copy deck (Appendix A). If still silent, copy the nearest similar element on another screen. |
| A behaviour is not specified | Choose the **safest honest** option: fail visibly, never silently; never invent data; never show a number the API did not return. |
| Two sentences conflict | The more specific one wins. Record it in `docs/decisions.md`. |
| You need a library not on the allowed list | Do not add it. Solve it with the allowed libraries. If impossible, record a decision and use the smallest possible dependency. |
| You are unsure between two options | Pick the simpler one. |

Record every such choice in `docs/decisions.md` in this format, one line each, numbered in order:

```
A-1: <decision> — <reason in one sentence>
```

### 3.3 What only the ML Lead can decide (send a blocker)

- Missing or broken template files (section 2.4).
- Any change to `contracts/schema.sql` or to the API contract.
- The real database file and checkpoint (delivered by the ML Lead, section 12).
- A bug in the `Runtime` class behaviour (section 11).
- What a feature label should say.

**Everything else is your decision.**

### 3.4 Contract changes

| Change | Rule |
|---|---|
| Add an optional field or a new table | Propose it to the ML Lead. If accepted, the schema minor version goes up. |
| Rename, remove or retype anything | Breaking. Needs both leads. Do not do it on your own. |

You never edit `contracts/schema.sql`. You may propose changes.

### 3.5 Claims policy for any wording you write

| You may write | You must not write |
|---|---|
| "Shows which features were active" | "Explains why the model answered this way" |
| "Check whether a concept was active" | "Proves bias", "detects bias" |
| "A test of what a feature does" | "Fixes wrong answers", "instantly fixes" |
| A metric with its value from the API | Any metric or percentage you typed yourself |

---

## 4. Build order

Owners: **S** = server person, **W** = web person, **Q** = QA person, **All** = everyone. If one person does several roles, keep the order. If fewer people exist than roles, merge roles in this priority: S first, then W, then Q.

| Stage | Owner | Build | Needs | Done when |
|---|---|---|---|---|
| 1 | All | Readiness check (section 2.3) | "template pushed" | Every command in 2.3 passes |
| 2 | S | Tooling and fixtures (sections 5, 8) | 1 | `sample.db` generates, is deterministic and passes the fixture checks in section 8.6 |
| 3 | S | Server core: settings, database, errors, models, `/health`, CORS, logging (section 9) | 2 | `/health` matches section 7; server refuses a bad database with a clear message |
| 4 | S | Read endpoints: features, detail, random, runs, metrics, steering examples (section 9) | 3 | Server tests for every endpoint pass |
| 5 | S | Mock service, `/inspect`, `/steer` (sections 9, 9.7) | 4 | Mock `/inspect` and `/steer` pass their tests |
| 6 | S | OpenAPI export, `check_db.py`, `demo_check.py` (section 9.9) | 5 | `contracts/openapi.json` is committed and `make check-contracts` passes. **Announce "openapi on main" to the team.** |
| 7 | W | Web foundation: Next.js app, design tokens, layout, banner, number formatting, shared components (section 10) | 1 | App builds, layout renders, no API needed |
| 8 | W | API client, generated types, stores | 6, 7 | `npm run gen:types` works; client calls `/health` and the banner reacts |
| 9 | W | Feature Explorer | 8 | FR criteria in section 10.4 met |
| 10 | W | Feature Detail and `TokenHeatmap` | 8 | Section 10.5 met |
| 11 | W | Text Inspector | 8, 10 | Section 10.6 met |
| 12 | W | Steering Panel | 8 | Section 10.7 met |
| 13 | W | Metrics Dashboard | 8 | Section 10.8 met |
| 14 | W | Home, not-found, error boundary, accessibility and projector pass | 9 to 13 | Section 10.9 checklist passes |
| 15 | Q | Finish tests and CI (section 13) | grows with every stage | CI green; Playwright flow passes |
| 16 | S | Real service against the `Runtime` interface, tested with a fake runtime (section 11) | 5 | Real-service tests pass without a GPU |
| 17 | All | Integration when the ML Lead delivers real data (section 12) | 14, 16, and the handoff message | Section 12 checklist passes |
| 18 | All | Final acceptance (section 14) | 17 | Every item in section 14 is ticked |

Stages 7 to 14 may run in parallel with stages 2 to 6 only where the "Needs" column allows it. Stage 8 cannot start before the server announces "openapi on main".

**Gate rule:** do not start a stage until its "Needs" are done. Do not call a stage finished until its "Done when" is true **and** CI is green on your branch.

---

## 5. Repository, tooling and commands

### 5.1 What you create

```
neuronlens/
├── Makefile
├── scripts/
│   └── demo_check.py
├── .github/workflows/ci.yml
├── docs/decisions.md
├── contracts/
│   ├── schema.sql                  # given (read-only for you)
│   ├── openapi.json                # you generate and commit
│   └── fixtures/
│       ├── make_fixtures.py        # you write
│       └── sample.db               # you generate and commit (under 5 MB)
├── server/
│   ├── requirements.txt            # fastapi, uvicorn[standard], pydantic>=2
│   ├── requirements-dev.txt        # pytest, httpx, ruff, mypy
│   ├── pyproject.toml              # ruff, mypy, pytest settings
│   └── app/
│       ├── main.py
│       ├── settings.py
│       ├── db.py
│       ├── errors.py
│       ├── models.py
│       ├── check_db.py
│       ├── export_openapi.py
│       ├── routers/                # health.py features.py inspect.py steer.py metrics.py runs.py
│       ├── services/               # base.py mock.py real.py tokenize.py
│       └── tests/
└── web/
    ├── package.json
    ├── tailwind.config.ts
    ├── app/                        # routes (section 10.2)
    ├── components/
    ├── lib/
    │   ├── api.ts
    │   ├── types.gen.ts            # generated, never hand-edited
    │   ├── fmt.ts
    │   └── stores/
    └── tests/
```

### 5.2 Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `NEURONLENS_MODE` | `mock` | `mock` or `real` |
| `NEURONLENS_DB` | `<repo>/contracts/fixtures/sample.db` | Path of the SQLite file |
| `NEURONLENS_CKPT` | none | Checkpoint path, real mode only |
| `NEURONLENS_CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Comma-separated allowed origins |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Where the browser finds the server |

Every default must work with no `.env` file, so a fresh clone runs with zero configuration.

### 5.3 Commands (raw form works everywhere; `make` targets are shortcuts)

| Task | Raw command |
|---|---|
| Server setup | `cd server && python3.11 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt -r requirements-dev.txt` (Windows activate: `.venv\Scripts\activate`) |
| Web setup | `cd web && npm install` |
| Build fixtures | `python3 contracts/fixtures/make_fixtures.py --out contracts/fixtures/sample.db` |
| Run server | `cd server && uvicorn app.main:app --reload --port 8000` |
| Run web | `cd web && npm run dev` |
| Server tests | `cd server && pytest` |
| Web tests | `cd web && npm test` |
| End-to-end tests | `cd web && npx playwright test` (first time: `npx playwright install chromium`) |
| Lint | `cd server && ruff check . && mypy app` and `cd web && npm run lint && npm run typecheck` |
| Export API description | `cd server && python -m app.export_openapi > ../contracts/openapi.json` |
| Generate web types | `cd web && npx openapi-typescript ../contracts/openapi.json -o lib/types.gen.ts` |
| Demo check | `python3 scripts/demo_check.py` |

`Makefile` targets to provide: `setup`, `fixtures`, `dev-server`, `dev-web`, `test`, `e2e`, `lint`, `openapi`, `gen-types`, `check-contracts`, `demo-check`. `check-contracts` rebuilds fixtures, runs `check_db.py` on them, regenerates `openapi.json` and `types.gen.ts`, and fails if `git diff --exit-code` shows any difference.

### 5.4 Git rules

- Branch from `main`: `feat/<area>-<thing>`, `fix/<thing>`, `docs/<thing>`, `test/<thing>`.
- Commit messages: `feat: …`, `fix: …`, `docs: …`, `test: …`, `chore: …`.
- Every change goes through a pull request with at least one review. `main` must always pass CI.
- Never commit: `data/`, `*.pt`, `*.pth`, `*.safetensors`, `.env`, `.venv/`, `node_modules/`, `.next/`, `test-results/`, `playwright-report/`. The template's `.gitignore` covers them. **`contracts/fixtures/sample.db` must be committed**, so never add a blanket `*.db` ignore.

### 5.5 CI skeleton (`.github/workflows/ci.yml`)

```yaml
name: ci
on: [push, pull_request]
jobs:
  server:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r server/requirements.txt -r server/requirements-dev.txt
      - run: cd server && ruff check . && mypy app && pytest
      - run: python3 contracts/fixtures/make_fixtures.py --out /tmp/check.db
      - run: cd server && python -m app.check_db /tmp/check.db
  web:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: "20" }
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r server/requirements.txt
      - run: cd server && python -m app.export_openapi > /tmp/openapi.json && diff /tmp/openapi.json ../contracts/openapi.json
      - run: cd web && npm ci && npm run typecheck && npm run lint && npm test
      - run: cd web && npx openapi-typescript ../contracts/openapi.json -o /tmp/types.gen.ts && diff /tmp/types.gen.ts lib/types.gen.ts
      - run: cd web && npx playwright install --with-deps chromium && npx playwright test
```

---

## 6. The database contract

`contracts/schema.sql` is given. It is reproduced here so you do not need to look elsewhere. The file is one SQLite database, opened **read-only** by the server. It describes **one primary run** (`meta.run_id`). Other runs have rows only in `runs`, `metrics`, `density_hist` and `training_curve`.

```sql
PRAGMA foreign_keys = ON;

-- Required keys: schema_version, run_id, model_name, hook_layer,
-- d_model, d_sae, k, created_at, is_synthetic ("true"|"false")
CREATE TABLE meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE runs (
  run_id              TEXT PRIMARY KEY,
  model_name          TEXT    NOT NULL,
  hook_layer          INTEGER NOT NULL,
  d_model             INTEGER NOT NULL,
  d_sae               INTEGER NOT NULL,
  k                   INTEGER NOT NULL,
  n_train_activations INTEGER,
  checkpoint_name     TEXT,
  created_at          TEXT    NOT NULL,
  notes               TEXT
);

CREATE TABLE features (
  feature_id      INTEGER PRIMARY KEY,           -- index into the SAE dictionary
  label           TEXT,                          -- NULL until named
  label_status    TEXT    NOT NULL DEFAULT 'unlabelled'
                  CHECK (label_status IN ('unlabelled','named','ambiguous')),
  density         REAL    NOT NULL,              -- fraction of tokens where it fires
  max_activation  REAL    NOT NULL,
  mean_activation REAL,
  is_dead         INTEGER NOT NULL DEFAULT 0 CHECK (is_dead IN (0,1)),
  notes           TEXT
);
CREATE INDEX idx_features_label   ON features(label);
CREATE INDEX idx_features_density ON features(density);
CREATE INDEX idx_features_maxact  ON features(max_activation);

CREATE TABLE feature_examples (
  example_id       INTEGER PRIMARY KEY AUTOINCREMENT,
  feature_id       INTEGER NOT NULL REFERENCES features(feature_id),
  rank             INTEGER NOT NULL,             -- 1 = strongest
  tokens_json      TEXT    NOT NULL,             -- JSON array of strings, window <= 64 tokens
  activations_json TEXT    NOT NULL,             -- JSON array of floats, same length
  peak_index       INTEGER NOT NULL,             -- index of the max token inside the window
  max_activation   REAL    NOT NULL,
  UNIQUE (feature_id, rank)
);
CREATE INDEX idx_examples_feature ON feature_examples(feature_id);

CREATE TABLE feature_logits (
  feature_id INTEGER NOT NULL REFERENCES features(feature_id),
  direction  TEXT    NOT NULL CHECK (direction IN ('promoted','suppressed')),
  rank       INTEGER NOT NULL,                   -- 1 = strongest
  token      TEXT    NOT NULL,
  score      REAL    NOT NULL,
  PRIMARY KEY (feature_id, direction, rank)
);

-- names used: fvu, l0, dead_pct, ce_original, ce_zero_ablation,
--             ce_reconstructed, ce_recovered
CREATE TABLE metrics (
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  name   TEXT NOT NULL,
  value  REAL NOT NULL,
  PRIMARY KEY (run_id, name)
);

CREATE TABLE density_hist (                      -- log10(density) bins
  run_id TEXT    NOT NULL REFERENCES runs(run_id),
  bin_lo REAL    NOT NULL,
  bin_hi REAL    NOT NULL,
  count  INTEGER NOT NULL,
  PRIMARY KEY (run_id, bin_lo)
);

CREATE TABLE training_curve (
  run_id   TEXT    NOT NULL REFERENCES runs(run_id),
  step     INTEGER NOT NULL,
  loss     REAL    NOT NULL,
  fvu      REAL,
  dead_pct REAL,
  PRIMARY KEY (run_id, step)
);

-- Pre-recorded steering results: used as the demo safety net and by mock mode.
CREATE TABLE steering_examples (
  example_id    INTEGER PRIMARY KEY AUTOINCREMENT,
  feature_id    INTEGER NOT NULL REFERENCES features(feature_id),
  mode          TEXT    NOT NULL CHECK (mode IN ('amplify','ablate')),
  strength      REAL    NOT NULL,
  prompt        TEXT    NOT NULL,
  seed          INTEGER NOT NULL,
  baseline_text TEXT    NOT NULL,
  steered_text  TEXT    NOT NULL,
  is_curated    INTEGER NOT NULL DEFAULT 0 CHECK (is_curated IN (0,1))
);
CREATE INDEX idx_steer_feature ON steering_examples(feature_id);
```

### 6.1 Units and meanings (binding; the ML track follows the same table)

| Field | Unit and rule |
|---|---|
| `features.density` | Fraction between 0 and 1. A dead feature has `0.0`. |
| `features.max_activation`, `mean_activation` | Raw activation value, 0 or more. Dead features have `0.0`. |
| `metrics` `fvu` | Fraction, usually between 0 and 1. Lower is better. |
| `metrics` `l0` | Average number of active features per token, a plain number. |
| `metrics` `dead_pct` | **Percent**, between 0 and 100. |
| `metrics` `ce_original`, `ce_zero_ablation`, `ce_reconstructed` | Cross-entropy loss, a plain number. |
| `metrics` `ce_recovered` | **Fraction**, normally between 0 and 1 (display as a percentage). |
| `training_curve.dead_pct` | Percent between 0 and 100, same as the metric. |
| `density_hist` | Bins over `log10(density)`. **Live features only.** Dead features are not in the histogram; they appear in the `dead_pct` metric. `count` is the number of features in the bin. |
| `feature_examples.tokens_json` | JSON array of strings exactly as the tokenizer produced them, **including leading spaces and newlines**. |
| `feature_examples.activations_json` | JSON array of numbers, same length as `tokens_json`. |

### 6.2 Invariants you can rely on

- `feature_id` values are dense: `0` to `d_sae - 1`, no gaps.
- Dead features have a row but no examples and no logits.
- Each live feature has between 1 and 32 examples and, in the real data, 10 promoted and 10 suppressed logit tokens.
- `meta.schema_version` is `MAJOR.MINOR`. This guide targets major version `1`.
- **Never hard-code** `12288`, `768`, `32` or `8`. Read sizes from `meta`. Fixture data uses `d_sae = 512` to prove this.

### 6.3 How to open it read-only

```python
from pathlib import Path
import sqlite3

def connect(path: str) -> sqlite3.Connection:
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn
```

`Path.as_uri()` produces a correct URI on Windows as well as Linux and macOS.

---

## 7. The API contract

Base path: `/api/v1`. JSON in, JSON out. Keys are `snake_case`. The server is the source of truth for the contract: its OpenAPI output is committed as `contracts/openapi.json` and the web types are generated from it. **Never hand-write response types in the web app.**

### 7.1 Error format (every error from every endpoint)

```json
{ "error": { "code": "FEATURE_NOT_FOUND", "message": "No feature with id 99999" } }
```

| Code | HTTP | When |
|---|---|---|
| `INVALID_INPUT` | 422 | Validation failed, including out-of-range values and a dead feature given to `/steer` |
| `TEXT_TOO_LONG` | 413 | Text or prompt is more than 256 tokens |
| `FEATURE_NOT_FOUND` | 404 | Unknown `feature_id` |
| `RUN_NOT_FOUND` | 404 | Unknown `run_id` |
| `NOT_FOUND` | 404 | Unknown route |
| `MODEL_NOT_LOADED` | 503 | No inference service is available |
| `SCHEMA_MISMATCH` | 500 | Database major version is not 1 |
| `INTERNAL` | 500 | Anything else; the traceback is logged, never sent |

### 7.2 `GET /health`

```json
{
  "status": "ok",
  "mode": "mock",
  "is_synthetic": true,
  "schema_version": "1.0",
  "model_name": "gpt2",
  "hook_layer": 8,
  "d_model": 768,
  "d_sae": 512,
  "k": 32,
  "run_id": "fixture-001"
}
```
`is_synthetic` is `true` when `meta.is_synthetic == "true"`.

### 7.3 `GET /features`

| Param | Type | Default | Rule |
|---|---|---|---|
| `query` | string | none | Case-insensitive substring of `label`. If the trimmed value is all digits, also match `feature_id` exactly. Empty string means no filter. |
| `status` | `unlabelled` \| `named` \| `ambiguous` | all | |
| `include_dead` | bool | `false` | When `false`, dead features are excluded |
| `sort` | `feature_id` \| `density` \| `max_activation` | `feature_id` | Whitelist only |
| `order` | `asc` \| `desc` | `asc` | |
| `limit` | int 1 to 200 | 50 | |
| `offset` | int 0 or more | 0 | |

Ties are broken by `feature_id` ascending so pagination is stable.

```json
{
  "total": 472,
  "limit": 50,
  "offset": 0,
  "items": [
    {
      "feature_id": 3,
      "label": "[sample] cooking 4",
      "label_status": "named",
      "density": 0.0021,
      "max_activation": 14.7,
      "is_dead": false
    }
  ]
}
```
`total` is the count after filters, before pagination.

### 7.4 `GET /features/{feature_id}`

```json
{
  "feature_id": 3,
  "label": "[sample] cooking 4",
  "label_status": "named",
  "density": 0.0021,
  "max_activation": 14.7,
  "is_dead": false,
  "mean_activation": 3.2,
  "notes": null,
  "examples": [
    {
      "rank": 1,
      "tokens": ["preheat", " the", " oven", "."],
      "activations": [0.0, 0.0, 14.7, 4.4],
      "peak_index": 2,
      "max_activation": 14.7
    }
  ],
  "logits": {
    "promoted":   [{ "token": " oven", "score": 2.0 }],
    "suppressed": [{ "token": " the", "score": -0.5 }]
  }
}
```
`examples` is ordered by `rank` ascending. `logits` lists are ordered by `rank` ascending. Unknown id: `FEATURE_NOT_FOUND`.

### 7.5 `GET /features/random`

Returns the same shape as 7.4 for a random **non-dead** feature. Optional `status` filter. If none qualify: `FEATURE_NOT_FOUND`. **This route must be registered before `/features/{feature_id}`.**

### 7.6 `POST /inspect`

Request:
```json
{ "text": "Hello there, my friend", "top_n": 5 }
```
`text`: not empty after trimming, at most 256 tokens. `top_n`: 1 to 20, default 5.

Response:
```json
{
  "model_name": "gpt2",
  "tokens": [
    {
      "index": 0,
      "text": "Hello",
      "is_outlier_position": true,
      "features": [
        { "feature_id": 3, "activation": 6.3, "label": "[sample] cooking 4" }
      ]
    }
  ]
}
```
`features` per token are ordered by activation descending and contain at most `top_n` entries. `label` is the stored label or `null`. `is_outlier_position` is `true` only for `index == 0`. Token `text` keeps its leading space.

### 7.7 `POST /steer`

Request:
```json
{
  "prompt": "The weather here is",
  "feature_id": 3,
  "mode": "amplify",
  "strength": 4.0,
  "max_new_tokens": 60,
  "seed": 42
}
```

| Field | Rule |
|---|---|
| `prompt` | Not empty after trimming; at most 256 tokens |
| `feature_id` | Must exist and must not be dead, otherwise `FEATURE_NOT_FOUND` or `INVALID_INPUT` |
| `mode` | `amplify` or `ablate` |
| `strength` | Float 0 to 10, default 4.0. Used only by `amplify`. `1.0` means the feature's own maximum activation. |
| `max_new_tokens` | Int 1 to 120, default 60 |
| `seed` | Int 0 to 2147483647, default 42. The baseline and the steered run use the **same seed**. |

Response:
```json
{
  "feature": { "feature_id": 3, "label": "[sample] cooking 4" },
  "mode": "amplify",
  "strength": 4.0,
  "seed": 42,
  "baseline_text": "The weather here is mild and the sky is clear.",
  "steered_text": "The weather here is preheat the oven and simmer."
}
```

### 7.8 `GET /steering-examples`

Query: `feature_id` (optional int), `curated_only` (bool, default `false`). Works in both modes.

```json
{
  "items": [
    {
      "example_id": 1,
      "feature_id": 3,
      "mode": "amplify",
      "strength": 4.0,
      "prompt": "The weather here is",
      "seed": 42,
      "baseline_text": "…",
      "steered_text": "…",
      "is_curated": true
    }
  ]
}
```
Order: curated first, then `example_id` ascending.

### 7.9 `GET /runs`

```json
{
  "items": [
    {
      "run_id": "fixture-001",
      "model_name": "gpt2",
      "hook_layer": 8,
      "d_model": 768,
      "d_sae": 512,
      "k": 32,
      "n_train_activations": 100000000,
      "checkpoint_name": null,
      "created_at": "1970-01-01T00:00:00Z",
      "notes": null,
      "is_primary": true
    }
  ]
}
```
Order: primary run first, then `created_at` descending.

### 7.10 `GET /metrics`

Query: `run_id` (optional; default is `meta.run_id`). Unknown id: `RUN_NOT_FOUND`.

```json
{
  "run_id": "fixture-001",
  "metrics": [
    { "name": "ce_recovered", "value": 0.915 },
    { "name": "fvu", "value": 0.121 }
  ],
  "density_hist": [ { "bin_lo": -6.0, "bin_hi": -5.5, "count": 12 } ],
  "training_curve": [ { "step": 100, "loss": 0.91, "fvu": 0.4, "dead_pct": 12.0 } ]
}
```
`metrics` order: `ce_recovered`, `fvu`, `l0`, `dead_pct`, `ce_original`, `ce_zero_ablation`, `ce_reconstructed`, then any others alphabetically. `density_hist` ordered by `bin_lo`, `training_curve` by `step`. Either may be an empty list; the UI must handle that.

### 7.11 Cross-cutting rules

- CORS allows the origins in `NEURONLENS_CORS_ORIGINS`.
- Identical `/steer` requests return identical output.
- Request bodies are never stored or logged.
- No outbound network calls at runtime.
- Targets, measured and reported rather than promised: list and detail under 500 ms; `/inspect` under 2 s for 128 tokens with the real model; `/steer` under 10 s for 60 new tokens with the real model.

---

## 8. Fixtures: the fake data you generate

You write `contracts/fixtures/make_fixtures.py`. It builds `sample.db`, the synthetic database that the whole app track develops against until real data arrives.

### 8.1 Rules

- Standard library only. Deterministic: `rng = random.Random(1234)`; never use Python's `hash()` (it changes between runs).
- Usage: `python3 contracts/fixtures/make_fixtures.py --out contracts/fixtures/sample.db`. It deletes an existing output file, then applies `contracts/schema.sql`, then inserts data.
- Everything is obviously synthetic: every label starts with `[sample] `.

### 8.2 Contents

**meta:** `schema_version=1.0`, `run_id=fixture-001`, `model_name=gpt2`, `hook_layer=8`, `d_model=768`, `d_sae=512`, `k=32`, `created_at=1970-01-01T00:00:00Z` (an epoch sentinel, clearly not real), `is_synthetic=true`.

**runs:**

| run_id | k | d_sae | n_train_activations |
|---|---|---|---|
| fixture-001 (primary) | 32 | 512 | 100000000 |
| fixture-002 | 16 | 512 | 100000000 |
| fixture-003 | 64 | 512 | 100000000 |

All with `model_name=gpt2`, `hook_layer=8`, `d_model=768`, `created_at=1970-01-01T00:00:00Z`, `checkpoint_name` and `notes` NULL.

**features:** 512 rows, `feature_id` 0 to 511.

| ids | Kind | Details |
|---|---|---|
| 0 to 119 | `named` | Label `[sample] {theme} {n}`: theme index `id // 10`, `n = id % 10 + 1`. 32 examples each. |
| 120 to 139 | `ambiguous` | Label `[sample] mixed {id}`. 32 examples each. |
| 140 to 471 | `unlabelled` | Label NULL. 8 examples each. |
| 472 to 511 | dead | `is_dead=1`, `density=0.0`, `max_activation=0.0`, `mean_activation=0.0`, `unlabelled`, no examples, no logits |

For every live feature (ids 0 to 471): `density = round(10 ** rng.uniform(-5, -1.3), 6)`, `max_activation = round(rng.uniform(4, 30), 2)`, `mean_activation = round(max_activation * rng.uniform(0.15, 0.4), 2)`.

**Themes** (index 0 to 11, each with six phrases). Themes for ids 120 to 471 reuse `theme = id % 12`.

| # | Theme | Phrases |
|---|---|---|
| 0 | greetings | `Hello there, my friend` · `Good morning everyone` · `Nice to meet you` · `Welcome back` · `Hi, how are you` · `Greetings and salutations` |
| 1 | python code | `def main():` · `import numpy as np` · `for i in range(10):` · `return result` · `print(value)` · `if x is None:` |
| 2 | legal language | `pursuant to the agreement` · `the parties hereby agree` · `notwithstanding the foregoing` · `in witness whereof` · `subject to the terms` · `shall be binding upon` |
| 3 | cooking | `preheat the oven` · `add a pinch of salt` · `simmer for ten minutes` · `chop the onions` · `stir until smooth` · `season to taste` |
| 4 | sports | `scored in the final minute` · `the team won the match` · `a penalty kick` · `the crowd cheered` · `the coach announced` · `a hat trick` |
| 5 | finance | `interest rates rose` · `the stock market closed higher` · `earnings beat expectations` · `invest in bonds` · `the budget deficit` · `revenue growth slowed` |
| 6 | medical | `the patient reported` · `blood pressure was normal` · `prescribed an antibiotic` · `symptoms include fever` · `a routine checkup` · `clinical trial results` |
| 7 | weather | `heavy rain is expected` · `clear skies ahead` · `strong winds from the north` · `the temperature dropped` · `a thunderstorm warning` · `light snow fell` |
| 8 | geography | `the river flows south` · `a mountain range` · `the capital city` · `on the coast` · `across the border` · `a dense forest` |
| 9 | emotions | `I feel so happy` · `she was deeply sad` · `an angry response` · `nervous and excited` · `a sense of relief` · `feeling lonely` |
| 10 | technology news | `the company released` · `a new smartphone` · `software update available` · `the startup raised funding` · `cloud computing services` · `an open source project` |
| 11 | questions | `what is the reason` · `how does it work` · `why did that happen` · `can you explain` · `where should we go` · `who wrote this` |

**feature_examples:** for a live feature with theme `T`, example number `r` (1-based):
1. Pick 2 to 3 phrases of theme `T` at random and join them with `" "`; tokenize with the regex `\s*\w+|\s*[^\w\s]` (this keeps each token's leading space). Keep at most `n` tokens, where `n` is a random integer from 10 to 24 (windows shorter than `n` are fine).
2. `peak_index` = index of a random token inside the first chosen phrase.
3. `peak activation = round(max_activation * (1 - 0.02 * (r - 1)), 2)`. Its two neighbours get `round(0.3 * peak, 2)`. Every other token in the window is `0.0`.
4. `rank = r`, `max_activation` = peak activation.

**feature_logits:** for each live feature, 10 `promoted` rows: lowercase words from the theme's phrases, prefixed with a space, unique, topped up from the next theme if fewer than 10; `score = round(2.0 - 0.15 * (rank - 1), 2)`. 10 `suppressed` rows from the pool `" the"," of"," and"," a"," to"," in"," is"," that"," it"," for"`; `score = round(-0.4 - 0.1 * (rank - 1), 2)`.

**metrics:**

| name | fixture-001 | fixture-002 | fixture-003 |
|---|---|---|---|
| fvu | 0.121 | 0.19 | 0.08 |
| l0 | 32.0 | 16.0 | 64.0 |
| dead_pct | 7.81 | 11.2 | 5.1 |
| ce_original | 3.3 | 3.3 | 3.3 |
| ce_zero_ablation | 9.8 | 9.8 | 9.8 |
| ce_reconstructed | 3.85 | 4.4 | 3.55 |
| ce_recovered | 0.915 | 0.831 | 0.962 |

**density_hist** for `fixture-001` only: bins of width 0.5 on `log10(density)` from `-6.0` to `-1.0` (ten bins; `bin_lo = -6.0, -5.5, … -1.5`), counts computed from the 472 live densities. The other two runs have no histogram and no training curve: this is deliberate, to exercise the empty states.

**training_curve** for `fixture-001`: `step = 100, 200, … 5000` (50 rows). `loss = round(1.0 * exp(-step/1500) + 0.08, 4)`, `fvu = round(0.9 * exp(-step/1200) + 0.11, 4)`, `dead_pct = round(40 * exp(-step/800) + 7.8, 2)`.

**steering_examples:** 11 rows. For features 0, 1, 10, 11, 20, 21, 30, 31, 40, 41: `mode=amplify`, `strength=4.0`, `seed=42`, a prompt chosen from `["The weather here is", "My favourite thing is", "My favourite place is"]`, `baseline_text = prompt + " mild and the sky is clear."`, `steered_text = prompt + " " + one phrase of that feature's theme + "."`. `is_curated=1` for the first five. One more row: feature 0, `mode=ablate`, `strength=0.0`, `seed=42`, `is_curated=1`.

### 8.3 Size

The finished `sample.db` must be **under 5 MB**. If it is larger, reduce examples for `unlabelled` features first.

### 8.4 Determinism

Running the script twice must produce an identical logical dump (`sqlite3 .dump` output compared with `diff`). Do not compare file bytes.

### 8.5 Safety

`is_synthetic=true` makes the web app show the sample-data banner. Never copy a fixture label or number into slides, the README or the report.

### 8.6 Checks for stage 2

- Row counts: `features` 512, `feature_examples` 140 × 32 + 332 × 8 = 7,136, `feature_logits` 472 × 20 = 9,440, `steering_examples` 11, `training_curve` 50, `density_hist` 10, `metrics` 21, `runs` 3.
- `schema.sql` applies cleanly and every `CHECK` constraint holds.
- Sum of `density_hist.count` is 472.
- Size is under 5 MB and the dump is identical across two runs.

---

## 9. Server spec

### 9.1 Stack and rules

Python 3.11, FastAPI, Uvicorn, Pydantic v2, standard-library `sqlite3` (no ORM). Dev tools: pytest, httpx, ruff, mypy. Use ordinary `def` endpoints (FastAPI runs them in a thread pool). Use a new read-only connection per request, closed afterward.

### 9.2 Startup (fail fast)

At startup, before serving anything:
1. Open the database read-only. Read `meta`.
2. All required keys must exist: `schema_version`, `run_id`, `model_name`, `hook_layer`, `d_model`, `d_sae`, `k`, `created_at`, `is_synthetic`. Otherwise exit with a message naming the missing key.
3. `schema_version` major must be `1`. Otherwise exit with a `SCHEMA_MISMATCH` message.
4. Build the service for the current mode (section 9.7 and 11).
5. In real mode, check the runtime's `model_name`, `hook_layer`, `d_model`, `d_sae` and `k` equal `meta`. On any difference, exit naming the field.

### 9.3 `models.py` (use exactly these names so the generated web types are predictable)

```python
from typing import Literal
from pydantic import BaseModel, Field, field_validator

LabelStatus = Literal["unlabelled", "named", "ambiguous"]
Mode = Literal["amplify", "ablate"]


class Health(BaseModel):
    status: Literal["ok"]
    mode: Literal["mock", "real"]
    is_synthetic: bool
    schema_version: str
    model_name: str
    hook_layer: int
    d_model: int
    d_sae: int
    k: int
    run_id: str


class FeatureSummary(BaseModel):
    feature_id: int
    label: str | None
    label_status: LabelStatus
    density: float
    max_activation: float
    is_dead: bool


class FeatureList(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[FeatureSummary]


class Example(BaseModel):
    rank: int
    tokens: list[str]
    activations: list[float]
    peak_index: int
    max_activation: float


class LogitEntry(BaseModel):
    token: str
    score: float


class Logits(BaseModel):
    promoted: list[LogitEntry]
    suppressed: list[LogitEntry]


class FeatureDetail(FeatureSummary):
    mean_activation: float | None
    notes: str | None
    examples: list[Example]
    logits: Logits


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


class InspectRequest(BaseModel):
    text: str = Field(min_length=1)
    top_n: int = Field(default=5, ge=1, le=20)

    _check_text = field_validator("text")(_not_blank)


class InspectFeature(BaseModel):
    feature_id: int
    activation: float
    label: str | None


class InspectToken(BaseModel):
    index: int
    text: str
    is_outlier_position: bool
    features: list[InspectFeature]


class InspectResponse(BaseModel):
    model_name: str
    tokens: list[InspectToken]


class SteerRequest(BaseModel):
    prompt: str = Field(min_length=1)
    feature_id: int = Field(ge=0)
    mode: Mode
    strength: float = Field(default=4.0, ge=0.0, le=10.0)
    max_new_tokens: int = Field(default=60, ge=1, le=120)
    seed: int = Field(default=42, ge=0, le=2147483647)

    _check_prompt = field_validator("prompt")(_not_blank)


class FeatureRef(BaseModel):
    feature_id: int
    label: str | None


class SteerResponse(BaseModel):
    feature: FeatureRef
    mode: Mode
    strength: float
    seed: int
    baseline_text: str
    steered_text: str


class SteeringExample(BaseModel):
    example_id: int
    feature_id: int
    mode: Mode
    strength: float
    prompt: str
    seed: int
    baseline_text: str
    steered_text: str
    is_curated: bool


class SteeringExampleList(BaseModel):
    items: list[SteeringExample]


class RunInfo(BaseModel):
    run_id: str
    model_name: str
    hook_layer: int
    d_model: int
    d_sae: int
    k: int
    n_train_activations: int | None
    checkpoint_name: str | None
    created_at: str
    notes: str | None
    is_primary: bool


class RunList(BaseModel):
    items: list[RunInfo]


class MetricItem(BaseModel):
    name: str
    value: float


class DensityBin(BaseModel):
    bin_lo: float
    bin_hi: float
    count: int


class CurvePoint(BaseModel):
    step: int
    loss: float
    fvu: float | None
    dead_pct: float | None


class MetricsResponse(BaseModel):
    run_id: str
    metrics: list[MetricItem]
    density_hist: list[DensityBin]
    training_curve: list[CurvePoint]
```

The `_not_blank` validators make blank or whitespace-only text and prompts fail validation, which the error handler in section 9.4 turns into `INVALID_INPUT`.

### 9.4 Errors (`errors.py`)

Define `class ApiError(Exception)` with `code`, `message`, `status`. Register handlers so that:
- `ApiError` returns `{"error": {"code", "message"}}` with its status.
- `RequestValidationError` returns `INVALID_INPUT` (422) with a message built from the first error: field name and reason, for example `strength: must be at most 10`.
- Unknown routes return `NOT_FOUND` (404).
- Any other exception is logged with its traceback and returns `INTERNAL` (500) with the message `Internal server error`.

### 9.5 Endpoint behaviour

**Query for `/features`.** Build the `WHERE` clause from fixed fragments; **never interpolate user input into SQL**.
- Whitelist maps: `sort` to a column (`feature_id`, `density`, `max_activation`), `order` to `ASC` or `DESC`. Anything else is a validation error.
- Label search: escape `\`, `%` and `_` in the query, then use `label LIKE ? ESCAPE '\'` with `%query%`. If the trimmed query is all digits, use `(feature_id = ? OR label LIKE ? ESCAPE '\')`.
- `include_dead=false` adds `is_dead = 0`. `status` adds `label_status = ?`.
- `ORDER BY <column> <order>, feature_id ASC`. Run a `COUNT(*)` with the same `WHERE` for `total`.

**`/features/{feature_id}`.** Read the feature row (404 if absent), its examples ordered by `rank` (parse `tokens_json` and `activations_json` with `json.loads`), and its logits ordered by `direction`, `rank`. Convert `is_dead` and `is_curated` from 0/1 to booleans.

**`/features/random`.** `SELECT feature_id FROM features WHERE is_dead = 0 [AND label_status = ?] ORDER BY RANDOM() LIMIT 1`, then return the detail response.

**`/runs`.** All rows from `runs`; `is_primary` is `run_id == meta.run_id`. Primary first, then `created_at` descending.

**`/metrics`.** Resolve the run (default primary; unknown gives `RUN_NOT_FOUND`), then return metrics in the order given in section 7.10, plus the histogram and curve.

**`/steering-examples`.** Filter by `feature_id` and `curated_only` when given. Order: `is_curated DESC, example_id ASC`.

**`/inspect`.**
1. `n = service.count_tokens(text)`; if `n > 256` raise `TEXT_TOO_LONG` with message `Text is {n} tokens; the limit is 256.`
2. `raw = service.inspect(text, top_n)`.
3. One query for all labels: `SELECT feature_id, label FROM features WHERE feature_id IN (…)` using bound parameters.
4. Build the response; `is_outlier_position = (index == 0)`; sort features by activation descending.

**`/steer`.**
1. Load the feature (404 if absent). If `is_dead`, raise `INVALID_INPUT` with message `Feature {id} is dead and never fires; pick another feature.`
2. `count_tokens(prompt) > 256` raises `TEXT_TOO_LONG`.
3. `baseline, steered = service.steer(prompt=…, feature_id=…, mode=…, strength=…, max_activation=<feature max_activation>, max_new_tokens=…, seed=…)`.
4. Return the response, echoing the request's `mode`, `strength` and `seed`.

### 9.6 Service interface (`services/base.py`)

```python
from typing import Protocol


class Service(Protocol):
    def count_tokens(self, text: str) -> int: ...

    def inspect(self, text: str, top_n: int) -> list[dict]:
        """Return [{"index": int, "text": str,
        "features": [{"feature_id": int, "activation": float}, ...]}, ...]."""

    def steer(
        self,
        *,
        prompt: str,
        feature_id: int,
        mode: str,
        strength: float,
        max_activation: float,
        max_new_tokens: int,
        seed: int,
    ) -> tuple[str, str]:
        """Return (baseline_text, steered_text)."""
```

### 9.7 Mock service (`services/mock.py` and `services/tokenize.py`)

The mock must be **deterministic** and need no model.

```python
import hashlib
import re

TOKEN_RE = re.compile(r"\s*\w+|\s*[^\w\s]")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def mock_features(token: str, top_n: int, live_ids: list[int]) -> list[tuple[int, float]]:
    """Deterministically pick up to top_n live features for a token."""
    chosen: dict[int, float] = {}
    j = 0
    while len(chosen) < top_n and j < top_n * 4:
        h = hashlib.sha256(f"{token}|{j}".encode("utf-8")).digest()
        feature_id = live_ids[int.from_bytes(h[:4], "big") % len(live_ids)]
        activation = round(1.0 + (int.from_bytes(h[4:6], "big") / 65535) * 9.0, 2)
        chosen.setdefault(feature_id, activation)
        j += 1
    return sorted(chosen.items(), key=lambda kv: (-kv[1], kv[0]))
```

- `count_tokens(text)` is `len(tokenize(text))`. (Real mode uses the real tokenizer; the mock approximates.)
- `inspect` tokenizes with `tokenize`, and for each token uses `mock_features(token, top_n, live_ids)`. `live_ids` is the sorted list of non-dead `feature_id` values read from the database when the service is created.
- `steer`:
  1. If `steering_examples` has a row matching `(feature_id, mode, strength, prompt, seed)`, return its `baseline_text` and `steered_text`.
  2. Otherwise, if there is any row for that `feature_id` and `mode`, return its texts.
  3. Otherwise return `prompt + " [sample baseline output]"` and `prompt + f" [sample steered output: feature {feature_id}, {mode}, strength {strength}]"`.

### 9.8 Logging

One log line per request: method, path, status, duration in milliseconds. Never log bodies or query strings that contain user text.

### 9.9 Tools you also write

- **`app/export_openapi.py`**: prints `json.dumps(app.openapi(), indent=2, sort_keys=True)` followed by a newline. Used by `make openapi`.
- **`app/check_db.py <path>`**: validates a database and prints a report; exits non-zero on any failure. Checks:
  1. every table and column in `contracts/schema.sql` exists (compare `PRAGMA table_info`),
  2. all required `meta` keys exist and `schema_version` major is 1,
  3. `feature_id` values are exactly `0..d_sae-1`,
  4. dead features have no examples and no logits,
  5. every live feature has at least one example,
  6. 50 random `feature_examples` rows have `tokens_json` and `activations_json` parsing to arrays of equal length with `peak_index` in range,
  7. units: `ce_recovered` and `fvu` present and numeric, `dead_pct` between 0 and 100, densities between 0 and 1,
  8. `density_hist` counts sum to the number of live features (a warning, not a failure, if the table is empty),
  9. prints row counts for every table.
- **`scripts/demo_check.py`**: fetches `http://localhost:8000/api/v1/health` with `urllib`; exits non-zero (with a plain explanation) if `mode` is `mock` or `is_synthetic` is `true`, or if the server cannot be reached.

### 9.10 Server tests (names are binding)

`test_health_shape`, `test_startup_rejects_missing_meta`, `test_startup_rejects_wrong_major`, `test_features_default_page`, `test_features_search_label`, `test_features_search_numeric_id`, `test_features_sort_and_order`, `test_features_filter_status`, `test_features_exclude_dead_by_default`, `test_features_pagination_stable`, `test_features_rejects_bad_sort`, `test_features_like_wildcards_escaped`, `test_feature_detail`, `test_feature_detail_404`, `test_feature_random_not_dead`, `test_feature_random_route_not_shadowed`, `test_runs_primary_first`, `test_metrics_default_run`, `test_metrics_unknown_run`, `test_metrics_empty_hist_ok`, `test_steering_examples_filters`, `test_inspect_ok`, `test_inspect_deterministic`, `test_inspect_empty_text_rejected`, `test_inspect_too_long`, `test_steer_ok`, `test_steer_deterministic`, `test_steer_dead_feature_rejected`, `test_steer_bad_strength`, `test_unknown_route_error_format`, `test_error_format_everywhere`. Tests build a fresh `sample.db` in a temporary folder using `make_fixtures.py`.

---

## 10. Web spec

### 10.1 Stack and design system

**Allowed dependencies.** Runtime: `next@14`, `react@18`, `react-dom@18`, `typescript`, `tailwindcss@3` (with `postcss`, `autoprefixer`), `zustand`, `recharts`, `lucide-react`, `clsx`. Dev: `vitest`, `@testing-library/react`, `@testing-library/jest-dom`, `jsdom`, `@playwright/test`, `openapi-typescript`, `eslint`, `eslint-config-next`, `prettier`. Commit `package-lock.json`. **Tailwind must stay on v3.** Use the App Router. Add `"use client"` to any file that uses hooks, stores or browser APIs.

**Look and feel.** Dark theme only, matching the project's slide deck. Large type and high contrast, because it will be shown on a projector.

```css
:root {
  --bg: #0A0A0A;        --surface: #141414;   --surface-2: #1C1C1C;
  --border: #2E2E2E;    --text: #F2F2F2;      --muted: #A0A0A0;
  --accent: #A3D729;    --accent-ink: #0A0A0A;
  --warn: #FFB454;      --pink: #F0568C;      --danger: #FF6B6B;  --ok: #6FD08C;
}
```

Expose these as Tailwind colours (`bg`, `surface`, `surface2`, `border`, `text`, `muted`, `accent`, `warn`, `pink`, `danger`, `ok`) by referencing the CSS variables in `tailwind.config.ts`.

| Aspect | Rule |
|---|---|
| Fonts | **System stacks only** (no web fonts, so the app works offline): `ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif` and `ui-monospace, SFMono-Regular, Menlo, Consolas, monospace` for tokens |
| Sizes | Body 16 px. Page title 32 px bold. Section title 22 px semibold. Tokens 15 px mono. Table text 15 px. |
| Layout | Content width max 1200 px, centred, 24 px side padding. Cards: `surface` background, 1 px `border`, 8 px radius, 16 to 24 px padding. |
| Focus | 2 px solid `accent`, 2 px offset, on every interactive element. Never remove outlines. |
| Motion | Minimal. Honour `prefers-reduced-motion`. |
| Width support | Designed for 1024 px and wider. Below 768 px, columns stack into one; it must remain usable but is not polished. |
| Buttons | Primary: `accent` background, `accent-ink` text. Secondary: `surface-2` background, `text`, 1 px `border`. Minimum height 40 px. |

### 10.2 Routes and file layout

| Route | File |
|---|---|
| `/` | `app/page.tsx` |
| `/features` | `app/features/page.tsx` (wrap `useSearchParams` users in `<Suspense>`) |
| `/features/[id]` | `app/features/[id]/page.tsx` |
| `/inspect` | `app/inspect/page.tsx` |
| `/steer` | `app/steer/page.tsx` (reads `?feature=`) |
| `/metrics` | `app/metrics/page.tsx` |
| not found | `app/not-found.tsx` |
| error boundary | `app/error.tsx` |
| layout | `app/layout.tsx` with header, `ModeBanner`, footer note |

### 10.3 Shared building blocks

**Header.** Left: `NeuronLens` wordmark (accent colour) with a Lucide `ScanSearch` icon. Centre: links Features, Inspect, Steer, Metrics (current page highlighted, `aria-current="page"`). Right: a compact chip `{model_name} · layer {hook_layer} · {d_sae} features · k={k}` from `/health` (number formatted with thousands separators).

**`ModeBanner`.** Full-width bar directly under the header, on every page:
- `is_synthetic` true: `warn` background, dark text: banner.synthetic.
- else `mode == "mock"`: `warn` background: banner.mock.
- health request failed: `danger` background: banner.offline with the API URL.
- otherwise render nothing.

**Footer.** One muted line on every page: note.limit.

**`lib/api.ts`.** One function per endpoint, typed with `types.gen.ts`. Rules: base URL from `NEXT_PUBLIC_API_URL` (fallback `http://localhost:8000`) plus `/api/v1`; JSON requests; 10 s timeout for reads and 60 s for `/steer`, using `AbortController`; on a non-2xx response parse the error format and throw `ApiError(code, message, status)`; on network failure throw `ApiError("NETWORK", …)`. A helper `errorMessage(error)` maps codes to the copy in Appendix A.

**Stores (Zustand, in `lib/stores/`).**
- `useHealthStore`: `{ health, status: "idle"|"loading"|"ok"|"error", load() }`. `load()` runs once from the layout.
- `useInspectorStore`: `{ text, topN, result, status, error, activeIndex, pinnedIndex, setText, run(), reset() }`.
- `useSteeringStore`: `{ featureId, mode, strength, prompt, maxNewTokens, seed, result, resultSource: "live"|"saved"|null, status, error, setField(), run(), loadSaved(example) }`.

Stores keep their state while the user moves between pages. Never use `localStorage` for anything the app needs.

**`lib/fmt.ts`** (all numbers shown anywhere go through these):

| Function | Rule | Example |
|---|---|---|
| `fmtInt(n)` | `toLocaleString("en-US")` | `12,288` |
| `fmtDensity(d)` | `d == 0` gives `0%`; percent `p = d * 100`; `p >= 1` gives one decimal; `p >= 0.01` gives two decimals; otherwise `<0.01%` | `0.21%` |
| `fmtActivation(a)` | one decimal | `14.7` |
| `fmtFraction(v)` | three decimals | `0.121` |
| `fmtPercentFromFraction(v)` | `(v * 100).toFixed(1) + "%"` | `91.5%` |
| `fmtPercent(v)` | value already a percent, one decimal | `7.8%` |
| `fmtLoss(v)` | three decimals | `3.300` |

**`TokenText` helper** (used everywhere a token is displayed): never trim. Render with `white-space: pre`. Display `\n` as `↵` and `\t` as `⇥`. A token that is only spaces is shown as `␠`. A token that is the empty string is shown as `∅`.

**State components.** `Skeleton` (grey pulsing blocks), `EmptyState({ text })`, `ErrorState({ error, onRetry })` (shows `errorMessage(error)` and a **Try again** button where retrying makes sense).

### 10.4 Feature Explorer (`/features`)

```
┌──────────────────────────────────────────────────────────────────┐
│ Features                                       [ Random feature ]│
│ [ search box…………………… ] [Status ▾] [Sort ▾] [↑↓] [☐ Show dead]  │
├──────────────────────────────────────────────────────────────────┤
│  ID    Label                         Density   Max activation     │
│  3     [sample] cooking 4  [named]   0.21%     14.7               │
│  …                                                               │
├──────────────────────────────────────────────────────────────────┤
│ Showing 1–50 of 472                         [ ← Prev ] [ Next → ]│
└──────────────────────────────────────────────────────────────────┘
```

- URL holds the state: `q`, `status`, `sort`, `order`, `dead` (`1` when shown), `page` (1-based). Omit parameters that equal their defaults.
- Search is debounced by 300 ms and resets to page 1. Page size is fixed at 50 (`offset = (page - 1) * 50`).
- A row is a link to `/features/{id}` (the whole row clickable, keyboard focusable). Unlabelled features show the text "Unlabelled feature" in muted colour. Show a small badge for `ambiguous` (warn colour) and `dead` (muted).
- **Random feature** calls `/features/random` and navigates to the result.
- States: loading shows 10 skeleton rows; no results shows `empty.features`; error shows `ErrorState` with retry.
- Accept: first page renders from local data in under 500 ms; typing `3` finds feature 3; sorting by density descending puts the densest first.

### 10.5 Feature Detail (`/features/[id]`)

```
┌──────────────────────────────────────────────────────────────────┐
│ ← Features                                                       │
│ #3  [sample] cooking 4   [named]                                 │
│ [ Steer this feature ] [ Random feature ]                        │
│ ┌ Density ┐ ┌ Max activation ┐ ┌ Mean activation ┐               │
│ Notes (only if present)                                          │
│ Top examples (32)                                                │
│  #1  preheat the [oven] .        ← heatmap row                   │
│  …                                                               │
│ ┌ Pushes the model towards ┐ ┌ Pushes the model away from ┐      │
│ │  oven   2.00             │ │  the   -0.50               │      │
└──────────────────────────────────────────────────────────────────┘
```

- Invalid or unknown id shows an error state with a link back to the list.
- Dead feature: show a banner-style note `feature.dead` and no examples section.
- Live feature with no examples: show `empty.examples`.
- **`TokenHeatmap`** props: `tokens: string[]`, `activations: number[]`, `peakIndex?: number`.
  - `max = max(activations)`. For each token with `a > 0`: `t = clamp(a / max, 0, 1)`, background `rgba(163, 215, 41, ${0.15 + 0.75 * t})`, text colour `accent-ink` when `t > 0.55`, otherwise `text`.
  - Tokens with `a <= 0` have no background.
  - The peak token gets a 2 px `accent` outline.
  - Every token is focusable (`tabIndex=0`) with `title` and `aria-label` equal to `token "{text}", activation {a.toFixed(2)}`. **Meaning must never rely on colour alone.**
- Logit lists: show the token (mono) and its score to two decimals. Promoted scores in `ok` colour, suppressed in `pink`.

### 10.6 Text Inspector (`/inspect`)

```
┌──────────────────────────────────────────────────────────────────┐
│ Inspect text                                                      │
│ ┌──────────────────────────────────────────────┐                  │
│ │ Paste or type text (up to 256 tokens)         │   [ Inspect ]   │
│ └──────────────────────────────────────────────┘                  │
│ ┌ tokens ───────────────────────────┐ ┌ Token: " friend" ────────┐│
│ │ Hello there , my friend           │ │ #3  [sample] cooking 4   ││
│ │ (coloured chips)                  │ │     ████████ 6.3         ││
│ └───────────────────────────────────┘ └──────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
```

- `Inspect` is disabled while the text is empty or a request is running. A 256-token limit is enforced by the server; show its message on `TEXT_TOO_LONG`.
- Each token is a button-like chip (mono font). Background strength uses the same colour function as the heatmap, with `a` = that token's strongest feature activation and `max` = the largest such value among tokens **excluding index 0**.
- Token with `is_outlier_position` is drawn muted with a dashed border and `title` = outlier tooltip text (Appendix A).
- The side panel shows the **active token**: the hovered or focused token, otherwise the pinned one (click or Enter pins). For each feature: `#id`, label (or "Unlabelled feature"), a horizontal bar whose width is `activation / max activation of this token's features`, the number, and the whole row links to `/features/{id}`.
- States: before the first run show an `EmptyState` with the hint text; while running show skeleton chips; on error show `ErrorState`.
- Accept: every token shows its features; keyboard users can tab through tokens and read the panel; clicking a feature opens its page.

### 10.7 Steering Panel (`/steer`)

```
┌───────────────────────────────┬─────────────────────────────────┐
│ Feature [ 3 ▾ search ] label  │  Baseline          Steered       │
│ Mode (●) Amplify ( ) Ablate   │  ┌─────────────┐  ┌─────────────┐│
│ Strength [──●────] 4.0        │  │ text…       │  │ text…       ││
│ Prompt [ textarea ]           │  └─────────────┘  └─────────────┘│
│ Tokens [60 ▾]  Seed [42] [🎲] │  seed 42 · amplify ×4.0 · Live   │
│ [ Run steering ]              │                                  │
│ Saved examples [ choose… ▾ ]  │  Steering is a test, not a fix.  │
└───────────────────────────────┴─────────────────────────────────┘
```

- `/steer?feature=3` pre-fills the feature. The feature field is a number input with a small search list: as the user types text, call `/features?query=…&limit=8` and show matches (label and id); choosing one sets `featureId`.
- Defaults: mode `amplify`, strength `4`, prompt `The weather here is`, max tokens `60` (options 30, 60, 120), seed `42`. **Randomize seed** sets a random integer from 0 to 99999.
- Strength: a slider 0 to 10 with step 0.5 plus a number input, both bound to the same value. Disabled (and visibly dimmed) when mode is `ablate`.
- **Run steering** posts to `/steer`. While running show `loading.steer` and skeleton output cards. The previous result stays visible until the new one arrives.
- Result cards: left "Baseline", right "Steered" with a caption `feature #{id} · {label} · {mode} ×{strength} · seed {seed}` and a badge **Live** or **Saved example**. Preserve line breaks in the text (`white-space: pre-wrap`).
- **Saved examples**: a select filled from `/steering-examples` (filtered by `feature_id` if one is set; otherwise curated only). An option reads `#{feature_id} · {mode} ×{strength} · "{prompt, first 28 characters}…"`. Choosing one fills the form and **immediately shows the stored result** with the Saved badge, without calling `/steer`. This keeps the demo safe when the server is offline.
- Always show note.steering below the result.
- `INVALID_INPUT` for a dead feature shows its message from the server.

### 10.8 Metrics Dashboard (`/metrics`)

```
┌──────────────────────────────────────────────────────────────────┐
│ Metrics          Run [ fixture-001 ▾ ]                            │
│ model gpt2 · layer 8 · 512 features · k=32 · 100,000,000 activations│
│ ┌ CE recovered ┐ ┌ FVU ┐ ┌ Dead features ┐ ┌ L0 ┐                 │
│ │    91.5%     │ │0.121│ │     7.8%      │ │32.0│                 │
│ Cross-entropy:  original 3.300 · zero ablation 9.800 · reconstructed 3.850 │
│ Density of features (log10)        Training curve                  │
│ [ bar chart ]                      [ line chart: loss, fvu ]       │
└──────────────────────────────────────────────────────────────────┘
```

- Four tiles in this order: **CE recovered** (largest, accent colour, `fmtPercentFromFraction`), **FVU** (`fmtFraction`), **Dead features** (`fmtPercent`), **L0** (one decimal). Each tile has a one-line plain-language helper under its number (Appendix A). A missing metric shows `—`.
- Run selector appears only when `/runs` returns more than one run; changing it refetches `/metrics?run_id=`. Default is the primary run.
- Charts use Recharts inside a fixed-height container (320 px) wrapped in a client component. Histogram: bars, x axis = bin midpoint (`(bin_lo + bin_hi) / 2`, one decimal) labelled `log10(density)`, y axis = number of features. Curve: lines for `loss` and `fvu` against `step`.
- Empty histogram or curve shows `empty.hist` or `empty.curve`.
- Never compute, estimate or round-trip any metric yourself; show exactly what the API returned, formatted by `fmt.ts`.

### 10.9 Home, not found, errors, polish (stage 14)

- **Home:** page title `NeuronLens`, tagline, intro paragraph (Appendix A), four cards linking to Features, Inspect, Steer and Metrics (each with its one-line description), and a "Currently loaded" card listing model, hook layer, number of features, k, run id and mode from `/health`.
- **Not found** and **error boundary** pages use the copy in Appendix A and a link home.
- **Accessibility pass:** every interactive element reachable by keyboard in a logical order; visible focus; a "Skip to content" link as the first focusable element; `aria-live="polite"` on loading and result regions; every form control has a label; heatmap and chip meaning never colour-only; contrast at least 4.5 to 1 for text.
- **Projector pass:** text at least 15 px everywhere important; nothing essential in muted tiny text; test at 1280 by 720.

---

## 11. The real service and the ML interface

The ML Lead provides one class in `ml/neuronlens/runtime.py`. The template contains this **stub**, which you code against:

```python
class Runtime:
    """Loads the frozen model and the trained SAE once and answers live requests."""

    model_name: str
    hook_layer: int
    d_model: int
    d_sae: int
    k: int

    @classmethod
    def load(cls, checkpoint_path: str, device: str = "cuda") -> "Runtime":
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        """Number of tokens the model's tokenizer produces for text."""
        raise NotImplementedError

    def inspect(self, text: str, top_n: int) -> list[dict]:
        """One dict per token: {"index": int, "text": str,
        "features": [{"feature_id": int, "activation": float}, ...]},
        features sorted by activation descending, at most top_n each.
        Activations are the SAE feature activations on the same scale
        as features.max_activation in the database."""
        raise NotImplementedError

    def steer(
        self,
        prompt: str,
        feature_id: int,
        mode: str,            # "amplify" or "ablate"
        strength: float,
        max_activation: float,
        max_new_tokens: int,
        seed: int,
    ) -> dict:
        """Return {"baseline_text": str, "steered_text": str}.
        Both generations use the same seed."""
        raise NotImplementedError
```

### 11.1 What you build (stage 16)

`services/real.py`:

```python
import threading


class RealService:
    def __init__(self, runtime) -> None:
        self._runtime = runtime
        self._lock = threading.Lock()   # one GPU request at a time

    def count_tokens(self, text: str) -> int:
        with self._lock:
            return self._runtime.count_tokens(text)

    def inspect(self, text: str, top_n: int) -> list[dict]:
        with self._lock:
            return self._runtime.inspect(text, top_n)

    def steer(self, *, prompt, feature_id, mode, strength,
              max_activation, max_new_tokens, seed):
        with self._lock:
            out = self._runtime.steer(
                prompt, feature_id, mode, strength,
                max_activation, max_new_tokens, seed,
            )
        return out["baseline_text"], out["steered_text"]
```

Loading happens only in real mode, **lazily**, so mock mode never needs PyTorch:

```python
def build_real_service(checkpoint_path: str):
    from neuronlens.runtime import Runtime   # imported only in real mode
    return RealService(Runtime.load(checkpoint_path))
```

### 11.2 How you test it without a GPU

Write `FakeRuntime` in the tests. It implements the same methods deterministically (for example, reuse `mock_features`) and has the five attributes. Run **the same router tests** against `RealService(FakeRuntime(...))` and against `MockService`. Add one test, `test_mock_and_real_shapes_match`, asserting both produce responses with identical keys and types.

Also test: runtime attribute mismatch with `meta` makes startup fail with the field named; a runtime exception returns `INTERNAL` and the server keeps serving the next request.

### 11.3 Installing for real mode

Real mode needs the ML package on the Python path: `pip install -e ml` (this pulls in PyTorch). Only the machine with the GPU does this. Everyone else stays in mock mode.

---

## 12. Integration when the ML Lead delivers real data

Wait for the ML Lead's message. It must contain: the database file name, its size, and its SHA-256 checksum. For the real model it also contains the checkpoint file name. If any of these is missing, send a blocker (Appendix B). Do not start earlier.

There are two deliveries. The first may be poor quality on purpose (the features may look meaningless); its job is to expose integration problems. Treat both the same way.

### 12.1 Steps

1. Put the file at `data/neuronlens.db`. Verify the checksum: `sha256sum data/neuronlens.db` (Windows: `certutil -hashfile data\neuronlens.db SHA256`). If it differs, ask for the file again.
2. Run `python -m app.check_db data/neuronlens.db` from `server/`. If it fails, post a blocker listing the failing checks.
3. Start the server against it: `NEURONLENS_DB=../data/neuronlens.db uvicorn app.main:app --port 8000` (Windows: set the variable first). It is still in **mock mode**, so `/inspect` and `/steer` are simulated and the banner says so; everything else is real.
4. Click through every screen. Specifically check:
   - the Explorer first page and a search return in under 500 ms (note the size of the database in your report),
   - no screen crashes on unlabelled, ambiguous or dead features, on very long tokens, or on tokens with unusual characters,
   - the metrics screen shows the real numbers with the right units (CE recovered is shown as a percentage, dead features as a percent),
   - sizes in the header chip match the database (no leftover fixture values).
5. Fix anything that is **your** bug. If something is wrong with the data itself, post a blocker with: the table, the `feature_id` or row, what you expected, what you saw.
6. When the ML Lead delivers the checkpoint, set `NEURONLENS_MODE=real` and `NEURONLENS_CKPT=<path>` on the GPU machine and restart. The server must pass the startup checks in section 9.2. If it exits naming a mismatching field, post a blocker naming the field and both values.
7. Test real `/inspect` and `/steer`: five varied sentences and three steering runs, including one with `ablate`. Check: same inputs give the same steering output twice; latencies against the targets in section 7.11 (report the measured numbers); no request leaves the server in a bad state; the banner disappears (real mode, `is_synthetic` false).
8. Run `python3 scripts/demo_check.py`. It must pass.

### 12.2 Reporting an integration issue

```
ISSUE
Where: <endpoint or screen>
Input: <exact request or clicks>
Expected: <what the guide says>
Actual: <what happened, with the error text or screenshot>
```

---

## 13. Tests and CI

### 13.1 Web unit tests (Vitest and Testing Library)

`fmt.test.ts` covers every function in section 10.3, including density edge cases (`0`, `0.5`, `0.0001`, `0.000001`). `TokenHeatmap.test.tsx` checks: tokens with zero activation have no background, the peak token has the outline, whitespace and newline tokens render visibly, `aria-label` text is correct, and an all-zero array does not divide by zero. `ModeBanner.test.tsx` checks all four banner states. `api.test.ts` checks that an error body becomes an `ApiError` with the right code and that a timeout becomes `NETWORK`. `steeringStore.test.ts` checks that loading a saved example fills the form and sets `resultSource` to `saved` without a network call.

### 13.2 End-to-end test (Playwright, mock mode)

Configure `webServer` entries to start the API on port 8000 (mock mode, fixture database) and the web app on port 3000. One spec walks the demo:

1. Open `/`: the sample-data banner is visible.
2. Open `/features`: 50 rows appear; search `cooking`; open the first result; at least one example heatmap is visible.
3. Click **Steer this feature**: the feature field is prefilled.
4. Open `/inspect`: paste `Hello there, my friend`, click **Inspect**; tokens render; hovering a token shows features in the side panel.
5. Open `/steer?feature=0`: choose a saved example; the two result cards appear with the **Saved example** badge; click **Run steering**; the badge changes to **Live**.
6. Open `/metrics`: four tiles appear, and the histogram and curve render.
7. Keyboard: tab through the Explorer; a row can be opened with Enter.

### 13.3 CI

The workflow in section 5.5 must be green before any merge.

---

## 14. Final acceptance

The app track is finished only when **every** line is true, on the real data, in real mode, on the GPU machine:

- [ ] `python3 scripts/demo_check.py` passes (not mock, not synthetic).
- [ ] The demo walk in section 13.2 passes on real data, including real `/inspect` and real `/steer`.
- [ ] A saved steering example loads with the server stopped.
- [ ] All Explorer, Detail, Inspector, Steering and Metrics requirements in section 10 are met.
- [ ] The wording claims policy (section 3.5) holds on every screen; no screen contains "proves", "detects" or "fixes".
- [ ] No hard-coded model sizes anywhere (grep for `12288`, `768`, `512` in `web/` and `server/app/`).
- [ ] CI is green on `main`.
- [ ] `docs/decisions.md` lists every decision you made.
- [ ] Nothing from `data/` or any checkpoint is committed.
- [ ] A short recording of the working demo exists, plus screenshots of each screen, stored outside git.

---

## 15. Gotchas

1. Register `/features/random` **before** `/features/{feature_id}`, or `random` is parsed as an id and returns 422.
2. On Windows, build the SQLite URI with `Path(...).resolve().as_uri()`; hand-built `file:` strings break.
3. `LIKE` treats `%` and `_` as wildcards. Escape them, and use `ESCAPE '\'`.
4. Never build SQL by string concatenation with user values. Whitelist `sort` and `order`.
5. Do not trim, collapse or normalise token strings anywhere. Leading spaces carry meaning.
6. `hash()` in Python is randomised per process. Use `hashlib` for anything that must be deterministic.
7. Tailwind v4 is a different, incompatible setup. Stay on v3.
8. Next.js 14: any component using `useSearchParams` must sit inside `<Suspense>` or the production build fails.
9. `NEXT_PUBLIC_*` variables are baked in at build time. Restart the dev server after changing one.
10. Recharts needs a fixed-height parent and must run in a client component.
11. Use system font stacks. Web fonts download at build time and break offline use.
12. Do not add a blanket `*.db` rule to `.gitignore`; the fixture database must be committed.
13. Do not commit generated code that is stale: CI regenerates `openapi.json` and `types.gen.ts` and fails if they differ.
14. Never show the mock or fixture data without the banner. The banner depends on `is_synthetic` and `mode`, not on the page.
15. First-token position: always render it, de-emphasised, never hidden.
16. If `make` is not available, use the raw commands in section 5.3.
17. FastAPI's generated OpenAPI describes 422 responses in its own default shape, not the error format in section 7.1. That is expected; the web client parses the error format itself in `lib/api.ts`.

---

## Appendix A: copy deck

All user-facing strings live in one file, `web/lib/copy.ts`, exactly as below.

| Key | Text |
|---|---|
| `app.name` | NeuronLens |
| `app.tagline` | See what's inside a language model. |
| `home.intro` | NeuronLens decodes a language model's internal activations into features you can browse, inspect and test. |
| `home.features` | Browse the concepts the decoder found. |
| `home.inspect` | Paste text and see which features fire on each token. |
| `home.steer` | Turn a feature up or down and compare the output. |
| `home.metrics` | See how faithful the decoder is. |
| `banner.synthetic` | SAMPLE DATA — everything here is synthetic and does not come from a trained model. |
| `banner.mock` | MOCK MODE — Inspect and Steer results are simulated. |
| `banner.offline` | Cannot reach the server at {url}. Start it and reload. |
| `note.steering` | Steering is a test of what a feature does. It is not a fix for wrong answers. |
| `note.limit` | Shows which features were active. It does not explain why the model chose a particular answer. |
| `feature.unlabelled` | Unlabelled feature |
| `feature.dead` | Dead feature: it never fired on the evaluation text. |
| `inspect.placeholder` | Paste or type text (up to 256 tokens) |
| `inspect.hint` | Paste some text and press Inspect to see which features fire on each token. |
| `inspect.outlier` | First token position. Values here are unusually large, so it is de-emphasised. |
| `loading.default` | Loading… |
| `loading.steer` | Generating… this can take several seconds. |
| `empty.features` | No features match these filters. |
| `empty.examples` | No examples are stored for this feature. |
| `empty.metrics` | No metrics were recorded for this run. |
| `empty.hist` | No density histogram for this run. |
| `empty.curve` | No training curve for this run. |
| `button.inspect` | Inspect |
| `button.steer` | Run steering |
| `button.random` | Random feature |
| `button.steerFeature` | Steer this feature |
| `button.tryAgain` | Try again |
| `button.randomSeed` | Randomize seed |
| `label.baseline` | Baseline |
| `label.steered` | Steered |
| `badge.saved` | Saved example |
| `badge.live` | Live |
| `tile.ceRecovered` | CE recovered — how much of the model's performance survives when the decoder's version replaces the real activations. |
| `tile.fvu` | FVU — the share of the original variation the decoder fails to rebuild. Lower is better. |
| `tile.dead` | Dead features — features that never fire. Lower is better. |
| `tile.l0` | L0 — average number of features active per token. |
| `notfound.title` | Page not found |
| `error.title` | Something went wrong |
| `error.generic` | Something went wrong. Try again. |
| `error.NETWORK` | Could not reach the server. |
| `error.INVALID_INPUT` | Some input is not valid: {message} |
| `error.TEXT_TOO_LONG` | That text is too long. The limit is 256 tokens. |
| `error.FEATURE_NOT_FOUND` | That feature does not exist. |
| `error.RUN_NOT_FOUND` | That run does not exist. |
| `error.NOT_FOUND` | Not found. |
| `error.MODEL_NOT_LOADED` | The model is not loaded on the server. |
| `error.SCHEMA_MISMATCH` | The server's database version is not supported. |
| `error.INTERNAL` | The server hit an error. Try again. |

---

## Appendix B: message templates

**Blocker** (the only question you should need to ask):
```
BLOCKER
Stage: <number and name>
What I need: <one sentence>
What I checked: <commands and files>
What I am doing meanwhile: <another stage, or nothing>
```

**Handoff you announce to the team:**
```
DONE: Stage <number> <name>
Merged: <pull request>
Next unblocked: <stage numbers>
```

**Decision record** (one line in `docs/decisions.md`):
```
A-<n>: <decision> — <reason>
```

---

*End of the App Track Build Guide. Build in the order of section 4. When the spec is silent, follow section 3.*
