# NeuronLens — Product Requirements Document (PRD)

| | |
|---|---|
| **Product** | NeuronLens: Interpreting Language Model Internals with Sparse Autoencoders |
| **Document** | PRD + technical spec, v1.0 |
| **Status** | Draft for kickoff |
| **Date** | 2026-10-04 |
| **ML Lead / Doc owner** | Prabodh |
| **App Lead(s)** | TBD (fill in at kickoff) |
| **Repo** | `neuronlens` (monorepo) |
| **Where this file lives** | `docs/PRD.md` |

> **How to read this document.** Sections 1–5 explain *what* and *why*. Sections 6–10 are the *contract* both tracks build against and are the most important part for the app team. Sections 11–13 are the per-track specs. Sections 14–19 are workflow, schedule, risks and decisions. If you only have ten minutes, read §2, §5, §6, §7 and §14.

---

## Table of contents

1. Summary
2. Problem, goals and non-goals
3. Users and user stories
4. Product scope and the claims we make
5. How the work is split (two tracks, one contract)
6. Architecture and monorepo layout
7. The data contract: SQLite schema
8. The data contract: HTTP API
9. Mock mode and fixtures
10. Requirements (functional and non-functional)
11. ML spec (ML track)
12. Server spec (App track)
13. Web spec (App track)
14. Development workflow
15. Phases, milestones and handoffs
16. Testing and quality
17. Risks
18. Decisions log and open questions
19. Glossary and references

---

## 1. Summary

A language model's internal state is a list of numbers that a human cannot read. NeuronLens trains a **sparse autoencoder (SAE)** on those numbers and turns them into a dictionary of named **features** (concepts). A web app lets a user:

1. **Browse** the features, each with the text that triggers it.
2. **Inspect** any pasted text to see which features fire on which tokens.
3. **Steer** the model by forcing a feature up or down and compare the output with and without the change.
4. **Review the evidence** — evaluation metrics that show how faithful the SAE is.

The base language model (GPT-2 small) is **frozen** and never modified. Only the SAE is trained.

**One-line description:** a tool that decodes a language model's internal activations into human-readable features, and lets you browse and test them from a browser.

---

## 2. Problem, goals and non-goals

### 2.1 Problem
When a language model produces a wrong, biased or unwanted output, there is no way to look inside and see what it was representing. Models store many more concepts than they have numbers, so concepts overlap (**superposition**) and single numbers mean several unrelated things (**polysemanticity**). Inspecting neurons one by one tells you nothing.

### 2.2 Goals

| ID | Goal | How we know we met it |
|---|---|---|
| G1 | Train a TopK SAE on GPT-2 small that reconstructs activations faithfully | Targets in §11.6 reported on a held-out set |
| G2 | Make the learned features inspectable by a non-expert | A new user can find a feature, read its examples and understand it in under a minute |
| G3 | Show a causal effect, not just correlation | At least one feature whose steering visibly and reproducibly changes generation |
| G4 | Be honest and reproducible | Every number in the demo comes from a logged run; the repo reproduces results from configs |
| G5 | Let two people work in parallel from day one | The app track ships all four screens against fixtures before real data exists |

### 2.3 Non-goals (this release)

- Training or fine-tuning a language model. The base model is never updated.
- Explaining why the model chose one particular output. We show which concepts were active; we do not claim a complete causal account.
- Automatically fixing wrong answers or factual mistakes. Steering is a **test**, not a repair.
- Models larger than GPT-2 small (the Qwen2.5-1.5B config is a documented stretch, out of scope for v1).
- Multi-user accounts, authentication, cloud deployment, or production traffic.
- Automatic labelling of all features. Labels are curated by hand for a subset.

---

## 3. Users and user stories

### 3.1 Personas

| Persona | Who | What they need |
|---|---|---|
| **The Evaluator** | Hackathon judge or college examiner with no ML background | Understand the idea in seconds and see something work live |
| **The Researcher** | ML student or engineer curious about model internals | Browse features, check the evaluation, reproduce the run |
| **The Auditor** (illustrative) | Someone who must check a model for sensitive concepts | See whether concepts like gender or region are active on a given text |

### 3.2 User stories

| ID | As a… | I want to… | So that… |
|---|---|---|---|
| US-1 | Evaluator | paste a sentence and see which concepts lit up per word | I immediately see what the tool does |
| US-2 | Evaluator | click a concept and see texts that trigger it | I believe the concept means something |
| US-3 | Evaluator | turn a concept up and compare outputs side by side | I see that the concept affects the model |
| US-4 | Researcher | search and sort features by density and strength | I can find interesting and broken features |
| US-5 | Researcher | click a random feature | I can judge quality without cherry-picking |
| US-6 | Researcher | see FVU, CE-recovered, dead-feature % and plots | I can judge how faithful the SAE is |
| US-7 | Auditor | check whether a named concept was active on a text | I have a starting point for an audit (not a verdict) |
| US-8 | Any user | see a clear banner if the data is sample data | I am never misled by fixtures |

---

## 4. Product scope and the claims we make

### 4.1 Screens in scope

1. **Feature Explorer** — searchable, sortable list.
2. **Feature Detail** — examples with token heatmap, stats, logit attribution.
3. **Text Inspector** — paste text, see per-token features.
4. **Steering Panel** — choose feature, strength and prompt; compare baseline and steered output.
5. **Metrics Dashboard** — evaluation results and plots.

### 4.2 Claims policy (binding on UI copy, README, slides and demo)

| We may say | We must not say |
|---|---|
| "Shows which concepts were active" | "Explains why the model gave this answer" |
| "Check whether a concept like gender was active" | "Proves bias", "detects bias" |
| "Test what a feature does by steering it" | "Fixes wrong answers", "instantly fixes behaviour" |
| "We implemented a published method (TopK SAE)" | "We invented", "nobody has done this" |
| "Keeps X% of the model's performance" (only with a logged number) | Any metric not backed by a run in `metrics` |

Anything shown from fixtures must carry the **sample data banner** (FR-60).

---

## 5. How the work is split (two tracks, one contract)

The ML work is the long, uncertain part. The app work must not wait for it. Both tracks therefore build against a **frozen data contract** (§7 and §8) and a **mock mode** (§9).

```
        ML TRACK (Prabodh)                         APP TRACK (teammates)
        ─────────────────                          ─────────────────────
  A0 env ─ A1 pipeline ─ A2 SAE smoke ─ …          B1 server in mock mode
        │                    │                           │
        │            H2: first valid                B2 Explorer + Detail
        │            (junk-quality) DB ────────────▶ B3 Inspector
        │                                           B4 Steering UI
  A3 full training ─ A4 eval ─ A5 index ─ A6 steer  B5 Metrics page
        │                                           B6 tests, polish
        └──────── H3: final DB + checkpoint ───────▶ C integration
```

### 5.1 Ground rules

1. **The contract is the boundary.** `contracts/schema.sql` and the API in §8 are the only things the tracks share. Internals on either side can change freely.
2. **ML may improvise; the contract may not.** Training will involve failed runs and rewrites. None of that should leak into the app. The ML track only has to keep `build_index.py` producing a database that satisfies the schema.
3. **Additive changes are cheap; breaking changes are expensive.** See §14.4.
4. **Integration starts early.** At handoff H2 the ML track delivers a real, valid database even if the features are poor, so integration bugs show up in week 2, not week 6.
5. **App-track developers never need a GPU, CUDA or PyTorch.** Mock mode runs on any laptop.

### 5.2 Roster

| Role | Name | Owns |
|---|---|---|
| ML Lead | Prabodh | `ml/`, `configs/`, `contracts/schema.sql` (jointly), labels |
| App Lead | TBD | `server/`, `web/`, `contracts/` (jointly) |
| App Dev | TBD | assigned screens in `web/`, tests, docs |
| Demo / QA | TBD | demo script, recording, README, final checks |

---

## 6. Architecture and monorepo layout

### 6.1 Runtime architecture

```
 text corpus ─▶ frozen GPT-2 ─▶ hook @ layer 8 ─▶ activation stream
                                                        │
                                              shuffle buffer + normalise
                                                        │
                                                 TopK SAE (trained)
                                                        │
                    ┌───────────────────────────────────┼────────────────────────┐
                    ▼                                   ▼                        ▼
              evaluation suite                  feature interpretation     steering function
              (FVU, L0, CE-recovered)           (top examples, logits)     (clamp / amplify)
                    │                                   │                        │
                    └──────────────▶  build_index.py ◀──┘                        │
                                            │                                    │
                                     neuronlens.db (SQLite)                      │
                                            │                                    │
                                            ▼                                    ▼
                                 FastAPI server  ◀────── real mode imports ml/ ──┘
                                            │
                                      HTTP /api/v1
                                            │
                                      Next.js web app
```

Two ways the server gets its answers:

- **Read-only data** (features, examples, metrics) always comes from the SQLite file.
- **Live compute** (`/inspect`, `/steer`) comes from the real model and SAE in **real mode**, or from deterministic fakes in **mock mode**.

### 6.2 Monorepo layout

```
neuronlens/
├── README.md
├── Makefile
├── .env.example
├── .gitignore
├── .github/
│   └── workflows/ci.yml
├── docs/
│   ├── PRD.md                      # this file
│   ├── results.md                  # filled in by ML track as runs complete
│   └── figures/
├── contracts/                      # THE SHARED BOUNDARY
│   ├── schema.sql                  # SQLite schema (source of truth)
│   ├── openapi.json                # exported from FastAPI, committed
│   └── fixtures/
│       ├── make_fixtures.py        # builds fixtures/sample.db
│       └── sample.db               # synthetic data (committed, small)
├── ml/                             # ML TRACK
│   ├── pyproject.toml              # installable package: neuronlens
│   ├── requirements.txt
│   ├── configs/
│   │   ├── gpt2-small.yaml
│   │   └── smoke.yaml
│   ├── labels.json                 # human-written feature labels (committed)
│   ├── neuronlens/
│   │   ├── config.py
│   │   ├── activations.py          # hooks, streaming, shuffle buffer, normalisation
│   │   ├── sae.py                  # TopK SAE, AuxK, init, decoder constraints
│   │   ├── train.py
│   │   ├── evals.py                # FVU, L0, density, CE-recovered
│   │   ├── interpret.py            # top examples, logit attribution
│   │   ├── steering.py             # clamp / amplify during generation
│   │   └── build_index.py          # writes neuronlens.db to contracts/schema.sql
│   ├── scripts/
│   │   ├── run_training.py
│   │   ├── run_evals.py
│   │   └── make_index.py
│   └── tests/
├── server/                         # APP TRACK
│   ├── requirements.txt            # fastapi, uvicorn, pydantic — no torch
│   ├── app/
│   │   ├── main.py
│   │   ├── settings.py             # env vars, mode switch
│   │   ├── db.py                   # read-only SQLite access
│   │   ├── models.py               # Pydantic response models
│   │   ├── routers/                # features.py, inspect.py, steer.py, metrics.py, health.py
│   │   └── services/
│   │       ├── mock.py             # deterministic fakes (mock mode)
│   │       └── real.py             # imports neuronlens lazily (real mode)
│   └── tests/
├── web/                            # APP TRACK
│   ├── package.json
│   ├── app/                        # Next.js App Router pages
│   ├── components/
│   ├── lib/
│   │   ├── api.ts                  # typed client
│   │   ├── types.gen.ts            # generated from contracts/openapi.json
│   │   └── stores/                 # Zustand
│   └── tests/
├── data/                           # GITIGNORED: checkpoints, real db, caches
└── scripts/
    └── dev.sh
```

### 6.3 Ownership

| Path | Owner | Who may change it |
|---|---|---|
| `ml/` | ML Lead | ML Lead; others via PR |
| `server/`, `web/` | App Lead | App track; ML Lead via PR |
| `contracts/` | **Both** | **PR approved by both leads** |
| `docs/PRD.md` | ML Lead | Anyone via PR; decisions go in §18 |
| `ml/labels.json` | ML Lead | Anyone via PR (labels need a reviewer) |

---

## 7. The data contract: SQLite schema

`contracts/schema.sql` is the source of truth. The file is a single SQLite database, opened **read-only** by the server. It describes **one primary run** (`meta.run_id`). Other runs (for example a k-sweep) appear only in `runs`, `metrics`, `density_hist` and `training_curve`.

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

### 7.1 Rules that follow from the schema

- **Never hard-code `12288`, `768`, `32` or `8`.** Read `d_sae`, `d_model`, `k` and `hook_layer` from `meta`. The ML track may switch the model, change the expansion factor, or use a pretrained SAE as a fallback, and the app must keep working.
- `feature_id` is dense in `[0, d_sae)`. Dead features still have rows (`is_dead = 1`) so counts add up.
- Examples are windows of at most 64 tokens around the peak token, so rows stay small. Expect the real database to be **several hundred MB** (estimate: ~32 examples × thousands of live features); it is **never committed to git**.
- `schema_version` is `MAJOR.MINOR`. The server refuses to start if the major version differs from the one it was built for.
- `is_synthetic = "true"` for `fixtures/sample.db`. The UI must show the sample data banner whenever this is true.

---

## 8. The data contract: HTTP API

Base path: `/api/v1`. JSON in, JSON out, `snake_case` keys. The source of truth is the FastAPI app; its OpenAPI output is committed as `contracts/openapi.json`, and `web/lib/types.gen.ts` is generated from it (`npm run gen:types`). Do not hand-write response types.

### 8.1 Error format (all endpoints)

```json
{ "error": { "code": "FEATURE_NOT_FOUND", "message": "No feature with id 99999" } }
```

| Code | HTTP | Meaning |
|---|---|---|
| `INVALID_INPUT` | 422 | Validation failed (bad field, out-of-range value) |
| `TEXT_TOO_LONG` | 413 | Input exceeds the token limit |
| `FEATURE_NOT_FOUND` | 404 | Unknown `feature_id` |
| `RUN_NOT_FOUND` | 404 | Unknown `run_id` |
| `MODEL_NOT_LOADED` | 503 | Real mode but model or checkpoint unavailable |
| `SCHEMA_MISMATCH` | 500 | Database major version not supported |
| `INTERNAL` | 500 | Anything else |

### 8.2 Endpoints

#### `GET /health`
```json
{
  "status": "ok",
  "mode": "mock",
  "is_synthetic": true,
  "schema_version": "1.0",
  "model_name": "gpt2",
  "hook_layer": 8,
  "d_model": 768,
  "d_sae": 12288,
  "k": 32,
  "run_id": "fixture-001"
}
```

#### `GET /features`
Query parameters:

| Param | Type | Default | Notes |
|---|---|---|---|
| `query` | string | – | Case-insensitive substring of `label`, or an integer matching `feature_id` |
| `status` | `unlabelled` \| `named` \| `ambiguous` | all | |
| `include_dead` | bool | `false` | |
| `sort` | `feature_id` \| `density` \| `max_activation` | `feature_id` | |
| `order` | `asc` \| `desc` | `asc` | |
| `limit` | int 1–200 | 50 | |
| `offset` | int ≥ 0 | 0 | |

```json
{
  "total": 11873,
  "limit": 50,
  "offset": 0,
  "items": [
    {
      "feature_id": 8412,
      "label": "Tamil script",
      "label_status": "named",
      "density": 0.0021,
      "max_activation": 14.7,
      "is_dead": false
    }
  ]
}
```

#### `GET /features/{feature_id}`
```json
{
  "feature_id": 8412,
  "label": "Tamil script",
  "label_status": "named",
  "density": 0.0021,
  "max_activation": 14.7,
  "mean_activation": 3.2,
  "is_dead": false,
  "notes": null,
  "examples": [
    {
      "rank": 1,
      "tokens": ["வ", "ண", "க்", "கம்", " ", "எப்படி"],
      "activations": [0.0, 0.0, 9.1, 14.7, 0.0, 2.2],
      "peak_index": 3,
      "max_activation": 14.7
    }
  ],
  "logits": {
    "promoted":   [{ "token": " என்", "score": 1.84 }],
    "suppressed": [{ "token": " the", "score": -0.91 }]
  }
}
```
`examples` has up to 32 entries ordered by `rank`. `logits` lists have up to 10 entries each, strongest first.

#### `GET /features/random`
Returns the same shape as `GET /features/{feature_id}` for a random **non-dead** feature. Optional `status` filter.

#### `POST /inspect`
Request:
```json
{ "text": "Vanakkam, epdi irukkeenga?", "top_n": 5 }
```
`text` is limited to **256 tokens**. `top_n` is 1–20.

Response:
```json
{
  "model_name": "gpt2",
  "tokens": [
    {
      "index": 0,
      "text": "Van",
      "is_outlier_position": true,
      "features": [
        { "feature_id": 8412, "activation": 6.3, "label": "Tamil script" }
      ]
    }
  ]
}
```
Features per token are sorted by activation, strongest first. Position 0 is flagged `is_outlier_position` because the first token's activation is an extreme outlier in GPT-2; the UI should de-emphasise it.

#### `POST /steer`
Request:
```json
{
  "prompt": "The weather today is",
  "feature_id": 8412,
  "mode": "amplify",
  "strength": 4.0,
  "max_new_tokens": 60,
  "seed": 42
}
```

| Field | Rule |
|---|---|
| `mode` | `amplify` adds the feature direction; `ablate` sets the feature to zero |
| `strength` | float in `[0, 10]`. Used only by `amplify`; ignored by `ablate`. `1.0` = add the feature at its maximum observed activation |
| `max_new_tokens` | 1–120 |
| `seed` | int. The **same seed is used for the baseline and the steered run**, so any difference is caused by the intervention |

Response:
```json
{
  "feature": { "feature_id": 8412, "label": "Tamil script" },
  "mode": "amplify",
  "strength": 4.0,
  "seed": 42,
  "baseline_text": "The weather today is sunny and warm …",
  "steered_text": "The weather today is வெயில் மற்றும் …"
}
```

#### `GET /steering-examples`
Query: `feature_id` (optional), `curated_only` (bool). Returns pre-recorded results from `steering_examples`. This is the **demo safety net** and works in both modes.

```json
{
  "items": [
    {
      "example_id": 3,
      "feature_id": 8412,
      "mode": "amplify",
      "strength": 4.0,
      "prompt": "The weather today is",
      "seed": 42,
      "baseline_text": "…",
      "steered_text": "…",
      "is_curated": true
    }
  ]
}
```

#### `GET /runs`
```json
{ "items": [ { "run_id": "gpt2-l8-k32", "k": 32, "d_sae": 12288, "n_train_activations": 100000000, "created_at": "2026-10-20T10:00:00Z" } ] }
```

#### `GET /metrics`
Query: `run_id` (default: the primary run).
```json
{
  "run_id": "gpt2-l8-k32",
  "metrics": [
    { "name": "fvu", "value": 0.12 },
    { "name": "ce_recovered", "value": 0.86 }
  ],
  "density_hist": [ { "bin_lo": -6.0, "bin_hi": -5.5, "count": 410 } ],
  "training_curve": [ { "step": 100, "loss": 0.91, "fvu": 0.4, "dead_pct": 12.0 } ]
}
```

### 8.3 Cross-cutting rules

- **CORS:** allow `http://localhost:3000` in development.
- **Determinism:** `/steer` with identical inputs returns identical output (seeded). Mock mode is deterministic too.
- **No persistence of user input.** `/inspect` and `/steer` inputs are not stored or logged beyond request metadata.
- **No outbound calls at runtime.** The app runs fully offline once the model and database are on disk.
- **Latency targets (to be measured, not promised):** list and detail under 500 ms; `/inspect` under 2 s for 128 tokens on the target laptop; `/steer` under 10 s for 60 new tokens.

---

## 9. Mock mode and fixtures

Mock mode is what lets the app track work before any ML exists.

### 9.1 Mode switch

```
NEURONLENS_MODE=mock    # default; no GPU, no torch, uses fixtures
NEURONLENS_MODE=real    # needs the neuronlens package, a checkpoint and a GPU
NEURONLENS_DB=./contracts/fixtures/sample.db
NEURONLENS_CKPT=./data/checkpoints/<run>.pt        # real mode only
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 9.2 What mock mode does

| Endpoint | Mock behaviour |
|---|---|
| `/features*`, `/metrics`, `/runs`, `/steering-examples` | Read `sample.db` exactly as real mode reads the real database |
| `/inspect` | Deterministic fake: hash each token to pick a few feature ids from the database and generate stable fake activations |
| `/steer` | If a matching row exists in `steering_examples`, return it; otherwise return a clearly templated placeholder pair |
| `/health` | `"mode": "mock"`, `"is_synthetic": true` |

### 9.3 Fixtures (`contracts/fixtures/make_fixtures.py`)

- Creates `sample.db` from `schema.sql`.
- About **500 synthetic features** with plausible-looking but obviously fake labels (prefix labels with `[sample]`), a spread of densities including a few dead features, 8–32 synthetic examples per feature, logits, one fake run with metrics, a density histogram, a training curve and ~10 steering examples.
- Must be generated by a seeded script so it is reproducible, and committed at a small size (target under 5 MB).
- CI rebuilds it and fails if the result does not validate against `schema.sql`.

### 9.4 Safety rules for mock data

- `is_synthetic = "true"` in `meta`; the web app shows a persistent **SAMPLE DATA** banner (FR-60).
- **The demo must never run in mock mode.** Add a pre-demo check (`make demo-check`) that fails if `/health` reports `mock` or `is_synthetic: true`.
- Never copy fixture labels into slides or the report.

---

## 10. Requirements

Priority: **M** = must have, **S** = should have, **C** = could have, **W** = won't have in v1.

### 10.1 Feature Explorer

| ID | Requirement | Pri | Acceptance criteria |
|---|---|---|---|
| FR-10 | Paginated list of features (50 per page) | M | First page renders in under 500 ms from local data; total count shown |
| FR-11 | Search by label substring or numeric id | M | Typing `8412` jumps to that feature; `tamil` returns labelled matches |
| FR-12 | Sort by id, density, max activation; filter by label status; hide dead by default | M | Sort and filter state reflected in the URL |
| FR-13 | "Random feature" button | M | Opens a random non-dead feature, labelled or not |

### 10.2 Feature Detail

| ID | Requirement | Pri | Acceptance criteria |
|---|---|---|---|
| FR-20 | Show up to 32 top examples with token-level heatmap | M | Intensity encodes activation; hover shows the numeric value (not colour-only) |
| FR-21 | Show label, status, density, max and mean activation, dead flag | M | Values match `GET /features/{id}` |
| FR-22 | Show top promoted and suppressed tokens | S | Up to 10 each, with scores |
| FR-23 | "Steer this feature" shortcut | S | Opens the Steering Panel with the feature prefilled |

### 10.3 Text Inspector

| ID | Requirement | Pri | Acceptance criteria |
|---|---|---|---|
| FR-30 | Paste text (≤ 256 tokens) and see per-token features | M | Every token shows its top-N features; outlier position de-emphasised |
| FR-31 | Hover a token for its features; click a feature to open its detail page | M | Works with keyboard focus too |
| FR-32 | Loading, empty and error states | M | Over-length text shows a clear message, not a crash |

### 10.4 Steering Panel

| ID | Requirement | Pri | Acceptance criteria |
|---|---|---|---|
| FR-40 | Choose feature, strength and prompt; show baseline and steered text side by side | M | Both outputs visible together; labelled |
| FR-41 | Same seed for both runs; seed visible and editable | M | Re-running with the same inputs gives the same text |
| FR-42 | `amplify` and `ablate` modes | S | Mode toggle; strength control disabled for `ablate` |
| FR-43 | Load a saved example from `steering_examples` | M | One click fills the form and shows the stored result (demo safety net) |
| FR-44 | Warn that steering is a test, not a fix | M | One-line note on the panel, in line with §4.2 |

### 10.5 Metrics Dashboard

| ID | Requirement | Pri | Acceptance criteria |
|---|---|---|---|
| FR-50 | Show FVU, L0, dead %, CE original / zero-ablation / reconstructed / recovered | M | Values equal `GET /metrics`; CE-recovered is the most prominent |
| FR-51 | Density histogram and training curve | S | Plots render from the API data |
| FR-52 | Compare runs (k-sweep) | C | Select run via `run_id` |

### 10.6 Platform

| ID | Requirement | Pri | Acceptance criteria |
|---|---|---|---|
| FR-60 | Persistent **SAMPLE DATA** banner when `is_synthetic` is true | M | Visible on every page |
| FR-61 | Show model name, hook layer, `d_sae` and `k` from `/health` | M | No hard-coded values anywhere in the UI |
| FR-62 | Consistent error format and a user-friendly error page | M | Server errors follow §8.1 |
| FR-63 | Works offline after setup | M | No network requests to third parties at runtime |

### 10.7 ML

| ID | Requirement | Pri | Acceptance criteria |
|---|---|---|---|
| FR-70 | Stream activations with a shuffle buffer and normalisation; never store raw activations | M | Pipeline sustains the throughput gate in §11.3 |
| FR-71 | TopK SAE with AuxK loss, unit-norm decoder, specified init | M | Tests in §16 pass |
| FR-72 | Save checkpoint together with config and normalisation constants | M | A checkpoint alone is enough to run inference |
| FR-73 | Evaluation suite: FVU, L0, density, CE-recovered with zero-ablation baseline | M | Numbers written to `metrics`; identity-SAE sanity test gives 100% recovered |
| FR-74 | `build_index.py` writes a database valid against `contracts/schema.sql` | M | CI validates the output schema; row counts logged |
| FR-75 | Top-32 activating examples per live feature; logit attribution; density stats | M | Rows present in `feature_examples`, `feature_logits`, `features` |
| FR-76 | Steering function (amplify and ablate) with seeded generation | M | Same seed gives identical baseline text across runs |
| FR-77 | k-sweep at 16 / 32 / 64 | C | Three runs in `runs` and `metrics` |
| FR-78 | Qwen2.5-1.5B configuration | W | Out of scope for v1 |

### 10.8 Non-functional requirements

| ID | Area | Requirement |
|---|---|---|
| NFR-1 | Portability | App track runs on Windows, macOS or Linux with no GPU, in mock mode |
| NFR-2 | Reproducibility | Every run is defined by a config file; seeds, library versions and git commit are logged with each run |
| NFR-3 | Performance | Latency targets in §8.3, measured on the RTX 4070 laptop and reported as measured values |
| NFR-4 | Privacy | Runs locally; inputs not stored; no third-party calls at runtime |
| NFR-5 | Accessibility | Keyboard navigable; heatmaps never rely on colour alone; sufficient text contrast |
| NFR-6 | Code quality | Type-checked (`mypy` or `pyright` for server, `tsc` for web), linted, formatted; CI must pass before merge |
| NFR-7 | Honesty | UI copy and docs follow the claims policy in §4.2 |

---

## 11. ML spec (ML track)

### 11.1 Environment

| Item | Value |
|---|---|
| Hardware | RTX 4070 Laptop, 8 GB VRAM |
| OS | Windows with **WSL2 + CUDA** for training (Triton does not run natively on Windows, so `torch.compile` needs WSL2) |
| Python | 3.11 |
| Core libraries | PyTorch, `transformers`, `datasets`, `numpy`, `pyyaml`, `tensorboard` |
| Pinning | All versions pinned in `ml/requirements.txt`; pip needs `--break-system-packages` only on system Python, so use a venv |

### 11.2 Model and hook

| Item | Value |
|---|---|
| Model | `gpt2` (GPT-2 small, 124M) loaded with `GPT2LMHeadModel.from_pretrained("gpt2")`, `.eval()`, `torch.no_grad()`, fp16 |
| Hook point | Output of transformer block 8, i.e. the residual stream after block 8 (`model.transformer.h[8]`; the block returns a tuple, take element 0) |
| `d_model` | 768 |
| Sequence length | 1024 tokens per row |
| First position | The activation at position 0 is an extreme outlier in GPT-2. **Exclude position 0 from training** and flag it in `/inspect` |

### 11.3 Activation pipeline (`activations.py`)

| Requirement | Detail |
|---|---|
| Source | `HuggingFaceFW/fineweb-edu` streamed (the `sample-10BT` config) via `datasets` streaming. Verify it loads on day one; fall back to another streamed web-text corpus if not |
| Streaming | Generate activations on the fly and discard after use. 100M activations in fp16 at 768 dims would be ~150 GB, so nothing large is ever written to disk |
| Shuffle buffer | About 500,000 activation vectors in memory. Fill from the stream, sample batches at random, refill the drawn slots. Without this, batches are correlated and training destabilises |
| Normalisation | From a sample, compute a mean `mu` and a scalar `s` so that the mean L2 norm of `(x - mu) * s` equals `sqrt(d_model)`. Persist `mu` and `s` with the checkpoint. Inference and steering must use the same transform |
| Batch | 4096 activations per step |
| **Throughput gate** | Sustain over 2,000 steps per minute while training. If not, fix the pipeline before touching the SAE |

### 11.4 SAE (`sae.py`)

| Item | Value |
|---|---|
| Architecture | `z = TopK(W_enc (x - b_dec) + b_enc)`; `x_hat = W_dec z + b_dec` |
| Sizes | `d_sae = 16 × d_model = 12288`; `k = 32` |
| Sparsity | Exactly `k` features active per token (TopK). No L1 term |
| Decoder | Columns unit-normalised; renormalise after every optimiser step; remove the gradient component parallel to each column before the step |
| Init | `W_enc = W_dec^T`; `b_dec` = geometric median of a sample batch; `b_enc = 0` |
| Loss | `MSE(x, x_hat) + (1/32) · AuxK` |
| AuxK | Track last-fired step per feature; mark features unfired for a threshold window as dead; take the top 512 dead features and train them to reconstruct the main residual error `x - x_hat` |
| Optimiser | Adam, betas (0.9, 0.999), lr 3e-4 with cosine decay |
| Precision | bf16 autocast |
| Budget | Target **100M** activations. If time-boxed, train on whatever fits and **report the true number** |

### 11.5 Stages

| Stage | Purpose | Settings | Output |
|---|---|---|---|
| **0 — Smoke** | Prove the loop runs end to end | ~5M activations, ~1–2 h | Disposable checkpoint; **a valid but low-quality database for handoff H2** |
| **1 — Full run** | The deliverable | ~100M activations, ~8–12 h | Final checkpoint, evaluation, database for handoff H3 |
| Stretch | k-sweep (16/32/64) | Same, three runs | Extra rows in `runs` and `metrics` |

If Stage 1 is at risk, a **pretrained GPT-2 small residual-stream SAE** from the public SAELens releases (as far as I know, `gpt2-small-res-jb`) can serve as a fallback so the app, steering and demo are never blocked. It will have a different `d_sae`, which is why the app reads sizes from `meta`. Verify availability and licence before relying on it, and label it honestly as pretrained if used.

### 11.6 Evaluation (`evals.py`) and targets

Evaluated on **held-out** text not used in training.

| Metric | Definition | Target |
|---|---|---|
| `fvu` | Residual variance ÷ total variance | below 0.15 |
| `l0` | Mean active features per token | exactly `k` |
| `dead_pct` | Features that never fire on the evaluation set | below 5% |
| `ce_original` | Cross-entropy of the unmodified model | – |
| `ce_zero_ablation` | Cross-entropy with the hook-layer activation replaced by zeros | – |
| `ce_reconstructed` | Cross-entropy with the activation replaced by the SAE reconstruction (de-normalised) | – |
| `ce_recovered` | `(ce_zero_ablation − ce_reconstructed) / (ce_zero_ablation − ce_original)` | above 0.80 |

Targets are goals, not guarantees. If they are missed, report the real numbers and a diagnosis. Also produce the density log-histogram and flag the two failure spikes: near zero (dead) and near one (always on).

### 11.7 Interpretation (`interpret.py`)

1. Scan a held-out corpus (target **10M tokens**; reduce if slow) and record, per feature: density, mean and max activation.
2. Keep the **top 32 examples** per live feature as windows of ≤ 64 tokens with per-token activations.
3. **Logit attribution:** `W_dec[:, i] @ W_U` (unembedding). Store the top 10 promoted and top 10 suppressed tokens.
4. **Labels** are human-written in `ml/labels.json` (`{ "8412": { "label": "...", "status": "named" } }`) and merged by `build_index.py`. A label needs agreement between examples, logits and a steering check; otherwise mark it `ambiguous`.

### 11.8 Steering (`steering.py`)

Intervene at the hook layer during generation, seeded:

- **amplify:** `x' = x + strength · max_act_i · d_i`, where `d_i = W_dec[:, i]`. The SAE works in normalised space, so convert the delta back to raw scale by dividing by `s` before adding it to the raw residual stream.
- **ablate:** subtract the feature's current contribution (`z_i · d_i`, de-normalised) from the residual stream.
- Run the baseline and the steered generation with the **same seed**.
- Curate 5–10 results into `steering_examples` for the demo safety net.

### 11.9 Index builder (`build_index.py`)

Reads the checkpoint, evaluation results, interpretation outputs and `labels.json`, and writes one SQLite file that **must validate against `contracts/schema.sql`**. It logs row counts per table and fails loudly if a required `meta` key is missing. This script is the ML track's only obligation to the app track.

### 11.10 ML deliverables

Final checkpoint with config and normalisation constants; `neuronlens.db`; `docs/results.md` with the metrics table, density histogram, training curve and honest notes on failures; at least one feature with a demonstrated causal steering effect.

---

## 12. Server spec (App track)

| Item | Decision |
|---|---|
| Framework | FastAPI + Uvicorn, Pydantic v2 response models |
| Database access | Read-only SQLite connection to `NEURONLENS_DB` |
| Startup checks | Verify required `meta` keys and schema major version; fail fast with `SCHEMA_MISMATCH` |
| Modes | `services/mock.py` and `services/real.py` behind one interface; `real.py` imports `neuronlens`, `torch` lazily so mock mode never needs them |
| OpenAPI | `make openapi` exports `contracts/openapi.json`; CI fails if it is stale |
| Real mode | Loads GPT-2 and the SAE once at startup; `/inspect` and `/steer` run under a lock so concurrent requests cannot corrupt the GPU state |
| Logging | Structured request logs (method, path, status, duration); never log request bodies |
| Tests | See §16 |

---

## 13. Web spec (App track)

| Item | Decision |
|---|---|
| Framework | Next.js (App Router) + TypeScript; pin the version in `package.json` |
| Styling | Tailwind CSS **v3** (pinned; v4 has breaking changes) |
| State | Zustand for cross-page state (selected feature, steering form, inspector text) |
| Charts | Recharts for the density histogram, training curve and metric bars |
| Icons | Lucide React |
| API client | `lib/api.ts` using types generated into `lib/types.gen.ts` |

### 13.1 Routes

| Route | Screen | Requirements |
|---|---|---|
| `/` | Landing with one-line description, health summary and links | FR-61 |
| `/features` | Explorer | FR-10 to FR-13 |
| `/features/[id]` | Detail | FR-20 to FR-23 |
| `/inspect` | Text Inspector | FR-30 to FR-32 |
| `/steer` | Steering Panel | FR-40 to FR-44 |
| `/metrics` | Metrics Dashboard | FR-50 to FR-52 |

### 13.2 Core components

`ModeBanner` (FR-60), `FeatureTable`, `FeatureCard`, `TokenHeatmap` (used by Detail and Inspector; shows numeric value on hover and focus), `SnippetList`, `LogitList`, `SteeringCompare`, `MetricTile`, `DensityHistogram`, `LossCurve`, `ErrorState`, `EmptyState`, `Skeleton`.

### 13.3 UX rules

- Never show a spinner without a way to see what failed. Every request has loading, empty and error states.
- Steering results always show the seed and mode, and the note from FR-44.
- Do not use alarming or definitive language ("proves", "detects", "fixes").
- Design for a projector: large type, high contrast, and nothing important in tiny text.

---

## 14. Development workflow

### 14.1 Setup

```
git clone <repo> && cd neuronlens
make setup           # app track: server venv + web npm install (no GPU needed)
make fixtures        # builds contracts/fixtures/sample.db
make dev-server      # FastAPI in mock mode on :8000
make dev-web         # Next.js on :3000
make test            # all non-GPU tests
make lint
make openapi         # refresh contracts/openapi.json
make gen-types       # regenerate web/lib/types.gen.ts
make demo-check      # fails if the server is in mock mode or the data is synthetic
```
ML track additionally: `make ml-setup` (inside WSL2 with CUDA), then `python ml/scripts/run_training.py --config ml/configs/smoke.yaml`.

Versions: Python 3.11, Node 20 LTS, npm.

### 14.2 Git

- Trunk-based: `main` is always working. Short-lived branches `feat/<area>-<thing>`, `fix/…`, `docs/…`.
- Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
- Every change goes through a PR with at least one review. PRs touching `contracts/` need **both leads**.
- Large artefacts (checkpoints, real database, activations) are **never committed**. They live in `data/` (gitignored) and are shared by a release asset or shared drive; record the file name and checksum in `docs/results.md`.

### 14.3 CI (`.github/workflows/ci.yml`)

1. Lint and format check (Python and TypeScript).
2. Type checks (`tsc`, Python type checker).
3. Server tests and web tests in mock mode.
4. **Contract checks:** build `sample.db` from `schema.sql`; check `openapi.json` and `types.gen.ts` are up to date; run `build_index` validation against a tiny synthetic input.
5. ML unit tests that need no GPU (shapes, TopK, normalisation round trip, schema writer).

GPU-dependent runs are done locally by the ML Lead and logged, not in CI.

### 14.4 Contract change policy

| Change | Allowed how |
|---|---|
| Add a nullable column, a new table or an optional API field | Allowed. Bump `schema_version` **minor**. Update fixtures and `openapi.json` in the same PR |
| Rename or remove a column or field, change a type or meaning | **Breaking.** Bump `schema_version` **major**. Needs both leads' approval and a migration note |
| Change anything inside `ml/` or inside `server/` internals | Free, as long as the contract still holds |

When the ML track needs a new piece of data, add it as an **optional** field first, so the app track is never blocked.

### 14.5 Definition of done (any ticket)

Code merged to `main`; tests added or updated; CI green; types regenerated if the API changed; docs updated if behaviour changed; no new claim that breaks §4.2.

---

## 15. Phases, milestones and handoffs

The deadline is not fixed in this document. The schedule below is a **6-week reference plan**. If time is shorter, apply the cut list in §15.4 rather than skipping testing.

### 15.1 Phases

| Phase | Owner | Content | Exit criterion |
|---|---|---|---|
| **0 — Foundation** | ML Lead (template) + App Lead | Push monorepo template, freeze contract v1.0, fixtures, CI, `make` targets | A fresh clone runs `make setup && make fixtures && make dev-server && make dev-web` and shows sample data |
| **A — ML** | ML Lead | A0 env and WSL2 · A1 activation pipeline · A2 SAE and smoke run · A3 full training · A4 evaluation · A5 interpretation and index · A6 steering and labels | Final `neuronlens.db` passes schema validation; metrics logged; one feature steers visibly |
| **B — App (no ML)** | App track | B1 server in mock mode · B2 Explorer and Detail · B3 Inspector · B4 Steering Panel · B5 Metrics page · B6 tests, polish, accessibility | All five screens complete against `sample.db`; CI green; Playwright smoke test passes |
| **C — Integration** | Both | Point the server at the real database; run real `/inspect` and `/steer`; fix contract gaps; rehearse the demo | `make demo-check` passes; demo script (§15.5) runs end to end |

### 15.2 Reference schedule

| Week | ML track | App track | Gate |
|---|---|---|---|
| 1 | Phase 0, A0 env, A1 pipeline | Phase 0, B1 server (mock) | **H1: contract v1.0 frozen** (day 2) |
| 2 | A2 SAE and smoke run | B2 Explorer and Detail | **H2: first real (poor-quality) database delivered** |
| 3 | A3 full training (overnight runs) | B3 Inspector, B4 Steering UI | – |
| 4 | A4 evaluation, A5 interpretation and index | B5 Metrics page, polish | – |
| 5 | A6 steering and labels; **H3: final database** | Phase C integration begins | **H3: final database and checkpoint** |
| 6 | Hardening, `results.md` | Tests, recording, README | Demo rehearsal, `make demo-check` |

### 15.3 Handoffs

| ID | When | What is handed over | Why it matters |
|---|---|---|---|
| H1 | Day 2 | Frozen `schema.sql`, API spec and `sample.db` | Lets the app track start immediately |
| H2 | End of week 2 | A real `neuronlens.db` from the smoke run, even if the features are junk | Surfaces integration bugs (sizes, encodings, performance) early |
| H3 | Week 5 | Final database, checkpoint, `results.md`, curated `steering_examples` | Starts real integration and demo prep |

### 15.4 Cut list (if time runs short)

Drop in this order. Items marked M in §10 are not on the list.

1. k-sweep and run comparison (FR-52, FR-77)
2. Density histogram and training curve in the UI (FR-51)
3. `ablate` mode (FR-42)
4. Logit attribution in the UI (FR-22)
5. Extra polish and animations

If training time is the limit, train on fewer activations and report the true number.

### 15.5 Demo script (acceptance test for v1)

1. Open `/inspect`, paste a prepared sentence; features appear per token. *(US-1)*
2. Click a feature; its 32 examples share a theme. *(US-2)*
3. Open `/steer` from that feature; amplify it; baseline and steered outputs differ visibly with the same seed. *(US-3)*
4. Open `/metrics`; CE-recovered and dead-feature % shown. *(US-6)*
5. Click **Random feature** once. *(US-5)*
6. Fallback: **Load saved example** works with the server offline. *(FR-43)*

---

## 16. Testing and quality

### 16.1 ML

| Test | What it proves |
|---|---|
| TopK produces exactly `k` non-zero activations per row | Sparsity is structural |
| Decoder columns have unit norm after an optimiser step | Constraint is enforced |
| Normalise then de-normalise returns the original within tolerance | Inference and steering use a valid transform |
| Hook captures shape `[batch, seq, 768]` at layer 8 | Hook is attached to the right place |
| **Identity SAE gives CE-recovered ≈ 1.0; zero SAE gives ≈ 0.0** | The CE-recovered metric is wired correctly |
| Same seed gives identical generation text | Steering comparison is fair |
| `build_index` output validates against `schema.sql` | Contract holds |

### 16.2 Server

Unit tests for each router; contract tests asserting responses match the Pydantic models and the committed `openapi.json`; a test that mock and real services return the **same shapes**; error-path tests (unknown feature, over-length text, bad strength).

### 16.3 Web

Type check (`tsc`); component tests for `TokenHeatmap` and `SteeringCompare`; a **Playwright smoke test** that walks the demo script in mock mode; manual keyboard and contrast check.

### 16.4 Manual before the demo

Run `make demo-check`; run the demo script three times; test with the network disabled; confirm the recording and screenshots are current.

---

## 17. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Features are poor or uninterpretable | Medium | High | GPT-2 small has published reference SAEs, so a bad result points at our code. Compare against a pretrained SAE if needed |
| Dictionary fills with dead features | Medium | High | AuxK from the start; log dead % every 100 steps; stop early if it climbs |
| Throughput too low to reach target data | High | Medium | Pipeline gate in §11.3; use WSL2 for `torch.compile`; reduce the activation budget and report the true number |
| Training unstable | Medium | Medium | Specified init, unit-norm decoder and gradient projection, normalised inputs — all together |
| App track blocked by ML | Medium | High | Frozen contract, mock mode, handoff H2 with a junk-quality real database |
| Contract churn | Medium | High | Additive-only policy (§14.4); both leads must approve |
| Real database too large to share | Medium | Low | Not in git; reduce examples per feature if needed; share by release asset with a checksum |
| Mock data mistaken for real in the demo | Low | High | SAMPLE DATA banner, `is_synthetic`, `make demo-check` |
| Steering effect is weak or incoherent | Medium | Medium | Curate a clear example early; keep saved examples as a safety net; describe steering as a test |
| Overclaiming in slides or UI | Medium | High | Claims policy §4.2 reviewed in every PR that touches copy |
| 8 GB VRAM exceeded | Low | Medium | GPT-2 small plus SAE needs roughly 3 GB; keep batches modest |
| Scope creep into larger models | Medium | High | FR-78 is **W**; only reconsider after H3 |

---

## 18. Decisions log and open questions

### 18.1 Decisions

| ID | Decision | Reason |
|---|---|---|
| D-1 | TopK SAE with AuxK, not an L1 penalty | Exact sparsity, no coefficient tuning, no shrinkage |
| D-2 | GPT-2 small, layer 8, 16× expansion, k = 32 | Small enough for a laptop; published reference results exist |
| D-3 | FineWeb-Edu, streamed | Reliable streaming loader; verify on day one |
| D-4 | TensorBoard for tracking | Works offline, no account needed |
| D-5 | Single read-only SQLite file as the data boundary | Simple, portable, no database server |
| D-6 | Mock mode is the default | App track needs no GPU |
| D-7 | Additive-only contract changes within a major version | Protects the app track from ML iteration |
| D-8 | Types generated from OpenAPI | Single source of truth, no drift |
| D-9 | Steering is described as a test, never a fix | Honest, defensible claims |

### 18.2 Open questions

| ID | Question | Owner | Needed by |
|---|---|---|---|
| Q-1 | What is the hackathon or submission deadline? Adjust §15.2 accordingly | Team | Day 1 |
| Q-2 | Who fills the App Lead, App Dev and Demo/QA roles? | Team | Day 1 |
| Q-3 | Where will large artefacts (database, checkpoint) be shared? | ML Lead | Week 2 |
| Q-4 | Is a usable gender-pronoun feature present? Decide whether the bias example is in the demo | ML Lead | Week 4 |
| Q-5 | Is a pretrained fallback SAE available and licensed for our use? | ML Lead | Week 1 |

---

## 19. Glossary and references

### 19.1 Glossary

| Term | Meaning |
|---|---|
| **Activation** | The vector of numbers (768 in GPT-2 small) the model holds for one token at one layer |
| **Residual stream** | The shared channel that layers read from and write to; our hook point |
| **Hook** | A PyTorch callback that captures or modifies a layer's output during a forward pass |
| **Frozen** | Weights never change; no gradients flow into the model |
| **Superposition** | Storing more concepts than dimensions by overlapping them |
| **Polysemantic / monosemantic** | One unit meaning several things / meaning one thing |
| **SAE** | Sparse autoencoder: expands activations into a large dictionary but uses only a few entries at a time |
| **Dictionary / feature** | The set of learned directions / one entry in it |
| **TopK** | Keep only the k largest activations, zero the rest |
| **AuxK** | Auxiliary loss that revives dead features |
| **Dead feature** | A feature that no longer fires and receives no gradient |
| **FVU** | Fraction of variance unexplained by the reconstruction |
| **L0** | Number of active features per token |
| **CE-recovered** | Share of the gap between zero-ablation and the original cross-entropy that the reconstruction recovers |
| **Logit attribution** | Projecting a feature's decoder direction through the unembedding to see which tokens it promotes |
| **Steering / clamping** | Changing a feature's value during generation to test its causal effect |
| **Shuffle buffer** | In-memory pool that decorrelates activation batches |
| **Mock mode** | Server mode that serves fixtures and deterministic fakes, with no GPU |

### 19.2 References

1. Elhage et al., *Toy Models of Superposition*, Anthropic, 2022.
2. Bricken et al., *Towards Monosemanticity*, Anthropic, 2023.
3. Templeton et al., *Scaling Monosemanticity*, Anthropic, 2024.
4. Gao et al., *Scaling and Evaluating Sparse Autoencoders*, OpenAI, 2024 (TopK and AuxK).
5. Lieberum et al., *Gemma Scope*, Google DeepMind, 2024.
6. Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2), OpenAI, 2019.

---

*NeuronLens PRD v1.0 — change decisions in §18; change the contract only by the rules in §14.4.*
