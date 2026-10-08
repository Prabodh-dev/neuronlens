# NeuronLens

NeuronLens is a web app that shows which concepts a language model's learned features were active on in any text, and lets users test those concepts by steering (turning a feature up or down and comparing outputs). Model: GPT-2 small, layer 8, TopK sparse autoencoder (k=32, 12,288 features).

## Folder layout
- `contracts/`: `schema.sql`, the future `openapi.json`, and `fixtures/` (synthetic `sample.db` and its generator)
- `ml/`: ML code (SAE training, interpretation, runtime), added by Prabodh
- `server/`: FastAPI backend (Mirdul)
- `web/`: Next.js frontend (Anand)
- `demo/`: demo script, screenshots, backup recording (Anand)
- `docs/`: handbook, phases, PRD, app build guide, feature cards
- `data/`: local-only data and checkpoints (git-ignored)

## Run in mock mode
Mock mode reads the synthetic fixture database; no model or checkpoint is needed.

```
cp .env.example .env        # NEURONLENS_MODE=mock
python contracts/fixtures/make_fixture.py   # regenerate sample.db (optional)
```

Then start the server and web app following `server/README.md` and `web/README.md` once those exist.

## Claims policy
- Say "shows which concepts were active".
- Steering is a test, not a fix.
- Never write "proves", "detects", "fixes", "explains why", "first" or "novel".

## Honest limits
- GPT-2 small, one layer only.
- The strongest features are word-piece features.
- Some features are ambiguous.
- Steering too hard breaks text.
