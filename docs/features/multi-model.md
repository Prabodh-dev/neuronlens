# MultiModel: a second small model (experiment)
User sees: A second model offered next to GPT-2 in the app, labelled experimental, when its server on port 8001 is running. With only the GPT-2 server up the app is unchanged.
Endpoint / data used: the existing endpoints, unchanged. Own files: `data/neuronlens-neo.db`, a checkpoint under `~/runs/`, `demo/autolab_fillers_<model>.json`.
Recorded example (feature ids, prompt, strength): none recorded. Examples recorded for GPT-2 do not carry over; feature numbers mean different things in each model. Auto-Lab runs live on it with three goal buttons.
Honest limit shown on screen: the "experimental" tag and the line "Smaller training run than GPT-2, so expect weaker features." Its metrics are its own measurements.
Status: included in the app as experimental (trained, evaluated, indexed and labelled; shown through the model chooser)

## Measured results (this model, on its own)

EleutherAI/gpt-neo-125M, layer 8, 12,288 features, k=32, trained on 37.5 million tokens.

- On 256 held-out sequences it keeps 97.46% of next-word ability (ce_recovered 0.9746). FVU is 0.0248. FVU is not comparable across models: it depends on how much the activations vary at the hooked layer, which differs from model to model.
- 61 of the 12,288 features are dead (0.50%).
- 12,223 of the 12,227 live features have a label from Gemini: 8,383 named and 3,840 ambiguous. The labels are automatic and were not checked by hand.
- Auto-Lab was checked on six goals on this model and three are offered: music, court law and weather rain. It runs live; no runs are recorded for it.
- The filler list (`demo/autolab_fillers_gpt-neo-125m.json`) is cut at features that come back for more than 20 of 200 words (10%); the gap at the cutoff is small (22 words against 20).

These are this model's own measurements. Nothing here is compared with another model.

## What it is

NeuronLens runs on GPT-2 small (layer 8, 12,288 features, k=32). This adds a second model with the same recipe so that the pipeline can be tried on something else. The candidate is GPT-Neo-125M; the fallback is DistilGPT-2. The choice of model is a command-line option (`--model`, default `gpt2`), so GPT-2 runs as before.

From the transformers source (modeling_gpt2.py, modeling_gpt_neo.py): GPT-2, DistilGPT-2 and GPT-Neo all keep the blocks at `model.transformer.h[i]` and the unembedding at `model.lm_head`, and the config exposes `hidden_size`. The GPT-Neo class default is a larger model (hidden size 2048, 24 layers); the 768 width, 12 layers and the GPT-2 vocabulary of the 125M model come from its published config, which could not be read here. `compat_check.py` checks them on the real model.

All layout-specific code is in `ml/neuronlens/modelutil.py` (block lookup, backbone, width, unembedding). Models are loaded with `AutoModelForCausalLM` and `AutoTokenizer`.

## Files and names

| Thing | GPT-2 (as before) | Second model |
| --- | --- | --- |
| Config | `ml/configs/gpt2-small.yaml` | `ml/configs/gpt-neo-125m.yaml` |
| Run folder | `~/runs/gpt2-l8-k32/` | `~/runs/gpt-neo-125m-l8-k32/` |
| Database | `data/neuronlens.db` | `data/neuronlens-neo.db` |
| Server port | 8000 | 8001 |
| Auto-Lab fillers | `demo/autolab_fillers.json` | `demo/autolab_fillers_gpt-neo-125m.json` |

The checkpoint remembers its model (`cfg.model_name`), so the index build, the evaluation and the server use the right one without being told. The server also reads `NEURONLENS_MODEL_NAME` if it is set. `/health` reports the database's model name, layer and sizes, and in real mode the server refuses a checkpoint whose model name, layer or width differs from the database, so the name it shows is the model that answers.

## Run it end to end

Run each line from the repo root with the venv active. The GPT-2 labels in `ml/labels.json` belong to GPT-2 features and must not be used for this model.

1. Check the model (tokenizer, shape, position-0 effect):

```
PYTHONPATH=ml python ml/scripts/compat_check.py --model EleutherAI/gpt-neo-125M --layer 8
```

2. Measure throughput and get a token budget for a time limit:

```
PYTHONPATH=ml python ml/scripts/run_training.py ml/configs/gpt-neo-125m.yaml --measure --minutes 60
```

3. Train with the budget it printed:

```
PYTHONPATH=ml python ml/scripts/run_training.py ml/configs/gpt-neo-125m.yaml --tokens 100000000
```

4. Evaluate (FVU, CE recovered; no database needed):

```
PYTHONPATH=ml python ml/scripts/run_evals.py --ckpt ~/runs/gpt-neo-125m-l8-k32/final.pt --out-json ~/runs/gpt-neo-125m-l8-k32/eval.json
```

5. Build the database:

```
PYTHONPATH=ml python ml/scripts/make_index.py --ckpt ~/runs/gpt-neo-125m-l8-k32/final.pt --out data/neuronlens-neo.db --schema contracts/schema.sql --n-seqs 3000 --notes experimental
```

6. Optional: write the 256-sequence evaluation into that database:

```
PYTHONPATH=ml python ml/scripts/run_evals.py --ckpt ~/runs/gpt-neo-125m-l8-k32/final.pt --db data/neuronlens-neo.db
```

7. Labels (with your own Gemini API key): `ml/scripts/auto_label.py` takes any database through `--db`, and names the model and layer from the database's meta rows in its prompt.
8. Start the server on port 8001:

```
NEURONLENS_MODE=real NEURONLENS_DB=data/neuronlens-neo.db NEURONLENS_CKPT=$HOME/runs/gpt-neo-125m-l8-k32/final.pt NEURONLENS_MODEL_NAME=EleutherAI/gpt-neo-125M uvicorn app.main:app --app-dir server --port 8001
```

9. Auto-Lab fillers for this model, against that server:

```
python server/scripts/autolab_check.py --url http://127.0.0.1:8001 --find-fillers demo/autolab_fillers_gpt-neo-125m.json
```

`--fillers-file` selects a filler list when running the checks; its default is the GPT-2 list.

10. Web app against it (a second terminal): `cd web`, then `NEXT_PUBLIC_API_BASE=http://localhost:8001/api/v1 npm run dev -- --port 3001`.

## Gates

Each step passes or the model is dropped from the app; none of them is a performance target.

1. `compat_check.py` prints COMPATIBLE: the token ids match GPT-2's on 50 sentences, the width is 768 and the layer exists, and position 0 stands out from later positions (ratio at least 3, a working threshold that is not calibrated). If the last one fails, dropping position 0 would discard a normal token and the recipe needs a decision before training.
2. Training finishes and `eval.json` exists. Its FVU and CE recovered are shown as measured.
3. The database builds, and `/health` on port 8001 reports the model name, layer 8 and sizes 768 and 12,288.
4. Labels: Gemini labelled the features. They are automatic and were not checked by hand; a spot-check of 50 (`auto_label.py sheet`) is still open.
5. The filler file exists, and Auto-Lab goals were tried on this model before being shown: six were checked and three are offered.

## Claims rules

- The page and any text say "experimental". The database's `runs.notes` carries it.
- Metrics are this model's own measured values. No sentence says it is better or worse than GPT-2; the two runs differ in model, layer behaviour and data statistics.
- Steering is a test, not a fix; results are described as tuned.
- Co-fire counts (`data/cofire.json`) were built for the GPT-2 run and are not available for this one until they are built for it.

## Limits

- One extra model, one layer (8), one seed, one recipe.
- Labels are automatic and not checked by hand, so some names will be wrong or too broad.
- The ✓ thresholds of Auto-Lab were tuned on GPT-2 runs; the page says so for this model.
- The "about 30 seconds" Auto-Lab estimate for this model is a round figure, not a timed measurement.
- Co-fire counts and recorded steering examples exist for GPT-2 only.
