# ML track

Sparse autoencoder (SAE) on GPT-2 small: training, evaluation, interpretation, and the runtime used by the server.

## What this does

- Hook: residual stream after block 8 of GPT-2 small (d_model 768). The first token is dropped (outlier position).
- SAE: TopK (k=32), 12,288 features (16x), AuxK loss for dead features, unit-norm decoder columns.
- Data: FineWeb-Edu sample/10BT, read from a local parquet shard. Activations are normalised so the mean L2 norm is sqrt(d_model).
- Interpretation: top examples per feature, logit attribution (promoted/suppressed tokens), density histogram, labels from labels.json.
- Steering: amplify adds strength x max_activation x decoder direction at every position except the first; ablate subtracts the feature's own activation. Baseline and steered runs use the same seed.

## Results (final run, gpt2-l8-k32)

Final evaluation: 256 sequences, seed 999.

| Metric                                      | Value                                            |
| ------------------------------------------- | ------------------------------------------------ |
| FVU                                         | 0.1626                                           |
| L0                                          | 32                                               |
| Dead features                               | 0.42% (small eval), about 0.008% (3M-token scan) |
| CE original / zero-ablation / reconstructed | 3.3089 / 14.7674 / 3.4714                        |
| CE recovered                                | 0.9858                                           |

The comparison with the published GPT-2 layer-8 SAE is in the section below.

## Run

Needs a CUDA GPU (developed on an RTX 4070 Laptop, 8 GB, in WSL2).

    cd ml
    pip install -r requirements.txt
    PYTHONPATH=. python scripts/run_training.py configs/gpt2-small.yaml
    PYTHONPATH=. python scripts/make_index.py --ckpt <ckpt> --out ../data/neuronlens.db --schema ../contracts/schema.sql --n-seqs <n> --labels labels.json --data-dir <fineweb-dir>
    PYTHONPATH=. python scripts/browse.py --top 10
    PYTHONPATH=. python scripts/steer_examples.py
    pytest tests

## Comparison with the published GPT-2 SAE

Compares our SAE with OpenAI's TopK SAE for GPT-2 small, `resid_post_mlp` layer 8, `v5_32k` (32,768 latents, k=32). Both are scored by the same `evals.evaluate` on the same token sequences (positions 1 and up), the same model and the same hook (output of `transformer.h[8]`). The published SAE is layer-normed inside itself; the script undoes our normalisation and applies its own (see the header of `scripts/compare_published.py` for what was verified and what was not). The capacity differs (12,288 vs 32,768 latents), so this is not a fair fight on equal size.

    cd ml
    PYTHONPATH=. python scripts/compare_published.py --ours ~/runs/gpt2-l8-k32/final.pt --data-dir ~/data/fineweb/sample/10BT --n-seqs 256 --seed 999 --out-json ~/runs/gpt2-l8-k32/compare.json

Results: Result (256 sequences, seed 999): ours FVU 0.1626, CE recovered 0.9858; OpenAI v5_32k L8 FVU 0.4307, CE recovered 0.9454. Not equal capacity or data: 12,288 vs 32,768 latents, different training data. The published SAE's registry metadata lists variance explained 0.74 on its own data, so the gap partly reflects data shift; the adapter was not cross-checked against OpenAI's own code.

## Phase 4: steering checks
`neuronlens/steering_checks.py` holds the functions behind the Phase 4 steering endpoints. Steering is a test of what a feature does, not a fix; every function runs the baseline and the steered text with the same seed, and the steering maths is unchanged.

- X1 `steering_check`: baseline text, steered text, whether they differ, and the target feature's activation on each text (re-encoded without steering; it reads the feature on the finished text and does not judge the output).
- X3 `seed_matched_compare`: token-by-token comparison; reports `first_divergence_index`, the tokens before it, the tokens from it onward, and `tokens_changed` counted from there. The raw percent is also returned, with a note that it overstates the direct effect because later tokens drift.
- X2 `strength_sweep`: one feature at several strengths (default 0, 0.25, 0.5, 0.75, 1, 1.5, 2), with a `degenerate` flag and `reason` per strength. The thresholds are named constants in the module and are not calibrated on real outputs yet.
- X4 `prompt_only_baseline`: what the model writes from the prompt alone.
- W5 `multi_steering_check`: 2 or 3 features steered together (amplify only, directions summed in one hook), compared with the baseline; one activation reading per feature on the finished text. `scripts/phase4_demo.py --combine` runs it from the command line.

Exact inputs, outputs, error cases and example JSON are in `docs/phase4-ml-api.md`. To produce demo examples on the GPU machine (reads the database read-only, writes `demo/phase4_examples.json`):

    cd ml
    export NEURONLENS_DB=../data/neuronlens.db NEURONLENS_CKPT=../data/gpt2-l8-k32.pt
    PYTHONPATH=. python scripts/phase4_demo.py

The prompts in `scripts/phase4_demo.py` are starting points and have not been tried on the real model.

To try other features without editing code, use explore mode. It runs each `--feature` on each `--prompt`, prints one line per strength (start of the text, the degenerate flag, the index where it differs from the baseline) and writes `demo/explore.json`, never `demo/phase4_examples.json`:

    PYTHONPATH=. python scripts/phase4_demo.py --feature 7 --feature 13 --prompt "He turned the picture" --strengths 0,0.5,0.75,1,1.5,2 [--mode ablate] [--explore-out PATH]

## Phase 4 data files
Two scripts build data files for the app (they run on the GPU machine, write to the git-ignored `data/` folder, and do not touch the database). Details, fields and limits are in `docs/phase4-data-files.md`.

- `scripts/build_cofire.py`: how often two features fired on the same tokens in a sample (default 500,000 tokens); top 10 partners per feature by Jaccard. A run with `--tokens 3000000 --seed 999` (unseen text) took about 5.6 minutes and gave a 6.5 MB file; 12,250 of 12,288 features have partners. Small samples are weak: 500,000 tokens gave counts of only 5 to 20 for the features checked.
- `scripts/build_map.py`: a 2D map of the decoder directions (PCA, then UMAP if umap-learn is installed). Checked, not for display: on three features the nearest points in the map did not match the feature, so distance in it should not be shown to users as meaning (details in `docs/phase4-data-files.md`). The script and its tests stay.

## Layout

- neuronlens/activations.py: model loading, activation hook and streaming buffer
- neuronlens/sae.py: TopK SAE
- neuronlens/train.py: training loop
- neuronlens/evals.py: FVU, L0, dead %, cross-entropy evals
- neuronlens/interpret.py, build_index.py: feature examples, logit attribution, database build
- neuronlens/textdecode.py: byte-aware token decoding (characters split across tokens)
- neuronlens/steering.py: steering hooks and generation
- neuronlens/runtime.py: Runtime used by the server (load, count_tokens, inspect, steer)
- scripts/: command-line entry points
- configs/: training configs

## Notes

- Checkpoints and the real database are not in git (data/ is ignored); they are shared by link.
- Prior work: see the Disclosures section of the root README.md for what existed before the hackathon and what was built during it.
- Honest limits: GPT-2 small and one layer; the strongest features are word-piece features; some features are ambiguous; steering too hard breaks text.
