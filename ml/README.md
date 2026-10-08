# ML track

Sparse autoencoder (SAE) on GPT-2 small: training, evaluation, interpretation, and the runtime used by the server.

## What this does

- Hook: residual stream after block 8 of GPT-2 small (d_model 768). The first token is dropped (outlier position).
- SAE: TopK (k=32), 12,288 features (16x), AuxK loss for dead features, unit-norm decoder columns.
- Data: FineWeb-Edu sample/10BT, read from a local parquet shard. Activations are normalised so the mean L2 norm is sqrt(d_model).
- Interpretation: top examples per feature, logit attribution (promoted/suppressed tokens), density histogram, labels from labels.json.
- Steering: amplify adds strength x max_activation x decoder direction at every position except the first; ablate subtracts the feature's own activation. Baseline and steered runs use the same seed.

## Results (final run, gpt2-l8-k32)

| Metric                                      | Value                                           |
| ------------------------------------------- | ----------------------------------------------- |
| FVU                                         | 0.1637                                          |
| L0                                          | 32                                              |
| Dead features                               | 2.6% (small eval), about 0.008% (3M-token scan) |
| CE original / zero-ablation / reconstructed | 3.2924 / 14.6982 / 3.4558                       |
| CE recovered                                | 0.9857                                          |

The FVU comparison against the published GPT-2 layer-8 SAE is still to do.

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

## Layout

- neuronlens/activations.py: model loading, activation hook and streaming buffer
- neuronlens/sae.py: TopK SAE
- neuronlens/train.py: training loop
- neuronlens/evals.py: FVU, L0, dead %, cross-entropy evals
- neuronlens/interpret.py, build_index.py: feature examples, logit attribution, database build
- neuronlens/steering.py: steering hooks and generation
- neuronlens/runtime.py: Runtime used by the server (load, count_tokens, inspect, steer)
- scripts/: command-line entry points
- configs/: training configs

## Notes

- Checkpoints and the real database are not in git (data/ is ignored); they are shared by link.
- This code was written before the hackathon and brought into this repo; the repo, API, web app and verification features are built during the event.
- Honest limits: GPT-2 small and one layer; the strongest features are word-piece features; some features are ambiguous; steering too hard breaks text.
