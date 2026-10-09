# Phase 4 data files: co-fire counts (W4) and the 2D map (W7)

For Mirdul. Two JSON files built by Prabodh on the GPU laptop. Neither goes into the database, and neither is in git. This file lists where they are, the exact fields, how they were built and what they can and cannot say. The endpoint shape and the file location below are proposals, not decisions.

The values in the examples are example values, not measured. No real file has been built yet; the sizes are estimates until it is.

## Files

| File | Built by | Default output | Estimated size |
| --- | --- | --- | --- |
| co-fire counts | `ml/scripts/build_cofire.py` | `data/cofire.json` | up to about 6 MB (12,288 lists of at most 10 partners, compact JSON). A unit test checks that a full file stays under 8 MB |
| 2D map | `ml/scripts/build_map.py` | `data/map2d.json` | about 0.2 to 0.3 MB (12,288 points) |

Both live in the git-ignored `data/` folder next to `neuronlens.db` and the checkpoint, and are shared separately in the same way. `.gitignore` also ignores `cofire*.json` and `map2d*.json` anywhere in the repository, so a different `--out` path is not committed by accident. Both files record the `run_id` they were built for; compare it with the database's `meta.run_id` before using a file.

## `cofire.json`

```json
{
  "run_id": "gpt2-l8-k32",
  "tokens_scanned": 500000,
  "seed": 42,
  "k": 32,
  "min_count": 5,
  "partners": {
    "0": [
      {"feature_id": 117, "co_count": 4210, "jaccard": 0.3187},
      {"feature_id": 9034, "co_count": 951, "jaccard": 0.2214}
    ],
    "1": [],
    "...": "one key for every feature id from 0 to d_sae - 1"
  }
}
```

(The numbers above are example values.)

| Field | Type | Notes |
| --- | --- | --- |
| `run_id` | str | the run the checkpoint belongs to (`runs.run_id`) |
| `tokens_scanned` | int | number of scored tokens counted, exactly as scanned (position 0 of each sequence is not counted) |
| `seed` | int | seed of the token stream |
| `k` | int | the SAE's TopK value (32) |
| `min_count` | int | a pair needs at least this many shared tokens to be listed (default 5) |
| `partners` | object | key is the feature id as a string; every feature has a key |
| `partners[id]` | list | up to 10 entries, best at the top; an empty list for a feature that never fired in the sample, and for dead features when the script was given the database |
| `feature_id` | int | the partner |
| `co_count` | int | tokens on which both features were active |
| `jaccard` | float | `co_count / (count_a + count_b - co_count)`, rounded to 4 decimals |

Order: by Jaccard, highest at the top. Ties are ordered by larger `co_count`, then smaller `feature_id`. The file does not repeat each feature's own firing count; `co_count / jaccard` cannot recover it exactly, so if the UI needs "fires on N tokens" it should come from the database (`features.density` is a fraction over the final scan, a different sample). If that count is wanted in the file, say so and it can be added.

A feature counts as active on a token when its SAE activation is above 0. With TopK 32 that is up to 32 features per token.

## `map2d.json`

```json
{
  "run_id": "gpt2-l8-k32",
  "method": "pca50+umap2d",
  "seed": 42,
  "params": {
    "pca_dims": 50, "n_neighbors": 15, "min_dist": 0.1, "metric": "cosine",
    "umap_learn": "0.5.x", "range": [-1.0, 1.0], "aspect_ratio_preserved": true,
    "decimals": 3, "n_features": 12288, "decoder_norm_min": 1.0, "decoder_norm_max": 1.0
  },
  "points": [[-0.412, 0.087], [0.233, -0.671], "..."]
}
```

(The numbers above are example values.)

| Field | Type | Notes |
| --- | --- | --- |
| `run_id` | str | as above |
| `method` | str | `"pca50+umap2d"` when umap-learn was installed, otherwise `"pca2d"` (PCA straight to 2D) |
| `seed` | int | the UMAP random state; recorded but not used by `pca2d` (`params.seed_used` is `false` then) |
| `params` | object | the settings used. For `pca2d`: `pca_dims` is 2 and there are no UMAP settings. Always present: `range`, `aspect_ratio_preserved`, `decimals`, `n_features`, `decoder_norm_min`, `decoder_norm_max` (the decoder rows are expected to have norm 1) |
| `points` | list of [x, y] | the list index is the feature id; 3 decimals |

Coordinates are shifted to the centre of their bounding box and scaled by one common factor, so the longer axis spans exactly -1 to 1 and the aspect ratio is kept. The shorter axis spans less.

## How they were built

Run from `ml/` on the GPU laptop with the checkpoint:

```
export NEURONLENS_CKPT=../data/gpt2-l8-k32.pt
PYTHONPATH=. python scripts/build_cofire.py --tokens 500000 --seed 42 --db ../data/neuronlens.db
PYTHONPATH=. python scripts/build_map.py --seed 42
```

Expected run time, estimated from the sizes involved and not measured: the co-fire script about 2 to 5 minutes including model load (most of it is the count matrix updates and reading the data); the map script about 1 to 2 minutes with UMAP (a fresh UMAP install also compiles code on its initial run, which adds time) and about 10 seconds with PCA alone. Both would be slower on a laptop without a GPU.

- **Co-fire.** The same token pipeline as the final scan: FineWeb-Edu sequences of 1,024 tokens, GPT-2 small, the output of `transformer.h[8]`, position 0 of each sequence dropped (the code is shared with `interpret.scan`, not copied). For every token, each pair of active features is counted. The 12,288 x 12,288 count matrix is about 600 MB; it is kept in host memory and filled in row blocks computed on the GPU, so the GPU holds one block at a time. `--db` is opened read-only and only used to blank dead features.
- **Map.** The 12,288 decoder directions (unit norm) are reduced to 50 dimensions with PCA, then to 2D with UMAP (cosine distance, 15 neighbours, minimum distance 0.1). Without umap-learn the script uses PCA to 2D alone and says so in `method`. No model or data is loaded.

## What they can and cannot say

**Co-fire.** It only says that two features fire on the same tokens in this sample, which is `tokens_scanned` tokens (500,000 by default) from one seed. It is not a statement about meaning, and a partner is not a cause. Features that fire often can share many tokens with many others, which the Jaccard score only partly offsets. The 5-token minimum removes pairs seen only a few times but does not remove chance overlap in a sample this size. With the default seed 42 the tokens are the same stream the SAE was trained on; another seed gives text it did not train on. A different sample would give different counts. Features with fewer than 10 qualifying partners have shorter lists, and some have none.

**Map.** It is a picture of decoder-direction similarity: two points near each other have decoder directions that point roughly the same way in the 768-dimensional space. Distance is a rough guide only. UMAP generally keeps neighbourhoods better than long distances, so cluster sizes and the gaps between clusters should not be read as measurements. A different seed or a different method gives a different layout, and the `pca2d` fallback looks different from the UMAP layout. Dead features are plotted too, at the position of their untrained decoder direction; mark them or leave them out using `features.is_dead`. The map says nothing about which features fire on the same tokens; that is the co-fire file.

Wording for the UI: say "fire on the same tokens" for co-fire, and "decoder directions that point the same way" for the map. Avoid wording that suggests a causal link or a shared meaning.

## Proposed endpoint shape (not decided)

Nothing here was added to `contracts/` or `server/`.

- Where the files live: the same `data/` folder, with two optional environment variables next to `NEURONLENS_DB` and `NEURONLENS_CKPT`, for example `NEURONLENS_COFIRE` and `NEURONLENS_MAP`. If a file is missing or its `run_id` differs from the database's, the endpoint could answer that the data is not available rather than fail the whole server.
- Co-fire: `GET /features/{feature_id}/cofire` could return `{"feature_id", "tokens_scanned", "partners": [{"feature_id", "label", "co_count", "jaccard"}]}`, with the labels joined from the database as the other endpoints do. An empty list is a valid answer.
- Map: `GET /map` could return `{"method", "seed", "points": [{"feature_id", "x", "y", "is_dead", "label"}]}` built from the file and the database, or pass the compact `points` list through as it is (about 0.3 MB).
