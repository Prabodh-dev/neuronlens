# Phase 4 data files: co-fire counts (W4) and the 2D map (W7)

For Mirdul. Two JSON files built by Prabodh on the GPU laptop. Neither goes into the database, and neither is in git. This file lists where they are, the exact fields, how they were built and what they can and cannot say. The endpoint shape and the file location below are proposals, not decisions.

Both files have been built once on the GPU laptop; what those runs showed is in "What the real runs showed" below. The JSON snippets in the field descriptions use example values, not values from those runs. The real examples are in their own section.

## Files

| File | Built by | Default output | Size in the real run |
| --- | --- | --- | --- |
| co-fire counts | `ml/scripts/build_cofire.py` | `data/cofire.json` | 6.5 MB (3,000,000 tokens scanned). A unit test checks that a full file stays under 8 MB |
| 2D map | `ml/scripts/build_map.py` | `data/map2d.json` | 0.18 MB (12,288 points). Not for display, see below |

Both live in the git-ignored `data/` folder next to `neuronlens.db` and the checkpoint, and are shared separately in the same way. `.gitignore` also ignores `cofire*.json` and `map2d*.json` anywhere in the repository, so a different `--out` path is not committed by accident. Both files record the `run_id` they were built for; compare it with the database's `meta.run_id` before using a file.

## `cofire.json`

```json
{
  "run_id": "gpt2-l8-k32",
  "tokens_scanned": 3000000,
  "seed": 999,
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

Run from `ml/` on the GPU laptop with the checkpoint. The real runs used these settings:

```
export NEURONLENS_CKPT=../data/gpt2-l8-k32.pt
PYTHONPATH=. python scripts/build_cofire.py --tokens 3000000 --seed 999
PYTHONPATH=. python scripts/build_map.py --seed 42
```

The script defaults are `--tokens 500000 --seed 42`. The real co-fire run used seed 999, which is text the SAE was not trained on (seed 42 is the training stream), and 3,000,000 tokens.

Measured on the GPU laptop, one run each: the co-fire run took about 5.6 minutes and wrote 6.5 MB; the map run (PCA to 50 dimensions, then UMAP to 2D, seed 42) took about 78 seconds of CPU and wrote 0.18 MB. These are single runs, not benchmarks.

- **Co-fire.** The same token pipeline as the final scan: FineWeb-Edu sequences of 1,024 tokens, GPT-2 small, the output of `transformer.h[8]`, position 0 of each sequence dropped (the code is shared with `interpret.scan`, not copied). For every token, each pair of active features is counted. The 12,288 x 12,288 count matrix is about 600 MB; it is kept in host memory and filled in row blocks computed on the GPU, so the GPU holds one block at a time. `--db` is opened read-only and only used to blank dead features.
- **Map.** The 12,288 decoder directions (unit norm) are reduced to 50 dimensions with PCA, then to 2D with UMAP (cosine distance, 15 neighbours, minimum distance 0.1). Without umap-learn the script uses PCA to 2D alone and says so in `method`. No model or data is loaded.

## What the real runs showed

### Co-fire (3,000,000 tokens, seed 999)

12,250 of the 12,288 features have partners and 38 have an empty list.

Three features from this one run, with a selection of their partners written as `label (co_count, jaccard)`. The partners are chosen from each list and are not shown in file order, and the file holds up to 10 per feature:

| Feature | Partners |
| --- | --- |
| 1863 "supreme – supreme court" | "supreme – court or command" (158, 0.17); "central – government or bank" (67, 0.067); "courts – legal system" (90, 0.034) |
| 7 "hon – honing, honking" | "hon- – honorable and honors" (82, 0.059); "hon – honors, honorable" (93, 0.046); "ton- – herbal and health" (76, 0.051) |
| 13 "upside – upside down" | "reverse – reverse engineering" (51, 0.044); "vertical – or horizontal" (22, 0.025); "top – of a physical object" (73, 0.023); "click – web link instructions" (29, 0.038) |

How to read this: these are three features from one run. The evidence is thinner for rare features: a pair that shares 22 tokens out of 3,000,000 is a small count, and the Jaccard values are low (0.023 to 0.17). Small samples are weak: a 500,000-token sample gave counts of only 5 to 20 for the same features, close to the minimum of 5 that lets a pair be listed. The UI must show the sample size (`tokens_scanned`) and the counts (`co_count`), and must use the words "fired on the same tokens", for example "fired on the same tokens 158 times in a sample of 3,000,000 tokens". It should not present a partner as a cause or as a feature with the same meaning.

### Map (PCA to 50, then UMAP to 2D, seed 42)

**Checked, not for display.** On three features (13, 7 and 1863), the five nearest points in the 2D map did not match the feature. The nearest points were: for 13, "ary", "multi", "di", "brother", "little"; for 7, "que", "tom", "inter", "cav", "fast"; for 1863, "pre", "primary", "b", "ipel", "and". So distance in this map should not be shown to users as meaning. If a map is wanted later, it should be built from the co-fire data. The script and its tests stay in the repository, and the file can still be built, but it is not proposed for the app.

## What they can and cannot say

**Co-fire.** It only says that two features fired on the same tokens in this sample, which is `tokens_scanned` tokens (500,000 by default, 3,000,000 in the real run) from one seed. It is not a statement about meaning, and a partner is not a cause. Features that fire often can share many tokens with many others, which the Jaccard score only partly offsets. The 5-token minimum removes pairs seen only a few times but does not remove chance overlap in a sample this size. With the default seed 42 the tokens are the same stream the SAE was trained on; the real run used seed 999, text it did not train on. A different sample would give different counts. Features with fewer than 10 qualifying partners have shorter lists, and some have none.

**Map (not for display, see "Checked, not for display").** It is a picture of decoder-direction similarity: two points near each other have decoder directions that point roughly the same way in the 768-dimensional space. Distance is a rough guide only. UMAP generally keeps neighbourhoods better than long distances, so cluster sizes and the gaps between clusters should not be read as measurements. A different seed or a different method gives a different layout, and the `pca2d` fallback looks different from the UMAP layout. Dead features are plotted too, at the position of their untrained decoder direction; mark them or leave them out using `features.is_dead`. The map says nothing about which features fire on the same tokens; that is the co-fire file.

Wording for the UI: for co-fire, say "fired on the same tokens" and show the sample size and the counts. Avoid wording that suggests a causal link or a shared meaning. The map is not for display; if its layout is ever described, the only accurate description is "decoder directions that point roughly the same way", and the check above shows that is not enough to show users.

## Proposed endpoint shape (not decided)

Nothing here was added to `contracts/` or `server/`.

- Where the files live: the same `data/` folder, with an optional environment variable next to `NEURONLENS_DB` and `NEURONLENS_CKPT`, for example `NEURONLENS_COFIRE` (a second one for the map would only be needed if the map is ever used). If a file is missing or its `run_id` differs from the database's, the endpoint could answer that the data is not available rather than fail the whole server.
- Co-fire: `GET /features/{feature_id}/cofire` could return `{"feature_id", "tokens_scanned", "partners": [{"feature_id", "label", "co_count", "jaccard"}]}`, with the labels joined from the database as the other endpoints do. An empty list is a valid answer.
- Map: no endpoint is proposed. The file exists, but the check above says distance in it should not be shown to users as meaning.
