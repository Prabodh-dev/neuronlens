"""W7: a 2D map of the SAE features, from their decoder directions.

The decoder directions (one unit-norm row per feature) are reduced to 50 dimensions with PCA
and then to 2D with UMAP when umap-learn is installed; otherwise PCA to 2D alone is used, and
the file says which. The result is a picture of decoder-direction similarity: nearby points have
similar decoder directions, but distance is a rough guide only, and a different seed or method
gives a different layout. Coordinates are shifted and scaled together (aspect ratio kept) so
the longer axis spans RANGE, then rounded to 3 decimals.
"""
import json

import numpy as np

PCA_DIMS = 50
RANGE = (-1.0, 1.0)
UMAP_PARAMS = {"n_neighbors": 15, "min_dist": 0.1, "metric": "cosine"}
NDIGITS = 3


def pca_scores(X, dims):
    """Centre X (n, d) and project on its top `dims` principal axes. Axis signs are fixed
    (the largest-magnitude loading of each axis is positive) so the output is repeatable."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    dims = max(1, min(dims, Xc.shape[0] - 1, Xc.shape[1]))
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Vt = Vt[:dims].copy()
    for i in range(Vt.shape[0]):
        if Vt[i, np.argmax(np.abs(Vt[i]))] < 0:
            Vt[i] = -Vt[i]
    return Xc @ Vt.T


def scale_to_range(xy, lo=RANGE[0], hi=RANGE[1]):
    """Shift both axes to the centre of their bounding box and scale them by one common
    factor so the longer axis spans exactly [lo, hi]. The aspect ratio is kept."""
    xy = np.asarray(xy, dtype=np.float64)
    mn, mx = xy.min(axis=0), xy.max(axis=0)
    centre, half = (mn + mx) / 2.0, float(((mx - mn) / 2.0).max())
    if half == 0.0:
        half = 1.0
    mid, radius = (lo + hi) / 2.0, (hi - lo) / 2.0
    return (xy - centre) / half * radius + mid


def reduce_2d(W, seed=42, method="auto"):
    """W: decoder directions (n_features, d_model). Returns (xy float64 (n, 2) before scaling,
    method name, params). method: "auto" (UMAP if installed, else PCA), "umap" or "pca"."""
    if method not in ("auto", "umap", "pca"):
        raise ValueError("method must be auto, umap or pca")
    umap_mod = None
    if method in ("auto", "umap"):
        try:
            import umap as umap_mod
        except ImportError:
            if method == "umap":
                raise RuntimeError("umap-learn is not installed; install it with "
                                   "`pip install umap-learn` or use --method pca")
    if umap_mod is None:
        xy = pca_scores(W, 2)
        return xy, "pca2d", {"pca_dims": 2, "seed_used": False}
    z = pca_scores(W, PCA_DIMS)
    xy = umap_mod.UMAP(n_components=2, random_state=seed, **UMAP_PARAMS).fit_transform(z)
    return (np.asarray(xy, dtype=np.float64), "pca50+umap2d",
            {"pca_dims": int(z.shape[1]), **UMAP_PARAMS,
             "umap_learn": getattr(umap_mod, "__version__", "unknown")})


def build_map(W, run_id, seed=42, method="auto"):
    """The JSON document: points[i] = [x, y] for feature i."""
    W = np.asarray(W, dtype=np.float64)
    xy, name, params = reduce_2d(W, seed, method)
    pts = np.round(scale_to_range(xy), NDIGITS) + 0.0   # + 0.0 turns -0.0 into 0.0
    norms = np.linalg.norm(W, axis=1)
    params = {**params, "range": list(RANGE), "aspect_ratio_preserved": True,
              "decimals": NDIGITS, "n_features": int(W.shape[0]),
              "decoder_norm_min": round(float(norms.min()), 4),
              "decoder_norm_max": round(float(norms.max()), 4)}
    return {"run_id": run_id, "method": name, "seed": int(seed), "params": params,
            "points": pts.tolist()}


def write_json(doc, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))
