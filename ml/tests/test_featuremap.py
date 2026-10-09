import importlib.util
import json
import os
import sys
import types

import numpy as np
import pytest

from neuronlens import featuremap as fm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def decoder(n=80, d=16, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n, d))
    return W / np.linalg.norm(W, axis=1, keepdims=True)           # unit-norm rows


@pytest.fixture()
def no_umap(monkeypatch):
    monkeypatch.setitem(sys.modules, "umap", None)                 # import umap -> ImportError


def fake_umap(monkeypatch, seen):
    class UMAP:
        def __init__(self, **kw):
            seen.update(kw)

        def fit_transform(self, z):
            seen["input_shape"] = z.shape
            return z[:, :2] * 3.0 + 1.0

    mod = types.ModuleType("umap")
    mod.UMAP, mod.__version__ = UMAP, "0.0-test"
    monkeypatch.setitem(sys.modules, "umap", mod)


def test_pca_only_when_umap_is_missing(no_umap):
    doc = fm.build_map(decoder(), "run-x", seed=42)
    assert doc["method"] == "pca2d" and doc["params"]["pca_dims"] == 2
    assert doc["params"]["seed_used"] is False
    with pytest.raises(RuntimeError, match="pip install umap-learn"):
        fm.build_map(decoder(), "run-x", method="umap")


def test_schema_shape_range_and_rounding(no_umap):
    W = decoder()
    doc = fm.build_map(W, "run-x", seed=7)
    assert list(doc) == ["run_id", "method", "seed", "params", "points"]
    assert doc["run_id"] == "run-x" and doc["seed"] == 7
    pts = np.array(doc["points"])
    assert pts.shape == (80, 2)                                     # index = feature id
    assert pts.min() >= -1.0 and pts.max() <= 1.0
    assert max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) == pytest.approx(2.0, abs=0.002)   # longer axis spans the range
    assert all(round(v, 3) == v for v in pts.ravel())
    p = doc["params"]
    assert p["range"] == [-1.0, 1.0] and p["aspect_ratio_preserved"] is True and p["decimals"] == 3
    assert p["n_features"] == 80 and p["decoder_norm_min"] == p["decoder_norm_max"] == 1.0
    json.dumps(doc)


def test_aspect_ratio_is_kept():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 2.0]])
    s = fm.scale_to_range(xy)
    assert s[:, 0].min() == -1.0 and s[:, 0].max() == 1.0
    assert np.ptp(s[:, 1]) == pytest.approx(0.4)                    # 2/10 of the long axis
    flat = fm.scale_to_range(np.zeros((3, 2)))
    assert np.all(flat == 0.0)                                      # no division by zero


def test_deterministic_for_a_fixed_seed_and_fixed_axis_signs(no_umap):
    W = decoder()
    a, b = fm.build_map(W, "r", 42), fm.build_map(W, "r", 42)
    assert a == b
    z = fm.pca_scores(W, 5)
    for i in range(z.shape[1]):
        assert abs(z[:, i]).max() > 0
    assert fm.pca_scores(W, 5).tolist() == z.tolist()


def test_nearby_directions_land_near_each_other(no_umap):
    W = decoder(60, 16)
    W[1] = W[0] + 1e-3 * np.random.default_rng(3).normal(size=16)     # a near copy of feature 0
    W[1] /= np.linalg.norm(W[1])
    pts = np.array(fm.build_map(W, "r")["points"])
    d01 = np.linalg.norm(pts[0] - pts[1])
    others = np.linalg.norm(pts[2:] - pts[0], axis=1)
    assert d01 < np.median(others)


def test_umap_path_uses_pca50_cosine_and_the_seed(monkeypatch):
    seen = {}
    fake_umap(monkeypatch, seen)
    doc = fm.build_map(decoder(120, 64), "run-x", seed=123)
    assert doc["method"] == "pca50+umap2d" and doc["params"]["umap_learn"] == "0.0-test"
    assert seen["random_state"] == 123 and seen["n_components"] == 2
    assert seen["metric"] == "cosine" and seen["n_neighbors"] == 15 and seen["min_dist"] == 0.1
    assert seen["input_shape"] == (120, 50) and doc["params"]["pca_dims"] == 50
    assert np.array(doc["points"]).shape == (120, 2)
    # pca_dims never exceeds what the data allows
    seen.clear()
    assert fm.build_map(decoder(30, 8), "r")["params"]["pca_dims"] == 8


def test_script_needs_a_checkpoint(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("build_map", os.path.join(ROOT, "ml", "scripts", "build_map.py"))
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.delenv("NEURONLENS_CKPT", raising=False)
    with pytest.raises(SystemExit, match="NEURONLENS_CKPT"):
        script.main([])
    with pytest.raises(SystemExit, match="checkpoint not found"):
        script.main(["--ckpt", str(tmp_path / "missing.pt")])
