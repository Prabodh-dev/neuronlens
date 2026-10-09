import os
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
from neuronlens import build_index  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
F, N_TOK = 4, 1000


def _stub_heavy_parts(monkeypatch):
    """Keep the real build() database-writing path; replace only the GPU/model steps."""
    ck = {"layer": 8, "d_model": 768, "d_sae": F, "k": 32, "cfg": {"run_id": "test-run"},
          "metrics": {"fvu": 0.16, "l0": 32.0, "dead_pct": 1.0, "ce_orig": 3.3, "ce_zero": 14.7,
                      "ce_recon": 3.4, "ce_recovered": 0.98}}
    monkeypatch.setattr(build_index, "load_sae", lambda path, device: (SimpleNamespace(d_sae=F), ck))
    monkeypatch.setattr(build_index, "load_model", lambda device: (object(), object()))
    monkeypatch.setattr(build_index, "Collector", lambda model, layer: object())
    monkeypatch.setattr(build_index, "token_sequences", lambda tok, d, seed: iter(range(10)))
    fire = np.array([500, 100, 0, 10])
    S = {"n_tok": N_TOK, "fire": fire, "asum": fire * 1.5, "top_v": np.full((32, F), 2.0)}
    monkeypatch.setattr(build_index, "scan", lambda *a, **k: S)
    ex = {f: [(0, ["a", "b"], [0.1, 2.0], 1, 2.0)] for f in (0, 1, 3)}
    monkeypatch.setattr(build_index, "build_examples", lambda *a, **k: ex)
    logits = {f: {"promoted": [(" the", 1.0)], "suppressed": [(" a", -1.0)]} for f in range(F)}
    monkeypatch.setattr(build_index, "logit_attribution", lambda *a, **k: logits)
    monkeypatch.setattr(build_index, "_training_curve", lambda run_dir: [])


def test_build_writes_schema_version_1_0(tmp_path, monkeypatch):
    _stub_heavy_parts(monkeypatch)
    out = str(tmp_path / "built.db")
    build_index.build(str(tmp_path / "x.pt"), out, os.path.join(ROOT, "contracts", "schema.sql"),
                      n_seqs=2, labels_path=str(tmp_path / "none.json"), device="cpu")
    con = sqlite3.connect(out)
    meta = dict(con.execute("SELECT key, value FROM meta"))
    assert meta["schema_version"] == "1.0"
    assert meta["run_id"] == "test-run" and meta["is_synthetic"] == "false"
    assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert con.execute("PRAGMA foreign_key_check").fetchall() == []
    con.close()


def test_constant_matches_the_contract_fixture():
    con = sqlite3.connect(os.path.join(ROOT, "contracts", "fixtures", "sample.db"))
    fixture = con.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
    con.close()
    assert build_index.SCHEMA_VERSION == fixture == "1.0"
