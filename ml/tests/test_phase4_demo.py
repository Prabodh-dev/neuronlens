import importlib.util
import json
import os
import sqlite3

import pytest

from tests.test_steering_checks import FakeRuntime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
spec = importlib.util.spec_from_file_location(
    "phase4_demo", os.path.join(ROOT, "ml", "scripts", "phase4_demo.py"))
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


@pytest.fixture()
def db(tmp_path):
    path = str(tmp_path / "t.db")
    con = sqlite3.connect(path)
    con.executescript(open(os.path.join(ROOT, "contracts", "schema.sql")).read())
    con.execute("INSERT INTO meta VALUES ('run_id','test-run')")
    for fid, label, dead, mx in [(7, "hon- – honing, honk", 0, 12.0), (13, "upside – upside down", 0, 9.0)]:
        con.execute("INSERT INTO features (feature_id, label, label_status, density, "
                    "max_activation, is_dead) VALUES (?,?,?,?,?,?)",
                    (fid, label, "named", 0.01, mx, dead))
    con.commit()
    con.close()
    return path


def test_build_examples_fields_and_read_only(db):
    before = open(db, "rb").read()
    con = demo.open_readonly(db)
    msgs = []
    out = demo.build_examples(FakeRuntime(baseline_len=20), con, max_new_tokens=20, log=msgs.append)
    assert [(e["feature_id"], e["mode"], e["strength"]) for e in out] == [
        (7, "amplify", 3.0), (13, "amplify", 3.0), (7, "ablate", 0.0)]
    e = out[0]
    for key in ("prompt", "seed", "feature_id", "mode", "strength", "baseline", "steered",
                "first_divergence_index", "label", "max_activation", "tokens_changed",
                "percent_changed_raw_note", "sweep", "differs"):
        assert key in e
    assert e["differs"] and e["first_divergence_index"] == 5 and e["label_matches_expected"]
    assert [r["strength"] for r in e["sweep"]["results"]] == list(demo.sc.DEFAULT_SWEEP_STRENGTHS)
    assert all("degenerate" in r and "reason" in r for r in e["sweep"]["results"])
    assert out[2]["sweep"] is None and msgs == []        # ablate has no sweep
    json.dumps(out)                                       # serialisable
    with pytest.raises(sqlite3.OperationalError):
        con.execute("UPDATE features SET label='x'")
    con.close()
    assert open(db, "rb").read() == before


def test_skips_missing_dead_and_warns_on_label_mismatch(db):
    con = sqlite3.connect(db)
    con.execute("UPDATE features SET is_dead=1 WHERE feature_id=13")
    con.execute("UPDATE features SET label='something else' WHERE feature_id=7")
    con.commit()
    con.close()
    msgs = []
    out = demo.build_examples(FakeRuntime(), demo.open_readonly(db), max_new_tokens=20,
                              sweep=False, log=msgs.append)
    assert [e["feature_id"] for e in out] == [7, 7]
    assert all(not e["label_matches_expected"] for e in out)
    assert any("skip feature 13" in m for m in msgs) and any("WARNING feature 7" in m for m in msgs)
    assert all(e["sweep"] is None for e in out)


def test_summary_is_readable(db):
    out = demo.build_examples(FakeRuntime(), demo.open_readonly(db), max_new_tokens=20)
    text = demo.summary(out)
    assert "feature 7 (hon- – honing, honk) amplify x3" in text and "sweep" in text
    assert "first divergence at new token 5" in text


def test_main_requires_real_mode_env(monkeypatch):
    monkeypatch.delenv("NEURONLENS_DB", raising=False)
    monkeypatch.delenv("NEURONLENS_CKPT", raising=False)
    with pytest.raises(SystemExit, match="NEURONLENS_DB"):
        demo.main([])
