import importlib.util
import os
import shutil
import sqlite3

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
spec = importlib.util.spec_from_file_location(
    "run_evals", os.path.join(ROOT, "ml", "scripts", "run_evals.py"))
run_evals = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_evals)

METRICS = {"fvu": 0.1637, "l0": 32.0, "dead_pct": 2.6, "ce_orig": 3.2924,
           "ce_zero": 14.6982, "ce_recon": 3.4558, "ce_recovered": 0.9857}


def _db(tmp_path):
    path = str(tmp_path / "t.db")
    shutil.copyfile(os.path.join(ROOT, "contracts", "fixtures", "sample.db"), path)
    return path


def _metrics(path):
    con = sqlite3.connect(path)
    rows = dict(con.execute("SELECT name, value FROM metrics").fetchall())
    con.close()
    return rows


def test_update_in_place_keeps_dead_pct(tmp_path):
    path = _db(tmp_path)
    before = _metrics(path)
    written = run_evals.update_metrics(path, METRICS, update_dead=False)
    after = _metrics(path)
    assert "dead_pct" not in written
    assert after["dead_pct"] == before["dead_pct"]
    assert after["fvu"] == 0.1637
    assert after["ce_original"] == 3.2924
    assert after["ce_zero_ablation"] == 14.6982
    assert after["ce_reconstructed"] == 3.4558
    assert after["ce_recovered"] == 0.9857
    assert len(after) == len(before)  # updated, not duplicated


def test_update_dead_flag_overwrites(tmp_path):
    path = _db(tmp_path)
    run_evals.update_metrics(path, METRICS, update_dead=True)
    assert _metrics(path)["dead_pct"] == 2.6


def test_inserts_when_row_missing(tmp_path):
    path = _db(tmp_path)
    con = sqlite3.connect(path)
    con.execute("DELETE FROM metrics WHERE name='l0'")
    con.commit()
    con.close()
    run_evals.update_metrics(path, METRICS)
    assert _metrics(path)["l0"] == 32.0
