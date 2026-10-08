import importlib.util
import json
import os
import shutil
import sqlite3

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
spec = importlib.util.spec_from_file_location(
    "repair", os.path.join(ROOT, "ml", "scripts", "repair_example_tokens.py"))
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def _db(tmp_path):
    path = str(tmp_path / "t.db")
    shutil.copyfile(os.path.join(ROOT, "contracts", "fixtures", "sample.db"), path)
    return sqlite3.connect(path)


def _rebuilt(con, damage=False):
    ex = {}
    for f, rank, toks, acts, pk in con.execute(
            "SELECT feature_id, rank, tokens_json, activations_json, peak_index "
            "FROM feature_examples ORDER BY feature_id, rank").fetchall():
        t = [x.replace("\ufffd", "'") for x in json.loads(toks)]
        t[0] = "FIXED"
        a = json.loads(acts)
        if damage:
            a[0] += 5.0
        ex.setdefault(f, []).append((rank, t, a, pk, 0.0))
    return ex


def test_only_tokens_json_changes_and_counts(tmp_path):
    con = _db(tmp_path)
    con.execute("UPDATE feature_examples SET tokens_json = replace(tokens_json, 'a', char(65533)) "
                "WHERE example_id % 2 = 0")
    con.commit()
    bad, total = repair.count_fffd(con)
    assert bad > 0
    before = repair.fingerprint(con)
    con.execute("BEGIN")
    updated, mism, missing = repair.apply_rebuilt(con, _rebuilt(con))
    con.commit()
    assert updated == total and mism == missing == 0
    assert repair.fingerprint(con) == before  # nothing but tokens_json touched
    assert repair.count_fffd(con)[0] == 0
    assert all(json.loads(t)[0] == "FIXED"
               for (t,) in con.execute("SELECT tokens_json FROM feature_examples"))


def test_mismatch_rolls_back(tmp_path):
    con = _db(tmp_path)
    old = con.execute("SELECT tokens_json FROM feature_examples ORDER BY example_id").fetchall()
    con.execute("BEGIN")
    with pytest.raises(RuntimeError, match="nothing written"):
        repair.apply_rebuilt(con, _rebuilt(con, damage=True))
    con.rollback()
    assert con.execute("SELECT tokens_json FROM feature_examples ORDER BY example_id").fetchall() == old
