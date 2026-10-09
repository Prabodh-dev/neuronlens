import importlib.util
import os
import sqlite3

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
spec = importlib.util.spec_from_file_location(
    "check_labels", os.path.join(ROOT, "ml", "scripts", "check_labels.py"))
cl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cl)

GOOD = [(1, "alzheimer – disease", "named", 0), (2, "hon- – honing, honk", "named", 0),
        (3, "semicolon – between clauses", "ambiguous", 0)]
BAD = [(10, "x" * 33, "named", 0),                       # too long
       (11, 'the "word"', "named", 0),                   # quote marks
       (12, "post (prefix)", "named", 0),                # parentheses
       (13, "ends with a period.", "named", 0),          # ending punctuation
       (14, "Capital Letter", "ambiguous", 0),           # not lowercase
       (15, "it’s – contraction", "named", 0)]           # curly quote


def make_db(tmp_path, rows):
    path = str(tmp_path / "t.db")
    con = sqlite3.connect(path)
    con.executescript(open(os.path.join(ROOT, "contracts", "schema.sql")).read())
    for fid, label, status, dead in rows:
        con.execute("INSERT INTO features (feature_id, label, label_status, density, "
                    "max_activation, is_dead) VALUES (?,?,?,?,?,?)",
                    (fid, label, status, 0.01, 1.0, dead))
    con.commit()
    con.close()
    return path


def test_problems_rules():
    assert cl.problems("alzheimer – disease") == []
    assert cl.problems("hon-") == []  # a word piece may end in a hyphen
    assert cl.problems("x" * 32) == [] and cl.problems("x" * 33) == ["too long"]
    assert cl.problems("a 'b'") == ["quote mark"]
    assert cl.problems("a (b)") == ["parenthesis"]
    assert cl.problems("a b,") == ["ends in punctuation"]
    assert cl.problems("A b") == ["not lowercase"]


def test_clean_database_exits_zero_and_reports(tmp_path, capsys):
    rows = GOOD + [(4, "alzheimer – disease", "named", 0),            # repeated label
                   (20, None, "unlabelled", 0), (21, None, "unlabelled", 1),
                   (22, None, "unlabelled", 0)]
    db = make_db(tmp_path, rows)
    before = open(db, "rb").read()
    assert cl.main(["--db", db]) == 0
    out = capsys.readouterr().out
    assert "named: 3" in out and "ambiguous: 1" in out and "unlabelled: 3" in out
    assert "2 x alzheimer – disease" in out
    assert "unlabelled features that are not dead: 2" in out and "20, 22" in out
    assert "21" not in out.split("not dead")[1]  # the dead one is left out
    assert "labels breaking the standard: 0" in out
    assert open(db, "rb").read() == before


def test_breaking_labels_exit_one(tmp_path, capsys):
    db = make_db(tmp_path, GOOD + BAD)
    assert cl.main(["--db", db, "--max-list", "1"]) == 1
    out = capsys.readouterr().out
    assert "too long: 1" in out and "quote mark: 2" in out and "parenthesis: 1" in out
    assert "ends in punctuation: 1" in out and "not lowercase: 1" in out
    assert "labels breaking the standard: 6" in out
    assert "... and 1 more" in out  # quote mark list capped at 1


def test_top_repeated_limited_to_fifteen(tmp_path):
    rows = [(i, f"label {i % 20}", "named", 0) for i in range(100)]
    con = cl.open_readonly(make_db(tmp_path, rows))
    assert len(cl.analyse(con)["repeated"]) == 15
    con.close()


def test_read_only_and_missing_db(tmp_path):
    db = make_db(tmp_path, GOOD)
    con = cl.open_readonly(db)
    with pytest.raises(sqlite3.OperationalError):
        con.execute("UPDATE features SET label='x'")
    con.close()
    with pytest.raises(SystemExit):
        cl.main(["--db", str(tmp_path / "missing.db")])
    assert not (tmp_path / "missing.db").exists()
