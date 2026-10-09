import importlib.util
import json
import os
import sqlite3

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
spec = importlib.util.spec_from_file_location(
    "review_labels", os.path.join(ROOT, "ml", "scripts", "review_labels.py"))
rl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl)


@pytest.fixture()
def db(tmp_path):
    path = str(tmp_path / "tiny.db")
    con = sqlite3.connect(path)
    con.executescript(open(os.path.join(ROOT, "contracts", "schema.sql")).read())
    for fid in (6, 7):
        con.execute("INSERT INTO features (feature_id, density, max_activation) VALUES (?,?,?)",
                    (fid, 0.01, 9.0))
    ctx = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"]  # 13 tokens
    ex = [  # (feature, rank, tokens, acts, peak_index, max)
        (6, 1, ["x ", "multi", "idis", "ciplinary", " work"], [0, 0.1, 5.5, 0.2, 0], 2, 5.5),
        (6, 2, ["low ", "multi", "idis"], [0, 0.1, 2.0], 2, 2.0),
        (7, 1, ["They", "", "’", "re"], [0.1, 3.0, 0.5, 0.0], 1, 3.0),  # peak on an empty piece
        (7, 2, ctx * 3, [0.0] * 20 + [4.0] + [0.0] * 18, 0, 4.0),       # peak far from stored hint
    ]
    for f, r, t, a, pk, mx in ex:
        con.execute("INSERT INTO feature_examples (feature_id, rank, tokens_json, activations_json,"
                    " peak_index, max_activation) VALUES (?,?,?,?,?,?)",
                    (f, r, json.dumps(t, ensure_ascii=False), json.dumps(a), pk, mx))
    for i, t in enumerate([" the", " a", " of", " in", " to", " and", " is", " for", " ZZ"], 1):
        con.execute("INSERT INTO feature_logits VALUES (6,'promoted',?,?,?)", (i, t, 10.0 - i))
    con.execute("INSERT INTO feature_logits VALUES (6,'suppressed',1,' no',1.0)")
    con.commit()
    con.close()
    return path


def test_peak_is_bracketed_and_activation_printed(db):
    con = rl.open_readonly(db)
    lines = rl.build_report(con, {6: {"label": "multi- prefix", "type": "token", "status": "named"}},
                            6, 6, 6)
    text = "\n".join(lines)
    assert "Feature 6 | proposed label: multi- prefix | type: token | status: named" in text
    assert "1. [5.50] x multi[[idis]]ciplinary work" in text
    assert "2. [2.00] low multi[[idis]]" in text  # ordered by activation
    assert "promoted: \" the\" (9.00)" in text and text.count('" ZZ"') == 0  # top 8 only
    assert " no" not in text.split("promoted:")[1]  # suppressed not shown
    con.close()


def test_empty_peak_piece_spans_to_completed_character(db):
    con = rl.open_readonly(db)
    text = "\n".join(rl.build_report(con, {}, 7, 7, 6))
    assert "They[[’]]re" in text
    assert "(none in labels file)" in text


def test_context_is_twelve_tokens_each_side():
    toks = [f"t{i} " for i in range(40)]
    acts = [0.0] * 40
    acts[20] = 1.0
    text, v = rl.mark_example(toks, acts)
    assert text.startswith("t8 ") and text.endswith("t32 ") and "[[t20 ]]" in text and v == 1.0


def test_database_opened_read_only(db):
    con = rl.open_readonly(db)
    with pytest.raises(sqlite3.OperationalError):
        con.execute("UPDATE features SET label='x'")
    with pytest.raises(sqlite3.OperationalError):
        con.execute("INSERT INTO meta VALUES ('k','v')")
    con.close()
    before = open(db, "rb").read()
    assert open(db, "rb").read() == before
    with pytest.raises(FileNotFoundError):
        rl.open_readonly(db + ".missing")  # read-only open never creates a file
    assert not os.path.exists(db + ".missing")


def test_labels_formats(tmp_path):
    f = tmp_path
    (f / "d.json").write_text(json.dumps({"6": "plain", "7": {"label": "obj", "status": "ambiguous",
                                                               "type": "word"}}))
    (f / "l.json").write_text(json.dumps([{"feature_id": 8, "proposed_label": "x", "kind": "t"}]))
    (f / "j.jsonl").write_text('{"id": 9, "label": "y"}\n{"id": 10, "label": "z", "status": "named"}\n')
    (f / "w.json").write_text(json.dumps({"labels": {"11": "wrapped"}}))
    (f / "one.json").write_text(json.dumps({"6": {"label": "lone"}}))
    d = rl.load_labels(f / "d.json")
    assert d[6]["label"] == "plain" and d[7] == {"label": "obj", "type": "word", "status": "ambiguous"}
    assert rl.load_labels(f / "l.json")[8] == {"label": "x", "type": "t", "status": None}
    j = rl.load_labels(f / "j.jsonl")
    assert j[9]["label"] == "y" and j[10]["status"] == "named"
    assert rl.load_labels(f / "w.json")[11]["label"] == "wrapped"
    assert rl.load_labels(f / "one.json")[6]["label"] == "lone"


def test_cli_end_to_end_and_out_file(db, tmp_path, capsys):
    labels = tmp_path / "l.json"
    labels.write_text(json.dumps({"6": "multi- prefix"}))
    out = tmp_path / "sheet.txt"
    before = open(db, "rb").read()
    rl.main(["--db", db, "--labels", str(labels), "--start", "6", "--end", "7",
             "--examples", "1", "--out", str(out)])
    printed = capsys.readouterr().out
    assert printed == out.read_text(encoding="utf-8")
    assert "Feature 6" in printed and "Feature 7" in printed and "\n\nFeature 7" in printed
    assert open(db, "rb").read() == before  # database untouched


def test_ids_file_csv_with_header_and_plain_list(tmp_path):
    csv_path = tmp_path / "spot.csv"
    csv_path.write_text(
        'feature_id,auto_label,top_examples\r\n7,"x, y","a || b\nmore"\r\n6,z,c\r\n7,dup,d\r\n',
        encoding="utf-8")
    assert rl.load_ids(csv_path) == [7, 6]
    plain = tmp_path / "ids.txt"
    plain.write_text("6\n\n 19 \n7\n")
    assert rl.load_ids(plain) == [6, 19, 7]
    bad = tmp_path / "bad.csv"
    bad.write_text("name,label\na,b\n")
    with pytest.raises(ValueError, match="no feature id column"):
        rl.load_ids(bad)
    bad.write_text("feature_id\nabc\n")
    with pytest.raises(ValueError, match="not a feature id"):
        rl.load_ids(bad)


def test_ids_file_uses_database_labels_and_ignores_range(db, tmp_path, capsys):
    con = sqlite3.connect(db)
    con.execute("UPDATE features SET label='multi- prefix', label_status='named', "
                "notes='auto-labelled  by\nmodel' WHERE feature_id=6")
    con.commit()
    con.close()
    ids = tmp_path / "ids.csv"
    ids.write_text("feature_id\n7\n6\n99\n")
    before = open(db, "rb").read()
    rl.main(["--db", db, "--ids-file", str(ids), "--start", "1", "--end", "3", "--examples", "1"])
    out = capsys.readouterr().out
    heads = [ln for ln in out.splitlines() if ln.startswith("Feature ")]
    assert heads == [
        "Feature 7 | proposed label: (empty) | status: unlabelled",
        "Feature 6 | proposed label: multi- prefix | status: named | notes: auto-labelled by model",
        "Feature 99 | proposed label: (none in database)"]
    assert "x multi[[idis]]ciplinary work" in out and "(feature not in database)" in out
    assert open(db, "rb").read() == before


def test_labels_file_still_wins_when_given(db, tmp_path, capsys):
    labels = tmp_path / "l.json"
    labels.write_text(json.dumps({"6": "from file"}))
    rl.main(["--db", db, "--labels", str(labels), "--start", "6", "--end", "6", "--examples", "1"])
    assert "Feature 6 | proposed label: from file" in capsys.readouterr().out
