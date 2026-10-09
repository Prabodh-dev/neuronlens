import importlib.util
import json
import os
import sqlite3

import pytest

from tests.test_steering_checks import FakeRuntime


class BigRuntime(FakeRuntime):
    """FakeRuntime with the real number of features, for the default examples (ids up to 3587)."""
    d_sae = 12288

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
    for fid, label, dead, mx in [(7, "hon- – honing, honk", 0, 12.0), (13, "upside – upside down", 0, 9.0),
                                 (3446, "music – music, bands", 0, 15.0),
                                 (1863, "supreme – supreme court", 0, 19.7),
                                 (3587, "alzheimer – disease", 0, 11.0)]:
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
    out = demo.build_examples(BigRuntime(baseline_len=20), con, max_new_tokens=20, log=msgs.append)
    assert [(e["feature_id"], e["prompt"], e["mode"], e["strength"]) for e in out] == [
        (3446, "The band walked on stage and", "amplify", 0.5),
        (3446, "She sat down at the piano and", "amplify", 0.5),
        (1863, "She opened the letter and read", "amplify", 0.5),
        (13, "The acrobat hung", "amplify", 0.5),
        (1863, "The lawyers argued before the", "ablate", 0.0),
        (3587, "My grandmother was diagnosed with", "amplify", 1.0),
        (7, "She looked out of the window and saw", "amplify", 1.0)]
    assert max(e["strength"] for e in out) <= 3.0         # nothing above the API cap
    assert all(e["seed"] == 42 and e["max_new_tokens"] == 20 for e in out)
    e = out[0]
    for key in ("prompt", "seed", "feature_id", "mode", "strength", "baseline", "steered",
                "first_divergence_index", "label", "max_activation", "tokens_changed",
                "percent_changed_raw_note", "sweep", "differs"):
        assert key in e
    assert e["differs"] and e["first_divergence_index"] == 5 and e["label_matches_expected"]
    assert [r["strength"] for r in e["sweep"]["results"]] == list(demo.sc.DEFAULT_SWEEP_STRENGTHS)
    assert all("degenerate" in r and "reason" in r for r in e["sweep"]["results"])
    assert [x["sweep"] is not None for x in out] == [True, True, True, True, False, True, True]
    assert msgs == []                                     # every label matched, nothing skipped
    json.dumps(out)                                       # serialisable
    with pytest.raises(sqlite3.OperationalError):
        con.execute("UPDATE features SET label='x'")
    con.close()
    assert open(db, "rb").read() == before


def test_notes_are_on_the_limit_examples_in_neutral_wording(db):
    out = demo.build_examples(BigRuntime(baseline_len=20), demo.open_readonly(db),
                              max_new_tokens=20, sweep=False)
    assert [("note" in x) for x in out] == [False, False, True, False, True, True, True]
    assert "did not appear in the text" in out[2]["note"] and "The man walked into the room and" in out[2]["note"]
    assert "about the same" in out[4]["note"] and "18.57 to 18.57" in out[4]["note"]
    assert "re-encodes the finished text" in out[4]["note"] and "Supreme" in out[4]["note"]
    assert "repeated" in out[5]["note"] and "little effect" in out[5]["note"] and "at 0.5" in out[5]["note"]
    assert "word fragments" in out[6]["note"] and "rather than toward honking" in out[6]["note"]
    for x in out:
        if "note" in x:
            low = x["note"].lower()
            assert not any(w in low for w in ("proves", "detects", "fixes", "explains why", "novel"))


def test_skips_missing_dead_and_warns_on_label_mismatch(db):
    con = sqlite3.connect(db)
    con.execute("UPDATE features SET is_dead=1 WHERE feature_id=13")
    con.execute("UPDATE features SET label='something else' WHERE feature_id=7")
    con.commit()
    con.close()
    msgs = []
    out = demo.build_examples(BigRuntime(), demo.open_readonly(db), max_new_tokens=20,
                              sweep=False, log=msgs.append)
    assert [e["feature_id"] for e in out] == [3446, 3446, 1863, 1863, 3587, 7]   # 13 skipped
    assert [e["label_matches_expected"] for e in out] == [True, True, True, True, True, False]
    assert any("skip feature 13" in m for m in msgs) and any("WARNING feature 7" in m for m in msgs)
    assert all(e["sweep"] is None for e in out)


def test_summary_is_readable(db):
    out = demo.build_examples(BigRuntime(), demo.open_readonly(db), max_new_tokens=20)
    text = demo.summary(out)
    assert "feature 3446 (music – music, bands) amplify x0.5" in text and "sweep" in text
    assert "feature 1863 (supreme – supreme court) ablate" in text
    assert "feature 13 (upside – upside down) amplify x0.5" in text
    assert "feature 7 (hon- – honing, honk) amplify x1" in text
    assert "first divergence at new token 5" in text


def test_main_requires_real_mode_env(monkeypatch):
    monkeypatch.delenv("NEURONLENS_DB", raising=False)
    monkeypatch.delenv("NEURONLENS_CKPT", raising=False)
    with pytest.raises(SystemExit, match="NEURONLENS_DB"):
        demo.main([])


# ------------------------------------------------------------------ explore mode

def test_parse_strengths():
    assert demo.parse_strengths("0,0.5, 0.75,1,1.5,2") == [0.0, 0.5, 0.75, 1.0, 1.5, 2.0]
    assert demo.parse_strengths("2,2,1") == [2.0, 2.0, 1.0]       # used exactly as given
    for bad in ("", "a,b", "1,,x", "nan", "inf"):
        with pytest.raises(ValueError):
            demo.parse_strengths(bad)


def test_explore_amplify_uses_exactly_the_given_strengths(db):
    rt = FakeRuntime(baseline_len=20)
    lines = []
    runs = demo.explore(rt, demo.open_readonly(db), [7, 13, 7], ["P one ", "P two "],
                        [0.0, 0.75, 1.5], max_new_tokens=20, log=lines.append)
    assert [(r["feature_id"], r["prompt"]) for r in runs] == [
        (7, "P one "), (7, "P two "), (13, "P one "), (13, "P two ")]   # each feature x each prompt, deduped
    for r in runs:
        assert [x["strength"] for x in r["results"]] == [0.0, 0.75, 1.5]
        assert r["mode"] == "amplify" and r["label_status"] == "named"
        assert r["results"][0]["differs_from_baseline"] is False
        assert r["results"][0]["first_divergence_index"] is None
        assert r["results"][1]["first_divergence_index"] == 5
        assert set(r["results"][1]) == {"strength", "text", "continuation", "differs_from_baseline",
                                        "first_divergence_index", "degenerate", "reason", "metrics",
                                        *demo.ACTIVATION_KEYS}
    out = "\n".join(lines)
    assert "feature 7 | hon- – honing, honk | named | prompt: 'P one '" in out
    assert out.count("x0.75") == 4 and "div=5" in out and "div=-" in out
    assert sum(l.startswith("  baseline") for l in lines) == 4
    json.dumps(runs)


def test_explore_marks_degenerate_and_truncates_to_120_chars(db):
    long_new = [ord("z")] * 150
    rt = FakeRuntime(base_ids=[ord(c) for c in "abcdefghij" * 3], steered_ids=long_new)
    lines = []
    runs = demo.explore(rt, demo.open_readonly(db), [7], ["Hi "], [3.0], max_new_tokens=120,
                        log=lines.append)
    r = runs[0]["results"][0]
    assert r["degenerate"] is True and len(r["continuation"]) == 150
    row = next(l for l in lines if l.startswith("  x3"))
    assert "DEGENERATE: mostly repeated tokens" in row
    assert repr("z" * 120) in row and "z" * 121 not in row


def test_explore_ablate_runs_once_and_ignores_strengths(db):
    rt = FakeRuntime(baseline_len=20)
    lines = []
    runs = demo.explore(rt, demo.open_readonly(db), [7], ["Hi "], [1, 2, 3], mode="ablate",
                        max_new_tokens=20, log=lines.append)
    assert len(runs[0]["results"]) == 1 and runs[0]["results"][0]["strength"] == 0.0
    assert runs[0]["results"][0]["differs_from_baseline"] is True
    assert any(l.startswith("  ablate") for l in lines)
    assert [c[3].mode for c in rt.calls if c[3]] == ["ablate"]


def test_explore_skips_dead_and_missing_features(db):
    con = sqlite3.connect(db)
    con.execute("UPDATE features SET is_dead=1 WHERE feature_id=13")
    con.commit()
    con.close()
    lines = []
    runs = demo.explore(FakeRuntime(), demo.open_readonly(db), [13, 99, 7], ["Hi "], [1.0],
                        max_new_tokens=20, log=lines.append)
    assert [r["feature_id"] for r in runs] == [7]
    assert any("skip feature 13" in l and "dead feature" in l for l in lines)
    assert any("skip feature 99" in l and "not in the database" in l for l in lines)


def _env(monkeypatch, db, tmp_path):
    ckpt = tmp_path / "x.pt"
    ckpt.write_bytes(b"")
    monkeypatch.setenv("NEURONLENS_DB", db)
    monkeypatch.setenv("NEURONLENS_CKPT", str(ckpt))
    monkeypatch.setattr(demo, "load_runtime", lambda path: BigRuntime(baseline_len=20))


def test_main_explore_writes_only_the_explore_file(db, tmp_path, monkeypatch, capsys):
    _env(monkeypatch, db, tmp_path)
    out_dir = tmp_path / "demo"
    out_dir.mkdir()
    examples = out_dir / "phase4_examples.json"
    examples.write_text('{"keep": true}')
    explore_out = out_dir / "explore.json"
    before = open(db, "rb").read()
    demo.main(["--feature", "7", "--feature", "13", "--prompt", "A ", "--prompt", "B ",
               "--strengths", "0,0.5,0.75,1,1.5,2", "--max-new-tokens", "20",
               "--out", str(examples), "--explore-out", str(explore_out)])
    data = json.loads(explore_out.read_text(encoding="utf-8"))
    assert data["mode"] == "amplify" and data["strengths"] == [0, 0.5, 0.75, 1, 1.5, 2]
    assert data["run_id"] == "test-run" and len(data["runs"]) == 4
    assert examples.read_text() == '{"keep": true}'          # never touched
    assert open(db, "rb").read() == before                   # database read-only
    assert "feature 13 | upside – upside down | named" in capsys.readouterr().out


def test_explore_out_defaults_to_demo_explore_json(db, tmp_path, monkeypatch):
    _env(monkeypatch, db, tmp_path)
    monkeypatch.setattr(demo, "REPO_ROOT", tmp_path)       # default paths hang off REPO_ROOT
    demo.main(["--feature", "7", "--prompt", "A ", "--max-new-tokens", "20", "--strengths", "1"])
    assert (tmp_path / "demo" / "explore.json").is_file()
    assert not (tmp_path / "demo" / "phase4_examples.json").exists()


def test_main_explore_argument_errors(db, tmp_path, monkeypatch):
    _env(monkeypatch, db, tmp_path)
    for argv in (["--feature", "7"],                                     # no prompt
                 ["--prompt", "x"],                                      # no feature
                 ["--strengths", "1,2"],                                 # explore option alone
                 ["--mode", "ablate"],
                 ["--feature", "7", "--prompt", "x", "--strengths", "a,b"],
                 ["--feature", "7", "--prompt", "x", "--out", str(tmp_path / "same.json"),
                  "--explore-out", str(tmp_path / "same.json")]):       # would overwrite --out
        with pytest.raises(SystemExit) as e:
            demo.main(argv)
        assert e.value.code == 2
    # out-of-range strength is reported cleanly, nothing is written
    out = tmp_path / "e.json"
    with pytest.raises(SystemExit, match="strength"):
        demo.main(["--feature", "7", "--prompt", "x", "--strengths", "50", "--explore-out", str(out)])
    assert not out.exists()


def test_default_mode_unchanged_by_explore_options(db, tmp_path, monkeypatch):
    _env(monkeypatch, db, tmp_path)
    out = tmp_path / "phase4_examples.json"
    demo.main(["--out", str(out), "--max-new-tokens", "20"])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert [e["feature_id"] for e in data["examples"]] == [3446, 3446, 1863, 13, 1863, 3587, 7]
    assert not (tmp_path / "explore.json").exists()


# ------------------------------------------------------------ activation fields and lines

def _up(hi, lo):
    return lambda ids, fid: [hi if 65 <= i <= 90 else lo for i in ids]


def test_default_examples_store_activation_fields_and_summary_line(db):
    rt = BigRuntime(baseline_len=20, act_fn=_up(8.0, 1.0))
    out = demo.build_examples(rt, demo.open_readonly(db), max_new_tokens=20, sweep=False)
    for e in out:
        for k in demo.ACTIVATION_KEYS:
            assert k in e
    a = out[0]
    assert a["target_activation_baseline"] == 1.0 and a["target_activation_steered"] == 8.0
    assert a["activation_verdict"] == "higher"
    json.dumps(out)
    text = demo.summary(out)
    assert "activation: baseline 1.00 -> steered 8.00 (higher)" in text
    assert text.count("activation: baseline") == len(out)       # one line per example


def test_activation_line_format():
    d = {"target_activation_baseline": 0.5, "target_activation_steered": 12.3456,
         "activation_verdict": "lower"}
    assert demo.activation_line(d) == "activation: baseline 0.50 -> steered 12.35 (lower)"


def test_explore_prints_an_activation_line_per_strength_and_for_ablate(db):
    lines = []
    runs = demo.explore(FakeRuntime(baseline_len=20, act_fn=_up(8.0, 1.0)), demo.open_readonly(db),
                        [7], ["Hi "], [0.0, 0.5, 1.0], max_new_tokens=20, log=lines.append)
    act = [l for l in lines if "activation: baseline" in l]
    assert len(act) == 3 and all(l.startswith("    activation:") for l in act)
    assert "(about the same)" in act[0] and "(higher)" in act[1]       # strength 0 matches baseline
    assert [r["activation_verdict"] for r in runs[0]["results"]] == ["about the same", "higher", "higher"]
    ab = []
    demo.explore(FakeRuntime(baseline_len=20, act_fn=_up(8.0, 1.0)), demo.open_readonly(db), [7],
                 ["Hi "], [1.0], mode="ablate", max_new_tokens=20, log=ab.append)
    assert sum("activation: baseline" in l for l in ab) == 1


def test_main_explore_json_has_activation_fields(db, tmp_path, monkeypatch):
    _env(monkeypatch, db, tmp_path)
    out = tmp_path / "explore.json"
    demo.main(["--feature", "7", "--prompt", "A ", "--strengths", "0.5", "--max-new-tokens", "20",
               "--explore-out", str(out)])
    r = json.loads(out.read_text(encoding="utf-8"))["runs"][0]["results"][0]
    assert all(k in r for k in demo.ACTIVATION_KEYS)


# ------------------------------------------------------------------------- --combine

def test_explore_combined_prints_text_flag_and_one_activation_line_per_feature(db):
    lines = []
    runs = demo.explore_combined(FakeRuntime(baseline_len=20, act_fn=_up(8.0, 1.0)), demo.open_readonly(db),
                                 [7, 13, 7], ["Hi "], [0.0, 0.5, 1.0], max_new_tokens=20, log=lines.append)
    assert len(runs) == 1 and [f["feature_id"] for f in runs[0]["features"]] == [7, 13]   # deduped, 2 features
    assert [r["strength"] for r in runs[0]["results"]] == [0.0, 0.5, 1.0]
    for r in runs[0]["results"]:
        assert [a["feature_id"] for a in r["activations"]] == [7, 13]
        assert all(set(a) == {"feature_id", "target_activation_baseline", "target_activation_steered",
                              "activation_verdict"} for a in r["activations"])
    assert runs[0]["results"][0]["differs"] is False and runs[0]["results"][1]["differs"] is True
    assert runs[0]["results"][1]["first_divergence_index"] == 5
    act = [l for l in lines if "activation: baseline" in l]
    assert len(act) == 6 and all(l.startswith("    feature ") for l in act)       # 3 strengths x 2 features
    assert any(l.startswith("    feature 13 (upside – upside down): activation: baseline") for l in act)
    assert sum(l.startswith("  baseline") for l in lines) == 1                    # shown once per prompt
    rows = [l for l in lines if l.startswith("  x")]
    assert len(rows) == 3 and rows[0].count("ok") == 1 and "div=-" in rows[0] and "div=5" in rows[1]
    assert lines[0].startswith("combine 7 (hon- – honing, honk) + 13 (upside – upside down) | prompt: 'Hi '")
    json.dumps(runs)


def test_explore_combined_marks_a_loop_and_skips_dead_features(db):
    base = [ord(c) for c in "abcdefghij" * 3]
    lines = []
    runs = demo.explore_combined(FakeRuntime(base_ids=base, steered_ids=[ord("z")] * 30),
                                 demo.open_readonly(db), [7, 13], ["Hi "], [2.0], max_new_tokens=60,
                                 log=lines.append)
    assert runs[0]["results"][0]["degenerate"] is True
    assert any("DEGENERATE: mostly repeated tokens" in l for l in lines)
    con = sqlite3.connect(db)
    con.execute("UPDATE features SET is_dead=1 WHERE feature_id=13")
    con.commit()
    con.close()
    msgs = []
    with pytest.raises(demo.sc.SteeringCheckError, match="needs 2 to 3 usable features, got 1"):
        demo.explore_combined(FakeRuntime(), demo.open_readonly(db), [7, 13], ["Hi "], [1.0], log=msgs.append)
    assert any("skip feature 13" in m and "dead feature" in m for m in msgs)
    with pytest.raises(demo.sc.SteeringCheckError):
        demo.explore_combined(FakeRuntime(), demo.open_readonly(db), [7, 99], ["Hi "], [1.0], log=lambda *_: None)


def test_main_combine_writes_the_explore_file_only(db, tmp_path, monkeypatch, capsys):
    _env(monkeypatch, db, tmp_path)
    out = tmp_path / "explore.json"
    examples = tmp_path / "phase4_examples.json"
    examples.write_text('{"keep": true}')
    before = open(db, "rb").read()
    demo.main(["--combine", "--feature", "7", "--feature", "13", "--prompt", "A ", "--strengths", "0,0.5",
               "--max-new-tokens", "20", "--out", str(examples), "--explore-out", str(out)])
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["mode"] == "combine" and data["strengths"] == [0, 0.5] and data["run_id"] == "test-run"
    assert [f["feature_id"] for f in data["runs"][0]["features"]] == [7, 13]
    assert all(len(r["activations"]) == 2 for r in data["runs"][0]["results"])
    assert examples.read_text() == '{"keep": true}' and open(db, "rb").read() == before
    assert capsys.readouterr().out.count("activation: baseline") == 4            # 2 strengths x 2 features


@pytest.mark.parametrize("argv", [
    ["--combine", "--feature", "7", "--prompt", "x"],                            # one feature
    ["--combine", "--feature", "7", "--feature", "7", "--prompt", "x"],          # the same feature twice
    ["--combine", "--feature", "1", "--feature", "2", "--feature", "3", "--feature", "4", "--prompt", "x"],
    ["--combine", "--mode", "ablate", "--feature", "7", "--feature", "13", "--prompt", "x"],
    ["--combine"],                                                               # no explore arguments
])
def test_main_combine_argument_errors(db, tmp_path, monkeypatch, argv):
    _env(monkeypatch, db, tmp_path)
    with pytest.raises(SystemExit) as e:
        demo.main(argv)
    assert e.value.code == 2


def test_combine_does_not_change_the_default_examples(db, tmp_path, monkeypatch):
    _env(monkeypatch, db, tmp_path)
    out = tmp_path / "phase4_examples.json"
    demo.main(["--out", str(out), "--max-new-tokens", "20"])
    ids = [e["feature_id"] for e in json.loads(out.read_text(encoding="utf-8"))["examples"]]
    assert ids == [3446, 3446, 1863, 13, 1863, 3587, 7]
