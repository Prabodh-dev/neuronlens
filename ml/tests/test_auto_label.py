import csv
import importlib.util
import json
import os
import shutil
import sqlite3
import threading

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
spec = importlib.util.spec_from_file_location(
    "auto_label", os.path.join(ROOT, "ml", "scripts", "auto_label.py"))
auto_label = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auto_label)

QUIET = lambda *_: None  # noqa: E731


def _db(tmp_path):
    path = str(tmp_path / "t.db")
    shutil.copyfile(os.path.join(ROOT, "contracts", "fixtures", "sample.db"), path)
    return path


def _rows(path):
    con = sqlite3.connect(path)
    rows = {r[0]: r[1:] for r in con.execute(
        "SELECT feature_id, label, label_status, notes, is_dead FROM features")}
    con.close()
    return rows


def _fid(prompt):
    return int(prompt.split()[1].rstrip("."))


def _labels(out):
    return json.load(open(out, encoding="utf-8"))["labels"]


class FakeLabeller:
    """Stands in for a Gemini slot: answers from the feature id in the prompt."""

    def __init__(self, name="fake", fail_ids=(), always_fail=False, model="fake-model"):
        self.name, self.model, self.usage = name, model, auto_label.Usage()
        self.calls, self.fail_ids, self.always_fail = [], set(fail_ids), always_fail
        self.lock = threading.Lock()

    def __call__(self, system, prompt):
        fid = _fid(prompt)
        with self.lock:
            self.calls.append((system, prompt))
        if self.always_fail or fid in self.fail_ids:
            raise RuntimeError("boom")
        self.usage.add(100, 10)
        status = "ambiguous" if fid % 5 == 0 else "named"
        return ({"label": f"  Fake `label` {fid}.  ", "status": status, "reason": "test"},
                "", (100, 10))


# ---------- DB reading and prompt ----------

def test_candidates_are_only_unlabelled_live(tmp_path):
    path = _db(tmp_path)
    rows = _rows(path)
    con = auto_label.connect_ro(path)
    cands = auto_label.candidate_ids(con)
    con.close()
    expected = sorted(f for f, (label, status, _, dead) in rows.items()
                      if status == "unlabelled" and label is None and not dead)
    assert cands == expected and cands


def test_parse_ids_ranges():
    assert auto_label.parse_ids("3,5-7, 10") == [3, 5, 6, 7, 10]
    assert auto_label.parse_ids("1863") == [1863]
    with pytest.raises(ValueError):
        auto_label.parse_ids("9-3")


def test_ids_range_limits_candidates(tmp_path):
    con = auto_label.connect_ro(_db(tmp_path))
    all_c = auto_label.candidate_ids(con)
    lo, hi = all_c[0], all_c[5]
    picked = auto_label.candidate_ids(con, auto_label.parse_ids(f"{lo}-{hi}"))
    con.close()
    assert picked == [f for f in all_c if lo <= f <= hi]


def test_prompt_marks_peak_and_follows_standard(tmp_path):
    con = auto_label.connect_ro(_db(tmp_path))
    fid = auto_label.candidate_ids(con)[0]
    ev = auto_label.feature_evidence(con, fid)
    con.close()
    assert all("[[" in s and "]]" in s for s in ev["examples"])
    assert ev["promoted"] and ev["suppressed"]
    assert auto_label.build_prompt(ev).startswith(f"Feature {fid}.")
    system = auto_label.SYSTEM
    assert '"ambiguous"' in system and "at most 32 characters" in system
    examples = system.split("Examples of good labels:")[1]
    assert not set(auto_label.QUOTES) & set(examples)  # no quote marks in examples
    for ex in ("alzheimer – disease", "after – named after", "in – before a year",
               "space – after a heading", "we – after than"):
        assert ex in examples and auto_label.clean_label(ex) == ex
    assert auto_label.SCHEMA["properties"]["status"]["enum"] == ["named", "ambiguous"]


def test_snippet_window():
    toks = [f" t{i}" for i in range(40)]
    s = auto_label.snippet(toks, 20)
    assert s.startswith("...") and s.endswith("...") and "[[ t20]]" in s
    assert auto_label.snippet([" a", " b"], 0) == "[[ a]] b"


def test_parse_answer_tolerates_wrappers():
    assert auto_label.parse_answer('{"label": "x", "status": "named"}')["label"] == "x"
    assert auto_label.parse_answer('```json\n{"label": "y", "status": "ambiguous"}\n```')["label"] == "y"
    assert auto_label.parse_answer('<think>hmm {no}</think> ok {"label": "z", "status": "named"}')["label"] == "z"
    with pytest.raises(ValueError):
        auto_label.parse_answer("no json here")


# ---------- label standard ----------

def test_clean_rejects_bad_answers():
    with pytest.raises(ValueError):
        auto_label.clean({"label": "", "status": "named"})
    with pytest.raises(ValueError):
        auto_label.clean({"label": "x", "status": "maybe"})
    for generic in ("text", "Language", "various topics", "words.", "'`'", "text – words"):
        with pytest.raises(ValueError):
            auto_label.clean({"label": generic, "status": "named"})
    # key word lost (a bracket stripped by cleanup) or context that says nothing new
    for bad in ("( – function call", "– function call", "old – the word old",
                "ul- – word fragment", "hon- – word start", "lig- – lig prefix"):
        with pytest.raises(ValueError):
            auto_label.clean({"label": bad, "status": "named"})
    for good in ("open bracket – function call", "old – titles and names",
                 "hon- – honing, honk", "lip- – biochemical prefix", "food lists",
                 "d – 3d printing"):
        assert auto_label.clean({"label": good, "status": "named"})["label"] == good
    assert auto_label.clean({"label": "mix", "status": "ambiguous"})["status"] == "ambiguous"
    res = auto_label.clean({"label": "x", "status": "named", "reason": " a  b "})
    assert set(res) == {"label", "status", "reason"} and res["reason"] == "a b"


@pytest.mark.parametrize("raw, want", [
    ("'after' in 'named after'", "after in named after"),
    ('the word "weapons"', "the word weapons"),
    ("`code` identifiers", "code identifiers"),
    ("“In” before a year.", "in before a year"),
    ("Legal Contracts!", "legal contracts"),
    ("punctuation marks (periods, dashes, semicolons in lists)", "punctuation marks periods"),
    ("hon- word start", "hon- word start"),
    ("x" * 50, "x" * 32),
    ("'Alzheimer' - disease", "alzheimer – disease"),
    ("after — named after", "after – named after"),
    ("in: before a year", "in: before a year"),
    ("in : before a year", "in – before a year"),
    ("hon- – word start", "hon- – word start"),
    ("anti-slavery", "anti-slavery"),
    ("supreme – supreme court of the united states", "supreme – supreme court"),
    ("longwordlongwordlongword – context", "longwordlongwordlongword"),
    ("career – professional life or work", "career – professional life"),
    ("in – before a year", "in – before a year"),  # short labels keep "a"
])
def test_clean_label_follows_standard(raw, want):
    got = auto_label.clean_label(raw)
    assert got == want and len(got) <= auto_label.MAX_LABEL_LEN


# ---------- propose: rotation, resume, failures ----------

def test_propose_rotates_across_slots_and_resumes(tmp_path):
    path, out = _db(tmp_path), str(tmp_path / "labels.json")
    before = _rows(path)
    servers = [FakeLabeller(f"gemini:s{i}") for i in range(3)]
    lines = []
    done, failed = auto_label.propose(path, out, servers, limit=30, log=lines.append)
    assert (done, failed) == (30, 0)
    run = json.load(open(out, encoding="utf-8"))
    assert run["model"] == "fake-model"  # stored once, not per label
    data = run["labels"]
    assert len(data) == 30
    assert all(set(v) == {"label", "status"} for v in data.values())
    assert all(v["label"] == f"fake label {k}" for k, v in data.items())
    reasons = json.load(open(auto_label.reasons_path(out), encoding="utf-8"))
    assert set(reasons) == set(data) and set(reasons.values()) == {"test"}
    assert sum(len(s.calls) for s in servers) == 30
    assert any(v["status"] == "ambiguous" for v in data.values())
    usage = [line for line in lines if line.startswith("usage ")]
    assert len(usage) == 3 and all("input tokens" in u for u in usage)
    # resume: answered features are not asked again
    again = FakeLabeller()
    auto_label.propose(path, out, [again], limit=5, log=QUIET)
    assert not {_fid(p) for _, p in again.calls} & {int(k) for k in data}
    assert len(_labels(out)) == 35
    assert _rows(path) == before  # propose never writes to the DB


def test_one_model_per_run(tmp_path):
    path, out = _db(tmp_path), str(tmp_path / "labels.json")
    with pytest.raises(SystemExit):
        auto_label.propose(path, out, [FakeLabeller("a", model="m1"),
                                       FakeLabeller("b", model="m2")], limit=2, log=QUIET)
    auto_label.propose(path, out, [FakeLabeller(model="m1")], limit=2, log=QUIET)
    with pytest.raises(SystemExit):  # resuming a file made by another model
        auto_label.propose(path, out, [FakeLabeller(model="m2")], limit=2, log=QUIET)


def test_failed_feature_moves_to_another_slot(tmp_path):
    path, out = _db(tmp_path), str(tmp_path / "labels.json")
    con = auto_label.connect_ro(path)
    ids = auto_label.candidate_ids(con)[:6]
    con.close()
    flaky = FakeLabeller("gemini:flaky", fail_ids=ids[:3])
    good = FakeLabeller("gemini:good")
    done, failed = auto_label.propose(path, out, [flaky, good], ids=ids, log=QUIET)
    assert (done, failed) == (6, 0)
    assert len(_labels(out)) == 6
    assert {_fid(p) for _, p in good.calls} >= set(ids[:3])


def test_dead_slot_is_dropped_others_finish(tmp_path):
    path, out = _db(tmp_path), str(tmp_path / "labels.json")
    dead = FakeLabeller("gemini:dead", always_fail=True)
    good = FakeLabeller("gemini:good")
    lines = []
    done, failed = auto_label.propose(path, out, [dead, good], limit=20, log=lines.append)
    assert (done, failed) == (20, 0)
    assert any("dropped a gemini:dead worker" in line for line in lines)
    assert len(dead.calls) == auto_label.MAX_SERVER_FAILS


def test_usage_per_feature(tmp_path):
    path, out = _db(tmp_path), str(tmp_path / "labels.json")
    con = auto_label.connect_ro(path)
    ids = auto_label.candidate_ids(con)[:4]
    con.close()
    flaky = FakeLabeller("gemini:flaky", fail_ids=[ids[0]])
    good = FakeLabeller("gemini:good")
    lines = []
    auto_label.propose(path, out, [flaky, good], ids=ids, log=lines.append)
    usage = json.load(open(auto_label.usage_path(out), encoding="utf-8"))
    assert set(usage) == {str(f) for f in ids}
    for f in ids:
        u = usage[str(f)]
        failed = u["failed_calls"]
        assert u["calls"] == failed + 1  # one success, after any failed tries
        assert (u["input_tokens"], u["output_tokens"]) == (100, 10)  # failed calls cost 0 here
    assert sum(u["failed_calls"] for u in usage.values()) == len(
        [c for c in flaky.calls if _fid(c[1]) == ids[0]])
    assert any("1 call, 100 in / 10 out" in line for line in lines)
    summary = next(line for line in lines if line.startswith("run total:"))
    assert "400 input tokens" in summary and "100 in" in summary


def test_parse_failure_still_counts_tokens(tmp_path):
    path, out = _db(tmp_path), str(tmp_path / "labels.json")

    class RawLabeller(FakeLabeller):  # like Gemini: answer None, raw text to parse
        def __call__(self, system, prompt):
            self.calls.append((system, prompt))
            return None, "not json", (50, 5)

    lab = RawLabeller("gemini:raw")
    con = auto_label.connect_ro(path)
    fid = auto_label.candidate_ids(con)[0]
    con.close()
    assert auto_label.propose(path, out, [lab], ids=[fid], log=QUIET) == (0, 1)
    u = json.load(open(auto_label.usage_path(out), encoding="utf-8"))[str(fid)]
    assert u == {"calls": 3, "failed_calls": 3, "input_tokens": 150, "output_tokens": 15}


def test_feature_gives_up_after_max_attempts(tmp_path):
    path, out = _db(tmp_path), str(tmp_path / "labels.json")
    con = auto_label.connect_ro(path)
    ids = auto_label.candidate_ids(con)[:2]
    con.close()
    bad = FakeLabeller("gemini:bad", fail_ids=[ids[0]])
    done, failed = auto_label.propose(path, out, [bad], ids=ids, log=QUIET)
    assert (done, failed) == (1, 1)
    assert sum(_fid(p) == ids[0] for _, p in bad.calls) == auto_label.MAX_ATTEMPTS


# ---------- Gemini slots ----------

class _FakeGemini:
    def __init__(self, model, rpm, api_key=None, slot=0, name=None):
        self.model, self.name, self.api_key = model, f"gemini:{name}({model})", api_key


def test_make_labellers_uses_one_key(monkeypatch):
    monkeypatch.setattr(auto_label, "GeminiLabeller", _FakeGemini)
    monkeypatch.setattr(auto_label, "GEMINI_MODELS", [
        {"name": "a", "model": "g", "api_key": "ka"}, {"name": "b", "model": "g", "api_key": "kb"}])
    assert [lab.api_key for lab in auto_label.make_labellers(1)] == ["ka"]  # no rotation
    two = auto_label.make_labellers(2)
    assert len(two) == 2 and two[0] is two[1]  # two workers, same key and rate limiter
    assert [lab.api_key for lab in auto_label.make_labellers(1, slot="b")] == ["kb"]
    with pytest.raises(SystemExit):
        auto_label.make_labellers(1, slot="nope")


# ---------- CLI: range, --apply, backup ----------

def _main(monkeypatch, argv):
    monkeypatch.setattr(auto_label, "make_labellers", lambda *a, **k: [FakeLabeller()])
    return auto_label.main(argv)


def test_start_end_range(tmp_path, monkeypatch):
    path, out = _db(tmp_path), str(tmp_path / "l.json")
    con = auto_label.connect_ro(path)
    cands = auto_label.candidate_ids(con)
    con.close()
    lo, hi = cands[2], cands[8]
    _main(monkeypatch, ["propose", "--db", path, "--out", out,
                        "--start", str(lo), "--end", str(hi)])
    assert sorted(map(int, _labels(out))) == [f for f in cands if lo <= f <= hi]
    out2 = str(tmp_path / "l2.json")
    _main(monkeypatch, ["propose", "--db", path, "--out", out2, "--start", str(cands[-3])])
    assert sorted(map(int, _labels(out2))) == cands[-3:]
    with pytest.raises(SystemExit):
        _main(monkeypatch, ["propose", "--db", path, "--ids", "1", "--start", "1"])


def test_without_apply_db_untouched(tmp_path, monkeypatch, capsys):
    path, out = _db(tmp_path), str(tmp_path / "l.json")
    before = _rows(path)
    _main(monkeypatch, ["propose", "--db", path, "--out", out, "--limit", "3"])
    assert _rows(path) == before and not os.path.exists(path + ".bak3")
    assert "DB not touched" in capsys.readouterr().out


def test_apply_flag_backs_up_then_writes(tmp_path, monkeypatch):
    path, out = _db(tmp_path), str(tmp_path / "l.json")
    before = _rows(path)
    _main(monkeypatch, ["propose", "--db", path, "--out", out, "--limit", "3", "--apply"])
    assert _rows(path + ".bak3") == before
    after = _rows(path)
    written = [f for f in after if after[f] != before[f]]
    assert len(written) == 3
    for f in written:
        assert after[f][0] == f"fake label {f}"
        assert after[f][2].endswith("auto-labelled (fake-model)")  # no reason in the DB
    calls = []
    monkeypatch.setattr(auto_label, "make_labellers", lambda *a, **k: calls.append(1))
    with pytest.raises(SystemExit):  # .bak3 exists: refuse before any LLM call
        auto_label.main(["propose", "--db", path, "--out", out, "--apply"])
    assert not calls and _rows(path) == after


# ---------- apply and sheet ----------

def test_apply_never_overwrites_human_labels(tmp_path):
    path, labels = _db(tmp_path), str(tmp_path / "labels.json")
    rows = _rows(path)
    human = next(f for f, r in rows.items() if r[1] == "named")
    ambiguous = next(f for f, r in rows.items() if r[1] == "ambiguous")
    dead = next(f for f, r in rows.items() if r[3])
    free = [f for f, r in rows.items() if r[1] == "unlabelled" and r[0] is None and not r[3]][:2]
    proposal = {str(f): {"label": f"auto {f}", "status": "named"}
                for f in [human, ambiguous, dead] + free}
    auto_label.save_run(labels, "m", proposal, {k: "SECRET REASON" for k in proposal})
    assert auto_label.apply_labels(path, labels) == (2, 3)
    after = _rows(path)
    for f in (human, ambiguous, dead):
        assert after[f] == rows[f]
    for f in free:
        assert after[f][:2] == (f"auto {f}", "named") and "auto-labelled (m)" in after[f][2]
        assert "SECRET" not in after[f][2]  # reasons never reach the DB
    assert _rows(path + ".bak3") == rows
    with pytest.raises(SystemExit):  # never clobbers an existing backup
        auto_label.apply_labels(path, labels)
    assert auto_label.apply_labels(path, labels, backup=False) == (0, 5)


def test_sheet(tmp_path):
    path, labels, sheet = _db(tmp_path), str(tmp_path / "l.json"), str(tmp_path / "s.csv")
    con = auto_label.connect_ro(path)
    ids = auto_label.candidate_ids(con)[:10]
    con.close()
    auto_label.save_run(labels, "m", {str(f): {"label": f"auto {f}", "status": "named"}
                                      for f in ids}, {})
    auto_label.apply_labels(path, labels, backup=False)
    assert auto_label.write_sheet(path, sheet, n=50) == 10
    rows = list(csv.reader(open(sheet, encoding="utf-8")))
    assert rows[0][:2] == ["feature_id", "auto_label"] and len(rows) == 11


def test_dry_run_makes_no_call(tmp_path, capsys):
    assert auto_label.main(["propose", "--db", _db(tmp_path), "--dry-run",
                            "--ids", "0-600"]) == 0
    out = capsys.readouterr().out
    assert "candidate features" in out and "Top snippets:" in out
    assert "Examples of good labels" in out


# ---------- --batch ----------

class FakeBatch(FakeLabeller):
    """Answers a batched request; can leave ids out or give one a generic label."""

    def __init__(self, name="fake", drop=(), generic=(), **kw):
        super().__init__(name, **kw)
        self.drop, self.generic, self.schemas = set(drop), set(generic), []

    def __call__(self, system, prompt, schema=None):
        import re
        fids = [int(m) for m in re.findall(r"^Feature (\d+)\.", prompt, flags=re.M)]
        with self.lock:
            self.calls.append((system, prompt))
            self.schemas.append(schema)
            drop = {f for f in fids if f in self.drop}
            self.drop -= drop  # only the first time: the retry answers it
        if self.always_fail:
            raise RuntimeError("boom")
        self.usage.add(1000, 100)
        items = [{"id": f, "label": "text" if f in self.generic else f"Batch `x` {f}",
                  "status": "named", "reason": f"r{f}"} for f in fids if f not in drop]
        return None, json.dumps({"features": items}), (1000, 100)


def _cands(path, n):
    con = auto_label.connect_ro(path)
    ids = auto_label.candidate_ids(con)[:n]
    con.close()
    return ids


def test_batch_groups_features_per_call(tmp_path):
    path, out = _db(tmp_path), str(tmp_path / "labels.json")
    ids = _cands(path, 25)
    lab = FakeBatch()
    lines = []
    assert auto_label.propose(path, out, [lab], ids=ids, log=lines.append, batch=10) == (25, 0)
    assert len(lab.calls) == 3  # 10 + 10 + 5
    assert all(s is auto_label.BATCH_SCHEMA for s in lab.schemas)
    assert "Return JSON only: {\"features\"" in lab.calls[0][0]
    assert lab.calls[0][1].startswith("Label these 10 features.")
    data = _labels(out)
    assert data[str(ids[0])]["label"] == f"batch x {ids[0]}"  # still cleaned per feature
    reasons = json.load(open(auto_label.reasons_path(out), encoding="utf-8"))
    assert reasons[str(ids[0])] == f"r{ids[0]}"
    usage = json.load(open(auto_label.usage_path(out), encoding="utf-8"))
    assert usage[str(ids[0])] == {"calls": 1, "failed_calls": 0,
                                  "input_tokens": 100, "output_tokens": 10}  # 1/10 share
    assert usage[str(ids[-1])]["input_tokens"] == 200  # last batch has 5
    summary = next(line for line in lines if line.startswith("run total:"))
    assert summary.startswith("run total: 3 calls (0 failed), 3000 input tokens")
    assert any("1 call of 10" in line for line in lines)


def test_batch_retries_only_bad_features(tmp_path):
    path, out = _db(tmp_path), str(tmp_path / "labels.json")
    ids = _cands(path, 10)
    lab = FakeBatch(drop=[ids[2]], generic=[ids[5]])
    done, failed = auto_label.propose(path, out, [lab], ids=ids, log=QUIET, batch=10)
    assert (done, failed) == (9, 1)  # dropped id answered on retry; generic one never passes
    assert len(lab.calls) == 1 + 2  # first batch, then [ids[2], ids[5]], then [ids[5]]
    assert lab.calls[1][1].startswith("Label these 2 features.")
    data = _labels(out)
    assert str(ids[2]) in data and str(ids[5]) not in data
    usage = json.load(open(auto_label.usage_path(out), encoding="utf-8"))
    assert usage[str(ids[5])]["calls"] == 3 and usage[str(ids[5])]["failed_calls"] == 3
    assert usage[str(ids[0])]["calls"] == 1


def test_batch_bad_reply_and_dead_key(tmp_path):
    path, out = _db(tmp_path), str(tmp_path / "labels.json")
    ids = _cands(path, 4)

    class Junk(FakeBatch):
        def __call__(self, system, prompt, schema=None):
            self.calls.append((system, prompt))
            return None, '{"oops": 1}', (10, 1)

    assert auto_label.propose(path, out, [Junk()], ids=ids, log=QUIET, batch=4) == (0, 4)
    dead = FakeBatch(always_fail=True)
    lines = []
    out2 = str(tmp_path / "l2.json")
    assert auto_label.propose(path, out2, [dead], ids=_cands(path, 30), log=lines.append,
                              batch=10) == (0, 30)
    assert any("dropped a fake worker" in line for line in lines)
    assert len(dead.calls) == auto_label.MAX_SERVER_FAILS  # all 30 counted as failed


def test_split_batch():
    ans = {"features": [{"id": 3, "label": "a"}, {"id": "4", "label": "b"},
                        {"id": 99, "label": "c"}, {"label": "no id"}, {"id": 3, "label": "dup"}]}
    got = auto_label.split_batch(ans, [3, 4, 5])
    assert set(got) == {3, 4} and got[3]["label"] == "a"
    with pytest.raises(ValueError):
        auto_label.split_batch({"oops": 1}, [3])


def test_dry_run_batch(tmp_path, capsys):
    assert auto_label.main(["propose", "--db", _db(tmp_path), "--dry-run", "--batch", "3"]) == 0
    out = capsys.readouterr().out
    assert "Label these 3 features." in out and out.count("Top snippets:") == 3


def test_progress_line(tmp_path, monkeypatch, capsys):
    path, out = _db(tmp_path), str(tmp_path / "l.json")
    monkeypatch.setattr(auto_label, "make_labellers", lambda *a, **k: [FakeBatch()])
    assert auto_label.main(["propose", "--db", path, "--out", out, "--limit", "25",
                            "--batch", "10", "--progress"]) == 0
    text = capsys.readouterr().out
    assert "labelling 25/25 (100%) | 3 calls, 0 failed | 3k in / 300 out" in text
    assert "] " not in text.split("labelling")[0] or "fake):" not in text  # no per-feature lines
    assert "fake): named" not in text
    assert "done: 25 labelled, 0 failed" in text


def test_all_slots(monkeypatch):
    monkeypatch.setattr(auto_label, "GeminiLabeller", _FakeGemini)
    monkeypatch.setattr(auto_label, "GEMINI_MODELS", [
        {"name": "a", "model": "g", "api_key": "ka"}, {"name": "b", "model": "g", "api_key": "kb"},
        {"name": "c", "model": "g", "api_key": "kc"}])
    labs = auto_label.make_labellers(1, all_slots=True)
    assert [lab.api_key for lab in labs] == ["ka", "kb", "kc"]
    assert len(auto_label.make_labellers(2, all_slots=True)) == 6
    with pytest.raises(SystemExit):
        auto_label.make_labellers(1, slot="a", all_slots=True)


def test_all_slots_dead_key_work_moves(tmp_path):
    path, out = _db(tmp_path), str(tmp_path / "l.json")
    dead = FakeBatch("gemini:dead", always_fail=True)
    good = [FakeBatch(f"gemini:k{i}") for i in range(3)]
    lines = []
    done, failed = auto_label.propose(path, out, [dead] + good, ids=_cands(path, 60),
                                      log=lines.append, batch=10)
    assert (done, failed) == (60, 0)  # the dead key's batches were answered by the others
    assert 0 < len(dead.calls) <= auto_label.MAX_SERVER_FAILS
