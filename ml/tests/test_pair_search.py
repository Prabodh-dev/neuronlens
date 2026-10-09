import importlib.util
import itertools
import json
import os
import sqlite3

import pytest

from neuronlens import pair_search as ps
from neuronlens import steering_checks as sc
from neuronlens.steering_checks import MultiSteeringSpec, SteeringSpec
from tests.test_steering_checks import FakeRuntime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
P = ps.PROMPTS


def row(fid, label=None, status="named", density=0.001, mx=10.0, dead=0):
    return {"feature_id": fid, "label": label or f"f{fid}", "label_status": status,
            "density": density, "max_activation": mx, "is_dead": dead}


# ---------------------------------------------------------------- candidates

def pool_rows():
    rows = [row(i) for i in range(100, 160)]
    rows += [row(200, status="unlabelled"), row(201, status="ambiguous"),
             row(202, dead=1), row(203, density=0.0004), row(204, density=0.021),
             row(205, density=0.0005), row(206, density=0.02), row(207, mx=0.0)]
    rows += [row(3446, status="unlabelled", density=0.5), row(1863, density=0.0001), row(13),
             row(3587, dead=1)]                                   # known: one is dead
    return rows


def test_sample_includes_known_and_applies_the_filters():
    got = {r["feature_id"] for r in ps.sample_candidates(pool_rows(), 1000, 42)}
    assert {3446, 1863, 13} <= got                                # known stay even if unnamed / rare
    assert 3587 not in got                                        # but a dead one is dropped
    assert not got & {200, 201, 202, 203, 204, 207}               # unnamed, ambiguous, dead, density, no max
    assert {205, 206} <= got                                      # density bounds are inclusive
    assert got >= set(range(100, 160))


def test_sample_size_is_the_total_and_deterministic():
    a = ps.sample_candidates(pool_rows(), 20, 42)
    assert len(a) == 20 and [r["feature_id"] for r in a] == sorted(r["feature_id"] for r in a)
    assert {3446, 1863, 13} <= {r["feature_id"] for r in a}
    assert a == ps.sample_candidates(pool_rows(), 20, 42)         # same seed, same sample
    assert a != ps.sample_candidates(pool_rows(), 20, 7)          # another seed, another sample
    tiny = ps.sample_candidates(pool_rows(), 2, 42)               # fewer than the known ones: they all stay
    assert {r["feature_id"] for r in tiny} == {3446, 1863, 13}


# ---------------------------------------------------------------- the rules

@pytest.mark.parametrize("verdict,degenerate,differs,expected", [
    ("higher", False, True, True), ("higher", True, True, False), ("higher", False, False, False),
    ("about the same", False, True, False), ("lower", False, True, False)])
def test_single_works_rule(verdict, degenerate, differs, expected):
    assert ps.single_works(verdict, degenerate, differs) is expected


def test_qualification_needs_two_of_three_prompts():
    rec = lambda *w: [{"works": x} for x in w]
    assert ps.qualifies(rec(True, True, False)) and ps.qualifies(rec(True, True, True))
    assert ps.qualifies(rec(False, True, True))
    assert not ps.qualifies(rec(True, False, False)) and not ps.qualifies(rec(False, False, False))
    assert ps.MIN_WORKING_PROMPTS == 2


@pytest.mark.parametrize("verdicts,degenerate,differs,expected", [
    (["higher", "higher"], False, True, True),
    (["higher", "about the same"], False, True, False),          # only one feature rises
    (["lower", "higher"], False, True, False),
    (["higher", "higher"], True, True, False),                   # loop warning
    (["higher", "higher"], False, False, False),                 # text unchanged
    (["higher"], False, True, False)])
def test_pair_works_requires_both_higher_no_loop_and_a_change(verdicts, degenerate, differs, expected):
    assert ps.pair_works(verdicts, degenerate, differs) is expected


def q(fid, works_on, change):
    return {"feature_id": fid, "label": f"f{fid}", "max_activation": 10.0, "works_on": list(works_on),
            "mean_change": change, "per_prompt": []}


def test_pair_selection_ranking_and_cap():
    qs = [q(1, P[:2], 0.9), q(2, P[:2], 0.5), q(3, P[2:], 0.8), q(4, P[1:], 0.1), q(5, P[:2], 0.5)]
    pairs = ps.select_pairs(qs, 100)
    ids = [(a["feature_id"], b["feature_id"]) for a, b, _, _ in pairs]
    assert (1, 3) not in ids and (2, 3) not in ids                # no shared prompt where both worked
    assert ids[0] == (1, 2) and pairs[0][2] == list(P[:2]) and pairs[0][3] == pytest.approx(1.4)
    assert ids.index((1, 5)) == 1                                 # tie with (1, 2): smaller ids first
    assert ids.index((2, 5)) < ids.index((2, 4)) and ids[-1] in {(1, 4), (4, 5), (2, 4)}
    scores = [s for *_, s in pairs]
    assert scores == sorted(scores, reverse=True)
    assert len(ps.select_pairs(qs, 2)) == 2 and [(a["feature_id"], b["feature_id"]) for a, b, *_ in ps.select_pairs(qs, 2)] == ids[:2]


def test_ranking_of_working_pairs_by_the_smaller_change():
    def run(a, b, ca, cb, works=True, prompt=P[0], strength=0.5):
        return {"features": [{"feature_id": a, "activation_change": ca}, {"feature_id": b, "activation_change": cb}],
                "min_change": min(ca, cb), "sum_change": ca + cb, "works": works, "prompt": prompt,
                "strength": strength}
    runs = [run(1, 2, 0.9, 0.2), run(3, 4, 0.5, 0.4), run(5, 6, 0.45, 0.45), run(7, 8, 0.95, 0.9, works=False),
            run(9, 10, 0.4, 0.5)]
    got = ps.rank_working(runs)
    assert [r["features"][0]["feature_id"] for r in got] == [5, 3, 9, 1]    # min 0.45, 0.4 (sum .9), 0.4, 0.2
    assert all(r["works"] for r in got)
    assert [r["features"][0]["feature_id"] for r in ps.rank_working(list(reversed(runs)))] == [5, 3, 9, 1]


# ---------------------------------------------------------------- the search, with a stub runtime

def key6(ids):
    return "".join(chr(i) for i in ids[:6])


class TableRuntime(FakeRuntime):
    """FakeRuntime whose features react per prompt: table[(feature_id, prompt)] is "up" (the
    activation is high on the steered text) or "flat". `loopers` are single features whose steered
    text loops; `pair_loops` makes any combination loop."""
    d_sae = 12288

    def __init__(self, table, loopers=(), pair_loops=False):
        def act(ids, fid):
            kind = table.get((fid, key6(ids)), "flat")
            if kind == "up":
                return [8.0 if 65 <= i <= 90 or i == 126 else 1.0 for i in ids]
            return [3.0] * len(ids)
        super().__init__(baseline_len=60, diverge_at=5, act_fn=act)
        self.loopers, self.pair_loops = set(loopers), pair_loops

    def generate(self, prompt, max_new_tokens, seed, steering=None):
        loop = False
        if isinstance(steering, SteeringSpec):
            loop = steering.feature_id in self.loopers
        elif isinstance(steering, MultiSteeringSpec):
            loop = self.pair_loops
        self.steered_ids = [126] * 20 if loop else None
        return super().generate(prompt, max_new_tokens, seed, steering)


def keys(*idx):
    return [P[i][:6] for i in idx]


def table_for(spec):
    t = {}
    for fid, prompts in spec.items():
        for i in prompts:
            t[(fid, P[i][:6])] = "up"
    return t


ROWS = [row(10), row(20), row(30), row(40), row(50)]
TABLE = table_for({10: (0, 1, 2), 20: (0, 1), 30: (2,), 50: (0, 1, 2)})   # 40 never reacts


def ticker():
    c = itertools.count()
    return lambda: float(next(c))


def run_search(rt=None, **kw):
    rt = rt or TableRuntime(TABLE, loopers={50})
    lines = []
    doc = ps.search(rt, ROWS, kw.pop("candidates", 10), kw.pop("max_pairs", 60), kw.pop("seed", 42),
                    max_new_tokens=20, log=lines.append, clock=ticker(), **kw)
    return doc, lines


def test_run_single_matches_steering_check():
    rt = TableRuntime(TABLE)
    base = rt.generate(P[0], 20, 42, None)
    r = ps.run_single(rt, P[0], base, ROWS[0], 0.5, 42, 20)
    x1 = sc.steering_check(TableRuntime(TABLE), P[0], 10, "amplify", 0.5, 10.0, 20, 42)
    assert r["verdict"] == x1["activation_verdict"] == "higher"
    assert r["baseline_activation"] == x1["target_activation_baseline"]
    assert r["steered_activation"] == x1["target_activation_steered"]
    assert r["activation_change"] == x1["activation_change"] and r["differs"] == x1["differs"]
    assert r["works"] is True and len(r["steered_text"]) <= ps.SINGLE_TEXT_CHARS


def test_search_stages_end_to_end():
    doc, lines = run_search()
    assert [c["feature_id"] for c in doc["candidates"]] == [10, 20, 30, 40, 50]
    qual = {x["feature_id"]: x for x in doc["qualifying"]}
    assert set(qual) == {10, 20}                    # 30 works on 1 prompt, 40 never, 50 loops
    assert qual[10]["works_on"] == list(P) and qual[20]["works_on"] == list(P[:2])
    assert all(len(x["per_prompt"]) == 3 for x in qual.values())
    assert all(r["steered_text"] for r in qual[10]["per_prompt"])
    # pairs: (10, 20) shares prompts 0 and 1, at strengths 0.25 and 0.5 -> 4 runs, all working
    assert doc["pairs_tested"] == 1 and doc["pair_runs"] == 4 and len(doc["working_pairs"]) == 4
    first = doc["working_pairs"][0]
    assert [f["feature_id"] for f in first["features"]] == [10, 20]
    assert all(f["verdict"] == "higher" for f in first["features"]) and first["degenerate"] is False
    assert first["seed"] == 42 and first["prompt"] in P[:2] and first["strength"] in (0.25, 0.5)
    assert len(first["steered_text"]) <= ps.PAIR_TEXT_CHARS
    s = doc["summary"]
    assert (s["candidates_tested"], s["qualifying"], s["pairs_tested"], s["working_pairs"]) == (5, 2, 1, 4)
    assert s["generations"] == 3 + 5 * 3 + 4                      # baselines + singles + pair runs (baseline reused)
    out = "\n".join(lines)
    assert "qualifies: 10" in out and "working pair: 10 + 20" in out
    assert "10 (f10), works on 3 of 3 prompts" in out and "->" in out
    assert "progress:" in out and ps.format_summary(doc).count("\n") >= 5


def test_a_combination_that_loops_does_not_work():
    doc, lines = run_search(TableRuntime(TABLE, pair_loops=True, loopers={50}))
    assert doc["summary"]["qualifying"] == 2                       # feature 50 loops on its own
    assert doc["pair_runs"] > 0 and doc["working_pairs"] == []
    assert "no pair worked" in "\n".join(lines)


def test_no_qualifying_candidate_means_no_pairs():
    doc, lines = run_search(TableRuntime({}, loopers=()))
    assert doc["summary"]["qualifying"] == 0 and doc["pairs_tested"] == 0 and doc["working_pairs"] == []
    assert "no candidate qualified" in "\n".join(lines)


def test_both_must_be_higher_in_the_combination():
    class OneRises(TableRuntime):
        def feature_activations(self, ids, fid):
            acts = super().feature_activations(ids, fid)
            return acts if fid == 10 or not any(self.last_multi) else [3.0] * len(acts)

        def generate(self, prompt, n, seed, steering=None):
            self.last_multi = [isinstance(steering, MultiSteeringSpec)]
            return super().generate(prompt, n, seed, steering)
    doc, _ = run_search(OneRises(TABLE, loopers={50}))
    assert doc["pair_runs"] == 4 and doc["working_pairs"] == []   # feature 20 does not rise when combined


def test_progress_line_every_ten_generations():
    doc, lines = run_search()
    prog = [l for l in lines if l.startswith("progress:")]
    assert len(prog) == doc["summary"]["generations"] // 10 and prog[0].startswith("progress: 10 generations")
    assert "stage 1" in prog[0]


def test_search_is_deterministic_for_a_fixed_seed():
    big = [row(i) for i in range(100, 400)]
    run = lambda seed: ps.search(TableRuntime({}), big, 25, 60, seed, max_new_tokens=20,
                                 log=lambda *_: None, clock=ticker())
    a, b, c = run(42), run(42), run(7)
    assert a == b                                                  # same seed, same document
    assert a["candidates"] != c["candidates"]                      # another seed, another sample
    assert len(a["candidates"]) == 25


def test_baseline_option_of_multi_steering_check():
    rt = TableRuntime(TABLE)
    steers = [{"feature_id": 10, "strength": 0.5, "max_activation": 10.0},
              {"feature_id": 20, "strength": 0.5, "max_activation": 10.0}]
    plain = sc.multi_steering_check(rt, P[0], steers, 20, 42)
    n_before = len(rt.calls)
    base = rt.generate(P[0], 20, 42, None)
    given = sc.multi_steering_check(rt, P[0], steers, 20, 42, baseline=base)
    assert given == plain
    assert [c[3] for c in rt.calls[n_before + 1:]] == [MultiSteeringSpec(((10, 0.5, 10.0), (20, 0.5, 10.0)))]


# ---------------------------------------------------------------- the script

def load_script():
    spec = importlib.util.spec_from_file_location("find_pairs", os.path.join(ROOT, "ml", "scripts", "find_pairs.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_reads_the_db_read_only_and_writes_only_the_output(tmp_path, monkeypatch, capsys):
    script = load_script()
    db = str(tmp_path / "t.db")
    con = sqlite3.connect(db)
    con.executescript(open(os.path.join(ROOT, "contracts", "schema.sql")).read())
    con.execute("INSERT INTO meta VALUES ('run_id','test-run')")
    for r in ROWS:
        con.execute("INSERT INTO features (feature_id, label, label_status, density, max_activation, is_dead) "
                    "VALUES (?,?,?,?,?,?)", (r["feature_id"], r["label"], "named", 0.001, 10.0, 0))
    con.commit()
    con.close()
    ckpt = tmp_path / "x.pt"
    ckpt.write_bytes(b"")
    monkeypatch.setenv("NEURONLENS_DB", db)
    monkeypatch.setenv("NEURONLENS_CKPT", str(ckpt))
    monkeypatch.setattr(script, "load_runtime", lambda path: TableRuntime(TABLE, loopers={50}))
    out = tmp_path / "pairs_search.json"
    before = open(db, "rb").read()
    script.main(["--candidates", "10", "--max-new-tokens", "20", "--out", str(out)])
    assert open(db, "rb").read() == before
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["run_id"] == "test-run" and data["summary"]["working_pairs"] == 4
    assert {"seed", "prompts", "qualifying", "working_pairs", "summary"} <= set(data)
    printed = capsys.readouterr().out
    assert "candidates tested: 5" in printed and "qualifying: 2" in printed and "total time:" in printed
    for argv in (["--candidates", "0"], ["--max-pairs", "0"]):
        with pytest.raises(SystemExit) as e:
            script.main(argv)
        assert e.value.code == 2
    monkeypatch.delenv("NEURONLENS_DB")
    with pytest.raises(SystemExit, match="NEURONLENS_DB"):
        script.main([])
