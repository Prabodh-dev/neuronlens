"""Auto-Lab: the scoring rule in server/scripts/autolab_check.py, the chain run against the
mock, and the mock replaying demo/autolab_examples.json exactly."""
import importlib.util
import json

import pytest
from fastapi.testclient import TestClient

from app.config import REPO_ROOT, Settings
from app.main import create_app
from app.services import mock

from .conftest import API

spec = importlib.util.spec_from_file_location(
    "autolab_check", REPO_ROOT / "server" / "scripts" / "autolab_check.py")
al = importlib.util.module_from_spec(spec)
spec.loader.exec_module(al)

W, X, B = al.WORKED, al.NO_EFFECT, al.BREAKS


# ---------- the scoring rule ----------

@pytest.mark.parametrize("unsteered, steered, loop, want", [
    (0.0, 3.0, False, W),      # rise exactly 3.0, unsteered 0: the 2x part holds by itself
    (0.0, 2.99, False, X),     # rise just under 3.0
    (3.0, 6.0, False, W),      # rise 3.0 and exactly 2x
    (3.5, 6.6, False, X),      # rise 3.1 but under 2x unsteered
    (0.0, 12.44, True, B),     # a loop is "breaks down" whatever the rise
    (0.0, 0.0, True, B),
    (5.0, 4.0, False, X),      # went down
    (0.0, 0.0, False, X),      # strength 0, the control
])
def test_score_cell(unsteered, steered, loop, want):
    assert al.score_cell(unsteered, steered, loop) == want


def _row(fid, verdicts, rises=None):
    rises = rises or [1.0] * len(verdicts)
    return {"feature_id": fid, "label": f"f{fid}",
            "cells": [{"strength": s, "verdict": v, "rise": r}
                      for s, v, r in zip(al.STRENGTHS, verdicts, rises)]}


def test_all_no_effect_is_nothing_worked():
    rows = [_row(f, [X] * 7) for f in range(5)]
    assert al.recommend(rows) == {"status": al.NOTHING}


def test_all_breaks_down_is_nothing_worked():
    rows = [_row(f, [X] + [B] * 6) for f in range(5)]
    assert al.recommend(rows) == {"status": al.NOTHING}


def test_worked_at_one_strength_only_is_not_reliable():
    rows = [_row(1, [X, X, W, X, W, B, B]),   # 0.5 and 1.0 worked, not next to each other
            _row(2, [X, W, B, B, B, B, B])]   # 0.25 alone
    assert al.recommend(rows) == {"status": al.NOT_RELIABLE}
    assert al.STATUS_TEXT[al.NOT_RELIABLE] == "Worked at one strength only, not reliable"
    assert al.STATUS_TEXT[al.NOTHING] == "Nothing worked"


def test_strength_0_is_ignored():
    assert al.recommend([_row(1, [W, W, X, X, X, X, X])]) == {"status": al.NOT_RELIABLE}


def test_most_worked_cells_then_lowest_stable_strength():
    a = _row(1, [X, X, W, W, B, B, B], [0, 0, 9, 12, 0, 0, 0])           # 2 ✓, best rise 12
    b = _row(2, [X, W, X, W, W, W, B], [0, 3, 0, 4, 5, 6, 0])            # 4 ✓, stable from 0.75
    rec = al.recommend([a, b])
    assert (rec["feature_id"], rec["strength"], rec["rise"], rec["worked_cells"]) == (2, 0.75, 4, 4)


def test_tie_goes_to_higher_rise_at_best_cell():
    a = _row(1, [X, X, W, W, X, X, X], [0, 0, 4, 5, 0, 0, 0])
    b = _row(2, [X, X, X, W, W, X, X], [0, 0, 0, 3, 20, 0, 0])
    assert (al.recommend([a, b])["feature_id"], al.recommend([a, b])["strength"]) == (2, 0.75)


def test_only_features_with_a_stable_cell():
    a = _row(1, [X, W, X, W, X, W, X], [0, 9, 0, 9, 0, 9, 0])             # 3 ✓, none stable
    b = _row(2, [X, X, X, X, W, W, X], [0, 0, 0, 0, 3, 3, 0])             # 2 ✓, stable
    assert al.recommend([a, b])["feature_id"] == 2


def test_ship_needs_2_of_3_prompts():
    ok, no = {"status": al.RECOMMENDED}, {"status": al.NOT_RELIABLE}
    results = [{"goal": g, "prompt": p, "recommendation": r} for g, p, r in [
        ("music", "p1", ok), ("music", "p2", ok), ("music", "p3", no),
        ("rain", "p1", ok), ("rain", "p2", no), ("rain", "p3", None)]]
    s = al.ship_summary(results)
    assert (s["music"]["ship"], s["music"]["recommended_on"]) == (True, 2)
    assert (s["rain"]["ship"], s["rain"]["recommended_on"]) == (False, 1)


def test_rule_constants():
    """The README and Anand's UI copy these; change them together."""
    assert (al.RISE_MIN, al.RATIO_MIN, al.SEED, al.CANDIDATES, al.MAX_DENSITY,
            al.SHIP_MIN_PROMPTS, al.SEARCH_TOP_N, al.MIN_CANDIDATES, al.FILLER_SHARE) == \
        (3.0, 2.0, 42, 5, 0.01, 2, 8, 3, 0.10)
    assert al.STRENGTHS == [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    assert len(al.COMMON_WORDS) == 200 == len(set(al.COMMON_WORDS))


def test_parse_goals():
    text = "# comment\n\nmusic\ncourt ruling | The man walked into the room and\n"
    assert al.parse_goals(text, "P") == [("music", "P"),
                                         ("court ruling", "The man walked into the room and")]
    with pytest.raises(SystemExit):
        al.parse_goals(" | prompt only", "P")


# ---------- a recorded file and the mock ----------

PROMPT = "The band walked on stage and"


def _sweep(fid, label, cells):
    """cells: (steered activation, degenerate) per strength; unsteered is 0.0."""
    req = {"prompt": PROMPT, "feature_id": fid, "strengths": al.STRENGTHS, "seed": 42,
           "max_new_tokens": 60, "with_activations": True}
    res = {"feature": {"feature_id": fid, "label": label}, "mode": "amplify", "seed": 42,
           "max_new_tokens": 60, "baseline_text": PROMPT + " the crowd booed.",
           "baseline_degenerate": False, "baseline_reason": None,
           "results": [{"strength": s, "text": PROMPT + f" text at {s}",
                        "differs_from_baseline": s > 0, "first_divergence_index": 3 if s else None,
                        "degenerate": d, "reason": "repeating loop" if d else None,
                        "metrics": {"new_tokens": 3, "repeated_token_fraction": 0.0,
                                    "repeated_trigram_fraction": 0.0, "non_word_fraction": 0.0,
                                    "replacement_chars": 0},
                        "target_activation_baseline": 0.0, "target_activation_steered": a}
                       for s, (a, d) in zip(al.STRENGTHS, cells)]}
    return {"request": req, "response": res}


# 3446 is not in the 512-feature sample DB: recorded answers must not need the DB
GOAL = {"goal": "music", "prompt": PROMPT,
        "search_request": {"query": " music", "top_n": 8, "max_density": 0.01},
        "search": {"query": " music", "model_name": "gpt2", "simulated": False,
                   "tokens": [" music"],
                   "features": [{"feature_id": 3446, "label": "music", "max_activation": 9.1,
                                 "relative_to_max": 0.4, "tokens_active": 1, "peak_token": 0,
                                 "density": 0.001},
                                {"feature_id": 13, "label": "upside", "max_activation": 2.0,
                                 "relative_to_max": 0.1, "tokens_active": 1, "peak_token": 0,
                                 "density": 0.002},
                                {"feature_id": 7, "label": "weather", "max_activation": 1.5,
                                 "relative_to_max": 0.1, "tokens_active": 1, "peak_token": 0,
                                 "density": 0.002}]},
        "sweeps": [_sweep(3446, "music", [(0, False), (1.5, False), (12.44, False),
                                          (14, False), (15, True), (16, True), (17, True)]),
                   _sweep(13, "upside", [(0, False)] * 7),
                   _sweep(7, "weather", [(0, False)] * 7)]}


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    path = tmp_path / "autolab_examples.json"
    path.write_text(json.dumps({"run_id": "gpt2-l8-k32", "goals": [GOAL]}), encoding="utf-8")
    monkeypatch.setattr(mock, "AUTOLAB_PATH", path)
    with TestClient(create_app(Settings())) as c:
        yield c


def _via(client):
    """autolab_check.call, answered by a TestClient instead of the network."""
    def call(base, path, body=None, timeout=10):
        r = client.get(path) if body is None else client.post(path, json=body)
        if r.status_code >= 400:
            raise al.CallFailed(f"{path}: HTTP {r.status_code} {r.json()['error']['code']}")
        return r.json(), 0.01
    return call


def _untag(text):
    return text.replace("[simulated]", "", 1)


def assert_mock_replays(client, goals):
    """Every recorded search and sweep comes back from the mock exactly, except that it is
    marked simulated and its texts carry the [simulated] tag. A request recorded more than
    once (the same goal words on several prompts, the same feature for two goals) is served
    from its first recording, so later copies are compared with that one."""
    seen = set()
    for g in goals:
        key = json.dumps(g["search_request"], sort_keys=True)
        got = client.post(f"{API}/features/search", json=g["search_request"]).json()
        assert got["simulated"] is True
        if key not in seen:
            seen.add(key)
            assert {**got, "simulated": g["search"]["simulated"]} == g["search"], g["goal"]
        for sw in g["sweeps"]:
            key = json.dumps({k: sw["request"][k] for k in sw["request"]
                              if k != "with_activations"}, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            got = client.post(f"{API}/steer/sweep", json=sw["request"]).json()
            assert got["baseline_text"].startswith(sw["request"]["prompt"] + "[simulated]")
            got["baseline_text"] = _untag(got["baseline_text"])
            for r in got["results"]:
                assert "[simulated]" in r["text"]
                r["text"] = _untag(r["text"])
            assert got == sw["response"], (g["goal"], sw["request"]["feature_id"])


def test_mock_replays_recorded_file(recorded):
    assert_mock_replays(recorded, [GOAL])


def test_repeated_requests_use_the_first_recording(tmp_path, monkeypatch):
    """18 lines = 6 goals x 3 prompts: the same search is recorded 3 times. A later copy
    that differs in the last digits must not break the replay or the drift test."""
    copy = json.loads(json.dumps(GOAL))
    copy["prompt"] = "She opened the door and"
    copy["search"]["features"][0]["max_activation"] += 1e-6
    copy["sweeps"] = []
    path = tmp_path / "autolab_examples.json"
    path.write_text(json.dumps({"goals": [GOAL, copy]}), encoding="utf-8")
    monkeypatch.setattr(mock, "AUTOLAB_PATH", path)
    with TestClient(create_app(Settings())) as c:
        assert_mock_replays(c, [GOAL, copy])
        got = c.post(f"{API}/features/search", json=GOAL["search_request"]).json()
        assert got["features"][0]["max_activation"] == 9.1


def test_demo_file_matches_mock(client):
    """Fails on any difference between demo/autolab_examples.json and what the mock serves."""
    if not mock.AUTOLAB_PATH.exists():
        pytest.skip("demo/autolab_examples.json not recorded yet (autolab_check.py --record)")
    goals = json.loads(mock.AUTOLAB_PATH.read_text(encoding="utf-8"))["goals"]
    assert goals, "the recorded file has no goals"
    assert_mock_replays(client, goals)


def test_chain_on_mock(recorded, monkeypatch):
    monkeypatch.setattr(al, "call", _via(recorded))
    res = al.run_chain("http://test", "music", PROMPT)
    assert not res["failures"] and res["simulated"] is True
    marks = {r["feature_id"]: [c["verdict"] for c in r["cells"]] for r in res["rows"]}
    assert marks[3446] == [X, X, W, W, B, B, B]
    assert marks[13] == marks[7] == [X] * 7         # all no effect
    rec = res["recommendation"]
    assert (rec["status"], rec["feature_id"], rec["strength"]) == (al.RECOMMENDED, 3446, 0.5)


# ---------- --replay and Prabodh's recorded run (demo/autolab_examples.json) ----------

def _replay_recording():
    """Prabodh's recording, scored the way the chain scores it: with the filler list."""
    return al.replay(str(mock.AUTOLAB_PATH), al.load_fillers(al.FILLERS_PATH))[1]


def _replayed(goal, prompt):
    hits = [r for r in _replay_recording() if (r["goal"], r["prompt"]) == (goal, prompt)]
    assert hits, f"{goal!r} on {prompt!r} is not in the recorded file"
    return hits[0]["recommendation"]


def test_recording_matches_the_filler_list():
    """The recording was made with the committed filler list: no filler was swept."""
    fillers = al.load_fillers(al.FILLERS_PATH)
    data = json.loads(mock.AUTOLAB_PATH.read_text(encoding="utf-8"))
    assert len(fillers) == 9 and sorted(fillers) == data["rule"]["filler_ids"]
    assert all(sw["request"]["feature_id"] not in fillers
               for g in data["goals"] for sw in g["sweeps"])


def test_court_law_on_man_picks_8362_not_9489():
    rec = _replayed("court law", "The man walked into the room and")
    assert rec["status"] == al.RECOMMENDED and rec["feature_id"] == 8362


def test_sports_on_band_ships_after_the_rerecord():
    """Was "not reliable" in the first recording (top 5, no leading space). With the
    leading space, top 8 and the filler list, sports finds #4901 on this prompt."""
    rec = _replayed("sports game", "The band walked on stage and")
    assert (rec["status"], rec["feature_id"], rec["strength"]) == (al.RECOMMENDED, 4901, 1.5)


def test_all_6_goals_ship():
    ships = al.ship_summary(_replay_recording())
    assert set(ships) == {"music", "court law", "cooking food", "sports game", "space planet",
                          "weather rain"}
    assert all(g["ship"] for g in ships.values())
    assert ships["sports game"]["recommended_on"] == 3


def test_replay_needs_no_server(monkeypatch, capsys):
    monkeypatch.setattr(al, "call", None)  # any server call would fail
    assert al.main(["--replay", str(mock.AUTOLAB_PATH)]) == 0
    assert "Buttons (a goal ships" in capsys.readouterr().out


def test_recorded_sweep_without_flag_has_no_activations(recorded):
    req = {**GOAL["sweeps"][0]["request"]}
    del req["with_activations"]
    got = recorded.post(f"{API}/steer/sweep", json=req).json()
    assert all("target_activation_steered" not in r for r in got["results"])


def test_other_requests_get_stand_ins(recorded):
    r = recorded.post(f"{API}/features/search", json={"query": "music", "top_n": 5}).json()
    assert all(f["feature_id"] != 3446 for f in r["features"])   # no density cut: not recorded
    r = recorded.post(f"{API}/steer/sweep", json={**GOAL["sweeps"][0]["request"], "seed": 7})
    assert r.status_code == 404                                    # 3446 is not in the sample DB


def test_old_server_without_activations_is_a_failure(monkeypatch):
    sweep = {"results": [{"strength": 0.0, "degenerate": False, "reason": None}]}
    search = {"simulated": False, "features": [{"feature_id": i, "label": "x"} for i in (1, 2, 3)]}
    monkeypatch.setattr(al, "call", lambda base, path, body=None, timeout=10:
                        (search if "search" in path else sweep, 0.01))
    res = al.run_chain("http://test", "music", PROMPT)
    assert any("older than with_activations" in f for f in res["failures"])


def test_record_refuses_a_mock_server(client, monkeypatch, tmp_path):
    monkeypatch.setattr(al, "call", _via(client))
    with pytest.raises(SystemExit, match="real-mode"):
        al.main(["--goal", "music", "--record", str(tmp_path / "x.json")])


def test_record_writes_the_mock_format(recorded, monkeypatch, tmp_path):
    """--record output is exactly what the mock reads (health is faked as real)."""
    call = _via(recorded)

    def real_health(base, path, body=None, timeout=10):
        out, s = call(base, path, body, timeout)
        return ({**out, "mode": "real"} if path.endswith("/health") else out), s
    monkeypatch.setattr(al, "call", real_health)
    out = tmp_path / "rec.json"
    assert al.main(["--goal", "music", "--prompt", PROMPT, "--record", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["rule"]["rise_min"] == 3.0 and len(data["goals"]) == 1
    g = data["goals"][0]
    assert g["search_request"] == GOAL["search_request"]
    assert [s["request"]["feature_id"] for s in g["sweeps"]] == [3446, 13, 7]
    assert mock.load_autolab_examples(out)[0]["goal"] == "music"


# ---------- search experiments and --fillers ----------

def test_search_request_leading_space_top_8():
    assert al.search_request("music") == {"query": " music", "top_n": 8, "max_density": 0.01}
    assert al.search_request(" court law ", "relative") == {
        "query": " court law", "top_n": 8, "max_density": 0.01, "sort": "relative"}


def test_fillers_on_the_recorded_run():
    """In Prabodh's run, #1375 peaked on the first query word (token 0, right after the
    newline prefix) in every goal it showed up in."""
    found = al.fillers(_replay_recording())
    assert {1375, 5023, 3356} <= set(found)
    assert len(found[1375]["goals"]) == 6                  # in every goal's top 8
    assert all(pk == 0 for pk, _, _ in found[1375]["goals"].values())


def test_fillers_counts_goal_words_not_prompts():
    def res(goal, prompt, fid, peak):
        return {"goal": goal, "prompt": prompt, "search": {"tokens": ["a", " b"], "features": [
            {"feature_id": fid, "label": "x", "density": 0.004, "peak_token": peak,
             "relative_to_max": 0.5}]}}
    found = al.fillers([res("music", "p1", 1, 0), res("music", "p2", 1, 0),
                        res("music", "p3", 1, 0), res("rain", "p1", 2, 0), res("food", "p1", 2, 1)])
    assert list(found) == [2]                       # #1: one goal on 3 prompts is not a filler
    assert found[2]["goals"] == {"rain": (0, "a", 0.5), "food": (1, " b", 0.5)}


def test_search_only_runs_no_sweeps(recorded, monkeypatch, capsys):
    calls = []
    via = _via(recorded)

    def counting(base, path, body=None, timeout=10):
        calls.append(path)
        return via(base, path, body, timeout)
    monkeypatch.setattr(al, "call", counting)
    assert al.main(["--goal", "music", "--search-only", "--fillers"]) == 0
    assert not any("sweep" in p for p in calls)
    assert "peak token" in capsys.readouterr().out


def test_experiment_options_cannot_record_or_replay(tmp_path):
    for argv in (["--goal", "x", "--record", str(tmp_path / "r.json"), "--sort", "relative"],
                 ["--replay", str(mock.AUTOLAB_PATH), "--search-only"]):
        with pytest.raises(SystemExit):
            al.main(argv)


# ---------- candidate selection: leading space, top 8, fillers dropped, first 5 ----------

def _feats(*ids):
    return [{"feature_id": i, "label": f"f{i}"} for i in ids]


def test_select_drops_fillers_and_keeps_the_first_5_in_order():
    kept, dropped = al.select_candidates(_feats(1, 1375, 2, 3, 5023, 4, 5, 6), {1375, 5023})
    assert [f["feature_id"] for f in kept] == [1, 2, 3, 4, 5]
    assert dropped == [1375, 5023]


def test_select_without_a_filler_list_keeps_the_first_5():
    kept, dropped = al.select_candidates(_feats(*range(8)), set())
    assert [f["feature_id"] for f in kept] == [0, 1, 2, 3, 4] and dropped == []


def _fake_server(features, sweeps):
    """autolab_check.call answered from fixed search features; records every path."""
    paths = []

    def call(base, path, body=None, timeout=10):
        paths.append(path)
        if path.endswith("/features/search"):
            return {"simulated": False, "tokens": [" x"], "features": features}, 0.01
        return sweeps(body["feature_id"]), 0.01
    return call, paths


def _sweep_response(rises):
    return {"results": [{"strength": s, "degenerate": False, "reason": None,
                         "target_activation_baseline": 0.0, "target_activation_steered": r}
                        for s, r in zip(al.STRENGTHS, rises)]}


def test_fewer_than_3_left_is_no_good_match_and_runs_no_sweeps(monkeypatch):
    call, paths = _fake_server(_feats(1375, 5023, 3356, 8559, 7, 9), lambda fid: None)
    monkeypatch.setattr(al, "call", call)
    res = al.run_chain("http://t", "music", PROMPT, filler_ids={1375, 5023, 3356, 8559})
    assert res["recommendation"] == {"status": al.NO_MATCH}
    assert res["dropped_fillers"] == [1375, 5023, 3356, 8559]
    assert not any("sweep" in p for p in paths)
    assert al.STATUS_TEXT[al.NO_MATCH].startswith("No good match")


def test_chain_sweeps_only_the_kept_candidates(monkeypatch):
    call, paths = _fake_server(_feats(1375, 1, 2, 5023, 3, 4, 5, 6),
                               lambda fid: _sweep_response([0, 0, 4, 5, 0, 0, 0]))
    monkeypatch.setattr(al, "call", call)
    res = al.run_chain("http://t", "music", PROMPT, filler_ids={1375, 5023})
    assert [r["feature_id"] for r in res["rows"]] == [1, 2, 3, 4, 5]
    assert sum("sweep" in p for p in paths) == 5
    assert (res["recommendation"]["status"], res["recommendation"]["feature_id"]) == \
        (al.RECOMMENDED, 1)


def test_replay_applies_the_same_selection():
    """music's top 8 holds 5 fillers, so exactly 3 candidates are left; one more dropped
    id leaves 2, which is "no good match"."""
    by = {(r["goal"], r["prompt"]): r for r in _replay_recording()}
    music = by[("music", "The band walked on stage and")]
    assert music["dropped_fillers"] == [1375, 5023, 2347, 3356, 2109]
    assert [r["feature_id"] for r in music["rows"]] == [3446, 3482, 7982]
    fillers = al.load_fillers(al.FILLERS_PATH) | {3446}
    _, results = al.replay(str(mock.AUTOLAB_PATH), fillers)
    music = {(r["goal"], r["prompt"]): r for r in results}[("music",
                                                            "The band walked on stage and")]
    assert music["recommendation"] == {"status": al.NO_MATCH}


def test_mock_replays_the_new_chain(client, monkeypatch):
    """With the re-recorded file the mock answers the UI chain with the real results: the
    same recommendation as --replay, marked simulated."""
    monkeypatch.setattr(al, "call", _via(client))
    res = al.run_chain("http://t", "music", PROMPT, filler_ids=al.load_fillers(al.FILLERS_PATH))
    assert not res["failures"] and res["simulated"] is True
    assert res["recommendation"] == _replayed("music", PROMPT)
    assert (res["recommendation"]["feature_id"], res["recommendation"]["strength"]) == (3482, 0.25)


# ---------- --find-fillers ----------

def test_filler_list_cutoff_and_near_misses():
    counts = {1375: 150, 5023: 120, 2347: 31, 9: 20, 8: 3, 7: 1}
    above, below = al.filler_list(counts, {1375: "more"}, 200)
    assert [f["feature_id"] for f in above] == [1375, 5023, 2347]   # more than 20 of 200
    assert [f["feature_id"] for f in below] == [9, 8, 7]            # 20 is not more than 10%
    assert above[0] == {"feature_id": 1375, "label": "more", "count": 150, "share": 0.75}


def test_find_fillers_writes_the_list(client, monkeypatch, tmp_path):
    """Every word returns #1375 and #5023; #2347 every 4th word; one word's own feature."""
    seen = []

    def call(base, path, body=None, timeout=10):
        if path.endswith("/health"):
            return {**client.get(path).json(), "mode": "real", "run_id": "gpt2-l8-k32"}, 0.01
        seen.append(body)
        i = len(seen)
        ids = [1375, 5023] + ([2347] if i % 4 == 0 else []) + [10000 + i]
        return {"features": _feats(*ids)}, 0.01
    monkeypatch.setattr(al, "call", call)
    out = tmp_path / "autolab_fillers.json"
    assert al.main(["--find-fillers", str(out)]) == 0
    assert len(seen) == 200 and all(b["query"].startswith(" ") and b["top_n"] == 8 for b in seen)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert [f["feature_id"] for f in data["fillers"]] == [1375, 5023, 2347]
    assert [f["count"] for f in data["fillers"]] == [200, 200, 50]
    assert data["cutoff"] == {"share": 0.1, "more_than_words": 20.0} and data["words"] == 200
    assert data["run_id"] == "gpt2-l8-k32" and len(data["date"]) == 10
    assert al.load_fillers(out) == {1375, 5023, 2347}


def test_find_fillers_refuses_a_mock_server(client, monkeypatch, tmp_path):
    monkeypatch.setattr(al, "call", _via(client))
    with pytest.raises(SystemExit, match="real-mode"):
        al.main(["--find-fillers", str(tmp_path / "f.json")])


def test_missing_filler_list_drops_nothing(tmp_path):
    assert al.load_fillers(tmp_path / "none.json") == set()
