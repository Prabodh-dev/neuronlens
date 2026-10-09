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


def test_all_no_effect_gives_no_button():
    rows = [_row(f, [X] * len(al.STRENGTHS)) for f in range(5)]
    assert al.recommend(rows) is None


def test_all_breaks_down_gives_no_button():
    rows = [_row(f, [X] + [B] * (len(al.STRENGTHS) - 1)) for f in range(5)]
    assert al.recommend(rows) is None


def test_recommend_lowest_strength_then_larger_rise():
    n = len(al.STRENGTHS)
    a = _row(1, [X, X, X, X, W, B, B][:n], [0, 0, 0, 0, 20, 0, 0][:n])     # worked at 1.0
    b = _row(2, [X, X, W, W, B, B, B][:n], [0, 0, 4, 9, 0, 0, 0][:n])      # worked at 0.5
    c = _row(3, [X, X, W, B, B, B, B][:n], [0, 0, 6, 0, 0, 0, 0][:n])      # 0.5, bigger rise
    rec = al.recommend([a, b, c])
    assert (rec["feature_id"], rec["strength"], rec["rise"]) == (3, 0.5, 6)


def test_rule_constants():
    """The README and Anand's UI copy these; change them together."""
    assert (al.RISE_MIN, al.RATIO_MIN, al.SEED, al.CANDIDATES, al.MAX_DENSITY) == \
        (3.0, 2.0, 42, 5, 0.01)
    assert al.STRENGTHS == [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


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
        "search_request": {"query": "music", "top_n": 5, "max_density": 0.01},
        "search": {"query": "music", "model_name": "gpt2", "simulated": False,
                   "tokens": [" music"],
                   "features": [{"feature_id": 3446, "label": "music", "max_activation": 9.1,
                                 "relative_to_max": 0.4, "tokens_active": 1, "peak_token": 0,
                                 "density": 0.001},
                                {"feature_id": 13, "label": "upside", "max_activation": 2.0,
                                 "relative_to_max": 0.1, "tokens_active": 1, "peak_token": 0,
                                 "density": 0.002}]},
        "sweeps": [_sweep(3446, "music", [(0, False), (1.5, False), (12.44, False),
                                          (14, False), (15, True), (16, True), (17, True)]),
                   _sweep(13, "upside", [(0, False)] * 7)]}


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
    marked simulated and its texts carry the [simulated] tag."""
    for g in goals:
        got = client.post(f"{API}/features/search", json=g["search_request"]).json()
        assert got["simulated"] is True
        assert {**got, "simulated": g["search"]["simulated"]} == g["search"], g["goal"]
        for sw in g["sweeps"]:
            got = client.post(f"{API}/steer/sweep", json=sw["request"]).json()
            assert got["baseline_text"].startswith(sw["request"]["prompt"] + "[simulated]")
            got["baseline_text"] = _untag(got["baseline_text"])
            for r in got["results"]:
                assert "[simulated]" in r["text"]
                r["text"] = _untag(r["text"])
            assert got == sw["response"], (g["goal"], sw["request"]["feature_id"])


def test_mock_replays_recorded_file(recorded):
    assert_mock_replays(recorded, [GOAL])


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
    assert marks[13] == [X] * 7                     # all no effect
    rec = res["recommendation"]
    assert (rec["feature_id"], rec["strength"]) == (3446, 0.5)


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
    search = {"simulated": False, "features": [{"feature_id": 1, "label": "x"}]}
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
    assert [s["request"]["feature_id"] for s in g["sweeps"]] == [3446, 13]
    assert mock.load_autolab_examples(out)[0]["goal"] == "music"
