"""Phase 4 steering checks: /steer/compare (X3), /steer/sweep (X2), /steer/prompt-only (X4).
Mock mode answers from demo/phase4_examples.json for the recorded inputs; real mode is
tested with a stand-in ml Runtime."""
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models import DEFAULT_SWEEP_STRENGTHS
from app.services import real
from app.services.checks import steering_checks

from .conftest import API

ACROBAT = {"prompt": "The acrobat hung", "feature_id": 13}


def _dead_feature(client):
    items = client.get(f"{API}/features?include_dead=true&limit=200").json()["items"]
    return next(i["feature_id"] for i in items if i["is_dead"])


# ---------- X3 compare ----------

def test_compare_recorded_example(client):
    r = client.post(f"{API}/steer/compare", json={**ACROBAT, "mode": "amplify", "strength": 0.5})
    assert r.status_code == 200
    b = r.json()
    assert b["feature"]["feature_id"] == 13 and "feature_id" not in b
    assert b["feature"]["label"] == client.get(f"{API}/features/13").json()["label"]
    assert (b["mode"], b["strength"], b["seed"], b["max_new_tokens"]) == ("amplify", 0.5, 42, 60)
    assert b["baseline_text"].startswith("The acrobat hung[simulated] his head and walked toward")
    assert b["steered_text"].startswith("The acrobat hung[simulated] his head and walked up to")
    assert b["diverges"] and b["first_divergence_index"] == 4
    assert b["identical_prefix_text"] == " his head and walked"
    assert b["baseline_from_divergence_text"].startswith(" toward the door")
    assert b["steered_from_divergence_text"].startswith(" up to the roof")
    assert b["prompt_tokens"] == 3
    assert 0 < b["tokens_changed"] <= b["tokens_compared_from_divergence"]
    assert b["percent_changed_raw_note"] == steering_checks().RAW_PERCENT_NOTE


def test_compare_zero_strength_is_a_control(client):
    b = client.post(f"{API}/steer/compare",
                    json={**ACROBAT, "mode": "amplify", "strength": 0}).json()
    assert not b["diverges"] and b["first_divergence_index"] is None
    assert b["baseline_text"] == b["steered_text"]
    assert b["tokens_changed"] == 0 and b["percent_changed_from_divergence"] is None
    assert b["percent_changed_raw"] == 0.0 and b["steered_from_divergence"] == []


def test_compare_ablate_reports_zero_strength(client):
    b = client.post(f"{API}/steer/compare", json={
        "prompt": "Drivers began to honk their horns while the traffic", "feature_id": 7,
        "mode": "ablate", "strength": 2.5}).json()
    assert b["strength"] == 0.0 and b["mode"] == "ablate"
    assert b["diverges"] and "[simulated]" in b["steered_text"]


def test_compare_stand_in_for_other_prompts(client):
    req = {"prompt": "Hello there", "feature_id": 13, "mode": "amplify", "strength": 2.0}
    a = client.post(f"{API}/steer/compare", json=req).json()
    assert a == client.post(f"{API}/steer/compare", json=req).json()  # deterministic
    assert "[simulated]" in a["baseline_text"] and a["diverges"]


# ---------- X2 sweep ----------

def test_sweep_default_strengths(client):
    b = client.post(f"{API}/steer/sweep", json=ACROBAT).json()
    assert b["mode"] == "amplify" and b["feature"]["feature_id"] == 13
    assert [r["strength"] for r in b["results"]] == DEFAULT_SWEEP_STRENGTHS
    zero = b["results"][0]
    assert zero["text"] == b["baseline_text"] and not zero["differs_from_baseline"]
    assert zero["first_divergence_index"] is None
    assert b["results"][2]["first_divergence_index"] == 4  # strength 0.5, as in compare
    loops = [r for r in b["results"] if r["degenerate"]]
    assert loops and all(r["reason"] for r in loops)
    assert all(r["reason"] is None for r in b["results"] if not r["degenerate"])
    assert set(b["results"][0]["metrics"]) == {"new_tokens", "repeated_token_fraction",
                                               "repeated_trigram_fraction", "non_word_fraction",
                                               "replacement_chars"}
    assert b["baseline_degenerate"] is False and b["baseline_reason"] is None


def test_sweep_keeps_given_order(client):
    b = client.post(f"{API}/steer/sweep", json={**ACROBAT, "strengths": [2, 0.5, 0]}).json()
    assert [r["strength"] for r in b["results"]] == [2.0, 0.5, 0.0]


def test_sweep_strength_limits(client):
    for strengths in ([], [3.5], [-0.1], [0.5] * 13):
        r = client.post(f"{API}/steer/sweep", json={**ACROBAT, "strengths": strengths})
        assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_INPUT", strengths
    assert client.post(f"{API}/steer/sweep",
                       json={**ACROBAT, "strengths": [3] * 12}).status_code == 200


# ---------- X4 prompt-only ----------

def test_prompt_only_matches_baseline(client):
    p = client.post(f"{API}/steer/prompt-only", json={"prompt": "The acrobat hung"}).json()
    c = client.post(f"{API}/steer/compare",
                    json={**ACROBAT, "mode": "amplify", "strength": 0.5}).json()
    assert p["text"] == c["baseline_text"]
    assert p["text"] == "The acrobat hung[simulated]" + p["continuation"]
    assert p["new_tokens"] == c["baseline_new_tokens"]
    assert (p["prompt"], p["seed"], p["max_new_tokens"]) == ("The acrobat hung", 42, 60)


def test_steer_uses_recorded_example(client):
    b = client.post(f"{API}/steer", json={**ACROBAT, "mode": "amplify", "strength": 0.5}).json()
    c = client.post(f"{API}/steer/compare",
                    json={**ACROBAT, "mode": "amplify", "strength": 0.5}).json()
    assert (b["baseline_text"], b["steered_text"]) == (c["baseline_text"], c["steered_text"])


# ---------- errors ----------

def test_errors(client):
    dead = _dead_feature(client)
    cases = [
        ("steer/compare", {"prompt": "Hi", "feature_id": 99999, "mode": "amplify"}, 404,
         "FEATURE_NOT_FOUND"),
        ("steer/sweep", {"prompt": "Hi", "feature_id": 99999}, 404, "FEATURE_NOT_FOUND"),
        ("steer/compare", {"prompt": "Hi", "feature_id": dead, "mode": "amplify"}, 422,
         "INVALID_INPUT"),
        ("steer/sweep", {"prompt": "Hi", "feature_id": dead}, 422, "INVALID_INPUT"),
        ("steer/compare", {"prompt": "Hi", "feature_id": 13, "mode": "amplify",
                           "strength": 3.5}, 422, "INVALID_INPUT"),
        ("steer/prompt-only", {"prompt": "   "}, 422, "INVALID_INPUT"),
        ("steer/prompt-only", {"prompt": "Hi", "max_new_tokens": 121}, 422, "INVALID_INPUT"),
        ("steer/prompt-only", {"prompt": "word " * 300}, 413, "TEXT_TOO_LONG"),
        ("steer/sweep", {"prompt": "word " * 300, "feature_id": 13}, 413, "TEXT_TOO_LONG"),
    ]
    for path, body, status, code in cases:
        r = client.post(f"{API}/{path}", json=body)
        assert (r.status_code, r.json()["error"]["code"]) == (status, code), (path, body)


# ---------- real mode with a stand-in ml Runtime ----------

class FakeMlRuntime:
    """Stands in for neuronlens.runtime.Runtime: generate() returns a Generation."""

    def __init__(self, d_sae=512):
        self.d_sae = d_sae
        self.calls = []
        self.wrapper = None

    def count_tokens(self, text):
        return len(text.split())

    def generate(self, prompt, max_new_tokens, seed, steering=None):
        assert self.wrapper._lock._is_owned()  # the whole check holds the model
        self.calls.append(steering)
        sc = steering_checks()
        extra = [] if steering is None or steering.strength == 0 and steering.mode == "amplify" \
            else [f" s{steering.strength}"]
        pieces = prompt.split(" ")
        pieces = [pieces[0]] + [" " + p for p in pieces[1:]] + [" a", " b"] + extra + [" c"]
        ids = [hash(p) % 50257 for p in pieces]
        return sc.Generation("".join(pieces), ids, len(prompt.split(" ")), pieces)


def _real_client(monkeypatch, fake):
    def load(cls, ckpt):
        wrapper = cls(fake)
        fake.wrapper = wrapper
        return wrapper
    monkeypatch.setattr(real.RealRuntime, "load", classmethod(load))
    return TestClient(create_app(Settings(mode="real", ckpt="x.pt")))


def test_real_mode_forwards_generate_under_lock(monkeypatch):
    fake = FakeMlRuntime()
    with _real_client(monkeypatch, fake) as c:
        max_act = c.get(f"{API}/features/13").json()["max_activation"]
        b = c.post(f"{API}/steer/sweep", json={"prompt": "Up we go", "feature_id": 13,
                                               "strengths": [0, 1.5], "seed": 7}).json()
        assert [r["strength"] for r in b["results"]] == [0.0, 1.5] and b["seed"] == 7
        assert b["results"][1]["text"] == "Up we go a b s1.5 c" and "[simulated]" not in b["baseline_text"]
        assert fake.calls[0] is None and len(fake.calls) == 3  # baseline + one per strength
        assert all(s.max_activation == max_act and s.feature_id == 13 for s in fake.calls[1:])
        p = c.post(f"{API}/steer/prompt-only", json={"prompt": "Up we go"}).json()
        assert p["continuation"] == " a b c" and p["new_tokens"] == 3


def test_real_mode_maps_ml_errors(monkeypatch):
    with _real_client(monkeypatch, FakeMlRuntime(d_sae=10)) as c:  # DB has 13, model does not
        r = c.post(f"{API}/steer/compare", json={"prompt": "Hi there", "feature_id": 13,
                                                 "mode": "amplify"})
        assert r.status_code == 404 and r.json()["error"]["code"] == "FEATURE_NOT_FOUND"


def test_real_mode_not_loaded(monkeypatch):
    monkeypatch.delenv("NEURONLENS_CKPT", raising=False)
    with TestClient(create_app(Settings(mode="real", ckpt=""))) as c:
        for path, body in (("steer/compare", {"prompt": "hi", "feature_id": 3, "mode": "amplify"}),
                           ("steer/sweep", {"prompt": "hi", "feature_id": 3}),
                           ("steer/prompt-only", {"prompt": "hi"})):
            r = c.post(f"{API}/{path}", json=body)
            assert r.status_code == 503 and r.json()["error"]["code"] == "MODEL_NOT_LOADED"
