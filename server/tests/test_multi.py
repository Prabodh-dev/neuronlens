"""W5 POST /steer/multi. Mock serves the recorded example straight from
demo/phase4_examples.json (the only source); other inputs get a stand-in."""
import json

from fastapi.testclient import TestClient

from app.config import REPO_ROOT, Settings
from app.main import create_app
from app.services import real
from app.services.checks import steering_checks

from .conftest import API

DEMO = json.loads((REPO_ROOT / "demo" / "phase4_examples.json").read_text(encoding="utf-8"))
EX = DEMO["multi"][0]
RECORDED = {"prompt": EX["prompt"], "seed": EX["seed"], "max_new_tokens": EX["max_new_tokens"],
            "steers": [{"feature_id": s["feature_id"], "strength": s["strength"]}
                       for s in EX["steers"]],
            "with_single_controls": True}


def _untag(text):
    return text.replace("[simulated]", "", 1)


# ---------- the recorded example (mock) ----------

def test_recorded_example_matches_demo_file(client):
    """Fails if the mock response and demo/phase4_examples.json ever differ."""
    b = client.post(f"{API}/steer/multi", json=RECORDED).json()
    labels = {f["feature_id"]: f["label"] for f in EX["features"]}
    assert b["simulated"] is True
    assert (b["prompt"], b["seed"], b["max_new_tokens"]) == (EX["prompt"], EX["seed"],
                                                              EX["max_new_tokens"])
    assert b["steers"] == [{"feature_id": s["feature_id"], "label": labels[s["feature_id"]],
                            "strength": s["strength"]} for s in EX["steers"]]
    assert _untag(b["baseline_text"]) == EX["baseline"]
    assert _untag(b["steered_text"]) == EX["steered"]
    assert b["baseline_text"].startswith(EX["prompt"] + "[simulated]")
    for key in ("differs", "first_divergence_index", "degenerate", "reason"):
        assert b[key] == EX[key], key
    assert b["activations"] == [{**a, "label": labels[a["feature_id"]]} for a in EX["activations"]]
    runs = [{**r, "steered_text": _untag(r["steered_text"])} for r in b["single_feature_runs"]]
    assert runs == [{**r, "label": labels[r["feature_id"]]} for r in EX["single_feature_runs"]]


def test_recorded_numbers(client):
    b = client.post(f"{API}/steer/multi", json=RECORDED).json()
    assert [round(a["target_activation_steered"], 2) for a in b["activations"]] == [4.20, 5.89]
    assert [round(r["target_activation_steered"], 2) for r in b["single_feature_runs"]] == \
        [2.29, 4.45]
    assert all(a["activation_verdict"] == "higher" for a in b["activations"])


def test_default_strength_is_half(client):
    body = {"prompt": EX["prompt"], "steers": [{"feature_id": s["feature_id"]}
                                               for s in EX["steers"]]}
    b = client.post(f"{API}/steer/multi", json=body).json()
    assert [s["strength"] for s in b["steers"]] == [0.5, 0.5]  # matches the recorded run
    assert b["single_feature_runs"] is None  # controls only when asked


# ---------- stand-in (mock) ----------

def test_stand_in(client):
    body = {"prompt": "Hello there", "steers": [{"feature_id": 13}, {"feature_id": 7, "strength": 1}],
            "with_single_controls": True}
    a = client.post(f"{API}/steer/multi", json=body).json()
    assert a == client.post(f"{API}/steer/multi", json=body).json()  # deterministic
    assert a["simulated"] and "[simulated]" in a["steered_text"] and a["differs"]
    assert [s["feature_id"] for s in a["steers"]] == [13, 7]
    assert a["steers"][0]["label"] == client.get(f"{API}/features/13").json()["label"]
    assert [x["feature_id"] for x in a["activations"]] == [13, 7]
    assert all(x["activation_verdict"] == "higher" for x in a["activations"])
    assert [r["strength"] for r in a["single_feature_runs"]] == [0.5, 1.0]
    assert isinstance(a["degenerate"], bool)


def test_errors(client):
    dead = next(i["feature_id"] for i in client.get(
        f"{API}/features?include_dead=true&limit=200").json()["items"] if i["is_dead"])
    two = [{"feature_id": 13}, {"feature_id": 7}]
    cases = [({"prompt": "Hi", "steers": [{"feature_id": 13}]}, 422, "INVALID_INPUT"),
             ({"prompt": "Hi", "steers": two + [{"feature_id": 8}, {"feature_id": 9}]}, 422,
              "INVALID_INPUT"),
             ({"prompt": "Hi", "steers": [{"feature_id": 13}, {"feature_id": 13}]}, 422,
              "INVALID_INPUT"),
             ({"prompt": "Hi", "steers": [{"feature_id": 13, "strength": 3.5}, {"feature_id": 7}]},
              422, "INVALID_INPUT"),
             ({"prompt": "Hi", "steers": [{"feature_id": 13}, {"feature_id": dead}]}, 422,
              "INVALID_INPUT"),
             ({"prompt": "Hi", "steers": [{"feature_id": 13}, {"feature_id": 99999}]}, 404,
              "FEATURE_NOT_FOUND"),
             ({"prompt": "  ", "steers": two}, 422, "INVALID_INPUT"),
             ({"prompt": "word " * 300, "steers": two}, 413, "TEXT_TOO_LONG")]
    for body, status, code in cases:
        r = client.post(f"{API}/steer/multi", json=body)
        assert (r.status_code, r.json()["error"]["code"]) == (status, code), body
    ok = client.post(f"{API}/steer/multi",
                     json={"prompt": "Hi", "steers": [{"feature_id": 13, "strength": 3},
                                                      {"feature_id": 7, "strength": 0}]})
    assert ok.status_code == 200


# ---------- real mode with a stand-in ml Runtime ----------

class FakeMlRuntime:
    d_sae = 512

    def __init__(self):
        self.calls = []
        self.wrapper = None

    def count_tokens(self, text):
        return len(text.split())

    def generate(self, prompt, max_new_tokens, seed, steering=None):
        assert self.wrapper._lock._is_owned()
        sc = steering_checks()
        self.calls.append(steering)
        if steering is None:
            extra = []
        elif isinstance(steering, sc.MultiSteeringSpec):
            extra = [f" m{f}" for f, _, _ in steering.steers]
        else:
            extra = [f" s{steering.feature_id}"]
        pieces = [prompt, " a"] + extra
        return sc.Generation("".join(pieces), [hash(p) % 50257 for p in pieces], 1, pieces)

    def feature_activations(self, token_ids, feature_id):
        assert self.wrapper._lock._is_owned()
        return [0.0] + [2.0] * (len(token_ids) - 1) if len(token_ids) > 2 else [0.0] * len(token_ids)


def test_real_mode(monkeypatch):
    fake = FakeMlRuntime()

    def load(cls, ckpt):
        fake.wrapper = cls(fake)
        return fake.wrapper
    monkeypatch.setattr(real.RealRuntime, "load", classmethod(load))
    with TestClient(create_app(Settings(mode="real", ckpt="x.pt"))) as c:
        max_act = {f: c.get(f"{API}/features/{f}").json()["max_activation"] for f in (13, 7)}
        b = c.post(f"{API}/steer/multi", json={"prompt": "Up", "with_single_controls": True,
                                               "steers": [{"feature_id": 13, "strength": 0.5},
                                                          {"feature_id": 7, "strength": 1.5}]}).json()
        assert not b["simulated"] and b["steered_text"] == "Up a m13 m7"
        multi = fake.calls[1]
        assert multi.steers == ((13, 0.5, max_act[13]), (7, 1.5, max_act[7]))  # max from the DB
        assert [r["steered_text"] for r in b["single_feature_runs"]] == ["Up a s13", "Up a s7"]
        assert len(fake.calls) == 4  # baseline, combined, one control per feature
        assert [a["activation_verdict"] for a in b["activations"]] == ["higher", "higher"]


def test_real_mode_not_loaded(monkeypatch):
    monkeypatch.delenv("NEURONLENS_CKPT", raising=False)
    with TestClient(create_app(Settings(mode="real", ckpt=""))) as c:
        r = c.post(f"{API}/steer/multi", json={"prompt": "hi", "steers": [{"feature_id": 3},
                                                                          {"feature_id": 4}]})
        assert r.status_code == 503 and r.json()["error"]["code"] == "MODEL_NOT_LOADED"
