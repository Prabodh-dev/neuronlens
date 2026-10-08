"""Real mode without a GPU: a missing checkpoint must not break DB pages, and a loaded
Runtime must be called through the same API. The ml Runtime is replaced by a stand-in."""
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.services import real

from .conftest import API


class FakeRuntime:
    def __init__(self):
        self.calls = []

    def count_tokens(self, text):
        return len(text.split())

    def inspect(self, text, top_n):
        return [{"index": i, "text": w, "features": [] if i == 0 else
                 [{"feature_id": 3, "activation": 1.5}]} for i, w in enumerate(text.split())]

    def steer(self, prompt, feature_id, mode, strength, max_activation, max_new_tokens, seed):
        self.calls.append((feature_id, mode, strength, max_activation, max_new_tokens, seed))
        return {"baseline_text": prompt + " base", "steered_text": prompt + " steered"}


def test_real_mode_without_checkpoint(monkeypatch):
    monkeypatch.delenv("NEURONLENS_CKPT", raising=False)
    with TestClient(create_app(Settings(mode="real", ckpt=""))) as c:
        assert c.get(f"{API}/health").json()["mode"] == "real"
        assert c.get(f"{API}/features").status_code == 200
        assert c.get(f"{API}/metrics").status_code == 200
        for path, body in (("inspect", {"text": "hi"}),
                           ("steer", {"prompt": "hi", "feature_id": 3, "mode": "amplify"})):
            r = c.post(f"{API}/{path}", json=body)
            assert r.status_code == 503 and r.json()["error"]["code"] == "MODEL_NOT_LOADED"
            assert "NEURONLENS_CKPT" in r.json()["error"]["message"]


def test_real_mode_missing_checkpoint_file():
    with TestClient(create_app(Settings(mode="real", ckpt="data/does-not-exist.pt"))) as c:
        r = c.post(f"{API}/inspect", json={"text": "hi"})
        assert r.status_code == 503 and "checkpoint not found" in r.json()["error"]["message"]


def test_real_mode_uses_runtime(monkeypatch):
    fake = FakeRuntime()
    monkeypatch.setattr(real.RealRuntime, "load", classmethod(lambda cls, ckpt: cls(fake)))
    with TestClient(create_app(Settings(mode="real", ckpt="x.pt"))) as c:
        toks = c.post(f"{API}/inspect", json={"text": "a b c"}).json()["tokens"]
        assert toks[0]["is_outlier_position"] and toks[1]["features"][0]["feature_id"] == 3
        assert toks[1]["features"][0]["label"] == \
            c.get(f"{API}/features/3").json()["label"]
        body = c.post(f"{API}/steer", json={"prompt": "Hi", "feature_id": 3,
                                            "mode": "amplify", "strength": 1.5,
                                            "seed": 9}).json()
        assert body["steered_text"] == "Hi steered" and body["seed"] == 9
        max_act = c.get(f"{API}/features/3").json()["max_activation"]
        assert fake.calls == [(3, "amplify", 1.5, max_act, 60, 9)]
        r = c.post(f"{API}/inspect", json={"text": "w " * 257})
        assert r.status_code == 413
