"""X6 POST /inspect/report and X7 POST /inspect/compare (both aggregate /inspect)."""
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.routes.analysis import aggregate, split_sentences
from app.services import real

from .conftest import API


def _toks(*texts):
    return [{"index": i, "text": t, "features": []} for i, t in enumerate(texts)]


# ---------- sentence split ----------

def test_split_on_end_marks_followed_by_space_or_end():
    toks = _toks("It", " rains", ".", " Then", " sun", "!", " Why", "?")
    assert split_sentences(toks) == [(0, 3), (3, 6), (6, 8)]


def test_split_newlines_and_no_space_after_mark():
    toks = _toks("Hi", ".", "\n", "\n", "Next", " line", "3", ".", "5", " ok")
    # "3.5" is not split (no space after "."); the blank "\n" pieces are dropped
    assert split_sentences(toks) == [(0, 2), (4, 10)]


def test_known_wrong_split_documented():
    toks = _toks("Mr", ".", " Smith", " came", ".")
    assert split_sentences(toks) == [(0, 2), (2, 5)]  # "Mr." splits early (documented)


def test_aggregate_counts_and_peak():
    toks = [{"index": 0, "text": "A", "features": []},
            {"index": 1, "text": " b", "features": [{"feature_id": 5, "activation": 2.0},
                                                   {"feature_id": 6, "activation": 0.0}]},
            {"index": 2, "text": " c", "features": [{"feature_id": 5, "activation": 3.0}]}]
    assert aggregate(toks) == {5: {"max": 3.0, "count": 2, "peak": 2}}


# ---------- X6 report (mock) ----------

def test_report_mock(client):
    r = client.post(f"{API}/inspect/report",
                    json={"text": "The court ruled. The cat sat!\nDone", "top_n": 3})
    assert r.status_code == 200
    b = r.json()
    assert b["simulated"] and b["model_name"] == "gpt2"
    assert [s["text"] for s in b["sentences"]] == ["The court ruled.", " The cat sat!", "Done"]
    assert [s["index"] for s in b["sentences"]] == [0, 1, 2]
    first = b["sentences"][0]
    assert (first["start_token"], first["end_token"]) == (0, 4)
    for s in b["sentences"]:
        acts = [f["max_activation"] for f in s["features"]]
        assert len(acts) <= 3 and acts == sorted(acts, reverse=True)
        for f in s["features"]:
            assert s["start_token"] <= f["peak_token"] < s["end_token"]
            assert f["peak_token"] != 0 and f["tokens_active"] >= 1
            detail = client.get(f"{API}/features/{f['feature_id']}").json()
            assert f["label"] == detail["label"]
            assert abs(f["relative_to_max"] - f["max_activation"] / detail["max_activation"]) < 1e-3


def test_report_single_sentence_without_end_mark(client):
    b = client.post(f"{API}/inspect/report", json={"text": "no end mark here"}).json()
    assert len(b["sentences"]) == 1 and b["sentences"][0]["text"] == "no end mark here"


# ---------- X7 compare (mock) ----------

def test_compare_mock(client):
    b = client.post(f"{API}/inspect/compare", json={"text_a": "The cat sat",
                                                    "text_b": "The court ruled", "top_n": 5}).json()
    assert b["simulated"] and (b["tokens_a"], b["tokens_b"]) == (3, 3)
    diffs = [abs(f["difference"]) for f in b["features"]]
    assert len(diffs) == 5 and diffs == sorted(diffs, reverse=True)
    for f in b["features"]:
        assert abs(f["difference"] - (f["max_activation_b"] - f["max_activation_a"])) < 1e-6
        if f["tokens_active_a"] == 0:
            assert f["max_activation_a"] == 0 and f["relative_to_max_a"] == 0


def test_compare_same_text_has_no_difference(client):
    b = client.post(f"{API}/inspect/compare", json={"text_a": "Same words", "text_b": "Same words"}).json()
    assert all(f["difference"] == 0 for f in b["features"])


def test_errors(client):
    cases = [("inspect/report", {"text": "  "}, 422, "INVALID_INPUT"),
             ("inspect/report", {"text": "hi", "top_n": 21}, 422, "INVALID_INPUT"),
             ("inspect/report", {"text": "word " * 300}, 413, "TEXT_TOO_LONG"),
             ("inspect/compare", {"text_a": "hi", "text_b": " "}, 422, "INVALID_INPUT"),
             ("inspect/compare", {"text_a": "hi", "text_b": "x", "top_n": 51}, 422, "INVALID_INPUT"),
             ("inspect/compare", {"text_a": "hi", "text_b": "word " * 300}, 413, "TEXT_TOO_LONG")]
    for path, body, status, code in cases:
        r = client.post(f"{API}/{path}", json=body)
        assert (r.status_code, r.json()["error"]["code"]) == (status, code), (path, body)


# ---------- real mode with a stand-in ml Runtime ----------

class FakeRuntime:
    """Every token after the first has 32 active features; feature 3 is the strongest."""

    def __init__(self):
        self.top_ns = []

    def count_tokens(self, text):
        return len(text.split())

    def inspect(self, text, top_n):
        self.top_ns.append(top_n)
        words = text.split(" ")
        out = []
        for i, w in enumerate(words):
            feats = [] if i == 0 else [{"feature_id": f, "activation": 10.0 - f * 0.1}
                                       for f in range(3, 3 + top_n)]
            out.append({"index": i, "text": w if i == 0 else " " + w, "features": feats})
        return out


def test_real_mode_uses_all_active_features(monkeypatch):
    fake = FakeRuntime()
    monkeypatch.setattr(real.RealRuntime, "load", classmethod(lambda cls, ckpt: cls(fake)))
    with TestClient(create_app(Settings(mode="real", ckpt="x.pt"))) as c:
        b = c.post(f"{API}/inspect/report", json={"text": "One two. Three four", "top_n": 2}).json()
        assert fake.top_ns == [32]  # the SAE's k, not the requested top_n
        assert not b["simulated"]
        assert [s["text"] for s in b["sentences"]] == ["One two.", " Three four"]
        assert [f["feature_id"] for f in b["sentences"][0]["features"]] == [3, 4]
        assert b["sentences"][0]["features"][0]["tokens_active"] == 1
        cmp = c.post(f"{API}/inspect/compare", json={"text_a": "a b", "text_b": "a b c"}).json()
        assert fake.top_ns[1:] == [32, 32]
        assert all(f["difference"] == 0 for f in cmp["features"])  # same features, same max
        assert (cmp["tokens_a"], cmp["tokens_b"]) == (2, 3)


def test_real_mode_not_loaded(monkeypatch):
    monkeypatch.delenv("NEURONLENS_CKPT", raising=False)
    with TestClient(create_app(Settings(mode="real", ckpt=""))) as c:
        for path, body in (("inspect/report", {"text": "hi"}),
                           ("inspect/compare", {"text_a": "hi", "text_b": "yo"})):
            r = c.post(f"{API}/{path}", json=body)
            assert r.status_code == 503 and r.json()["error"]["code"] == "MODEL_NOT_LOADED"
