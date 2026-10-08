from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from neuronlens.runtime import Runtime  # noqa: E402
from tests.test_textdecode import ByteTok, real_tokenizer  # noqa: E402

FFFD = "�"
TEXT = "They’re happy"
D, F = 6, 20


class CallableTok(ByteTok):
    def __call__(self, text, return_tensors=None):
        return SimpleNamespace(input_ids=torch.tensor([self.encode(text)]))


class FakeSAE:
    d_sae = F

    def encode(self, x):
        g = torch.Generator().manual_seed(0)
        w = torch.randn(x.shape[-1], F, generator=g)
        return torch.relu(x @ w), None


def make_runtime(tok):
    rt = Runtime.__new__(Runtime)  # skip loading GPT-2 and the checkpoint
    rt.tok, rt.sae, rt.device, rt.d_sae = tok, FakeSAE(), "cpu", F
    rt.mu, rt.scale = torch.zeros(D), torch.tensor(1.0)
    rt.col = SimpleNamespace(out=None)

    def transformer(ids):  # stands in for the hooked GPT-2 forward pass
        g = torch.Generator().manual_seed(1)
        rt.col.out = torch.randn(1, ids.shape[1], D, generator=g)

    rt.model = SimpleNamespace(transformer=transformer)
    return rt


def check(tok):
    rt = make_runtime(tok)
    ids = tok(TEXT, return_tensors="pt").input_ids[0].tolist()
    out = rt.inspect(TEXT, top_n=3)
    pieces = [t["text"] for t in out]
    assert len(out) == len(ids) and [t["index"] for t in out] == list(range(len(ids)))
    assert not any(FFFD in p for p in pieces)
    assert "".join(pieces) == TEXT
    # activation data is exactly what the SAE gives for each position
    acts, _ = rt.sae.encode((rt.col.out[0] - rt.mu) * rt.scale)
    vals, idx = acts.topk(3, dim=-1)
    for t, row in enumerate(out):
        want = [] if t == 0 else [{"feature_id": i, "activation": v}
                                  for v, i in zip(vals[t].tolist(), idx[t].tolist()) if v > 0]
        assert row["features"] == want
    return pieces


def test_inspect_split_apostrophe_stand_in_tokenizer():
    pieces = check(CallableTok())
    assert "’" in pieces and "" in pieces  # earlier byte pieces of the character are empty


def test_inspect_real_gpt2_tokenizer():
    check(real_tokenizer())
