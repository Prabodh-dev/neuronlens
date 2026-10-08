import importlib.util
import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
spec = importlib.util.spec_from_file_location(
    "compare_published", os.path.join(ROOT, "ml", "scripts", "compare_published.py"))
cp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cp)

D = 8


def _norm(d=D, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(d, generator=g), torch.tensor(0.37) + 0 * torch.randn(1, generator=g)[0]


def _identity_sd(d=D):
    eye = torch.eye(d)
    return {"pre_bias": torch.zeros(d),
            "encoder.weight": torch.cat([eye, -eye]),     # (2d, d)
            "latent_bias": torch.zeros(2 * d),
            "decoder.weight": torch.cat([eye, -eye]).T}   # (d, 2d)


def test_roundtrip_exact_for_identity_like_sae():
    mu, scale = _norm()
    # all 2d latents kept: relu(x_ln) - relu(-x_ln) = x_ln, so recon == input
    ad = cp.PublishedSAEAdapter.from_state_dict(_identity_sd(), mu, scale, k=2 * D)
    x_norm = torch.randn(50, D) * 3
    recon, acts, pre = ad(x_norm)
    assert recon.shape == x_norm.shape
    assert acts.shape == (50, 2 * D) and pre.shape == (50, 2 * D)
    assert torch.allclose(recon, x_norm, atol=1e-4)


def test_random_fake_sae_shapes_and_topk():
    mu, scale = _norm()
    g = torch.Generator().manual_seed(1)
    sd = {"pre_bias": torch.randn(D, generator=g), "encoder.weight": torch.randn(20, D, generator=g),
          "latent_bias": torch.randn(20, generator=g), "decoder.weight": torch.randn(D, 20, generator=g)}
    ad = cp.PublishedSAEAdapter.from_state_dict(sd, mu, scale, k=4)
    assert (ad.d_model, ad.d_sae) == (D, 20)
    recon, acts, _ = ad(torch.randn(30, D))
    assert recon.shape == (30, D)
    assert ((acts > 0).sum(-1) <= 4).all() and (acts >= 0).all()


def test_rejects_unknown_format_and_finds_k():
    mu, scale = _norm()
    with pytest.raises(ValueError, match="missing"):
        cp.PublishedSAEAdapter.from_state_dict({"W_enc": torch.zeros(2, 2)}, mu, scale, k=1)
    sd = _identity_sd()
    with pytest.raises(ValueError, match="find k"):
        cp.PublishedSAEAdapter.from_state_dict(sd, mu, scale)
    sd["activation_state_dict"] = {"k": 5}
    assert cp.PublishedSAEAdapter.from_state_dict(sd, mu, scale).k == 5


class _StubModel(nn.Module):
    """Mimics model.transformer.h[i] and model(ids).logits for evals.evaluate."""

    def __init__(self, vocab=30, d=D, layers=3):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.transformer = SimpleNamespace(h=nn.ModuleList([nn.Linear(d, d) for _ in range(layers)]))
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        h = self.emb(ids)
        for blk in self.transformer.h:
            h = h + blk(h)
        return SimpleNamespace(logits=self.head(h))


def test_evaluate_with_adapter_on_stub_model():
    from neuronlens.evals import evaluate
    torch.manual_seed(0)
    model = _StubModel().eval()
    seqs = [torch.randint(0, 30, (16,)) for _ in range(4)]
    mu, scale = _norm()
    ad = cp.PublishedSAEAdapter.from_state_dict(_identity_sd(), mu, scale, k=2 * D)
    m = evaluate(model, ad, seqs, 1, mu, scale, device="cpu", batch=2)
    assert m["fvu"] < 1e-4
    assert abs(m["ce_recon"] - m["ce_orig"]) < 1e-3
    assert m["l0"] <= 2 * D


def test_table_builder():
    m = {"fvu": 0.16, "l0": 32.0, "ce_orig": 3.3, "ce_zero": 14.7, "ce_recon": 3.45,
         "ce_recovered": 0.985}
    rows = [cp.make_row("ours", 12288, 32, m, "note a"), cp.make_row("pub", 32768, 32, m, "note b")]
    out = cp.format_table(rows)
    assert "12,288" in out and "32,768" in out and "note [pub]: note b" in out
    assert rows[0]["ce_zero_ablation"] == 14.7
