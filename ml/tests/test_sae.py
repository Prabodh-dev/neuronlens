import torch
from neuronlens.sae import TopKSAE


def _data(n, d=32, m=64, active=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    D = torch.randn(m, d, generator=g)
    D /= D.norm(dim=1, keepdim=True)
    c = torch.zeros(n, m)
    idx = torch.rand(n, m, generator=g).topk(active, dim=1).indices
    c.scatter_(1, idx, torch.rand(n, active, generator=g) + 0.5)
    return c @ D * 3


def test_shapes_and_topk():
    sae = TopKSAE(32, 128, k=4)
    x = torch.randn(10, 32)
    recon, acts, pre = sae(x)
    assert recon.shape == x.shape and acts.shape == (10, 128)
    assert ((acts > 0).sum(-1) <= 4).all()


def test_decoder_norm():
    sae = TopKSAE(32, 128, k=4)
    with torch.no_grad():
        sae.W_dec.mul_(5)
    sae.normalize_decoder()
    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(128), atol=1e-5)


def test_parallel_grad_removed():
    sae = TopKSAE(32, 128, k=4)
    loss, _ = sae.loss(torch.randn(16, 32))
    loss.backward()
    sae.remove_parallel_grad()
    assert (sae.W_dec.grad * sae.W_dec).sum(1).abs().max() < 1e-5


def test_training_reduces_loss():
    torch.manual_seed(0)
    x = _data(8192)
    sae = TopKSAE(32, 128, k=4, aux_k=16, dead_after=20)
    sae.init_b_dec(x[:2000])
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    first = None
    for _ in range(600):
        b = x[torch.randint(0, len(x), (256,))]
        loss, m = sae.loss(b)
        if first is None:
            first = float(m["mse"])
        opt.zero_grad()
        loss.backward()
        sae.remove_parallel_grad()
        opt.step()
        sae.normalize_decoder()
    assert torch.isfinite(loss)
    assert float(m["mse"]) < 0.5 * first, (first, float(m["mse"]))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)