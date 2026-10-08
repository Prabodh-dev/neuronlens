import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate(model, sae, seqs, layer, mu, scale, device="cuda", batch=4):
    """seqs: list of (1024,) token tensors. Returns metrics dict.
    Positions >= 1 are patched/scored (position 0 is the outlier we drop)."""
    block = model.transformer.h[layer]
    d = sae.d_model
    st = {"sse": 0.0, "sxx": 0.0, "n": 0, "l0": 0.0,
          "sx": torch.zeros(d, device=device, dtype=torch.float64)}
    fired = torch.zeros(sae.d_sae, dtype=torch.bool, device=device)
    mode = {"m": None}
    mu_d, scale_d = mu.to(device), scale.to(device)

    def hook(mod, inp, out):
        if mode["m"] is None:
            return None
        h = out[0] if isinstance(out, tuple) else out
        new = h.clone()
        if mode["m"] == "zero":
            new[:, 1:] = 0
        else:
            hs = h[:, 1:]
            x = (hs.float().reshape(-1, d) - mu_d) * scale_d
            recon, acts, _ = sae(x)
            st["sse"] += (recon - x).pow(2).sum().item()
            st["sxx"] += x.double().pow(2).sum().item()
            st["sx"] += x.double().sum(0)
            st["n"] += x.shape[0]
            st["l0"] += (acts > 0).float().sum().item()
            fired.__ior__((acts > 0).any(0))
            new[:, 1:] = (recon / scale_d + mu_d).reshape(hs.shape).to(h.dtype)
        return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

    totals = {"orig": 0.0, "zero": 0.0, "recon": 0.0}
    count = 0
    handle = block.register_forward_hook(hook)
    try:
        for i in range(0, len(seqs), batch):
            ids = torch.stack(seqs[i:i + batch]).to(device)
            tgt = ids[:, 2:].reshape(-1)
            count += tgt.numel()
            for m, key in ((None, "orig"), ("zero", "zero"), ("recon", "recon")):
                mode["m"] = m
                logits = model(ids).logits[:, 1:-1].float()
                totals[key] += F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), tgt, reduction="sum").item()
    finally:
        handle.remove()

    ce = {k: v / count for k, v in totals.items()}
    total_var = st["sxx"] - (st["sx"].pow(2).sum().item() / st["n"])
    return {
        "fvu": st["sse"] / total_var,
        "l0": st["l0"] / st["n"],
        "dead_pct": 100.0 * (1 - fired.float().mean().item()),
        "ce_orig": ce["orig"], "ce_zero": ce["zero"], "ce_recon": ce["recon"],
        "ce_recovered": (ce["zero"] - ce["recon"]) / (ce["zero"] - ce["orig"]),
    }