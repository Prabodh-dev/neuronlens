from collections import defaultdict

import numpy as np
import torch

from .modelutil import backbone, unembedding
from .sae import TopKSAE
from .textdecode import token_pieces

TOP_N = 32
WIN_BEFORE, WIN_AFTER = 48, 16  # window = 64 tokens max


def load_sae(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    sae = TopKSAE(ck["d_model"], ck["d_sae"], ck["k"]).to(device)
    sae.load_state_dict(ck["state_dict"])
    return sae.eval(), ck


@torch.no_grad()
def _acts(model, col, sae, ids, mu, scale):
    backbone(model)(ids)  # fills col.out through the hook
    h = col.out[:, 1:, :].float()  # position p -> index p-1
    B, T1, d = h.shape
    acts, _ = sae.encode((h.reshape(-1, d) - mu) * scale)
    return acts.reshape(B, T1, -1)


@torch.no_grad()
def scan(model, col, sae, ck, seqs, device="cuda", batch=4):
    """Pass 1: density stats + best token per sequence for every feature."""
    F = sae.d_sae
    mu, scale = ck["mu"].to(device), ck["scale"].to(device)
    top_v = torch.full((TOP_N, F), -1.0, device=device)
    top_s = torch.zeros((TOP_N, F), dtype=torch.long, device=device)
    top_p = torch.zeros_like(top_s)
    fire = torch.zeros(F, dtype=torch.long, device=device)
    asum = torch.zeros(F, device=device)
    n_tok = 0
    for i in range(0, len(seqs), batch):
        ids = torch.stack(seqs[i:i + batch]).to(device)
        acts = _acts(model, col, sae, ids, mu, scale)
        B, T1, _ = acts.shape
        flat = acts.reshape(-1, F)
        fire += (flat > 0).sum(0)
        asum += flat.sum(0)
        n_tok += flat.shape[0]
        v, idx = acts.max(1)  # one peak per sequence -> varied examples
        sid = (i + torch.arange(B, device=device)).unsqueeze(1).expand(B, F)
        cv = torch.cat([top_v, v])
        cs = torch.cat([top_s, sid])
        cp = torch.cat([top_p, idx + 1])
        top_v, ni = cv.topk(TOP_N, dim=0)
        top_s, top_p = cs.gather(0, ni), cp.gather(0, ni)
        if (i // batch) % 100 == 0:
            print(f"scan {i}/{len(seqs)}")
    return {"top_v": top_v.cpu().numpy(), "top_s": top_s.cpu().numpy(),
            "top_p": top_p.cpu().numpy(), "fire": fire.cpu().numpy(),
            "asum": asum.cpu().numpy(), "n_tok": n_tok}


@torch.no_grad()
def build_examples(model, col, tok, sae, ck, seqs, S, device="cuda", batch=4):
    """Pass 2: cut 64-token windows and record per-token activations."""
    top_v, top_s, top_p = S["top_v"], S["top_s"], S["top_p"]
    mu, scale = ck["mu"].to(device), ck["scale"].to(device)
    need = defaultdict(list)
    rs, fs = np.nonzero(top_v > 0)
    for r, f in zip(rs, fs):
        need[int(top_s[r, f])].append((int(f), int(r)))
    ex = defaultdict(list)
    order = sorted(need)
    for i in range(0, len(order), batch):
        chunk = order[i:i + batch]
        ids = torch.stack([seqs[s] for s in chunk]).to(device)
        acts = _acts(model, col, sae, ids, mu, scale)
        for b, s in enumerate(chunk):
            items = need[s]
            sub = acts[b][:, [f for f, _ in items]].cpu().numpy()
            toks = seqs[s].tolist()
            text = token_pieces(tok, toks)  # decoded over the whole sequence so windows cut mid-character stay clean
            for j, (f, r) in enumerate(items):
                p = int(top_p[r, f])
                lo, hi = max(0, p - WIN_BEFORE), min(len(toks), p + WIN_AFTER)
                a = np.concatenate([[0.0], sub[:, j]])[lo:hi]
                ex[f].append((r, text[lo:hi],
                              [round(float(x), 2) for x in a], p - lo,
                              float(top_v[r, f])))
        if (i // batch) % 100 == 0:
            print(f"examples {i}/{len(order)}")
    return {f: sorted(v, key=lambda e: e[0]) for f, v in ex.items()}


@torch.no_grad()
def logit_attribution(model, tok, sae, top=10, chunk=512):
    """Direct effect of each decoder direction on the output logits."""
    WU = unembedding(model).float()  # (V, d)
    out = {}
    for s in range(0, sae.d_sae, chunk):
        eff = sae.W_dec[s:s + chunk].float() @ WU.T  # (c, V)
        pv, pi = eff.topk(top, dim=1)
        sv, si = (-eff).topk(top, dim=1)
        for j in range(eff.shape[0]):
            out[s + j] = {
                "promoted": [(tok.decode([int(t)]), float(v)) for t, v in zip(pi[j], pv[j])],
                "suppressed": [(tok.decode([int(t)]), -float(v)) for t, v in zip(si[j], sv[j])],
            }
    return out