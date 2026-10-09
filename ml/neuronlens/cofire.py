"""W4: how often two features fire on the same token (co-fire counts).

A feature is active on a token when its SAE activation is above 0 (TopK with k = 32, so up to
32 features per token). For every scored token the active features are counted pairwise. The
tokens come from the same pipeline as the final scan: GPT-2 small, hook = output of
transformer.h[8], position 0 dropped (interpret._acts). This only says that two features fire
on the same tokens in the sample that was scanned.
"""
import itertools
import json

import numpy as np
import torch

from .interpret import _acts

MIN_CO_COUNT = 5        # a pair needs at least this many shared tokens to be listed
TOP_PARTNERS = 10       # partners kept per feature
ROW_BLOCK = 3072        # rows of the count matrix computed at once (bounds GPU memory)
SCORED_PER_SEQUENCE = 1023  # a 1024-token sequence scores positions 1 to 1023
_EXACT_FLOAT32 = 1 << 24    # integer counts are exact in float32 below this


class CofireCounter:
    """Pairwise co-occurrence counts of active features, plus each feature's own count.

    counts[a, b] is the number of tokens on which features a and b were both active, and
    counts[a, a] is the number of tokens on which a fired. The (d_sae x d_sae) int32 matrix is
    about 600 MB for 12,288 features. By default it lives in host memory (accumulate="cpu") and
    each batch is added in row blocks computed on the device, so the device only ever holds a
    ROW_BLOCK x d_sae block; accumulate="device" keeps the whole matrix on the device.
    """

    def __init__(self, d_sae, device="cpu", row_block=ROW_BLOCK, accumulate="cpu"):
        if accumulate not in ("cpu", "device"):
            raise ValueError("accumulate must be 'cpu' or 'device'")
        self.d_sae, self.device, self.row_block, self.accumulate = d_sae, device, row_block, accumulate
        self.n_tokens = 0
        if accumulate == "cpu":
            self.counts = np.zeros((d_sae, d_sae), dtype=np.int32)
        else:
            self.counts = torch.zeros((d_sae, d_sae), dtype=torch.int32, device=device)

    @torch.no_grad()
    def add(self, active):
        """active: bool tensor (n_tokens, d_sae)."""
        n = active.shape[0]
        if n >= _EXACT_FLOAT32:
            raise ValueError("too many tokens in one add(); split the batch")
        a = active.to(self.device).to(torch.float32)
        for r0 in range(0, self.d_sae, self.row_block):
            r1 = min(r0 + self.row_block, self.d_sae)
            block = (a[:, r0:r1].T @ a).round().to(torch.int32)   # exact: counts < 2**24
            if self.accumulate == "cpu":
                self.counts[r0:r1] += block.cpu().numpy()
            else:
                self.counts[r0:r1] += block
        self.n_tokens += n

    def numpy_counts(self):
        return self.counts if self.accumulate == "cpu" else self.counts.cpu().numpy()


def top_partners(counts, min_count=MIN_CO_COUNT, top_n=TOP_PARTNERS, skip=()):
    """For every feature, its top partners ranked by Jaccard = c_ab / (c_a + c_b - c_ab).

    Pairs below min_count shared tokens are left out. Ties in Jaccard are ordered by larger
    raw co-count, then smaller feature id. Features that never fired, and features in `skip`
    (for example dead features), get an empty list, and are never listed as a partner either.
    Returns {feature_id: [{"feature_id", "co_count", "jaccard"}, ...]}.
    """
    counts = np.asarray(counts)
    fire = np.diagonal(counts).astype(np.int64)
    skip = set(int(s) for s in skip)
    blocked = fire == 0
    for s in skip:
        blocked[s] = True
    ids = np.arange(counts.shape[0])
    out = {}
    for a in range(counts.shape[0]):
        if blocked[a]:
            out[a] = []
            continue
        c_ab = counts[a].astype(np.int64)
        ok = (c_ab >= min_count) & ~blocked & (ids != a)
        cand = np.nonzero(ok)[0]
        if cand.size == 0:
            out[a] = []
            continue
        co = c_ab[cand]
        jac = co / (fire[a] + fire[cand] - co)   # correctly rounded: equal fractions are equal
        order = np.lexsort((cand, -co, -jac))[:top_n]
        out[a] = [{"feature_id": int(cand[i]), "co_count": int(co[i]), "jaccard": float(jac[i])}
                  for i in order]
    return out


@torch.no_grad()
def scan_cofire(model, col, sae, ck, seqs, n_tokens, counter, device="cuda", batch=4, log=print):
    """Feed token sequences through the model and count co-firing until n_tokens scored tokens
    have been counted (the last batch is cut so the total is exact) or the data runs out.
    Returns the number of tokens counted."""
    mu, scale = ck["mu"].to(device), ck["scale"].to(device)
    it, done = iter(seqs), 0
    while done < n_tokens:
        chunk = list(itertools.islice(it, batch))
        if not chunk:
            break
        ids = torch.stack(chunk).to(device)
        acts = _acts(model, col, sae, ids, mu, scale).reshape(-1, sae.d_sae)  # position 0 dropped
        acts = acts[: n_tokens - done]
        counter.add(acts > 0)
        done += acts.shape[0]
        if (done // (batch * SCORED_PER_SEQUENCE)) % 20 == 0:
            log(f"cofire {done}/{n_tokens} tokens")
    return done


def build_output(run_id, tokens_scanned, seed, k, min_count, partners, ndigits=4):
    """The JSON document. partners: result of top_partners()."""
    return {"run_id": run_id, "tokens_scanned": int(tokens_scanned), "seed": int(seed),
            "k": int(k), "min_count": int(min_count),
            "partners": {str(f): [{"feature_id": p["feature_id"], "co_count": p["co_count"],
                                   "jaccard": round(p["jaccard"], ndigits)} for p in lst]
                         for f, lst in partners.items()}}


def write_json(doc, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))
