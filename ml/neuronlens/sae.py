import torch
import torch.nn as nn
import torch.nn.functional as F


def geometric_median(x, iters=100, eps=1e-6):
    y = x.mean(0)
    for _ in range(iters):
        w = 1.0 / (x - y).norm(dim=1).clamp_min(eps)
        y_new = (x * w[:, None]).sum(0) / w.sum()
        done = (y_new - y).norm() < 1e-5
        y = y_new
        if done:
            break
    return y


class TopKSAE(nn.Module):
    def __init__(self, d_model=768, d_sae=12288, k=32, aux_k=512,
                 aux_coef=1 / 32, dead_after=500):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.aux_k, self.aux_coef, self.dead_after = aux_k, aux_coef, dead_after
        W = torch.randn(d_sae, d_model)
        W /= W.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W)               # (d_sae, d_model), unit-norm rows
        self.W_enc = nn.Parameter(W.t().clone())   # init W_enc = W_dec^T
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.register_buffer("since_fired", torch.zeros(d_sae, dtype=torch.long))

    def pre_acts(self, x):
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x):
        pre = self.pre_acts(x)
        vals, idx = pre.topk(self.k, dim=-1)
        acts = torch.zeros_like(pre).scatter(-1, idx, F.relu(vals))
        return acts, pre

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, x):
        acts, pre = self.encode(x)
        return self.decode(acts), acts, pre

    def loss(self, x):
        recon, acts, pre = self(x)
        mse = F.mse_loss(recon.float(), x.float())
        aux = torch.zeros((), device=x.device)
        dead = self.since_fired > self.dead_after
        n_dead = int(dead.sum())
        if n_dead > 0:
            e = (x - recon).detach().float()
            k = min(self.aux_k, n_dead)
            masked = pre.float().masked_fill(~dead, float("-inf"))
            vals, idx = masked.topk(k, dim=-1)
            a = torch.zeros_like(masked).scatter(-1, idx, F.relu(vals))
            aux = F.mse_loss(a @ self.W_dec.float(), e)
        total = mse + self.aux_coef * aux
        self._update_fired(acts)
        return total, {"mse": mse.detach(), "aux": aux.detach(), "n_dead": n_dead}

    @torch.no_grad()
    def _update_fired(self, acts):
        fired = (acts > 0).any(0)
        self.since_fired += 1
        self.since_fired[fired] = 0

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec /= self.W_dec.norm(dim=1, keepdim=True)

    @torch.no_grad()
    def remove_parallel_grad(self):
        g = self.W_dec.grad
        if g is not None:
            g -= (g * self.W_dec).sum(1, keepdim=True) * self.W_dec

    @torch.no_grad()
    def init_b_dec(self, x):
        self.b_dec.copy_(geometric_median(x.float()))