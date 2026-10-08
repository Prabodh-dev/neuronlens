import math
import os
import time
import itertools

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from .activations import ActivationStream, token_sequences
from .evals import evaluate
from .sae import TopKSAE

_MISSING = object()


def _flat(d, out=None):
    out = {} if out is None else out
    for k, v in d.items():
        if isinstance(v, dict):
            _flat(v, out)
        else:
            out[k] = v
    return out


def load_cfg(path):
    with open(path) as f:
        return _flat(yaml.safe_load(f))


def pick(cfg, names, default=_MISSING):
    for n in names:
        if n in cfg:
            return cfg[n]
    if default is _MISSING:
        raise KeyError(f"none of {names} in config; keys are {sorted(cfg)}")
    return default


def save_ckpt(path, sae, cfg, stream, layer, step, metrics):
    torch.save({
        "state_dict": sae.state_dict(), "cfg": cfg,
        "mu": stream.mu.cpu(), "scale": stream.scale.cpu(),
        "d_model": sae.d_model, "d_sae": sae.d_sae, "k": sae.k,
        "layer": layer, "step": step, "metrics": metrics,
    }, path)


def train(config_path, device="cuda", out_root="~/runs"):
    cfg = load_cfg(config_path)
    run_id = pick(cfg, ["run_id"])
    d_sae = int(pick(cfg, ["d_sae"]))
    k = int(pick(cfg, ["k", "topk"]))
    total = int(pick(cfg, ["n_train_activations"]))
    buf = int(pick(cfg, ["shuffle_buffer", "buffer_size"]))
    bs = int(pick(cfg, ["batch_size", "batch"], 4096))
    lr = float(pick(cfg, ["lr", "learning_rate"], 3e-4))
    aux_k = int(pick(cfg, ["aux_k"], 512))
    aux_coef = float(pick(cfg, ["aux_coef"], 1 / 32))
    seed = int(pick(cfg, ["seed"], 42))
    layer = int(pick(cfg, ["hook_layer", "layer"], 8))
    dead_after = int(pick(cfg, ["dead_steps_threshold"], 500))
    data_dir = pick(cfg, ["data_dir"], "~/data/fineweb/sample/10BT")

    steps = total // bs
    warm = max(1, steps // 100)
    eval_every = max(steps // 10, 200)
    out = os.path.join(os.path.expanduser(out_root), run_id)
    os.makedirs(out, exist_ok=True)
    tb = SummaryWriter(out)
    torch.manual_seed(seed)

    stream = ActivationStream(layer, buf, bs, data_dir, seed, device)
    d_model = stream.model.config.n_embd
    sae = TopKSAE(d_model, d_sae, k, aux_k, aux_coef, dead_after).to(device)
    eval_seqs = list(itertools.islice(token_sequences(stream.tok, data_dir, seed=123), 32))

    it = iter(stream)
    sae.init_b_dec(next(it))
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    print(f"run {run_id}: {steps} steps, batch {bs}, d_sae {d_sae}, k {k}")

    t0 = time.time()
    metrics = {}
    for step in range(1, steps + 1):
        cur = lr * min(1.0, step / warm) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = cur
        x = next(it)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, m = sae.loss(x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        sae.remove_parallel_grad()
        opt.step()
        sae.normalize_decoder()

        if step % 50 == 0:
            dead_train = 100.0 * (sae.since_fired > dead_after).float().mean().item()
            tb.add_scalar("train/mse", m["mse"].item(), step)
            tb.add_scalar("train/aux", m["aux"].item(), step)
            tb.add_scalar("train/dead_pct_train", dead_train, step)
            tb.add_scalar("train/lr", cur, step)
            spm = step / ((time.time() - t0) / 60)
            print(f"step {step}/{steps} mse {m['mse'].item():.4f} "
                  f"aux {m['aux'].item():.4f} dead% {dead_train:.1f} steps/min {spm:.0f}")

        if step % eval_every == 0 or step == steps:
            sae.eval()
            metrics = evaluate(stream.model, sae, eval_seqs, layer,
                               stream.mu, stream.scale, device)
            sae.train()
            for kk, v in metrics.items():
                tb.add_scalar(f"eval/{kk}", v, step)
            print("EVAL", {kk: round(v, 4) for kk, v in metrics.items()})
            save_ckpt(os.path.join(out, "last.pt"), sae, cfg, stream, layer, step, metrics)

    save_ckpt(os.path.join(out, "final.pt"), sae, cfg, stream, layer, steps, metrics)
    print("saved", os.path.join(out, "final.pt"))
    return sae, metrics