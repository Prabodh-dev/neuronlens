import math
import os
import re
import time
import itertools

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from .activations import ActivationStream, token_sequences
from .evals import evaluate
from .modelutil import DEFAULT_MODEL, d_model_of, n_layers, slug
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


def apply_overrides(cfg, model=None, layer=None, tokens=None):
    """Command-line overrides on a loaded config; with none of them cfg is returned unchanged.

    model: Hugging Face name. Sets model_name and renames the run to <slug>-l<layer>-k<k>
           (EleutherAI/gpt-neo-125M, layer 8, k 32 -> gpt-neo-125m-l8-k32). Giving the model
           the config already has changes nothing.
    layer: hook after block N. Renames the run the same way (gpt2-l8-k32 -> gpt2-l3-k32).
    tokens: token budget, i.e. n_train_activations (one activation per scored token).
    """
    if model is None and layer is None and tokens is None:
        return cfg
    cfg = dict(cfg)
    new_model = model is not None and model != cfg.get("model_name", DEFAULT_MODEL)
    if layer is not None:
        layer = int(layer)
        if layer < 0:
            raise ValueError(f"layer must be 0 or more, got {layer}")
        cfg.pop("layer", None)
        cfg["hook_layer"] = layer
    if new_model:
        cfg["model_name"] = model
    if new_model or layer is not None:
        cur = int(pick(cfg, ["hook_layer", "layer"], 8))
        k = int(pick(cfg, ["k", "topk"], 32))
        if new_model:
            cfg["run_id"] = f"{slug(cfg['model_name'])}-l{cur}-k{k}"
        else:
            run_id = str(cfg["run_id"])
            if re.search(r"-l\d+-", run_id):
                cfg["run_id"] = re.sub(r"-l\d+-", f"-l{cur}-", run_id, count=1)
            else:
                cfg["run_id"] = f"{run_id}-l{cur}"
    if tokens is not None:
        if int(tokens) <= 0:
            raise ValueError("tokens must be positive")
        cfg["n_train_activations"] = int(tokens)
    return cfg


def budget_for(rate, minutes, batch_size):
    """Token budget that fits in `minutes` at `rate` tokens per second, rounded down to a whole
    number of batches (training uses total // batch_size steps)."""
    return int(rate * 60 * minutes) // batch_size * batch_size


def throughput(step_fn, seconds, batch_size, warmup_steps=20, clock=time.perf_counter):
    """Call step_fn() repeatedly; after warmup_steps untimed calls, time calls for `seconds`.
    Returns (tokens per second, timed steps). One step trains on batch_size tokens."""
    for _ in range(warmup_steps):
        step_fn()
    t0 = clock()
    steps = 0
    while True:
        step_fn()
        steps += 1
        elapsed = clock() - t0
        if elapsed >= seconds:
            return steps * batch_size / elapsed, steps


def measure(config_path, device="cuda", out_root=None, seconds=120, minutes=None,
            model=None, layer=None):
    """Throughput run: the same data stream, SAE and update step as train(), for about
    `seconds`, nothing saved. Prints tokens per second and, with `minutes`, the token budget
    that fits (evaluation time is not included, so allow a little extra)."""
    cfg = apply_overrides(load_cfg(config_path), model, layer)
    d_sae = int(pick(cfg, ["d_sae"]))
    k = int(pick(cfg, ["k", "topk"]))
    buf = int(pick(cfg, ["shuffle_buffer", "buffer_size"]))
    bs = int(pick(cfg, ["batch_size", "batch"], 4096))
    lr = float(pick(cfg, ["lr", "learning_rate"], 3e-4))
    aux_k = int(pick(cfg, ["aux_k"], 512))
    aux_coef = float(pick(cfg, ["aux_coef"], 1 / 32))
    seed = int(pick(cfg, ["seed"], 42))
    layer = int(pick(cfg, ["hook_layer", "layer"], 8))
    dead_after = int(pick(cfg, ["dead_steps_threshold"], 500))
    data_dir = pick(cfg, ["data_dir"], "~/data/fineweb/sample/10BT")
    model_name = pick(cfg, ["model_name"], DEFAULT_MODEL)

    torch.manual_seed(seed)
    stream = ActivationStream(layer, buf, bs, data_dir, seed, device, model_name)
    sae = TopKSAE(d_model_of(stream.model), d_sae, k, aux_k, aux_coef, dead_after).to(device)
    it = iter(stream)
    sae.init_b_dec(next(it))
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    def step_fn():
        x = next(it)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = sae.loss(x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        sae.remove_parallel_grad()
        opt.step()
        sae.normalize_decoder()
        if device == "cuda":
            torch.cuda.synchronize()

    print(f"measuring {model_name}, layer {layer}, batch {bs}: about {seconds} s of training steps")
    rate, steps = throughput(step_fn, seconds, bs)
    print(f"throughput: {rate:,.0f} tokens/sec ({steps} steps timed)")
    if minutes is not None:
        n = budget_for(rate, minutes, bs)
        print(f"budget that fits in {minutes:g} minutes: {n:,} tokens  ->  --tokens {n}")
    return rate


def train(config_path, device="cuda", out_root="~/runs", model=None, layer=None, tokens=None):
    cfg = apply_overrides(load_cfg(config_path), model, layer, tokens)
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
    model_name = pick(cfg, ["model_name"], DEFAULT_MODEL)

    steps = total // bs
    warm = max(1, steps // 100)
    eval_every = max(steps // 10, 200)
    out = os.path.join(os.path.expanduser(out_root), run_id)
    os.makedirs(out, exist_ok=True)
    tb = SummaryWriter(out)
    torch.manual_seed(seed)

    stream = ActivationStream(layer, buf, bs, data_dir, seed, device, model_name)
    if layer >= n_layers(stream.model):
        raise ValueError(f"{model_name} has {n_layers(stream.model)} blocks; layer {layer} does not exist")
    d_model = d_model_of(stream.model)
    if int(pick(cfg, ["d_model"], d_model)) != d_model:
        raise ValueError(f"config says d_model {cfg['d_model']} but {model_name} has {d_model}")
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