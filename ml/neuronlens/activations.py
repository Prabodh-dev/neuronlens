"""Streaming GPT-2 activations: hook -> shuffle buffer -> normalised batches."""
import glob
import os
import random

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .modelutil import DEFAULT_MODEL, backbone, d_model_of, get_block


def load_model(device="cuda", name=DEFAULT_MODEL):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).half().to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def token_sequences(tok, data_dir, seq_len=1024, seed=42):
    files = sorted(glob.glob(os.path.join(os.path.expanduser(data_dir), "*.parquet")))
    rng = random.Random(seed)
    buf = []
    while True:
        rng.shuffle(files)
        for f in files:
            pf = pq.ParquetFile(f)
            groups = list(range(pf.num_row_groups))
            rng.shuffle(groups)
            for g in groups:
                texts = pf.read_row_group(g, columns=["text"]).column("text").to_pylist()
                rng.shuffle(texts)
                for t in texts:
                    buf.extend(tok.encode(t) + [tok.eos_token_id])
                    while len(buf) >= seq_len:
                        yield torch.tensor(buf[:seq_len])
                        buf = buf[seq_len:]


class Collector:
    """Forward hook that grabs the residual stream after block `layer`."""

    def __init__(self, model, layer):
        self.out = None
        get_block(model, layer).register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        self.out = h.detach()


@torch.no_grad()
def raw_activations(model, seqs, layer, device="cuda", seq_batch=16):
    col = Collector(model, layer)
    batch = []
    for s in seqs:
        batch.append(s)
        if len(batch) == seq_batch:
            ids = torch.stack(batch).to(device)
            batch = []
            backbone(model)(ids)
            a = col.out[:, 1:, :]  # drop position 0 (outlier)
            yield a.reshape(-1, a.shape[-1]).cpu()


class ActivationStream:
    def __init__(self, layer=8, buffer_size=500_000, batch_size=4096,
                 data_dir="~/data/fineweb/sample/10BT",
                 seed=42, device="cuda", model_name=DEFAULT_MODEL):
        self.device, self.batch_size = device, batch_size
        self.model, self.tok = load_model(device, model_name)
        seqs = token_sequences(self.tok, data_dir, seed=seed)
        self.src = raw_activations(self.model, seqs, layer, device)
        d = d_model_of(self.model)
        self.buf = torch.empty(buffer_size, d, dtype=torch.float16)
        n = 0
        while n < buffer_size:
            c = next(self.src)[: buffer_size - n]
            self.buf[n:n + len(c)] = c
            n += len(c)
        sample = self.buf[:100_000].float()
        self.mu = sample.mean(0)
        self.scale = (d ** 0.5) / (sample - self.mu).norm(dim=1).mean()
        self.mu_d, self.scale_d = self.mu.to(device), self.scale.to(device)

    def _refill(self, idx):
        got, parts = 0, []
        while got < len(idx):
            c = next(self.src)
            parts.append(c)
            got += len(c)
        self.buf[idx] = torch.cat(parts)[: len(idx)]

    def __iter__(self):
        N = len(self.buf)
        while True:
            idx = torch.randperm(N)[: N // 2]
            out = self.buf[idx].clone()
            for i in range(0, len(out) - self.batch_size + 1, self.batch_size):
                x = out[i:i + self.batch_size].to(self.device).float()
                yield (x - self.mu_d) * self.scale_d
            self._refill(idx)