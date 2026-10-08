import os

import torch

from .activations import Collector, load_model
from .interpret import load_sae
from .steering import generate, make_hook


class Runtime:
    """Loads the frozen model and the trained SAE once; answers live requests."""

    def __init__(self, model, tok, sae, ck, device):
        self.model, self.tok, self.sae, self.device = model, tok, sae, device
        self.model_name = "gpt2"
        self.hook_layer = ck["layer"]
        self.d_model, self.d_sae, self.k = ck["d_model"], ck["d_sae"], ck["k"]
        self.mu = ck["mu"].to(device)
        self.scale = ck["scale"].to(device)
        self.col = Collector(model, self.hook_layer)

    @classmethod
    def load(cls, checkpoint_path, device="cuda"):
        sae, ck = load_sae(os.path.expanduser(checkpoint_path), device)
        model, tok = load_model(device)
        return cls(model, tok, sae, ck, device)

    def count_tokens(self, text):
        return len(self.tok.encode(text))

    @torch.no_grad()
    def inspect(self, text, top_n):
        ids = self.tok(text, return_tensors="pt").input_ids[:, :1024].to(self.device)
        self.model.transformer(ids)
        h = self.col.out[0].float()  # (T, d)
        acts, _ = self.sae.encode((h - self.mu) * self.scale)
        vals, idx = acts.topk(min(top_n, self.d_sae), dim=-1)
        vals, idx = vals.cpu().tolist(), idx.cpu().tolist()
        out = []
        for t in range(ids.shape[1]):
            feats = [] if t == 0 else [  # position 0 is an outlier; SAE not trained on it
                {"feature_id": i, "activation": v}
                for v, i in zip(vals[t], idx[t]) if v > 0]
            out.append({"index": t, "text": self.tok.decode([int(ids[0, t])]),
                        "features": feats})
        return out

    def steer(self, prompt, feature_id, mode, strength, max_activation,
              max_new_tokens, seed):
        base = generate(self.model, self.tok, prompt, max_new_tokens, seed,
                        None, self.hook_layer, self.device)
        hook = make_hook(self.sae, feature_id, mode, strength, max_activation,
                         self.mu, self.scale)
        steered = generate(self.model, self.tok, prompt, max_new_tokens, seed,
                           hook, self.hook_layer, self.device)
        return {"baseline_text": base, "steered_text": steered}