import os

import torch

from .activations import Collector, load_model
from .modelutil import DEFAULT_MODEL, backbone, get_block, model_name_from_ck
from .interpret import load_sae
from .steering import generate, generate_ids, make_hook, make_multi_hook
from .steering_checks import Generation, MultiSteeringSpec
from .textdecode import token_pieces


class Runtime:
    """Loads the frozen model and the trained SAE once; answers live requests."""

    def __init__(self, model, tok, sae, ck, device, model_name=None):
        self.model, self.tok, self.sae, self.device = model, tok, sae, device
        self.model_name = model_name or model_name_from_ck(ck)
        self.hook_layer = ck["layer"]
        self.d_model, self.d_sae, self.k = ck["d_model"], ck["d_sae"], ck["k"]
        self.mu = ck["mu"].to(device)
        self.scale = ck["scale"].to(device)
        self.col = Collector(model, self.hook_layer)

    @classmethod
    def load(cls, checkpoint_path, device="cuda", model_name=None):
        """model_name: Hugging Face name; if not given, NEURONLENS_MODEL_NAME, then the name the
        checkpoint was trained on (gpt2 for every existing checkpoint)."""
        sae, ck = load_sae(os.path.expanduser(checkpoint_path), device)
        name = (model_name or os.environ.get("NEURONLENS_MODEL_NAME", "").strip()
                or model_name_from_ck(ck))
        model, tok = load_model(device, name)
        return cls(model, tok, sae, ck, device, name)

    def count_tokens(self, text):
        return len(self.tok.encode(text))

    def _token_acts(self, ids):
        """SAE activations (T, d_sae) for ids of shape (1, T): one forward pass, then the SAE.
        Shared by inspect() and feature_activations() so both read the same numbers."""
        backbone(self.model)(ids)
        h = self.col.out[0].float()  # (T, d)
        acts, _ = self.sae.encode((h - self.mu) * self.scale)
        return acts

    @torch.no_grad()
    def feature_activations(self, token_ids, feature_id):
        """One feature's activation at every position of a finished token sequence, read with
        no steering hook attached (same path as inspect). Position 0 is reported as 0.0, as
        inspect() reports no features there. Token ids are used as they are: nothing is
        re-tokenised, so positions line up with the generated tokens."""
        block = get_block(self.model, self.hook_layer)
        extra = [h for h in getattr(block, "_forward_hooks", {}).values()
                 if getattr(h, "__self__", None) is not self.col]
        if extra:
            raise RuntimeError("a forward hook other than the activation collector is attached "
                               "to the hooked block; refusing to read activations")
        ids = torch.tensor([list(token_ids)[:1024]], device=self.device)
        acts = self._token_acts(ids)[:, feature_id].cpu().tolist()
        if acts:
            acts[0] = 0.0
        return acts

    @torch.no_grad()
    def inspect(self, text, top_n):
        ids = self.tok(text, return_tensors="pt").input_ids[:, :1024].to(self.device)
        acts = self._token_acts(ids)
        vals, idx = acts.topk(min(top_n, self.d_sae), dim=-1)
        vals, idx = vals.cpu().tolist(), idx.cpu().tolist()
        # decode the whole text as one byte stream: tokens can split a character (e.g. a
        # curly apostrophe), so decoding each id alone gives U+FFFD. One string per token.
        pieces = token_pieces(self.tok, ids[0].tolist())
        out = []
        for t in range(ids.shape[1]):
            feats = [] if t == 0 else [  # position 0 is an outlier; SAE not trained on it
                {"feature_id": i, "activation": v}
                for v, i in zip(vals[t], idx[t]) if v > 0]
            out.append({"index": t, "text": pieces[t],
                        "features": feats})
        return out

    def generate(self, prompt, max_new_tokens, seed, steering=None):
        """One seeded generation, optionally steered (a steering_checks.SteeringSpec, or a
        MultiSteeringSpec for several amplify steers at once).
        Used by the Phase 4 checks; the steering hook is the same as in steer()."""
        hook = None
        if isinstance(steering, MultiSteeringSpec):   # several amplify steers in one hook
            hook = make_multi_hook(self.sae, [(f, s, m) for f, s, m in steering.steers],
                                   self.mu, self.scale)
        elif steering is not None:
            hook = make_hook(self.sae, steering.feature_id, steering.mode, steering.strength,
                             steering.max_activation, self.mu, self.scale)
        ids, prompt_len = generate_ids(self.model, self.tok, prompt, max_new_tokens, seed,
                                       hook, self.hook_layer, self.device)
        return Generation(self.tok.decode(ids, skip_special_tokens=True), ids, prompt_len,
                          token_pieces(self.tok, ids))

    def steer(self, prompt, feature_id, mode, strength, max_activation,
              max_new_tokens, seed):
        base = generate(self.model, self.tok, prompt, max_new_tokens, seed,
                        None, self.hook_layer, self.device)
        hook = make_hook(self.sae, feature_id, mode, strength, max_activation,
                         self.mu, self.scale)
        steered = generate(self.model, self.tok, prompt, max_new_tokens, seed,
                           hook, self.hook_layer, self.device)
        return {"baseline_text": base, "steered_text": steered}