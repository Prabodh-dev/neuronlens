"""Mock Runtime: same interface as ml/neuronlens/runtime.py, answered from the fixture DB.

Everything here is deterministic (same input, same output) and synthetic. Steering text is
marked [simulated] so it can never be mistaken for real model output.

Generation (used by /steer and the Phase 4 checks) returns the real outputs recorded in
demo/phase4_examples.json when prompt, seed, max_new_tokens and the steering match one of
them, still marked [simulated]; anything else gets a filler stand-in. Tokens here are a
rough word split, not GPT-2 tokens, so counts and divergence indices differ from real mode.
Feature activations (X1) follow the same rule: the recorded max and mean for a recorded
text, a deterministic stand-in otherwise.
"""
import contextlib
import hashlib
from collections import OrderedDict
import json
import random
import re

from ..config import REPO_ROOT
from ..db import Database

# Rough GPT-2-style split: words keep their leading space, punctuation stands alone.
_TOKEN_RE = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+")

_FILLER = [" the", " day", " went", " on", ",", " and", " people", " said", " that", " it",
           " was", " a", " new", " idea", " for", " the", " town", ".", " Later", " they"]

SIMULATED_TAG = "[simulated]"
EXAMPLES_PATH = REPO_ROOT / "demo" / "phase4_examples.json"
VOCAB = 50257  # stand-in token ids stay in GPT-2's id range


def _seed(*parts) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def load_examples(path=EXAMPLES_PATH) -> dict:
    """{(prompt, seed, max_new_tokens): {"baseline": text, (feature, mode, strength): text,
    ("acts", feature, "baseline" | (feature, mode, strength)): (max, mean)}}"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for ex in data.get("examples", []):
        key = (ex["prompt"], ex["seed"], ex["max_new_tokens"])
        entry = out.setdefault(key, {"baseline": ex["baseline"]})
        strength = float(ex["strength"]) if ex["mode"] == "amplify" else 0.0
        spec = (ex["feature_id"], ex["mode"], strength)
        entry[spec] = ex["steered"]
        if ex.get("target_activation_baseline") is not None:
            entry[("acts", ex["feature_id"], "baseline")] = (
                ex["target_activation_baseline"], ex["target_activation_baseline_mean"])
            entry[("acts", ex["feature_id"], spec)] = (
                ex["target_activation_steered"], ex["target_activation_steered_mean"])
        for r in (ex.get("sweep") or {}).get("results", []):
            entry.setdefault((ex["feature_id"], "amplify", float(r["strength"])), r["text"])
    return out


class MockRuntime:
    is_simulated = True

    def __init__(self, db: Database, examples: dict | None = None):
        self.db = db
        self._live = [(r["feature_id"], r["max_activation"]) for r in db.query(
            "SELECT feature_id, max_activation FROM features WHERE is_dead = 0 "
            "ORDER BY feature_id")]
        self.d_sae = db.one("SELECT COUNT(*) AS n FROM features")["n"]
        self._examples = load_examples() if examples is None else examples
        self._generated = OrderedDict()  # token ids -> how they were made, for X1 activations

    def count_tokens(self, text: str) -> int:
        return len(tokenize(text))

    def inspect(self, text: str, top_n: int) -> list[dict]:
        out = []
        for i, tok in enumerate(tokenize(text)):
            feats = []
            if i > 0 and self._live:  # position 0 is an outlier, as in the real Runtime
                rng = random.Random(_seed("inspect", tok))
                picks = rng.sample(self._live, min(top_n, len(self._live)))
                feats = [{"feature_id": fid, "activation": round(m * rng.uniform(0.2, 1.0), 3)}
                         for fid, m in picks]
                feats.sort(key=lambda f: (-f["activation"], f["feature_id"]))
            out.append({"index": i, "text": tok, "features": feats})
        return out

    def steer(self, prompt, feature_id, mode, strength, max_activation, max_new_tokens, seed):
        return {"baseline_text": self._text(prompt, max_new_tokens, seed),
                "steered_text": self._text(prompt, max_new_tokens, seed,
                                           (feature_id, mode, strength))}

    # ---- used by the Phase 4 checks (ml/neuronlens/steering_checks.py) ----

    def generate(self, prompt, max_new_tokens, seed, steering=None):
        """A steering_checks.Generation. The [simulated] tag is glued to the last prompt
        piece, so it shows in every text but is not counted as a generated token."""
        from .checks import steering_checks  # noqa: PLC0415
        spec = None if steering is None else (steering.feature_id, steering.mode,
                                              steering.strength)
        prompt_pieces = tokenize(prompt) or [prompt]
        new_pieces = tokenize(self._continuation_text(prompt, max_new_tokens, seed, spec))
        pieces = prompt_pieces[:-1] + [prompt_pieces[-1] + SIMULATED_TAG] + new_pieces
        ids = [_seed("tok", p) % VOCAB for p in prompt_pieces + new_pieces]
        self._remember(ids, (prompt, seed, max_new_tokens), spec,
                       None if steering is None else steering.max_activation)
        return steering_checks().Generation("".join(pieces), ids, len(prompt_pieces), pieces)

    def feature_activations(self, token_ids, feature_id):
        """Per-position activation of one feature on a text this runtime generated: the
        recorded max (and mean) for a recorded example, else a stand-in where amplify
        raises the feature in proportion to strength. Position 0 and the prompt are 0.0."""
        n = len(token_ids)
        made = self._generated.get(tuple(token_ids))
        if made is None or n == 0:
            return [0.0] * n
        key, spec, max_activation, prompt_len = made
        recorded = self._examples.get(key, {})
        stats = recorded.get(("acts", feature_id, "baseline" if spec is None else spec))
        if stats is None:
            stats = (0.0, 0.0)  # stand-in: the feature is quiet on unsteered filler
            if spec is not None and spec[0] == feature_id and spec[1] == "amplify" and spec[2] > 0:
                peak = round(0.2 * spec[2] * (max_activation or 0.0), 4)
                stats = (peak, peak / 4)
        peak, mean = stats
        first = max(prompt_len, 1)
        new = n - first
        acts = [0.0] * n
        if new <= 0 or peak <= 0:
            return acts
        rest = max(0.0, (mean * new - peak) / (new - 1)) if new > 1 else 0.0
        for i in range(first, n):
            acts[i] = min(rest, peak)
        acts[first] = peak
        return acts

    @staticmethod
    def exclusive():
        return contextlib.nullcontext()

    def _remember(self, ids, key, spec, max_activation):
        prompt_len = len(tokenize(key[0]) or [key[0]])
        self._generated[tuple(ids)] = (key, spec, max_activation, prompt_len)
        while len(self._generated) > 256:
            self._generated.popitem(last=False)

    # ---- internals ----

    def _text(self, prompt, max_new_tokens, seed, spec=None) -> str:
        return f"{prompt}{SIMULATED_TAG}{self._continuation_text(prompt, max_new_tokens, seed, spec)}"

    def _continuation_text(self, prompt, max_new_tokens, seed, spec) -> str:
        """New text only. spec is None (baseline) or (feature_id, mode, strength)."""
        if spec is not None:
            spec = (spec[0], spec[1], float(spec[2]) if spec[1] == "amplify" else 0.0)
        recorded = self._examples.get((prompt, seed, max_new_tokens))
        if recorded is not None:
            text = recorded["baseline"] if spec is None or (spec[1] == "amplify" and
                                                            spec[2] == 0) else recorded.get(spec)
            if text is not None and text.startswith(prompt):
                return text[len(prompt):]
        base_words = self._filler(seed, max_new_tokens)
        if spec is None:
            return "".join(base_words)
        feature_id, mode, strength = spec
        steered_words = list(base_words)
        if mode == "amplify" and strength > 0:
            rng = random.Random(_seed("steer", feature_id, strength, seed))
            marker = f" [feature {feature_id}]"
            share = min(1.0, strength / 3)
            steered_words = [marker if rng.random() < share else w for w in base_words]
        elif mode == "ablate":
            rng = random.Random(_seed("ablate", feature_id, seed))
            steered_words = [w for w in base_words if rng.random() > 0.15] or base_words[:1]
        return "".join(steered_words)

    @staticmethod
    def _filler(seed: int, n: int) -> list[str]:
        rng = random.Random(_seed("base", seed))
        return [rng.choice(_FILLER) for _ in range(min(n, 40))]
