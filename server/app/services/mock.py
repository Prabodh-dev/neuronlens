"""Mock Runtime: same interface as ml/neuronlens/runtime.py, answered from the fixture DB.

Everything here is deterministic (same input, same output) and synthetic. Steering text is
marked [simulated] so it can never be mistaken for real model output.
"""
import hashlib
import random
import re

from ..db import Database

# Rough GPT-2-style split: words keep their leading space, punctuation stands alone.
_TOKEN_RE = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+")

_FILLER = [" the", " day", " went", " on", ",", " and", " people", " said", " that", " it",
           " was", " a", " new", " idea", " for", " the", " town", ".", " Later", " they"]


def _seed(*parts) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


class MockRuntime:
    is_simulated = True

    def __init__(self, db: Database):
        self.db = db
        self._live = [(r["feature_id"], r["max_activation"]) for r in db.query(
            "SELECT feature_id, max_activation FROM features WHERE is_dead = 0 "
            "ORDER BY feature_id")]

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
        base_words = self._continuation(seed, max_new_tokens)
        steered_words = list(base_words)
        if mode == "amplify" and strength > 0:
            rng = random.Random(_seed("steer", feature_id, strength, seed))
            marker = f" [feature {feature_id}]"
            share = min(1.0, strength / 3)
            steered_words = [marker if rng.random() < share else w for w in base_words]
        elif mode == "ablate":
            rng = random.Random(_seed("ablate", feature_id, seed))
            steered_words = [w for w in base_words if rng.random() > 0.15] or base_words[:1]
        return {"baseline_text": f"{prompt}[simulated]{''.join(base_words)}",
                "steered_text": f"{prompt}[simulated]{''.join(steered_words)}"}

    @staticmethod
    def _continuation(seed: int, n: int) -> list[str]:
        rng = random.Random(_seed("base", seed))
        return [rng.choice(_FILLER) for _ in range(min(n, 40))]
