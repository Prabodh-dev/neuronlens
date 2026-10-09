import pytest

from neuronlens import steering_checks as sc
from neuronlens.steering_checks import Generation, SteeringCheckError


class FakeRuntime:
    """Deterministic stand-in for Runtime. Tokens are single characters; the baseline is a
    fixed function of (prompt, seed). A steering spec changes the output from token index
    `div` on, where div depends on the strength, mimicking drift after a first divergence."""

    d_sae = 100

    def __init__(self, baseline_len=20, diverge_at=5, steered_ids=None, base_ids=None):
        self.calls = []
        self.baseline_len, self.diverge_at = baseline_len, diverge_at
        self.steered_ids, self.base_ids = steered_ids, base_ids

    def generate(self, prompt, max_new_tokens, seed, steering=None):
        self.calls.append((prompt, max_new_tokens, seed, steering))
        prompt_ids = [ord(c) for c in prompt]
        base = self.base_ids if self.base_ids is not None else \
            [97 + (seed + i) % 26 for i in range(min(self.baseline_len, max_new_tokens))]
        new = list(base)
        if steering is not None and steering.strength != 0.0 or (
                steering is not None and steering.mode == "ablate"):
            if self.steered_ids is not None:
                new = list(self.steered_ids)
            else:
                new = base[:self.diverge_at] + [65 + (i % 26) for i in range(len(base) - self.diverge_at)]
        ids = prompt_ids + new
        pieces = [chr(i) for i in ids]
        return Generation("".join(pieces), ids, len(prompt_ids), pieces)


# ----------------------------------------------------------------------------- X1

def test_x1_returns_texts_and_flag():
    rt = FakeRuntime()
    r = sc.steering_check(rt, "Hello ", 7, "amplify", 3.0, 10.0, max_new_tokens=20, seed=5)
    assert set(r) == {"feature_id", "mode", "strength", "seed", "max_new_tokens",
                      "baseline_text", "steered_text", "differs"}
    assert r["differs"] is True and r["baseline_text"].startswith("Hello ")
    assert r["baseline_text"] != r["steered_text"]
    assert r["strength"] == 3.0 and r["seed"] == 5


def test_x1_same_seed_for_both_runs_and_spec_passed():
    rt = FakeRuntime()
    sc.steering_check(rt, "Hi", 7, "amplify", 2.0, 9.5, seed=11)
    (_, _, s0, spec0), (_, _, s1, spec1) = rt.calls
    assert s0 == s1 == 11 and spec0 is None
    assert (spec1.feature_id, spec1.mode, spec1.strength, spec1.max_activation) == (7, "amplify", 2.0, 9.5)


def test_x1_strength_zero_matches_baseline_and_ablate_ignores_strength():
    rt = FakeRuntime()
    assert sc.steering_check(rt, "Hi", 7, "amplify", 0, 9.5)["differs"] is False
    r = sc.steering_check(rt, "Hi", 7, "ablate", 3.0, 9.5)
    assert r["strength"] == 0.0 and rt.calls[-1][3].strength == 0.0 and r["differs"] is True


@pytest.mark.parametrize("kwargs,code", [
    (dict(prompt="  "), "INVALID_INPUT"),
    (dict(mode="clamp"), "INVALID_INPUT"),
    (dict(strength=-1), "INVALID_INPUT"),
    (dict(strength=sc.STRENGTH_MAX + 1), "INVALID_INPUT"),
    (dict(strength=float("nan")), "INVALID_INPUT"),
    (dict(max_new_tokens=0), "INVALID_INPUT"),
    (dict(max_new_tokens=sc.MAX_NEW_TOKENS + 1), "INVALID_INPUT"),
    (dict(seed=-1), "INVALID_INPUT"),
    (dict(max_activation=0.0), "INVALID_INPUT"),
    (dict(feature_id=100), "FEATURE_NOT_FOUND"),
    (dict(feature_id=-1), "FEATURE_NOT_FOUND"),
])
def test_x1_errors(kwargs, code):
    args = dict(prompt="Hi", feature_id=7, mode="amplify", strength=1.0, max_activation=5.0,
                max_new_tokens=20, seed=1)
    args.update(kwargs)
    with pytest.raises(SteeringCheckError) as e:
        sc.steering_check(FakeRuntime(), **args)
    assert e.value.code == code


def test_x1_ablate_accepts_zero_max_activation_check_only_for_amplify():
    # ablate does not use max_activation, so only amplify rejects 0
    assert sc.steering_check(FakeRuntime(), "Hi", 7, "ablate", 1, 0.0)["mode"] == "ablate"


# ------------------------------------------------------------- Runtime.generate wiring

def test_runtime_generate_uses_same_hook_and_seed(monkeypatch):
    from types import SimpleNamespace
    from neuronlens import runtime as rtmod
    from neuronlens.textdecode import token_pieces
    from tests.test_textdecode import ByteTok

    class DecodingTok(ByteTok):
        def decode(self, ids, skip_special_tokens=True):
            return "".join(token_pieces(self, ids))

    seen = {}
    monkeypatch.setattr(rtmod, "make_hook", lambda *a: seen.setdefault("hook_args", a) and "HOOK")

    def fake_generate_ids(model, tok, prompt, n, seed, hook, layer, device):
        seen["gen"] = (prompt, n, seed, hook, layer, device)
        return [72, 105, 0xE2, 0x80, 0x99], 2

    monkeypatch.setattr(rtmod, "generate_ids", fake_generate_ids)
    rt = rtmod.Runtime.__new__(rtmod.Runtime)
    rt.tok, rt.sae, rt.model, rt.device = DecodingTok(), "SAE", "MODEL", "cpu"
    rt.hook_layer, rt.mu, rt.scale = 8, "MU", "SCALE"
    g = rt.generate("Hi", 5, 3, sc.SteeringSpec(7, "amplify", 2.0, 9.0))
    assert seen["hook_args"] == ("SAE", 7, "amplify", 2.0, 9.0, "MU", "SCALE")
    assert seen["gen"] == ("Hi", 5, 3, "HOOK", 8, "cpu")
    assert g.prompt_len == 2 and len(g.pieces) == 5 and g.continuation == "’"
    base = rt.generate("Hi", 5, 3, None)
    assert seen["gen"][3] is None and base.token_ids == g.token_ids
