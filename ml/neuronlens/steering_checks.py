"""Steering checks for the Phase 4 endpoints (X1 to X4).

Steering is a test of what a feature does, not a fix. Every function here runs seeded
generation: the baseline and the steered run use the same seed, so a difference between
them comes from the intervention. The steering maths is unchanged (steering.make_hook):
amplify adds strength x max_activation x decoder direction at every position except the
first; ablate subtracts the feature's own activation.

The functions take a runtime object with `generate(...)` (neuronlens.runtime.Runtime) and
an `d_sae` attribute, so they can be tested without a GPU.
"""
import math
from dataclasses import dataclass

# ---- input limits (the ML side accepts a wider strength range than the current API) ----
STRENGTH_MAX = 10.0       # amplify strength, in units of the feature's max activation
MAX_NEW_TOKENS = 120      # same cap as the API (SteerRequest.max_new_tokens)
SEED_MAX = 2147483647     # same cap as the API
MODES = ("amplify", "ablate")


class SteeringCheckError(ValueError):
    """Bad input. `code` uses the API's error codes: INVALID_INPUT or FEATURE_NOT_FOUND."""

    def __init__(self, message, code="INVALID_INPUT"):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SteeringSpec:
    feature_id: int
    mode: str
    strength: float        # ignored (and reported as 0.0) for ablate
    max_activation: float  # the feature's max_activation from the database


@dataclass(frozen=True)
class Generation:
    """One seeded generation. token_ids and pieces cover prompt + new tokens, in order."""
    text: str              # full decoded text, prompt included (what /steer returns)
    token_ids: list
    prompt_len: int
    pieces: list           # one display string per token (textdecode.token_pieces)

    @property
    def new_ids(self):
        return self.token_ids[self.prompt_len:]

    @property
    def new_pieces(self):
        return self.pieces[self.prompt_len:]

    @property
    def continuation(self):
        """The newly generated text only."""
        return "".join(self.new_pieces)


def validate(rt, prompt, max_new_tokens, seed, feature_id=None, mode=None, strength=None,
             max_activation=None):
    if not isinstance(prompt, str) or not prompt.strip():
        raise SteeringCheckError("prompt must not be empty")
    if not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
        raise SteeringCheckError(f"max_new_tokens must be an integer from 1 to {MAX_NEW_TOKENS}")
    if not isinstance(seed, int) or not 0 <= seed <= SEED_MAX:
        raise SteeringCheckError(f"seed must be an integer from 0 to {SEED_MAX}")
    if feature_id is None:
        return
    d_sae = getattr(rt, "d_sae", None)
    if not isinstance(feature_id, int) or feature_id < 0 or (d_sae is not None and feature_id >= d_sae):
        raise SteeringCheckError(f"feature {feature_id} not found", code="FEATURE_NOT_FOUND")
    if mode not in MODES:
        raise SteeringCheckError(f"mode must be one of {', '.join(MODES)}")
    if mode == "amplify":
        if strength is None or not math.isfinite(strength) or not 0 <= strength <= STRENGTH_MAX:
            raise SteeringCheckError(f"strength must be a number from 0 to {STRENGTH_MAX:g}")
    if max_activation is None or not math.isfinite(max_activation) or max_activation < 0:
        raise SteeringCheckError("max_activation must be a non-negative number")
    if mode == "amplify" and max_activation == 0:
        raise SteeringCheckError("feature has max_activation 0 (dead feature); amplify would do nothing")


def _spec(feature_id, mode, strength, max_activation):
    return SteeringSpec(feature_id, mode, float(strength) if mode == "amplify" else 0.0,
                        float(max_activation))


def _run_pair(rt, prompt, feature_id, mode, strength, max_activation, max_new_tokens, seed):
    validate(rt, prompt, max_new_tokens, seed, feature_id, mode, strength, max_activation)
    spec = _spec(feature_id, mode, strength, max_activation)
    base = rt.generate(prompt, max_new_tokens, seed, None)
    steered = rt.generate(prompt, max_new_tokens, seed, spec)
    return base, steered, spec


# ---------------------------------------------------------------- X1: steering check

def steering_check(rt, prompt, feature_id, mode, strength, max_activation,
                   max_new_tokens=60, seed=42):
    """X1. Baseline and steered text for one feature, same seed, and whether they differ.

    Returns {feature_id, mode, strength, seed, max_new_tokens, baseline_text, steered_text,
    differs}. `differs` is True when the two full texts are not identical. For ablate,
    `strength` is reported as 0.0 because it is ignored.
    """
    base, steered, spec = _run_pair(rt, prompt, feature_id, mode, strength, max_activation,
                                    max_new_tokens, seed)
    return {"feature_id": feature_id, "mode": mode, "strength": spec.strength, "seed": seed,
            "max_new_tokens": max_new_tokens, "baseline_text": base.text,
            "steered_text": steered.text, "differs": base.text != steered.text}
