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


# ------------------------------------------------------- X3: seed-matched compare

RAW_PERCENT_NOTE = (
    "percent_changed_raw counts every differing token against all generated tokens. It "
    "overstates the direct effect of the intervention: after the first divergence the two "
    "runs continue from different text, so later tokens drift even where the intervention "
    "no longer acts. Use first_divergence_index and tokens_changed instead.")


def compare_generations(baseline, steered):
    """Token-by-token comparison of two generations from the same prompt.

    Only newly generated tokens are compared (the prompt is identical). Indices are
    0-based within the new tokens. A position past the end of the shorter output counts as
    different. See RAW_PERCENT_NOTE for why the raw percent is not the effect size.
    """
    if baseline.token_ids[:baseline.prompt_len] != steered.token_ids[:steered.prompt_len]:
        raise ValueError("the two generations do not share the same prompt")
    b_ids, s_ids = baseline.new_ids, steered.new_ids
    n = max(len(b_ids), len(s_ids))
    div = next((i for i in range(n) if i >= len(b_ids) or i >= len(s_ids) or b_ids[i] != s_ids[i]),
               None)
    b_pieces, s_pieces = baseline.new_pieces, steered.new_pieces
    out = {"prompt_tokens": baseline.prompt_len, "baseline_new_tokens": len(b_ids),
           "steered_new_tokens": len(s_ids), "diverges": div is not None,
           "first_divergence_index": div,
           "percent_changed_raw_note": RAW_PERCENT_NOTE}
    if div is None:
        out.update(identical_prefix_tokens=list(b_pieces), identical_prefix_text="".join(b_pieces),
                   baseline_from_divergence=[], baseline_from_divergence_text="",
                   steered_from_divergence=[], steered_from_divergence_text="",
                   tokens_changed=0, tokens_compared_from_divergence=0,
                   percent_changed_from_divergence=None, percent_changed_raw=0.0)
        return out
    changed = sum(1 for i in range(div, n)
                  if i >= len(b_ids) or i >= len(s_ids) or b_ids[i] != s_ids[i])
    compared = n - div
    out.update(identical_prefix_tokens=list(b_pieces[:div]),
               identical_prefix_text="".join(b_pieces[:div]),
               baseline_from_divergence=list(b_pieces[div:]),
               baseline_from_divergence_text="".join(b_pieces[div:]),
               steered_from_divergence=list(s_pieces[div:]),
               steered_from_divergence_text="".join(s_pieces[div:]),
               tokens_changed=changed, tokens_compared_from_divergence=compared,
               percent_changed_from_divergence=round(100.0 * changed / compared, 1),
               percent_changed_raw=round(100.0 * changed / n, 1))
    return out


def seed_matched_compare(rt, prompt, feature_id, mode, strength, max_activation,
                         max_new_tokens=60, seed=42):
    """X3. Run baseline and steered with the same seed and compare them token by token.

    Returns {feature_id, mode, strength, seed, max_new_tokens, baseline_text, steered_text}
    plus every field of compare_generations.
    """
    base, steered, spec = _run_pair(rt, prompt, feature_id, mode, strength, max_activation,
                                    max_new_tokens, seed)
    out = {"feature_id": feature_id, "mode": mode, "strength": spec.strength, "seed": seed,
           "max_new_tokens": max_new_tokens, "baseline_text": base.text,
           "steered_text": steered.text}
    out.update(compare_generations(base, steered))
    return out
