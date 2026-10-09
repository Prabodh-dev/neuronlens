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


# ------------------------------------------------- X2: strength sweep + degeneration

DEFAULT_SWEEP_STRENGTHS = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0)
MAX_SWEEP_POINTS = 12     # most strengths per sweep (each one is a full generation)

# Degeneration rule. An output is marked degenerate when ANY of these holds, judged on the
# newly generated text only (never the prompt). The limits are heuristics chosen by reading
# the failure modes of over-steered output; they have NOT been calibrated on real GPT-2
# outputs yet. phase4_demo.py prints the measured values so they can be adjusted.
MIN_TOKENS_FOR_RULES = 8             # fewer new tokens: the repeat and word rules are skipped
REPEAT_UNIGRAM_FRACTION = 0.6        # share of new tokens that repeat an earlier token (1 - distinct/total)
REPEAT_TRIGRAM_FRACTION = 0.3        # share of token trigrams that repeat an earlier trigram (loops)
MIN_TRIGRAMS_FOR_RULE = 6            # trigram rule needs at least this many trigrams
NON_WORD_FRACTION = 0.5              # share of whitespace-separated chunks that are not words
MIN_CHUNKS_FOR_RULE = 5              # word rule needs at least this many chunks
NON_WORD_LETTER_RATIO = 0.5          # a chunk is a non-word if fewer than this share of its characters are letters or digits
REPLACEMENT_CHAR_FRACTION = 0.02     # U+FFFD share of characters above this: degenerate
REPLACEMENT_CHAR_MIN_COUNT = 3       # ... or this many U+FFFD characters in total
# A single U+FFFD at the very end is ignored: cutting generation at max_new_tokens can end
# in the middle of a multi-byte character.

REPLACEMENT_CHAR = "�"


def _is_word(chunk):
    stripped = chunk.strip("\"'`.,;:!?()[]{}<>-–—…*_/\\|")
    if not stripped or REPLACEMENT_CHAR in stripped:
        return False
    good = sum(1 for c in stripped if c.isalnum() or c in "'’-")
    return good / len(stripped) >= NON_WORD_LETTER_RATIO and any(c.isalnum() for c in stripped)


def assess_degeneration(token_ids, text):
    """Is this generated continuation degenerate? Returns
    {degenerate: bool, reason: str|None, metrics: {...}}.

    token_ids: the newly generated token ids; text: their decoded text. Rules (constants
    above): empty output; mostly repeated tokens (unigram or trigram); mostly non-words;
    U+FFFD above a threshold. `reason` joins every rule that fired, with the measured value.
    """
    n = len(token_ids)
    body = text[:-1] if text.endswith(REPLACEMENT_CHAR) else text  # ignore one trailing U+FFFD
    fffd = body.count(REPLACEMENT_CHAR)
    chunks = text.split()
    words = sum(1 for c in chunks if _is_word(c))
    trigrams = [tuple(token_ids[i:i + 3]) for i in range(n - 2)]
    metrics = {
        "new_tokens": n,
        "repeated_token_fraction": round(1 - len(set(token_ids)) / n, 3) if n else 0.0,
        "repeated_trigram_fraction": round(1 - len(set(trigrams)) / len(trigrams), 3) if trigrams else 0.0,
        "non_word_fraction": round(1 - words / len(chunks), 3) if chunks else 0.0,
        "replacement_chars": fffd,
    }
    reasons = []
    if n == 0 or not text.strip():
        reasons.append("no new text generated")
    if n >= MIN_TOKENS_FOR_RULES:
        if metrics["repeated_token_fraction"] >= REPEAT_UNIGRAM_FRACTION:
            reasons.append(f"mostly repeated tokens ({metrics['repeated_token_fraction']:.0%} "
                           f"repeats, limit {REPEAT_UNIGRAM_FRACTION:.0%})")
        if len(trigrams) >= MIN_TRIGRAMS_FOR_RULE and \
                metrics["repeated_trigram_fraction"] >= REPEAT_TRIGRAM_FRACTION:
            reasons.append(f"repeating loop ({metrics['repeated_trigram_fraction']:.0%} of token "
                           f"trigrams repeat, limit {REPEAT_TRIGRAM_FRACTION:.0%})")
        if len(chunks) >= MIN_CHUNKS_FOR_RULE and metrics["non_word_fraction"] >= NON_WORD_FRACTION:
            reasons.append(f"mostly non-words ({metrics['non_word_fraction']:.0%} of chunks, "
                           f"limit {NON_WORD_FRACTION:.0%})")
    if fffd and (fffd >= REPLACEMENT_CHAR_MIN_COUNT or fffd / max(len(body), 1) > REPLACEMENT_CHAR_FRACTION):
        reasons.append(f"{fffd} replacement characters (U+FFFD)")
    return {"degenerate": bool(reasons), "reason": "; ".join(reasons) or None, "metrics": metrics}


def _assess(gen):
    return assess_degeneration(gen.new_ids, gen.continuation)


def strength_sweep(rt, prompt, feature_id, max_activation, strengths=DEFAULT_SWEEP_STRENGTHS,
                   max_new_tokens=60, seed=42):
    """X2. Amplify one feature at several strengths with the same seed.

    The baseline is generated once. Returns {feature_id, mode: "amplify", seed,
    max_new_tokens, baseline_text, baseline_degenerate, baseline_reason, results: [...]}
    with one result per strength, in the order given:
    {strength, text, differs_from_baseline, first_divergence_index, degenerate, reason, metrics}.
    Strength 0 adds nothing, so it should match the baseline (a control).
    """
    strengths = [float(s) for s in strengths]
    if not strengths or len(strengths) > MAX_SWEEP_POINTS:
        raise SteeringCheckError(f"strengths must hold 1 to {MAX_SWEEP_POINTS} values")
    for s in strengths:
        validate(rt, prompt, max_new_tokens, seed, feature_id, "amplify", s, max_activation)
    base = rt.generate(prompt, max_new_tokens, seed, None)
    base_assess = _assess(base)
    results = []
    for s in strengths:
        g = rt.generate(prompt, max_new_tokens, seed, _spec(feature_id, "amplify", s, max_activation))
        cmp = compare_generations(base, g)
        a = _assess(g)
        results.append({"strength": s, "text": g.text, "differs_from_baseline": g.text != base.text,
                        "first_divergence_index": cmp["first_divergence_index"],
                        "degenerate": a["degenerate"], "reason": a["reason"], "metrics": a["metrics"]})
    return {"feature_id": feature_id, "mode": "amplify", "seed": seed,
            "max_new_tokens": max_new_tokens, "baseline_text": base.text,
            "baseline_degenerate": base_assess["degenerate"], "baseline_reason": base_assess["reason"],
            "results": results}


# ---------------------------------------------------- X4: prompt-only baseline

def prompt_only_baseline(rt, prompt, max_new_tokens=60, seed=42):
    """X4. What the model writes from the prompt by itself: no steering, same seed.

    Shown separately from the steered runs. Returns {prompt, seed, max_new_tokens, text,
    continuation, new_tokens}; `text` is the prompt plus the new text, `continuation` is
    the new text only. With the same prompt, seed and max_new_tokens it equals the
    baseline_text of the steering functions.
    """
    validate(rt, prompt, max_new_tokens, seed)
    g = rt.generate(prompt, max_new_tokens, seed, None)
    return {"prompt": prompt, "seed": seed, "max_new_tokens": max_new_tokens,
            "text": g.text, "continuation": g.continuation, "new_tokens": len(g.new_ids)}
