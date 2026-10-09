import pytest

from neuronlens import steering_checks as sc
from neuronlens.steering_checks import Generation, SteeringCheckError


def _steers_something(steering):
    """The fake runtime changes its output after the divergence point whenever the steering
    adds something: a single spec with strength, an ablate, or a multi spec whose summed
    strength x max_activation is not zero."""
    if steering is None:
        return False
    if hasattr(steering, "steers"):
        return sum(st * m for _, st, m in steering.steers) != 0
    return steering.strength != 0.0 or steering.mode == "ablate"


ACT_FIELDS = ("target_activation_baseline", "target_activation_steered",
              "target_activation_baseline_mean", "target_activation_steered_mean",
              "activation_change", "activation_verdict")


class FakeRuntime:
    """Deterministic stand-in for Runtime. Tokens are single characters; the baseline is a
    fixed function of (prompt, seed). A steering spec changes the output from token index
    `div` on, where div depends on the strength, mimicking drift after a first divergence."""

    d_sae = 100

    def __init__(self, baseline_len=20, diverge_at=5, steered_ids=None, base_ids=None, act_fn=None):
        self.calls = []
        self.events = []        # "generate" / "encode" in call order
        self.encode_calls = []  # (token_ids, feature_id) for every feature_activations call
        self.hook_active = False
        self.act_fn = act_fn    # (token_ids, feature_id) -> per-token activations
        self.baseline_len, self.diverge_at = baseline_len, diverge_at
        self.steered_ids, self.base_ids = steered_ids, base_ids

    def feature_activations(self, token_ids, feature_id):
        """Stub of Runtime.feature_activations: known per-token activations, position 0 is 0.0."""
        assert not self.hook_active, "re-encoding must run with no steering hook attached"
        self.events.append("encode")
        self.encode_calls.append((list(token_ids), feature_id))
        if self.act_fn is not None:
            acts = list(self.act_fn(token_ids, feature_id))
        else:
            acts = [(i % 10) / 10 for i in token_ids]
        acts[0] = 0.0
        return acts

    def generate(self, prompt, max_new_tokens, seed, steering=None):
        self.hook_active = steering is not None   # a steered run has its hook only while it runs
        try:
            return self._generate(prompt, max_new_tokens, seed, steering)
        finally:
            self.hook_active = False
            self.events.append("generate")

    def _generate(self, prompt, max_new_tokens, seed, steering=None):
        self.calls.append((prompt, max_new_tokens, seed, steering))
        prompt_ids = [ord(c) for c in prompt]
        base = self.base_ids if self.base_ids is not None else \
            [97 + (seed + i) % 26 for i in range(min(self.baseline_len, max_new_tokens))]
        new = list(base)
        if _steers_something(steering):
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
                      "baseline_text", "steered_text", "differs", *ACT_FIELDS}
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


# ----------------------------------------------------------------------------- X3

def gen(prompt, new):
    ids = [ord(c) for c in prompt] + [ord(c) for c in new]
    pieces = [chr(i) for i in ids]
    return Generation("".join(pieces), ids, len(prompt), pieces)


def test_x3_first_divergence_and_split():
    r = sc.compare_generations(gen("P:", "abcdefgh"), gen("P:", "abcXYZgh"))
    assert r["diverges"] is True and r["first_divergence_index"] == 3
    assert r["identical_prefix_tokens"] == ["a", "b", "c"] and r["identical_prefix_text"] == "abc"
    assert r["baseline_from_divergence"] == list("defgh")
    assert r["steered_from_divergence"] == list("XYZgh")
    # positions 3,4,5 differ, 6,7 are equal again: counted from the divergence, 5 compared
    assert r["tokens_changed"] == 3 and r["tokens_compared_from_divergence"] == 5
    assert r["percent_changed_from_divergence"] == 60.0
    assert r["percent_changed_raw"] == 37.5  # 3 of 8 generated tokens
    assert "overstates" in r["percent_changed_raw_note"]
    assert r["prompt_tokens"] == 2


def test_x3_no_divergence():
    r = sc.compare_generations(gen("P:", "abcd"), gen("P:", "abcd"))
    assert r["diverges"] is False and r["first_divergence_index"] is None
    assert r["tokens_changed"] == 0 and r["percent_changed_raw"] == 0.0
    assert r["percent_changed_from_divergence"] is None
    assert r["identical_prefix_text"] == "abcd" and r["steered_from_divergence"] == []


def test_x3_length_difference_counts_as_divergence():
    r = sc.compare_generations(gen("P:", "abcd"), gen("P:", "ab"))
    assert r["first_divergence_index"] == 2 and r["tokens_changed"] == 2
    assert r["steered_from_divergence"] == [] and r["baseline_from_divergence"] == ["c", "d"]
    assert r["baseline_new_tokens"] == 4 and r["steered_new_tokens"] == 2


def test_x3_first_token_differs_and_empty_outputs():
    r = sc.compare_generations(gen("P:", "abc"), gen("P:", "xbc"))
    assert r["first_divergence_index"] == 0 and r["identical_prefix_tokens"] == []
    assert r["tokens_changed"] == 1 and r["percent_changed_raw"] == 33.3
    e = sc.compare_generations(gen("P:", ""), gen("P:", ""))
    assert e["diverges"] is False and e["percent_changed_raw"] == 0.0


def test_x3_prompt_mismatch_is_an_error():
    with pytest.raises(ValueError):
        sc.compare_generations(gen("P:", "ab"), gen("Q:", "ab"))


def test_x3_pieces_keep_split_characters_whole():
    # the apostrophe's bytes are two tokens; the leading piece is an empty string
    b = Generation("x", [1, 2, 3, 4], 1, ["a", "", "’", "z"])
    s = Generation("x", [1, 2, 3, 9], 1, ["a", "", "’", "q"])
    r = sc.compare_generations(b, s)
    assert r["identical_prefix_tokens"] == ["", "’"] and r["first_divergence_index"] == 2


def test_x3_seed_matched_compare_end_to_end():
    rt = FakeRuntime(baseline_len=20, diverge_at=5)
    r = sc.seed_matched_compare(rt, "Hello ", 7, "amplify", 3.0, 10.0, max_new_tokens=20, seed=1)
    assert r["first_divergence_index"] == 5 and r["tokens_changed"] > 0
    assert rt.calls[0][2] == rt.calls[1][2] == 1
    assert r["baseline_text"].startswith("Hello ") and r["strength"] == 3.0
    same = sc.seed_matched_compare(rt, "Hello ", 7, "amplify", 0, 10.0, max_new_tokens=20, seed=1)
    assert same["diverges"] is False


# ----------------------------------------------------------------------------- X2

def ids_of(text):
    """Word-level stand-in tokenizer: one id per whitespace-separated chunk."""
    vocab = {}
    return [vocab.setdefault(w, len(vocab)) for w in text.split()], text


NORMAL = ("The council met on Tuesday to discuss the new bridge, and several residents spoke "
          "about traffic, noise and the cost of building it. After a long debate the members "
          "agreed to ask engineers for another report before voting, which pleased most of "
          "those who had come to listen.")


def test_x2_normal_text_is_not_degenerate():
    a = sc.assess_degeneration(*ids_of(NORMAL))
    assert a["degenerate"] is False and a["reason"] is None
    assert a["metrics"]["new_tokens"] == len(NORMAL.split())


def test_x2_repeated_tokens():
    a = sc.assess_degeneration(*ids_of("the " * 30))
    assert a["degenerate"] and "repeated tokens" in a["reason"]


def test_x2_phrase_loop_is_flagged():
    loop = "he went to the shop and bought bread " * 6   # 8 distinct words, 48 tokens
    a = sc.assess_degeneration(*ids_of(loop))
    assert a["degenerate"] and "loop" in a["reason"]


def test_x2_mostly_non_words():
    junk = "#$%^ &*() ;;;; ---- ~~~~ ==== +++ @@@ xq#z 12@@ !!?? ,,,, ...."
    a = sc.assess_degeneration(*ids_of(junk))
    assert a["degenerate"] and "non-words" in a["reason"]


def test_x2_replacement_characters():
    ids = list(range(20))
    text = "a fine sentence " + "�" * 3 + " with some bad bytes in the middle of it"
    a = sc.assess_degeneration(ids, text)
    assert a["degenerate"] and "U+FFFD" in a["reason"]
    # one trailing U+FFFD (generation cut mid-character) is ignored
    cut = NORMAL + "�"
    assert sc.assess_degeneration(*ids_of(cut))["degenerate"] is False


def test_x2_short_and_empty_outputs():
    assert sc.assess_degeneration([1, 1, 1], "the the the")["degenerate"] is False  # below MIN_TOKENS
    e = sc.assess_degeneration([], "")
    assert e["degenerate"] and e["reason"] == "no new text generated"


def test_x2_thresholds_are_named_constants():
    for name in ("MIN_TOKENS_FOR_RULES", "REPEAT_UNIGRAM_FRACTION", "REPEAT_TRIGRAM_FRACTION",
                 "NON_WORD_FRACTION", "REPLACEMENT_CHAR_FRACTION", "REPLACEMENT_CHAR_MIN_COUNT"):
        assert hasattr(sc, name)


def test_x2_default_strengths_stay_within_the_api_cap():
    assert sc.DEFAULT_SWEEP_STRENGTHS == (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
    assert max(sc.DEFAULT_SWEEP_STRENGTHS) <= 3.0          # SteerRequest.strength cap
    assert sc.STRENGTH_MAX == 10.0                          # the ML side still accepts 0 to 10
    assert len(sc.DEFAULT_SWEEP_STRENGTHS) <= sc.MAX_SWEEP_POINTS


def test_x2_ml_still_accepts_strengths_above_the_api_cap():
    r = sc.strength_sweep(FakeRuntime(), "Hi ", 7, 10.0, strengths=[5.0, 10.0], max_new_tokens=20)
    assert [x["strength"] for x in r["results"]] == [5.0, 10.0]


def test_x2_sweep_default_strengths_order_and_baseline_once():
    rt = FakeRuntime(baseline_len=20)
    r = sc.strength_sweep(rt, "Hello ", 7, 10.0, max_new_tokens=20, seed=3)
    assert [x["strength"] for x in r["results"]] == list(sc.DEFAULT_SWEEP_STRENGTHS)
    assert sum(1 for c in rt.calls if c[3] is None) == 1          # one baseline
    assert {c[2] for c in rt.calls} == {3}                          # same seed everywhere
    zero = r["results"][0]
    assert zero["strength"] == 0.0 and zero["differs_from_baseline"] is False
    assert zero["first_divergence_index"] is None
    assert all(x["differs_from_baseline"] and x["first_divergence_index"] == 5 for x in r["results"][1:])
    assert set(r["results"][0]) == {"strength", "text", "differs_from_baseline",
                                    "first_divergence_index", "degenerate", "reason", "metrics"}
    assert r["mode"] == "amplify" and r["baseline_text"].startswith("Hello ")


def test_x2_sweep_marks_degenerate_strengths():
    base = [ord(c) for c in "abcdefghijklmnopqrst"]
    rt = FakeRuntime(base_ids=base, steered_ids=[ord("z")] * 20)
    r = sc.strength_sweep(rt, "Hi ", 7, 10.0, strengths=[0, 3], max_new_tokens=20)
    assert r["results"][0]["degenerate"] is False
    assert r["results"][1]["degenerate"] is True and "repeated tokens" in r["results"][1]["reason"]


@pytest.mark.parametrize("strengths", [[], [1.0] * (sc.MAX_SWEEP_POINTS + 1), [1.0, -1.0],
                                       [sc.STRENGTH_MAX + 1]])
def test_x2_sweep_errors(strengths):
    with pytest.raises(SteeringCheckError):
        sc.strength_sweep(FakeRuntime(), "Hi", 7, 10.0, strengths=strengths)


# ----------------------------------------------------------------------------- X4

def test_x4_prompt_only_baseline_matches_the_steering_baseline():
    rt = FakeRuntime(baseline_len=12)
    r = sc.prompt_only_baseline(rt, "Once upon ", max_new_tokens=12, seed=9)
    assert set(r) == {"prompt", "seed", "max_new_tokens", "text", "continuation", "new_tokens"}
    assert r["text"] == "Once upon " + r["continuation"] and r["new_tokens"] == 12
    assert rt.calls[-1][3] is None  # no steering
    x1 = sc.steering_check(rt, "Once upon ", 7, "amplify", 2.0, 5.0, max_new_tokens=12, seed=9)
    assert x1["baseline_text"] == r["text"]
    assert sc.prompt_only_baseline(rt, "Once upon ", 12, 9) == r  # deterministic


@pytest.mark.parametrize("kwargs", [dict(prompt=""), dict(max_new_tokens=0), dict(seed=-5)])
def test_x4_errors(kwargs):
    args = dict(prompt="Hi", max_new_tokens=10, seed=1)
    args.update(kwargs)
    with pytest.raises(SteeringCheckError):
        sc.prompt_only_baseline(FakeRuntime(), **args)


# ------------------------------------------------------------ X1: target activation

def upper_is(hi, lo):
    """Activation stub: uppercase letters (the steered text) get `hi`, lowercase ones `lo`."""
    return lambda ids, fid: [hi if 65 <= i <= 90 else lo for i in ids]


def late_lower_is(hi, lo):
    """Activation stub: lowercase letters from 'g' on (present only in the baseline after the
    divergence point) get `hi`; everything else, including the shared prefix, gets `lo`."""
    return lambda ids, fid: [hi if 103 <= i <= 122 else lo for i in ids]


def x1(rt, strength=3.0, max_activation=10.0, mode="amplify", prompt="Hello ", **kw):
    return sc.steering_check(rt, prompt, 7, mode, strength, max_activation,
                             max_new_tokens=20, seed=1, **kw)


def test_x1_activation_higher():
    r = x1(FakeRuntime(act_fn=upper_is(8.0, 1.0)))
    assert r["target_activation_baseline"] == 1.0 and r["target_activation_steered"] == 8.0
    assert r["activation_change"] == pytest.approx(0.7) and r["activation_verdict"] == "higher"


def test_x1_activation_lower():
    r = x1(FakeRuntime(act_fn=late_lower_is(8.0, 1.0)))
    assert r["target_activation_baseline"] == 8.0 and r["target_activation_steered"] == 1.0
    assert r["activation_change"] == pytest.approx(-0.7) and r["activation_verdict"] == "lower"


def test_x1_activation_about_the_same():
    r = x1(FakeRuntime(act_fn=lambda ids, f: [3.0] * len(ids)))
    assert r["activation_change"] == 0.0 and r["activation_verdict"] == "about the same"
    assert r["differs"] is True       # the texts differ even though the activation does not


def test_x1_identical_texts_are_consistent():
    rt = FakeRuntime(act_fn=upper_is(8.0, 1.0))
    r = x1(rt, strength=0)            # amplify at 0 adds nothing: same text
    assert r["differs"] is False
    assert r["activation_change"] == 0.0 and r["activation_verdict"] == "about the same"
    assert r["target_activation_baseline"] == r["target_activation_steered"]
    assert r["target_activation_baseline_mean"] == r["target_activation_steered_mean"]
    assert len(rt.encode_calls) == 1  # the baseline numbers are reused


def test_x1_ablate_and_zero_max_activation():
    rt = FakeRuntime(act_fn=late_lower_is(8.0, 1.0))
    r = x1(rt, mode="ablate", strength=3.0)
    assert r["strength"] == 0.0 and r["activation_verdict"] == "lower"
    assert rt.encode_calls[-1][1] == 7 and rt.calls[-1][3].mode == "ablate"
    z = x1(FakeRuntime(act_fn=late_lower_is(8.0, 1.0)), mode="ablate", max_activation=0.0)
    assert z["activation_change"] == 0.0 and z["activation_verdict"] == "about the same"
    assert z["target_activation_baseline"] == 8.0 and z["target_activation_steered"] == 1.0


def test_activation_same_fraction_boundary():
    assert sc.ACTIVATION_SAME_FRACTION == 0.05
    assert sc.activation_verdict(0.0) == "about the same"
    assert sc.activation_verdict(0.0499) == "about the same"
    assert sc.activation_verdict(-0.0499) == "about the same"
    assert sc.activation_verdict(0.05) == "higher"        # exactly the constant
    assert sc.activation_verdict(-0.05) == "lower"
    # end to end: change of exactly 0.05 (baseline max 1.0, steered 2.0, max_activation 20)
    at = x1(FakeRuntime(act_fn=upper_is(2.0, 1.0)), max_activation=20.0)
    assert at["activation_change"] == 0.05 and at["activation_verdict"] == "higher"
    below = x1(FakeRuntime(act_fn=upper_is(1.99, 1.0)), max_activation=20.0)
    assert below["activation_verdict"] == "about the same"


def test_x1_activation_uses_new_tokens_only_and_max_and_mean():
    # prompt positions carry a huge value that must not count; position 0 is dropped anyway
    def acts(ids, fid):
        out = [99.0] * len(ids)
        n_prompt = len("Hello ")
        for i in range(n_prompt, len(ids)):
            out[i] = float(i - n_prompt)          # 0, 1, 2, ... over the new tokens
        return out
    r = x1(FakeRuntime(baseline_len=20, act_fn=acts))
    assert r["target_activation_baseline"] == 19.0
    assert r["target_activation_baseline_mean"] == pytest.approx(9.5)


def test_x1_empty_continuation_gives_zeros():
    rt = FakeRuntime(base_ids=[], steered_ids=[], act_fn=lambda ids, f: [5.0] * len(ids))
    r = sc.steering_check(rt, "Hi ", 7, "amplify", 1.0, 10.0, max_new_tokens=20, seed=1)
    assert r["target_activation_baseline"] == 0.0 and r["target_activation_steered_mean"] == 0.0


def test_x1_re_encoding_happens_after_generation_with_no_hook_active():
    rt = FakeRuntime(act_fn=upper_is(8.0, 1.0))
    x1(rt)
    assert rt.events == ["generate", "generate", "encode", "encode"]   # never inside a run
    assert rt.hook_active is False
    base_ids, steered_ids = rt.encode_calls[0][0], rt.encode_calls[1][0]
    assert base_ids != steered_ids and all(fid == 7 for _, fid in rt.encode_calls)


def test_x3_and_x2_activation_fields_are_opt_in():
    rt = FakeRuntime(act_fn=upper_is(8.0, 1.0))
    plain = sc.seed_matched_compare(rt, "Hello ", 7, "amplify", 3.0, 10.0, 20, 1)
    assert not any(k in plain for k in ACT_FIELDS) and rt.encode_calls == []
    withact = sc.seed_matched_compare(rt, "Hello ", 7, "amplify", 3.0, 10.0, 20, 1, with_activations=True)
    assert withact["activation_verdict"] == "higher" and withact["first_divergence_index"] == 5
    sw = sc.strength_sweep(FakeRuntime(act_fn=upper_is(8.0, 1.0)), "Hello ", 7, 10.0,
                           strengths=[0, 3], max_new_tokens=20, with_activations=True)
    assert sw["results"][0]["activation_verdict"] == "about the same"
    assert sw["results"][1]["activation_verdict"] == "higher"
    assert sw["target_activation_baseline"] == 1.0
    plain_sw = sc.strength_sweep(FakeRuntime(), "Hello ", 7, 10.0, strengths=[0, 3], max_new_tokens=20)
    assert not any(k in plain_sw["results"][0] for k in ACT_FIELDS)


# --------------------------------------------------- Runtime.feature_activations wiring

def _runtime_with_stub_model(extra_hook=False):
    from types import SimpleNamespace
    import torch
    from neuronlens import runtime as rtmod

    class Block:
        def __init__(self):
            self._forward_hooks = {}

    class FakeSAE:
        d_sae = 6
        def encode(self, x):          # known activations: feature f at position t = 10*f + t
            T = x.shape[0]
            a = torch.stack([torch.arange(T, dtype=torch.float32) + 10 * f for f in range(6)], dim=1)
            return a, None

    rt = rtmod.Runtime.__new__(rtmod.Runtime)
    rt.sae, rt.device, rt.d_sae, rt.hook_layer = FakeSAE(), "cpu", 6, 0
    rt.mu, rt.scale = torch.zeros(3), torch.tensor(1.0)
    block = Block()

    class Col:                         # Collector registers a bound method of itself
        out = None
        def _hook(self, *a): pass

    rt.col = Col()
    block._forward_hooks[1] = rt.col._hook
    if extra_hook:
        block._forward_hooks[2] = lambda *a: None

    def transformer(ids):
        rt.col.out = torch.zeros(1, ids.shape[1], 3)
    transformer.h = [block]
    rt.model = SimpleNamespace(transformer=transformer)
    return rt


def test_runtime_feature_activations_matches_inspect_path_and_zeroes_position_0():
    rt = _runtime_with_stub_model()
    acts = rt.feature_activations([5, 6, 7, 8], 2)
    assert acts == [0.0, 21.0, 22.0, 23.0]           # position 0 is reported as 0.0
    assert rt.feature_activations([5], 2) == [0.0]


def test_runtime_feature_activations_refuses_when_another_hook_is_attached():
    rt = _runtime_with_stub_model(extra_hook=True)
    with pytest.raises(RuntimeError, match="forward hook other than"):
        rt.feature_activations([1, 2, 3], 0)
