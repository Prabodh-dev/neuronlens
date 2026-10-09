import math
from types import SimpleNamespace

import pytest
import torch

from neuronlens import runtime as rtmod
from neuronlens import steering as st
from neuronlens import steering_checks as sc
from neuronlens.sae import TopKSAE
from neuronlens.steering_checks import MultiSteeringSpec, SteeringCheckError, SteeringSpec
from tests.test_steering_checks import FakeRuntime, late_lower_is, upper_is
from tests.test_textdecode import ByteTok


# ------------------------------------------------------------------ hook maths

D, FEATURES = 8, 20


@pytest.fixture()
def sae():
    torch.manual_seed(0)
    return TopKSAE(D, FEATURES, 3)


def run(hook, h):
    return hook(None, None, (h, "extra"))[0]


def delta_of(sae, f, s, m, scale, h):
    return run(st.make_hook(sae, f, "amplify", s, m, torch.zeros(D), scale), h) - h


def test_two_and_three_directions_are_summed(sae):
    scale, mu = torch.tensor(0.37), torch.zeros(D)
    h = torch.randn(2, 5, D)
    steers = [(1, 2.0, 3.0), (5, 1.0, 4.0), (9, 0.5, 8.0)]
    for n in (2, 3):
        multi = run(st.make_multi_hook(sae, steers[:n], mu, scale), h)
        expect = h.clone()
        for f, s, m in steers[:n]:
            expect[:, 1:] = expect[:, 1:] + delta_of(sae, f, s, m, scale, h)[:, 1:]
        assert torch.allclose(multi, expect, atol=1e-5)
        assert torch.equal(multi[:, 0], h[:, 0])                 # position 0 of the prompt untouched
        assert not torch.equal(multi[:, 1:], h[:, 1:])
    # the added vector is strength x max_activation x unit direction (divided by scale), summed
    d = multi[:, 1:] - h[:, 1:]
    want = sum((s * m / scale) * sae.W_dec[f] for f, s, m in steers)
    assert torch.allclose(d, want.expand_as(d), atol=1e-5)


def test_sum_does_not_depend_on_the_order_of_steers(sae):
    h, scale, mu = torch.randn(2, 5, D), torch.tensor(0.5), torch.zeros(D)
    steers = [(1, 2.0, 3.0), (5, 1.0, 4.0), (9, 0.5, 8.0)]
    ref = run(st.make_multi_hook(sae, steers, mu, scale), h)
    for perm in ([2, 0, 1], [1, 2, 0], [2, 1, 0]):
        out = run(st.make_multi_hook(sae, [steers[i] for i in perm], mu, scale), h)
        assert torch.equal(out, ref)                              # bitwise, not just close


def test_later_calls_steer_every_position_and_dtype_is_kept(sae):
    scale, mu = torch.tensor(1.0), torch.zeros(D)
    hook = st.make_multi_hook(sae, [(1, 1.0, 5.0), (2, 1.0, 5.0)], mu, scale)
    h = torch.randn(1, 4, D).half()
    first = run(hook, h)
    assert first.dtype == torch.float16 and torch.equal(first[:, 0], h[:, 0])
    one = torch.randn(1, 1, D).half()
    later = run(hook, one)
    assert not torch.equal(later, one)                            # a single new token is steered


def test_one_nonzero_steer_equals_the_single_feature_hook_bitwise(sae):
    scale, mu = torch.tensor(0.37), torch.zeros(D)
    for dtype in (torch.float32, torch.float16):
        h = torch.randn(2, 5, D).to(dtype)
        single = run(st.make_hook(sae, 4, "amplify", 1.7, 9.3, mu, scale), h)
        for steers in ([(4, 1.7, 9.3), (11, 0.0, 6.0)], [(11, 0.0, 6.0), (4, 1.7, 9.3)],
                       [(2, 0.0, 1.0), (4, 1.7, 9.3), (15, 0.0, 3.0)]):
            assert torch.equal(run(st.make_multi_hook(sae, steers, mu, scale), h), single)


def test_multi_hook_rejects_duplicates_and_all_zero_is_a_no_op(sae):
    with pytest.raises(ValueError):
        st.make_multi_hook(sae, [(1, 1.0, 2.0), (1, 2.0, 2.0)], torch.zeros(D), torch.tensor(1.0))
    h = torch.randn(1, 4, D)
    out = run(st.make_multi_hook(sae, [(1, 0.0, 2.0), (3, 0.0, 2.0)], torch.zeros(D), torch.tensor(1.0)), h)
    assert torch.equal(out, h)


# ------------------------------------------------------------------ Runtime wiring

def test_runtime_generate_dispatches_multi_specs(monkeypatch):
    seen = {}
    monkeypatch.setattr(rtmod, "make_multi_hook",
                        lambda sae, steers, mu, scale: seen.setdefault("multi", (sae, steers, mu, scale)) and "MULTI")
    monkeypatch.setattr(rtmod, "make_hook", lambda *a: seen.setdefault("single", a) and "SINGLE")
    hooks = []

    def fake_generate_ids(model, tok, prompt, n, seed, hook, layer, device):
        hooks.append(hook)
        return [72, 105], 1

    monkeypatch.setattr(rtmod, "generate_ids", fake_generate_ids)

    class Tok(ByteTok):
        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(i) for i in ids)

    rt = rtmod.Runtime.__new__(rtmod.Runtime)
    rt.tok, rt.sae, rt.model, rt.device, rt.hook_layer, rt.mu, rt.scale = Tok(), "SAE", "M", "cpu", 8, "MU", "SC"
    rt.generate("Hi", 5, 1, MultiSteeringSpec(((3, 1.0, 2.0), (7, 0.5, 4.0))))
    assert seen["multi"] == ("SAE", [(3, 1.0, 2.0), (7, 0.5, 4.0)], "MU", "SC") and hooks[-1] == "MULTI"
    rt.generate("Hi", 5, 1, SteeringSpec(3, "amplify", 1.0, 2.0))
    assert seen["single"] == ("SAE", 3, "amplify", 1.0, 2.0, "MU", "SC") and hooks[-1] == "SINGLE"
    rt.generate("Hi", 5, 1, None)
    assert hooks[-1] is None


# ------------------------------------------------------------------ the check function

def steer(f, s=1.0, m=10.0):
    return {"feature_id": f, "strength": s, "max_activation": m}


def multi(rt, steers=None, **kw):
    kw.setdefault("max_new_tokens", 20)
    kw.setdefault("seed", 1)
    return sc.multi_steering_check(rt, "Hello ", steers or [steer(7), steer(13)], **kw)


def test_output_fields_and_echo():
    rt = FakeRuntime(act_fn=upper_is(8.0, 1.0))
    r = multi(rt, [steer(7, 2, 10), steer(13, 0.5, 12.5), steer(40, 1, 3)])
    assert set(r) == {"prompt", "seed", "max_new_tokens", "steers", "baseline_text", "steered_text",
                      "differs", "first_divergence_index", "degenerate", "reason", "activations"}
    assert r["steers"] == [{"feature_id": 7, "strength": 2.0, "max_activation": 10.0},
                           {"feature_id": 13, "strength": 0.5, "max_activation": 12.5},
                           {"feature_id": 40, "strength": 1.0, "max_activation": 3.0}]
    assert r["prompt"] == "Hello " and r["seed"] == 1 and r["max_new_tokens"] == 20
    assert r["differs"] is True and r["first_divergence_index"] == 5
    assert r["baseline_text"].startswith("Hello ") and r["steered_text"] != r["baseline_text"]
    # one generation each: the baseline unsteered, the steered one with all three steers, same seed
    assert [c[2] for c in rt.calls] == [1, 1] and rt.calls[0][3] is None
    assert rt.calls[1][3] == MultiSteeringSpec(((7, 2.0, 10.0), (13, 0.5, 12.5), (40, 1.0, 3.0)))


def test_per_feature_activation_fields_and_verdicts():
    def per_feature(ids, fid):
        if fid == 7:       # rises on the steered text
            return upper_is(8.0, 1.0)(ids, fid)
        if fid == 13:      # falls
            return late_lower_is(8.0, 1.0)(ids, fid)
        return [3.0] * len(ids)   # unchanged
    rt = FakeRuntime(act_fn=per_feature)
    r = multi(rt, [steer(7), steer(13), steer(40)])
    acts = r["activations"]
    assert [a["feature_id"] for a in acts] == [7, 13, 40]
    assert all(set(a) == {"feature_id", "target_activation_baseline", "target_activation_steered",
                          "activation_verdict"} for a in acts)
    assert [a["activation_verdict"] for a in acts] == ["higher", "lower", "about the same"]
    assert (acts[0]["target_activation_baseline"], acts[0]["target_activation_steered"]) == (1.0, 8.0)
    assert (acts[1]["target_activation_baseline"], acts[1]["target_activation_steered"]) == (8.0, 1.0)
    # measured by re-encoding the finished texts after both generations, never inside a run
    assert rt.events[:2] == ["generate", "generate"] and set(rt.events[2:]) == {"encode"}
    assert len(rt.encode_calls) == 6                              # 3 features x 2 texts
    assert rt.hook_active is False


def test_threshold_is_the_x1_threshold():
    rt = FakeRuntime(act_fn=upper_is(2.0, 1.0))                   # change exactly 1.0/20 = 0.05
    r = multi(rt, [steer(7, 1, 20.0), steer(13, 1, 20.0)])
    assert [a["activation_verdict"] for a in r["activations"]] == ["higher", "higher"]
    assert sc.activation_verdict(0.0499) == "about the same"


def test_degenerate_flag_and_reason_come_from_the_steered_text():
    base = [ord(c) for c in "abcdefghijklmnopqrst"]
    r = multi(FakeRuntime(base_ids=base, steered_ids=[ord("z")] * 20))
    assert r["degenerate"] is True and "repeated tokens" in r["reason"]
    ok = multi(FakeRuntime(baseline_len=20))
    assert ok["degenerate"] is False and ok["reason"] is None


def test_all_zero_strengths_give_the_baseline_and_consistent_activations():
    rt = FakeRuntime(act_fn=upper_is(8.0, 1.0))
    r = multi(rt, [steer(7, 0), steer(13, 0)])
    assert r["differs"] is False and r["first_divergence_index"] is None and r["degenerate"] is False
    assert all(a["activation_verdict"] == "about the same" and
               a["target_activation_baseline"] == a["target_activation_steered"] for a in r["activations"])
    assert len(rt.encode_calls) == 2                              # baseline numbers reused


def test_one_nonzero_strength_matches_the_x1_result():
    fn = upper_is(8.0, 1.0)
    x1 = sc.steering_check(FakeRuntime(act_fn=fn), "Hello ", 7, "amplify", 3.0, 10.0, 20, 1)
    for steers in ([steer(7, 3.0, 10.0), steer(13, 0.0, 10.0)], [steer(13, 0.0, 10.0), steer(7, 3.0, 10.0)]):
        r = multi(FakeRuntime(act_fn=fn), steers)
        assert r["baseline_text"] == x1["baseline_text"] and r["steered_text"] == x1["steered_text"]
        assert r["differs"] == x1["differs"]
        mine = next(a for a in r["activations"] if a["feature_id"] == 7)
        assert mine["target_activation_baseline"] == x1["target_activation_baseline"]
        assert mine["target_activation_steered"] == x1["target_activation_steered"]
        assert mine["activation_verdict"] == x1["activation_verdict"]


def test_single_controls_option():
    rt = FakeRuntime(act_fn=upper_is(8.0, 1.0))
    r = multi(rt, [steer(7, 3.0), steer(13, 0.0)], with_single_controls=True)
    runs = r["single_feature_runs"]
    assert [x["feature_id"] for x in runs] == [7, 13]
    assert runs[0]["differs"] is True and runs[0]["first_divergence_index"] == 5
    assert runs[1]["differs"] is False and runs[1]["first_divergence_index"] is None
    assert {"feature_id", "strength", "steered_text", "differs", "first_divergence_index",
            "degenerate", "reason", "target_activation_baseline", "target_activation_steered",
            "activation_verdict"} == set(runs[0])
    singles = [c[3] for c in rt.calls if isinstance(c[3], SteeringSpec)]
    assert [(s.feature_id, s.mode, s.strength) for s in singles] == [(7, "amplify", 3.0), (13, "amplify", 0.0)]
    assert "single_feature_runs" not in multi(FakeRuntime())


BAD = [
    ("one steer", dict(steers=[steer(7)]), "INVALID_INPUT"),
    ("four steers", dict(steers=[steer(1), steer(2), steer(3), steer(4)]), "INVALID_INPUT"),
    ("no steers", dict(steers=[]), "INVALID_INPUT"),
    ("not a list", dict(steers="7,13"), "INVALID_INPUT"),
    ("missing key", dict(steers=[steer(7), {"feature_id": 13, "strength": 1.0}]), "INVALID_INPUT"),
    ("not a dict", dict(steers=[steer(7), 13]), "INVALID_INPUT"),
    ("duplicate ids", dict(steers=[steer(7), steer(7, 2.0)]), "INVALID_INPUT"),
    ("negative strength", dict(steers=[steer(7), steer(13, -1)]), "INVALID_INPUT"),
    ("strength above 10", dict(steers=[steer(7), steer(13, 10.5)]), "INVALID_INPUT"),
    ("nan strength", dict(steers=[steer(7), steer(13, float("nan"))]), "INVALID_INPUT"),
    ("dead feature", dict(steers=[steer(7), steer(13, 1, 0.0)]), "INVALID_INPUT"),
    ("negative max_activation", dict(steers=[steer(7), steer(13, 1, -2.0)]), "INVALID_INPUT"),
    ("blank prompt", dict(prompt="  "), "INVALID_INPUT"),
    ("bad seed", dict(seed=-1), "INVALID_INPUT"),
    ("bad max_new_tokens", dict(max_new_tokens=0), "INVALID_INPUT"),
    ("too many new tokens", dict(max_new_tokens=sc.MAX_NEW_TOKENS + 1), "INVALID_INPUT"),
    ("id outside the dictionary", dict(steers=[steer(7), steer(100)]), "FEATURE_NOT_FOUND"),
    ("negative id", dict(steers=[steer(7), steer(-1)]), "FEATURE_NOT_FOUND"),
]


@pytest.mark.parametrize("label,kwargs,code", BAD, ids=[b[0] for b in BAD])
def test_validation_errors(label, kwargs, code):
    rt = FakeRuntime()
    args = dict(prompt="Hello ", steers=[steer(7), steer(13)], max_new_tokens=20, seed=1)
    args.update(kwargs)
    with pytest.raises(SteeringCheckError) as e:
        sc.multi_steering_check(rt, args.pop("prompt"), args.pop("steers"), **args)
    assert e.value.code == code
    assert rt.calls == []                                         # nothing is generated for a bad request


def test_strength_zero_and_ten_are_accepted():
    r = multi(FakeRuntime(), [steer(7, 0.0), steer(13, 10.0)])
    assert [s["strength"] for s in r["steers"]] == [0.0, 10.0]


# ---------------------------------------------- end to end with the real hooks on a tiny model

class TinyLM(torch.nn.Module):
    """Position-wise toy model: embedding, one block (the hooked layer), output head."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(3)
        self.emb = torch.nn.Embedding(256, D)
        self.block = torch.nn.Linear(D, D)
        self.head = torch.nn.Linear(D, 256)

    def forward(self, ids):
        return self.head(self.block(self.emb(ids)))


def make_tiny_runtime(monkeypatch):
    model, sae = TinyLM().eval(), TopKSAE(D, FEATURES, 3)
    sae.W_dec.data = torch.nn.functional.normalize(torch.randn(FEATURES, D, generator=torch.Generator().manual_seed(5)), dim=1)

    def generate_ids(model_, tok, prompt, n, seed, hook, layer, device):
        ids = list(prompt.encode())
        handle = model.block.register_forward_hook(hook) if hook is not None else None
        try:
            with torch.no_grad():
                logits = model(torch.tensor([ids]))[0, -1]
                for _ in range(n):
                    nxt = int(logits.argmax())
                    ids.append(nxt)
                    logits = model(torch.tensor([[nxt]]))[0, -1]
        finally:
            if handle is not None:
                handle.remove()
        return ids, len(prompt.encode())

    monkeypatch.setattr(rtmod, "generate_ids", generate_ids)

    class Tok(ByteTok):
        def decode(self, ids, skip_special_tokens=True):
            return "".join(sc_pieces(self, ids))

    from neuronlens.textdecode import token_pieces as sc_pieces
    rt = rtmod.Runtime.__new__(rtmod.Runtime)
    rt.tok, rt.sae, rt.model, rt.device, rt.hook_layer = Tok(), sae, model, "cpu", 0
    rt.mu, rt.scale, rt.d_sae = torch.zeros(D), torch.tensor(1.0), FEATURES
    rt.feature_activations = lambda ids, f: [0.0] * len(ids)      # activations are not the point here
    return rt


def test_real_hooks_one_nonzero_steer_matches_x1_and_two_steers_differ_from_one(monkeypatch):
    rt = make_tiny_runtime(monkeypatch)
    x1 = sc.steering_check(rt, "Hello there ", 4, "amplify", 5.0, 20.0, 20, 1)
    assert x1["differs"] is True
    for steers in ([steer(4, 5.0, 20.0), steer(9, 0.0, 20.0)], [steer(9, 0.0, 20.0), steer(4, 5.0, 20.0)]):
        r = sc.multi_steering_check(rt, "Hello there ", steers, 20, 1)
        assert r["steered_text"] == x1["steered_text"] and r["baseline_text"] == x1["baseline_text"]
    both = sc.multi_steering_check(rt, "Hello there ", [steer(4, 5.0, 20.0), steer(9, 5.0, 20.0)], 20, 1)
    assert both["steered_text"] != x1["steered_text"]            # the second direction adds to the first
    again = sc.multi_steering_check(rt, "Hello there ", [steer(9, 5.0, 20.0), steer(4, 5.0, 20.0)], 20, 1)
    assert again["steered_text"] == both["steered_text"]          # order of the steers does not matter
