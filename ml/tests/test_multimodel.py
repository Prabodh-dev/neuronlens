"""Second-model support on tiny random models (GPT-2 and GPT-Neo configs): no downloads."""
import os
import sqlite3
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPTNeoConfig, GPTNeoForCausalLM

import neuronlens.activations as activations
import neuronlens.build_index as build_index
import neuronlens.runtime as runtime_mod
from neuronlens import modelutil
from neuronlens.activations import Collector
from neuronlens.evals import evaluate
from neuronlens.sae import TopKSAE
from neuronlens.steering import generate_ids, make_hook
from neuronlens.train import apply_overrides, budget_for, load_cfg, throughput

ML = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ML, "scripts"))
import compat_check  # noqa: E402

ROOT = os.path.abspath(os.path.join(ML, ".."))
D, LAYERS, VOCAB = 32, 4, 64


def tiny_gpt2():
    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(n_layer=LAYERS, n_embd=D, n_head=2, vocab_size=VOCAB,
                                      n_positions=64)).eval()


def tiny_neo():
    torch.manual_seed(0)
    return GPTNeoForCausalLM(GPTNeoConfig(
        num_layers=LAYERS, hidden_size=D, num_heads=2, vocab_size=VOCAB,
        max_position_embeddings=64, attention_types=[[["global", "local"], LAYERS // 2]],
        window_size=8)).eval()


MODELS = pytest.mark.parametrize("make", [tiny_gpt2, tiny_neo], ids=["gpt2", "gpt-neo"])


class FakeTok:
    eos_token_id = 0

    def __call__(self, text, return_tensors=None):
        ids = torch.tensor([[1 + (ord(c) % (VOCAB - 1)) for c in text]])
        return SimpleNamespace(input_ids=ids)

    def encode(self, text):
        return [1 + (ord(c) % (VOCAB - 1)) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(97 + i % 26) for i in ids)


# ---- the layout helpers ----

@MODELS
def test_helpers_find_blocks_width_and_unembedding(make):
    m = make()
    assert modelutil.get_block(m, 2) is m.transformer.h[2]
    assert modelutil.n_layers(m) == LAYERS
    assert modelutil.d_model_of(m) == D
    assert modelutil.unembedding(m).shape == (VOCAB, D)
    assert modelutil.unembedding(m) is m.lm_head.weight
    assert modelutil.backbone(m) is m.transformer


def test_model_name_from_ck_and_slug():
    assert modelutil.model_name_from_ck({"cfg": {"model_name": "EleutherAI/gpt-neo-125M"}}) \
        == "EleutherAI/gpt-neo-125M"
    assert modelutil.model_name_from_ck({"cfg": {"run_id": "x"}}) == "gpt2"
    assert modelutil.model_name_from_ck({}) == "gpt2"
    assert modelutil.slug("EleutherAI/gpt-neo-125M") == "gpt-neo-125m"
    assert modelutil.slug("distilgpt2") == "distilgpt2"


# ---- hook, evaluation, steering and generation on both layouts ----

@MODELS
@pytest.mark.parametrize("layer", [0, LAYERS - 1])
def test_collector_matches_the_block_output(make, layer):
    m = make()
    col = Collector(m, layer)
    seen = {}
    m.transformer.h[layer].register_forward_hook(
        lambda mod, i, o: seen.update(out=o[0] if isinstance(o, tuple) else o))
    with torch.no_grad():
        modelutil.backbone(m)(torch.randint(0, VOCAB, (2, 12)))
    assert torch.equal(col.out, seen["out"])


class IdentitySAE:
    d_model = D
    d_sae = D

    def __call__(self, x):
        return x, (x != 0).float(), x


@MODELS
@pytest.mark.parametrize("layer", [0, LAYERS - 1])
def test_evaluate_identity_sae(make, layer):
    m = make()
    seqs = [torch.randint(0, VOCAB, (24,)) for _ in range(4)]
    r = evaluate(m, IdentitySAE(), seqs, layer, torch.zeros(D), torch.ones(()),
                 device="cpu", batch=2)
    assert r["fvu"] < 1e-6
    assert r["ce_recovered"] == pytest.approx(1.0, abs=1e-3)


@MODELS
def test_steering_hook_changes_generation_and_is_removed(make):
    m, tok = make(), FakeTok()
    sae = TopKSAE(D, 16, 4)
    base, n = generate_ids(m, tok, "hello there", 6, 1, None, 1, "cpu")
    again, _ = generate_ids(m, tok, "hello there", 6, 1, None, 1, "cpu")
    hook = make_hook(sae, 3, "amplify", 5.0, 50.0, torch.zeros(D), torch.ones(()))
    steered, _ = generate_ids(m, tok, "hello there", 6, 1, hook, 1, "cpu")
    after, _ = generate_ids(m, tok, "hello there", 6, 1, None, 1, "cpu")
    assert base == again == after and n == len("hello there")
    assert steered != base
    assert not m.transformer.h[1]._forward_hooks


# ---- Runtime ----

def _ck(name=None):
    cfg = {"run_id": "t"} if name is None else {"run_id": "t", "model_name": name}
    return {"layer": 1, "d_model": D, "d_sae": 16, "k": 4, "cfg": cfg,
            "mu": torch.zeros(D), "scale": torch.ones(())}


@MODELS
def test_runtime_feature_activations_on_both_layouts(make):
    m = make()
    sae = TopKSAE(D, 16, 4).eval()
    rt = runtime_mod.Runtime(m, FakeTok(), sae, _ck(), "cpu")
    acts = rt.feature_activations([3, 4, 5, 6, 7], 2)
    assert len(acts) == 5 and acts[0] == 0.0 and all(a >= 0 for a in acts)
    assert rt.model_name == "gpt2" and rt.hook_layer == 1 and rt.d_model == D


def test_runtime_model_name_order(monkeypatch):
    asked = []
    monkeypatch.setattr(runtime_mod, "load_sae", lambda p, d: (TopKSAE(D, 16, 4).eval(), _ck(ckname[0])))
    monkeypatch.setattr(runtime_mod, "load_model", lambda device, name="gpt2": (
        asked.append(name), (tiny_gpt2(), FakeTok()))[1])
    ckname = [None]
    monkeypatch.delenv("NEURONLENS_MODEL_NAME", raising=False)
    assert runtime_mod.Runtime.load("x.pt", "cpu").model_name == "gpt2"
    ckname[0] = "EleutherAI/gpt-neo-125M"
    assert runtime_mod.Runtime.load("x.pt", "cpu").model_name == "EleutherAI/gpt-neo-125M"
    monkeypatch.setenv("NEURONLENS_MODEL_NAME", "distilgpt2")
    assert runtime_mod.Runtime.load("x.pt", "cpu").model_name == "distilgpt2"
    assert runtime_mod.Runtime.load("x.pt", "cpu", model_name="gpt2").model_name == "gpt2"
    assert asked == ["gpt2", "EleutherAI/gpt-neo-125M", "distilgpt2", "gpt2"]


# ---- the loader ----

def test_load_model_default_is_gpt2_and_name_is_passed(monkeypatch):
    seen = []

    class FakeAuto:
        @staticmethod
        def from_pretrained(name):
            seen.append(name)
            return tiny_gpt2()

    class FakeAutoTok:
        @staticmethod
        def from_pretrained(name):
            seen.append(name)
            return FakeTok()

    monkeypatch.setattr(activations, "AutoModelForCausalLM", FakeAuto)
    monkeypatch.setattr(activations, "AutoTokenizer", FakeAutoTok)
    model, tok = activations.load_model("cpu")
    assert seen == ["gpt2", "gpt2"] and not model.training
    assert all(not p.requires_grad for p in model.parameters())
    activations.load_model("cpu", "EleutherAI/gpt-neo-125M")
    assert seen[2:] == ["EleutherAI/gpt-neo-125M"] * 2


def test_build_index_writes_the_model_name(tmp_path, monkeypatch):
    import numpy as np
    F = 4
    ck = {"layer": 8, "d_model": 768, "d_sae": F, "k": 32,
          "cfg": {"run_id": "neo-run", "model_name": "EleutherAI/gpt-neo-125M"},
          "metrics": {"fvu": 0.2, "ce_recovered": 0.9}}
    loaded = []
    monkeypatch.setattr(build_index, "load_sae", lambda p, d: (SimpleNamespace(d_sae=F), ck))
    monkeypatch.setattr(build_index, "load_model",
                        lambda device, name="gpt2": (loaded.append(name), (object(), object()))[1])
    monkeypatch.setattr(build_index, "Collector", lambda m, l: object())
    monkeypatch.setattr(build_index, "token_sequences", lambda t, d, seed: iter(range(10)))
    fire = np.array([500, 100, 0, 10])
    monkeypatch.setattr(build_index, "scan", lambda *a, **k: {
        "n_tok": 1000, "fire": fire, "asum": fire * 1.5, "top_v": np.full((32, F), 2.0)})
    monkeypatch.setattr(build_index, "build_examples", lambda *a, **k: {})
    monkeypatch.setattr(build_index, "logit_attribution", lambda *a, **k: {})
    monkeypatch.setattr(build_index, "_training_curve", lambda d: [])
    out = str(tmp_path / "neo.db")
    build_index.build(str(tmp_path / "x.pt"), out, os.path.join(ROOT, "contracts", "schema.sql"),
                      n_seqs=2, labels_path=str(tmp_path / "none.json"), device="cpu",
                      notes="experimental")
    con = sqlite3.connect(out)
    assert dict(con.execute("SELECT key, value FROM meta"))["model_name"] == "EleutherAI/gpt-neo-125M"
    assert con.execute("SELECT model_name, notes FROM runs").fetchone() == \
        ("EleutherAI/gpt-neo-125M", "experimental")
    con.close()
    assert loaded == ["EleutherAI/gpt-neo-125M"]


# ---- training options ----

CFG = {"run_id": "gpt2-l8-k32", "model_name": "gpt2", "hook_layer": 8, "k": 32,
       "n_train_activations": 100_000_000}


def test_no_overrides_leave_the_config_alone():
    assert apply_overrides(CFG) is CFG
    real = load_cfg(os.path.join(ML, "configs", "gpt2-small.yaml"))
    assert apply_overrides(real, None, None, None) == real
    assert apply_overrides(real, "gpt2")["run_id"] == real["run_id"] == "gpt2-l8-k32"


def test_model_override_names_the_run_and_sets_the_model():
    out = apply_overrides(CFG, model="EleutherAI/gpt-neo-125M")
    assert out["model_name"] == "EleutherAI/gpt-neo-125M"
    assert out["run_id"] == "gpt-neo-125m-l8-k32" and out["hook_layer"] == 8
    assert CFG["model_name"] == "gpt2"
    out = apply_overrides(CFG, model="distilgpt2", layer=3)
    assert out["run_id"] == "distilgpt2-l3-k32" and out["hook_layer"] == 3


def test_layer_and_tokens_overrides():
    out = apply_overrides(CFG, layer=3, tokens=5_000_000)
    assert out["run_id"] == "gpt2-l3-k32" and out["hook_layer"] == 3
    assert out["n_train_activations"] == 5_000_000
    with pytest.raises(ValueError):
        apply_overrides(CFG, layer=-1)
    with pytest.raises(ValueError):
        apply_overrides(CFG, tokens=0)


def test_neo_config_file_is_consistent():
    cfg = load_cfg(os.path.join(ML, "configs", "gpt-neo-125m.yaml"))
    gpt2 = load_cfg(os.path.join(ML, "configs", "gpt2-small.yaml"))
    assert cfg["model_name"] == "EleutherAI/gpt-neo-125M"
    assert cfg["run_id"] == "gpt-neo-125m-l8-k32"
    for key in ("d_model", "d_sae", "k", "batch_size", "lr", "n_train_activations", "seed"):
        assert cfg[key] == gpt2[key]


def test_budget_rounds_down_to_whole_batches():
    assert budget_for(10_000, 10, 4096) == 6_000_000 // 4096 * 4096
    assert budget_for(10_000, 10, 4096) % 4096 == 0
    assert budget_for(1, 1, 4096) == 0


def test_throughput_counts_timed_steps_only():
    t = [0.0]
    calls = []

    def step():
        calls.append(1)
        t[0] += 0.5

    rate, steps = throughput(step, seconds=10, batch_size=100, warmup_steps=3,
                             clock=lambda: t[0])
    assert steps == 20 and len(calls) == 23
    assert rate == pytest.approx(20 * 100 / 10.0)


# ---- compat_check ----

def test_there_are_50_distinct_sentences():
    assert len(compat_check.SENTENCES) == 50 == len(set(compat_check.SENTENCES))


def test_tokenizer_check_pass_and_fail():
    ok, bad = compat_check.check_tokenizer(FakeTok(), FakeTok())
    assert ok and bad == []

    class Other(FakeTok):
        def encode(self, text):
            return super().encode(text)[:-1] if text.startswith("The end") else super().encode(text)

    ok, bad = compat_check.check_tokenizer(Other(), FakeTok())
    assert not ok and bad == ["The end."]


@MODELS
def test_shape_check(make):
    m = make()
    assert compat_check.check_shape(m, 2, expected_width=D)[0]
    assert not compat_check.check_shape(m, LAYERS, expected_width=D)[0]   # no such block
    assert not compat_check.check_shape(m, 2)[0]                          # width is not 768


@MODELS
def test_first_token_ratio_runs_on_both_layouts(make):
    m = make()
    r = compat_check.first_token_ratio(m, FakeTok(), 1, sentences=compat_check.SENTENCES[:6])
    assert r > 0 and r == r


def test_first_token_ratio_sees_an_outlier_at_position_0():
    m = tiny_gpt2()
    block = m.transformer.h[1]

    def boost(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h[:, 0] *= 50
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    plain = compat_check.first_token_ratio(m, FakeTok(), 1, sentences=compat_check.SENTENCES[:8])
    # the boost hook is registered before first_token_ratio adds its collector, so the
    # collector reads the boosted output
    handle = block.register_forward_hook(boost)
    try:
        col_ratio = compat_check.first_token_ratio(m, FakeTok(), 1,
                                                    sentences=compat_check.SENTENCES[:8])
    finally:
        handle.remove()
    assert col_ratio > 10 * plain and col_ratio > compat_check.FIRST_TOKEN_RATIO_MIN


def test_verdict():
    ok, line = compat_check.verdict({"a": (True, ""), "b": (True, "")})
    assert ok and line.startswith("COMPATIBLE")
    ok, line = compat_check.verdict({"a": (True, ""), "b": (False, "")})
    assert not ok and "NOT COMPATIBLE" in line and "b" in line
