import importlib.util
import json
import os
import sqlite3
from collections import Counter
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from neuronlens import cofire as cf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
F, D = 24, 4


# ---- counter ---------------------------------------------------------------------------

def brute(active):
    n = active.shape[1]
    out = np.zeros((n, n), dtype=np.int64)
    for row in active:
        on = np.nonzero(row)[0]
        for a in on:
            out[a, a] += 1
        for a, b in combinations(on, 2):
            out[a, b] += 1
            out[b, a] += 1
    return out


@pytest.mark.parametrize("accumulate", ["cpu", "device"])
def test_counter_matches_brute_force_in_row_blocks(accumulate):
    rng = np.random.default_rng(0)
    active = rng.random((300, F)) < 0.2
    c = cf.CofireCounter(F, "cpu", row_block=7, accumulate=accumulate)   # blocks of 7, 7, 7, 3
    c.add(torch.tensor(active[:100]))
    c.add(torch.tensor(active[100:]))                                    # two batches add up
    got = c.numpy_counts()
    assert c.n_tokens == 300 and got.dtype == np.int32
    assert (got == brute(active)).all()
    assert (got == got.T).all()
    assert (np.diagonal(got) == active.sum(0)).all()                     # own firing counts


# ---- top partners ----------------------------------------------------------------------

def counts_from(fire, pairs):
    n = len(fire)
    m = np.zeros((n, n), dtype=np.int32)
    for i, f in enumerate(fire):
        m[i, i] = f
    for (a, b), c in pairs.items():
        m[a, b] = m[b, a] = c
    return m


def test_ranking_by_jaccard_with_tie_order():
    # feature 0 fires 10 times. Jaccard with each partner:
    #   1: 5/(10+10-5)=1/3   2: 6/(10+20-6)=1/4   3: 6/(10+14-6)=1/3   4: 5/(10+10-5)=1/3
    m = counts_from([10, 10, 20, 14, 10], {(0, 1): 5, (0, 2): 6, (0, 3): 6, (0, 4): 5})
    got = cf.top_partners(m, min_count=5)[0]
    assert [p["feature_id"] for p in got] == [3, 1, 4, 2]       # ties: larger co_count, then smaller id
    assert [p["co_count"] for p in got] == [6, 5, 5, 6]
    assert got[0]["jaccard"] == pytest.approx(1 / 3) and got[3]["jaccard"] == pytest.approx(1 / 4)
    assert got[1]["jaccard"] == got[2]["jaccard"]                # exactly equal, so the tie is real
    # symmetric matrix: feature 1 lists feature 0 with the same numbers
    assert cf.top_partners(m)[1][0] == {"feature_id": 0, "co_count": 5, "jaccard": pytest.approx(1 / 3)}


def test_min_count_filter():
    m = counts_from([10, 10, 10], {(0, 1): 4, (0, 2): 5})        # 1 has the same Jaccard-ish but only 4 shared
    assert [p["feature_id"] for p in cf.top_partners(m, min_count=5)[0]] == [2]
    assert [p["feature_id"] for p in cf.top_partners(m, min_count=4)[0]] == [2, 1]
    assert cf.top_partners(m, min_count=6)[0] == []


def test_top_n_limit():
    n = 14
    fire = [100] * n
    pairs = {(0, j): 10 + j for j in range(1, n)}
    got = cf.top_partners(counts_from(fire, pairs), min_count=5, top_n=10)[0]
    assert len(got) == 10 and got[0]["feature_id"] == 13        # highest co-count has the highest Jaccard here


def test_never_fired_and_skipped_features_get_empty_lists_and_are_not_listed():
    m = counts_from([10, 10, 0, 10], {(0, 1): 8, (0, 3): 8})
    out = cf.top_partners(m, min_count=5)
    assert out[2] == []                                          # never fired
    assert [p["feature_id"] for p in out[0]] == [1, 3]
    out = cf.top_partners(m, min_count=5, skip=[3])             # feature 3 marked dead elsewhere
    assert out[3] == [] and [p["feature_id"] for p in out[0]] == [1]
    assert set(out) == {0, 1, 2, 3}                              # every feature id has an entry


def test_a_feature_is_never_its_own_partner():
    m = counts_from([10, 10], {(0, 1): 9})
    assert all(p["feature_id"] != 0 for p in cf.top_partners(m)[0])


# ---- scan with a stub model and SAE ----------------------------------------------------

class StubSAE:
    """Feature set per token id v (carried in x[:, 0]): {v % F, (7*v + 1) % F}, plus feature 0
    when v is even. Position 0 of every sequence has id 999, which would add its own features
    if it were not dropped."""
    d_sae, k = F, 32

    def encode(self, x):
        v = x[:, 0].round().long()
        acts = torch.zeros(x.shape[0], F)
        acts[torch.arange(x.shape[0]), v % F] = 1.0
        acts[torch.arange(x.shape[0]), (7 * v + 1) % F] = 1.0
        acts[(v % 2 == 0).nonzero().flatten(), 0] = 1.0
        return acts, None


def make_stub(seqs):
    col = SimpleNamespace(out=None)

    def transformer(ids):
        out = torch.zeros(ids.shape[0], ids.shape[1], D)
        out[:, :, 0] = ids.float()
        col.out = out

    return SimpleNamespace(transformer=transformer), col


def stub_sequences(n_seq, length=9, marker=999):
    rng = np.random.default_rng(1)
    return [torch.tensor([marker] + rng.integers(0, 200, length - 1).tolist()) for _ in range(n_seq)]


def reference(seqs, n_tokens):
    rows = []
    for s in seqs:
        for v in s[1:].tolist():                     # position 0 dropped
            on = {v % F, (7 * v + 1) % F} | ({0} if v % 2 == 0 else set())
            rows.append([i in on for i in range(F)])
    return np.array(rows)[:n_tokens]


def run_scan(seqs, n_tokens, batch=2, **kw):
    model, col = make_stub(seqs)
    counter = cf.CofireCounter(F, "cpu", row_block=10, **kw)
    ck = {"mu": torch.zeros(D), "scale": torch.tensor(1.0)}
    done = cf.scan_cofire(model, col, StubSAE(), ck, iter(seqs), n_tokens, counter,
                          device="cpu", batch=batch, log=lambda *_: None)
    return done, counter


def test_scan_counts_match_reference_drop_position_zero_and_cut_exactly():
    seqs = stub_sequences(7)                         # 7 sequences x 8 scored tokens = 56
    done, c = run_scan(seqs, 21)                      # not a multiple of the batch: cut mid-sequence
    assert done == 21 and c.n_tokens == 21
    assert (c.numpy_counts() == brute(reference(seqs, 21))).all()
    done_all, c_all = run_scan(seqs, 10_000)          # data runs out
    assert done_all == 56 and (c_all.numpy_counts() == brute(reference(seqs, 56))).all()


def test_scan_is_deterministic():
    seqs = stub_sequences(5)
    a = run_scan(seqs, 30)[1].numpy_counts()
    b = run_scan(seqs, 30)[1].numpy_counts()
    assert (a == b).all()
    one = cf.top_partners(a, 1)
    two = cf.top_partners(b, 1)
    assert one == two


# ---- output document -------------------------------------------------------------------

def test_output_schema_and_rounding(tmp_path):
    m = counts_from([10, 10, 0], {(0, 1): 7})
    doc = cf.build_output("run-x", 1234, 42, 32, 5, cf.top_partners(m))
    assert list(doc) == ["run_id", "tokens_scanned", "seed", "k", "min_count", "partners"]
    assert doc["run_id"] == "run-x" and doc["tokens_scanned"] == 1234 and doc["k"] == 32
    assert set(doc["partners"]) == {"0", "1", "2"} and doc["partners"]["2"] == []
    entry = doc["partners"]["0"][0]
    assert list(entry) == ["feature_id", "co_count", "jaccard"]
    assert entry == {"feature_id": 1, "co_count": 7, "jaccard": round(7 / 13, 4)}
    path = tmp_path / "c.json"
    cf.write_json(doc, path)
    assert json.loads(path.read_text()) == doc and " " not in path.read_text()   # compact


def test_realistic_size_stays_small(tmp_path):
    n = 12288
    partners = {f: [{"feature_id": (f + j) % n, "co_count": 1234567, "jaccard": 0.123456}
                    for j in range(1, 11)] for f in range(n)}
    path = tmp_path / "big.json"
    cf.write_json(cf.build_output("r", 500000, 42, 32, 5, partners), path)
    assert path.stat().st_size < 8_000_000        # a few MB at most, even with every list full


# ---- script ----------------------------------------------------------------------------

def load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "ml", "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_reads_dead_features_read_only_and_needs_a_checkpoint(tmp_path, monkeypatch):
    script = load_script("build_cofire")
    db = str(tmp_path / "t.db")
    con = sqlite3.connect(db)
    con.executescript(open(os.path.join(ROOT, "contracts", "schema.sql")).read())
    for fid, dead in [(0, 0), (1, 1), (2, 1)]:
        con.execute("INSERT INTO features (feature_id, density, max_activation, is_dead) "
                    "VALUES (?,?,?,?)", (fid, 0.0, 1.0, dead))
    con.commit()
    con.close()
    before = open(db, "rb").read()
    assert script.dead_features(db) == [1, 2] and open(db, "rb").read() == before
    monkeypatch.delenv("NEURONLENS_CKPT", raising=False)
    with pytest.raises(SystemExit, match="NEURONLENS_CKPT"):
        script.main([])
    with pytest.raises(SystemExit, match="checkpoint not found"):
        script.main(["--ckpt", str(tmp_path / "missing.pt")])
