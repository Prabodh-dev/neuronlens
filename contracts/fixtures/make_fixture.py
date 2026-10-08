#!/usr/bin/env python3
"""Generate contracts/fixtures/sample.db: a small, deterministic, synthetic database.

Standard library only. Every text field is marked [sample]; nothing here is real data.
Usage: python contracts/fixtures/make_fixture.py
"""
import json
import math
import random
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = HERE.parent / "schema.sql"
OUT = HERE / "sample.db"
CREATED_AT = "1970-01-01T00:00:00Z"
RUN_ID = "fixture-001"

N_FEATURES = 512
N_NAMED, N_UNLABELLED, N_AMBIGUOUS, N_DEAD = 400, 60, 30, 22

TOPICS = ["sports", "cooking", "finance", "weather", "music", "travel", "medicine",
          "law", "space", "gardening", "cars", "movies", "school", "tech", "history", "animals"]
WORDS = ["the", " game", " won", " last", " night", " and", " the", " team", " said", " it",
         " was", " great", ",", " but", " later", " they", " went", " home", ".", " Then",
         " a", " new", " plan", " began", " to", " take", " shape", " in", " town"]
LOGIT_TOKENS = [" the", " of", " and", " ing", " ly", " ed", " in", " to", " is", " for",
                " with", " on", " as", " by", " at", " from", " that", " this", " an", " or"]


def make_features(rng):
    kinds = (["named"] * N_NAMED + ["unlabelled"] * N_UNLABELLED
             + ["ambiguous"] * N_AMBIGUOUS + ["dead"] * N_DEAD)
    rng.shuffle(kinds)
    rows, named_n = [], 0
    for fid, kind in enumerate(kinds):
        if kind == "dead":
            rows.append((fid, None, "unlabelled", 0.0, 0.0, 0.0, 1, "[sample] dead feature"))
            continue
        density = 10 ** rng.uniform(-5, -1)
        max_act = rng.uniform(2, 20)
        mean_act = max_act * rng.uniform(0.1, 0.4)
        label, notes = None, "[sample] synthetic feature"
        if kind == "named":
            named_n += 1
            label = f"[sample] {TOPICS[named_n % len(TOPICS)]} {named_n}"
        elif kind == "ambiguous":
            notes = "[sample] ambiguous synthetic feature"
        rows.append((fid, label, kind if kind != "unlabelled" else "unlabelled",
                     density, max_act, mean_act, 0, notes))
    return rows


def main():
    rng = random.Random(1234)
    if OUT.exists():
        OUT.unlink()
    con = sqlite3.connect(OUT)
    con.executescript(SCHEMA.read_text())
    con.execute("PRAGMA foreign_keys = ON")

    meta = {"schema_version": "1.0", "run_id": RUN_ID, "model_name": "gpt2", "hook_layer": "8",
            "d_model": "768", "d_sae": "512", "k": "32", "created_at": CREATED_AT,
            "is_synthetic": "true"}
    con.executemany("INSERT INTO meta VALUES (?,?)", meta.items())
    con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)",
                (RUN_ID, "gpt2", 8, 768, 512, 32, None, "[sample] fixture, no checkpoint",
                 CREATED_AT, "[sample] synthetic fixture run, not real data"))

    feats = make_features(rng)
    con.executemany("INSERT INTO features VALUES (?,?,?,?,?,?,?,?)", feats)

    live = [f for f in feats if f[6] == 0]
    for f in live:
        fid, max_act = f[0], f[4]
        for rank in range(1, 6):
            n = rng.randint(8, 16)
            toks = ["[sample]"] + [rng.choice(WORDS) for _ in range(n - 1)]
            peak = rng.randrange(n)
            peak_val = round(max_act * (1 - 0.05 * (rank - 1)), 3)
            acts = [round(rng.uniform(0, peak_val * 0.4), 3) for _ in range(n)]
            acts[peak] = peak_val
            con.execute(
                "INSERT INTO feature_examples (feature_id, rank, tokens_json, activations_json,"
                " peak_index, max_activation) VALUES (?,?,?,?,?,?)",
                (fid, rank, json.dumps(toks), json.dumps(acts), peak, peak_val))
        for direction in ("promoted", "suppressed"):
            for rank, tok in enumerate(rng.sample(LOGIT_TOKENS, 5), start=1):
                score = round(rng.uniform(0.5, 3.0) * (1 if direction == "promoted" else -1)
                              * (1 - 0.1 * (rank - 1)), 3)
                con.execute("INSERT INTO feature_logits VALUES (?,?,?,?,?)",
                            (fid, direction, rank, tok, score))

    metrics = {"fvu": 0.121, "l0": 32, "dead_pct": 4.3, "ce_original": 3.3,
               "ce_zero_ablation": 14.7, "ce_reconstructed": 3.45, "ce_recovered": 0.915}
    con.executemany("INSERT INTO metrics VALUES (?,?,?)",
                    [(RUN_ID, k, float(v)) for k, v in metrics.items()])

    counts = [0] * 10
    for f in live:
        idx = min(9, max(0, int(math.floor((math.log10(f[3]) + 5) / 0.5))))
        counts[idx] += 1
    con.executemany("INSERT INTO density_hist VALUES (?,?,?,?)",
                    [(RUN_ID, -5 + 0.5 * i, -5 + 0.5 * (i + 1), c) for i, c in enumerate(counts)])

    for i in range(30):
        step = (i + 1) * 100
        t = i / 29
        con.execute("INSERT INTO training_curve VALUES (?,?,?,?,?)",
                    (RUN_ID, step, round(1.5 * math.exp(-3 * t) + 0.1 + rng.uniform(0, 0.01), 4),
                     round(0.8 * math.exp(-3 * t) + 0.12 + rng.uniform(0, 0.005), 4),
                     round(min(4.3, 20 * t) + rng.uniform(0, 0.1), 2)))

    named = [f for f in live if f[2] == "named"]
    for i in range(8):
        f = named[i * 7]
        mode = "amplify" if i % 2 == 0 else "ablate"
        strength = [0.5, 1.0, 1.5, 2.0][(i // 2) % 4] if mode == "amplify" else 1.0
        con.execute(
            "INSERT INTO steering_examples (feature_id, mode, strength, prompt, seed, baseline_text,"
            " steered_text, is_curated) VALUES (?,?,?,?,?,?,?,?)",
            (f[0], mode, strength, f"[sample] prompt {i + 1}: The story begins",
             1000 + i, f"[sample] baseline continuation {i + 1}",
             f"[sample] {mode} continuation {i + 1} (synthetic, not real model output)",
             1 if i < 4 else 0))

    con.commit()
    con.execute("VACUUM")
    con.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
