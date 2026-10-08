import argparse
import json
import sqlite3

from neuronlens.runtime import Runtime

p = argparse.ArgumentParser()
p.add_argument("specs")
p.add_argument("--ckpt", required=True)
p.add_argument("--db", default="../data/neuronlens.db")
p.add_argument("--max-new-tokens", type=int, default=60)
p.add_argument("--save", action="store_true")
a = p.parse_args()

rt = Runtime.load(a.ckpt)
con = sqlite3.connect(a.db)
for s in json.load(open(a.specs)):
    f, mode = s["feature_id"], s["mode"]
    strength = s.get("strength", 0.0 if mode == "ablate" else 4.0)
    seed = s.get("seed", 42)
    mx = con.execute("select max_activation from features where feature_id=?", (f,)).fetchone()[0]
    out = rt.steer(s["prompt"], f, mode, strength, mx, a.max_new_tokens, seed)
    print(f"--- #{f} {mode} x{strength} seed {seed} prompt: {s['prompt']!r}")
    print(" BASE :", out["baseline_text"])
    print(" STEER:", out["steered_text"])
    if a.save:
        con.execute(
            "insert into steering_examples (feature_id,mode,strength,prompt,seed,"
            "baseline_text,steered_text,is_curated) values (?,?,?,?,?,?,?,?)",
            (f, mode, strength, s["prompt"], seed, out["baseline_text"],
             out["steered_text"], int(s.get("curated", False))))
        con.commit()