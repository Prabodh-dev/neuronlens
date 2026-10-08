"""Evaluate an SAE checkpoint and write the results into the database metrics table.

    PYTHONPATH=. python scripts/run_evals.py --ckpt <ckpt> --db <db>
"""
import argparse
import itertools
import json
import os
import shutil
import sqlite3

# same names as neuronlens.build_index.METRIC_NAMES (duplicated so the update
# logic imports without torch)
METRIC_NAMES = {"fvu": "fvu", "l0": "l0", "dead_pct": "dead_pct",
                "ce_orig": "ce_original", "ce_zero": "ce_zero_ablation",
                "ce_recon": "ce_reconstructed", "ce_recovered": "ce_recovered"}


def update_metrics(db_path, metrics, update_dead=False):
    """Update the metrics table of db_path in place for the run in meta.

    UPDATE the existing row; INSERT only if there is none. dead_pct is left
    alone unless update_dead is True. Returns {db_name: value} of what was written.
    """
    con = sqlite3.connect(db_path)
    try:
        row = con.execute("SELECT value FROM meta WHERE key='run_id'").fetchone()
        if row is None:
            raise ValueError("meta has no run_id")
        run_id = row[0]
        written = {}
        for key, name in METRIC_NAMES.items():
            if key not in metrics:
                continue
            if key == "dead_pct" and not update_dead:
                print("dead_pct: keeping the existing database value (3M-token scan, more "
                      "accurate than this small eval); pass --update-dead to overwrite")
                continue
            value = float(metrics[key])
            cur = con.execute("UPDATE metrics SET value=? WHERE run_id=? AND name=?",
                              (value, run_id, name))
            if cur.rowcount == 0:
                con.execute("INSERT INTO metrics (run_id, name, value) VALUES (?,?,?)",
                            (run_id, name, value))
            written[name] = value
        con.commit()
        return written
    finally:
        con.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--data-dir", default="~/data/fineweb/sample/10BT")
    p.add_argument("--n-seqs", type=int, default=256)
    p.add_argument("--seed", type=int, default=999)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--out-json", default=None)
    p.add_argument("--update-dead", action="store_true")
    a = p.parse_args()

    import torch
    from neuronlens.activations import load_model, token_sequences
    from neuronlens.evals import evaluate
    from neuronlens.interpret import load_sae

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt, db = os.path.expanduser(a.ckpt), os.path.expanduser(a.db)

    sae, ck = load_sae(ckpt, device)
    loaded = load_model(device)
    if isinstance(loaded, tuple):
        model, tok = loaded
    else:
        from transformers import AutoTokenizer
        model, tok = loaded, AutoTokenizer.from_pretrained("gpt2")

    seqs = list(itertools.islice(token_sequences(tok, a.data_dir, seed=a.seed), a.n_seqs))
    print(f"evaluating on {len(seqs)} sequences (seed {a.seed})")
    metrics = evaluate(model, sae, seqs, ck["layer"], ck["mu"], ck["scale"],
                       device, batch=a.batch)
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    if a.out_json:
        out = dict(metrics, n_seqs=a.n_seqs, seed=a.seed)
        with open(os.path.expanduser(a.out_json), "w") as f:
            json.dump(out, f, indent=2)

    shutil.copyfile(db, db + ".bak")
    written = update_metrics(db, metrics, a.update_dead)
    print(f"updated {db} (backup at {db}.bak): {', '.join(sorted(written))}")


if __name__ == "__main__":
    main()
