"""Repair feature_examples.tokens_json in an existing database (U+FFFD from per-token decoding).

feature_examples stores only decoded strings (no token ids, no sequence id or position),
and a lone byte token decoded to U+FFFD has lost its bytes, so the text cannot be
recovered from what is saved. The only way to repair is --rebuild-examples: re-scan the
same data with the same seed and checkpoint and rewrite ONLY tokens_json.

    cd ml
    PYTHONPATH=. python scripts/repair_example_tokens.py --rebuild-examples \
        --ckpt <ckpt> --db ../data/neuronlens.db --data-dir ~/data/fineweb/sample/10BT

Without --rebuild-examples it only reports the counts and changes nothing.

Safety: the database is copied to <db>.bak2 first (refuses to overwrite an existing
.bak2); one transaction; only the tokens_json column is updated; every row is matched to
the rebuilt example by (feature_id, rank) and checked against the stored
activations_json / peak_index / length. Rows that do not match are left untouched, and
if too many do not match (wrong --n-seqs, seed or checkpoint) nothing is written.
"""
import argparse
import hashlib
import itertools
import json
import os
import shutil
import sqlite3
import sys

FFFD = "�"
OTHER_TABLES = ("meta", "runs", "features", "feature_logits", "metrics",
                "density_hist", "training_curve", "steering_examples")


def count_fffd(con):
    total = con.execute("SELECT COUNT(*) FROM feature_examples").fetchone()[0]
    bad = con.execute("SELECT COUNT(*) FROM feature_examples "
                      "WHERE instr(tokens_json, char(65533)) > 0").fetchone()[0]
    return bad, total


def fingerprint(con):
    """Hash of every table except feature_examples (and of that table minus tokens_json)."""
    out = {}
    for t in OTHER_TABLES:
        h = hashlib.sha256()
        try:
            for row in con.execute(f"SELECT * FROM {t} ORDER BY 1, 2"):
                h.update(repr(row).encode())
        except sqlite3.OperationalError:  # table absent in an older schema
            continue
        out[t] = h.hexdigest()
    h = hashlib.sha256()
    for row in con.execute("SELECT example_id, feature_id, rank, activations_json, "
                           "peak_index, max_activation FROM feature_examples ORDER BY example_id"):
        h.update(repr(row).encode())
    out["feature_examples (all columns except tokens_json)"] = h.hexdigest()
    return out


def apply_rebuilt(con, ex, tol=0.02, max_mismatch_frac=0.02):
    """Update tokens_json from rebuilt examples ex[feature_id] -> [(r, toks, acts, pk, mx)].

    Runs inside the caller's transaction. Returns (updated, mismatched, missing).
    Raises RuntimeError (caller rolls back) if too many rows do not match.
    """
    updated = mismatched = missing = 0
    rows = con.execute("SELECT example_id, feature_id, rank, tokens_json, activations_json, "
                       "peak_index FROM feature_examples").fetchall()
    for example_id, f, rank, old_tokens, acts_json, peak in rows:
        lst = ex.get(f)
        if lst is None or rank - 1 >= len(lst):
            missing += 1
            continue
        _, toks, acts, pk, _ = lst[rank - 1]
        old_acts = json.loads(acts_json)
        old_toks = json.loads(old_tokens)
        same = (pk == peak and len(toks) == len(old_acts) == len(old_toks)
                and all(abs(a - b) <= tol for a, b in zip(acts, old_acts)))
        if not same:
            mismatched += 1
            continue
        if toks != old_toks:
            con.execute("UPDATE feature_examples SET tokens_json=? WHERE example_id=?",
                        (json.dumps(toks, ensure_ascii=False), example_id))
            updated += 1
    bad = mismatched + missing
    if rows and bad / len(rows) > max_mismatch_frac:
        raise RuntimeError(
            f"{bad} of {len(rows)} rows did not match the rebuilt examples "
            f"(> {max_mismatch_frac:.0%}); wrong --n-seqs, seed or checkpoint? nothing written")
    return updated, mismatched, missing


def rebuild(a):
    import torch
    from neuronlens.activations import Collector, load_model, token_sequences
    from neuronlens.interpret import build_examples, load_sae, scan

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae, ck = load_sae(os.path.expanduser(a.ckpt), device)
    from neuronlens.modelutil import model_name_from_ck
    model, tok = load_model(device, model_name_from_ck(ck))
    col = Collector(model, ck["layer"])
    seqs = list(itertools.islice(
        token_sequences(tok, a.data_dir, seed=a.seed), a.n_seqs))  # same as build_index
    print("sequences:", len(seqs))
    S = scan(model, col, sae, ck, seqs, device)
    return build_examples(model, col, tok, sae, ck, seqs, S, device)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--db", default="../data/neuronlens.db")
    p.add_argument("--rebuild-examples", action="store_true",
                   help="re-scan the data and rewrite tokens_json (required to repair)")
    p.add_argument("--ckpt", help="checkpoint used to build the database")
    p.add_argument("--data-dir", default="~/data/fineweb/sample/10BT")
    p.add_argument("--n-seqs", type=int, default=3000, help="same value as make_index.py")
    p.add_argument("--seed", type=int, default=123, help="same seed as build_index (123)")
    a = p.parse_args()

    db = os.path.expanduser(a.db)
    con = sqlite3.connect(db)
    bad, total = count_fffd(con)
    print(f"before: {bad} of {total} feature_examples rows contain U+FFFD")
    if not a.rebuild_examples:
        print("The saved strings do not contain token ids, and lone byte tokens were saved as "
              "U+FFFD, so the text cannot be recovered from the database.\n"
              "Re-run with --rebuild-examples --ckpt <ckpt> to rebuild from the data.")
        return 2
    if not a.ckpt:
        sys.exit("--ckpt is required with --rebuild-examples")
    con.close()

    backup = db + ".bak2"
    if os.path.exists(backup):
        sys.exit(f"{backup} already exists; move it away first (not overwriting a backup)")
    shutil.copyfile(db, backup)
    print("backup:", backup)

    ex = rebuild(a)

    con = sqlite3.connect(db)
    before = fingerprint(con)
    try:
        con.execute("BEGIN")
        updated, mismatched, missing = apply_rebuilt(con, ex)
        con.commit()
    except Exception:
        con.rollback()
        raise
    after = fingerprint(con)
    print(f"updated tokens_json in {updated} rows; {mismatched} rows did not match the rebuild "
          f"and {missing} had no rebuilt example (left unchanged)")
    bad2, total2 = count_fffd(con)
    print(f"after:  {bad2} of {total2} feature_examples rows contain U+FFFD")
    print("integrity_check:", con.execute("PRAGMA integrity_check").fetchone()[0])
    changed = [k for k in before if before[k] != after[k]]
    print("all other tables and columns unchanged:", "yes" if not changed else f"NO {changed}")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
