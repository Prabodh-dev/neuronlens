"""Automated search for a W5 demo example (two features that both read "higher" together).

    cd ml
    export NEURONLENS_DB=../data/neuronlens.db NEURONLENS_CKPT=../data/gpt2-l8-k32.pt
    PYTHONPATH=. python scripts/find_pairs.py [--candidates 150] [--max-pairs 60] [--seed 42] \
        [--out ../demo/pairs_search.json]

Stage 1 steers each candidate feature alone (amplify 0.5) on three neutral prompts; a candidate
qualifies when it reads "higher" without a loop warning on at least 2 of them. Stage 2 steers
pairs of qualifying candidates together at 0.25 and 0.5 (the same strength on both) on the
prompts where both singles worked. The database is opened read-only and never written. The
output file is scratch (demo/pairs_search.json is ignored by git). The rules are in
neuronlens/pair_search.py. This is a search for a demonstration, not a ranking of features.
"""
import argparse
import os
import sqlite3
import sys
from pathlib import Path

from neuronlens import pair_search as ps

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve(value):
    p = Path(os.path.expanduser(value))
    if p.is_absolute() or (Path.cwd() / p).exists():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


def open_readonly(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"database not found: {path}")
    con = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)
    con.execute("PRAGMA query_only = ON")
    return con


def load_runtime(ckpt_path):
    import torch
    from neuronlens.runtime import Runtime
    return Runtime.load(str(ckpt_path), "cuda" if torch.cuda.is_available() else "cpu")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--candidates", type=int, default=ps.DEFAULT_CANDIDATES,
                   help="candidate features to test, the known four included (default 150)")
    p.add_argument("--max-pairs", type=int, default=ps.DEFAULT_MAX_PAIRS)
    p.add_argument("--seed", type=int, default=42,
                   help="seed for the candidate sample and for every generation")
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--out", default=str(REPO_ROOT / "demo" / "pairs_search.json"))
    a = p.parse_args(argv)
    if a.candidates < 1 or a.max_pairs < 1:
        p.error("--candidates and --max-pairs must be at least 1")
    db, ckpt = os.environ.get("NEURONLENS_DB"), os.environ.get("NEURONLENS_CKPT")
    if not db or not ckpt:
        sys.exit("set NEURONLENS_DB and NEURONLENS_CKPT (real mode); see the header of this file")
    try:
        con = open_readonly(resolve(db))
    except OSError as e:
        sys.exit(f"error: {e}")
    if not resolve(ckpt).is_file():
        sys.exit(f"error: checkpoint not found: {resolve(ckpt)}")
    try:
        rows = ps.feature_rows(con)
        run_id = con.execute("SELECT value FROM meta WHERE key='run_id'").fetchone()
    finally:
        con.close()
    rt = load_runtime(resolve(ckpt))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save(doc):
        doc["run_id"] = run_id[0] if run_id else None
        ps.write_json(doc, out)

    doc = ps.search(rt, rows, a.candidates, a.max_pairs, a.seed, max_new_tokens=a.max_new_tokens,
                    checkpoint=save)
    save(doc)
    print(ps.format_summary(doc))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
