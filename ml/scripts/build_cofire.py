"""W4: co-fire counts for the Phase 4 data files. Runs on the GPU machine; writes one JSON file.

    cd ml
    export NEURONLENS_CKPT=../data/gpt2-l8-k32.pt
    PYTHONPATH=. python scripts/build_cofire.py [--tokens 500000] [--seed 42] \
        [--out ../data/cofire.json] [--db ../data/neuronlens.db]

For every scored token (GPT-2 small, output of transformer.h[8], position 0 dropped) the
active SAE features are counted pairwise. For each feature the top 10 partners by Jaccard,
c_ab / (c_a + c_b - c_ab), are kept, with at least --min-count shared tokens. Features that
never fired in the sample get an empty list; with --db (opened read-only) features marked
dead there get one too. Nothing is written to the database.

The count matrix is about 600 MB (12,288 x 12,288 int32). It is kept in host memory and
filled in row blocks computed on the GPU, so the GPU holds only one block at a time
(--row-block, --accumulate device keeps the whole matrix on the GPU instead).

Note on --seed: 42 is the seed the SAE was trained with, so the default sample is the same
token stream as training. Another seed gives tokens the SAE did not train on.
"""
import argparse
import os
import sqlite3
import sys
from pathlib import Path

from neuronlens import cofire as cf

REPO_ROOT = Path(__file__).resolve().parents[2]


def dead_features(db_path):
    con = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    try:
        return [r[0] for r in con.execute("SELECT feature_id FROM features WHERE is_dead = 1")]
    finally:
        con.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", default=os.environ.get("NEURONLENS_CKPT"),
                   help="checkpoint (default: NEURONLENS_CKPT)")
    p.add_argument("--tokens", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(REPO_ROOT / "data" / "cofire.json"))
    p.add_argument("--data-dir", default="~/data/fineweb/sample/10BT")
    p.add_argument("--min-count", type=int, default=cf.MIN_CO_COUNT)
    p.add_argument("--top", type=int, default=cf.TOP_PARTNERS)
    p.add_argument("--db", default=None, help="optional database, read-only, to blank dead features")
    p.add_argument("--batch", type=int, default=4, help="sequences per forward pass")
    p.add_argument("--row-block", type=int, default=cf.ROW_BLOCK)
    p.add_argument("--accumulate", choices=("cpu", "device"), default="cpu")
    a = p.parse_args(argv)
    if not a.ckpt:
        sys.exit("set NEURONLENS_CKPT or pass --ckpt")
    ckpt = Path(os.path.expanduser(a.ckpt))
    if not ckpt.is_file():
        sys.exit(f"error: checkpoint not found: {ckpt}")
    dead = dead_features(os.path.expanduser(a.db)) if a.db else []

    import torch
    from neuronlens.activations import Collector, load_model, token_sequences
    from neuronlens.interpret import load_sae
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae, ck = load_sae(str(ckpt), device)
    model, tok = load_model(device)
    col = Collector(model, ck["layer"])
    seqs = token_sequences(tok, a.data_dir, seed=a.seed)
    counter = cf.CofireCounter(sae.d_sae, device, a.row_block, a.accumulate)
    done = cf.scan_cofire(model, col, sae, ck, seqs, a.tokens, counter, device, a.batch)
    if done < a.tokens:
        print(f"warning: the data ran out after {done} tokens")
    partners = cf.top_partners(counter.numpy_counts(), a.min_count, a.top, skip=dead)
    doc = cf.build_output(ck["cfg"]["run_id"], done, a.seed, sae.k, a.min_count, partners)
    out = Path(os.path.expanduser(a.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    cf.write_json(doc, out)
    listed = sum(1 for v in partners.values() if v)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB): {done} tokens, "
          f"{listed} of {sae.d_sae} features have partners, "
          f"{sum(1 for v in partners.values() if not v)} have an empty list")


if __name__ == "__main__":
    main()
