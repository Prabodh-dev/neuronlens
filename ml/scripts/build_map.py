"""W7: 2D map of the 12,288 features for the Phase 4 data files. Writes one JSON file.

    cd ml
    export NEURONLENS_CKPT=../data/gpt2-l8-k32.pt
    PYTHONPATH=. python scripts/build_map.py [--seed 42] [--out ../data/map2d.json] \
        [--method auto|umap|pca]

Takes the decoder directions (one unit-norm row per feature), reduces them to 50 dimensions
with PCA and then to 2D with UMAP if umap-learn is installed (`pip install umap-learn`);
otherwise PCA to 2D alone. The file records which. The points are a picture of decoder-direction
similarity; distance is a rough guide only and another seed or method gives another layout.
No model or data is loaded and nothing is written to the database.
"""
import argparse
import os
import sys
from pathlib import Path

from neuronlens import featuremap as fm

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", default=os.environ.get("NEURONLENS_CKPT"),
                   help="checkpoint (default: NEURONLENS_CKPT)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(REPO_ROOT / "data" / "map2d.json"))
    p.add_argument("--method", choices=("auto", "umap", "pca"), default="auto")
    a = p.parse_args(argv)
    if not a.ckpt:
        sys.exit("set NEURONLENS_CKPT or pass --ckpt")
    ckpt = Path(os.path.expanduser(a.ckpt))
    if not ckpt.is_file():
        sys.exit(f"error: checkpoint not found: {ckpt}")
    from neuronlens.interpret import load_sae
    sae, ck = load_sae(str(ckpt), "cpu")
    try:
        doc = fm.build_map(sae.W_dec.detach().float().numpy(), ck["cfg"]["run_id"], a.seed, a.method)
    except RuntimeError as e:
        sys.exit(f"error: {e}")
    out = Path(os.path.expanduser(a.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    fm.write_json(doc, out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB): method {doc['method']}, "
          f"{len(doc['points'])} points, seed {doc['seed']}")


if __name__ == "__main__":
    main()
