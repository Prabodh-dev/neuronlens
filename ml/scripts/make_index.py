import argparse

from neuronlens.build_index import build

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--out", default="../data/neuronlens.db")
p.add_argument("--schema", default="../contracts/schema.sql")
p.add_argument("--n-seqs", type=int, default=3000)
p.add_argument("--labels", default="labels.json")
p.add_argument("--data-dir", default="~/data/fineweb/sample/10BT")
a = p.parse_args()
build(a.ckpt, a.out, a.schema, a.n_seqs, a.labels, a.data_dir)