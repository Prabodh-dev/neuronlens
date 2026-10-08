import argparse

from neuronlens.train import train

p = argparse.ArgumentParser()
p.add_argument("config")
p.add_argument("--out", default="~/runs")
a = p.parse_args()
train(a.config, out_root=a.out)