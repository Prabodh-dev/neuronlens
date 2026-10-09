import argparse

from neuronlens.train import measure, train

p = argparse.ArgumentParser()
p.add_argument("config")
p.add_argument("--out", default="~/runs")
p.add_argument("--model", default=None,
               help="Hugging Face model name (default: the config's model_name, gpt2). "
                    "The run folder becomes <out>/<model>-l<layer>-k<k>")
p.add_argument("--layer", type=int, default=None,
               help="hook after transformer block N (default: the config's hook_layer, 8)")
p.add_argument("--tokens", type=int, default=None,
               help="token budget (default: the config's n_train_activations)")
p.add_argument("--measure", action="store_true",
               help="about two minutes of training steps, print tokens/sec, save nothing")
p.add_argument("--minutes", type=float, default=None,
               help="with --measure: print the token budget that fits in this many minutes")
p.add_argument("--seconds", type=int, default=120, help="with --measure: length of the run")
a = p.parse_args()
if a.measure:
    measure(a.config, seconds=a.seconds, minutes=a.minutes, model=a.model, layer=a.layer)
else:
    if a.minutes is not None:
        p.error("--minutes only goes with --measure")
    train(a.config, out_root=a.out, model=a.model, layer=a.layer, tokens=a.tokens)
