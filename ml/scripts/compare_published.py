"""Like-for-like comparison of our SAE with the published OpenAI GPT-2 small SAE.

    cd ml
    PYTHONPATH=. python scripts/compare_published.py --ours ~/runs/gpt2-l8-k32/final.pt \
        --data-dir ~/data/fineweb/sample/10BT --n-seqs 256 --seed 999 [--out-json cmp.json]

WHAT WAS VERIFIED (from the official sources, not from memory)
  openai/sparse_autoencoder, paths.py:
    v5_32k(location, layer_index) -> "{location}_v5_32k/autoencoders/{layer_index}.pt"
    under az://openaipublic/sparse-autoencoder/gpt2-small/ ; location in
    resid_delta_attn | resid_delta_mlp | resid_post_attn | resid_post_mlp ; layers 0-11.
    Docstring: 2^15 = 32,768 latents, TopK(32), no L1, LAYER-NORMED inputs; the training
    token count is marked TODO there (so it is not known from the source).
  openai/sparse_autoencoder, README: location "resid_post_mlp" -> transformer_lens hook
    blocks.{L}.hook_resid_post.
  openai/sparse_autoencoder, model.py (Autoencoder):
    state-dict keys: pre_bias, encoder.weight (n_latents, d_model), latent_bias,
    decoder.weight (d_model, n_latents), plus bookkeeping buffers and the activation
    name; normalize=True when the activation is TopK.
    preprocess: LN(x): mu = x.mean(-1); x = x - mu; std = x.std(-1) (unbiased);
                x = x / (std + 1e-5)
    encode:     latents = TopK(k, ReLU)( (x_ln - pre_bias) @ encoder.weight.T + latent_bias )
    decode:     recon_ln = latents @ decoder.weight.T + pre_bias
                recon    = recon_ln * std + mu        (undoes the LN)
  SAELens pretrained_saes.yaml: release "gpt2-small-resid-post-v5-32k" lists
    id blocks.8.hook_resid_post, path v5_32k_layer_8.pt, l0 32.0,
    variance_explained 0.74 (registry metadata, not measured by this script).
  SAELens sae.py: its "layer_norm" normalisation is the same formula as OpenAI's LN.

  => "layer 8 resid_post" = residual stream AFTER transformer block 8 (zero-indexed)
     = output of HF GPT2 transformer.h[8] = our hook. The hook points match.

NOT VERIFIED (no access to the weight hosts from the authoring environment)
  * That https://openaipublic.blob.core.windows.net/sparse-autoencoder/<path> serves the
    az://openaipublic/sparse-autoencoder/<path> blobs. The host is the one the openai repo
    README uses for its viewer; the az:// -> https mapping is inferred, not tested.
  * The contents of the Hugging Face mirror jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs
    (file names / tensor keys / whether it is a raw OpenAI state dict or a SAELens
    safetensors folder). The loader below accepts only the raw OpenAI key layout and
    stops with a clear error listing the keys it found otherwise. It never guesses.
  * The exact nesting of "k" inside the saved state dict. It is looked up in a few
    places; if it cannot be found the script stops unless --published-k is given.
  * What data / how many tokens the published SAE was trained on, and whether position 0
    was excluded in training. The comparison drops position 0 for both SAEs, as in evals.

FVU is defined as in neuronlens.evals.evaluate: over the scored positions (>= 1),
    FVU = sum((recon - x)^2) / sum((x - mean(x))^2)
i.e. squared error over TOTAL VARIANCE after subtracting the per-dimension mean,
accumulated over every scored token. It is computed in our normalised space
(x - mu) * scale; because both SAEs are mapped into that same space by a common affine
map, the ratio equals the FVU in raw residual units.
"""
import argparse
import itertools
import json
import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

OPENAI_BASE = "https://openaipublic.blob.core.windows.net/sparse-autoencoder/gpt2-small"
OPENAI_RELEASE = "resid_post_mlp_v5_32k"  # location + release, from paths.py
OPENAI_LN_EPS = 1e-5                      # from model.py LN()
DEFAULT_K = 32                            # TopK(32) per the v5_32k docstring in paths.py


def openai_url(layer):
    return f"{OPENAI_BASE}/{OPENAI_RELEASE}/autoencoders/{layer}.pt"


class PublishedSAEAdapter(nn.Module):
    """Wraps the OpenAI TopK SAE so evals.evaluate can call it like our TopKSAE.

    evaluate() hands us x_norm = (x_raw - mu) * scale (our normalisation) and expects
    (recon_norm, acts, aux) back in that same space. Steps:
      1. undo our normalisation:           x_raw = x_norm / scale + mu
      2. published preprocessing (LN):     x_ln = (x_raw - m) / (std + eps),
         m = per-token mean over d_model, std = per-token unbiased std
      3. published encoder:                pre = (x_ln - pre_bias) @ W_enc.T + latent_bias
      4. TopK(k) then ReLU on the kept values -> acts
      5. published decoder:                recon_ln = acts @ W_dec.T + pre_bias
      6. undo the published LN:            recon_raw = recon_ln * std + m
      7. back into our space:              recon_norm = (recon_raw - mu) * scale
    """

    def __init__(self, pre_bias, enc_w, latent_bias, dec_w, k, mu, scale,
                 eps=OPENAI_LN_EPS):
        super().__init__()
        self.d_sae, self.d_model = enc_w.shape
        self.k, self.eps = int(k), eps
        for name, t in (("pre_bias", pre_bias), ("enc_w", enc_w),
                        ("latent_bias", latent_bias), ("dec_w", dec_w),
                        ("mu", mu), ("scale", scale)):
            self.register_buffer(name, t.detach().float().clone())

    @classmethod
    def from_state_dict(cls, sd, mu, scale, k=None):
        needed = ("pre_bias", "encoder.weight", "latent_bias", "decoder.weight")
        missing = [n for n in needed if n not in sd]
        if missing:
            raise ValueError(
                f"not an OpenAI-format Autoencoder state dict; missing {missing}. "
                f"Keys found: {sorted(str(x) for x in sd.keys())[:20]}")
        if k is None:
            k = _find_k(sd)
        enc_w, dec_w = sd["encoder.weight"], sd["decoder.weight"]
        if tuple(dec_w.shape) != (enc_w.shape[1], enc_w.shape[0]):
            raise ValueError(f"decoder.weight {tuple(dec_w.shape)} does not match "
                             f"encoder.weight {tuple(enc_w.shape)}")
        return cls(sd["pre_bias"], enc_w, sd["latent_bias"], dec_w, k, mu, scale)

    def forward(self, x_norm):
        x_raw = x_norm / self.scale + self.mu                       # 1
        m = x_raw.mean(dim=-1, keepdim=True)                        # 2
        c = x_raw - m
        std = c.std(dim=-1, keepdim=True)
        x_ln = c / (std + self.eps)
        pre = F.linear(x_ln - self.pre_bias, self.enc_w, self.latent_bias)   # 3
        vals, idx = pre.topk(self.k, dim=-1)                        # 4
        acts = torch.zeros_like(pre).scatter(-1, idx, F.relu(vals))
        recon_ln = acts @ self.dec_w.T + self.pre_bias              # 5
        recon_raw = recon_ln * std + m                              # 6
        recon_norm = (recon_raw - self.mu) * self.scale             # 7
        return recon_norm, acts, pre


def _find_k(sd):
    """Look for TopK's k in the places model.py could have saved it."""
    cands = [sd.get("k")]
    ad = sd.get("activation_state_dict")
    if isinstance(ad, dict):
        cands.append(ad.get("k"))
    cands.append(sd.get("activation_state_dict.k"))
    for c in cands:
        if c is not None:
            return int(c)
    raise ValueError("could not find k in the state dict; pass --published-k "
                     f"(the v5_32k docstring says TopK({DEFAULT_K}))")


def resolve_published(path, layer, hf_file=None):
    """Return a local file path for the published checkpoint."""
    if path is None:
        path = openai_url(layer)
    path = os.path.expanduser(path)
    if os.path.isfile(path):
        return path
    cache = os.path.expanduser("~/.cache/neuronlens/published")
    os.makedirs(cache, exist_ok=True)
    if path.startswith(("http://", "https://")):
        dest = os.path.join(cache, path.split("//", 1)[1].replace("/", "_"))
        if not os.path.exists(dest):
            print("downloading", path)
            urllib.request.urlretrieve(path, dest)
        return dest
    if "/" in path and hf_file:  # Hugging Face repo id + file inside it
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise SystemExit("huggingface_hub is needed for repo ids: "
                             "pip install huggingface_hub") from e
        return hf_hub_download(repo_id=path, filename=hf_file)
    raise SystemExit(f"cannot resolve --published-path {path!r}: not a local file or URL; "
                     "for a Hugging Face repo id also pass --published-file")


def make_row(name, n_latents, k, metrics, note):
    return {"sae": name, "n_latents": n_latents, "k": k,
            "fvu": metrics["fvu"], "l0": metrics["l0"],
            "ce_original": metrics["ce_orig"], "ce_zero_ablation": metrics["ce_zero"],
            "ce_reconstructed": metrics["ce_recon"], "ce_recovered": metrics["ce_recovered"],
            "note": note}


COLS = [("sae", "SAE"), ("n_latents", "latents"), ("k", "k"), ("fvu", "FVU"),
        ("l0", "L0"), ("ce_original", "CE orig"), ("ce_zero_ablation", "CE zero"),
        ("ce_reconstructed", "CE recon"), ("ce_recovered", "CE recov")]


def format_table(rows):
    """Plain-text table; the note column is printed under the table (it is long)."""
    def cell(r, key):
        v = r[key]
        if key == "n_latents":
            return f"{v:,}"
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    grid = [[h for _, h in COLS]] + [[cell(r, key) for key, _ in COLS] for r in rows]
    widths = [max(len(row[i]) for row in grid) for i in range(len(COLS))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in grid]
    lines.insert(1, "  ".join("-" * w for w in widths))
    for r in rows:
        lines.append(f"note [{r['sae']}]: {r['note']}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ours", required=True, help="our checkpoint (.pt)")
    p.add_argument("--data-dir", default="~/data/fineweb/sample/10BT")
    p.add_argument("--n-seqs", type=int, default=256)
    p.add_argument("--seed", type=int, default=999)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--published-path", default=None,
                   help="local OpenAI-format .pt, an https URL, or a Hugging Face repo id "
                        "(default: the OpenAI blob URL for resid_post_mlp_v5_32k)")
    p.add_argument("--published-file", default=None,
                   help="file inside the HF repo when --published-path is a repo id")
    p.add_argument("--published-layer", type=int, default=8,
                   help="layer of the published SAE (default 8)")
    p.add_argument("--published-k", type=int, default=None)
    p.add_argument("--out-json", default=None)
    a = p.parse_args()

    from neuronlens.activations import load_model, token_sequences
    from neuronlens.evals import evaluate
    from neuronlens.interpret import load_sae

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ours, ck = load_sae(os.path.expanduser(a.ours), device)
    layer = ck["layer"]
    from neuronlens.modelutil import model_name_from_ck
    if model_name_from_ck(ck) != "gpt2":
        raise SystemExit(f"the published SAE is for GPT-2 small; this checkpoint is for "
                         f"{model_name_from_ck(ck)}; refusing to compare")
    if layer != a.published_layer:
        raise SystemExit(f"hook mismatch: our SAE hooks the output of block {layer} but the "
                         f"published SAE is for layer {a.published_layer}; refusing to compare")
    print(f"hook check: ours = output of transformer.h[{layer}] (residual after block {layer}, "
          f"zero-indexed); published = resid_post_mlp layer {a.published_layer} = "
          f"blocks.{a.published_layer}.hook_resid_post -> same point")
    mu, scale = ck["mu"], ck["scale"]

    ckpt_path = resolve_published(a.published_path, a.published_layer, a.published_file)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pub = PublishedSAEAdapter.from_state_dict(sd, mu.to(device), scale.to(device),
                                              k=a.published_k).to(device).eval()
    print(f"published SAE: {pub.d_sae:,} latents, d_model {pub.d_model}, k {pub.k} "
          f"(from {ckpt_path})")
    if pub.d_model != ours.d_model:
        raise SystemExit("d_model mismatch between the two SAEs")

    loaded = load_model(device)
    if isinstance(loaded, tuple):
        model, tok = loaded
    else:
        from transformers import AutoTokenizer
        model, tok = loaded, AutoTokenizer.from_pretrained("gpt2")
    seqs = list(itertools.islice(token_sequences(tok, a.data_dir, seed=a.seed), a.n_seqs))
    print(f"evaluating both SAEs on the same {len(seqs)} sequences (seed {a.seed})")

    m_ours = evaluate(model, ours, seqs, layer, mu, scale, device, batch=a.batch)
    m_pub = evaluate(model, pub, seqs, layer, mu, scale, device, batch=a.batch)
    if abs(m_ours["ce_orig"] - m_pub["ce_orig"]) > 1e-6:
        print("WARNING: CE original differs between runs; the two evaluations were not "
              "on identical inputs")

    n_train = ck.get("cfg", {}).get("n_train_activations")
    common = ("Not equal capacity: 12,288 vs 32,768 latents. Positions >= 1 only, same "
              "sequences, same hook.")
    rows = [
        make_row("ours (gpt2-l8-k32)", ours.d_sae, ours.k, m_ours,
                 f"{common} Trained on FineWeb-Edu"
                 + (f", {n_train:,} activations" if n_train else "") + "."),
        make_row("OpenAI v5_32k L8", pub.d_sae, pub.k, m_pub,
                 f"{common} Training tokens/data: not stated in the source (token count is "
                 "TODO in paths.py); input is per-token layer-normed inside the SAE. "
                 "SAELens registry metadata lists variance_explained 0.74 (not measured here)."),
    ]
    print(format_table(rows))

    if a.out_json:
        with open(os.path.expanduser(a.out_json), "w") as f:
            json.dump({"n_seqs": len(seqs), "seed": a.seed, "layer": layer,
                       "fvu_definition": "sum sq error / total variance about the mean, "
                                         "positions >= 1, in (x-mu)*scale space",
                       "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
