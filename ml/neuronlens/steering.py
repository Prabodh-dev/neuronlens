import torch


@torch.no_grad()
def generate_ids(model, tok, prompt, max_new_tokens, seed, hook_fn=None,
                 layer=8, device="cuda"):
    """Seeded sampling; returns (all token ids as a list, number of prompt tokens)."""
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    handle = None
    if hook_fn is not None:
        handle = model.transformer.h[layer].register_forward_hook(hook_fn)
    try:
        torch.manual_seed(seed)
        out = model.generate(
            ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens, do_sample=True,
            temperature=0.8, top_p=0.95, pad_token_id=tok.eos_token_id)
    finally:
        if handle is not None:
            handle.remove()
    return out[0].tolist(), ids.shape[1]


def generate(model, tok, prompt, max_new_tokens, seed, hook_fn=None,
             layer=8, device="cuda"):
    out, _ = generate_ids(model, tok, prompt, max_new_tokens, seed, hook_fn, layer, device)
    return tok.decode(out, skip_special_tokens=True)


def _position_hook(transform):
    """Forward hook that applies transform(part) to the block output at every position except
    position 0 of the prompt (the first call is the whole prompt; later calls carry only new
    tokens, which are all steered)."""
    state = {"first": True}

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        new = h.clone()
        sl = slice(1, None) if state["first"] else slice(None)
        state["first"] = False
        part = h[:, sl].float()
        new[:, sl] = transform(part).to(h.dtype)
        return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new

    return hook


def amplify_delta(sae, steers, scale):
    """The vector added at each steered position: for every (feature_id, strength,
    max_activation), strength x max_activation x decoder direction (decoder rows are unit
    norm in normalised space; dividing by scale puts it back in raw units). Several terms are
    summed in order of feature id, so the result does not depend on the order given. With one
    steer this is exactly that one term."""
    terms = [(strength * max_activation / scale) * sae.W_dec[feature_id].float()
             for feature_id, strength, max_activation in sorted(steers)]
    delta = terms[0]
    for t in terms[1:]:
        delta = delta + t
    return delta


def make_multi_hook(sae, steers, mu, scale):
    """Amplify several features in one hook. steers: [(feature_id, strength, max_activation)].
    Amplify only; the single-feature hook below is the same code path with one steer."""
    if len({f for f, _, _ in steers}) != len(steers):
        raise ValueError("duplicate feature ids in steers")
    delta = amplify_delta(sae, steers, scale)
    return _position_hook(lambda part: part + delta)


def make_hook(sae, feature_id, mode, strength, max_activation, mu, scale):
    d = sae.W_dec[feature_id].float()  # unit direction in normalised space
    if mode == "amplify":
        delta = amplify_delta(sae, [(feature_id, strength, max_activation)], scale)
        return _position_hook(lambda part: part + delta)

    def ablate(part):  # remove this feature's own contribution
        x = (part - mu) * scale
        acts, _ = sae.encode(x.reshape(-1, x.shape[-1]))
        a = acts[:, feature_id].reshape(part.shape[:-1])
        return part - (a / scale).unsqueeze(-1) * d

    return _position_hook(ablate)
