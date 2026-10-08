import torch


@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens, seed, hook_fn=None,
             layer=8, device="cuda"):
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
    return tok.decode(out[0], skip_special_tokens=True)


def make_hook(sae, feature_id, mode, strength, max_activation, mu, scale):
    d = sae.W_dec[feature_id].float()  # unit direction in normalised space
    state = {"first": True}  # first call = whole prompt; skip position 0

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        new = h.clone()
        sl = slice(1, None) if state["first"] else slice(None)
        state["first"] = False
        part = h[:, sl].float()
        if mode == "amplify":
            part = part + (strength * max_activation / scale) * d
        else:  # ablate: remove this feature's own contribution
            x = (part - mu) * scale
            acts, _ = sae.encode(x.reshape(-1, x.shape[-1]))
            a = acts[:, feature_id].reshape(part.shape[:-1])
            part = part - (a / scale).unsqueeze(-1) * d
        new[:, sl] = part.to(h.dtype)
        return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new

    return hook