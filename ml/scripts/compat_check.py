"""Is a model usable for NeuronLens the way GPT-2 small is? Three checks, then a verdict.

    cd ml
    PYTHONPATH=. python scripts/compat_check.py --model EleutherAI/gpt-neo-125M --layer 8

(a) tokenizer: the model's token ids equal GPT-2's on 50 fixed sentences. The training data
    is tokenised with the model's own tokenizer, so a different vocabulary would still train,
    but the result could not be compared with the GPT-2 run.
(b) shape: hidden width equals the SAE's d_model (768) and the chosen layer exists.
(c) first token: the norm of the hooked activation at position 0 divided by the mean norm at
    later positions, over the same 50 sentences. We drop position 0 because in GPT-2 it is an
    outlier; if another model has no such outlier the same rule drops a normal token. The
    GPT-2 ratio is measured the same way and printed next to it. FIRST_TOKEN_RATIO_MIN is a
    working threshold, not calibrated.

Exit code 0 if every check passes, 1 otherwise. Needs the model files (downloaded once).
"""
import argparse
import sys

REFERENCE_MODEL = "gpt2"
EXPECTED_WIDTH = 768
FIRST_TOKEN_RATIO_MIN = 3.0   # position 0 at least this many times the later positions

SENTENCES = [
    "The band walked on stage and the crowd went quiet.",
    "The Supreme Court ruled on the case on Tuesday.",
    "She said, \"I don't know what you mean,\" and left.",
    "def add(a, b):\n    return a + b",
    "Water boils at 100 degrees Celsius at sea level.",
    "It's a beautiful day; let's go outside!",
    "The price rose from $4.99 to $12.50 in 2023.",
    "Café, naïve, façade, and résumé are all borrowed words.",
    "東京は日本の首都です。",
    "Привет, как дела сегодня?",
    "The quick brown fox jumps over the lazy dog.",
    "   Leading spaces and  double  spaces.",
    "HTTP 404: the page could not be found.",
    "She sells sea shells by the sea shore.",
    "To be, or not to be, that is the question.",
    "The mitochondria is the powerhouse of the cell.",
    "Email me at someone@example.com or call 555-0100.",
    "1, 2, 3, 4, 5, 6, 7, 8, 9, 10.",
    "I can't believe they're not here yet; we've waited an hour.",
    "The moon orbits the Earth about every 27 days.",
    "NASA launched the rocket at dawn.",
    "He walked into the room and sat down quietly.",
    "Rain fell on the old stone bridge all night.",
    "A neural network learns weights from examples.",
    "The recipe needs two cups of flour and one egg.",
    "Unbelievably, the underground railroad was reorganised.",
    "x = [i ** 2 for i in range(10)]",
    "The team scored in the final minute of the game.",
    "Once upon a time there was a small village by the sea.",
    "Is this the real life? Is this just fantasy?",
    "The doctor recommended rest and plenty of fluids.",
    "Our flight was delayed by three hours because of fog.",
    "Photosynthesis converts light into chemical energy.",
    "She opened the letter and began to read.",
    "The stock market closed higher on Friday.",
    "Bonjour tout le monde, comment allez-vous?",
    "The library closes at nine o'clock tonight.",
    "A sudden noise made everyone turn around.",
    "Mix the ingredients slowly, then bake for 20 minutes.",
    "The mayor announced a new plan for public transport.",
    "SELECT name, age FROM users WHERE age > 30;",
    "They argued about politics until midnight.",
    "The ancient temple stood on a hill above the town.",
    "Thank you very much for your help yesterday.",
    "The children laughed and ran across the field.",
    "It was the best of times, it was the worst of times.",
    "Temperatures will fall below freezing this weekend.",
    "The software update adds a dark mode and little else.",
    "Emoji test: \U0001F600 \U0001F680 and a snowman ☃.",
    "The end.",
]
assert len(SENTENCES) == 50


def check_tokenizer(tok, ref_tok, sentences=SENTENCES):
    """(passed, list of sentences whose ids differ)."""
    bad = [s for s in sentences if tok.encode(s) != ref_tok.encode(s)]
    return not bad, bad


def check_shape(model, layer, expected_width=EXPECTED_WIDTH):
    """(passed, text). Width must match the SAE; the layer must exist."""
    from neuronlens.modelutil import d_model_of, n_layers
    width, layers = d_model_of(model), n_layers(model)
    ok = width == expected_width and 0 <= layer < layers
    return ok, f"hidden width {width} (need {expected_width}), {layers} blocks, layer {layer}"


def first_token_ratio(model, tok, layer, sentences=SENTENCES, device="cpu"):
    """Mean over sentences of: norm at position 0 / mean norm at positions 1 and later, for
    the output of block `layer`. Sentences with fewer than 2 tokens are skipped."""
    import torch
    from neuronlens.activations import Collector
    from neuronlens.modelutil import backbone
    col = Collector(model, layer)
    ratios = []
    with torch.no_grad():
        for s in sentences:
            ids = torch.tensor([tok.encode(s)], device=device)
            if ids.shape[1] < 2:
                continue
            backbone(model)(ids)
            n = col.out[0].float().norm(dim=-1)
            ratios.append((n[0] / n[1:].mean()).item())
    return sum(ratios) / len(ratios)


def verdict(results):
    """results: {name: (passed, text)} -> (all passed, closing line)."""
    failed = [k for k, (ok, _) in results.items() if not ok]
    if not failed:
        return True, "COMPATIBLE: every check passed (it does not say the SAE will train as well)"
    return False, "NOT COMPATIBLE: failed " + ", ".join(failed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, default=8)
    p.add_argument("--no-reference", action="store_true",
                   help="do not measure GPT-2's first-token ratio for comparison")
    a = p.parse_args()

    import torch
    from transformers import AutoTokenizer
    from neuronlens.activations import load_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(device, a.model)
    ref_tok = AutoTokenizer.from_pretrained(REFERENCE_MODEL)

    results = {}
    ok, bad = check_tokenizer(tok, ref_tok)
    results["tokenizer"] = (ok, f"{len(SENTENCES) - len(bad)} of {len(SENTENCES)} sentences "
                                f"give the same ids as {REFERENCE_MODEL}")
    results["shape"] = check_shape(model, a.layer)
    if results["shape"][0]:
        r = first_token_ratio(model, tok, a.layer, device=device)
        text = f"position 0 norm / later positions = {r:.2f} (need at least {FIRST_TOKEN_RATIO_MIN})"
        if not a.no_reference and a.model != REFERENCE_MODEL:
            ref_model, ref_t = load_model(device, REFERENCE_MODEL)
            text += f"; {REFERENCE_MODEL} at layer {a.layer}: " \
                    f"{first_token_ratio(ref_model, ref_t, a.layer, device=device):.2f}"
        results["first token"] = (r >= FIRST_TOKEN_RATIO_MIN, text)
    else:
        results["first token"] = (False, "skipped: the shape check failed")

    print(f"model {a.model}, layer {a.layer}")
    for name, (ok, text) in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {text}")
    for s in bad[:5]:
        print(f"        differs: {s!r}")
    ok, line = verdict(results)
    print(line)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
