"""Phase 4 demo examples: steering checks on the real model (needs the GPU and the real files).

    cd ml
    export NEURONLENS_DB=../data/neuronlens.db
    export NEURONLENS_CKPT=../data/gpt2-l8-k32.pt
    PYTHONPATH=. python scripts/phase4_demo.py [--out ../demo/phase4_examples.json]

Runs the X1 to X4 functions (neuronlens.steering_checks) for three examples: feature 7
("hon") amplified at 3x, feature 13 ("upside") amplified at 3x, and one ablate example.
Each amplify example also gets a strength sweep with the degeneration flags. The database is
opened read-only; the only file written is the JSON output. Steering is a test of what a
feature does, not a fix: the outputs shift the text and are not an improvement.

The prompts below are starting points. They have not been tried on the real model, so edit
EXAMPLES, run it, and keep the ones that read well.
"""
import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

from neuronlens import steering_checks as sc

REPO_ROOT = Path(__file__).resolve().parents[2]

EXAMPLES = [
    {"feature_id": 7, "expect_label": "hon", "mode": "amplify", "strength": 3.0,
     "prompt": "She looked out of the window and saw"},
    {"feature_id": 13, "expect_label": "upside", "mode": "amplify", "strength": 3.0,
     "prompt": "He turned the picture"},
    {"feature_id": 7, "expect_label": "hon", "mode": "ablate", "strength": 1.0,
     "prompt": "Drivers began to honk their horns while the traffic"},
]
SEED = 42
MAX_NEW_TOKENS = 60


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


def feature_row(con, fid):
    return con.execute("SELECT label, label_status, max_activation, is_dead FROM features "
                       "WHERE feature_id = ?", (fid,)).fetchone()


def build_examples(rt, con, examples=EXAMPLES, seed=SEED, max_new_tokens=MAX_NEW_TOKENS,
                   sweep=True, log=print):
    """Run the checks for each example; returns the list written to the JSON file."""
    out = []
    for ex in examples:
        fid = ex["feature_id"]
        row = feature_row(con, fid)
        if row is None or row[3] or not row[2]:
            log(f"skip feature {fid}: not in the database, dead, or max_activation is 0")
            continue
        label, status, max_act, _ = row
        label_ok = bool(label) and ex["expect_label"].lower() in label.lower()
        if not label_ok:
            log(f"WARNING feature {fid}: label is {label!r}, expected it to mention "
                f"{ex['expect_label']!r}; check the feature id")
        cmp = sc.seed_matched_compare(rt, ex["prompt"], fid, ex["mode"], ex["strength"], max_act,
                                      max_new_tokens, seed)
        check = {"feature_id": fid, "label": label, "label_status": status,
                 "label_matches_expected": label_ok, "max_activation": max_act,
                 "prompt": ex["prompt"], "seed": seed, "mode": ex["mode"],
                 "strength": cmp["strength"], "max_new_tokens": max_new_tokens,
                 "baseline": cmp["baseline_text"], "steered": cmp["steered_text"],
                 "differs": cmp["baseline_text"] != cmp["steered_text"],
                 "first_divergence_index": cmp["first_divergence_index"],
                 "tokens_changed": cmp["tokens_changed"],
                 "tokens_compared_from_divergence": cmp["tokens_compared_from_divergence"],
                 "percent_changed_from_divergence": cmp["percent_changed_from_divergence"],
                 "percent_changed_raw": cmp["percent_changed_raw"],
                 "percent_changed_raw_note": cmp["percent_changed_raw_note"],
                 "sweep": None}
        if sweep and ex["mode"] == "amplify":
            sw = sc.strength_sweep(rt, ex["prompt"], fid, max_act, max_new_tokens=max_new_tokens,
                                   seed=seed)
            check["sweep"] = {"baseline_degenerate": sw["baseline_degenerate"],
                              "baseline_reason": sw["baseline_reason"],
                              "results": [{k: r[k] for k in ("strength", "text", "degenerate",
                                                              "reason", "first_divergence_index",
                                                              "metrics")}
                                          for r in sw["results"]]}
        out.append(check)
    return out


def summary(examples, width=160):
    lines = []
    for e in examples:
        head = f"feature {e['feature_id']} ({e['label']}) {e['mode']}"
        if e["mode"] == "amplify":
            head += f" x{e['strength']:g}"
        lines += [head, f"  prompt:   {e['prompt']}",
                  f"  baseline: {e['baseline'][len(e['prompt']):][:width]!r}",
                  f"  steered:  {e['steered'][len(e['prompt']):][:width]!r}"]
        if e["first_divergence_index"] is None:
            lines.append("  the two outputs are identical")
        else:
            lines.append(f"  first divergence at new token {e['first_divergence_index']}; "
                         f"{e['tokens_changed']} of {e['tokens_compared_from_divergence']} "
                         f"tokens from there differ ({e['percent_changed_from_divergence']}%)")
        if e["sweep"]:
            lines.append("  sweep (strength: degenerate?):")
            for r in e["sweep"]["results"]:
                flag = f"DEGENERATE - {r['reason']}" if r["degenerate"] else "ok"
                lines.append(f"    {r['strength']:g}: {flag}")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=str(REPO_ROOT / "demo" / "phase4_examples.json"))
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--no-sweep", action="store_true", help="skip the strength sweeps (faster)")
    a = p.parse_args(argv)
    db, ckpt = os.environ.get("NEURONLENS_DB"), os.environ.get("NEURONLENS_CKPT")
    if not db or not ckpt:
        sys.exit("set NEURONLENS_DB and NEURONLENS_CKPT (real mode); see the header of this file")
    try:
        con = open_readonly(resolve(db))
    except OSError as e:
        sys.exit(f"error: {e}")
    if not resolve(ckpt).is_file():
        sys.exit(f"error: checkpoint not found: {resolve(ckpt)}")
    import torch
    from neuronlens.runtime import Runtime
    rt = Runtime.load(str(resolve(ckpt)), "cuda" if torch.cuda.is_available() else "cpu")
    try:
        examples = build_examples(rt, con, seed=a.seed, max_new_tokens=a.max_new_tokens,
                                  sweep=not a.no_sweep)
        run_id = con.execute("SELECT value FROM meta WHERE key='run_id'").fetchone()
    finally:
        con.close()
    result = {"note": "Steering is a test of what a feature does, not a fix. These outputs shift "
                      "the text; they are not an improvement.",
              "run_id": run_id[0] if run_id else None, "examples": examples}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(summary(examples))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
