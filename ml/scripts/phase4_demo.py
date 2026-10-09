"""Phase 4 demo examples: steering checks on the real model (needs the GPU and the real files).

    cd ml
    export NEURONLENS_DB=../data/neuronlens.db
    export NEURONLENS_CKPT=../data/gpt2-l8-k32.pt
    PYTHONPATH=. python scripts/phase4_demo.py [--out ../demo/phase4_examples.json]

Runs the X1 to X4 functions (neuronlens.steering_checks) for three examples: feature 7
(label "upside") amplified at 0.5 on two prompts, feature 7 (label "hon") amplified at 1.0 as
a limit example, and one ablate example on feature 7.
Each amplify example also gets a strength sweep with the degeneration flags. The database is
opened read-only; the only file written is the JSON output. Steering is a test of what a
feature does, not a fix: the outputs shift the text and are not an improvement.

The prompts below are starting points. They have not been tried on the real model, so edit
EXAMPLES, run it, and keep the ones that read well.

Explore mode tries other features without editing code. It runs every --feature on every
--prompt and writes demo/explore.json (never demo/phase4_examples.json):

    PYTHONPATH=. python scripts/phase4_demo.py --feature 7 --feature 13 \
        --prompt "She looked out of the window and saw" --prompt "He turned the picture" \
        --strengths 0,0.5,0.75,1,1.5,2 [--mode amplify|ablate] [--explore-out PATH]

For each feature and prompt it prints the database label and status, then one line per
strength with the start of the generated text, the degenerate flag and the index of the
earliest token that differs from the baseline. With --mode ablate the strengths do not apply:
the feature is ablated once. Dead features are skipped with a message.
"""
import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

from neuronlens import steering_checks as sc

REPO_ROOT = Path(__file__).resolve().parents[2]

# Strengths follow what real runs showed: amplify at 3x produced repetition loops for these
# features, so the amplify examples use 0.5 to 1.0 (the API cap on strength is 3).
LIMIT_NOTE_7 = (
    "Kept as an example of a limit. In real runs this feature, amplified, shifted the text toward "
    "word fragments (for example \"ing\", \"ours\") rather than toward honking, the concept in "
    "its label, and produced repetition loops from strength 3. One feature, one prompt, one seed.")

EXAMPLES = [
    {"feature_id": 13, "expect_label": "upside", "mode": "amplify", "strength": 0.5,
     "prompt": "The acrobat hung"},
    {"feature_id": 13, "expect_label": "upside", "mode": "amplify", "strength": 0.5,
     "prompt": "He turned the picture"},
    {"feature_id": 7, "expect_label": "hon", "mode": "amplify", "strength": 1.0,
     "prompt": "She looked out of the window and saw", "note": LIMIT_NOTE_7},
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
        if ex.get("note"):
            check["note"] = ex["note"]
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


CONTINUATION_CHARS = 120


def continuation_of(prompt, text):
    """The generated part of a full text; the whole text if it does not start with the prompt."""
    return text[len(prompt):] if text.startswith(prompt) else text


def parse_strengths(value):
    """'0,0.5,1' -> [0.0, 0.5, 1.0]; used exactly as given (order and repeats kept)."""
    try:
        out = [float(x) for x in value.split(",") if x.strip()]
    except ValueError:
        raise ValueError(f"--strengths must be a comma list of numbers, got {value!r}")
    if not out or not all(x == x and abs(x) != float("inf") for x in out):
        raise ValueError(f"--strengths must be a comma list of numbers, got {value!r}")
    return out


def short_reason(reason):
    """The leading rule of a degeneration reason, without its numbers."""
    return reason.split(";")[0].split(" (")[0] if reason else ""


def _flag(degenerate, reason):
    return f"DEGENERATE: {short_reason(reason)}" if degenerate else "ok"


def explore(rt, con, features, prompts, strengths, mode="amplify", seed=SEED,
            max_new_tokens=MAX_NEW_TOKENS, log=print):
    """Run every feature on every prompt. Prints a compact view and returns the results.

    amplify: one sweep over exactly `strengths`. ablate: one ablated run (strength is ignored).
    The database is only read.
    """
    runs = []
    for fid in dict.fromkeys(features):
        row = feature_row(con, fid)
        if row is None:
            log(f"skip feature {fid}: not in the database")
            continue
        label, status, max_act, dead = row
        if dead or not max_act:
            log(f"skip feature {fid} ({label or 'no label'}): "
                f"{'dead feature' if dead else 'max_activation is 0'}")
            continue
        for prompt in prompts:
            log(f"feature {fid} | {label or '(no label)'} | {status} | prompt: {prompt!r}")
            if mode == "amplify":
                sw = sc.strength_sweep(rt, prompt, fid, max_act, strengths=strengths,
                                       max_new_tokens=max_new_tokens, seed=seed)
                base_text = sw["baseline_text"]
                results = [{"strength": r["strength"], "text": r["text"],
                            "continuation": continuation_of(prompt, r["text"]),
                            "differs_from_baseline": r["differs_from_baseline"],
                            "first_divergence_index": r["first_divergence_index"],
                            "degenerate": r["degenerate"], "reason": r["reason"],
                            "metrics": r["metrics"]} for r in sw["results"]]
            else:
                sc.validate(rt, prompt, max_new_tokens, seed, fid, "ablate", 0.0, max_act)
                base = rt.generate(prompt, max_new_tokens, seed, None)
                g = rt.generate(prompt, max_new_tokens, seed,
                                sc.SteeringSpec(fid, "ablate", 0.0, float(max_act)))
                a = sc.assess_degeneration(g.new_ids, g.continuation)
                base_text = base.text
                results = [{"strength": 0.0, "text": g.text,
                            "continuation": continuation_of(prompt, g.text),
                            "differs_from_baseline": g.text != base.text,
                            "first_divergence_index":
                                sc.compare_generations(base, g)["first_divergence_index"],
                            "degenerate": a["degenerate"], "reason": a["reason"],
                            "metrics": a["metrics"]}]
            log(f"  baseline            {continuation_of(prompt, base_text)[:CONTINUATION_CHARS]!r}")
            for r in results:
                div = "-" if r["first_divergence_index"] is None else r["first_divergence_index"]
                name = f"x{r['strength']:g}" if mode == "amplify" else "ablate"
                log(f"  {name:<7} div={div!s:<4} {_flag(r['degenerate'], r['reason']):<38}"
                    f" {r['continuation'][:CONTINUATION_CHARS]!r}")
            log("")
            runs.append({"feature_id": fid, "label": label, "label_status": status,
                         "max_activation": max_act, "prompt": prompt, "seed": seed,
                         "mode": mode, "max_new_tokens": max_new_tokens,
                         "baseline_text": base_text, "results": results})
    return runs


def load_runtime(ckpt_path):
    import torch
    from neuronlens.runtime import Runtime
    return Runtime.load(str(ckpt_path), "cuda" if torch.cuda.is_available() else "cpu")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=str(REPO_ROOT / "demo" / "phase4_examples.json"))
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--no-sweep", action="store_true", help="skip the strength sweeps (faster)")
    p.add_argument("--feature", type=int, action="append", default=[],
                   help="explore mode: feature id (repeatable)")
    p.add_argument("--prompt", action="append", default=[],
                   help="explore mode: prompt (repeatable; each feature runs on each prompt)")
    p.add_argument("--strengths", default=None,
                   help="explore mode: comma list used exactly for the sweep "
                        "(default 0,0.25,0.5,0.75,1,1.5,2)")
    p.add_argument("--mode", choices=sc.MODES, default=None,
                   help="explore mode: amplify (default) or ablate")
    p.add_argument("--explore-out", default=str(REPO_ROOT / "demo" / "explore.json"),
                   help="explore mode: output file (default demo/explore.json)")
    a = p.parse_args(argv)
    exploring = bool(a.feature or a.prompt)
    if exploring and not (a.feature and a.prompt):
        p.error("explore mode needs at least one --feature and at least one --prompt")
    if not exploring and (a.strengths is not None or a.mode is not None):
        p.error("--strengths and --mode belong to explore mode (--feature and --prompt)")
    if exploring:
        try:
            strengths = parse_strengths(a.strengths) if a.strengths else list(sc.DEFAULT_SWEEP_STRENGTHS)
        except ValueError as e:
            p.error(str(e))
        if Path(a.explore_out).resolve() == Path(a.out).resolve():
            p.error("--explore-out must not be the same file as --out (phase4_examples.json)")
    db, ckpt = os.environ.get("NEURONLENS_DB"), os.environ.get("NEURONLENS_CKPT")
    if not db or not ckpt:
        sys.exit("set NEURONLENS_DB and NEURONLENS_CKPT (real mode); see the header of this file")
    try:
        con = open_readonly(resolve(db))
    except OSError as e:
        sys.exit(f"error: {e}")
    if not resolve(ckpt).is_file():
        sys.exit(f"error: checkpoint not found: {resolve(ckpt)}")
    rt = load_runtime(resolve(ckpt))
    if exploring:
        try:
            runs = explore(rt, con, a.feature, a.prompt, strengths, a.mode or "amplify",
                           a.seed, a.max_new_tokens)
            run_id = con.execute("SELECT value FROM meta WHERE key='run_id'").fetchone()
        except sc.SteeringCheckError as e:
            sys.exit(f"error: {e}")
        finally:
            con.close()
        out = Path(a.explore_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "note": "Steering is a test of what a feature does, not a fix. These outputs shift "
                    "the text; they are not an improvement. A result marked ok can still be poor.",
            "run_id": run_id[0] if run_id else None, "mode": a.mode or "amplify",
            "strengths": strengths if (a.mode or "amplify") == "amplify" else None,
            "seed": a.seed, "max_new_tokens": a.max_new_tokens, "runs": runs},
            indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {out}")
        return
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
