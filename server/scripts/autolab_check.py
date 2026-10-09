#!/usr/bin/env python3
"""Auto-Lab check: runs the same chain as the Auto-Lab UI against a running server and prints
which features x strengths worked, so we can decide which goal buttons ship.

    python server/scripts/autolab_check.py --prompt "The band walked on stage and" --goal "music"
    python server/scripts/autolab_check.py --goals-file goals.txt --repeat 5 --url http://127.0.0.1:8000

Chain (seed 42): POST /features/search with the goal words (top 5, max_density 0.01), then
POST /steer/sweep with with_activations for each of the 5 candidates. Each cell is scored
with the rule below.

Recommendation (rule v2), strength 0 (the control) left out: a worked cell is "stable" if
the strength next to it (before or after) also worked. Only features with a stable worked
cell can be recommended; among them, the one with the most worked cells (tie: the higher
rise at its best cell), at its lowest stable strength. Worked cells but none stable: "Worked
at one strength only, not reliable", no recommendation. No worked cell: "Nothing worked".
A goal ships if it has a recommendation on at least 2 of its prompts.

    python server/scripts/autolab_check.py --replay demo/autolab_examples.json
re-scores a recorded file with the current rule, without a server or a GPU.

Steering is a test, not a fix: a cell that "worked" means the feature fired more on the
finished text at that strength without the text looping. It does not mean the text is
better. The thresholds are tuned on a few prompts and are provisional until the laptop test.

--goals-file: one goal per line, "goal words" or "goal words | prompt" (the prompt defaults
to --prompt). Blank lines and lines starting with # are skipped.
Only the standard library is used, so it runs anywhere Python 3.9+ does.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# ---------- Auto-Lab rule (the UI uses the same constants) ----------
SEED = 42
MAX_NEW_TOKENS = 60
CANDIDATES = 5                       # features taken from search
MAX_DENSITY = 0.01                   # search hides features firing on more than 1% of tokens
STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]   # the /steer/sweep default; 0 is the control
RISE_MIN = 3.0                       # worked: steered - unsteered >= 3.0 ...
RATIO_MIN = 2.0                      # ... and steered >= 2x unsteered, and the text does not loop
WORKED, NO_EFFECT, BREAKS = "worked", "no effect", "breaks down"
MARK = {WORKED: "✓", NO_EFFECT: "✗", BREAKS: "⚠️"}
# --------------------------------------------------------------------

SEARCH_TIMEOUT = 30                  # the UI's limits (web/src/lib/api/client.ts)
SWEEP_TIMEOUT = 120
TARGET_S, WARN_S = 35.0, 40.0        # one chain: target, and "tell us right away"


def score_cell(unsteered, steered, degenerate):
    """worked / no effect / breaks down for one feature at one strength."""
    if degenerate:
        return BREAKS
    rise = steered - unsteered
    if rise >= RISE_MIN and steered >= RATIO_MIN * unsteered:
        return WORKED
    return NO_EFFECT


RECOMMENDED, NOT_RELIABLE, NOTHING = "recommended", "not reliable", "nothing worked"
STATUS_TEXT = {NOT_RELIABLE: "Worked at one strength only, not reliable",
               NOTHING: "Nothing worked"}
SHIP_MIN_PROMPTS = 2                 # a goal ships if it has a recommendation on 2 of 3 prompts


def stable_cells(cells):
    """(stable, worked): the worked cells whose neighbour strength (before or after,
    strength 0 left out) also worked, and all worked cells, in strength order."""
    steered = sorted((c for c in cells if c["strength"] > 0), key=lambda c: c["strength"])
    ok = [c["verdict"] == WORKED for c in steered]
    stable = [c for i, c in enumerate(steered)
              if ok[i] and ((i > 0 and ok[i - 1]) or (i + 1 < len(ok) and ok[i + 1]))]
    return stable, [c for c, o in zip(steered, ok) if o]


def recommend(rows):
    """Rule v2. rows: [{feature_id, label, cells: [{strength, verdict, rise}]}] in search
    order. Only features with a stable worked cell can be recommended; among them the one
    with the most worked cells (tie: the higher rise at its best worked cell, then search
    order), at its lowest stable strength. Returns {"status"}, plus feature_id, label,
    strength, rise and worked_cells when the status is "recommended"."""
    best, any_worked = None, False
    for r in rows:
        stable, worked = stable_cells(r["cells"])
        any_worked = any_worked or bool(worked)
        if not stable:
            continue
        key = (len(worked), max(c["rise"] for c in worked))
        if best is None or key > best[0]:
            best = (key, r, stable[0])
    if best:
        (n, _), r, c = best
        return {"status": RECOMMENDED, "feature_id": r["feature_id"], "label": r["label"],
                "strength": c["strength"], "rise": c["rise"], "worked_cells": n}
    return {"status": NOT_RELIABLE if any_worked else NOTHING}


def ship_summary(results):
    """{goal words: {"ship", "recommended_on", "prompts": {prompt: recommendation}}}. A goal
    ships if it has a recommendation on at least SHIP_MIN_PROMPTS of its prompts."""
    goals = {}
    for r in results:
        g = goals.setdefault(r["goal"], {"prompts": {}})
        g["prompts"].setdefault(r["prompt"], r["recommendation"])
    for g in goals.values():
        n = sum(1 for rec in g["prompts"].values() if rec and rec["status"] == RECOMMENDED)
        g["ship"], g["recommended_on"] = n >= SHIP_MIN_PROMPTS, n
    return goals


def rule_dict():
    return {"version": 2, "seed": SEED, "max_new_tokens": MAX_NEW_TOKENS,
            "candidates": CANDIDATES, "max_density": MAX_DENSITY, "strengths": STRENGTHS,
            "rise_min": RISE_MIN, "ratio_min": RATIO_MIN, "ship_min_prompts": SHIP_MIN_PROMPTS}


def parse_goals(text, default_prompt):
    goals = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        goal, sep, prompt = (p.strip() for p in line.partition("|"))
        prompt = prompt if sep else default_prompt
        if not goal or not prompt:
            raise SystemExit(f"bad goals line (need goal words and a prompt): {line!r}")
        goals.append((goal, prompt))
    return goals


# ---------- HTTP ----------

class CallFailed(Exception):
    pass


def call(base, path, body=None, timeout=10):
    """(json, seconds). Raises CallFailed with the status and the server's error code."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(base.rstrip("/") + path, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="GET" if body is None else "POST")
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8")), time.perf_counter() - t
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode("utf-8")).get("error", {})
        except ValueError:
            err = {}
        raise CallFailed(f"{path}: HTTP {e.code} {err.get('code', '')} {err.get('message', '')}"
                         .strip()) from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        kind = "timed out" if "timed out" in str(e) else str(e)
        raise CallFailed(f"{path}: {kind} after {time.perf_counter() - t:.1f}s") from e


# ---------- the chain ----------

def run_chain(base, goal, prompt):
    """One Auto-Lab chain. Returns a dict with rows, recommendation, timings and failures,
    and under "raw" every request and response (what --record saves)."""
    t0 = time.perf_counter()
    out = {"goal": goal, "prompt": prompt, "rows": [], "failures": [], "calls": []}
    search_req = {"query": goal, "top_n": CANDIDATES, "max_density": MAX_DENSITY}
    try:
        search, s = call(base, "/api/v1/features/search", search_req, SEARCH_TIMEOUT)
        out["calls"].append(("search", s))
        out["raw"] = {"goal": goal, "prompt": prompt, "search_request": search_req,
                      "search": search, "sweeps": []}
    except CallFailed as e:
        out["failures"].append(str(e))
        out["total_s"] = time.perf_counter() - t0
        out["recommendation"] = None
        return out
    out["simulated"] = search.get("simulated")
    for f in search["features"][:CANDIDATES]:
        body = {"prompt": prompt, "feature_id": f["feature_id"], "strengths": STRENGTHS,
                "seed": SEED, "max_new_tokens": MAX_NEW_TOKENS, "with_activations": True}
        try:
            sweep, s = call(base, "/api/v1/steer/sweep", body, SWEEP_TIMEOUT)
            out["calls"].append((f"sweep #{f['feature_id']}", s))
            out["raw"]["sweeps"].append({"request": body, "response": sweep})
        except CallFailed as e:
            out["failures"].append(str(e))
            continue
        out["rows"].append(score_sweep(f, sweep, out["failures"]))
    out["recommendation"] = recommend(out["rows"])
    out["total_s"] = time.perf_counter() - t0
    return out


def score_sweep(feature, sweep, failures):
    """One table row from a search feature and its /steer/sweep response."""
    cells = []
    for r in sweep["results"]:
        if r.get("target_activation_steered") is None:
            failures.append("/steer/sweep returned no activations: the server is older than "
                            "with_activations")
            break
        u, st = r["target_activation_baseline"], r["target_activation_steered"]
        cells.append({"strength": r["strength"], "unsteered": u, "steered": st, "rise": st - u,
                      "degenerate": r["degenerate"], "reason": r["reason"],
                      "verdict": score_cell(u, st, r["degenerate"])})
    return {"feature_id": feature["feature_id"], "label": feature.get("label"),
            "search_activation": feature.get("max_activation"), "cells": cells}


def replay(path):
    """(file data, results) re-scored from a --record file with the current rule, without a
    server: the file keeps the raw search and sweep responses, so a rule change needs no
    rerun."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    results = []
    for g in data["goals"]:
        out = {"goal": g["goal"], "prompt": g["prompt"], "rows": [], "failures": [],
               "calls": [], "total_s": 0.0, "simulated": g["search"].get("simulated")}
        sweeps = {sw["request"]["feature_id"]: sw["response"] for sw in g["sweeps"]}
        for f in g["search"]["features"][:CANDIDATES]:
            if f["feature_id"] in sweeps:
                out["rows"].append(score_sweep(f, sweeps[f["feature_id"]], out["failures"]))
            else:
                out["failures"].append(f"no recorded sweep for #{f['feature_id']}")
        out["recommendation"] = recommend(out["rows"])
        results.append(out)
    return data, results


def show(res, verbose=False):
    print(f"\nGoal: {res['goal']!r}   Prompt: {res['prompt']!r}   seed {SEED}"
          + ("   [mock: simulated data]" if res.get("simulated") else ""))
    if res["rows"]:
        head = f"  {'feature':<36}" + "".join(f"{s:>7g}" for s in STRENGTHS)
        print(head)
        for r in res["rows"]:
            name = f"#{r['feature_id']} {r['label'] or 'unlabelled'}"[:35]
            marks = {c["strength"]: MARK[c["verdict"]] for c in r["cells"]}
            # ⚠️ is two code points wide in most terminals; pad so the columns line up
            print(f"  {name:<36}" + "".join(
                f"{marks.get(s, '-'):>{6 if marks.get(s) == MARK[BREAKS] else 7}}"
                for s in STRENGTHS))
            if verbose:
                for c in r["cells"]:
                    print(f"      {c['strength']:>4g}: unsteered {c['unsteered']:.2f}  steered "
                          f"{c['steered']:.2f}  rise {c['rise']:+.2f}  {c['verdict']}"
                          + (f" ({c['reason']})" if c["reason"] else ""))
        print(f"  ✓ worked (rise ≥ {RISE_MIN:g} and ≥ {RATIO_MIN:g}× unsteered, no loop)   "
              f"✗ no effect   ⚠️ breaks down (loop)   strength 0 is the unsteered control")
    rec = res["recommendation"]
    if rec and rec["status"] == RECOMMENDED:
        print(f"  Recommendation: #{rec['feature_id']} {rec['label']} at {rec['strength']:g} "
              f"(rise {rec['rise']:+.2f}, {rec['worked_cells']} ✓ cells). Tuned on this prompt "
              "and seed; a test, not a fix.")
    elif rec:
        print(f"  Recommendation: none. {STATUS_TEXT[rec['status']]}.")
    for f in res["failures"]:
        print(f"  FAILED: {f}")
    calls = ", ".join(f"{n} {s:.1f}s" for n, s in res["calls"])
    print(f"  Total {res['total_s']:.1f}s" + (f"  ({calls})" if verbose and calls else ""))


def print_ships(results):
    print(f"\nButtons (a goal ships with a recommendation on at least {SHIP_MIN_PROMPTS} "
          "prompts):")
    for goal, g in ship_summary(results).items():
        print(f"  {'SHIP     ' if g['ship'] else 'NO BUTTON'}  {goal!r}: recommended on "
              f"{g['recommended_on']} of {len(g['prompts'])} prompt(s)")
        for prompt, rec in g["prompts"].items():
            if rec is None:
                what = "failed"
            elif rec["status"] == RECOMMENDED:
                what = f"#{rec['feature_id']} {rec['label']} at {rec['strength']:g}"
            else:
                what = STATUS_TEXT[rec["status"]]
            print(f"      {prompt!r}: {what}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    # 127.0.0.1, not localhost: on Windows, Python's "localhost" tries IPv6 first and adds
    # about 2 s to every call, which would swamp the timing this script measures
    p.add_argument("--url", default="http://127.0.0.1:8000", help="server base URL")
    p.add_argument("--prompt", default="The band walked on stage and")
    p.add_argument("--goal", help="goal words, e.g. 'music'")
    p.add_argument("--goals-file", help="one goal per line: 'goal words' or 'goal words | prompt'")
    p.add_argument("--repeat", type=int, default=1, help="run every chain N times back to back")
    p.add_argument("--json", help="also write all results to this file")
    p.add_argument("--record", help="real server only: save every search and sweep response "
                                    "of the first run of each goal, for the mock "
                                    "(demo/autolab_examples.json)")
    p.add_argument("-v", "--verbose", action="store_true", help="numbers per cell and per call")
    p.add_argument("--replay", help="re-score a --record file with the current rule; no server")
    a = p.parse_args(argv)
    if a.replay:
        data, results = replay(a.replay)
        print(f"Replay of {a.replay}: run {data.get('run_id')}, recorded with rule "
              f"v{data.get('rule', {}).get('version', 1)}, scored with rule "
              f"v{rule_dict()['version']}")
        for res in results:
            show(res, a.verbose)
        print_ships(results)
        return 0
    if bool(a.goal) == bool(a.goals_file):
        p.error("give --goal or --goals-file (or --replay FILE)")
    goals = ([(a.goal, a.prompt)] if a.goal else
             parse_goals(open(a.goals_file, encoding="utf-8").read(), a.prompt))
    try:
        health, _ = call(a.url, "/api/v1/health")
    except CallFailed as e:
        raise SystemExit(f"cannot reach the server at {a.url}: {e}")
    print(f"Server {a.url}: {health['mode']} mode, run {health['run_id']}, "
          f"{health['model_name']} layer {health['hook_layer']}")
    if a.record and health["mode"] != "real":
        raise SystemExit("--record needs a real-mode server: the mock must replay real outputs")

    results, t_all = [], time.perf_counter()
    for i in range(a.repeat):
        for goal, prompt in goals:
            res = run_chain(a.url, goal, prompt)
            res["repeat"] = i + 1
            results.append(res)
            if i == 0 or res["failures"] or a.verbose:
                show(res, a.verbose)
            else:
                print(f"  repeat {i + 1}: {goal!r} {res['total_s']:.1f}s")
            if res["total_s"] > WARN_S:
                print(f"  WARNING: this chain took {res['total_s']:.1f}s, over {WARN_S:g}s. "
                      "Tell Prabodh right away.")
    total = time.perf_counter() - t_all

    times = [r["total_s"] for r in results]
    failed = [f for r in results for f in r["failures"]]
    print(f"\n{len(results)} chain(s) in {total:.1f}s. Per chain: max {max(times):.1f}s, "
          f"mean {sum(times) / len(times):.1f}s (target under {TARGET_S:g}s). "
          f"{len(failed)} failed or timed-out call(s).")
    if len(goals) > 1 or a.repeat > 1:
        print_ships(results[:len(goals)])
    if a.record:
        firsts = [r for r in results[:len(goals)] if r.get("raw") and not r["failures"]]
        with open(a.record, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"note": "Auto-Lab chains recorded on the real server by "
                               "server/scripts/autolab_check.py --record. The mock replays "
                               "them (marked simulated); do not edit by hand.",
                       "run_id": health["run_id"], "rule": rule_dict(),
                       "goals": [r["raw"] for r in firsts]}, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
        skipped = len(goals) - len(firsts)
        print(f"recorded {len(firsts)} goal(s) to {a.record}"
              + (f"; {skipped} skipped because a call failed" if skipped else ""))
    for r in results:
        r.pop("raw", None)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"url": a.url, "health": health, "rule": rule_dict(),
                       "total_s": total, "results": results}, fh, indent=1, ensure_ascii=False)
        print(f"wrote {a.json}")
    return 1 if failed else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
