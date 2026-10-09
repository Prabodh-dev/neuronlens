#!/usr/bin/env python3
"""Auto-Lab check: runs the same chain as the Auto-Lab UI against a running server and prints
which features x strengths worked, so we can decide which goal buttons ship.

    python server/scripts/autolab_check.py --prompt "The band walked on stage and" --goal "music"
    python server/scripts/autolab_check.py --goals-file goals.txt --repeat 5 --url http://127.0.0.1:8000

Chain (seed 42): POST /features/search with the goal words and a leading space (top 8,
max_density 0.01); drop the filler features (demo/autolab_fillers.json) and keep the first
5; fewer than 3 left is "no good match". Then POST /steer/sweep with with_activations for
each candidate. Each cell is scored with the rule below.

    python server/scripts/autolab_check.py --find-fillers demo/autolab_fillers.json
(real server) searches 200 common words the same way and lists the features that come back
for more than 10% of them: the filler list.

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
import datetime
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------- Auto-Lab rule (the UI uses the same constants) ----------
SEED = 42
MAX_NEW_TOKENS = 60
SEARCH_TOP_N = 8                     # search for the goal with a leading space, top 8 ...
CANDIDATES = 5                       # ... drop the filler features, keep the first 5
MIN_CANDIDATES = 3                   # fewer than 3 left: "no good match"
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

# ---------- the filler list (--find-fillers) ----------
REPO_ROOT = Path(__file__).resolve().parents[2]
FILLERS_PATH = REPO_ROOT / "demo" / "autolab_fillers.json"
FILLER_SHARE = 0.10                  # a filler shows up for more than 10% of common words
# 200 common English words on unrelated topics, searched like a goal (leading space, top 8)
COMMON_WORDS = """
time year people way day man thing woman life child world school state family student
group country problem hand part place case week company system program question work
government number night point home water room mother area money story fact month lot right
study book eye job word business issue side kind head house service friend father power
hour game line end member law car city community name president team minute idea kid body
information back parent face others level office door health person art war history party
result change morning reason research girl guy moment air teacher force education foot boy
age policy music market sense nation plan college interest death experience effect class
control care field development role effort rate heart drug show leader light voice wife
police mind price report decision son view relationship town road arm difference value
building action model season society tax director position player record paper space
ground form event official matter center couple site project activity star table need
court oil situation cost industry figure street image phone data picture practice piece
land product doctor wall patient worker news test movie north love support technology
step baby computer type attention film tree
""".split()


def score_cell(unsteered, steered, degenerate):
    """worked / no effect / breaks down for one feature at one strength."""
    if degenerate:
        return BREAKS
    rise = steered - unsteered
    if rise >= RISE_MIN and steered >= RATIO_MIN * unsteered:
        return WORKED
    return NO_EFFECT


RECOMMENDED, NOT_RELIABLE, NOTHING = "recommended", "not reliable", "nothing worked"
NO_MATCH = "no good match"
STATUS_TEXT = {NOT_RELIABLE: "Worked at one strength only, not reliable",
               NOTHING: "Nothing worked",
               NO_MATCH: f"No good match (fewer than {MIN_CANDIDATES} features left after "
                         "dropping the filler features)"}
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


def select_candidates(features, filler_ids):
    """(kept, dropped): the search features in order with the filler features dropped, the
    first CANDIDATES of them kept; dropped lists the filler ids that were in the results."""
    dropped = [f["feature_id"] for f in features if f["feature_id"] in filler_ids]
    kept = [f for f in features if f["feature_id"] not in filler_ids][:CANDIDATES]
    return kept, dropped


def load_fillers(path=FILLERS_PATH):
    """The filler ids from --find-fillers output, or an empty set if the file is missing."""
    try:
        with open(path, encoding="utf-8") as fh:
            return {f["feature_id"] for f in json.load(fh)["fillers"]}
    except OSError:
        return set()


def rule_dict(filler_ids=()):
    return {"version": 3, "seed": SEED, "max_new_tokens": MAX_NEW_TOKENS,
            "leading_space": True, "search_top_n": SEARCH_TOP_N, "candidates": CANDIDATES,
            "min_candidates": MIN_CANDIDATES, "filler_ids": sorted(filler_ids),
            "max_density": MAX_DENSITY, "strengths": STRENGTHS, "rise_min": RISE_MIN,
            "ratio_min": RATIO_MIN, "ship_min_prompts": SHIP_MIN_PROMPTS}


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

def search_request(goal, sort="activation"):
    """The UI's search request: the goal words with a leading space (so the first word gets
    the token it has mid-sentence), top SEARCH_TOP_N. sort is an experiment (--sort)."""
    req = {"query": " " + goal.strip(), "top_n": SEARCH_TOP_N, "max_density": MAX_DENSITY}
    if sort != "activation":
        req["sort"] = sort
    return req


def run_chain(base, goal, prompt, sort="activation", search_only=False, filler_ids=()):
    """One Auto-Lab chain. Returns a dict with rows, recommendation, timings and failures,
    and under "raw" every request and response (what --record saves). The defaults are the
    UI chain; sort and search_only are for experiments."""
    t0 = time.perf_counter()
    out = {"goal": goal, "prompt": prompt, "rows": [], "failures": [], "calls": []}
    search_req = search_request(goal, sort)
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
    out["search"] = search
    kept, out["dropped_fillers"] = select_candidates(search["features"], filler_ids)
    if search_only:
        out["recommendation"] = None
        out["total_s"] = time.perf_counter() - t0
        return out
    if len(kept) < MIN_CANDIDATES:
        out["recommendation"] = {"status": NO_MATCH}
        out["total_s"] = time.perf_counter() - t0
        return out
    for f in kept:
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


def replay(path, filler_ids=()):
    """(file data, results) re-scored from a --record file with the current rule and the
    same candidate selection as the chain, without a server: the file keeps the raw search
    and sweep responses, so a rule change needs no rerun. A recording made with another
    search (older files: no leading space, top 5) is scored on the features it has."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    results = []
    for g in data["goals"]:
        out = {"goal": g["goal"], "prompt": g["prompt"], "rows": [], "failures": [],
               "calls": [], "total_s": 0.0, "simulated": g["search"].get("simulated"),
               "search": g["search"]}
        kept, out["dropped_fillers"] = select_candidates(g["search"]["features"], filler_ids)
        if len(kept) < MIN_CANDIDATES:
            out["recommendation"] = {"status": NO_MATCH}
            results.append(out)
            continue
        sweeps = {sw["request"]["feature_id"]: sw["response"] for sw in g["sweeps"]}
        for f in kept:
            if f["feature_id"] in sweeps:
                out["rows"].append(score_sweep(f, sweeps[f["feature_id"]], out["failures"]))
            else:
                out["failures"].append(f"no recorded sweep for #{f['feature_id']}")
        out["recommendation"] = recommend(out["rows"])
        results.append(out)
    return data, results


def find_fillers(base, words=COMMON_WORDS):
    """Searches every word like a goal (leading space, top SEARCH_TOP_N) and counts how
    often each feature comes back. Returns (counts, labels, failures)."""
    counts, labels, failures = {}, {}, []
    for w in words:
        try:
            res, _ = call(base, "/api/v1/features/search", search_request(w), SEARCH_TIMEOUT)
        except CallFailed as e:
            failures.append(str(e))
            continue
        for f in res["features"]:
            counts[f["feature_id"]] = counts.get(f["feature_id"], 0) + 1
            labels[f["feature_id"]] = f.get("label")
    return counts, labels, failures


def filler_list(counts, labels, n_words, share=FILLER_SHARE):
    """(fillers, near_misses): features that came back for more than `share` of the words,
    most frequent first, and the next 5 below the cutoff (to show the gap)."""
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    row = lambda fid, n: {"feature_id": fid, "label": labels.get(fid), "count": n,  # noqa: E731
                          "share": round(n / n_words, 3)}
    above = [row(fid, n) for fid, n in ranked if n > share * n_words]
    below = [row(fid, n) for fid, n in ranked if n <= share * n_words][:5]
    return above, below


def fillers(results, min_goals=2):
    """{feature_id: info} for features in the search results for at least min_goals
    different goal words, with the query token each hit peaked on (token 0 is the first
    query word, right after the newline prefix) and its relative_to_max."""
    seen = {}
    for r in results:
        s = r.get("search") or {}
        toks = s.get("tokens", [])
        for f in s.get("features", []):
            e = seen.setdefault(f["feature_id"], {"label": f.get("label"), "goals": {},
                                                  "density": f.get("density")})
            pk = f.get("peak_token")
            e["goals"].setdefault(r["goal"], (pk, toks[pk] if pk is not None and pk < len(toks)
                                              else None, f.get("relative_to_max")))
    return {fid: e for fid, e in seen.items() if len(e["goals"]) >= min_goals}


def show(res, verbose=False):
    print(f"\nGoal: {res['goal']!r}   Prompt: {res['prompt']!r}   seed {SEED}"
          + ("   [mock: simulated data]" if res.get("simulated") else ""))
    if res.get("dropped_fillers"):
        print(f"  filler features dropped: {', '.join(f'#{i}' for i in res['dropped_fillers'])}")
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
    if res["recommendation"] is None and res.get("search"):  # --search-only
        s = res["search"]
        toks = s.get("tokens", [])
        print(f"  query tokens: {toks}")
        for f in s["features"]:
            pk = f.get("peak_token")
            tok = toks[pk] if pk is not None and pk < len(toks) else None
            print(f"  #{f['feature_id']:<6} {str(f.get('label'))[:34]:<34} "
                  f"act {f['max_activation']:6.2f}  relative {f.get('relative_to_max')}  "
                  f"density {f.get('density', 0):.4f}  peak token {pk} {tok!r}")
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


def print_fillers(results):
    found = fillers(results)
    print("\nFeatures in the search results for 2 or more goals (peak token 0 = the first "
          "query word, right after the newline prefix):")
    if not found:
        print("  none")
    for fid, e in sorted(found.items(), key=lambda kv: (-len(kv[1]["goals"]), kv[0])):
        first = sum(1 for pk, _, _ in e["goals"].values() if pk == 0)
        print(f"  #{fid} {e['label']!r}  density {e['density'] or 0:.4f}: in {len(e['goals'])} "
              f"goals, peaked on token 0 in {first} of them")
        for goal, (pk, tok, rel) in e["goals"].items():
            print(f"      {goal!r}: peak token {pk} {tok!r}, relative_to_max {rel}")


def run_find_fillers(base, health, out_path):
    counts, labels, failures = find_fillers(base)
    n = len(COMMON_WORDS) - len(failures)
    if not n:
        raise SystemExit("every search failed: " + "; ".join(failures[:3]))
    above, below = filler_list(counts, labels, n)
    cut = FILLER_SHARE * n
    print(f"\n{n} common words searched (leading space, top {SEARCH_TOP_N}, max_density "
          f"{MAX_DENSITY}). Cutoff: more than {FILLER_SHARE:.0%} of the words, so more than "
          f"{cut:g}.")
    for f in above:
        print(f"  FILLER  #{f['feature_id']:<6} {str(f['label'])[:36]:<36} {f['count']:>4} words "
              f"({f['share']:.0%})")
    for f in below:
        print(f"  below   #{f['feature_id']:<6} {str(f['label'])[:36]:<36} {f['count']:>4} words "
              f"({f['share']:.0%})")
    if above and below:
        print(f"  Gap at the cutoff: {above[-1]['count']} -> {below[0]['count']} words.")
    for e in failures:
        print(f"  FAILED: {e}")
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"note": "Filler features: features that come back from /features/search for "
                           "many unrelated words. Written by server/scripts/autolab_check.py "
                           "--find-fillers on the real server; Auto-Lab drops them from the "
                           "candidates. Do not edit by hand.",
                   "date": datetime.date.today().isoformat(), "run_id": health["run_id"],
                   "words": n, "search": {"leading_space": True, "top_n": SEARCH_TOP_N,
                                          "max_density": MAX_DENSITY},
                   "cutoff": {"share": FILLER_SHARE, "more_than_words": cut},
                   "fillers": above, "near_misses": below}, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print(f"wrote {len(above)} filler(s) to {out_path}")
    return 1 if failures else 0


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
    p.add_argument("--fillers", action="store_true",
                   help="also list features in the search results for 2+ goals and the query "
                        "token they peaked on (works with --replay)")
    p.add_argument("--search-only", action="store_true",
                   help="experiment: only the search step, with peak tokens (no sweeps, fast)")
    p.add_argument("--sort", choices=["activation", "relative"], default="activation",
                   help="experiment: the search sort (the UI uses activation)")
    p.add_argument("--fillers-file", default=str(FILLERS_PATH),
                   help="the filler list dropped from the candidates (default "
                        "demo/autolab_fillers.json)")
    p.add_argument("--find-fillers", metavar="OUT",
                   help="real server only: search the 200 common words and write the filler "
                        "list to OUT (normally demo/autolab_fillers.json)")
    a = p.parse_args(argv)
    experiment = a.search_only or a.sort != "activation"
    if a.replay and experiment:
        p.error("--replay re-scores the recorded UI chain; the experiment options need a server")
    if a.record and experiment:
        p.error("--record saves the UI chain only; drop the experiment options")
    filler_ids = load_fillers(a.fillers_file)
    if not a.find_fillers:
        print(f"Filler list: {len(filler_ids)} feature(s) from {a.fillers_file}"
              + ("" if filler_ids else " (missing or empty: nothing is dropped)"))
    if a.replay:
        data, results = replay(a.replay, filler_ids)
        print(f"Replay of {a.replay}: run {data.get('run_id')}, recorded with rule "
              f"v{data.get('rule', {}).get('version', 1)}, scored with rule "
              f"v{rule_dict()['version']}")
        for res in results:
            show(res, a.verbose)
        print_ships(results)
        if a.fillers:
            print_fillers(results)
        return 0
    if not a.find_fillers and bool(a.goal) == bool(a.goals_file):
        p.error("give --goal or --goals-file (or --replay FILE, or --find-fillers OUT)")
    try:
        health, _ = call(a.url, "/api/v1/health")
    except CallFailed as e:
        raise SystemExit(f"cannot reach the server at {a.url}: {e}")
    print(f"Server {a.url}: {health['mode']} mode, run {health['run_id']}, "
          f"{health['model_name']} layer {health['hook_layer']}")
    if (a.record or a.find_fillers) and health["mode"] != "real":
        raise SystemExit("--record and --find-fillers need a real-mode server: the mock must "
                         "replay real outputs")
    if a.find_fillers:
        return run_find_fillers(a.url, health, a.find_fillers)
    goals = ([(a.goal, a.prompt)] if a.goal else
             parse_goals(open(a.goals_file, encoding="utf-8").read(), a.prompt))

    results, t_all = [], time.perf_counter()
    for i in range(a.repeat):
        for goal, prompt in goals:
            res = run_chain(a.url, goal, prompt, a.sort, a.search_only, filler_ids)
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
    if not a.search_only and (len(goals) > 1 or a.repeat > 1):
        print_ships(results[:len(goals)])
    if a.fillers:
        print_fillers(results[:len(goals)])
    if a.record:
        firsts = [r for r in results[:len(goals)] if r.get("raw") and not r["failures"]]
        with open(a.record, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"note": "Auto-Lab chains recorded on the real server by "
                               "server/scripts/autolab_check.py --record. The mock replays "
                               "them (marked simulated); do not edit by hand.",
                       "run_id": health["run_id"], "rule": rule_dict(filler_ids),
                       "goals": [r["raw"] for r in firsts]}, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
        skipped = len(goals) - len(firsts)
        print(f"recorded {len(firsts)} goal(s) to {a.record}"
              + (f"; {skipped} skipped because a call failed" if skipped else ""))
    for r in results:
        r.pop("raw", None)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"url": a.url, "health": health, "rule": rule_dict(filler_ids),
                       "total_s": total, "results": results}, fh, indent=1, ensure_ascii=False)
        print(f"wrote {a.json}")
    return 1 if failed else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
