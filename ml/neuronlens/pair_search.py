"""Automated search for a W5 demo example: two features that both read "higher" when steered
together on the same prompt, with no loop warning.

Stage 1 (singles): every candidate feature is amplified alone at SINGLE_STRENGTH on three neutral
prompts. A candidate "works" on a prompt when its activation verdict is "higher", the steered text
is not degenerate and it differs from the baseline; it qualifies when it works on at least
MIN_WORKING_PROMPTS prompts.
Stage 2 (pairs): pairs of qualifying candidates are steered together (multi_steering_check) on the
prompts where both singles worked, at each PAIR_STRENGTHS value applied to both. A pair "works"
when both verdicts are "higher", the text is not degenerate and it differs from the baseline.

This is a search for a demonstration, a test of what the features do together. It does not show
that a combination is better, and a combination that works on one prompt is one example.
The database is only read.
"""
import itertools
import json
import random
import time

from . import steering_checks as sc

PROMPTS = ("The man walked into the room and", "She opened the letter and read",
           "The reporter wrote that the")
KNOWN_FEATURES = (3446, 1863, 13, 3587)       # always tested (when usable)
SINGLE_STRENGTH = 0.5
PAIR_STRENGTHS = (0.25, 0.5)
DENSITY_MIN, DENSITY_MAX = 0.0005, 0.02       # inclusive; fraction of tokens a feature fires on
MIN_WORKING_PROMPTS = 2
DEFAULT_CANDIDATES = 150
DEFAULT_MAX_PAIRS = 60
PROGRESS_EVERY = 10                           # generations between progress lines
SINGLE_TEXT_CHARS, PAIR_TEXT_CHARS = 100, 200


# ------------------------------------------------------------------ candidates

def feature_rows(con):
    """Rows of the features table (read only) as dicts."""
    cur = con.execute("SELECT feature_id, label, label_status, density, max_activation, is_dead "
                      "FROM features ORDER BY feature_id")
    return [dict(zip(("feature_id", "label", "label_status", "density", "max_activation", "is_dead"), r))
            for r in cur.fetchall()]


def _usable(r):
    return not r["is_dead"] and r["max_activation"] and r["max_activation"] > 0


def sample_candidates(rows, n=DEFAULT_CANDIDATES, seed=42, known=KNOWN_FEATURES):
    """The candidate features: the known ones (when present and usable) plus a random sample, with
    the given seed, of the features that are named, not dead and have a density between
    DENSITY_MIN and DENSITY_MAX. n is the total, known ones included. Sorted by feature id."""
    by_id = {r["feature_id"]: r for r in rows}
    chosen = [by_id[f] for f in known if f in by_id and _usable(by_id[f])]
    taken = {r["feature_id"] for r in chosen}
    pool = sorted(r["feature_id"] for r in rows
                  if r["feature_id"] not in taken and _usable(r) and r["label_status"] == "named"
                  and r["density"] is not None and DENSITY_MIN <= r["density"] <= DENSITY_MAX)
    take = max(0, min(n - len(chosen), len(pool)))
    chosen += [by_id[f] for f in random.Random(seed).sample(pool, take)]
    return sorted(chosen, key=lambda r: r["feature_id"])


# ------------------------------------------------------------------ progress

class ProgressRuntime:
    """Wraps a runtime: counts generations and prints a progress line every `every` of them."""

    def __init__(self, rt, every=PROGRESS_EVERY, log=print, clock=time.time):
        self._rt, self.every, self.log, self.clock = rt, every, log, clock
        self.generations, self.stage, self.start = 0, "", clock()

    def __getattr__(self, name):          # d_sae, feature_activations, ...
        return getattr(self._rt, name)

    def generate(self, *a, **k):
        out = self._rt.generate(*a, **k)
        self.generations += 1
        if self.generations % self.every == 0:
            self.log(f"progress: {self.generations} generations, "
                     f"{self.clock() - self.start:.0f} s, {self.stage}")
        return out


# ------------------------------------------------------------------ stage 1: singles

def single_works(verdict, degenerate, differs):
    """The rule for a single feature on one prompt."""
    return verdict == "higher" and not degenerate and bool(differs)


def run_single(rt, prompt, base, row, strength, seed, max_new_tokens):
    """One steering check for one feature (the same result fields as steering_check, with the
    baseline generated once per prompt and passed in)."""
    fid, max_act = row["feature_id"], float(row["max_activation"])
    steered = rt.generate(prompt, max_new_tokens, seed, sc.SteeringSpec(fid, "amplify", strength, max_act))
    act = sc.activation_report(rt, base, steered, fid, max_act)
    deg = sc.assess_degeneration(steered.new_ids, steered.continuation)
    differs = base.text != steered.text
    return {"feature_id": fid, "label": row["label"], "prompt": prompt, "strength": strength,
            "baseline_activation": act["target_activation_baseline"],
            "steered_activation": act["target_activation_steered"],
            "activation_change": act["activation_change"], "verdict": act["activation_verdict"],
            "degenerate": deg["degenerate"], "reason": deg["reason"], "differs": differs,
            "works": single_works(act["activation_verdict"], deg["degenerate"], differs),
            "steered_text": steered.text[len(prompt):][:SINGLE_TEXT_CHARS]
            if steered.text.startswith(prompt) else steered.text[:SINGLE_TEXT_CHARS]}


def qualifies(records, min_prompts=MIN_WORKING_PROMPTS):
    """A candidate qualifies when it works on at least min_prompts prompts."""
    return sum(1 for r in records if r["works"]) >= min_prompts


def run_singles(rt, candidates, prompts, baselines, seed, max_new_tokens, strength=SINGLE_STRENGTH,
                log=print):
    """Stage 1. Returns (all records, qualifying candidates). A qualifying candidate is
    {feature_id, label, max_activation, works_on (prompts), mean_change (over those prompts),
    per_prompt (all its records)}."""
    records, qualifying = [], []
    for i, row in enumerate(candidates, 1):
        mine = [run_single(rt, p, baselines[p], row, strength, seed, max_new_tokens) for p in prompts]
        records += mine
        if qualifies(mine):
            ok = [r for r in mine if r["works"]]
            qualifying.append({"feature_id": row["feature_id"], "label": row["label"],
                               "max_activation": float(row["max_activation"]),
                               "works_on": [r["prompt"] for r in ok],
                               "mean_change": sum(r["activation_change"] for r in ok) / len(ok),
                               "per_prompt": mine})
            log(f"qualifies: {row['feature_id']} ({row['label']}) works on {len(ok)} of {len(mine)} prompts")
    return records, qualifying


# ------------------------------------------------------------------ stage 2: pairs

def select_pairs(qualifying, max_pairs=DEFAULT_MAX_PAIRS):
    """Pairs of qualifying candidates that share at least one prompt where both singles worked,
    best at the top by the sum of the two candidates' mean activation_change (ties: smaller feature
    ids), cut at max_pairs. Returns [(a, b, shared_prompts, score)]."""
    out = []
    for a, b in itertools.combinations(sorted(qualifying, key=lambda q: q["feature_id"]), 2):
        shared = [p for p in a["works_on"] if p in b["works_on"]]
        if shared:
            out.append((a, b, shared, a["mean_change"] + b["mean_change"]))
    out.sort(key=lambda t: (-t[3], t[0]["feature_id"], t[1]["feature_id"]))
    return out[:max_pairs]


def pair_works(verdicts, degenerate, differs):
    """The rule for a pair on one prompt and strength: both verdicts "higher", no loop warning,
    text differs from the baseline."""
    return len(verdicts) >= 2 and all(v == "higher" for v in verdicts) and not degenerate and bool(differs)


def run_pair(rt, a, b, prompt, strength, baseline, seed, max_new_tokens):
    steers = [{"feature_id": q["feature_id"], "strength": strength, "max_activation": q["max_activation"]}
              for q in (a, b)]
    r = sc.multi_steering_check(rt, prompt, steers, max_new_tokens, seed, baseline=baseline)
    feats = []
    for q, act in zip((a, b), r["activations"]):
        change = (act["target_activation_steered"] - act["target_activation_baseline"]) / q["max_activation"]
        feats.append({"feature_id": q["feature_id"], "label": q["label"],
                      "baseline_activation": act["target_activation_baseline"],
                      "steered_activation": act["target_activation_steered"],
                      "activation_change": change, "verdict": act["activation_verdict"]})
    works = pair_works([f["verdict"] for f in feats], r["degenerate"], r["differs"])
    return {"features": feats, "prompt": prompt, "strength": strength, "seed": seed,
            "differs": r["differs"], "first_divergence_index": r["first_divergence_index"],
            "degenerate": r["degenerate"], "reason": r["reason"], "works": works,
            "min_change": min(f["activation_change"] for f in feats),
            "sum_change": sum(f["activation_change"] for f in feats),
            "steered_text": r["steered_text"][len(prompt):][:PAIR_TEXT_CHARS]
            if r["steered_text"].startswith(prompt) else r["steered_text"][:PAIR_TEXT_CHARS]}


def rank_working(pair_runs):
    """Working pair runs, best at the top: by the smaller of the two activation_change values, then
    their sum, then feature ids, prompt and strength."""
    ok = [r for r in pair_runs if r["works"]]
    return sorted(ok, key=lambda r: (-r["min_change"], -r["sum_change"],
                                     tuple(f["feature_id"] for f in r["features"]),
                                     r["prompt"], r["strength"]))


def run_pairs(rt, pairs, baselines, seed, max_new_tokens, strengths=PAIR_STRENGTHS, log=print):
    runs = []
    for a, b, shared, _ in pairs:
        for prompt in shared:
            for strength in strengths:
                run = run_pair(rt, a, b, prompt, strength, baselines[prompt], seed, max_new_tokens)
                runs.append(run)
                if run["works"]:
                    log(f"working pair: {a['feature_id']} + {b['feature_id']} on {prompt!r} at {strength:g}")
    return runs


# ------------------------------------------------------------------ the whole search

def search(rt, rows, candidates=DEFAULT_CANDIDATES, max_pairs=DEFAULT_MAX_PAIRS, seed=42,
           prompts=PROMPTS, max_new_tokens=60, log=print, checkpoint=None, clock=time.time):
    """Run both stages on a runtime. `rows` come from feature_rows(). `checkpoint(doc)` is called
    after stage 1 so a long run keeps its singles. Returns the result document."""
    t0 = clock()
    prog = rt if isinstance(rt, ProgressRuntime) else ProgressRuntime(rt, log=log, clock=clock)
    cands = sample_candidates(rows, candidates, seed)
    prog.stage = "baselines"
    baselines = {p: prog.generate(p, max_new_tokens, seed, None) for p in prompts}
    prog.stage = f"stage 1: singles ({len(cands)} candidates)"
    log(f"stage 1: {len(cands)} candidates x {len(prompts)} prompts at strength {SINGLE_STRENGTH:g}")
    singles, qualifying = run_singles(prog, cands, prompts, baselines, seed, max_new_tokens, log=log)
    doc = {"seed": seed, "prompts": list(prompts), "single_strength": SINGLE_STRENGTH,
           "pair_strengths": list(PAIR_STRENGTHS), "max_new_tokens": max_new_tokens,
           "candidates_requested": candidates,
           "candidates": [{"feature_id": r["feature_id"], "label": r["label"]} for r in cands],
           "qualifying": [{k: v for k, v in q.items() if k != "per_prompt"} | {
               "per_prompt": [{k: r[k] for k in ("prompt", "baseline_activation", "steered_activation",
                                                 "activation_change", "verdict", "degenerate",
                                                 "differs", "works", "steered_text")}
                              for r in q["per_prompt"]]} for q in qualifying],
           "singles_tested": len(singles)}
    log(format_qualifying(qualifying) if qualifying else "no candidate qualified")
    if checkpoint:
        checkpoint(doc)
    pairs = select_pairs(qualifying, max_pairs)
    prog.stage = f"stage 2: pairs ({len(pairs)} selected)"
    log(f"stage 2: {len(pairs)} pairs selected from {len(qualifying)} qualifying candidates")
    runs = run_pairs(prog, pairs, baselines, seed, max_new_tokens, log=log)
    working = rank_working(runs)
    log(format_working(working) if working else "no pair worked")
    doc["pairs_tested"] = len(pairs)
    doc["pair_runs"] = len(runs)
    doc["working_pairs"] = working
    doc["summary"] = {"candidates_tested": len(cands), "qualifying": len(qualifying),
                      "pairs_tested": len(pairs), "pair_runs": len(runs),
                      "working_pairs": len(working), "generations": prog.generations,
                      "seconds": round(clock() - t0, 1)}
    return doc


def format_summary(doc):
    s = doc["summary"]
    lines = [f"candidates tested: {s['candidates_tested']}", f"qualifying: {s['qualifying']}",
             f"pairs tested: {s['pairs_tested']} ({s['pair_runs']} runs)",
             f"working pairs: {s['working_pairs']}", f"generations: {s['generations']}",
             f"total time: {s['seconds']:.0f} s"]
    for r in doc["working_pairs"][:5]:
        f = r["features"]
        lines.append(f"  {f[0]['feature_id']} ({f[0]['label']}) + {f[1]['feature_id']} ({f[1]['label']}) "
                     f"on {r['prompt']!r} at {r['strength']:g}: "
                     f"{f[0]['baseline_activation']:.2f}->{f[0]['steered_activation']:.2f}, "
                     f"{f[1]['baseline_activation']:.2f}->{f[1]['steered_activation']:.2f}")
    return "\n".join(lines)


def format_qualifying(qualifying):
    """Readable lines for the qualifying candidates: label, per-prompt activation (baseline to
    steered) and the start of the steered text."""
    lines = []
    for q in qualifying:
        lines.append(f"{q['feature_id']} ({q['label']}), works on {len(q['works_on'])} of "
                     f"{len(q['per_prompt'])} prompts")
        for r in q["per_prompt"]:
            mark = "works" if r["works"] else "does not work"
            lines.append(f"  {r['prompt']!r}: {r['baseline_activation']:.2f} -> {r['steered_activation']:.2f} "
                         f"({r['verdict']}; {mark}) "
                         f"{r['steered_text'][:SINGLE_TEXT_CHARS]!r}")
    return "\n".join(lines)


def format_working(working):
    """Readable lines for the working pairs, best at the top."""
    lines = []
    for i, r in enumerate(working, 1):
        a, b = r["features"]
        lines.append(f"{i}. {a['feature_id']} ({a['label']}) + {b['feature_id']} ({b['label']}) | "
                     f"prompt {r['prompt']!r} | strength {r['strength']:g} | seed {r['seed']}")
        for f in r["features"]:
            lines.append(f"     {f['feature_id']}: {f['baseline_activation']:.2f} -> "
                         f"{f['steered_activation']:.2f} (change {f['activation_change']:.2f}, {f['verdict']})")
        lines.append(f"     text: {r['steered_text'][:PAIR_TEXT_CHARS]!r}")
    return "\n".join(lines)


def write_json(doc, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")
