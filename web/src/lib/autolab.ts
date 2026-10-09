import { api } from "./api/client";
import type { Example } from "./api/types";
import type { AutoLabGoal } from "./autolab-presets";
import { dropFillers } from "./autolab-fillers";

/* Auto-Lab: for a goal and a prompt, find the features that fired on the goal's words, try each one at
   every strength of the Every strength tab, and tag each strength. Every call runs one after another
   (the model has one lock). Same chain and rule as server/scripts/autolab_check.py (rule v2). Tags:
     worked     the feature's reading on the finished text rose clearly from the unsteered text
                (WORKED below) and the text did not loop
     none       no clear rise, no loop
     breaks     the sweep's loop tag is on
   Strength 0 is the unsteered control: it is shown, not tagged. It is a test on one prompt with one
   seed, not a fix. */

/** The "worked" thresholds, the same as RISE_MIN and RATIO_MIN in server/scripts/autolab_check.py:
 *  rise = steered - unsteered reading must be at least `minRise`, and the steered reading at least
 *  `minRatio` times the unsteered one. Provisional until the laptop test. */
export const WORKED = { minRise: 3.0, minRatio: 2 } as const;

/** The search asks for SEARCH_TOP features, drops the fillers (autolab-fillers.ts) and keeps the first
 *  AUTOLAB_CANDIDATES. With fewer than MIN_CANDIDATES left, nothing else runs ("No feature matches"). */
const SEARCH_TOP = 8;
export const MIN_CANDIDATES = 3;

export const AUTOLAB_SEED = 42;
export const AUTOLAB_LEN = 60; // the Every strength tab's default length
export const AUTOLAB_CANDIDATES = 5;
/** The Every strength tab's strengths without the unsteered 0: the columns before the first sweep lands. */
export const AUTOLAB_STRENGTHS = [0.25, 0.5, 0.75, 1, 1.5, 2];
const MAX_DENSITY = 0.01; // as W8: without it very common features fill the search

export type Tag = "worked" | "none" | "breaks";
/** One steered strength. `stable`: worked, and a neighbouring strength (0 left out) worked too. */
export type Cell = { strength: number; tag: Tag; base: number | null; steered: number | null; rise: number; stable: boolean; text: string; reason: string | null };
export type Snippet = { before: string; peak: string; after: string };
export type Candidate = { feature_id: number; label: string | null; example: Snippet | null };
/** `unsteered`: the reading on the unsteered text (the strength-0 control). */
export type Row = Candidate & { unsteered: number | null; cells: Cell[]; breaksAt: number | null };
/** Rule v2: a recommendation, or why there is none. */
export type Verdict =
  | { status: "recommended"; row: Row; cell: Cell; worked: number }
  | { status: "unreliable" } // worked cells, but none stable
  | { status: "nothing" };
export type Report = { goal: AutoLabGoal; prompt: string; seed: number; simulated: boolean; rows: Row[]; verdict: Verdict };
/** Progress: `rows` holds every row scored so far, the one being tried last. */
export type Step = { stage: "search" } | { stage: "feature"; i: number; n: number; candidates: Candidate[]; rows: Row[] };

export function worked(base: number, steered: number) {
  return steered - base >= WORKED.minRise && steered >= WORKED.minRatio * base;
}

/** Marks each worked cell whose neighbour strength (before or after, 0 left out) also worked. */
export function markStable(cells: Cell[]): Cell[] {
  const s = [...cells].sort((a, b) => a.strength - b.strength);
  return s.map((c, i) => ({ ...c, stable: c.tag === "worked" && ((i > 0 && s[i - 1].tag === "worked") || (i + 1 < s.length && s[i + 1].tag === "worked")) }));
}

/** Rule v2 (as recommend() in server/scripts/autolab_check.py). Only features with a stable worked cell
 *  can be recommended; among them the one with the most worked cells (tie: the higher rise at its best
 *  worked cell, then search order), at its lowest stable strength. */
export function pickBest(rows: Row[]): Verdict {
  let best: { n: number; rise: number; row: Row; cell: Cell } | null = null;
  let anyWorked = false;
  for (const row of rows) {
    const ok = row.cells.filter((c) => c.tag === "worked");
    anyWorked ||= ok.length > 0;
    const stable = row.cells.filter((c) => c.stable).sort((a, b) => a.strength - b.strength);
    if (!stable.length) continue;
    const n = ok.length;
    const rise = Math.max(...ok.map((c) => c.rise));
    if (!best || n > best.n || (n === best.n && rise > best.rise)) best = { n, rise, row, cell: stable[0] };
  }
  if (best) return { status: "recommended", row: best.row, cell: best.cell, worked: best.n };
  return { status: anyWorked ? "unreliable" : "nothing" };
}

const END = "<|endoftext|>"; // GPT-2's break between documents

/** A few words around the peak of a feature's top example, kept within the peak's own document. */
export function snippet(ex: Example | undefined): Snippet | null {
  if (!ex || !ex.tokens.length) return null;
  const t = ex.tokens;
  const p = Math.max(0, Math.min(t.length - 1, ex.peak_index));
  let a = Math.max(0, p - 6);
  for (let i = p - 1; i >= a; i--) if (t[i] === END) a = i + 1;
  let z = Math.min(t.length, p + 4);
  for (let i = p + 1; i < z; i++) if (t[i] === END) z = i;
  const flat = (x: string) => x.replace(/\s*\n\s*/g, " ");
  return { before: flat(t.slice(a, p).join("")), peak: t[p] === END ? "" : flat(t[p]), after: flat(t.slice(p + 1, z).join("")) };
}

/** Runs Auto-Lab. Resolves to "nomatch" when fewer than MIN_CANDIDATES features are left after the
 *  fillers are dropped (nothing else runs). A report from a mock server is marked `simulated`, and the
 *  screen labels it so. Throws on errors and when `signal` aborts. */
export async function runAutoLab(goal: AutoLabGoal, prompt: string, signal: AbortSignal, onStep: (s: Step) => void): Promise<Report | "nomatch"> {
  onStep({ stage: "search" });
  // a leading space, so the first word gets the token it has mid-sentence
  const found = await api.featureSearch({ query: " " + goal.words.trim(), top_n: SEARCH_TOP, max_density: MAX_DENSITY, sort: "activation" }, signal);
  const picked = dropFillers(found.features).slice(0, AUTOLAB_CANDIDATES);
  if (picked.length < MIN_CANDIDATES) return "nomatch";

  // each candidate's name and top example, so it is clear what fired on the words (reads only)
  const candidates: Candidate[] = [];
  for (const f of picked) {
    const d = await api.feature(f.feature_id, signal).catch(() => null);
    candidates.push({ feature_id: f.feature_id, label: d?.label ?? f.label, example: snippet(d?.examples[0]) });
  }

  const rows: Row[] = [];
  const n = candidates.length;
  for (const [k, c] of candidates.entries()) {
    onStep({ stage: "feature", i: k + 1, n, candidates, rows: [...rows] });
    // the same call as the Every strength tab (its default strengths, the same seed and length), with
    // the feature's reading on each finished text
    const sw = await api.steerSweep({ feature_id: c.feature_id, prompt, seed: AUTOLAB_SEED, max_new_tokens: AUTOLAB_LEN, with_activations: true }, signal);
    const control = sw.results.find((r) => r.strength === 0);
    let unsteered = control?.target_activation_steered ?? control?.target_activation_baseline ?? null;
    let cells: Cell[] = [];
    const partial = (): Row => ({ ...c, unsteered, cells: markStable(cells), breaksAt: cells.find((x) => x.tag === "breaks")?.strength ?? null });
    for (const r of sw.results.filter((x) => x.strength > 0)) {
      let base = r.target_activation_baseline ?? null;
      let steered = r.target_activation_steered ?? null;
      if (!r.degenerate && (base === null || steered === null) && r.differs_from_baseline) {
        // a server without with_activations: the activation check (X1) for this strength
        const chk = await api.steerCheck({ feature_id: c.feature_id, mode: "amplify", strength: r.strength, prompt, seed: AUTOLAB_SEED, max_new_tokens: AUTOLAB_LEN }, signal);
        base = chk.target_activation_baseline;
        steered = chk.target_activation_steered;
      }
      unsteered ??= base;
      // a text identical to the unsteered one cannot read higher: no effect
      const tag: Tag = r.degenerate ? "breaks" : base !== null && steered !== null && worked(base, steered) ? "worked" : "none";
      cells = [...cells, { strength: r.strength, tag, base, steered, rise: base !== null && steered !== null ? steered - base : 0, stable: false, text: r.text, reason: r.reason }];
      onStep({ stage: "feature", i: k + 1, n, candidates, rows: [...rows, partial()] });
    }
    rows.push(partial());
  }
  return { goal, prompt, seed: AUTOLAB_SEED, simulated: found.simulated, rows, verdict: pickBest(rows) };
}
