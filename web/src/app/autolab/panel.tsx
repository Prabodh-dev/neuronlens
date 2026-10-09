"use client";

import Link from "next/link";
import { useEffect, useRef, useState, type CSSProperties } from "react";
import { ErrorState } from "@/components/States";
import { ApiError } from "@/lib/api/client";
import {
  AUTOLAB_CANDIDATES,
  AUTOLAB_LEN,
  AUTOLAB_SEED,
  AUTOLAB_STRENGTHS,
  runAutoLab,
  WORKED,
  type Candidate,
  type Cell,
  type Report,
  type Row,
  type Step,
  type Tag,
  type Verdict,
} from "@/lib/autolab";
import { isBlockedGoal } from "@/lib/autolab-blocklist";
import { FILLERS } from "@/lib/autolab-fillers";
import { AUTOLAB_GOALS, AUTOLAB_PROMPTS, type AutoLabGoal } from "@/lib/autolab-presets";
import { featureName } from "@/lib/format";
import { steerLink } from "@/lib/share";
import { after, LiveTag, strengthLabel } from "../steer/checks";
import s from "./autolab.module.css";

/* Auto-Lab: pick a prompt and a goal (a button, or 1 to 3 typed words), press Run. The run and its tags
   are in lib/autolab.ts (rule v2, as server/scripts/autolab_check.py); this is the screen
   (docs/features/AutoLab.md), on its own page (/autolab). The grid fills
   in live, with the unsteered control first; once done, every cell can be clicked to see its readings
   and the rule behind its tag. A mock server's results are labelled Simulated, and
   NEXT_PUBLIC_MODEL_ONLY=1 blocks them on the demo machine. Candidates are "features that fired on
   these words", never "the feature for" a goal. */

type State =
  | { kind: "idle" }
  | { kind: "running"; step: Step; t0: number }
  | { kind: "done"; report: Report; secs: number }
  | { kind: "cancelled" }
  | { kind: "nomatch" }
  | { kind: "error"; error: ApiError };
type Pick = { feature_id: number; strength: number };
type Line = { id: number; text: string; tag: Tag | "info" };

const MARK: Record<Tag, { sym: string; word: string }> = {
  worked: { sym: "✓", word: "worked" },
  none: { sym: "✗", word: "no effect" },
  breaks: { sym: "⚠", word: "breaks down" },
};
const act = (v: number) => v.toFixed(2);

/** A typed goal: 1 to 3 words, trimmed. */
function typedGoal(text: string): { goal: AutoLabGoal | null; problem: string | null } {
  const words = text.trim().split(/\s+/).filter(Boolean);
  if (!words.length) return { goal: null, problem: null };
  if (words.length > 3) return { goal: null, problem: "Use 1 to 3 words." };
  if (isBlockedGoal(text)) return { goal: null, problem: "Try another word." };
  const w = words.join(" ");
  return { goal: { label: `“${w}”`, words: w }, problem: null };
}

/** Counts up to `to` as soon as it appears, and always ends on the exact value (even with animation
 *  frames paused or reduced motion on). */
function CountTo({ to }: { to: number }) {
  const [v, setV] = useState(0);
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setV(to);
      return;
    }
    const t0 = performance.now();
    let raf = 0;
    const tick = (t: number) => {
      const u = Math.min(1, (t - t0) / 800);
      setV(to * (1 - Math.pow(1 - u, 3)));
      if (u < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    const done = setTimeout(() => setV(to), 900);
    return () => {
      cancelAnimationFrame(raf);
      clearTimeout(done);
    };
  }, [to]);
  return <>{act(v)}</>;
}

function tally(rows: Row[]) {
  const t = { worked: 0, none: 0, breaks: 0 };
  for (const r of rows) for (const c of r.cells) t[c.tag]++;
  return t;
}

export function AutoLabPanel({ simulatedServer, run }: { simulatedServer: boolean; run: string | null }) {
  const [p, setP] = useState(0);
  const [g, setG] = useState(0); // a goal button, or -1 for the typed goal
  const [typed, setTyped] = useState("");
  const [state, setState] = useState<State>({ kind: "idle" });
  const [log, setLog] = useState<Line[]>([]);
  const [pick, setPick] = useState<Pick | null>(null);
  const ctl = useRef<AbortController | null>(null);
  const running = state.kind === "running";
  const own = typedGoal(typed);
  const goal = g >= 0 ? AUTOLAB_GOALS[g] : own.goal;

  async function go() {
    if (!goal || running) return;
    const c = new AbortController();
    ctl.current = c;
    const t0 = performance.now();
    let n = 0;
    const say = (text: string, tag: Line["tag"]) => setLog((l) => [{ id: ++n, text, tag }, ...l].slice(0, 6));
    const told = new Set<number>();
    let found = false;
    setLog([]);
    setPick(null);
    say(`Searching for features that fired on ${goal.label}…`, "info");
    setState({ kind: "running", step: { stage: "search" }, t0 });
    try {
      const r = await runAutoLab(goal, AUTOLAB_PROMPTS[p], c.signal, (step) => {
        setState({ kind: "running", step, t0 });
        if (step.stage !== "feature") return;
        if (!found) {
          found = true;
          say(`${step.n} features fired on these words. Trying each at ${AUTOLAB_STRENGTHS.length} strengths.`, "info");
        }
        for (const row of step.rows) {
          if (row.cells.length < AUTOLAB_STRENGTHS.length || told.has(row.feature_id)) continue;
          told.add(row.feature_id);
          const ok = row.cells.filter((x) => x.tag === "worked").map((x) => strengthLabel(x.strength));
          const br = row.breaksAt !== null ? ` · breaks down from ${strengthLabel(row.breaksAt)}` : "";
          say(`#${row.feature_id}: ${ok.length ? `✓ at ${ok.join(", ")}` : "no ✓"}${br}`, ok.length ? "worked" : row.breaksAt !== null ? "breaks" : "none");
        }
      });
      if (c.signal.aborted) return;
      setState(r === "nomatch" ? { kind: "nomatch" } : { kind: "done", report: r, secs: (performance.now() - t0) / 1000 });
    } catch (e) {
      if (c.signal.aborted) setState({ kind: "cancelled" });
      else setState({ kind: "error", error: e instanceof ApiError ? e : new ApiError("INTERNAL", String((e as Error)?.message ?? e), 0) });
    }
  }

  const report = state.kind === "done" ? state.report : null;
  const simulated = report ? report.simulated : simulatedServer;
  const liveRows = state.kind === "running" && state.step.stage === "feature" ? state.step.rows : report?.rows ?? [];

  return (
    <section className={`card ${s.lab}`} aria-labelledby="al-h">
      <div className={s.head}>
        <div>
          <h1 id="al-h" className={s.titleBig}>
            Auto-<span className={s.grad}>Lab</span>
          </h1>
          <p className={s.sub}>
            Pick a prompt and a goal. Auto-Lab tests up to {AUTOLAB_CANDIDATES} features that fired most on the goal&apos;s words, each at {AUTOLAB_STRENGTHS.length} strengths,
            and tags every test.
          </p>
        </div>
        {(running || report) && <LiveTag simulated={simulated} />}
      </div>

      <div className={s.setup}>
        <div className={s.group} role="group" aria-label="Prompt">
          <StepLabel n={1} text="Pick a prompt" />
          <div className={s.row}>
            {AUTOLAB_PROMPTS.map((t, i) => (
              <button key={t} type="button" className={s.choice} aria-pressed={p === i} onClick={() => setP(i)} disabled={running}>
                “{t}”
              </button>
            ))}
          </div>
        </div>

        <div className={s.group} role="group" aria-label="Goal">
          <StepLabel n={2} text="Pick a goal" />
          <div className={s.row}>
            {AUTOLAB_GOALS.map((x, i) => (
              <button
                key={x.label}
                type="button"
                className={`${s.choice} ${s.goal}`}
                aria-pressed={g === i}
                onClick={() => {
                  setG(i);
                  setTyped("");
                }}
                disabled={running}
              >
                {x.icon && (
                  <span className={s.icon} aria-hidden="true">
                    {x.icon}
                  </span>
                )}
                {x.label}
              </button>
            ))}
            <input
              className={s.own}
              value={typed}
              maxLength={60}
              placeholder="✎ or type 1 to 3 words"
              aria-label="Your own goal, 1 to 3 words"
              aria-invalid={g < 0 && !!own.problem}
              data-on={g < 0 && !!own.goal ? "1" : undefined}
              disabled={running}
              onChange={(e) => {
                // typing selects the typed goal; an empty box leaves Run disabled until a button is picked
                setTyped(e.target.value);
                setG(-1);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") go();
              }}
            />
          </div>
          {g < 0 && own.problem && <p className={s.problem}>{own.problem}</p>}
        </div>

        <div className={s.group}>
          <StepLabel n={3} text="Run it" />
          {running ? (
            <RunStatus step={state.step} t0={state.t0} rows={liveRows} log={log} onCancel={() => ctl.current?.abort()} />
          ) : (
            <div className={s.runRow}>
              <button type="button" className={s.run} onClick={go} disabled={!goal}>
                <span className={s.runGlyph} aria-hidden="true">
                  ▶
                </span>
                Run Auto-Lab
              </button>
              <span className={s.runSub}>
                {goal ? (
                  <>
                    {goal.icon && `${goal.icon} `}
                    {goal.label} on “{AUTOLAB_PROMPTS[p]}” · seed {AUTOLAB_SEED} · up to {AUTOLAB_CANDIDATES * AUTOLAB_STRENGTHS.length} tests · about 15 seconds
                  </>
                ) : (
                  "Pick a goal or type one"
                )}
              </span>
            </div>
          )}
        </div>
      </div>

      {state.kind === "nomatch" && (
        <div className={s.note} role="status">
          <b>No feature matches these words. Try another word.</b>
          <span>Fewer than 3 features were left once the ones that fire on almost any words were left out, so nothing else was run.</span>
        </div>
      )}
      {state.kind === "cancelled" && (
        <div className={s.note} role="status">
          <b>Cancelled.</b>
          <span>Nothing was scored.</span>
        </div>
      )}
      {state.kind === "error" && (
        <div className={s.note}>
          <ErrorState error={state.error} onRetry={go} compact />
        </div>
      )}

      {report && state.kind === "done" && <Result r={report} secs={state.secs} run={run} pick={pick} onPick={setPick} />}
      {(state.kind === "running" || report) && (
        // live: each row pops in as its sweep lands; result: the whole table plays one slow wave
        <Board
          key={report ? "result" : "live"}
          wave={!!report}
          candidates={state.kind === "running" ? (state.step.stage === "feature" ? state.step.candidates : null) : report!.rows}
          rows={liveRows}
          active={state.kind === "running" && state.step.stage === "feature" ? state.step.i - 1 : -1}
          pickCell={report ? verdictCell(report.verdict) : null}
          pick={report ? pick : null}
          onPick={report ? setPick : undefined}
        />
      )}

      <div className={s.foot}>
        <b>A test on this prompt, not a fix.</b>
        <span>
          One seed ({AUTOLAB_SEED}), {AUTOLAB_LEN} tokens, and up to {AUTOLAB_CANDIDATES} features that fired most on the goal&apos;s words, leaving out {FILLERS.size} that fire on almost any words.
          {simulated && " Mock mode: these results are simulated, not from the model."}
        </span>
      </div>
    </section>
  );
}

function StepLabel({ n, text }: { n: number; text: string }) {
  return (
    <span className={s.step}>
      <span className={s.num} aria-hidden="true">
        {n}
      </span>
      {text}
    </span>
  );
}

function Tally({ rows, big = false }: { rows: Row[]; big?: boolean }) {
  const t = tally(rows);
  return (
    <div className={`${s.tally} ${big ? s.tallyBig : ""}`} aria-label={`${t.worked} worked, ${t.none} no effect, ${t.breaks} breaks down`}>
      {(["worked", "none", "breaks"] as Tag[]).map((k) => (
        <span key={k} className={s.count} data-tag={k}>
          <span aria-hidden="true">{MARK[k].sym}</span>
          <b key={t[k]} className={s.bump}>
            {t[k]}
          </b>
          <span className={s.countWord}>{MARK[k].word}</span>
        </span>
      ))}
    </div>
  );
}

function RunStatus({ step, t0, rows, log, onCancel }: { step: Step; t0: number; rows: Row[]; log: Line[]; onCancel: () => void }) {
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    const id = setInterval(() => setNow(performance.now()), 200);
    return () => clearInterval(id);
  }, []);
  const done = rows.filter((r) => r.cells.length >= AUTOLAB_STRENGTHS.length).length;
  const n = step.stage === "feature" ? step.n : AUTOLAB_CANDIDATES;
  const frac = step.stage === "search" ? 0.04 : 0.08 + (0.92 * done) / Math.max(1, n);
  const label = step.stage === "search" ? "Finding features that fired on these words…" : `Trying feature ${step.i} of ${step.n}…`;
  return (
    <div className={s.status} role="status" aria-live="polite">
      <div className={s.statusTop}>
        <span className={s.spinner} aria-hidden="true" />
        <b>
          {label} <span className={s.eta}>about 15 seconds in all</span>
        </b>
        <span className={s.clock}>{Math.floor((now - t0) / 1000)} s</span>
        <button type="button" className="btn btn-ghost btn-sm" onClick={onCancel}>
          Cancel
        </button>
      </div>
      <span className={s.bar} aria-hidden="true">
        <i style={{ width: `${Math.round(frac * 100)}%` }} />
      </span>
      <div className={s.statusBottom}>
        <Tally rows={rows} />
        <ol className={s.log} aria-label="What Auto-Lab is doing">
          {log.map((l) => (
            <li key={l.id} data-tag={l.tag}>
              {l.text}
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}

/** A candidate's number, name and the words around the peak of its top example. */
function CandidateName({ c }: { c: Candidate }) {
  return (
    <>
      <span className={s.featName}>
        <span className={s.fid}>#{c.feature_id}</span> {featureName(c.label, c.feature_id)}
      </span>
      {c.example && (
        <span className={s.example} title="Top example: the words around its strongest token">
          {"…"}
          {c.example.before}
          <b>{c.example.peak}</b>
          {c.example.after}
          {"…"}
        </span>
      )}
    </>
  );
}

/** The grid: features down the side, the unsteered control, then the strengths. It fills in live while
 *  the run goes; once done, every cell is a button that opens it in the inspector. Worked cells next to
 *  each other are joined: that is what makes a ✓ stable. */
/** The slow cascade: each row shows its unsteered reading, then its strengths one by one, left to right.
 *  In the result (`wave`) each row also starts ROW_MS after the one above, so the table fills top-down. */
const FIRST_MS = 160;
const STEP_MS = 110;
const ROW_MS = 150;

/** False until the element is on screen, so an animation never plays where nobody can see it. */
function useSeen<T extends Element>() {
  const ref = useRef<T | null>(null);
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") {
      setSeen(true);
      return;
    }
    const io = new IntersectionObserver((es) => es.some((e) => e.isIntersecting) && (setSeen(true), io.disconnect()), { threshold: 0.12 });
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return [ref, seen] as const;
}

function Board(props: {
  candidates: Candidate[] | null;
  rows: Row[];
  active: number;
  wave?: boolean;
  pickCell?: { row: Row; cell: Cell } | null;
  pick?: Pick | null;
  onPick?: (p: Pick) => void;
}) {
  const { candidates, rows, active, wave = false, pickCell, pick, onPick } = props;
  const [ref, seen] = useSeen<HTMLDivElement>();
  const at = (k: number, j: number) => ({ "--d": `${(wave ? k * ROW_MS : 0) + (j < 0 ? 0 : FIRST_MS + j * STEP_MS)}ms`, "--rd": `${wave ? k * ROW_MS : 0}ms` }) as CSSProperties;
  const strengths = rows.find((r) => r.cells.length)?.cells.map((c) => c.strength) ?? AUTOLAB_STRENGTHS;
  const list: (Candidate | null)[] = candidates ?? Array.from({ length: AUTOLAB_CANDIDATES }, () => null);
  return (
    <div ref={ref} className={s.board} data-hold={seen ? undefined : "1"}>
      <div className={s.boardHead}>
        <h3 id="al-grid" className={s.caption}>
          Features that fired on these words, at each strength
        </h3>
        <span className={s.hint}>{onPick ? "click any cell" : "fills in as each feature is tried"}</span>
      </div>
      <div className={s.gridWrap}>
        <table className={s.grid} aria-labelledby="al-grid">
          <colgroup>
            <col />
            <col className={s.colCtrl} />
            {strengths.map((st) => (
              <col key={st} className={s.colStr} />
            ))}
            <col className={s.colBreaks} />
          </colgroup>
          <thead>
            <tr className={s.groupRow} aria-hidden="true">
              <td />
              <td />
              <td colSpan={strengths.length}>
                <span className={s.groupHead}>steered, at each strength</span>
              </td>
              <td />
            </tr>
            <tr>
              <th scope="col">Feature and top example</th>
              <th scope="col" className={s.ctrlHead} title="Strength 0: the unsteered text, the control. Its reading is what each strength is compared with.">
                unsteered
              </th>
              {strengths.map((st) => (
                <th key={st} scope="col" className={s.mono}>
                  {strengthLabel(st)}
                </th>
              ))}
              <th scope="col">Breaks down from</th>
            </tr>
          </thead>
          <tbody>
            {list.map((c, k) => {
              const row = c ? rows.find((r) => r.feature_id === c.feature_id) : undefined;
              const isActive = k === active && (!row || row.cells.length < strengths.length);
              return (
                <tr key={c?.feature_id ?? `wait-${k}`} data-active={isActive ? "1" : undefined} style={at(k, -1)}>
                  <th scope="row" className={s.feat}>
                    {c ? (
                      <Link href={`/features/${c.feature_id}`}>
                        <CandidateName c={c} />
                      </Link>
                    ) : (
                      <span className={s.ghost} aria-hidden="true" />
                    )}
                  </th>
                  <td className={s.ctrl} title="The feature's reading on the unsteered text" style={at(k, -1)}>
                    {row && row.unsteered !== null ? <span className={s.ctrlVal}>{act(row.unsteered)}</span> : <span className={s.dot} aria-hidden="true" />}
                  </td>
                  {strengths.map((st, j) => {
                    const cell = row?.cells.find((x) => x.strength === st);
                    if (!cell || !row)
                      return (
                        <td key={`${st}-wait`} className={s.cell} data-state={isActive ? "active" : "wait"} style={{ animationDelay: `${j * 120}ms` }}>
                          <span className={s.dot} aria-hidden="true" />
                          <span className="visually-hidden">not tried yet</span>
                        </td>
                      );
                    const prev = row.cells[j - 1];
                    const next = row.cells[j + 1];
                    const linkL = cell.tag === "worked" && prev?.tag === "worked";
                    const linkR = cell.tag === "worked" && next?.tag === "worked";
                    const isBest = pickCell?.row.feature_id === row.feature_id && pickCell.cell.strength === st;
                    const isPick = pick ? pick.feature_id === row.feature_id && pick.strength === st : isBest;
                    const label = `#${row.feature_id} at ${strengthLabel(st)}: ${MARK[cell.tag].word}${cell.stable ? ", stable" : ""}${isBest ? ", recommended" : ""}`;
                    return (
                      <td
                        key={`${st}-${cell.tag}`}
                        className={s.cell}
                        data-tag={cell.tag}
                        data-best={isBest ? "1" : undefined}
                        data-link-l={linkL ? "1" : undefined}
                        data-link-r={linkR ? "1" : undefined}
                        style={at(k, j)}
                      >
                        {/* the same button while it runs (not clickable yet) and after, so it is never rebuilt */}
                        <button
                          type="button"
                          className={s.cellBtn}
                          disabled={!onPick}
                          aria-pressed={onPick ? isPick : undefined}
                          aria-label={label}
                          onClick={() => onPick?.({ feature_id: row.feature_id, strength: st })}
                        >
                          {MARK[cell.tag].sym}
                        </button>
                      </td>
                    );
                  })}
                  <td className={s.breaks} style={at(k, strengths.length)}>
                    {row && row.cells.length >= strengths.length ? (
                      row.breaksAt !== null ? (
                        <span className={s.breaksAt}>⚠ {strengthLabel(row.breaksAt)}</span>
                      ) : (
                        <span className={s.dim}>not up to {strengthLabel(strengths[strengths.length - 1])}</span>
                      )
                    ) : null}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className={s.legend}>
        <span data-tag="worked">
          <b>✓ worked</b> its reading on the finished text rose by {WORKED.minRise.toFixed(1)} or more, to at least {WORKED.minRatio}× unsteered, no loop
        </span>
        <span data-tag="stable">
          <b>✓—✓ stable</b> the strength next to it worked too
        </span>
        <span data-tag="none">
          <b>✗ no effect</b> no clear rise, no loop
        </span>
        <span data-tag="breaks">
          <b>⚠ breaks down</b> the text loops
        </span>
      </p>
    </div>
  );
}

const VERDICT_TEXT: Record<Exclude<Verdict["status"], "recommended">, { title: string; sub: (n: number) => string }> = {
  unreliable: {
    title: "Worked at one strength only, not reliable",
    sub: () => "A ✓ counts for a recommendation only when the strength next to it worked too. None did here. Click any cell to see what it did.",
  },
  nothing: {
    title: "Nothing worked for this goal on this prompt",
    sub: (n) => `None of the ${n} features rose clearly at any strength without looping. Click any cell to see what it did.`,
  },
};

const verdictCell = (v: Verdict) => (v.status === "recommended" ? { row: v.row, cell: v.cell } : null);

function Result({ r, secs, run, pick, onPick }: { r: Report; secs: number; run: string | null; pick: Pick | null; onPick: (p: Pick) => void }) {
  const v = r.verdict;
  const rec = verdictCell(v);
  const sel = pick
    ? (() => {
        const row = r.rows.find((x) => x.feature_id === pick.feature_id);
        const cell = row?.cells.find((x) => x.strength === pick.strength);
        return row && cell ? { row, cell } : null;
      })()
    : rec;
  const t = tally(r.rows);
  return (
    <>
      <div className={s.summary}>
        <Tally rows={r.rows} big />
        <span className={s.summaryText}>
          {r.goal.icon && `${r.goal.icon} `}
          {r.goal.label} on “{r.prompt}” · {t.worked + t.none + t.breaks} tests · {Math.round(secs)} s
        </span>
      </div>
      {v.status !== "recommended" && (
        <div className={`${s.inspect} ${s.empty}`} role="status">
          <div className={s.inspectMain}>
            <span className={s.eyebrow}>No recommendation</span>
            <span className={s.inspectName}>{VERDICT_TEXT[v.status].title}</span>
            <span className={s.dim}>{VERDICT_TEXT[v.status].sub(r.rows.length)}</span>
          </div>
        </div>
      )}
      {sel && (
        <Inspector
          key={`${sel.row.feature_id}-${sel.cell.strength}`}
          r={r}
          row={sel.row}
          cell={sel.cell}
          isBest={!!rec && rec.row === sel.row && rec.cell === sel.cell}
          workedCount={v.status === "recommended" ? v.worked : sel.row.cells.filter((c) => c.tag === "worked").length}
          run={run}
          onPick={onPick}
        />
      )}
    </>
  );
}

/** One cell up close: its readings, the rule behind its tag, its text on demand, and links. */
function Inspector(props: { r: Report; row: Row; cell: Cell; isBest: boolean; workedCount: number; run: string | null; onPick: (p: Pick) => void }) {
  const { r, row, cell, isBest, workedCount, run, onPick } = props;
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const path = steerLink("http://x", { feature_id: row.feature_id, mode: "amplify", strength: cell.strength, seed: r.seed, prompt: r.prompt, max_new_tokens: AUTOLAB_LEN }, run).replace("http://x", "");
  const has = cell.base !== null && cell.steered !== null;
  const rise = has ? cell.steered! - cell.base! : 0;
  const rules = [
    { ok: has && rise >= WORKED.minRise, text: `rise ≥ ${WORKED.minRise.toFixed(1)}`, val: has ? `${rise >= 0 ? "+" : ""}${act(rise)}` : "same text" },
    { ok: has && cell.steered! >= WORKED.minRatio * cell.base!, text: `≥ ${WORKED.minRatio}× unsteered`, val: has ? (cell.base! > 0 ? `${(cell.steered! / cell.base!).toFixed(1)}×` : "from 0") : "—" },
    { ok: cell.tag !== "breaks", text: "no loop", val: cell.tag === "breaks" ? "loops" : "ok" },
  ];
  const stableRule = { ok: cell.stable, text: "stable", val: cell.stable ? "next strength ✓ too" : cell.tag === "worked" ? "no ✓ next to it" : "—" };

  async function copyLink() {
    try {
      await navigator.clipboard.writeText(window.location.origin + path);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked */
    }
  }

  return (
    <div className={s.inspect} data-tag={cell.tag} aria-live="polite">
      <div className={s.inspectMain}>
        <span className={s.eyebrow}>{isBest ? "Recommended · tuned for this goal on this prompt" : "Selected test"}</span>
        <Link href={`/features/${row.feature_id}`} className={s.inspectName}>
          <span className={s.fid}>#{row.feature_id}</span> {featureName(row.label, row.feature_id)}
        </Link>
        <div className={s.badges}>
          <span className={s.chip}>up {strengthLabel(cell.strength)}</span>
          <span className={s.tagBadge} data-tag={cell.tag}>
            {MARK[cell.tag].sym} {MARK[cell.tag].word}
          </span>
        </div>
        {has ? (
          <div className={s.big}>
            <span>
              <CountTo to={cell.base!} />
            </span>
            <span className={s.arrow} aria-hidden="true">
              →
            </span>
            <span className="visually-hidden"> to </span>
            <b>
              <CountTo to={cell.steered!} />
            </b>
          </div>
        ) : (
          <p className={s.dim}>{cell.tag === "breaks" ? "The text loops at this strength." : "Same text as unsteered at this strength, so no rise."}</p>
        )}
        <span className={s.dim}>
          #{row.feature_id}&apos;s reading on the finished text, unsteered → steered · seed {r.seed} · up {strengthLabel(cell.strength)}
        </span>
        <ul className={s.rules} aria-label="The rule for worked, and stability">
          {[...rules, stableRule].map((x, k) => (
            <li key={x.text} data-ok={x.ok ? "1" : "0"} style={{ animationDelay: `${120 + k * 60}ms` }}>
              <span aria-hidden="true">{x.ok ? "✓" : "✗"}</span> {x.text} <span className={s.ruleVal}>{x.val}</span>
            </li>
          ))}
        </ul>
        {isBest && (
          <span className={s.why}>
            Picked: {workedCount} of {row.cells.length} strengths worked, the most of any feature with a stable ✓ (a tie goes to the higher rise), shown at its lowest stable strength.
          </span>
        )}
        {cell.tag === "breaks" && cell.reason && <p className={s.reason}>Loop: {cell.reason}</p>}
        <div className={s.actions}>
          <button type="button" className={s.action} aria-expanded={open} onClick={() => setOpen((v) => !v)}>
            {open ? "Hide text" : "Show text"}
          </button>
          <a className={s.action} href={path} title="Rerun this test on the Steer page's Compare tab, side by side with the unsteered text">
            Open in Compare
          </a>
          <button type="button" className={s.action} onClick={copyLink}>
            {copied ? "Link copied" : "Copy link"}
          </button>
        </div>
      </div>
      <Readings row={row} pick={cell.strength} onPick={(st) => onPick({ feature_id: row.feature_id, strength: st })} />
      {open && (
        <p className={s.text}>
          <span className={s.promptText}>{r.prompt}</span>
          {after(cell.text, r.prompt)}
        </p>
      )}
    </div>
  );
}

/** A feature's reading at each strength, as bars by tag, after the unsteered control; click a bar to
 *  inspect that strength. */
function Readings({ row, pick, onPick }: { row: Row; pick: number; onPick: (strength: number) => void }) {
  const max = Math.max(1, row.unsteered ?? 0, ...row.cells.map((c) => c.steered ?? 0));
  const h = (v: number | null) => `${Math.max(3, ((v ?? 0) / max) * 100)}%`;
  return (
    <div className={s.chart}>
      <span className={s.eyebrow}>#{row.feature_id}&apos;s reading at each strength</span>
      <div className={s.bars} style={{ gridTemplateColumns: `repeat(${row.cells.length + 1}, minmax(0, 1fr))` }}>
        <div className={`${s.barCol} ${s.barCtrl}`} title="Strength 0: the unsteered text, the control">
          <span className={s.barVal}>{row.unsteered === null ? "—" : act(row.unsteered)}</span>
          <span className={s.barTrack}>
            <i style={{ height: h(row.unsteered) }} />
          </span>
          <span className={s.barKey}>unst.</span>
        </div>
        {row.cells.map((c, k) => (
          <button
            key={c.strength}
            type="button"
            className={s.barCol}
            data-tag={c.tag}
            aria-pressed={c.strength === pick}
            aria-label={`${strengthLabel(c.strength)}: ${MARK[c.tag].word}${c.steered !== null ? `, reading ${act(c.steered)}` : ""}`}
            onClick={() => onPick(c.strength)}
          >
            <span className={s.barVal}>{c.steered === null ? MARK[c.tag].sym : act(c.steered)}</span>
            <span className={s.barTrack}>
              <i style={{ height: h(c.steered), animationDelay: `${80 + k * 55}ms` }} />
            </span>
            <span className={s.barKey}>{strengthLabel(c.strength)}</span>
          </button>
        ))}
      </div>
      <span className={s.thresh}>✓ needs a rise of {WORKED.minRise.toFixed(1)} over the unsteered reading, and 2× it</span>
    </div>
  );
}
