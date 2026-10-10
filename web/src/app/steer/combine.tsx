"use client";

import { useState } from "react";
import { FeaturePicker, type PickedFeature } from "@/components/FeaturePicker";
import { useModelData } from "@/components/useModelData";
import { ErrorState, Skeleton } from "@/components/States";
import { api } from "@/lib/api/client";
import type { MultiSteerRequest, MultiSteerResponse } from "@/lib/api/types";
import { featureName } from "@/lib/format";
import type { CombinePair } from "@/lib/model-data";
import { useAction } from "@/lib/useApi";
import { after, LiveTag, strengthLabel } from "./checks";
import s from "./steer.module.css";

/* W5 (POST /steer/multi): two or three features turned up together on one prompt, with each one run
   alone beside it, shown as activation numbers only (their texts can be harsh: server/README.md). The
   readings show which concepts were active in the finished text, nothing more (docs/features/W5.md).
   Seed and strength are always shown. */

const STRENGTHS = [0.25, 0.5, 0.75, 1, 1.5, 2, 3]; // up to the steering cap of 3
type Slot = { f: PickedFeature | null; strength: number };
const empty = (): Slot => ({ f: null, strength: 0.5 });
const act = (v: number) => v.toFixed(2);

// this model's recorded pair (lib/model-data.ts), found by an automated search; none on other models
const isPair = (r: MultiSteerResponse, pair: CombinePair | null) =>
  !!pair &&
  r.prompt === pair.prompt &&
  r.seed === pair.seed &&
  r.steers.length === pair.ids.length &&
  r.steers.every((x) => pair.ids.includes(x.feature_id) && x.strength === pair.strength);

export function CombinePanel(props: {
  prompt: string;
  seed: number;
  maxTokens: number;
  onSetup: (v: { prompt: string; seed: number; maxTokens: number }) => void;
}) {
  const { prompt, seed, maxTokens } = props;
  const PAIR = useModelData()?.combinePair ?? null;
  const [slots, setSlots] = useState<Slot[]>([empty(), empty()]);
  const [labels, setLabels] = useState<Record<number, string | null>>({});
  const combine = useAction((body: MultiSteerRequest, sig) => api.steerMulti(body, sig));
  const res = combine.data;

  const ids = slots.map((x) => x.f?.feature_id);
  const dup = new Set(ids.filter((v) => v !== undefined)).size < ids.filter((v) => v !== undefined).length;
  const ready = slots.every((x) => x.f) && !dup && !!prompt.trim();

  const remember = (fs: PickedFeature[]) => setLabels((m) => ({ ...m, ...Object.fromEntries(fs.map((f) => [f.feature_id, f.label])) }));
  const go = (body: Omit<MultiSteerRequest, "with_single_controls">) => combine.run({ ...body, with_single_controls: true });
  const run = () =>
    ready && go({ prompt, seed, max_new_tokens: maxTokens, steers: slots.map((x) => ({ feature_id: x.f!.feature_id, strength: x.strength })) });

  function set(i: number, v: Partial<Slot>) {
    if (v.f) remember([v.f]);
    setSlots((cur) => cur.map((x, j) => (j === i ? { ...x, ...v } : x)));
  }

  async function loadPair() {
    if (!PAIR) return;
    const fs = await Promise.all(
      PAIR.ids.map((id) =>
        api
          .feature(id)
          .then((d) => ({ feature_id: id, label: d.label }))
          .catch(() => ({ feature_id: id, label: null })),
      ),
    );
    remember(fs);
    setSlots(fs.map((f) => ({ f, strength: PAIR.strength })));
    props.onSetup({ prompt: PAIR.prompt, seed: PAIR.seed, maxTokens: PAIR.maxTokens });
    go({ prompt: PAIR.prompt, seed: PAIR.seed, max_new_tokens: PAIR.maxTokens, steers: PAIR.ids.map((id) => ({ feature_id: id, strength: PAIR.strength })) });
  }

  return (
    <section className="card" aria-labelledby="cb-h">
      <div className="card-head">
        <h2 id="cb-h">Combine features</h2>
        {res && (
          <div className="row" style={{ gap: 8 }}>
            <span className="mono muted small">seed {res.seed}</span>
            <LiveTag simulated={res.simulated} />
          </div>
        )}
      </div>
      <div className="card-pad stack">
        <p className="small muted">Turn two or three features up together, with the prompt and seed on the left. Each one is also run alone.</p>
        {PAIR && (
          <button type="button" className={s.demo} style={{ alignSelf: "flex-start" }} onClick={loadPair} disabled={combine.loading}>
            <span className="mono">{PAIR.ids.map((id) => `#${id}`).join(" + ")}</span> up {strengthLabel(PAIR.strength)} · “
            {PAIR.prompt.length > 22 ? `${PAIR.prompt.slice(0, 22)}…` : PAIR.prompt}”
          </button>
        )}
        <ol className={s.slots}>
          {slots.map((x, i) => (
            <li key={i} className={s.slot}>
              <span className="mono muted small">{i + 1}</span>
              <FeaturePicker value={x.f} onChange={(f) => set(i, { f })} />
              <select
                className="select"
                aria-label={`Strength for feature ${i + 1}`}
                value={x.strength}
                onChange={(e) => set(i, { strength: Number(e.target.value) })}
              >
                {STRENGTHS.map((v) => (
                  <option key={v} value={v}>
                    up {strengthLabel(v)}
                  </option>
                ))}
              </select>
              {i === 2 ? (
                <button type="button" className="icon-btn" aria-label="Remove the third feature" onClick={() => setSlots((cur) => cur.slice(0, 2))}>
                  ×
                </button>
              ) : (
                <span />
              )}
            </li>
          ))}
        </ol>
        <div className="row" style={{ gap: 8 }}>
          <button className="btn" onClick={run} disabled={!ready || combine.loading}>
            {combine.loading ? "Generating…" : "Run together"}
          </button>
          {slots.length < 3 && (
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => setSlots((cur) => [...cur, empty()])}>
              Add a third
            </button>
          )}
        </div>
        {dup && <p className="small muted">Pick different features.</p>}
        {combine.error && <ErrorState error={combine.error} onRetry={run} compact />}
        {combine.loading && !res && (
          <div className="stack" style={{ gap: 10 }}>
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} h={40} />
            ))}
          </div>
        )}
      </div>
      {res && <CombineResult res={res} labels={labels} busy={combine.loading} pair={PAIR} />}
    </section>
  );
}

function CombineResult({ res, labels, busy, pair }: { res: MultiSteerResponse; labels: Record<number, string | null>; busy: boolean; pair: CombinePair | null }) {
  const found = isPair(res, pair);
  const alone = new Map((res.single_feature_runs ?? []).map((r) => [r.feature_id, r]));
  const strengthOf = (id: number) => res.steers.find((x) => x.feature_id === id)?.strength;
  return (
    <div className={busy ? s.dim : ""}>
      <div className={s.combineTop}>
        <span className="mono small">
          {res.steers.map((x) => `#${x.feature_id} up ${strengthLabel(x.strength)}`).join(" + ")} · seed {res.seed}
        </span>
        {found && <span className="pill pill-outline">Found by search, depends on the prompt</span>}
        {res.degenerate && <span className={s.warn}>Breaks down</span>}
        {res.degenerate && res.reason && <span className={s.breaks}>{res.reason}</span>}
      </div>
      {!res.differs && <p className={s.summary}>Both runs wrote the same text.</p>}
      <div className={s.cols}>
        <div className={s.col}>
          <div className={s.colHead}>
            <b>Normal</b>
            <span className="mono muted small">seed {res.seed}</span>
          </div>
          <p className={s.text}>
            <span className={s.promptText}>{res.prompt}</span>
            {after(res.baseline_text, res.prompt)}
          </p>
        </div>
        <div className={`${s.col} ${s.colSteered}`} data-bad={res.degenerate ? "1" : "0"}>
          <div className={s.colHead}>
            <b>Together</b>
            <span className="mono muted small">
              {res.steers.map((x) => strengthLabel(x.strength)).join(" + ")} · seed {res.seed}
            </span>
          </div>
          <p className={s.text}>
            <span className={s.promptText}>{res.prompt}</span>
            {after(res.steered_text, res.prompt)}
          </p>
        </div>
      </div>
      <div className={s.check}>
        <div className={s.checkTop}>
          <span className="eyebrow">On the finished text</span>
          <span className="mono muted small">seed {res.seed}</span>
        </div>
        <ol className={s.acts}>
          <li className={`${s.actRow} ${s.actHead}`} aria-hidden="true">
            <span />
            <span>normal</span>
            <span>alone</span>
            <span>together</span>
            <span />
          </li>
          {res.activations.map((a) => {
            const one = alone.get(a.feature_id);
            const st = strengthOf(a.feature_id);
            return (
              <li key={a.feature_id} className={s.actRow}>
                <span className={s.actName}>
                  <span className="mono">#{a.feature_id}</span>
                  {st !== undefined && <span className="mono muted small"> up {strengthLabel(st)}</span>} {featureName(a.label ?? labels[a.feature_id] ?? null, a.feature_id)}
                </span>
                <span className="mono" title="The normal run">
                  {act(a.target_activation_baseline)}
                </span>
                <span className="mono" title="This feature turned up by itself, same prompt, seed and strength">
                  {one ? act(one.target_activation_steered) : "—"}
                </span>
                <b className="mono" title="All the features turned up together">
                  {act(a.target_activation_steered)}
                </b>
                <span className={a.activation_verdict === "about the same" ? s.verdictSame : s.verdict}>{a.activation_verdict}</span>
              </li>
            );
          })}
        </ol>
      </div>
      <div className={s.foot}>
        <span className="small muted">
          {found && pair?.note}
          Each number is the feature&apos;s highest reading on the new text, re-read without steering: it shows which concepts were active, not whether the
          text is better. Alone: that feature turned up by itself, same prompt, seed and strength. The loop warning catches repetition, not all poor text.
        </span>
      </div>
    </div>
  );
}
