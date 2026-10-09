"use client";

import { useState } from "react";
import type { PickedFeature } from "@/components/FeaturePicker";
import { ErrorState, Skeleton } from "@/components/States";
import { api } from "@/lib/api/client";
import type { ApiError } from "@/lib/api/client";
import type { PromptOnlyRequest, SteerCheckResponse, SteerCompareResponse, SteerSweepRequest } from "@/lib/api/types";
import { t } from "@/lib/copy";
import { useAction } from "@/lib/useApi";
import s from "./steer.module.css";

/* Phase 4 steering checks on the Steer page: X1 the feature's activation on the finished text, X3 where
   the runs split, X2 every strength, X4 prompt only. Each shows its honest limit under the result
   (docs/features/X1.md to X4.md). Seed and strength are always shown next to steered text. */

/** The new text after the prompt. */
export function after(full: string, prompt: string) {
  return full.startsWith(prompt) ? full.slice(prompt.length) : full;
}

export const strengthLabel = (v: number) => `${+v.toFixed(2)}×`;

export function LiveTag({ simulated }: { simulated: boolean }) {
  return <span className={`pill ${simulated ? "pill-sim" : "pill-live"} pill-dot`}>{simulated ? "Simulated" : t("badge.live")}</span>;
}

/** X3: where the two runs split. The shared start is shown once, then each run from the split. */
export function SplitView({ c, prompt }: { c: SteerCompareResponse; prompt: string }) {
  const n = c.first_divergence_index ?? 0;
  return (
    <>
      <p className={s.summary}>
        {!c.diverges ? "Both runs wrote the same text." : n === 0 ? "It splits at the first new token." : `Same for the first ${n} new token${n === 1 ? "" : "s"}, then it splits.`}
      </p>
      <div className={s.shared}>
        <span className="eyebrow">{c.diverges ? "Same start" : "Both runs"}</span>
        <p className={s.text}>
          <span className={s.promptText}>{prompt}</span>
          {c.identical_prefix_text}
          {c.diverges && <span className={s.fork} aria-hidden="true" />}
        </p>
      </div>
      {c.diverges && (
        <div className={s.cols}>
          <div className={s.col}>
            <div className={s.colHead}>
              <b>Normal</b>
              <span className="mono muted small">seed {c.seed} · from token {n}</span>
            </div>
            <p className={s.text}>{c.baseline_from_divergence_text}</p>
          </div>
          <div className={`${s.col} ${s.colSteered}`}>
            <div className={s.colHead}>
              <b>{c.mode === "amplify" ? `Feature up ${strengthLabel(c.strength)}` : "Feature off"}</b>
              <span className="mono muted small">seed {c.seed} · from token {n}</span>
            </div>
            <p className={s.text}>{c.steered_from_divergence_text}</p>
          </div>
        </div>
      )}
      <div className={s.foot}>
        <span className="small muted">
          {c.diverges ? `${c.tokens_changed} of ${c.tokens_compared_from_divergence} tokens differ after the split. ` : ""}
          Same seed ({c.seed}) in both runs, so the split comes from the feature. After it, each run follows its own words, so not every later change is the
          feature&apos;s doing. One feature, one prompt, one seed.
        </span>
      </div>
    </>
  );
}

const act = (v: number) => v.toFixed(2);

/** X1: how strongly the feature fires on each finished text (re-encoded without steering). */
export function ActivationCheck({ c, error }: { c: SteerCheckResponse | null; error: ApiError | null }) {
  if (!c) return error ? <p className={s.checkErr}>Activation check unavailable: {error.message}</p> : null;
  return (
    <div className={s.check}>
      <div className={s.checkTop}>
        <span className="eyebrow">Activation check</span>
        <span className="mono muted small">
          seed {c.seed} · {c.mode === "amplify" ? `up ${strengthLabel(c.strength)}` : "off"}
        </span>
      </div>
      <p className={s.checkLine}>
        <span className="mono">#{c.feature.feature_id}</span> on the finished text: <b className="mono">{act(c.target_activation_baseline)}</b>
        <span aria-hidden="true"> → </span>
        <span className="visually-hidden"> to </span>
        <b className="mono">{act(c.target_activation_steered)}</b> <span className={c.activation_verdict === "about the same" ? s.verdictSame : s.verdict}>{c.activation_verdict}</span>
      </p>
      <p className="small muted">
        Read on the finished text without steering: it says whether the feature shows up, not whether the text is better. A feature already near its peak
        can&apos;t show an increase, and switched off it still fires if the concept&apos;s word is in the text.
      </p>
    </div>
  );
}

/** X2: the feature turned up at several strengths with one seed, each flagged if the text breaks down. */
export function SweepPanel(props: {
  feature: PickedFeature | null;
  prompt: string;
  seed: number;
  maxTokens: number;
  simulated: boolean;
  onUse: (strength: number) => void;
}) {
  const { feature, prompt, seed, maxTokens, simulated } = props;
  const sweep = useAction((body: SteerSweepRequest, sig) => api.steerSweep(body, sig));
  const res = sweep.data;
  const go = () => feature && sweep.run({ feature_id: feature.feature_id, prompt, seed, max_new_tokens: maxTokens });
  return (
    <section className="card" aria-labelledby="sw-h">
      <div className="card-head">
        <h2 id="sw-h">Every strength</h2>
        {res && (
          <div className="row" style={{ gap: 8 }}>
            <span className="mono muted small">seed {res.seed}</span>
            <LiveTag simulated={simulated} />
          </div>
        )}
      </div>
      <div className="card-pad stack">
        <p className="small muted">Turns the feature up from 0 to 2× with the same seed: see where the text starts to change, and where it breaks down.</p>
        <button className="btn" style={{ alignSelf: "flex-start" }} onClick={go} disabled={!feature || !prompt.trim() || sweep.loading}>
          {sweep.loading ? "Running the sweep…" : feature ? `Try every strength on #${feature.feature_id}` : "Pick a feature first"}
        </button>
        {sweep.error && <ErrorState error={sweep.error} onRetry={go} compact />}
        {sweep.loading && !res && (
          <div className="stack" style={{ gap: 10 }}>
            {[0, 1, 2, 3].map((i) => (
              <Skeleton key={i} h={40} />
            ))}
          </div>
        )}
      </div>
      {res && (
        <>
          <ol className={`${s.sweepList} ${sweep.loading ? s.dim : ""}`} aria-label={`Strength sweep for #${res.feature.feature_id}`}>
            <li className={s.sweepRow}>
              <span className={s.sweepK}>Normal</span>
              <div className={s.sweepBody}>
                <p className={s.clamp} title={res.baseline_text}>
                  {after(res.baseline_text, prompt)}
                </p>
                {res.baseline_degenerate && <p className={s.breaks}>Breaks down: {res.baseline_reason}</p>}
              </div>
            </li>
            {res.results.map((r) => (
              <li key={r.strength} className={s.sweepRow} data-bad={r.degenerate ? "1" : "0"}>
                <span className={s.sweepK}>{strengthLabel(r.strength)}</span>
                <div className={s.sweepBody}>
                  <div className={s.sweepTags}>
                    <span className="pill pill-outline">{r.differs_from_baseline ? `splits at token ${r.first_divergence_index}` : "same as normal"}</span>
                    {r.degenerate && <span className={s.warn}>Breaks down</span>}
                    {r.strength > 0 && (
                      <button type="button" className={s.use} onClick={() => props.onUse(r.strength)}>
                        Use {strengthLabel(r.strength)}
                      </button>
                    )}
                  </div>
                  {r.differs_from_baseline && (
                    <p className={s.clamp} title={r.text}>
                      {after(r.text, prompt)}
                    </p>
                  )}
                  {r.degenerate && r.reason && <p className={s.breaks}>{r.reason}</p>}
                </div>
              </li>
            ))}
          </ol>
          <div className={s.foot}>
            <span className="small muted">
              “Breaks down” catches repetition and broken text, not every kind of drift: a result not flagged can still be poor. Results depend on the prompt and
              the strength; this is one feature and one seed.
            </span>
          </div>
        </>
      )}
    </section>
  );
}

/** X4: say the idea in the prompt instead of steering, and put the two side by side. */
export function PromptOnlyPanel(props: {
  prompt: string;
  seed: number;
  maxTokens: number;
  simulated: boolean;
  steered: { prompt: string; text: string; seed: number; what: string } | null;
}) {
  const { prompt, seed, maxTokens, simulated, steered } = props;
  const [alt, setAlt] = useState<string | null>(null);
  const value = alt ?? prompt;
  const only = useAction((body: PromptOnlyRequest, sig) => api.promptOnly(body, sig));
  const res = only.data;
  const go = () => only.run({ prompt: value, seed, max_new_tokens: maxTokens });
  return (
    <section className="card" aria-labelledby="po-h">
      <div className="card-head">
        <h2 id="po-h">Prompt only</h2>
        {res && <LiveTag simulated={simulated} />}
      </div>
      <div className="card-pad stack">
        <p className="small muted">Instead of steering, say the idea in the prompt, then compare. Same seed, no feature touched.</p>
        <textarea className="textarea" rows={2} value={value} onChange={(e) => setAlt(e.target.value)} aria-label="Prompt to try without steering" />
        <div className="row" style={{ gap: 8 }}>
          <button className="btn" onClick={go} disabled={!value.trim() || only.loading}>
            {only.loading ? "Writing…" : "Write from the prompt only"}
          </button>
          {alt !== null && alt !== prompt && (
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => setAlt(null)}>
              Back to your prompt
            </button>
          )}
        </div>
        {only.error && <ErrorState error={only.error} onRetry={go} compact />}
      </div>
      {res && (
        <>
          <div className={`${s.cols} ${only.loading ? s.dim : ""}`}>
            <div className={s.col}>
              <div className={s.colHead}>
                <b>Prompt only</b>
                <span className="mono muted small">seed {res.seed}</span>
              </div>
              <p className={s.text}>
                <span className={s.promptText}>{res.prompt}</span>
                {res.continuation}
              </p>
            </div>
            <div className={`${s.col} ${s.colSteered}`}>
              <div className={s.colHead}>
                <b>{steered ? steered.what : "Steering"}</b>
                {steered && <span className="mono muted small">seed {steered.seed}</span>}
              </div>
              {steered ? (
                <p className={s.text}>
                  <span className={s.promptText}>{steered.prompt}</span>
                  {after(steered.text, steered.prompt)}
                </p>
              ) : (
                <p className="small muted">Run the comparison to see the steered text here.</p>
              )}
            </div>
          </div>
          <div className={s.foot}>
            <span className="small muted">
              One seeded sample, not what the model always writes for this prompt. Prompting changes the words going in; steering changes one feature inside.
              With your original prompt and seed it matches Normal.
            </span>
          </div>
        </>
      )}
    </section>
  );
}
