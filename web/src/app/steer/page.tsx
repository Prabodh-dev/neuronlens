"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useState } from "react";
import { FeaturePicker, type PickedFeature } from "@/components/FeaturePicker";
import { isMockModel, useHealth } from "@/components/HealthProvider";
import { RunNote } from "@/components/RunNote";
import { EmptyState, ErrorState, Skeleton } from "@/components/States";
import { api, serverHas, type ApiError } from "@/lib/api/client";
import type { SteerCheckResponse, SteerCompareResponse, SteerMode, SteerRequest, SteeringExample } from "@/lib/api/types";
import { diffWords } from "@/lib/diff";
import { featureName, fmtNum } from "@/lib/format";
import { readSteerLink, steerLink } from "@/lib/share";
import { t } from "@/lib/copy";
import { useAction, useApi } from "@/lib/useApi";
import { ActivationCheck, PromptOnlyPanel, SplitView, SweepPanel } from "./checks";
import { CombinePanel } from "./combine";
import s from "./steer.module.css";

type Shown = {
  source: "live" | "saved";
  curated?: boolean;
  feature: PickedFeature;
  mode: SteerMode;
  strength: number;
  seed: number;
  prompt: string;
  len?: number;
  baseline: string;
  steered: string;
  cmp?: SteerCompareResponse;
  chk?: SteerCheckResponse | null;
};
type Tab = "compare" | "sweep" | "prompt" | "combine";

// the demo runs checked on the GPU (demo/phase4_examples.json, seed 42); a click reruns them live.
// #1863 stays a recorded example only: it depends on the prompt and touches a hot-button topic.
const DEMOS = [
  { id: 3446, prompt: "The band walked on stage and", mode: "amplify" as SteerMode, strength: 0.5 },
  { id: 13, prompt: "The acrobat hung", mode: "amplify" as SteerMode, strength: 0.5 },
];

export default function SteerPage() {
  return (
    <Suspense fallback={<div className="page" />}>
      <Steer />
    </Suspense>
  );
}

function Steer() {
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const { data: h } = useHealth();
  const fromUrl = params.get("feature");
  const urlId = fromUrl !== null && /^\d+$/.test(fromUrl) ? Number(fromUrl) : null;

  const [feature, setFeature] = useState<PickedFeature | null>(null);
  const [mode, setMode] = useState<SteerMode>("amplify");
  const [strength, setStrength] = useState(0.5);
  const [prompt, setPrompt] = useState("The weather here is");
  const [seed, setSeed] = useState(42);
  const [maxTokens, setMaxTokens] = useState(60);
  const [shown, setShown] = useState<Shown | null>(null);
  const [tab, setTab] = useState<Tab>("compare");
  // W5: the Combine tab shows only on a server that has the endpoint
  const hasCombine = useApi((sig) => serverHas("/steer/multi", sig), "has:steer-multi").data === true;

  // Load the feature named in the URL so the picker shows its name.
  const urlFeature = useApi((sig) => api.feature(urlId as number, sig), urlId !== null && feature?.feature_id !== urlId ? `sf:${urlId}` : null);
  useEffect(() => {
    if (urlFeature.data) setFeature({ feature_id: urlFeature.data.feature_id, label: urlFeature.data.label });
  }, [urlFeature.data]);

  function choose(f: PickedFeature) {
    setFeature(f);
    setShown(null);
    router.replace(`${pathname}?feature=${f.feature_id}`, { scroll: false });
  }

  const saved = useApi((sig) => api.steeringExamples(feature?.feature_id ?? null, sig), `se:${feature?.feature_id ?? "all"}`);
  // X3: the run compares the two outputs token by token and finds where they split
  const steer = useAction((body: SteerRequest, sig) => api.steerCompare(body, sig));
  // X1: the same run, plus the feature's activation on each finished text
  const check = useAction((body: SteerRequest, sig) => api.steerCheck(body, sig));

  async function compareWith(body: SteerRequest) {
    setTab("compare");
    const [res, chk] = await Promise.all([steer.run(body), check.run(body)]);
    if (res)
      setShown({
        source: "live",
        feature: { feature_id: res.feature.feature_id, label: res.feature.label },
        mode: res.mode,
        strength: res.strength,
        seed: res.seed,
        prompt: body.prompt,
        len: body.max_new_tokens,
        baseline: res.baseline_text,
        steered: res.steered_text,
        cmp: res,
        chk,
      });
  }

  // S5: a shared link (/steer?feature=…&prompt=…) reruns that exact result on arrival. No run-once
  // guard: in development React mounts twice and cancels the first run, so the second must go ahead
  useEffect(() => {
    const l = readSteerLink(params);
    if (!l) return;
    setMode(l.mode);
    if (l.mode === "amplify") setStrength(l.strength);
    setPrompt(l.prompt);
    setSeed(l.seed);
    setMaxTokens(l.max_new_tokens);
    compareWith({ feature_id: l.feature_id, mode: l.mode, strength: l.strength, prompt: l.prompt, seed: l.seed, max_new_tokens: l.max_new_tokens });
    // once, on arrival
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function run() {
    if (!feature || !prompt.trim()) return;
    // a new run: drop the settings of a link this page was opened with
    if (params.get("prompt") !== null) router.replace(`${pathname}?feature=${feature.feature_id}`, { scroll: false });
    await compareWith({ feature_id: feature.feature_id, mode, strength: mode === "amplify" ? strength : 0, prompt, seed, max_new_tokens: maxTokens });
  }

  async function loadDemo(d: (typeof DEMOS)[number]) {
    let label: string | null = null;
    try {
      label = (await api.feature(d.id)).label;
    } catch {
      /* the run below reports any problem */
    }
    setFeature({ feature_id: d.id, label });
    router.replace(`${pathname}?feature=${d.id}`, { scroll: false });
    setMode(d.mode);
    if (d.mode === "amplify") setStrength(d.strength);
    setPrompt(d.prompt);
    setSeed(42);
    setMaxTokens(60);
    await compareWith({ feature_id: d.id, mode: d.mode, strength: d.mode === "amplify" ? d.strength : 0, prompt: d.prompt, seed: 42, max_new_tokens: 60 });
  }

  function showSaved(e: SteeringExample) {
    setShown({
      source: "saved",
      curated: e.is_curated,
      feature: { feature_id: e.feature_id, label: feature?.feature_id === e.feature_id ? feature.label : null },
      mode: e.mode,
      strength: e.strength,
      seed: e.seed,
      prompt: e.prompt,
      baseline: e.baseline_text,
      steered: e.steered_text,
    });
  }

  const savedItems = saved.data?.items ?? [];

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Steer</h1>
          <p className="lede">Turn one feature up or off. Same seed, two outputs.</p>
          <div className={s.demos}>
            <span className="small muted">Saved runs:</span>
            {DEMOS.map((d) => (
              <button key={d.id} type="button" className={s.demo} onClick={() => loadDemo(d)} disabled={steer.loading || check.loading}>
                <span className="mono">#{d.id}</span> {d.mode === "amplify" ? "up" : "off"} · “{d.prompt.length > 26 ? `${d.prompt.slice(0, 26)}…` : d.prompt}”
              </button>
            ))}
          </div>
        </div>
      </div>

      <RunNote />
      <div className={s.layout}>
        <form
          className={`card ${s.controls}`}
          onSubmit={(e) => {
            e.preventDefault();
            run();
          }}
        >
          <div className="field">
            <span className="label">Feature</span>
            {urlFeature.loading && !feature ? <Skeleton h={38} r={10} /> : <FeaturePicker value={feature} onChange={choose} />}
            {urlFeature.error && <span className="hint">Couldn&apos;t load feature #{urlId}: {urlFeature.error.message}</span>}
            {feature && (
              <Link href={`/features/${feature.feature_id}`} className={s.peek}>
                See its examples →
              </Link>
            )}
          </div>

          <div className="field">
            <span className="label">What to do</span>
            <div className={s.modes} role="radiogroup" aria-label="Mode">
              <button type="button" role="radio" aria-checked={mode === "amplify"} className={s.mode} onClick={() => setMode("amplify")}>
                <b>Force on</b>
                <span>Push it up</span>
              </button>
              <button type="button" role="radio" aria-checked={mode === "ablate"} className={s.mode} onClick={() => setMode("ablate")}>
                <b>Switch off</b>
                <span>Take it out</span>
              </button>
            </div>
          </div>

          <div className="field">
            <label htmlFor="st-strength" className={s.labelRow}>
              <span>Strength</span>
              <span className="mono">{mode === "amplify" ? `${fmtNum(strength, 2)}×` : "n/a"}</span>
            </label>
            <input
              id="st-strength"
              type="range"
              min={0}
              max={3}
              step={0.25}
              value={strength}
              disabled={mode === "ablate"}
              onChange={(e) => setStrength(Number(e.target.value))}
            />
            <div className={s.scale} aria-hidden="true">
              <span>0</span>
              <span>1× its peak</span>
              <span>2×</span>
              <span>3×</span>
            </div>
            <span className="hint">
              {mode === "amplify" ? (strength >= 2 ? "Too strong breaks the text." : "1× = its own peak.") : "Strength doesn't apply."}
            </span>
          </div>

          <div className="field">
            <label htmlFor="st-prompt">Prompt</label>
            <textarea id="st-prompt" className="textarea" rows={3} value={prompt} onChange={(e) => setPrompt(e.target.value)} />
          </div>

          <div className={s.pair}>
            <div className="field">
              <label htmlFor="st-seed">Seed</label>
              <div className="row" style={{ gap: 6, flexWrap: "nowrap" }}>
                <input
                  id="st-seed"
                  className="input mono"
                  type="number"
                  min={0}
                  max={2147483647}
                  value={seed}
                  onChange={(e) => setSeed(Math.max(0, Math.min(2147483647, Math.floor(Number(e.target.value) || 0))))}
                />
                <button type="button" className="icon-btn" title="New random seed" aria-label="New random seed" onClick={() => setSeed(Math.floor(Math.random() * 100000))}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                    <rect x="3.5" y="3.5" width="17" height="17" rx="4" />
                    <circle cx="8.5" cy="8.5" r="1.2" fill="currentColor" />
                    <circle cx="15.5" cy="15.5" r="1.2" fill="currentColor" />
                    <circle cx="12" cy="12" r="1.2" fill="currentColor" />
                  </svg>
                </button>
              </div>
            </div>
            <div className="field">
              <label htmlFor="st-len">Length</label>
              <select id="st-len" className="select" value={maxTokens} onChange={(e) => setMaxTokens(Number(e.target.value))}>
                {[30, 60, 90, 120].map((n) => (
                  <option key={n} value={n}>
                    {n} tokens
                  </option>
                ))}
              </select>
            </div>
          </div>

          <button className="btn btn-lg" type="submit" disabled={!feature || !prompt.trim() || steer.loading || check.loading}>
            {steer.loading || check.loading ? "Generating…" : t("button.steer")}
          </button>
          <p className={s.limit}>{t("note.steering")}</p>
        </form>

        <div className="stack" style={{ minWidth: 0 }}>
          <div className={s.tabs}>
            <div className="seg" role="group" aria-label="Which test">
              <button type="button" aria-pressed={tab === "compare"} onClick={() => setTab("compare")}>
                Compare
              </button>
              <button type="button" aria-pressed={tab === "sweep"} onClick={() => setTab("sweep")}>
                Every strength
              </button>
              <button type="button" aria-pressed={tab === "prompt"} onClick={() => setTab("prompt")}>
                Prompt only
              </button>
              {hasCombine && (
                <button type="button" aria-pressed={tab === "combine"} onClick={() => setTab("combine")}>
                  Combine
                </button>
              )}
            </div>
          </div>

          <div hidden={tab !== "sweep"}>
            <SweepPanel
              feature={feature}
              prompt={prompt}
              seed={seed}
              maxTokens={maxTokens}
              simulated={isMockModel(h)}
              onUse={(v) => {
                setMode("amplify");
                setStrength(v);
                if (feature) compareWith({ feature_id: feature.feature_id, mode: "amplify", strength: v, prompt, seed, max_new_tokens: maxTokens });
              }}
            />
          </div>
          <div hidden={tab !== "prompt"}>
            <PromptOnlyPanel
              prompt={prompt}
              seed={seed}
              maxTokens={maxTokens}
              simulated={isMockModel(h)}
              steered={
                shown
                  ? { prompt: shown.prompt, text: shown.steered, seed: shown.seed, what: shown.mode === "amplify" ? `Steering · up ${+shown.strength.toFixed(2)}×` : "Steering · off" }
                  : null
              }
            />
          </div>

          {hasCombine && (
            <div hidden={tab !== "combine"}>
              <CombinePanel
                prompt={prompt}
                seed={seed}
                maxTokens={maxTokens}
                onSetup={(v) => {
                  setPrompt(v.prompt);
                  setSeed(v.seed);
                  setMaxTokens(v.maxTokens);
                }}
              />
            </div>
          )}
          <div hidden={tab !== "compare"} className="stack">
          {steer.error && (
            <ErrorState
              error={steer.error}
              onRetry={run}
            />
          )}
          {steer.error?.code === "MODEL_NOT_LOADED" && savedItems.length > 0 && (
            <p className="small muted">Saved examples below were recorded earlier.</p>
          )}

          {steer.loading && !shown && <ResultSkeleton />}
          {shown ? (
            <Comparison shown={shown} simulated={isMockModel(h)} busy={steer.loading || check.loading} checkError={check.error} run={h?.run_id ?? null} />
          ) : (
            !steer.loading &&
            !steer.error && (
              <div className="card">
                <EmptyState title={feature ? "Ready" : "Pick a feature"} icon="spark">
                  {feature ? `Press ${t("button.steer")}.` : "Search by name or #."}
                </EmptyState>
              </div>
            )
          )}

          <section className="card" aria-labelledby="saved-h">
            <div className="card-head">
              <h2 id="saved-h">Saved examples{feature ? ` for #${feature.feature_id}` : ""}</h2>
              
            </div>
            {saved.loading && !saved.data && (
              <div className="card-pad stack">
                <Skeleton h={44} />
                <Skeleton h={44} />
              </div>
            )}
            {saved.error && (
              <div className="card-pad">
                <ErrorState error={saved.error} onRetry={saved.reload} compact />
              </div>
            )}
            {saved.data && !savedItems.length && (
              <p className={s.noneSaved}>
                None yet.
              </p>
            )}
            {savedItems.length > 0 && (
              <ul className={s.savedList}>
                {savedItems.map((e) => (
                  <li key={e.example_id}>
                    <button className={s.savedItem} onClick={() => showSaved(e)}>
                      <span className={s.savedTop}>
                        <span className="mono muted">#{e.feature_id}</span>
                        <span className={`pill ${e.mode === "amplify" ? "pill-named" : "pill-dead"}`}>
                          {e.mode === "amplify" ? `Amplify ${fmtNum(e.strength, 2)}×` : "Ablate"}
                        </span>
                        {e.is_curated && <span className="pill pill-outline">Curated</span>}
                        <span className="mono muted small">seed {e.seed}</span>
                      </span>
                      <span className={s.savedPrompt}>{e.prompt}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>
          </div>
        </div>
      </div>
    </div>
  );
}

function Comparison(props: { shown: Shown; simulated: boolean; busy: boolean; checkError: ApiError | null; run: string | null }) {
  const { shown, simulated, busy, checkError, run } = props;
  const segs = useMemo(() => diffWords(shown.baseline, shown.steered), [shown.baseline, shown.steered]);
  const [copied, setCopied] = useState(false);
  // S5: a link that reruns this exact result (live runs only: a saved run's length is not recorded)
  async function copyLink() {
    if (shown.len === undefined) return;
    const l = { feature_id: shown.feature.feature_id, mode: shown.mode, strength: shown.strength, seed: shown.seed, prompt: shown.prompt, max_new_tokens: shown.len };
    try {
      await navigator.clipboard.writeText(steerLink(window.location.origin, l, run));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked */
    }
  }
  return (
    <section className={`card ${s.result} ${busy ? s.dim : ""}`} aria-labelledby="cmp-h">
      <div className="card-head">
        <h2 id="cmp-h" className={s.resultTitle}>
          <span className="mono muted">#{shown.feature.feature_id}</span> {featureName(shown.feature.label, shown.feature.feature_id)}
        </h2>
        <div className="row" style={{ gap: 6 }}>
          {shown.source === "live" ? (
            <span className={`pill ${simulated ? "pill-sim" : "pill-live"} pill-dot`}>{simulated ? "Simulated" : t("badge.live")}</span>
          ) : (
            <span className="pill pill-saved pill-dot">{t("badge.saved")}</span>
          )}
          {shown.curated && <span className="pill pill-outline">Curated</span>}
          {shown.source === "live" && shown.len !== undefined && (
            <button type="button" className="btn btn-ghost btn-sm" onClick={copyLink} title="A link that reruns this result with the same feature, prompt, strength, seed and mode">
              {copied ? "Link copied" : "Copy link"}
            </button>
          )}
        </div>
      </div>
      {shown.cmp ? (
        <>
          <SplitView c={shown.cmp} prompt={shown.prompt} />
          <ActivationCheck c={shown.chk ?? null} error={shown.chk ? null : checkError} />
        </>
      ) : (
        <>
      <div className={s.promptRow}>
        <span className="eyebrow">Prompt</span>
        <p>{shown.prompt}</p>
      </div>
      <div className={s.cols}>
        <div className={s.col}>
          <div className={s.colHead}>
            <b>Before</b>
            <span className="mono muted small">seed {shown.seed}</span>
          </div>
          <p className={s.text}>{shown.baseline}</p>
        </div>
        <div className={`${s.col} ${s.colSteered}`}>
          <div className={s.colHead}>
            <b>{shown.mode === "amplify" ? `After · up ${fmtNum(shown.strength, 2)}×` : "After · off"}</b>
            <span className="mono muted small">seed {shown.seed}</span>
          </div>
          <p className={s.text}>
            {segs.map((sg, i) =>
              sg.changed ? (
                <mark key={i} className={s.changed}>
                  {sg.text}
                </mark>
              ) : (
                <span key={i}>{sg.text}</span>
              ),
            )}
          </p>
        </div>
      </div>
      <div className={s.foot}>
        <span className="small muted">
          <mark className={s.changed}>Highlighted</mark> = changed words
        </span>
      </div>
        </>
      )}
    </section>
  );
}

function ResultSkeleton() {
  return (
    <div className="card card-pad stack">
      <Skeleton w="40%" h={16} />
      <div className={s.cols} style={{ padding: 0 }}>
        <div className="stack" style={{ gap: 8 }}>
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} h={14} />
          ))}
        </div>
        <div className="stack" style={{ gap: 8 }}>
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} h={14} />
          ))}
        </div>
      </div>
    </div>
  );
}
