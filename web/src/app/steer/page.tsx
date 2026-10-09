"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useState } from "react";
import { FeaturePicker, type PickedFeature } from "@/components/FeaturePicker";
import { isMockModel, useHealth } from "@/components/HealthProvider";
import { EmptyState, ErrorState, Skeleton } from "@/components/States";
import { api } from "@/lib/api/client";
import type { SteerMode, SteerRequest, SteeringExample } from "@/lib/api/types";
import { changedShare, diffWords } from "@/lib/diff";
import { featureName, fmtNum } from "@/lib/format";
import { t } from "@/lib/copy";
import { useAction, useApi } from "@/lib/useApi";
import s from "./steer.module.css";

type Shown = {
  source: "live" | "saved";
  curated?: boolean;
  feature: PickedFeature;
  mode: SteerMode;
  strength: number;
  seed: number;
  prompt: string;
  baseline: string;
  steered: string;
};

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
  const [strength, setStrength] = useState(1);
  const [prompt, setPrompt] = useState("The weather here is");
  const [seed, setSeed] = useState(42);
  const [maxTokens, setMaxTokens] = useState(60);
  const [shown, setShown] = useState<Shown | null>(null);

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
  const steer = useAction((body: SteerRequest, sig) => api.steer(body, sig));

  async function run() {
    if (!feature || !prompt.trim()) return;
    const body: SteerRequest = {
      feature_id: feature.feature_id,
      mode,
      strength: mode === "amplify" ? strength : 0,
      prompt,
      seed,
      max_new_tokens: maxTokens,
    };
    const res = await steer.run(body);
    if (res)
      setShown({
        source: "live",
        feature: { feature_id: res.feature.feature_id, label: res.feature.label },
        mode: res.mode,
        strength: res.strength,
        seed: res.seed,
        prompt,
        baseline: res.baseline_text,
        steered: res.steered_text,
      });
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
        </div>
      </div>

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

          <button className="btn btn-lg" type="submit" disabled={!feature || !prompt.trim() || steer.loading}>
            {steer.loading ? "Generating…" : t("button.steer")}
          </button>
          <p className={s.limit}>{t("note.steering")}</p>
        </form>

        <div className="stack" style={{ minWidth: 0 }}>
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
            <Comparison shown={shown} simulated={isMockModel(h)} busy={steer.loading} />
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
  );
}

function Comparison({ shown, simulated, busy }: { shown: Shown; simulated: boolean; busy: boolean }) {
  const segs = useMemo(() => diffWords(shown.baseline, shown.steered), [shown.baseline, shown.steered]);
  const share = changedShare(segs);
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
        </div>
      </div>
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
          <mark className={s.changed}>Highlighted</mark> = changed · {Math.round(share * 100)}% of words
        </span>
      </div>
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
