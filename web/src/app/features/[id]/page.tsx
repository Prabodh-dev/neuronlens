"use client";

import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { FeatureTitle, StatusPill } from "@/components/FeatureBits";
import { HeatText } from "@/components/HeatText";
import { isSampleData, useHealth } from "@/components/HealthProvider";
import { LogitBars } from "@/components/LogitBars";
import { Cofire } from "./cofire";
import { EmptyState, ErrorState, Skeleton } from "@/components/States";
import { api } from "@/lib/api/client";
import type { Example, FeatureDetail } from "@/lib/api/types";
import { featureName, fmtDensity, fmtNum, oneIn } from "@/lib/format";
import { heatCss } from "@/lib/heat";
import { t } from "@/lib/copy";
import { useApi } from "@/lib/useApi";
import s from "./detail.module.css";

const FIRST = 8;

export default function FeaturePage() {
  const params = useParams<{ id: string }>();
  const id = Number(params.id);
  const valid = Number.isInteger(id) && id >= 0;
  const { data: h } = useHealth();
  const { data: f, error, loading, reload } = useApi((sig) => api.feature(id, sig), valid ? `feature:${id}` : null);

  useEffect(() => {
    if (f) document.title = `${featureName(f.label, f.feature_id)} · NeuronLens`;
  }, [f]);

  const lastId = h ? h.d_sae - 1 : null;

  return (
    <div className="page">
      <nav className={s.crumbs} aria-label="Feature navigation">
        <Link href="/features" className={s.back}>
          ← Features
        </Link>
        <span className="spacer" />
        {valid && id > 0 && (lastId === null || id - 1 <= lastId) && (
          <Link href={`/features/${id - 1}`} className="btn btn-ghost btn-sm" aria-label="Previous feature">
            ← #{id - 1}
          </Link>
        )}
        {valid && (lastId === null || id < lastId) && (
          <Link href={`/features/${id + 1}`} className="btn btn-ghost btn-sm" aria-label="Next feature">
            #{id + 1} →
          </Link>
        )}
      </nav>

      {!valid && (
        <div className="card">
          <EmptyState title="That isn't a feature number" action={<Link className="btn" href="/features">Browse features</Link>}>
            Feature numbers are whole numbers starting at 0.
          </EmptyState>
        </div>
      )}
      {valid && error && (
        <div className="card card-pad">
          <ErrorState error={error} onRetry={reload} />
        </div>
      )}
      {valid && loading && !f && <DetailSkeleton />}
      {f && f.feature_id === id && <Detail f={f} sim={isSampleData(h)} />}
    </div>
  );
}

function Detail({ f, sim }: { f: FeatureDetail; sim: boolean }) {
  const [showAll, setShowAll] = useState(false);
  const [copied, setCopied] = useState(false);
  const examples = showAll ? f.examples : f.examples.slice(0, FIRST);
  const freq = oneIn(f.density);

  const router = useRouter();
  const [rolling, setRolling] = useState(false);
  async function randomFeature() {
    setRolling(true);
    try {
      const r = await api.randomFeature();
      router.push(`/features/${r.feature_id}`);
    } catch {
      setRolling(false);
    }
  }

  async function copyLink() {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard blocked */
    }
  }

  return (
    <>
      <header className={`${s.header} bracket`}>
        <div className={s.titleBlock}>
          <span className="eyebrow">
            Feature <span className={s.fid}>#{f.feature_id}</span>
            {sim && " · sample data"}
          </span>
          <h1 className={s.title}>
            <FeatureTitle label={f.label} id={f.feature_id} />
          </h1>
          <div className="row" style={{ gap: 8 }}>
            <StatusPill status={f.label_status} dead={f.is_dead} />
            {f.notes && <span className="muted small">{f.notes}</span>}
          </div>
          <div className={s.actions}>
            {f.is_dead ? (
              <button className="btn" disabled title="A dead feature never fires, so there is nothing to steer.">
                {t("button.steerFeature")}
              </button>
            ) : (
              <Link href={`/steer?feature=${f.feature_id}`} className="btn btn-heat">
                {t("button.steerFeature")}
              </Link>
            )}
            <Link href={`/microscope?neuron=${f.feature_id}`} className="btn btn-ghost">
              Microscope
            </Link>
            <button className="btn btn-ghost" onClick={randomFeature} disabled={rolling}>
              {rolling ? t("loading.default") : t("button.random")}
            </button>
            <button className="btn btn-ghost" onClick={copyLink}>
              {copied ? "Link copied" : "Copy link"}
            </button>
          </div>
        </div>
        <dl className={s.readouts}>
          <div>
            <dt>Density</dt>
            <dd>{fmtDensity(f.density)}</dd>
            <span>{freq ?? "never fired"}</span>
          </div>
          <div>
            <dt>Peak</dt>
            <dd>{fmtNum(f.max_activation, 2)}</dd>
            
          </div>
          <div>
            <dt>Mean</dt>
            <dd>{fmtNum(f.mean_activation, 2)}</dd>
            
          </div>
        </dl>
      </header>

      <div className={s.grid}>
        <section className={s.panel} aria-labelledby="ex-h">
          <div className={s.panelHead}>
            <h2 id="ex-h">Top examples</h2>
            <span className="heat-legend">
              low <i /> high
            </span>
          </div>
          {f.is_dead || !f.examples.length ? (
            <EmptyState title={f.is_dead ? t("feature.dead") : t("empty.examples")} icon="spark" />
          ) : (
            <>
              <ol className={s.examples}>
                {examples.map((ex) => (
                  <ExampleRow key={ex.rank} ex={ex} max={f.max_activation} />
                ))}
              </ol>
              {f.examples.length > FIRST && (
                <div className={s.moreRow}>
                  <button className="btn btn-ghost btn-sm" onClick={() => setShowAll((v) => !v)}>
                    {showAll ? "Show fewer" : `Show all ${f.examples.length}`}
                  </button>
                </div>
              )}
            </>
          )}
        </section>

        <aside className={s.side}>
          <section className={s.panel} aria-labelledby="lg-h">
            <div className={s.panelHead}>
              <h2 id="lg-h">Next-word effect</h2>
            </div>
            <div className={s.panelBody}>
              <LogitBars promoted={f.logits.promoted} suppressed={f.logits.suppressed} />
            </div>
          </section>

          <Cofire id={f.feature_id} />

        </aside>
      </div>
    </>
  );
}

function ExampleRow({ ex, max }: { ex: Example; max: number }) {
  return (
    <li className={s.example}>
      <span className={s.rank}>{String(ex.rank).padStart(2, "0")}</span>
      <div className={s.exBody}>
        <HeatText
          className={s.exText}
          tokens={ex.tokens.map((t, i) => ({
            text: t,
            value: ex.activations[i] ?? 0,
            heat: (ex.activations[i] ?? 0) / (max || 1),
            peak: i === ex.peak_index,
            tip: (
              <>
                <span className="mono">{fmtNum(ex.activations[i] ?? 0)}</span>
                {i === ex.peak_index && " · peak"}
              </>
            ),
          }))}
        />
        <Trace values={ex.activations} peak={ex.peak_index} max={max} />
      </div>
      <span className={s.exMax} title="Peak activation in this passage">
        {fmtNum(ex.max_activation, 1)}
      </span>
    </li>
  );
}

/** Oscilloscope-style trace of activation along the passage. */
function Trace({ values, peak, max }: { values: number[]; peak: number; max: number }) {
  const n = values.length;
  if (n < 2) return null;
  const W = 100;
  const H = 18;
  const pts = values.map((v, i) => `${((i / (n - 1)) * W).toFixed(2)},${(H - (Math.max(0, v) / (max || 1)) * (H - 2) - 1).toFixed(2)}`);
  const px = (peak / (n - 1)) * W;
  return (
    <svg className={s.trace} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
      <polyline points={pts.join(" ")} fill="none" stroke={heatCss(0.62)} strokeWidth="1" vectorEffect="non-scaling-stroke" />
      <line x1={px} x2={px} y1={0} y2={H} stroke={heatCss(0.95)} strokeWidth="1" vectorEffect="non-scaling-stroke" strokeDasharray="2 2" />
    </svg>
  );
}

function DetailSkeleton() {
  return (
    <>
      <div className={s.header}>
        <div className={s.titleBlock}>
          <Skeleton w={160} h={11} />
          <Skeleton w="60%" h={48} />
          <Skeleton w={90} h={22} />
        </div>
        <div className={s.readouts}>
          {[0, 1, 2].map((i) => (
            <div key={i}>
              <Skeleton w="50%" h={10} />
              <Skeleton w="70%" h={30} />
            </div>
          ))}
        </div>
      </div>
      <div className={s.grid}>
        <div className={`${s.panel} ${s.panelBody} stack`}>
          {[0, 1, 2, 3, 4].map((i) => (
            <Skeleton key={i} h={40} />
          ))}
        </div>
        <div className={`${s.panel} ${s.panelBody} stack`}>
          {[0, 1, 2, 3, 4].map((i) => (
            <Skeleton key={i} h={12} />
          ))}
        </div>
      </div>
    </>
  );
}
