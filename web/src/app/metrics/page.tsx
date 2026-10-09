"use client";

import { useState } from "react";
import { Histogram, LineChart } from "@/components/charts";
import { CountUp } from "@/components/CountUp";
import { isSampleData, useHealth } from "@/components/HealthProvider";
import { EmptyState, ErrorState, Skeleton } from "@/components/States";
import { api } from "@/lib/api/client";
import type { MetricsResponse, Run } from "@/lib/api/types";
import { fmtInt, fmtNum, pctDigits } from "@/lib/format";
import { t } from "@/lib/copy";
import { useApi } from "@/lib/useApi";
import s from "./metrics.module.css";

type CurveKey = "loss" | "fvu" | "dead_pct";
const CURVES: { k: CurveKey; label: string; fmt: (v: number) => string }[] = [
  { k: "loss", label: "Loss", fmt: (v) => fmtNum(v, 2) },
  { k: "fvu", label: "Unexplained", fmt: (v) => fmtNum(v, 2) },
  { k: "dead_pct", label: "Dead %", fmt: (v) => `${v.toFixed(pctDigits(v))}%` },
];

function densityTick(v: number): string {
  const p = 10 ** v * 100;
  return p >= 1 ? `${Number(p.toPrecision(2))}%` : `${Number(p.toPrecision(1))}%`;
}

export default function MetricsPage() {
  const { data: h } = useHealth();
  const metrics = useApi((sig) => api.metrics(null, sig), "metrics");
  const runs = useApi((sig) => api.runs(sig), "runs");
  const m = metrics.data;
  const run = runs.data?.items.find((r) => r.run_id === m?.run_id) ?? runs.data?.items.find((r) => r.is_primary) ?? null;

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Metrics</h1>
          <p className="lede">How well the autoencoder copies the model.</p>
        </div>
        {isSampleData(h) && <span className="pill pill-sim">Sample data</span>}
      </div>

      {metrics.error && <ErrorState error={metrics.error} onRetry={metrics.reload} />}
      {metrics.loading && !m && (
        <div className={s.big}>
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className={s.num}>
              <Skeleton w="60%" h={56} />
              <Skeleton w="40%" h={12} />
            </div>
          ))}
        </div>
      )}
      {m && <Body m={m} run={run} />}
    </div>
  );
}

function Body({ m, run }: { m: MetricsResponse; run: Run | null }) {
  const get = (name: string) => m.metrics.find((x) => x.name === name)?.value ?? null;
  const ceRec = get("ce_recovered");
  const fvu = get("fvu");
  const l0 = get("l0");
  const dead = get("dead_pct");
  const ceO = get("ce_original");
  const ceR = get("ce_reconstructed");
  const ceZ = get("ce_zero_ablation");

  const available = CURVES.filter((c) => m.training_curve.some((p) => p[c.k] !== null && p[c.k] !== undefined));
  const [curve, setCurve] = useState<CurveKey>(available[0]?.k ?? "loss");
  const cdef = CURVES.find((c) => c.k === curve) ?? CURVES[0];
  const points = m.training_curve.filter((p) => p[curve] !== null && p[curve] !== undefined).map((p) => ({ x: p.step, y: p[curve] as number }));

  return (
    <>
      <section className={s.big} aria-label="Headline numbers">
        <Num label="Loss recovered" hint="Higher is better">
          {ceRec !== null ? <CountUp to={ceRec * 100} decimals={1} suffix="%" /> : "—"}
        </Num>
        <Num label="Unexplained" hint="Lower is better">
          {fvu !== null ? <CountUp to={fvu} decimals={3} /> : "—"}
        </Num>
        <Num label="Active per token" hint="On average">
          {l0 !== null ? <CountUp to={l0} decimals={1} /> : "—"}
        </Num>
        <Num label="Dead" hint="Never fired">
          {dead !== null ? <CountUp to={dead} decimals={pctDigits(dead)} suffix="%" /> : "—"}
        </Num>
      </section>

      {ceO !== null && ceR !== null && ceZ !== null && (
        <section className={s.loss} aria-labelledby="loss-h">
          <div className={s.lossHead}>
            <h2 id="loss-h">Next-word loss</h2>
            <span className="muted small">lower is better</span>
          </div>
          <LossBar label="Real model" v={ceO} max={ceZ} />
          <LossBar label="With autoencoder" v={ceR} max={ceZ} hot />
          <LossBar label="Layer removed" v={ceZ} max={ceZ} dim />
        </section>
      )}

      <div className={s.charts}>
        <section className="card" aria-labelledby="curve-h">
          <div className="card-head">
            <h2 id="curve-h">Training</h2>
            {available.length > 1 && (
              <div className="seg" role="group" aria-label="Curve">
                {available.map((c) => (
                  <button key={c.k} aria-pressed={curve === c.k} onClick={() => setCurve(c.k)}>
                    {c.label}
                  </button>
                ))}
              </div>
            )}
          </div>
          <div className="card-pad">
            {points.length ? (
              <LineChart
                key={curve}
                points={points}
                xLabel="step"
                yFormat={cdef.fmt}
                xFormat={(v) => (v >= 1000 ? `${fmtNum(v / 1000, v % 1000 ? 1 : 0)}k` : String(v))}
                label={`${cdef.label} by training step`}
              />
            ) : (
              <EmptyState title={t("empty.curve")} icon="chart" />
            )}
          </div>
        </section>

        <section className="card" aria-labelledby="hist-h">
          <div className="card-head">
            <h2 id="hist-h">How often features fire</h2>
          </div>
          <div className="card-pad">
            {m.density_hist.length ? (
              <Histogram
                bins={m.density_hist.map((b) => ({ lo: b.bin_lo, hi: b.bin_hi, count: b.count }))}
                xLabel="share of tokens (log)"
                xFormat={densityTick}
                label="Histogram of feature density"
              />
            ) : (
              <EmptyState title={t("empty.hist")} icon="chart" />
            )}
          </div>
        </section>
      </div>

      {run && (
        <p className={s.runLine}>
          <span>{run.model_name}</span>
          <span>layer {run.hook_layer}</span>
          <span>
            {fmtInt(run.d_model)} → {fmtInt(run.d_sae)}
          </span>
          <span>k {run.k}</span>
          {run.n_train_activations ? <span>{fmtInt(run.n_train_activations)} activations</span> : null}
          <span className="mono">{m.run_id}</span>
        </p>
      )}
    </>
  );
}

function Num({ label, hint, children }: { label: string; hint: string; children: React.ReactNode }) {
  return (
    <div className={s.num}>
      <b>{children}</b>
      <span>{label}</span>
      <small>{hint}</small>
    </div>
  );
}

function LossBar({ label, v, max, hot, dim }: { label: string; v: number; max: number; hot?: boolean; dim?: boolean }) {
  return (
    <div className={`${s.lossRow} ${hot ? s.hot : ""} ${dim ? s.dim : ""}`}>
      <span className={s.lossLabel}>{label}</span>
      <span className={s.lossTrack}>
        <i style={{ width: `${Math.max(1.5, (v / (max || 1)) * 100)}%` }} />
      </span>
      <span className="mono">{fmtNum(v, 3)}</span>
    </div>
  );
}
