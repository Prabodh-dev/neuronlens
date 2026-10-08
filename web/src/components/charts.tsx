"use client";

import { useEffect, useRef, useState } from "react";
import s from "./charts.module.css";

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [w, setW] = useState(600);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(260, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

/** Round tick values for a range. */
function ticks(lo: number, hi: number, n = 4): number[] {
  if (hi <= lo) return [lo];
  const raw = (hi - lo) / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((st) => st >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Number(v.toPrecision(10)));
  return out;
}

const M = { l: 52, r: 18, t: 16, b: 38 };

type LinePoint = { x: number; y: number };

export function LineChart({
  points,
  height = 260,
  xLabel,
  yFormat = (v) => String(v),
  xFormat = (v) => String(v),
  label,
}: {
  points: LinePoint[];
  height?: number;
  xLabel: string;
  yFormat?: (v: number) => string;
  xFormat?: (v: number) => string;
  label: string;
}) {
  const [ref, W] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  if (points.length < 2) return <p className="muted small">Not enough points to draw a curve.</p>;

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  const yMax = Math.max(...ys);
  const yMin = Math.min(0, Math.min(...ys));
  const yTicks = ticks(yMin, yMax * 1.05);
  const yTop = yTicks[yTicks.length - 1];
  const X = (v: number) => M.l + ((v - x0) / (x1 - x0 || 1)) * (W - M.l - M.r);
  const Y = (v: number) => M.t + (1 - (v - yMin) / (yTop - yMin || 1)) * (height - M.t - M.b);
  const d = points.map((p, i) => `${i ? "L" : "M"}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join("");
  const area = `${d}L${X(x1).toFixed(1)},${Y(yMin)}L${X(x0).toFixed(1)},${Y(yMin)}Z`;
  const last = points[points.length - 1];
  const hp = hover !== null ? points[hover] : null;

  function onMove(e: React.MouseEvent<SVGSVGElement>) {
    const r = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    let best = 0;
    let bd = Infinity;
    points.forEach((p, i) => {
      const dd = Math.abs(X(p.x) - px);
      if (dd < bd) {
        bd = dd;
        best = i;
      }
    });
    setHover(best);
  }

  return (
    <div ref={ref} className={s.wrap}>
      <svg
        width="100%"
        viewBox={`0 0 ${W} ${height}`}
        role="img"
        aria-label={label}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={M.l} x2={W - M.r} y1={Y(t)} y2={Y(t)} className={s.grid} />
            <text x={M.l - 8} y={Y(t) + 4} textAnchor="end" className={s.tick}>
              {yFormat(t)}
            </text>
          </g>
        ))}
        {ticks(x0, x1, Math.max(2, Math.floor(W / 120))).map((t) => (
          <text key={t} x={X(t)} y={height - M.b + 18} textAnchor="middle" className={s.tick}>
            {xFormat(t)}
          </text>
        ))}
        <text x={(M.l + W - M.r) / 2} y={height - 4} textAnchor="middle" className={s.axis}>
          {xLabel}
        </text>
        <path d={area} className={s.area} />
        <path d={d} className={s.line} pathLength={1} />
        <circle cx={X(last.x)} cy={Y(last.y)} r={4.5} className={s.end} />
        {hp && (
          <g>
            <line x1={X(hp.x)} x2={X(hp.x)} y1={M.t} y2={height - M.b} className={s.cross} />
            <circle cx={X(hp.x)} cy={Y(hp.y)} r={4} className={s.dot} />
          </g>
        )}
      </svg>
      {hp && (
        <div
          className={s.readout}
          style={{ left: `${(X(hp.x) / W) * 100}%`, transform: X(hp.x) > W * 0.7 ? "translateX(-105%)" : "translateX(8px)" }}
        >
          <span className="mono">{yFormat(hp.y)}</span>
          <span style={{ color: "#555" }}>
            {xLabel} {xFormat(hp.x)}
          </span>
        </div>
      )}
    </div>
  );
}

export type Bin = { lo: number; hi: number; count: number };

export function Histogram({
  bins,
  height = 260,
  xLabel,
  xFormat,
  label,
}: {
  bins: Bin[];
  height?: number;
  xLabel: string;
  xFormat: (v: number) => string;
  label: string;
}) {
  const [ref, W] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  if (!bins.length) return <p className="muted small">No bins to draw.</p>;
  const sorted = [...bins].sort((a, b) => a.lo - b.lo);
  const lo = sorted[0].lo;
  const hi = sorted[sorted.length - 1].hi;
  const maxC = Math.max(...sorted.map((b) => b.count), 1);
  const yTicks = ticks(0, maxC);
  const yTop = yTicks[yTicks.length - 1] || maxC;
  const X = (v: number) => M.l + ((v - lo) / (hi - lo || 1)) * (W - M.l - M.r);
  const Y = (v: number) => M.t + (1 - v / yTop) * (height - M.t - M.b);
  const xt = ticks(lo, hi, Math.max(2, Math.floor(W / 110))).filter((t) => Number.isInteger(t));
  const hb = hover !== null ? sorted[hover] : null;

  return (
    <div ref={ref} className={s.wrap}>
      <svg width="100%" viewBox={`0 0 ${W} ${height}`} role="img" aria-label={label} onMouseLeave={() => setHover(null)}>
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={M.l} x2={W - M.r} y1={Y(t)} y2={Y(t)} className={s.grid} />
            <text x={M.l - 8} y={Y(t) + 4} textAnchor="end" className={s.tick}>
              {t.toLocaleString("en-US")}
            </text>
          </g>
        ))}
        {sorted.map((b, i) => {
          const x = X(b.lo) + 1;
          const w = Math.max(1, X(b.hi) - X(b.lo) - 2);
          return (
            <g key={b.lo} onMouseEnter={() => setHover(i)}>
              <rect x={x - 1} y={M.t} width={w + 2} height={height - M.t - M.b} fill="transparent" />
              <rect x={x} y={Y(b.count)} width={w} height={Math.max(0, Y(0) - Y(b.count))} rx={3} className={hover === i ? s.barOn : s.barRect} style={{ animationDelay: `${i * 0.04}s` }} />
            </g>
          );
        })}
        {xt.map((t) => (
          <text key={t} x={X(t)} y={height - M.b + 18} textAnchor="middle" className={s.tick}>
            {xFormat(t)}
          </text>
        ))}
        <text x={(M.l + W - M.r) / 2} y={height - 4} textAnchor="middle" className={s.axis}>
          {xLabel}
        </text>
      </svg>
      {hb && (
        <div className={s.readout} style={{ left: `${(X((hb.lo + hb.hi) / 2) / W) * 100}%`, transform: "translateX(-50%)" }}>
          <span className="mono">{hb.count.toLocaleString("en-US")} features</span>
          <span style={{ color: "#555" }}>
            {xFormat(hb.lo)} to {xFormat(hb.hi)}
          </span>
        </div>
      )}
    </div>
  );
}
