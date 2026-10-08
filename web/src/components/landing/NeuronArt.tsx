"use client";

import { useMemo } from "react";
import s from "./landing.module.css";

/** A drawn neuron: soma, branching dendrites and an axon, with signals running along them. */

type Seg = { id: string; d: string; depth: number; toSoma: boolean; len: number };

function rng(seed: number) {
  let x = seed | 0 || 1;
  return () => {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    return ((x >>> 0) % 100000) / 100000;
  };
}

function build(seed: number) {
  const r = rng(seed);
  const pre = `n${seed}-`;
  const segs: Seg[] = [];
  const cx = 300;
  const cy = 250;
  let n = 0;

  function branch(x: number, y: number, ang: number, len: number, depth: number, w: number) {
    if (depth > 4 || len < 14) return;
    const bend = (r() - 0.5) * 0.9;
    const ex = x + Math.cos(ang) * len;
    const ey = y + Math.sin(ang) * len;
    const mx = x + Math.cos(ang + bend) * len * 0.55;
    const my = y + Math.sin(ang + bend) * len * 0.55;
    segs.push({ id: `${pre}d${n++}`, d: `M${x.toFixed(1)},${y.toFixed(1)} Q${mx.toFixed(1)},${my.toFixed(1)} ${ex.toFixed(1)},${ey.toFixed(1)}`, depth, toSoma: true, len });
    const kids = depth < 2 ? 2 + (r() > 0.5 ? 1 : 0) : r() > 0.35 ? 2 : 1;
    for (let k = 0; k < kids; k++) {
      const spread = (k - (kids - 1) / 2) * (0.5 + r() * 0.35);
      branch(ex, ey, ang + spread + (r() - 0.5) * 0.3, len * (0.62 + r() * 0.18), depth + 1, w * 0.7);
    }
  }

  const primaries = 7;
  for (let i = 0; i < primaries; i++) {
    // dendrites fan upward and sideways; the axon leaves downward
    const a = -Math.PI / 2 + (i - (primaries - 1) / 2) * 0.48 + (r() - 0.5) * 0.25;
    const sx = cx + Math.cos(a) * 26;
    const sy = cy + Math.sin(a) * 22;
    branch(sx, sy, a, 70 + r() * 30, 0, 3);
  }

  // axon
  const pts: [number, number][] = [[cx, cy + 26]];
  let y = cy + 26;
  let x = cx;
  while (y < 520) {
    y += 34;
    x += (r() - 0.5) * 26;
    pts.push([x, y]);
  }
  const axon = pts.reduce((acc, [px, py], i) => (i === 0 ? `M${px},${py}` : `${acc} L${px.toFixed(1)},${py.toFixed(1)}`), "");
  const end = pts[pts.length - 1];
  const terminals: Seg[] = [];
  const boutons: [number, number][] = [];
  for (let k = 0; k < 5; k++) {
    const a = Math.PI / 2 + (k - 2) * 0.45 + (r() - 0.5) * 0.2;
    const l = 30 + r() * 26;
    const ex = end[0] + Math.cos(a) * l;
    const ey = end[1] + Math.sin(a) * l * 0.7;
    terminals.push({ id: `${pre}t${k}`, d: `M${end[0].toFixed(1)},${end[1].toFixed(1)} Q${(end[0] + Math.cos(a) * l * 0.5).toFixed(1)},${(end[1] + 12).toFixed(1)} ${ex.toFixed(1)},${ey.toFixed(1)}`, depth: 5, toSoma: false, len: l });
    boutons.push([ex, ey]);
  }
  // myelin sheath segments along the axon
  const myelin = pts.slice(1, -1).map(([px, py], i) => {
    const [qx, qy] = pts[i];
    return { x1: qx + (px - qx) * 0.2, y1: qy + (py - qy) * 0.2, x2: qx + (px - qx) * 0.8, y2: qy + (py - qy) * 0.8 };
  });
  return { segs, axon, terminals, boutons, myelin, cx, cy, axonId: `${pre}axon`, pre };
}

export function NeuronArt({ seed = 7 }: { seed?: number }) {
  const g = useMemo(() => build(seed), [seed]);
  // signals on the outer dendrite tips travel in toward the cell body
  const signalPaths = g.segs.filter((sg) => sg.depth <= 1).slice(0, 12);

  return (
    <svg className={s.neuron} viewBox="0 0 600 600" role="img" aria-label="Illustration of a neuron with signals travelling along its branches">
      <defs>
        <radialGradient id={`${g.pre}soma`} cx="45%" cy="40%" r="60%">
          <stop offset="0" stopColor="#ffffff" />
          <stop offset="0.55" stopColor="#d9d9d9" />
          <stop offset="1" stopColor="#7a7a7a" />
        </radialGradient>
        <filter id={`${g.pre}glow`} x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="6" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <filter id={`${g.pre}soft`} x="-100%" y="-100%" width="300%" height="300%">
          <feGaussianBlur stdDeviation="18" />
        </filter>
      </defs>

      <circle cx={g.cx} cy={g.cy} r="90" fill="#fff" opacity="0.12" filter={`url(#${g.pre}soft)`} className={s.halo} />

      <g fill="none" stroke="#fff" strokeLinecap="round">
        {g.segs.map((sg) => (
          <path
            key={sg.id}
            id={sg.id}
            d={sg.d}
            pathLength={1}
            strokeWidth={Math.max(0.8, 3.4 - sg.depth * 0.65)}
            opacity={0.9 - sg.depth * 0.13}
            className={s.draw}
            style={{ animationDelay: `${0.15 + sg.depth * 0.22}s` }}
          />
        ))}
        <path id={g.axonId} d={g.axon} pathLength={1} strokeWidth={2.6} opacity={0.85} className={s.draw} style={{ animationDelay: "0.1s" }} />
        {g.myelin.map((m, i) => (
          <line key={i} x1={m.x1} y1={m.y1} x2={m.x2} y2={m.y2} strokeWidth={7} className={s.fade} style={{ animationDelay: `${0.6 + i * 0.05}s`, ["--o" as string]: 0.22 }} />
        ))}
        {g.terminals.map((t) => (
          <path key={t.id} id={t.id} d={t.d} pathLength={1} strokeWidth={1.2} opacity={0.7} className={s.draw} style={{ animationDelay: "1.1s" }} />
        ))}
      </g>
      {g.boutons.map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r={3.2} fill="#fff" className={s.fade} style={{ animationDelay: `${1.4 + i * 0.05}s` }} />
      ))}

      <g filter={`url(#${g.pre}glow)`}>
        <ellipse cx={g.cx} cy={g.cy} rx="30" ry="26" fill={`url(#${g.pre}soma)`} className={s.soma} />
        <circle cx={g.cx - 4} cy={g.cy - 3} r="8" fill="#000" opacity="0.35" />
      </g>

      {/* signals: dendrites → soma, then soma → axon terminals */}
      <g fill="#fff" filter={`url(#${g.pre}glow)`}>
        {signalPaths.map((sg, i) => (
          <circle key={sg.id} r="3" opacity="0">
            <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.1;0.85;1" dur="1.6s" begin={`${1.6 + i * 0.37}s`} repeatCount="indefinite" />
            <animateMotion dur="1.6s" begin={`${1.6 + i * 0.37}s`} repeatCount="indefinite" keyPoints="1;0" keyTimes="0;1" calcMode="linear">
              <mpath href={`#${sg.id}`} />
            </animateMotion>
          </circle>
        ))}
        {[0, 1, 2].map((k) => (
          <circle key={k} r="3.6" opacity="0">
            <animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.08;0.9;1" dur="2.2s" begin={`${2 + k * 0.73}s`} repeatCount="indefinite" />
            <animateMotion dur="2.2s" begin={`${2 + k * 0.73}s`} repeatCount="indefinite">
              <mpath href={`#${g.axonId}`} />
            </animateMotion>
          </circle>
        ))}
      </g>
    </svg>
  );
}
