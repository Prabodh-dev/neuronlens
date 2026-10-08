"use client";

import { useEffect, useRef } from "react";
import type { FeatureListItem } from "@/lib/api/types";
import { featureName } from "@/lib/format";
import s from "./NeuronField.module.css";

export type FieldPin = { id: number; mode: "amplify" | "ablate" } | null;

type Props = {
  /** Layout order: index 0 sits in the centre, the last one on the outer ring. */
  neurons: FeatureListItem[];
  /** feature_id → 0..1, how close to its own peak the neuron is firing right now. */
  activity: Map<number, number>;
  selected: number | null;
  pin: FieldPin;
  isDimmed?: (n: FeatureListItem) => boolean;
  onSelect: (id: number | null) => void;
  /** Glide to this neuron whenever `n` changes. */
  focus?: { id: number; n: number } | null;
};

const MAX_ZOOM = 40;
const TAU = Math.PI * 2;

type Wave = { i: number; t0: number; kind: "fire" | "burst" | "implode" };

type State = {
  n: number;
  x: Float32Array;
  y: Float32Array;
  ring: Uint16Array;
  rings: number;
  peak: Float32Array;
  act: Float32Array;
  tgt: Float32Array;
  hov: Float32Array;
  dim: Uint8Array;
  dead: Uint8Array;
  index: Map<number, number>;
  neurons: FeatureListItem[];
  // camera: current and target
  scale: number;
  ox: number;
  oy: number;
  tScale: number;
  tOx: number;
  tOy: number;
  w: number;
  h: number;
  hover: number;
  sel: number;
  selAt: number;
  linkAt: number;
  pin: FieldPin;
  bornAt: number;
  waves: Wave[];
  drag: null | { x: number; y: number; ox: number; oy: number; moved: boolean };
  sprite: HTMLCanvasElement | null;
  font: string;
};

export function NeuronField({ neurons, activity, selected, pin, isDimmed, onSelect, focus }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const tipRef = useRef<HTMLDivElement>(null);
  const zoomRef = useRef<HTMLSpanElement>(null);
  const st = useRef<State>({
    n: 0,
    x: new Float32Array(0),
    y: new Float32Array(0),
    ring: new Uint16Array(0),
    rings: 1,
    peak: new Float32Array(0),
    act: new Float32Array(0),
    tgt: new Float32Array(0),
    hov: new Float32Array(0),
    dim: new Uint8Array(0),
    dead: new Uint8Array(0),
    index: new Map(),
    neurons: [],
    scale: 1,
    ox: 0,
    oy: 0,
    tScale: 1,
    tOx: 0,
    tOy: 0,
    w: 600,
    h: 600,
    hover: -1,
    sel: -1,
    selAt: 0,
    linkAt: 0,
    pin: null,
    bornAt: 0,
    waves: [],
    drag: null,
    sprite: null,
    font: "monospace",
  });
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;

  // Layout: perfect concentric rings, evenly spaced. Most frequent features in the middle.
  useEffect(() => {
    const S = st.current;
    const n = neurons.length;
    S.n = n;
    S.neurons = neurons;
    S.x = new Float32Array(n);
    S.y = new Float32Array(n);
    S.ring = new Uint16Array(n);
    S.peak = new Float32Array(n);
    S.act = new Float32Array(n);
    S.tgt = new Float32Array(n);
    S.hov = new Float32Array(n);
    S.dim = new Uint8Array(n);
    S.dead = new Uint8Array(n);
    S.index = new Map();
    S.waves = [];
    let maxPeak = 0;
    for (const f of neurons) maxPeak = Math.max(maxPeak, f.max_activation);

    // ring k holds round(2πk) dots; the last ring is spread evenly around the full circle
    const sizes: number[] = [1];
    let placed = 1;
    for (let k = 1; placed < n; k++) {
      const c = Math.min(n - placed, Math.round(TAU * k));
      sizes.push(c);
      placed += c;
    }
    const K = sizes.length - 1 || 1;
    let i = 0;
    sizes.forEach((count, k) => {
      const off = k * 0.61;
      for (let j = 0; j < count && i < n; j++, i++) {
        const a = off + (j / count) * TAU;
        const r = k / K;
        S.x[i] = r * Math.cos(a);
        S.y[i] = r * Math.sin(a);
        S.ring[i] = k;
      }
    });
    S.rings = K;
    neurons.forEach((f, idx) => {
      S.peak[idx] = maxPeak ? f.max_activation / maxPeak : 0;
      S.dead[idx] = f.is_dead ? 1 : 0;
      S.index.set(f.feature_id, idx);
    });
    S.bornAt = performance.now();
  }, [neurons]);

  useEffect(() => {
    const S = st.current;
    const now = performance.now();
    const next = new Float32Array(S.n);
    activity.forEach((v, id) => {
      const i = S.index.get(id);
      if (i !== undefined) next[i] = v;
    });
    // A neuron that jumps up sends out a shockwave.
    for (let i = 0; i < S.n; i++) {
      if (next[i] > 0.35 && next[i] - S.tgt[i] > 0.25 && S.waves.length < 60) S.waves.push({ i, t0: now, kind: "fire" });
    }
    S.tgt = next;
    S.linkAt = now;
  }, [activity, neurons]);

  useEffect(() => {
    const S = st.current;
    for (let i = 0; i < S.n; i++) S.dim[i] = isDimmed && isDimmed(S.neurons[i]) ? 1 : 0;
  }, [isDimmed, neurons]);

  useEffect(() => {
    const S = st.current;
    const i = selected === null ? -1 : (S.index.get(selected) ?? -1);
    if (i !== S.sel) {
      S.selAt = performance.now();
      S.linkAt = S.selAt;
    }
    S.sel = i;
  }, [selected, neurons]);

  useEffect(() => {
    const S = st.current;
    const prev = S.pin;
    S.pin = pin;
    if (pin && (!prev || prev.id !== pin.id || prev.mode !== pin.mode)) {
      const i = S.index.get(pin.id);
      if (i !== undefined) S.waves.push({ i, t0: performance.now(), kind: pin.mode === "amplify" ? "burst" : "implode" });
    }
  }, [pin]);

  useEffect(() => {
    if (!focus) return;
    const S = st.current;
    const i = S.index.get(focus.id);
    if (i === undefined) return;
    const R = radius(S);
    S.tScale = Math.max(S.tScale, Math.min(MAX_ZOOM, Math.max(4, S.rings / 5)));
    S.tOx = -S.x[i] * R * S.tScale;
    S.tOy = -S.y[i] * R * S.tScale;
    clampTarget(S);
  }, [focus]);

  useEffect(() => {
    const S = st.current;
    const canvas = canvasRef.current!;
    const wrap = wrapRef.current!;
    const ctx = canvas.getContext("2d")!;
    S.sprite = buildSprite();
    S.font = getComputedStyle(document.documentElement).getPropertyValue("--font-mono").trim() || "monospace";

    const ro = new ResizeObserver(([e]) => {
      S.w = Math.max(260, Math.round(e.contentRect.width));
      S.h = Math.max(260, Math.round(e.contentRect.height));
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      canvas.width = Math.round(S.w * dpr);
      canvas.height = Math.round(S.h * dpr);
      clampTarget(S);
    });
    ro.observe(wrap);

    let raf = 0;
    const frame = (t: number) => {
      raf = requestAnimationFrame(frame);
      step(S);
      draw(ctx, S, t);
      if (zoomRef.current) zoomRef.current.textContent = `${S.scale < 10 ? S.scale.toFixed(1) : Math.round(S.scale)}×`;
    };
    raf = requestAnimationFrame(frame);

    const pos = (e: PointerEvent | MouseEvent | WheelEvent) => {
      const r = canvas.getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top] as const;
    };
    const tip = (i: number, x: number, y: number) => {
      const el = tipRef.current;
      if (!el) return;
      if (i < 0) {
        el.dataset.on = "0";
        return;
      }
      const f = S.neurons[i];
      const a = S.act[i];
      el.dataset.on = "1";
      el.innerHTML = "";
      const b = document.createElement("b");
      b.textContent = featureName(f.label, f.feature_id);
      const m = document.createElement("span");
      m.textContent = f.is_dead ? `#${f.feature_id} · dead` : a > 0.03 ? `#${f.feature_id} · ${Math.round(a * 100)}%` : `#${f.feature_id}`;
      el.append(b, m);
      el.style.transform = `translate(${Math.max(4, Math.min(x + 16, S.w - 220))}px, ${Math.min(y + 18, S.h - 56)}px)`;
    };

    const onDown = (e: PointerEvent) => {
      canvas.setPointerCapture(e.pointerId);
      const [x, y] = pos(e);
      S.drag = { x, y, ox: S.tOx, oy: S.tOy, moved: false };
    };
    const onMove = (e: PointerEvent) => {
      const [x, y] = pos(e);
      if (S.drag) {
        const dx = x - S.drag.x;
        const dy = y - S.drag.y;
        if (!S.drag.moved && Math.abs(dx) + Math.abs(dy) > 4) S.drag.moved = true;
        if (S.drag.moved) {
          S.tOx = S.ox = S.drag.ox + dx;
          S.tOy = S.oy = S.drag.oy + dy;
          clampTarget(S);
          S.ox = S.tOx;
          S.oy = S.tOy;
          canvas.style.cursor = "grabbing";
          tip(-1, 0, 0);
          return;
        }
      }
      S.hover = nearest(S, x, y);
      canvas.style.cursor = S.hover >= 0 ? "pointer" : "grab";
      tip(S.hover, x, y);
    };
    const onUp = (e: PointerEvent) => {
      const d = S.drag;
      S.drag = null;
      canvas.style.cursor = "grab";
      if (d && !d.moved) {
        const [x, y] = pos(e);
        const i = nearest(S, x, y);
        onSelectRef.current(i >= 0 ? S.neurons[i].feature_id : null);
      }
    };
    const onLeave = () => {
      S.hover = -1;
      tip(-1, 0, 0);
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const [x, y] = pos(e);
      zoomAt(S, x, y, Math.exp(-e.deltaY * 0.0018));
    };
    const onDbl = (e: MouseEvent) => {
      const [x, y] = pos(e);
      zoomAt(S, x, y, 2.5);
    };
    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointerleave", onLeave);
    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("dblclick", onDbl);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointerleave", onLeave);
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("dblclick", onDbl);
    };
  }, []);

  const zoomBy = (k: number) => zoomAt(st.current, st.current.w / 2, st.current.h / 2, k);
  const reset = () => {
    const S = st.current;
    S.tScale = 1;
    S.tOx = 0;
    S.tOy = 0;
  };

  return (
    <div
      ref={wrapRef}
      className={s.wrap}
      tabIndex={0}
      aria-label="Neuron field. Scroll to zoom, drag to move, click a neuron. Plus and minus zoom, zero resets."
      onKeyDown={(e) => {
        if (e.key === "+" || e.key === "=") zoomBy(1.7);
        else if (e.key === "-" || e.key === "_") zoomBy(1 / 1.7);
        else if (e.key === "0") reset();
        else return;
        e.preventDefault();
      }}
    >
      <canvas ref={canvasRef} className={s.canvas} />
      <div ref={tipRef} className={s.tip} data-on="0" />
      <div className={s.zoomBox}>
        <button type="button" className={s.zb} onClick={() => zoomBy(1 / 1.7)} aria-label="Zoom out">
          −
        </button>
        <span ref={zoomRef} className={s.zoomRead}>
          1.0×
        </span>
        <button type="button" className={s.zb} onClick={() => zoomBy(1.7)} aria-label="Zoom in">
          +
        </button>
        <button type="button" className={s.zb} onClick={reset} aria-label="Reset view" title="Reset (0)">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden="true">
            <path d="M4 12a8 8 0 1 0 2.4-5.7M4 4v4h4" />
          </svg>
        </button>
      </div>
    </div>
  );
}

/* ---------------- engine ---------------- */

function radius(S: { w: number; h: number }) {
  return Math.min(S.w, S.h) / 2 - 12;
}

function clampTarget(S: State) {
  const lim = radius(S) * S.tScale * 0.95;
  S.tOx = Math.max(-lim, Math.min(lim, S.tOx));
  S.tOy = Math.max(-lim, Math.min(lim, S.tOy));
}

function zoomAt(S: State, px: number, py: number, k: number) {
  const ns = Math.max(1, Math.min(MAX_ZOOM, S.tScale * k));
  const R = radius(S);
  const cx = S.w / 2;
  const cy = S.h / 2;
  const wx = (px - cx - S.tOx) / (R * S.tScale);
  const wy = (py - cy - S.tOy) / (R * S.tScale);
  S.tScale = ns;
  S.tOx = ns === 1 ? 0 : px - cx - wx * R * ns;
  S.tOy = ns === 1 ? 0 : py - cy - wy * R * ns;
  clampTarget(S);
}

function step(S: State) {
  // camera glides to its target
  const k = 0.16;
  S.scale += (S.tScale - S.scale) * k;
  S.ox += (S.tOx - S.ox) * k;
  S.oy += (S.tOy - S.oy) * k;
  if (Math.abs(S.tScale - S.scale) < 0.001) S.scale = S.tScale;
  // fast attack, slow decay: an afterglow trails each token
  const { act, tgt, hov, n } = S;
  for (let i = 0; i < n; i++) {
    const d = tgt[i] - act[i];
    if (d !== 0) {
      act[i] += d > 0 ? d * 0.25 : d * 0.06;
      if (Math.abs(tgt[i] - act[i]) < 0.002) act[i] = tgt[i];
    }
    const hv = i === S.hover || i === S.sel ? 1 : 0;
    if (hov[i] !== hv) hov[i] += (hv - hov[i]) * 0.25;
  }
}

function pitch(S: State) {
  return (radius(S) * S.scale) / Math.max(1, S.rings);
}

function nearest(S: State, px: number, py: number): number {
  const R = radius(S);
  const cx = S.w / 2;
  const cy = S.h / 2;
  if ((px - cx) ** 2 + (py - cy) ** 2 > (R + 6) ** 2) return -1;
  const lim = Math.max(10, pitch(S) * 0.75);
  let best = -1;
  let bd = lim * lim;
  for (let i = 0; i < S.n; i++) {
    const dx = cx + S.ox + S.x[i] * R * S.scale - px;
    const dy = cy + S.oy + S.y[i] * R * S.scale - py;
    const d = dx * dx + dy * dy - S.act[i] * lim * lim * 0.5;
    if (d < bd) {
      bd = d;
      best = i;
    }
  }
  return best;
}

function buildSprite(): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d")!;
  const gr = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  gr.addColorStop(0, "rgba(255,255,255,1)");
  gr.addColorStop(0.12, "rgba(255,255,255,0.85)");
  gr.addColorStop(0.35, "rgba(255,255,255,0.22)");
  gr.addColorStop(0.7, "rgba(255,255,255,0.05)");
  gr.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = gr;
  g.fillRect(0, 0, 128, 128);
  return c;
}

const easeOut = (x: number) => 1 - Math.pow(1 - Math.max(0, Math.min(1, x)), 3);

function draw(ctx: CanvasRenderingContext2D, S: State, t: number) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const { w, h } = S;
  const R = radius(S);
  const cx = w / 2;
  const cy = h / 2;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const sc = R * S.scale;
  const X = cx + S.ox;
  const Y = cy + S.oy;
  const p = pitch(S);
  const dotR = Math.max(0.9, Math.min(12, p * 0.2));
  const born = t - S.bornAt;

  // lens
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, TAU);
  const lens = ctx.createRadialGradient(cx, cy, 0, cx, cy, R);
  lens.addColorStop(0, "#0d0d0d");
  lens.addColorStop(1, "#030303");
  ctx.fillStyle = lens;
  ctx.fill();
  ctx.clip();

  // slow radar sweep keeps the lens alive while idle
  const sweep = (t / 7000) * TAU;
  const grad = ctx.createConicGradient ? ctx.createConicGradient(sweep, cx, cy) : null;
  if (grad) {
    grad.addColorStop(0, "rgba(255,255,255,0.07)");
    grad.addColorStop(0.08, "rgba(255,255,255,0)");
    grad.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = grad;
    ctx.fillRect(cx - R, cy - R, R * 2, R * 2);
  }

  // resting neurons, revealed ring by ring
  ctx.fillStyle = "#ffffff";
  for (let i = 0; i < S.n; i++) {
    const intro = easeOut((born - S.ring[i] * (900 / Math.max(1, S.rings))) / 500);
    if (intro <= 0) continue;
    const x = X + S.x[i] * sc;
    const y = Y + S.y[i] * sc;
    if (x < -20 || y < -20 || x > w + 20 || y > h + 20) continue;
    const r = dotR * intro * (1 + S.hov[i] * 0.8);
    if (S.dead[i]) {
      ctx.globalAlpha = 0.18 * intro;
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.arc(x, y, r * 0.75, 0, TAU);
      ctx.stroke();
      continue;
    }
    if (S.act[i] > 0.04) continue;
    ctx.globalAlpha = (S.dim[i] ? 0.05 : 0.13 + 0.17 * S.peak[i] + S.hov[i] * 0.55) * intro;
    if (r < 1.6) ctx.fillRect(x - r, y - r, r * 2, r * 2);
    else {
      ctx.beginPath();
      ctx.arc(x, y, r, 0, TAU);
      ctx.fill();
    }
  }

  // links from the selected neuron to everything firing with it on this token
  if (S.sel >= 0) {
    const sx = X + S.x[S.sel] * sc;
    const sy = Y + S.y[S.sel] * sc;
    const grow = easeOut((t - S.linkAt) / 700);
    const cand: number[] = [];
    for (let i = 0; i < S.n; i++) if (i !== S.sel && S.act[i] > 0.12 && !S.dim[i]) cand.push(i);
    cand.sort((a, b) => S.act[b] - S.act[a]);
    ctx.lineWidth = 1;
    for (const i of cand.slice(0, 14)) {
      const tx = X + S.x[i] * sc;
      const ty = Y + S.y[i] * sc;
      const mx = (sx + tx) / 2 + (ty - sy) * 0.18;
      const my = (sy + ty) / 2 - (tx - sx) * 0.18;
      ctx.globalAlpha = 0.15 + S.act[i] * 0.45;
      ctx.strokeStyle = "#ffffff";
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      // draw the curve partially as it grows
      const steps = 24;
      for (let k = 1; k <= Math.ceil(steps * grow); k++) {
        const u = Math.min(grow, k / steps);
        const qx = (1 - u) * (1 - u) * sx + 2 * (1 - u) * u * mx + u * u * tx;
        const qy = (1 - u) * (1 - u) * sy + 2 * (1 - u) * u * my + u * u * ty;
        ctx.lineTo(qx, qy);
      }
      ctx.stroke();
      // a spark travelling along each link
      const u = ((t / 1400 + i * 0.13) % 1) * grow;
      const qx = (1 - u) * (1 - u) * sx + 2 * (1 - u) * u * mx + u * u * tx;
      const qy = (1 - u) * (1 - u) * sy + 2 * (1 - u) * u * my + u * u * ty;
      ctx.globalAlpha = 0.9;
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(qx, qy, 1.6, 0, TAU);
      ctx.fill();
    }
  }

  // firing neurons: additive white bloom that breathes
  ctx.globalCompositeOperation = "lighter";
  const glow = Math.max(9, p * 1.3);
  const spr = S.sprite!;
  for (let i = 0; i < S.n; i++) {
    const a = S.act[i];
    if (a <= 0.04 || S.dead[i]) continue;
    const x = X + S.x[i] * sc;
    const y = Y + S.y[i] * sc;
    const pulse = 1 + Math.min(1, a * 1.5) * 0.16 * Math.sin(t / 260 + i * 1.7);
    const rg = glow * (1 + a * 3.2) * pulse;
    if (x < -rg || y < -rg || x > w + rg || y > h + rg) continue;
    ctx.globalAlpha = S.dim[i] ? 0.12 : Math.min(1, 0.35 + a * 0.75);
    ctx.drawImage(spr, x - rg, y - rg, rg * 2, rg * 2);
  }
  ctx.globalCompositeOperation = "source-over";
  ctx.fillStyle = "#ffffff";
  for (let i = 0; i < S.n; i++) {
    const a = S.act[i];
    if (a <= 0.04 || S.dead[i] || S.dim[i]) continue;
    const x = X + S.x[i] * sc;
    const y = Y + S.y[i] * sc;
    if (x < -10 || y < -10 || x > w + 10 || y > h + 10) continue;
    ctx.globalAlpha = 0.6 + a * 0.4;
    ctx.beginPath();
    ctx.arc(x, y, Math.max(1.4, dotR * (1 + a * 0.9 + S.hov[i] * 0.6)), 0, TAU);
    ctx.fill();
  }

  // shockwaves: firing, force-on bursts, switch-off implosions
  const alive: Wave[] = [];
  for (const wv of S.waves) {
    const dur = wv.kind === "fire" ? 900 : 1300;
    const u = (t - wv.t0) / dur;
    if (u >= 1) continue;
    alive.push(wv);
    const x = X + S.x[wv.i] * sc;
    const y = Y + S.y[wv.i] * sc;
    const e = easeOut(u);
    ctx.strokeStyle = "#ffffff";
    if (wv.kind === "implode") {
      ctx.globalAlpha = 1 - u;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, (1 - e) * glow * 5 + 2, 0, TAU);
      ctx.stroke();
    } else {
      const maxR = wv.kind === "burst" ? glow * 9 : glow * 3.2;
      const rings = wv.kind === "burst" ? 3 : 1;
      for (let k = 0; k < rings; k++) {
        const uk = Math.max(0, e - k * 0.12);
        ctx.globalAlpha = (1 - u) * (wv.kind === "burst" ? 0.8 : 0.45);
        ctx.lineWidth = wv.kind === "burst" ? 1.6 : 1;
        ctx.beginPath();
        ctx.arc(x, y, 3 + uk * maxR, 0, TAU);
        ctx.stroke();
      }
      if (wv.kind === "burst" && u < 0.35) {
        ctx.globalCompositeOperation = "lighter";
        ctx.globalAlpha = (0.35 - u) * 2.4;
        const rb = glow * 6 * e + 6;
        ctx.drawImage(spr, x - rb, y - rb, rb * 2, rb * 2);
        ctx.globalCompositeOperation = "source-over";
      }
    }
  }
  S.waves = alive;

  // names, once zoomed in far enough to read
  if (p > 40) {
    ctx.font = `500 ${Math.min(13, 9 + p / 40)}px ${S.font}`;
    ctx.textBaseline = "middle";
    let drawn = 0;
    for (let i = 0; i < S.n && drawn < 180; i++) {
      const x = X + S.x[i] * sc;
      const y = Y + S.y[i] * sc;
      if (x < 0 || y < 0 || x > w || y > h) continue;
      const f = S.neurons[i];
      const a = S.act[i];
      // named features get their label; the many unlabelled ones only their number
      ctx.globalAlpha = S.dim[i] ? 0.2 : a > 0.04 ? 1 : f.label ? 0.6 : 0.28;
      ctx.fillStyle = "#ffffff";
      const txt = f.label ? f.label : `#${f.feature_id}`;
      ctx.fillText(txt.length > 26 ? txt.slice(0, 25) + "…" : txt, x + dotR + 7, y);
      drawn++;
    }
  }

  // pinned neuron: halo when forced on, crossed ring when switched off
  if (S.pin) {
    const i = S.index.get(S.pin.id);
    if (i !== undefined) {
      const x = X + S.x[i] * sc;
      const y = Y + S.y[i] * sc;
      ctx.strokeStyle = "#ffffff";
      ctx.globalAlpha = 1;
      if (S.pin.mode === "amplify") {
        const r0 = Math.max(12, dotR * 3);
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 4]);
        ctx.lineDashOffset = -t / 40;
        ctx.beginPath();
        ctx.arc(x, y, r0 + 4, 0, TAU);
        ctx.stroke();
        ctx.setLineDash([]);
        const ph = (t / 1100) % 1;
        ctx.globalAlpha = (1 - ph) * 0.8;
        ctx.beginPath();
        ctx.arc(x, y, r0 + ph * r0 * 2.4, 0, TAU);
        ctx.stroke();
      } else {
        const r0 = Math.max(10, dotR * 2.4);
        ctx.globalAlpha = 0.85;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(x, y, r0, 0, TAU);
        ctx.moveTo(x - r0 * 0.7, y - r0 * 0.7);
        ctx.lineTo(x + r0 * 0.7, y + r0 * 0.7);
        ctx.stroke();
      }
    }
  }

  // selection: crosshair ring with a ripple
  if (S.sel >= 0) {
    const x = X + S.x[S.sel] * sc;
    const y = Y + S.y[S.sel] * sc;
    const pop = easeOut((t - S.selAt) / 450);
    const r0 = Math.max(9, dotR * 2.2) * (0.6 + 0.4 * pop);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(x, y, r0, 0, TAU);
    ctx.stroke();
    ctx.beginPath();
    for (const [dx, dy] of [
      [1, 0],
      [-1, 0],
      [0, 1],
      [0, -1],
    ]) {
      ctx.moveTo(x + dx * (r0 + 4), y + dy * (r0 + 4));
      ctx.lineTo(x + dx * (r0 + 11), y + dy * (r0 + 11));
    }
    ctx.stroke();
    const age = (t - S.selAt) / 1000;
    if (age < 1) {
      for (let k = 0; k < 2; k++) {
        const u = Math.max(0, age - k * 0.18);
        ctx.globalAlpha = (1 - u) * 0.7;
        ctx.beginPath();
        ctx.arc(x, y, r0 + easeOut(u) * 80, 0, TAU);
        ctx.stroke();
      }
    }
  }
  ctx.restore();

  // rim and ticks, fixed to the lens
  ctx.globalAlpha = 1;
  ctx.strokeStyle = "rgba(255,255,255,0.22)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, TAU);
  ctx.stroke();
  for (let k = 0; k < 72; k++) {
    const a = (k / 72) * TAU;
    const long = k % 6 === 0;
    const r1 = R + (long ? 7 : 3.5);
    ctx.strokeStyle = long ? "rgba(255,255,255,0.4)" : "rgba(255,255,255,0.14)";
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(a) * (R + 1.5), cy + Math.sin(a) * (R + 1.5));
    ctx.lineTo(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1);
    ctx.stroke();
  }
}
