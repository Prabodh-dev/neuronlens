"use client";

import { useEffect, useRef } from "react";
import type { FeatureListItem } from "@/lib/api/types";
import { featureName } from "@/lib/format";
import s from "./FeatureGrid.module.css";

export type GridPin = { id: number; mode: "amplify" | "ablate" } | null;

type Props = {
  /** Layout order: index 0 sits in the centre, then square rings outward. */
  neurons: FeatureListItem[];
  /** feature_id → 0..1, how close to its own peak the feature is firing right now. */
  activity: Map<number, number>;
  /** The feature to fly into, or null for the whole map. */
  open: number | null;
  pin: GridPin;
  /** Features that match the search; the rest dim. */
  highlight: Set<number> | null;
  onOpen: (id: number) => void;
  /** Called when the camera has arrived inside `open`. */
  onArrive: (id: number) => void;
};

/** Every feature as a square tile. Scroll to zoom (numbers, then names, appear in the tiles), drag to
 *  move, click a tile to fly into it. Flights pull back first when the target is far away. */
export function FeatureGrid({ neurons, activity, open, pin, highlight, onOpen, onArrive }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const tipRef = useRef<HTMLDivElement>(null);
  const zoomRef = useRef<HTMLSpanElement>(null);
  const cb = useRef({ onOpen, onArrive });
  const ctrl = useRef<{ zoom: (k: number) => void; reset: () => void } | null>(null);
  useEffect(() => {
    cb.current = { onOpen, onArrive };
  }, [onOpen, onArrive]);

  const S = useRef({
    n: 0,
    R: 1,
    gx: new Int16Array(0),
    gy: new Int16Array(0),
    at: new Int32Array(0), // grid cell → feature index, -1 when empty
    peak: new Float32Array(0),
    rank: new Float32Array(0),
    dead: new Uint8Array(0),
    named: new Uint8Array(0),
    act: new Float32Array(0),
    tgt: new Float32Array(0),
    hl: new Uint8Array(0),
    hlOn: false,
    index: new Map<number, number>(),
    list: [] as FeatureListItem[],
    // camera: centre in grid units and pixels per unit; free navigation eases toward t*
    cx: 0,
    cy: 0,
    sc: 0,
    tcx: 0,
    tcy: 0,
    tsc: 0,
    fly: null as null | { x0: number; y0: number; l0: number; x1: number; y1: number; l1: number; arc: number; t0: number; dur: number; done?: () => void },
    back: null as null | { cx: number; cy: number; sc: number },
    open: -1,
    hover: -1,
    pin: null as GridPin,
    waves: [] as { i: number; t0: number }[],
    drag: null as null | { x: number; y: number; cx: number; cy: number; moved: boolean },
    w: 600,
    h: 600,
    dpr: 1,
    fitted: false,
  });

  /* layout: a square spiral, most frequent features in the middle */
  useEffect(() => {
    const st = S.current;
    const n = neurons.length;
    st.n = n;
    st.list = neurons;
    st.gx = new Int16Array(n);
    st.gy = new Int16Array(n);
    st.peak = new Float32Array(n);
    st.rank = new Float32Array(n);
    st.dead = new Uint8Array(n);
    st.named = new Uint8Array(n);
    st.act = new Float32Array(n);
    st.tgt = new Float32Array(n);
    st.hl = new Uint8Array(n);
    st.index = new Map();
    let i = n > 0 ? 1 : 0;
    let R = 0;
    for (let r = 1; i < n; r++) {
      R = r;
      for (let x = -r; x <= r && i < n; x++, i++) [st.gx[i], st.gy[i]] = [x, -r];
      for (let y = -r + 1; y <= r && i < n; y++, i++) [st.gx[i], st.gy[i]] = [r, y];
      for (let x = r - 1; x >= -r && i < n; x--, i++) [st.gx[i], st.gy[i]] = [x, r];
      for (let y = r - 1; y >= -r + 1 && i < n; y--, i++) [st.gx[i], st.gy[i]] = [-r, y];
    }
    st.R = Math.max(1, R);
    const side = 2 * st.R + 1;
    st.at = new Int32Array(side * side).fill(-1);
    let max = 0;
    for (const f of neurons) max = Math.max(max, f.max_activation);
    neurons.forEach((f, k) => {
      st.at[(st.gy[k] + st.R) * side + (st.gx[k] + st.R)] = k;
      st.peak[k] = max ? f.max_activation / max : 0;
      // most frequent first, so the centre of the map glows and it fades outward
      st.rank[k] = 1 - k / Math.max(1, n - 1);
      st.dead[k] = f.is_dead ? 1 : 0;
      st.named[k] = f.label ? 1 : 0;
      st.index.set(f.feature_id, k);
    });
    st.fitted = false;
  }, [neurons]);

  useEffect(() => {
    const st = S.current;
    const next = new Float32Array(st.n);
    activity.forEach((v, id) => {
      const k = st.index.get(id);
      if (k !== undefined) next[k] = v;
    });
    st.tgt = next;
  }, [activity, neurons]);

  useEffect(() => {
    const st = S.current;
    st.hl.fill(0);
    st.hlOn = !!highlight && highlight.size > 0;
    highlight?.forEach((id) => {
      const k = st.index.get(id);
      if (k !== undefined) st.hl[k] = 1;
    });
  }, [highlight, neurons]);

  useEffect(() => {
    const st = S.current;
    const prev = st.pin;
    st.pin = pin;
    if (pin && pin.mode === "amplify" && (!prev || prev.id !== pin.id || prev.mode !== pin.mode)) {
      const k = st.index.get(pin.id);
      if (k !== undefined) st.waves.push({ i: k, t0: performance.now() });
    }
  }, [pin]);

  /* flying in and out */
  useEffect(() => {
    const st = S.current;
    const k = open === null ? -1 : (st.index.get(open) ?? -1);
    if (k === st.open) return;
    const wasOpen = st.open >= 0;
    st.open = k;
    if (!st.fitted && st.n) {
      st.sc = st.tsc = fitScale(st);
      st.cx = st.cy = st.tcx = st.tcy = 0;
      st.fitted = true;
    }
    if (k >= 0) {
      if (!wasOpen) st.back = { cx: st.tcx, cy: st.tcy, sc: st.tsc };
      // close enough that the tile nearly fills the view, then it opens
      flyTo(st, st.gx[k], st.gy[k], (Math.min(st.w, st.h) * 0.8) / 0.86, () => cb.current.onArrive(st.list[k].feature_id));
    } else {
      const b = st.back ?? { cx: 0, cy: 0, sc: fitScale(st) };
      flyTo(st, b.cx, b.cy, b.sc);
      st.back = null;
    }
  }, [open, neurons]);

  /* the engine */
  useEffect(() => {
    const st = S.current;
    const canvas = canvasRef.current!;
    const wrap = wrapRef.current!;
    const ctx = canvas.getContext("2d")!;
    const font = getComputedStyle(document.documentElement).getPropertyValue("--font-mono").trim() || "monospace";
    const sans = getComputedStyle(document.documentElement).getPropertyValue("--font-sans").trim() || "sans-serif";
    const sprite = buildSprite();

    function fit() {
      const w = Math.max(260, wrap.clientWidth);
      const h = Math.max(260, wrap.clientHeight);
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      if (w === st.w && h === st.h && dpr === st.dpr && canvas.width === Math.round(w * dpr)) return;
      st.w = w;
      st.h = h;
      st.dpr = dpr;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(wrap);

    const local = (e: { clientX: number; clientY: number }) => {
      const r = canvas.getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top] as const;
    };
    const cellAt = (x: number, y: number) => {
      if (!st.sc) return -1;
      const gx = Math.round(st.cx + (x - st.w / 2) / st.sc);
      const gy = Math.round(st.cy + (y - st.h / 2) / st.sc);
      if (Math.abs(gx) > st.R || Math.abs(gy) > st.R) return -1;
      return st.at[(gy + st.R) * (2 * st.R + 1) + (gx + st.R)];
    };
    const tip = (k: number, x: number, y: number) => {
      const el = tipRef.current;
      if (!el) return;
      if (k < 0 || st.open >= 0) {
        el.dataset.on = "0";
        return;
      }
      const f = st.list[k];
      el.dataset.on = "1";
      el.replaceChildren();
      const b = document.createElement("b");
      b.textContent = featureName(f.label, f.feature_id);
      const m = document.createElement("span");
      const a = st.act[k];
      m.textContent = f.is_dead ? `#${f.feature_id} · dead` : a > 0.03 ? `#${f.feature_id} · firing ${Math.round(a * 100)}%` : `#${f.feature_id} · click to open`;
      el.append(b, m);
      el.style.transform = `translate(${Math.max(6, Math.min(x + 16, st.w - 240))}px, ${Math.max(6, Math.min(y + 18, st.h - 60))}px)`;
    };
    const zoomAt = (x: number, y: number, k: number) => {
      const fs = fitScale(st);
      const ns = Math.max(fs * 0.85, Math.min(fs * 80, st.tsc * k));
      const wx = st.tcx + (x - st.w / 2) / st.tsc;
      const wy = st.tcy + (y - st.h / 2) / st.tsc;
      st.tsc = ns;
      st.tcx = wx - (x - st.w / 2) / ns;
      st.tcy = wy - (y - st.h / 2) / ns;
      clamp(st);
    };

    const onDown = (e: PointerEvent) => {
      if (st.open >= 0 || st.fly) return;
      canvas.setPointerCapture(e.pointerId);
      const [x, y] = local(e);
      st.drag = { x, y, cx: st.tcx, cy: st.tcy, moved: false };
    };
    const onMove = (e: PointerEvent) => {
      const [x, y] = local(e);
      if (st.drag) {
        const dx = x - st.drag.x;
        const dy = y - st.drag.y;
        if (!st.drag.moved && Math.abs(dx) + Math.abs(dy) > 4) st.drag.moved = true;
        if (st.drag.moved) {
          st.tcx = st.cx = st.drag.cx - dx / st.tsc;
          st.tcy = st.cy = st.drag.cy - dy / st.tsc;
          clamp(st);
          canvas.style.cursor = "grabbing";
          tip(-1, 0, 0);
          return;
        }
      }
      st.hover = st.open >= 0 ? -1 : cellAt(x, y);
      canvas.style.cursor = st.open >= 0 ? "default" : st.hover >= 0 ? "pointer" : "grab";
      tip(st.hover, x, y);
    };
    const onUp = (e: PointerEvent) => {
      const d = st.drag;
      st.drag = null;
      if (d && !d.moved) {
        const [x, y] = local(e);
        const k = cellAt(x, y);
        if (k >= 0) cb.current.onOpen(st.list[k].feature_id);
      }
    };
    const onLeave = () => {
      st.hover = -1;
      tip(-1, 0, 0);
    };
    const onWheel = (e: WheelEvent) => {
      if (st.open >= 0 || st.fly) return;
      e.preventDefault();
      const [x, y] = local(e);
      zoomAt(x, y, Math.exp(-e.deltaY * 0.0016));
    };
    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointerleave", onLeave);
    canvas.addEventListener("wheel", onWheel, { passive: false });
    ctrl.current = {
      zoom: (k) => st.open < 0 && zoomAt(st.w / 2, st.h / 2, k),
      reset: () => {
        if (st.open >= 0) return;
        flyTo(st, 0, 0, fitScale(st));
      },
    };

    const buckets: number[][] = Array.from({ length: 16 }, () => []);
    let prev = -1;
    let raf = 0;
    const frame = (t: number) => {
      raf = requestAnimationFrame(frame);
      const box = canvas.getBoundingClientRect();
      if (box.bottom <= 0 || box.top >= window.innerHeight || document.visibilityState !== "visible") {
        prev = t;
        return;
      }
      const dt = prev < 0 ? 16 : Math.max(0, Math.min(250, t - prev));
      prev = t;
      fit();
      if (!st.n) return;
      if (!st.fitted) {
        st.sc = st.tsc = fitScale(st);
        st.cx = st.cy = st.tcx = st.tcy = 0;
        st.fitted = true;
      }

      // camera
      if (st.fly) {
        const f = st.fly;
        const u = f.dur <= 1 ? 1 : Math.min(1, Math.max(0, (t - f.t0) / f.dur));
        const e = u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
        st.cx = f.x0 + (f.x1 - f.x0) * e;
        st.cy = f.y0 + (f.y1 - f.y0) * e;
        st.sc = Math.exp(f.l0 + (f.l1 - f.l0) * e - f.arc * Math.sin(Math.PI * u));
        if (u >= 1) {
          st.tcx = f.x1;
          st.tcy = f.y1;
          st.tsc = Math.exp(f.l1);
          st.fly = null;
          f.done?.();
        }
      } else {
        const k = 1 - Math.exp(-dt / 90);
        st.cx += (st.tcx - st.cx) * k;
        st.cy += (st.tcy - st.cy) * k;
        st.sc = Math.exp(Math.log(st.sc) + (Math.log(st.tsc) - Math.log(st.sc)) * k);
      }
      if (zoomRef.current) {
        const z = st.sc / fitScale(st);
        zoomRef.current.textContent = `${z < 10 ? z.toFixed(1) : Math.round(z)}×`;
      }

      // firing: fast attack, slow afterglow
      const rise = 1 - Math.exp(-dt / 60);
      const fall = 1 - Math.exp(-dt / 320);
      for (let i = 0; i < st.n; i++) {
        const d = st.tgt[i] - st.act[i];
        if (d !== 0) {
          st.act[i] += d * (d > 0 ? rise : fall);
          if (Math.abs(d) < 0.002) st.act[i] = st.tgt[i];
        }
      }

      const W = st.w;
      const H = st.h;
      const sc = st.sc;
      const gap = sc < 3 ? 0 : 0.14;
      const half = (sc * (1 - gap)) / 2;
      const ox = W / 2 - st.cx * sc;
      const oy = H / 2 - st.cy * sc;
      ctx.setTransform(st.dpr, 0, 0, st.dpr, 0, 0);
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, W, H);

      // the map's outline
      const edge = (st.R + 0.5) * sc;
      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.lineWidth = 1;
      ctx.strokeRect(ox - edge - 6, oy - edge - 6, edge * 2 + 12, edge * 2 + 12);

      // tiles, grouped by brightness so each group is one fill
      for (const b of buckets) b.length = 0;
      const lit: number[] = [];
      const near: number[] = [];
      const pinK = st.pin ? (st.index.get(st.pin.id) ?? -1) : -1;
      for (let i = 0; i < st.n; i++) {
        const px = ox + st.gx[i] * sc;
        const py = oy + st.gy[i] * sc;
        if (px + half < 0 || px - half > W || py + half < 0 || py - half > H) continue;
        let a = st.dead[i] ? 0.04 : 0.07 + 0.34 * Math.pow(st.rank[i], 1.6) + 0.1 * st.peak[i] + (st.named[i] ? 0.18 : 0);
        const act = i === pinK && st.pin?.mode === "ablate" ? 0 : i === pinK ? Math.max(st.act[i], 0.95) : st.act[i];
        if (act > 0.02) {
          a = Math.max(a, 0.35 + 0.65 * act);
          lit.push(i);
        }
        if (st.hlOn) a = st.hl[i] ? Math.max(a, 0.85) : a * 0.22;
        buckets[Math.min(15, Math.floor(a * 16))].push(i);
        if (sc >= 30) near.push(i);
      }
      const round = sc >= 18;
      for (let lvl = 0; lvl < 16; lvl++) {
        const b = buckets[lvl];
        if (!b.length) continue;
        ctx.fillStyle = `rgba(255,255,255,${((lvl + 0.6) / 16).toFixed(3)})`;
        ctx.beginPath();
        for (const i of b) {
          const px = ox + st.gx[i] * sc;
          const py = oy + st.gy[i] * sc;
          if (round) ctx.roundRect(px - half, py - half, half * 2, half * 2, sc * 0.08);
          else ctx.rect(px - half, py - half, half * 2, half * 2);
        }
        ctx.fill();
      }

      // glow on firing tiles
      ctx.globalCompositeOperation = "lighter";
      for (const i of lit) {
        const px = ox + st.gx[i] * sc;
        const py = oy + st.gy[i] * sc;
        const g = Math.max(16, sc * 3.2) * (0.6 + st.act[i] * 0.6);
        ctx.globalAlpha = Math.min(1, st.act[i] * 0.8);
        ctx.drawImage(sprite, px - g / 2, py - g / 2, g, g);
      }
      ctx.globalAlpha = 1;
      ctx.globalCompositeOperation = "source-over";

      // close up: numbers appear, then names, then how hard it fires
      if (near.length) {
        const fsId = Math.max(9, Math.min(15, sc * 0.13));
        const fsName = Math.max(11, Math.min(26, sc * 0.1));
        ctx.textBaseline = "top";
        for (const i of near) {
          const px = ox + st.gx[i] * sc;
          const py = oy + st.gy[i] * sc;
          const f = st.list[i];
          const lit2 = st.act[i] > 0.35 || (st.hlOn && st.hl[i]);
          const ink = lit2 ? "0,0,0" : "255,255,255";
          const fade = Math.min(1, (sc - 30) / 30);
          ctx.font = `500 ${fsId}px ${font}`;
          ctx.fillStyle = `rgba(${ink},${(0.75 * fade).toFixed(3)})`;
          ctx.fillText(`#${f.feature_id}`, px - half + sc * 0.08, py - half + sc * 0.08);
          if (sc >= 90) {
            const nameFade = Math.min(1, (sc - 90) / 50);
            ctx.font = `600 ${fsName}px ${sans}`;
            ctx.fillStyle = `rgba(${ink},${(0.95 * nameFade).toFixed(3)})`;
            // a name when there is one; unlabelled tiles just say so, quietly
            if (!f.label) ctx.fillStyle = `rgba(${ink},${(0.45 * nameFade).toFixed(3)})`;
            const lines = wrapText(ctx, f.label ?? "Unlabelled", half * 2 - sc * 0.16, 4);
            lines.forEach((ln, j) => ctx.fillText(ln, px - half + sc * 0.08, py - half + sc * 0.08 + fsId * 1.6 + j * fsName * 1.15));
            if (st.act[i] > 0.02) {
              ctx.font = `500 ${fsId}px ${font}`;
              ctx.fillStyle = `rgba(${ink},${(0.8 * nameFade).toFixed(3)})`;
              ctx.fillText(`firing ${Math.round(st.act[i] * 100)}%`, px - half + sc * 0.08, py + half - sc * 0.08 - fsId);
            }
          }
          if (st.dead[i]) {
            ctx.strokeStyle = "rgba(255,255,255,0.25)";
            ctx.beginPath();
            ctx.moveTo(px - half, py - half);
            ctx.lineTo(px + half, py + half);
            ctx.stroke();
          }
        }
      }

      // hover and pin marks
      const mark = (i: number, alpha: number, w = 1.5) => {
        const px = ox + st.gx[i] * sc;
        const py = oy + st.gy[i] * sc;
        const m = Math.max(3, sc * 0.08);
        ctx.strokeStyle = `rgba(255,255,255,${alpha})`;
        ctx.lineWidth = w;
        ctx.beginPath();
        ctx.roundRect(px - half - m, py - half - m, (half + m) * 2, (half + m) * 2, Math.max(3, sc * 0.12));
        ctx.stroke();
      };
      if (st.hover >= 0 && st.open < 0) mark(st.hover, 0.95);
      if (pinK >= 0) {
        mark(pinK, 0.7 + 0.3 * Math.sin(t / 220), 2);
        if (st.pin?.mode === "ablate") {
          const px = ox + st.gx[pinK] * sc;
          const py = oy + st.gy[pinK] * sc;
          ctx.strokeStyle = "rgba(255,255,255,0.8)";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(px - half, py - half);
          ctx.lineTo(px + half, py + half);
          ctx.moveTo(px + half, py - half);
          ctx.lineTo(px - half, py + half);
          ctx.stroke();
        }
      }
      // force-on bursts: square ripples
      st.waves = st.waves.filter((wv) => t - wv.t0 < 1400);
      for (const wv of st.waves) {
        const u = (t - wv.t0) / 1400;
        const px = ox + st.gx[wv.i] * sc;
        const py = oy + st.gy[wv.i] * sc;
        for (let r = 0; r < 3; r++) {
          const uu = Math.max(0, u - r * 0.15);
          if (uu <= 0) continue;
          const grow = half + (1 - Math.pow(1 - uu, 3)) * Math.max(40, sc * 2.5);
          ctx.globalAlpha = (1 - uu) * 0.7;
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 1.5;
          ctx.strokeRect(px - grow, py - grow, grow * 2, grow * 2);
        }
      }
      ctx.globalAlpha = 1;
    };
    raf = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointerleave", onLeave);
      canvas.removeEventListener("wheel", onWheel);
      ctrl.current = null;
    };
  }, []);

  return (
    <div
      ref={wrapRef}
      className={s.wrap}
      data-open={open !== null ? "1" : "0"}
      tabIndex={0}
      aria-label="Map of every feature. Scroll to zoom, drag to move, click a tile to open it. Plus and minus zoom, zero resets."
      onKeyDown={(e) => {
        if (e.key === "+" || e.key === "=") ctrl.current?.zoom(1.6);
        else if (e.key === "-" || e.key === "_") ctrl.current?.zoom(1 / 1.6);
        else if (e.key === "0") ctrl.current?.reset();
        else return;
        e.preventDefault();
      }}
    >
      <canvas ref={canvasRef} className={s.canvas} />
      <div ref={tipRef} className={s.tip} data-on="0" />
      <div className={s.zoomBox}>
        <button type="button" className={s.zb} onClick={() => ctrl.current?.zoom(1 / 1.6)} aria-label="Zoom out">
          −
        </button>
        <span ref={zoomRef} className={s.zoomRead}>
          1.0×
        </span>
        <button type="button" className={s.zb} onClick={() => ctrl.current?.zoom(1.6)} aria-label="Zoom in">
          +
        </button>
        <button type="button" className={s.zb} onClick={() => ctrl.current?.reset()} aria-label="Show the whole map" title="Whole map (0)">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden="true">
            <path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" />
          </svg>
        </button>
      </div>
    </div>
  );
}

/* ---------------- helpers ---------------- */

type St = {
  R: number;
  w: number;
  h: number;
  cx: number;
  cy: number;
  sc: number;
  tcx: number;
  tcy: number;
  tsc: number;
  fly: null | { x0: number; y0: number; l0: number; x1: number; y1: number; l1: number; arc: number; t0: number; dur: number; done?: () => void };
};

function fitScale(st: { R: number; w: number; h: number }) {
  return (Math.min(st.w, st.h) * 0.9) / (2 * st.R + 1);
}

function clamp(st: St) {
  const lim = st.R + 0.5;
  st.tcx = Math.max(-lim, Math.min(lim, st.tcx));
  st.tcy = Math.max(-lim, Math.min(lim, st.tcy));
}

/** Fly the camera to (x, y) at scale `sc`. If both ends are close up and far apart, pull back mid-way
 *  so the trip reads as zoom out, glide, zoom in. */
function flyTo(st: St, x: number, y: number, sc: number, done?: () => void) {
  const reduce = typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const l0 = Math.log(st.sc || sc);
  const l1 = Math.log(sc);
  const dist = Math.hypot(x - st.cx, y - st.cy);
  // the scale at which both points fit on screen
  const both = Math.log((Math.min(st.w, st.h) * 0.7) / Math.max(1, dist + 2));
  const arc = Math.max(0, Math.min(l0, l1) - both) * 0.9;
  st.fly = { x0: st.cx, y0: st.cy, l0, x1: x, y1: y, l1, arc, t0: performance.now(), dur: reduce ? 1 : 900 + Math.min(900, Math.abs(l1 - l0) * 140 + arc * 220), done };
}

function wrapText(ctx: CanvasRenderingContext2D, text: string, width: number, max: number) {
  const words = text.split(/\s+/);
  const lines: string[] = [];
  let line = "";
  for (const w of words) {
    const tryLine = line ? `${line} ${w}` : w;
    if (ctx.measureText(tryLine).width <= width || !line) line = tryLine;
    else {
      lines.push(line);
      line = w;
      if (lines.length === max - 1) break;
    }
  }
  if (line && lines.length < max) lines.push(line);
  const used = lines.join(" ").split(/\s+/).length;
  if (used < words.length && lines.length) lines[lines.length - 1] = lines[lines.length - 1].replace(/\s*\S*$/, "") + "…";
  return lines;
}

function buildSprite(): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d")!;
  const gr = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  gr.addColorStop(0, "rgba(255,255,255,0.9)");
  gr.addColorStop(0.2, "rgba(255,255,255,0.4)");
  gr.addColorStop(0.55, "rgba(255,255,255,0.08)");
  gr.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = gr;
  g.fillRect(0, 0, 128, 128);
  return c;
}
