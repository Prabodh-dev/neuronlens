"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { useHealth } from "@/components/HealthProvider";
import { API_BASE, api } from "@/lib/api/client";
import type { FeatureDetail, FeatureListItem, Health, InspectResponse, SteeringExample } from "@/lib/api/types";
import { t as copy } from "@/lib/copy";
import { diffWords } from "@/lib/diff";
import { fmtInt, showToken } from "@/lib/format";
import s from "./tour.module.css";

/** The landing hero: a guided tour that plays by itself and loops. It shows every feature, reads a
 *  sentence, flies into a few named features, steers one of them, then pulls back out.
 *  Everything on screen comes from the server: features, /inspect, /steer, examples, metrics. */

// stops the tour visits when the database has them; otherwise the strongest named features
const PREFERRED = [1863, 3587, 3446];
const PAGE = 200;
const TAU = Math.PI * 2;

type Named = { id: number; label: string };
type Frame = { token: number; feats: { id: number; a: number }[] };
type Run = Pick<SteeringExample, "prompt" | "seed" | "mode" | "baseline_text" | "steered_text"> & { live: boolean };
type Scene =
  | { kind: "overview"; chapter: string; dur: number }
  | { kind: "read"; chapter: string; dur: number }
  | { kind: "focus"; chapter: string; dur: number; f: FeatureDetail }
  | { kind: "steer"; chapter: string; dur: number; f: FeatureDetail; ex: Run }
  | { kind: "outro"; chapter: string; dur: number };

/* ---------------- data ---------------- */

/** Every feature, most frequent first, the same order the Microscope uses. */
function useNeurons(enabled: boolean) {
  const [items, setItems] = useState<FeatureListItem[] | null>(null);
  useEffect(() => {
    if (!enabled) return;
    const ctl = new AbortController();
    (async () => {
      try {
        const q = { include_dead: true, sort: "density" as const, order: "desc" as const, limit: PAGE };
        const first = await api.features({ ...q, offset: 0 }, ctl.signal);
        const pages = Math.ceil(first.total / PAGE);
        const all: FeatureListItem[][] = [first.items];
        for (let p = 1; p < pages; p += 6) {
          const batch = await Promise.all(
            Array.from({ length: Math.min(6, pages - p) }, (_, k) => api.features({ ...q, offset: (p + k) * PAGE }, ctl.signal)),
          );
          batch.forEach((b) => all.push(b.items));
        }
        setItems(all.flat());
      } catch {
        // without the list the tour still plays its overview
      }
    })();
    return () => ctl.abort();
  }, [enabled]);
  return items;
}

/** Up to three named features to visit, and the saved steering run whose prompt, feature and seed the tour uses. */
function useStops(named: Named[] | null) {
  const [stops, setStops] = useState<{ list: FeatureDetail[]; steer: SteeringExample | null } | null>(null);
  const key = named ? named.map((n) => n.id).join(",") : null;
  useEffect(() => {
    if (key === null) return;
    const ctl = new AbortController();
    (async () => {
      const ids = key ? key.split(",").map(Number) : [];
      const want = [...PREFERRED, ...ids.filter((id) => !PREFERRED.includes(id))];
      const list: FeatureDetail[] = [];
      for (const id of want) {
        if (list.length >= 3) break;
        try {
          const f = await api.feature(id, ctl.signal);
          if (f.label && !f.is_dead && f.examples.length) list.push(f);
        } catch {
          if (ctl.signal.aborted) return;
        }
      }
      let steer: SteeringExample | null = null;
      try {
        const ex = (await api.steeringExamples(null, ctl.signal)).items.filter((e) => list.some((f) => f.feature_id === e.feature_id));
        steer = ex.find((e) => e.is_curated && e.feature_id === list[0]?.feature_id) ?? ex.find((e) => e.is_curated) ?? ex[0] ?? null;
      } catch {
        if (ctl.signal.aborted) return;
      }
      setStops({ list, steer });
    })();
    return () => ctl.abort();
  }, [key]);
  return stops;
}

/** In real mode the steering chapter runs on the model, with the saved run's prompt, feature and seed. */
function useLiveRun(ex: SteeringExample | null) {
  const [run, setRun] = useState<Run | null>(null);
  useEffect(() => {
    setRun(null);
    if (!ex) return;
    const ctl = new AbortController();
    api
      .steer({ feature_id: ex.feature_id, mode: ex.mode, strength: ex.strength, prompt: ex.prompt, seed: ex.seed, max_new_tokens: 40 }, ctl.signal)
      .then((r) => setRun({ prompt: ex.prompt, seed: r.seed, mode: r.mode, baseline_text: r.baseline_text, steered_text: r.steered_text, live: true }))
      .catch(() => {
        // model not loaded or the call failed: the chapter is left out, never filled from storage
      });
    return () => ctl.abort();
  }, [ex]);
  return run;
}

/* ---------------- text helpers ---------------- */

const END = "<|endoftext|>";

/** A few tokens either side of the peak, cut at document boundaries. */
function snippet(f: FeatureDetail) {
  const ex = f.examples[0];
  const p = ex.peak_index;
  let before = ex.tokens.slice(Math.max(0, p - 9), p).join("");
  let after = ex.tokens.slice(p + 1, p + 7).join("");
  if (before.includes(END)) before = before.slice(before.lastIndexOf(END) + END.length);
  if (after.includes(END)) after = after.slice(0, after.indexOf(END));
  const clean = (t: string) => t.replace(/\s+/g, " ");
  return { before: clean(before).trimStart(), peak: showToken(ex.tokens[p]), after: clean(after).trimEnd() };
}

/** The continuation after the prompt, cut to a few words so it fits. */
function continuation(full: string, prompt: string, words = 18) {
  const rest = full.startsWith(prompt) ? full.slice(prompt.length) : full;
  const w = rest.replace(/\s+/g, " ").trim().split(" ");
  return w.slice(0, words).join(" ") + (w.length > words ? "…" : "");
}

const pad = (n: number) => String(n).padStart(2, "0");

/* ---------------- component ---------------- */

export function Tour({ h, named, demo, ceRec }: { h: Health | null; named: Named[] | null; demo: InspectResponse | null; ceRec: number | null }) {
  const router = useRouter();
  const { error: offline } = useHealth();
  const neurons = useNeurons(!!h);
  const stops = useStops(named);
  const real = h?.mode === "real";
  const liveRun = useLiveRun(real ? (stops?.steer ?? null) : null);
  const tokens = useMemo(() => demo?.tokens ?? [], [demo]);
  const frames = useMemo<Frame[]>(
    () =>
      tokens
        .filter((t) => !t.is_outlier_position && t.features.length > 0)
        .map((t) => ({ token: t.index, feats: t.features.map((f) => ({ id: f.feature_id, a: f.activation })) })),
    [tokens],
  );

  const scenes = useMemo<Scene[]>(() => {
    const out: Scene[] = [{ kind: "overview", chapter: "Overview", dur: 6500 }];
    if (frames.length) out.push({ kind: "read", chapter: "Reading", dur: 1600 + frames.length * 680 });
    for (const f of stops?.list ?? []) out.push({ kind: "focus", chapter: `#${f.feature_id}`, dur: 7600, f });
    // real mode: the model's own run; mock mode: the saved run, since mock steering is simulated
    const saved = stops?.steer;
    const ex: Run | null = real ? liveRun : saved && h?.mode === "mock" ? { ...saved, live: false } : null;
    const sf = saved ? stops?.list.find((f) => f.feature_id === saved.feature_id) : undefined;
    if (ex && sf) out.push({ kind: "steer", chapter: "Steering", dur: 9800, f: sf, ex });
    out.push({ kind: "outro", chapter: "Quality", dur: 7200 });
    return out;
  }, [frames, stops, real, liveRun, h?.mode]);

  const [at, setAt] = useState(0);
  const [tok, setTok] = useState(-1);
  const [playing, setPlaying] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fills = useRef<(HTMLElement | null)[]>([]);
  const ctrl = useRef<{ goto: (i: number) => void; play: (on: boolean) => void } | null>(null);

  // the engine reads the latest data through these
  const live = useRef({ scenes, frames, tokens, neurons, total: h?.d_sae ?? 0, router });
  useEffect(() => {
    live.current = { scenes, frames, tokens, neurons, total: h?.d_sae ?? 0, router };
  }, [scenes, frames, tokens, neurons, h, router]);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) setPlaying(false);
  }, []);
  useEffect(() => {
    ctrl.current?.play(playing);
  }, [playing]);

  /* ---------- the engine ---------- */
  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const font = getComputedStyle(document.documentElement).getPropertyValue("--font-mono").trim() || "monospace";

    const E = {
      n: 0,
      x: new Float32Array(0),
      y: new Float32Array(0),
      ring: new Uint16Array(0),
      rings: 1,
      peak: new Float32Array(0),
      dead: new Uint8Array(0),
      act: new Float32Array(0),
      tgt: new Float32Array(0),
      index: new Map<number, number>(),
      laidOut: null as FeatureListItem[] | null,
      waves: [] as { i: number; t0: number; big: boolean }[],
      camX: 0,
      camY: 0,
      logZ: Math.log(0.8),
      tilt: 1.1,
      spin: 0,
      fly: { x0: 0, y0: 0, z0: Math.log(0.8), tilt0: 1.1, t0: 0, dur: 1, arc: 0 },
      scene: 0,
      sceneT: 0,
      step: -1,
      playing: !reduce,
      bornAt: -1,
      axK: 0.5,
      ayK: 0.56,
      mx: 0,
      my: 0,
      pmx: 0,
      pmy: 0,
      w: 800,
      h: 600,
      dpr: 1,
      focus: null as null | { x: number; y: number; r: number; id: number },
      px: new Float32Array(0),
      py: new Float32Array(0),
    };

    // perfect concentric rings, most frequent features in the middle (as in the Microscope)
    function layout(n: number) {
      E.n = n;
      E.x = new Float32Array(n);
      E.y = new Float32Array(n);
      E.ring = new Uint16Array(n);
      E.peak = new Float32Array(n).fill(0.35);
      E.dead = new Uint8Array(n);
      E.act = new Float32Array(n);
      E.tgt = new Float32Array(n);
      E.px = new Float32Array(n);
      E.py = new Float32Array(n);
      const sizes = [1];
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
          E.x[i] = (k / K) * Math.cos(a);
          E.y[i] = (k / K) * Math.sin(a);
          E.ring[i] = k;
        }
      });
      E.rings = K;
    }
    function sync() {
      const L = live.current;
      const list = L.neurons;
      const n = list ? list.length : L.total;
      if (n > 0 && n !== E.n) layout(n);
      if (list && E.laidOut !== list) {
        E.laidOut = list;
        E.index = new Map();
        let max = 0;
        for (const f of list) max = Math.max(max, f.max_activation);
        list.forEach((f, i) => {
          E.index.set(f.feature_id, i);
          E.peak[i] = max ? f.max_activation / max : 0;
          E.dead[i] = f.is_dead ? 1 : 0;
        });
      }
    }

    const sprite = document.createElement("canvas");
    sprite.width = sprite.height = 128;
    {
      const g = sprite.getContext("2d")!;
      const gr = g.createRadialGradient(64, 64, 0, 64, 64, 64);
      gr.addColorStop(0, "rgba(255,255,255,1)");
      gr.addColorStop(0.14, "rgba(255,255,255,0.75)");
      gr.addColorStop(0.4, "rgba(255,255,255,0.18)");
      gr.addColorStop(1, "rgba(255,255,255,0)");
      g.fillStyle = gr;
      g.fillRect(0, 0, 128, 128);
    }

    const ease = (u: number) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);
    const easeOut = (u: number) => 1 - Math.pow(1 - Math.max(0, Math.min(1, u)), 3);
    const lerp = (a: number, b: number, u: number) => a + (b - a) * u;
    const rot = (i: number): [number, number] => {
      const c = Math.cos(E.spin);
      const sn = Math.sin(E.spin);
      return [E.x[i] * c - E.y[i] * sn, E.x[i] * sn + E.y[i] * c];
    };

    function target(sc: Scene | undefined) {
      const close = Math.log(Math.max(3, Math.min(9, E.rings / 9)));
      if (sc && (sc.kind === "focus" || sc.kind === "steer")) {
        const node = E.index.get(sc.f.feature_id) ?? -1;
        if (node >= 0) return { node, logZ: sc.kind === "steer" ? close + 0.2 : close, tilt: 0.34 };
      }
      if (sc?.kind === "read") return { node: -1, logZ: Math.log(1.06), tilt: 0.82 };
      if (sc?.kind === "outro") return { node: -1, logZ: Math.log(0.9), tilt: 1.05 };
      return { node: -1, logZ: 0, tilt: 1.05 };
    }

    function goto(i: number, t: number) {
      const list = live.current.scenes;
      const idx = ((i % list.length) + list.length) % list.length;
      const tg = target(list[idx]);
      const [nx, ny] = tg.node >= 0 ? rot(tg.node) : [0, 0];
      const dist = Math.hypot(nx - E.camX, ny - E.camY);
      const bothClose = tg.logZ > 0.6 && E.logZ > 0.6;
      E.fly = {
        x0: E.camX,
        y0: E.camY,
        z0: E.logZ,
        tilt0: E.tilt,
        t0: t,
        dur: reduce ? 1 : 1700 + Math.min(1300, dist * 2400),
        arc: reduce ? 0 : Math.min(1.5, dist * 2.4) * (bothClose ? 1 : 0.3),
      };
      E.scene = idx;
      E.sceneT = 0;
      E.step = -1;
      E.tgt.fill(0);
      if (tg.node >= 0) {
        E.tgt[tg.node] = 1;
        E.waves.push({ i: tg.node, t0: t + E.fly.dur * 0.8, big: true });
      }
      setAt(idx);
      setTok(-1);
    }

    // sizing: at start, on resize, and checked every frame in case a resize event is late
    function fit() {
      const w = Math.max(300, canvas.clientWidth);
      const h = Math.max(300, canvas.clientHeight);
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      if (w === E.w && h === E.h && dpr === E.dpr && canvas.width === Math.round(w * dpr)) return;
      E.w = w;
      E.h = h;
      E.dpr = dpr;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(canvas);

    // parallax and clicks on the focused feature
    const onMove = (e: PointerEvent) => {
      E.mx = (e.clientX / window.innerWidth - 0.5) * 2;
      E.my = (e.clientY / window.innerHeight - 0.5) * 2;
      const r = canvas.getBoundingClientRect();
      const f = E.focus;
      const over = !!f && Math.hypot(e.clientX - r.left - f.x, e.clientY - r.top - f.y) < f.r;
      canvas.style.cursor = over ? "pointer" : "default";
    };
    const onClick = (e: MouseEvent) => {
      const r = canvas.getBoundingClientRect();
      const f = E.focus;
      if (f && Math.hypot(e.clientX - r.left - f.x, e.clientY - r.top - f.y) < f.r) live.current.router.push(`/features/${f.id}`);
    };
    window.addEventListener("pointermove", onMove, { passive: true });
    canvas.addEventListener("click", onClick);


    ctrl.current = {
      goto: (i) => goto(i, performance.now()),
      play: (on) => (E.playing = on),
    };

    const buckets: number[][] = Array.from({ length: 16 }, () => []);
    let prev = -1;
    let raf = 0;

    const frame = (t: number) => {
      raf = requestAnimationFrame(frame);
      // pause while scrolled out of view (checked directly, so it never depends on observer timing)
      const box = canvas.getBoundingClientRect();
      const onScreen = box.bottom > 0 && box.top < window.innerHeight;
      if (!onScreen || document.visibilityState !== "visible") {
        prev = t;
        return;
      }
      // time-based, so slow machines play at the same speed; the first frame's timestamp can be
      // earlier than anything recorded before it, so clamp at zero (and cap long stalls)
      const dt = prev < 0 ? 16 : Math.max(0, Math.min(1000, t - prev));
      prev = t;
      fit();
      sync();
      if (E.n === 0) return;
      if (E.bornAt < 0) E.bornAt = t;
      const L = live.current;
      if (E.scene >= L.scenes.length) E.scene = 0;

      // clock: scenes advance only while playing and on screen
      if (E.playing) E.sceneT += dt;
      let sc = L.scenes[E.scene];
      if (sc && E.sceneT >= sc.dur) {
        goto(E.scene + 1, t);
        sc = L.scenes[E.scene];
      }
      for (let i = 0; i < L.scenes.length; i++) {
        const el = fills.current[i];
        if (el) el.style.transform = `scaleX(${i < E.scene ? 1 : i > E.scene ? 0 : Math.min(1, E.sceneT / (sc?.dur ?? 1))})`;
      }

      // reading: the next token's real features light up
      if (sc?.kind === "read" && L.frames.length) {
        const per = (sc.dur - 1600) / L.frames.length;
        const k = Math.min(L.frames.length - 1, Math.floor((E.sceneT - 600) / per));
        if (k >= 0 && k !== E.step) {
          E.step = k;
          E.tgt.fill(0);
          let top = 1e-6;
          for (const f of L.frames) for (const x of f.feats) top = Math.max(top, x.a);
          L.frames[k].feats.forEach((x, j) => {
            const i = E.index.get(x.id);
            if (i === undefined) return;
            E.tgt[i] = 0.25 + 0.75 * (x.a / top);
            if (j < 5) E.waves.push({ i, t0: t + j * 40, big: false });
          });
          setTok(L.frames[k].token);
        }
      }
      // steering: the forced-on feature keeps pulsing
      const tg = target(sc);
      if (sc?.kind === "steer" && tg.node >= 0 && E.sceneT > E.fly.dur) {
        const k = Math.floor((E.sceneT - E.fly.dur) / 1100);
        if (k !== E.step) {
          E.step = k;
          E.waves.push({ i: tg.node, t0: t, big: true });
        }
      }

      // camera: fly along an arc (pull back, then dive), then track the target as the field turns
      E.spin += reduce ? 0 : dt * (tg.node >= 0 ? 0.000018 : 0.00006);
      const [nx, ny] = tg.node >= 0 ? rot(tg.node) : [0, 0];
      const u = E.fly.dur <= 1 ? 1 : Math.min(1, (t - E.fly.t0) / E.fly.dur);
      const e = ease(u);
      E.camX = lerp(E.fly.x0, nx, e);
      E.camY = lerp(E.fly.y0, ny, e);
      E.logZ = lerp(E.fly.z0, tg.logZ, e) - E.fly.arc * Math.sin(Math.PI * u);
      E.tilt = lerp(E.fly.tilt0, tg.tilt, e);
      if (!reduce) {
        const k = 1 - Math.exp(-dt / 400);
        E.pmx += (E.mx - E.pmx) * k;
        E.pmy += (E.my - E.pmy) * k;
      }

      // firing: fast attack, slow afterglow
      const rise = 1 - Math.exp(-dt / 70);
      const fall = 1 - Math.exp(-dt / 300);
      for (let i = 0; i < E.n; i++) {
        const d = E.tgt[i] - E.act[i];
        if (d !== 0) {
          E.act[i] += d * (d > 0 ? rise : fall);
          if (Math.abs(d) < 0.002) E.act[i] = E.tgt[i];
        }
      }

      const W = E.w;
      const H = E.h;
      const narrow = W < 760;
      // where the camera's focus sits on screen: centred, nudged aside to leave room for the text a
      // chapter shows (keep in step with .callout, .runs and .sentence in tour.module.css)
      const kind = sc?.kind;
      const axT = !narrow && kind === "focus" ? 0.4 : 0.5;
      const ayT = narrow
        ? kind === "focus" ? 0.4 : kind === "steer" ? 0.33 : kind === "read" ? 0.47 : 0.56
        : kind === "steer" ? 0.42 : kind === "read" ? 0.47 : kind === "focus" ? 0.5 : 0.56;
      const glideK = 1 - Math.exp(-dt / 500);
      E.axK += (axT - E.axK) * glideK;
      E.ayK += (ayT - E.ayK) * glideK;
      const R = narrow ? Math.min(W * 0.46, H * 0.28) : Math.min(W * 0.34, H * 0.4);
      const tilt = E.tilt + E.pmy * 0.05;
      const K = {
        ax: E.axK * W + E.pmx * 16,
        ay: E.ayK * H + E.pmy * 10,
        R,
        Z: Math.exp(E.logZ),
        ct: Math.cos(tilt),
        st: Math.sin(tilt),
        cs: Math.cos(E.spin),
        sn: Math.sin(E.spin),
      };
      const pitch = (R * K.Z) / Math.max(1, E.rings);
      const zoomed = Math.max(0, Math.min(1, (K.Z - 1.4) / 3));
      const baseR = Math.max(narrow ? 0.72 : 1.05, Math.min(6.5, pitch * 0.26));
      const age = t - E.bornAt;
      const sweep = sc?.kind === "outro" ? (E.sceneT - 500) / 2000 : -1;
      const calm = !reduce && sc?.kind === "overview";

      ctx.setTransform(E.dpr, 0, 0, E.dpr, 0, 0);
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, W, H);
      // a soft light behind the field
      const bg = ctx.createRadialGradient(K.ax, K.ay, 0, K.ax, K.ay, R * 1.5);
      bg.addColorStop(0, "rgba(255,255,255,0.05)");
      bg.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, W, H);

      // dots, grouped by brightness so each group is one fill
      for (const b of buckets) b.length = 0;
      const lit: number[] = [];
      const RZ = K.R * K.Z;
      for (let i = 0; i < E.n; i++) {
        // turn the field, tilt it away from the viewer, and give near dots a little perspective
        const dx = E.x[i] * K.cs - E.y[i] * K.sn - E.camX;
        const dy = E.x[i] * K.sn + E.y[i] * K.cs - E.camY;
        const depth = Math.max(-0.9, Math.min(0.9, dy * K.st));
        const persp = 1 / (1 - depth * 0.42);
        const qx = K.ax + dx * RZ * persp;
        const qy = K.ay + dy * K.ct * RZ * persp;
        E.px[i] = qx;
        E.py[i] = qy;
        if (qx < -40 || qx > W + 40 || qy < -40 || qy > H + 40) continue;
        const born = reduce ? 1 : Math.max(0, Math.min(1, (age - (E.ring[i] / E.rings) * 1500) / 350));
        if (born <= 0) continue;
        let a = (0.36 + 0.55 * E.peak[i]) * (0.62 + 0.38 * ((depth + 1) / 2));
        // overview: a slow ripple runs out through the rings (decoration, not data)
        if (calm) a += 0.22 * Math.pow(Math.max(0, Math.sin(E.ring[i] * 0.22 - t * 0.0026)), 8);
        if (zoomed > 0) a *= 1 - zoomed * Math.min(0.75, Math.max(0, (Math.hypot(qx - K.ax, qy - K.ay) - 130) / 480));
        if (E.dead[i]) a *= 0.3;
        if (sweep >= 0) {
          const d = Math.abs(E.ring[i] / E.rings - sweep);
          if (d < 0.07) a += (1 - d / 0.07) * 0.55;
        }
        a = Math.min(1, a * born + E.act[i] * 0.9);
        const lvl = Math.min(15, Math.floor(a * 16));
        buckets[lvl].push(i);
        if (E.act[i] > 0.04) lit.push(i);
      }
      for (let lvl = 0; lvl < 16; lvl++) {
        const b = buckets[lvl];
        if (!b.length) continue;
        ctx.fillStyle = `rgba(255,255,255,${((lvl + 0.6) / 16).toFixed(3)})`;
        ctx.beginPath();
        for (const i of b) {
          const r = baseR * (1 + E.act[i] * 0.9);
          if (r < 1.3) ctx.rect(E.px[i] - r, E.py[i] - r, r * 2, r * 2);
          else {
            ctx.moveTo(E.px[i] + r, E.py[i]);
            ctx.arc(E.px[i], E.py[i], r, 0, TAU);
          }
        }
        ctx.fill();
      }

      // glow on firing features
      ctx.globalCompositeOperation = "lighter";
      for (const i of lit) {
        const gs = Math.max(18, baseR * 12) * (0.5 + E.act[i]);
        ctx.globalAlpha = Math.min(1, E.act[i] * 0.9);
        ctx.drawImage(sprite, E.px[i] - gs / 2, E.py[i] - gs / 2, gs, gs);
      }
      ctx.globalAlpha = 1;
      ctx.globalCompositeOperation = "source-over";

      // reading: links between the features that fired on this token, and the token itself
      if (sc?.kind === "read" && E.step >= 0 && L.frames[E.step]) {
        const ids = L.frames[E.step].feats.map((x) => E.index.get(x.id)).filter((i): i is number => i !== undefined);
        if (ids.length) {
          const a0 = ids[0];
          ctx.lineWidth = 1;
          for (const i of ids.slice(1, 14)) {
            ctx.strokeStyle = `rgba(255,255,255,${(0.08 + E.act[i] * 0.3).toFixed(3)})`;
            ctx.beginPath();
            ctx.moveTo(E.px[a0], E.py[a0]);
            const mx = (E.px[a0] + E.px[i]) / 2 + (E.py[i] - E.py[a0]) * 0.15;
            const my = (E.py[a0] + E.py[i]) / 2 - (E.px[i] - E.px[a0]) * 0.15;
            ctx.quadraticCurveTo(mx, my, E.px[i], E.py[i]);
            ctx.stroke();
          }
          const word = L.tokens[L.frames[E.step].token];
          if (word) chip(`“${showToken(word.text)}”`, E.px[a0], E.py[a0] - 28, true, 1);
        }
      }

      // shockwaves
      E.waves = E.waves.filter((wv) => t - wv.t0 < (wv.big ? 1500 : 950));
      for (const wv of E.waves) {
        const uu = (t - wv.t0) / (wv.big ? 1500 : 950);
        if (uu < 0) continue;
        const rings = wv.big ? 3 : 1;
        for (let k = 0; k < rings; k++) {
          const uk = Math.max(0, easeOut(uu) - k * 0.14);
          if (uk <= 0) continue;
          ctx.globalAlpha = (1 - uu) * (wv.big ? 0.75 : 0.45);
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = wv.big ? 1.5 : 1;
          ctx.beginPath();
          ctx.arc(E.px[wv.i], E.py[wv.i], Math.max(1, 4 + uk * (wv.big ? Math.max(60, pitch * 6) : Math.max(16, pitch * 2.2))), 0, TAU);
          ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;

      // the feature being visited: a reticle that closes in, its number, and the next words it pushes up
      E.focus = null;
      if (tg.node >= 0 && (sc?.kind === "focus" || sc?.kind === "steer")) {
        const fx = E.px[tg.node];
        const fy = E.py[tg.node];
        const k = easeOut((u - 0.72) / 0.28);
        if (k > 0) {
          const size = 22 + (1 - k) * 46;
          ctx.globalAlpha = k;
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 1.5;
          const arm = 9;
          ctx.beginPath();
          for (const [sx, sy] of [
            [-1, -1],
            [1, -1],
            [1, 1],
            [-1, 1],
          ]) {
            const cx = fx + sx * size;
            const cy = fy + sy * size;
            ctx.moveTo(cx, cy - sy * arm);
            ctx.lineTo(cx, cy);
            ctx.lineTo(cx - sx * arm, cy);
          }
          ctx.stroke();
          ctx.setLineDash([3, 5]);
          ctx.lineDashOffset = -t * 0.02;
          ctx.strokeStyle = "rgba(255,255,255,0.45)";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.arc(fx, fy, size * 0.72, 0, TAU);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.globalAlpha = 1;
          E.focus = { x: fx, y: fy, r: size + 6, id: sc.f.feature_id };

          if (sc.kind === "focus") {
            const next = sc.f.logits.promoted.slice(0, 5);
            const orbit = Math.max(narrow ? 92 : 104, size + (narrow ? 64 : 78));
            next.forEach((entry, j) => {
              const fade = Math.max(0, Math.min(1, (E.sceneT - E.fly.dur * 0.9 - j * 160) / 450));
              if (fade <= 0) return;
              const ang = -Math.PI / 2 + (j / next.length) * TAU + (reduce ? 0 : t * 0.00012);
              const ox = fx + Math.cos(ang) * orbit;
              const oy = fy + Math.sin(ang) * orbit * 0.8;
              ctx.globalAlpha = fade * 0.3;
              ctx.strokeStyle = "#fff";
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.moveTo(fx + Math.cos(ang) * (size + 4), fy + Math.sin(ang) * (size + 4) * 0.8);
              ctx.lineTo(ox, oy);
              ctx.stroke();
              chip(showToken(entry.token), ox, oy, false, fade);
            });
            ctx.globalAlpha = 1;
          }
          chip(sc.kind === "steer" ? `#${sc.f.feature_id} · ${sc.ex.mode === "amplify" ? "forced on" : "switched off"}` : `#${sc.f.feature_id}`, fx, fy + size + 18, false, k);
        }
      }
    };

    function chip(text: string, x: number, y: number, strong: boolean, alpha: number) {
      ctx.globalAlpha = alpha;
      ctx.font = `${strong ? 600 : 500} ${strong ? 13 : 12}px ${font}`;
      ctx.textBaseline = "middle";
      const w = ctx.measureText(text).width + 16;
      const hgt = strong ? 26 : 22;
      ctx.fillStyle = strong ? "#fff" : "rgba(0,0,0,0.9)";
      ctx.strokeStyle = strong ? "#fff" : "rgba(255,255,255,0.45)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(x - w / 2, y - hgt / 2, w, hgt, hgt / 2);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = strong ? "#000" : "#fff";
      ctx.textAlign = "center";
      ctx.fillText(text, x, y + 0.5);
      ctx.textAlign = "left";
      ctx.globalAlpha = 1;
    }

    raf = requestAnimationFrame(frame);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      window.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("click", onClick);
      ctrl.current = null;
    };
    // the engine reads changing data through `live`; it starts once
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const sc = scenes[Math.min(at, scenes.length - 1)];
  const total = h ? fmtInt(h.d_sae) : "";
  const k = `${at}-${sc?.kind}`;
  const tryIt = `/inspect?text=${encodeURIComponent(tokens.map((t) => t.text).join(""))}`;
  // the header (and its banner) steps aside during the tour, so the tour carries the mode line itself
  const mode = offline ? copy("banner.offline", { url: API_BASE }) : h?.is_synthetic ? copy("banner.synthetic") : h?.mode === "mock" ? copy("banner.mock") : null;

  return (
    <div className={s.tour} data-scene={sc?.kind} role="region" aria-label="Guided tour of the features inside GPT-2">
      <h1 className="visually-hidden">NeuronLens: put GPT-2 under a lens</h1>
      <canvas ref={canvasRef} className={s.canvas} aria-hidden="true" />
      <div className={s.shadeEdges} aria-hidden="true" />
      <div className={s.shadeCenter} data-on={sc?.kind === "overview" || sc?.kind === "outro" ? "1" : "0"} aria-hidden="true" />
      {mode && (
        <span className={s.mode} role="status">
          <i aria-hidden="true" />
          {mode}
        </span>
      )}

      {/* opening and closing: big text in the middle of the field */}
      {sc?.kind === "overview" && (
        <div className={s.center} key={`c-${k}`}>
          <span className={s.eyebrow}>{h ? `NeuronLens · ${h.model_name} · layer ${h.hook_layer}` : "NeuronLens"}</span>
          <p className={s.hero}>
            Put GPT-2 <span className={s.dim}>under a lens.</span>
          </p>
          <p className={s.sub}>{h ? `${total} features. Each dot is one the model learned.` : "Waiting for the server…"}</p>
          <div className={s.ctas}>
            <Link href="/microscope" className="btn btn-lg">
              Open microscope <span aria-hidden="true">→</span>
            </Link>
            <Link href={tokens.length ? tryIt : "/inspect"} className="btn btn-ghost btn-lg">
              Try a sentence
            </Link>
          </div>
        </div>
      )}
      {sc?.kind === "outro" && (
        <div className={s.center} key={`c-${k}`}>
          <span className={s.eyebrow}>Quality check</span>
          {ceRec !== null ? (
            <>
              <p className={`${s.hero} ${s.big}`}>{(ceRec * 100).toFixed(1)}%</p>
              <p className={s.sub}>
                of GPT-2&apos;s next-word ability is kept when our {total} features replace layer {h?.hook_layer}.
              </p>
            </>
          ) : (
            <p className={s.hero}>Look closer.</p>
          )}
          <div className={s.ctas}>
            <Link href="/microscope" className="btn btn-lg">
              Explore every feature <span aria-hidden="true">→</span>
            </Link>
            <Link href="/metrics" className="btn btn-ghost btn-lg">
              See the metrics
            </Link>
          </div>
        </div>
      )}

      {/* chapter title: top centre, only while a chapter needs one */}
      {(sc?.kind === "read" || sc?.kind === "focus" || sc?.kind === "steer") && (
        <div className={s.head} key={`h-${k}`}>
          {sc.kind === "read" && (
            <>
              <span className={s.eyebrow}>Reading a sentence</span>
              <p className={s.title}>{h ? `Every word fires ${h.k} of them.` : "Every word fires a few."}</p>
            </>
          )}
          {sc.kind === "focus" && (
            <>
              <span className={s.eyebrow}>Feature #{sc.f.feature_id}</span>
              <p className={`${s.title} ${sc.f.label!.length > 30 ? s.titleLong : ""}`}>{sc.f.label}</p>
            </>
          )}
          {sc.kind === "steer" && (
            <>
              <span className={s.eyebrow}>
                Steering · #{sc.f.feature_id} {sc.ex.mode === "amplify" ? "forced on" : "switched off"}
              </span>
              <p className={s.title}>{sc.ex.mode === "amplify" ? "Turn it up. The text changes." : "Switch it off. The text changes."}</p>
            </>
          )}
        </div>
      )}

      {/* the visited feature's evidence, beside it (below it on phones) */}
      {sc?.kind === "focus" &&
        (() => {
          const sn = snippet(sc.f);
          return (
            <div className={s.callout} key={`d-${k}`}>
              <span className={s.cardLabel}>Top example</span>
              <p className={s.example}>
                …{sn.before}
                <mark>{sn.peak}</mark>
                {sn.after}…
              </p>
              <span className={s.cardLabel}>Pushes up next</span>
              <div className={s.chips}>
                {sc.f.logits.promoted.slice(0, 5).map((l, i) => (
                  <span key={i}>{showToken(l.token)}</span>
                ))}
              </div>
              <Link href={`/features/${sc.f.feature_id}`} className={s.more}>
                Open feature <span aria-hidden="true">→</span>
              </Link>
            </div>
          );
        })()}

      <div className={s.dock}>
        {/* reading: the sentence, word by word */}
        {sc?.kind === "read" && (
          <p className={s.sentence} key={`s-${k}`} aria-label={tokens.map((t) => t.text).join("")}>
            {tokens.map((t) => (
              <span key={t.index} className={t.index === tok ? s.now : tok >= 0 && t.index < tok ? s.done : s.next}>
                {showToken(t.text)}
              </span>
            ))}
          </p>
        )}

        {/* steering: normal next to steered, live on the model in real mode */}
        {sc?.kind === "steer" &&
          (() => {
            const base = continuation(sc.ex.baseline_text, sc.ex.prompt);
            const steered = continuation(sc.ex.steered_text, sc.ex.prompt);
            return (
              <div className={s.runs} key={`r-${k}`}>
                <span className={s.cardLabel}>
                  “{sc.ex.prompt}…” · same prompt, seed {sc.ex.seed} · {sc.ex.live ? "live on the model" : "saved example"}
                </span>
                <div className={s.runGrid}>
                  <div className={s.run}>
                    <span className={s.runTag}>Normal</span>
                    <p>{base}</p>
                  </div>
                  <div className={`${s.run} ${s.runOn}`}>
                    <span className={s.runTag}>{sc.ex.mode === "amplify" ? "Feature on" : "Feature off"}</span>
                    <p className={s.wipe}>
                      {diffWords(base, steered).map((seg, i) =>
                        seg.changed ? <mark key={i}>{seg.text}</mark> : <span key={i}>{seg.text}</span>,
                      )}
                    </p>
                  </div>
                </div>
              </div>
            );
          })()}

        {/* chapters: progress, jump, pause */}
        <div className={s.bar}>
          <button type="button" className={s.ctrlBtn} onClick={() => setPlaying((p) => !p)} aria-label={playing ? "Pause tour" : "Play tour"} aria-pressed={!playing}>
            {playing ? (
              <svg width="10" height="12" viewBox="0 0 10 12" aria-hidden="true">
                <rect x="0" y="0" width="3" height="12" fill="currentColor" />
                <rect x="7" y="0" width="3" height="12" fill="currentColor" />
              </svg>
            ) : (
              <svg width="10" height="12" viewBox="0 0 10 12" aria-hidden="true">
                <path d="M0 0 L10 6 L0 12 Z" fill="currentColor" />
              </svg>
            )}
          </button>
          <ol className={s.chapters}>
            {scenes.map((c, i) => (
              <li key={`${c.kind}-${c.chapter}`}>
                <button
                  type="button"
                  onClick={() => {
                    ctrl.current?.goto(i);
                    setPlaying(true);
                  }}
                  aria-current={i === at ? "step" : undefined}
                  aria-label={`Go to ${c.chapter}`}
                >
                  <span className={s.track}>
                    <i
                      ref={(el) => {
                        fills.current[i] = el;
                      }}
                    />
                  </span>
                  <span className={s.chLabel}>{c.chapter}</span>
                </button>
              </li>
            ))}
          </ol>
          <span className={s.count} aria-hidden="true">
            {pad(at + 1)}/{pad(scenes.length)}
          </span>
        </div>
      </div>
    </div>
  );
}
