"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef } from "react";
import s from "./landing.module.css";

type Named = { id: number; label: string };

/** Every feature as one dot, laid out row by row by id. A lens follows the cursor and magnifies
 *  what is under it, and 32 dots fire at a time (k = 32). Click a dot in the lens to open it. */
export function LensField({ total, named }: { total: number; named: Named[] }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const router = useRouter();
  const labels = useRef(new Map<number, string>());
  const nav = useRef(router);

  useEffect(() => {
    labels.current = new Map(named.map((n) => [n.id, n.label]));
  }, [named]);
  useEffect(() => {
    nav.current = router;
  }, [router]);

  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext("2d")!;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const N = Math.max(1, total);
    const font = `500 11px ${getComputedStyle(document.documentElement).getPropertyValue("--font-mono").trim() || "monospace"}`;

    let w = 0;
    let h = 0;
    let dpr = 1;
    let cell = 10;
    let cols = 1;
    let rows = 1;
    let ox = 0;
    let oy = 0;
    let R = 140; // lens radius
    const M = 3.2; // magnification
    let base: HTMLCanvasElement | null = null;

    // soft glow sprite, drawn behind firing dots
    const glow = document.createElement("canvas");
    glow.width = glow.height = 64;
    {
      const g = glow.getContext("2d")!;
      const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
      grad.addColorStop(0, "rgba(255,255,255,0.9)");
      grad.addColorStop(0.25, "rgba(255,255,255,0.35)");
      grad.addColorStop(1, "rgba(255,255,255,0)");
      g.fillStyle = grad;
      g.fillRect(0, 0, 64, 64);
    }

    const pos = (id: number): [number, number] => [ox + (id % cols) * cell + cell / 2, oy + Math.floor(id / cols) * cell + cell / 2];

    function size() {
      const r = canvas.getBoundingClientRect();
      w = r.width;
      h = r.height;
      dpr = Math.min(2, window.devicePixelRatio || 1);
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      cell = Math.sqrt((w * h) / N);
      cols = Math.max(1, Math.floor(w / cell));
      rows = Math.ceil(N / cols);
      cell = Math.min(w / cols, h / rows);
      ox = (w - cols * cell) / 2;
      oy = (h - rows * cell) / 2;
      R = Math.max(86, Math.min(170, Math.min(w, h) * 0.22));

      // the resting field, drawn once: dim dots, fading toward the edges
      base = document.createElement("canvas");
      base.width = canvas.width;
      base.height = canvas.height;
      const b = base.getContext("2d")!;
      b.setTransform(dpr, 0, 0, dpr, 0, 0);
      const dr = Math.max(0.55, cell * 0.09);
      const cx = w / 2;
      const cy = h / 2;
      const far = Math.hypot(cx, cy);
      for (let id = 0; id < N; id++) {
        const [x, y] = pos(id);
        const f = 1 - Math.hypot(x - cx, y - cy) / far;
        b.fillStyle = `rgba(255,255,255,${(0.07 + 0.2 * f * f).toFixed(3)})`;
        b.fillRect(x - dr, y - dr, dr * 2, dr * 2);
      }
    }
    size();
    const ro = new ResizeObserver(size);
    ro.observe(canvas);

    // firing: every beat, 32 features light up and fade
    type Lit = { id: number; t0: number; amp: number };
    let lit: Lit[] = [];
    let lastBeat = -1e9;
    function beat(t: number) {
      const named = [...labels.current.keys()];
      for (let i = 0; i < 32; i++) {
        const pick = named.length && Math.random() < 0.06 ? named[Math.floor(Math.random() * named.length)] : Math.floor(Math.random() * N);
        lit.push({ id: pick, t0: t + i * 18, amp: 0.45 + Math.random() * 0.55 });
      }
    }
    const env = (u: number) => (u < 0 ? 0 : u < 0.12 ? u / 0.12 : Math.exp(-(u - 0.12) * 3.2));

    // lens follows the cursor, or drifts on its own
    const mouse = { x: 0, y: 0, inside: false, last: -1e9 };
    let lx = -1;
    let ly = -1;
    const onMove = (e: PointerEvent) => {
      const r = canvas.getBoundingClientRect();
      mouse.x = e.clientX - r.left;
      mouse.y = e.clientY - r.top;
      mouse.inside = mouse.x >= 0 && mouse.y >= 0 && mouse.x <= r.width && mouse.y <= r.height && e.pointerType === "mouse";
      if (mouse.inside) mouse.last = performance.now();
    };
    window.addEventListener("pointermove", onMove, { passive: true });

    // what was drawn inside the lens last frame, for hover and click
    let drawn: { x: number; y: number; id: number }[] = [];
    let hovered = -1;
    function pick(x: number, y: number) {
      if (Math.hypot(x - lx, y - ly) > R) return -1;
      let best = -1;
      let bd = (cell * M) / 2;
      for (const d of drawn) {
        const dd = Math.hypot(d.x - x, d.y - y);
        if (dd < bd) {
          bd = dd;
          best = d.id;
        }
      }
      return best;
    }
    const onClick = (e: MouseEvent) => {
      const r = canvas.getBoundingClientRect();
      const id = pick(e.clientX - r.left, e.clientY - r.top);
      if (id >= 0) nav.current.push(`/features/${id}`);
    };
    canvas.addEventListener("click", onClick);

    // pause when off screen
    let visible = true;
    const io = new IntersectionObserver(([e]) => (visible = e.isIntersecting));
    io.observe(canvas);

    const t0 = performance.now();
    let prev = t0;
    let raf = 0;
    const frame = (t: number) => {
      raf = requestAnimationFrame(frame);
      if (!visible || !base) return;
      const dt = Math.min(64, t - prev);
      prev = t;
      const age = t - t0;
      const intro = reduce ? 1 : Math.min(1, age / 1500);
      const ease = 1 - Math.pow(1 - intro, 3);

      // lens target
      const narrow = w < 760;
      let tx: number;
      let ty: number;
      if (mouse.inside && t - mouse.last < 4000) {
        tx = mouse.x;
        ty = mouse.y;
      } else {
        const k = reduce ? 0 : age / 1000;
        tx = (narrow ? 0.5 : 0.7) * w + Math.sin(k * 0.31) * w * (narrow ? 0.22 : 0.14);
        ty = (narrow ? 0.7 : 0.5) * h + Math.sin(k * 0.47 + 1) * h * (narrow ? 0.12 : 0.2);
      }
      if (lx < 0) {
        lx = tx;
        ly = ty;
      }
      const follow = 1 - Math.exp(-dt / 90);
      lx += (tx - lx) * follow;
      ly += (ty - ly) * follow;

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      // field, revealed from the centre on load
      const reach = Math.hypot(w, h) * 0.6 * ease;
      ctx.save();
      if (intro < 1) {
        ctx.beginPath();
        ctx.arc(w / 2, h / 2, reach, 0, Math.PI * 2);
        ctx.clip();
      }
      ctx.drawImage(base, 0, 0, w, h);
      ctx.restore();
      if (intro < 1) {
        ctx.strokeStyle = `rgba(255,255,255,${(0.35 * (1 - intro)).toFixed(3)})`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(w / 2, h / 2, reach, 0, Math.PI * 2);
        ctx.stroke();
      }

      // firing dots
      if (!reduce && t - lastBeat > 760) {
        lastBeat = t;
        beat(t);
      }
      if (reduce && lit.length === 0) beat(t - 300);
      const bright = new Map<number, number>();
      lit = lit.filter((l) => t - l.t0 < 1500 || reduce);
      for (const l of lit) {
        const v = reduce ? l.amp * 0.8 : l.amp * env((t - l.t0) / 1300);
        if (v <= 0.01) continue;
        bright.set(l.id, Math.max(bright.get(l.id) ?? 0, v));
      }
      for (const [id, v] of bright) {
        const [x, y] = pos(id);
        if (Math.hypot(x - w / 2, y - h / 2) > reach) continue;
        const gs = cell * (2.2 + v * 2.4);
        ctx.globalAlpha = v * 0.9;
        ctx.drawImage(glow, x - gs / 2, y - gs / 2, gs, gs);
        ctx.globalAlpha = 1;
        ctx.fillStyle = `rgba(255,255,255,${(0.4 + v * 0.6).toFixed(3)})`;
        const r = Math.max(0.9, cell * 0.14) * (1 + v);
        ctx.fillRect(x - r, y - r, r * 2, r * 2);
      }

      // the lens
      const la = reduce ? 1 : Math.min(1, Math.max(0, (age - 500) / 700));
      if (la > 0) {
        ctx.globalAlpha = la;
        ctx.save();
        ctx.beginPath();
        ctx.arc(lx, ly, R, 0, Math.PI * 2);
        ctx.clip();
        ctx.fillStyle = "#000";
        ctx.fillRect(lx - R, ly - R, R * 2, R * 2);

        // magnified grid with a little barrel distortion at the rim
        const src = (R / M) * 1.18;
        const c0 = Math.max(0, Math.floor((lx - src - ox) / cell));
        const c1 = Math.min(cols - 1, Math.ceil((lx + src - ox) / cell));
        const r0 = Math.max(0, Math.floor((ly - src - oy) / cell));
        const r1 = Math.min(rows - 1, Math.ceil((ly + src - oy) / cell));
        const dotR = Math.max(1.3, cell * 0.11 * M);
        drawn = [];
        const labelled: { x: number; y: number; id: number; text: string }[] = [];
        for (let rr = r0; rr <= r1; rr++) {
          for (let cc = c0; cc <= c1; cc++) {
            const id = rr * cols + cc;
            if (id >= N) continue;
            const sx = ox + cc * cell + cell / 2;
            const sy = oy + rr * cell + cell / 2;
            const d = Math.hypot(sx - lx, sy - ly);
            const u = d / (R / M);
            const f = M * (1 - 0.1 * u * u);
            const px = lx + (sx - lx) * f;
            const py = ly + (sy - ly) * f;
            const pd = Math.hypot(px - lx, py - ly);
            if (pd > R + dotR) continue;
            const edge = Math.max(0, 1 - Math.pow(pd / R, 4));
            const v = bright.get(id) ?? 0;
            const name = labels.current.get(id);
            if (v > 0.02) {
              const gs = dotR * (6 + v * 6);
              ctx.globalAlpha = la * v;
              ctx.drawImage(glow, px - gs / 2, py - gs / 2, gs, gs);
              ctx.globalAlpha = la;
            }
            ctx.fillStyle = `rgba(255,255,255,${((0.26 + v * 0.74) * (0.35 + 0.65 * edge)).toFixed(3)})`;
            ctx.beginPath();
            ctx.arc(px, py, dotR * (1 + v * 0.8), 0, Math.PI * 2);
            ctx.fill();
            if (name) {
              ctx.strokeStyle = `rgba(255,255,255,${(0.8 * edge).toFixed(3)})`;
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.arc(px, py, dotR + 4, 0, Math.PI * 2);
              ctx.stroke();
              if (pd < R * 0.78) labelled.push({ x: px, y: py, id, text: name });
            }
            drawn.push({ x: px, y: py, id });
          }
        }

        // hovered dot: a ring and a crosshair
        hovered = mouse.inside ? pick(mouse.x, mouse.y) : -1;
        canvas.style.cursor = hovered >= 0 ? "pointer" : "default";
        const focus = hovered >= 0 ? drawn.find((d) => d.id === hovered) : null;
        if (focus) {
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(focus.x, focus.y, dotR + 7, 0, Math.PI * 2);
          ctx.stroke();
        }
        ctx.strokeStyle = "rgba(255,255,255,0.1)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(lx - R, ly);
        ctx.lineTo(lx + R, ly);
        ctx.moveTo(lx, ly - R);
        ctx.lineTo(lx, ly + R);
        ctx.stroke();

        // labels for named features
        ctx.font = font;
        ctx.textBaseline = "middle";
        for (const l of labelled.slice(0, 3)) {
          const text = l.text.length > 26 ? l.text.slice(0, 25) + "…" : l.text;
          const tw = ctx.measureText(text).width;
          // labels open toward the centre of the lens so the rim never cuts them
          const bx = l.x > lx ? l.x - dotR - 9 - tw : l.x + dotR + 9;
          ctx.fillStyle = "rgba(0,0,0,0.85)";
          ctx.fillRect(bx - 4, l.y - 9, tw + 8, 18);
          ctx.fillStyle = "#fff";
          ctx.fillText(text, bx, l.y);
        }
        ctx.restore();

        // glass: rim, ticks, a highlight
        ctx.strokeStyle = "rgba(255,255,255,0.9)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(lx, ly, R, 0, Math.PI * 2);
        ctx.stroke();
        ctx.strokeStyle = "rgba(255,255,255,0.14)";
        ctx.lineWidth = 6;
        ctx.beginPath();
        ctx.arc(lx, ly, R - 7, Math.PI * 1.08, Math.PI * 1.42);
        ctx.stroke();
        const spin = reduce ? 0 : age * 0.00008;
        ctx.lineWidth = 1;
        for (let i = 0; i < 72; i++) {
          const a = spin + (i / 72) * Math.PI * 2;
          const long = i % 6 === 0;
          const r1i = R + 5;
          const r2i = R + (long ? 13 : 9);
          ctx.strokeStyle = long ? "rgba(255,255,255,0.55)" : "rgba(255,255,255,0.2)";
          ctx.beginPath();
          ctx.moveTo(lx + Math.cos(a) * r1i, ly + Math.sin(a) * r1i);
          ctx.lineTo(lx + Math.cos(a) * r2i, ly + Math.sin(a) * r2i);
          ctx.stroke();
        }

        // readout under the lens: the feature in the centre, or the one under the cursor
        const centre = (() => {
          const cc = Math.floor((lx - ox) / cell);
          const rr = Math.floor((ly - oy) / cell);
          const id = rr * cols + cc;
          return cc >= 0 && cc < cols && rr >= 0 && id < N ? id : -1;
        })();
        const show = hovered >= 0 ? hovered : centre;
        if (show >= 0) {
          const name = labels.current.get(show);
          const text = hovered >= 0 ? `#${show}  ${name ?? "unlabelled"}  ·  open →` : `#${show}${name ? "  " + name : ""}`;
          const clipped = text.length > 46 ? text.slice(0, 45) + "…" : text;
          ctx.font = font;
          ctx.textBaseline = "middle";
          const tw = ctx.measureText(clipped).width;
          const bw = tw + 22;
          const bx = Math.max(6, Math.min(w - bw - 6, lx - bw / 2));
          const by = Math.min(h - 32, ly + R + 22);
          ctx.fillStyle = "#000";
          ctx.strokeStyle = hovered >= 0 ? "rgba(255,255,255,0.8)" : "rgba(255,255,255,0.25)";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.roundRect(bx, by, bw, 24, 12);
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = hovered >= 0 ? "#fff" : "rgba(255,255,255,0.7)";
          ctx.fillText(clipped, bx + 11, by + 12.5);
        }
        ctx.globalAlpha = 1;
      }
    };
    raf = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
      window.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("click", onClick);
    };
  }, [total]);

  return <canvas ref={ref} className={s.field} aria-label={`A field of ${total.toLocaleString("en-US")} dots, one per feature, under a moving lens`} role="img" />;
}
