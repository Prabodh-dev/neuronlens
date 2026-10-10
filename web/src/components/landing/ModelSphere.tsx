"use client";

import { useEffect, useRef } from "react";

/** A slowly turning sphere of dots, one per model in the model selector. It is a picture of "a model's
 *  feature space", the same dots-on-black look as the landing page's neuron art; it carries no data, so
 *  both models get the identical sphere, and the experimental one only gets a dashed ring. It stands
 *  still when the visitor prefers reduced motion, and stops while off screen or in a hidden tab. */
const DOTS = 360;
const TILT = 0.38; // radians the axis leans toward the viewer

// points spread evenly over a unit sphere (golden-angle spiral)
const POINTS: [number, number, number][] = Array.from({ length: DOTS }, (_, i) => {
  const y = 1 - (2 * (i + 0.5)) / DOTS;
  const r = Math.sqrt(1 - y * y);
  const a = i * Math.PI * (3 - Math.sqrt(5));
  return [Math.cos(a) * r, y, Math.sin(a) * r];
});

export function ModelSphere({ dashed = false, fast = false }: { dashed?: boolean; fast?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const speed = useRef(fast);
  speed.current = fast;

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let angle = 0.6;
    let raf = 0;
    let visible = true;
    let last = performance.now();

    const draw = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = canvas.clientWidth;
      if (!w) return;
      if (canvas.width !== Math.round(w * dpr)) {
        canvas.width = canvas.height = Math.round(w * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, w);
      const R = w * 0.38;
      const c = w / 2;
      const ca = Math.cos(angle);
      const sa = Math.sin(angle);
      const ct = Math.cos(TILT);
      const st = Math.sin(TILT);
      for (const [x, y, z] of POINTS) {
        // turn about the vertical axis, then lean the axis toward the viewer
        const x1 = x * ca + z * sa;
        const z1 = -x * sa + z * ca;
        const y2 = y * ct - z1 * st;
        const z2 = y * st + z1 * ct;
        const depth = (z2 + 1) / 2; // 0 far side, 1 near side
        ctx.globalAlpha = 0.12 + 0.88 * depth ** 1.6;
        ctx.fillStyle = "#fff";
        ctx.beginPath();
        ctx.arc(c + x1 * R, c + y2 * R, 0.7 + 1.9 * depth, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 0.28;
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1;
      ctx.setLineDash(dashed ? [4, 6] : []);
      ctx.beginPath();
      ctx.ellipse(c, c, R * 1.28, R * 0.42, -0.35, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    };

    const tick = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      if (visible && !document.hidden) {
        angle += dt * (speed.current ? 0.9 : 0.32);
        draw();
      }
      raf = requestAnimationFrame(tick);
    };

    draw();
    const io = new IntersectionObserver(([e]) => (visible = e.isIntersecting));
    io.observe(canvas);
    if (!still) raf = requestAnimationFrame(tick);
    const onResize = () => draw();
    window.addEventListener("resize", onResize);
    return () => {
      cancelAnimationFrame(raf);
      io.disconnect();
      window.removeEventListener("resize", onResize);
    };
  }, [dashed]);

  return <canvas ref={ref} aria-hidden="true" style={{ display: "block", width: "100%", aspectRatio: "1" }} />;
}
