"use client";

import { useEffect, useRef } from "react";
import s from "./landing.module.css";

/** Drifting nodes that link up when close, react to the cursor, and pass signals along links. */
export function NetworkBg() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext("2d")!;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let w = 0;
    let h = 0;
    let dpr = 1;
    type Node = { x: number; y: number; vx: number; vy: number; r: number };
    let nodes: Node[] = [];
    const mouse = { x: -9999, y: -9999 };
    const pulses: { a: number; b: number; t0: number }[] = [];

    function size() {
      const rect = canvas.getBoundingClientRect();
      dpr = Math.min(2, window.devicePixelRatio || 1);
      w = rect.width;
      h = rect.height;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      const count = Math.round(Math.min(110, (w * h) / 14000));
      nodes = Array.from({ length: count }, () => ({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.25,
        vy: (Math.random() - 0.5) * 0.25,
        r: 0.8 + Math.random() * 1.6,
      }));
    }
    size();
    const ro = new ResizeObserver(size);
    ro.observe(canvas);

    const onMove = (e: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      mouse.x = e.clientX - rect.left;
      mouse.y = e.clientY - rect.top;
    };
    const onLeave = () => {
      mouse.x = mouse.y = -9999;
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerleave", onLeave);

    let raf = 0;
    let lastPulse = 0;
    const LINK = 150;
    const frame = (t: number) => {
      raf = requestAnimationFrame(frame);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      for (const n of nodes) {
        if (!reduce) {
          n.x += n.vx;
          n.y += n.vy;
        }
        if (n.x < -20) n.x = w + 20;
        if (n.x > w + 20) n.x = -20;
        if (n.y < -20) n.y = h + 20;
        if (n.y > h + 20) n.y = -20;
        // gentle pull toward the cursor
        const dx = mouse.x - n.x;
        const dy = mouse.y - n.y;
        const d2 = dx * dx + dy * dy;
        if (d2 < 220 * 220) {
          n.x += dx * 0.0025;
          n.y += dy * 0.0025;
        }
      }
      // links
      ctx.lineWidth = 1;
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i];
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const d = Math.hypot(dx, dy);
          if (d > LINK) continue;
          const near = Math.min(1, 200 / (Math.hypot(mouse.x - (a.x + b.x) / 2, mouse.y - (a.y + b.y) / 2) + 1));
          ctx.strokeStyle = `rgba(255,255,255,${((1 - d / LINK) * (0.1 + near * 0.35)).toFixed(3)})`;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
      // cursor links
      for (const n of nodes) {
        const d = Math.hypot(mouse.x - n.x, mouse.y - n.y);
        if (d < 180) {
          ctx.strokeStyle = `rgba(255,255,255,${(0.5 * (1 - d / 180)).toFixed(3)})`;
          ctx.beginPath();
          ctx.moveTo(mouse.x, mouse.y);
          ctx.lineTo(n.x, n.y);
          ctx.stroke();
        }
      }
      // nodes
      for (const n of nodes) {
        const d = Math.hypot(mouse.x - n.x, mouse.y - n.y);
        const lit = d < 180 ? 1 - d / 180 : 0;
        ctx.fillStyle = `rgba(255,255,255,${(0.35 + lit * 0.65).toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r + lit * 1.8, 0, Math.PI * 2);
        ctx.fill();
      }
      // signals hopping along links
      if (!reduce && t - lastPulse > 260 && nodes.length > 2) {
        lastPulse = t;
        const a = Math.floor(Math.random() * nodes.length);
        let best = -1;
        let bd = LINK;
        for (let j = 0; j < nodes.length; j++) {
          if (j === a) continue;
          const d = Math.hypot(nodes[a].x - nodes[j].x, nodes[a].y - nodes[j].y);
          if (d < bd) {
            bd = d;
            best = j;
          }
        }
        if (best >= 0) pulses.push({ a, b: best, t0: t });
      }
      for (let k = pulses.length - 1; k >= 0; k--) {
        const p = pulses[k];
        const u = (t - p.t0) / 900;
        if (u >= 1) {
          pulses.splice(k, 1);
          continue;
        }
        const A = nodes[p.a];
        const B = nodes[p.b];
        if (!A || !B) continue;
        const x = A.x + (B.x - A.x) * u;
        const y = A.y + (B.y - A.y) * u;
        const g = ctx.createRadialGradient(x, y, 0, x, y, 10);
        g.addColorStop(0, "rgba(255,255,255,0.95)");
        g.addColorStop(1, "rgba(255,255,255,0)");
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(x, y, 10, 0, Math.PI * 2);
        ctx.fill();
      }
    };
    raf = requestAnimationFrame(frame);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerleave", onLeave);
    };
  }, []);

  return <canvas ref={ref} className={s.network} aria-hidden="true" />;
}
