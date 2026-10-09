"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { fmtInt } from "@/lib/format";
import { isMockModel, isSampleData, useHealth } from "./HealthProvider";
import { Logo } from "./Logo";
import { ModeBanner } from "./ModeBanner";
import s from "./TopBar.module.css";

const NAV = [
  { href: "/microscope", label: "Microscope" },
  { href: "/features", label: "Features" },
  { href: "/inspect", label: "Inspect" },
  { href: "/steer", label: "Steer" },
  { href: "/autolab", label: "Auto-Lab" },
  { href: "/metrics", label: "Metrics" },
];

export function TopBar() {
  const path = usePathname();
  const { data: h, error } = useHealth();
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  // on the landing page the bar steps aside while the tour plays, and slides in once you scroll,
  // point at the top edge, or tab into it
  const immersive = path === "/";
  const [nearTop, setNearTop] = useState(false);
  const [glide, setGlide] = useState<{ x: number; w: number; on: boolean }>({ x: 0, w: 0, on: false });
  const headRef = useRef<HTMLElement>(null);
  const progRef = useRef<HTMLSpanElement>(null);

  // solid bar: track scroll for the hairline and the progress line
  useEffect(() => {
    const on = () => {
      const y = window.scrollY;
      setScrolled(y > 24);
      const max = document.documentElement.scrollHeight - window.innerHeight;
      progRef.current?.style.setProperty("--p", max > 0 ? Math.min(1, y / max).toFixed(4) : "0");
    };
    on();
    window.addEventListener("scroll", on, { passive: true });
    window.addEventListener("resize", on);
    return () => {
      window.removeEventListener("scroll", on);
      window.removeEventListener("resize", on);
    };
  }, [path]);

  // sticky panels on other pages sit below the header, whatever its height (banner or not)
  useEffect(() => {
    const el = headRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => document.documentElement.style.setProperty("--head-h", `${el.offsetHeight}px`));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => setOpen(false), [path]);

  useEffect(() => {
    if (!immersive) return;
    const onMove = (e: PointerEvent) => {
      if (e.pointerType !== "mouse") return;
      setNearTop((was) => (was ? e.clientY < 140 : e.clientY < 56));
    };
    window.addEventListener("pointermove", onMove, { passive: true });
    return () => window.removeEventListener("pointermove", onMove);
  }, [immersive]);

  function hover(e: React.PointerEvent<HTMLAnchorElement>) {
    const a = e.currentTarget;
    setGlide({ x: a.offsetLeft, w: a.offsetWidth, on: true });
  }

  const sim = h ? isSampleData(h) || isMockModel(h) : false;

  return (
    <header
      ref={headRef}
      className={`${s.bar} ${scrolled ? s.scrolled : ""} ${immersive ? s.immersive : ""} ${immersive && !scrolled && !nearTop && !open ? s.away : ""}`}
    >
      <a href="#main" className={s.skip}>
        Skip to content
      </a>
      <div className={s.inner}>
        <Link href="/" className={s.brand} aria-label="NeuronLens home">
          <span className={s.logo}>
            <Logo size={24} />
          </span>
          <span className={s.word}>NeuronLens</span>
        </Link>

        <nav className={`${s.nav} ${open ? s.navOpen : ""}`} aria-label="Main" onPointerLeave={() => setGlide((g) => ({ ...g, on: false }))}>
          <span className={s.glide} style={{ transform: `translateX(${glide.x}px)`, width: glide.w, opacity: glide.on ? 1 : 0 }} aria-hidden="true" />
          {NAV.map((n) => {
            const on = path === n.href || path.startsWith(n.href + "/");
            return (
              <Link key={n.href} href={n.href} className={s.link} aria-current={on ? "page" : undefined} onPointerEnter={hover}>
                {n.label}
              </Link>
            );
          })}
        </nav>

        <div className={s.right}>
          {/* one pill: server status, then the model it serves */}
          {(h || error) && (
            <span
              className={s.status}
              data-kind={error ? "off" : sim ? "sim" : "live"}
              tabIndex={h ? 0 : undefined}
              aria-label={h ? `${isSampleData(h) ? "Sample data" : isMockModel(h) ? "Mock" : "Live"}: ${h.model_name}, layer ${h.hook_layer}, ${fmtInt(h.d_sae)} features, k ${h.k}` : undefined}
            >
              <i aria-hidden="true" />
              <b>{error ? "Offline" : isSampleData(h!) ? "Sample data" : isMockModel(h!) ? "Mock" : "Live"}</b>
              {h && (
                <span className={s.chip} aria-hidden="true">
                  {h.model_name} · L{h.hook_layer} · {fmtInt(h.d_sae)} · k={h.k}
                </span>
              )}
            </span>
          )}
          <button className={s.menu} aria-label="Menu" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
            <span />
            <span />
          </button>
        </div>
      </div>
      <span ref={progRef} className={s.progress} aria-hidden="true" />
      <ModeBanner />
    </header>
  );
}
