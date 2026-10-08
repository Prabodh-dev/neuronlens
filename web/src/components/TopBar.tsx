"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
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
  { href: "/metrics", label: "Metrics" },
];

export function TopBar() {
  const path = usePathname();
  const { data: h, error } = useHealth();
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const on = () => setScrolled(window.scrollY > 8);
    on();
    window.addEventListener("scroll", on, { passive: true });
    return () => window.removeEventListener("scroll", on);
  }, []);
  useEffect(() => setOpen(false), [path]);

  return (
    <>
      <header className={`${s.bar} ${scrolled ? s.scrolled : ""}`}>
        <div className={s.inner}>
          <Link href="/" className={s.brand} aria-label="NeuronLens home">
            <Logo />
            <span>NeuronLens</span>
          </Link>

          <nav className={`${s.nav} ${open ? s.navOpen : ""}`} aria-label="Main">
            {NAV.map((n) => {
              const on = path === n.href || path.startsWith(n.href + "/");
              return (
                <Link key={n.href} href={n.href} className={s.link} aria-current={on ? "page" : undefined}>
                  {n.label}
                </Link>
              );
            })}
          </nav>

          <div className={s.right}>
            {h && (
              <span className={s.chip} title={`Run ${h.run_id}`}>
                {h.model_name} · layer {h.hook_layer} · {fmtInt(h.d_sae)} features · k={h.k}
              </span>
            )}
            {h && (
              <span className={s.status}>
                <i className={isSampleData(h) || isMockModel(h) ? s.dotSim : s.dotLive} />
                {isSampleData(h) ? "Sample data" : isMockModel(h) ? "Mock" : "Live"}
              </span>
            )}
            {error && (
              <span className={s.status}>
                <i className={s.dotOff} /> Offline
              </span>
            )}
            <Link href="/microscope" className={`btn btn-sm ${s.cta}`}>
              Open microscope
            </Link>
            <button className={s.menu} aria-label="Menu" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
              <span />
              <span />
            </button>
          </div>
        </div>
        <ModeBanner />
      </header>
    </>
  );
}
