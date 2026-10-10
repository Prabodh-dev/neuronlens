"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { CountUp } from "@/components/CountUp";
import { isSampleData, useHealth } from "@/components/HealthProvider";
import { DEMO_TEXT, LiveDemo } from "@/components/landing/LiveDemo";
import { NeuronArt } from "@/components/landing/NeuronArt";
import { Tour } from "@/components/landing/Tour";
import { useModelData } from "@/components/useModelData";
import s from "@/components/landing/landing.module.css";
import { api } from "@/lib/api/client";
import { fmtInt, showToken } from "@/lib/format";
import { modelLabel } from "@/lib/models";
import { tokenColors } from "@/lib/heat";
import { useApi } from "@/lib/useApi";

/** Adds data-in="1" once the element scrolls into view. Works for elements that mount late. */
function useReveal<T extends HTMLElement>() {
  const [el, setEl] = useState<T | null>(null);
  useEffect(() => {
    if (!el) return;
    const show = () => {
      el.dataset.in = "1";
      io.disconnect();
      window.removeEventListener("scroll", check);
    };
    const io = new IntersectionObserver(([e]) => e.isIntersecting && show(), { threshold: 0.15 });
    // fallback in case the observer is late: reveal once the top is well inside the viewport
    const check = () => el.getBoundingClientRect().top < window.innerHeight * 0.85 && show();
    io.observe(el);
    window.addEventListener("scroll", check, { passive: true });
    check();
    return () => {
      io.disconnect();
      window.removeEventListener("scroll", check);
    };
  }, [el]);
  return setEl;
}


// server and browser trig can differ in the last digits; round so hydration matches
const round = (v: number) => Math.round(v * 100) / 100;

// rings of dots for the microscope tile
const RINGS = [
  { r: 22, n: 8 },
  { r: 42, n: 14 },
  { r: 62, n: 20 },
  { r: 82, n: 26 },
];

export default function Landing() {
  const { data: h } = useHealth();
  const sample = isSampleData(h);
  const names = useApi((sig) => api.features({ status: "named", sort: "max_activation", order: "desc", limit: 60 }, sig), "landing:names");
  const metrics = useApi((sig) => api.metrics(null, sig), "landing:metrics");
  const ceRec = metrics.data?.metrics.find((m) => m.name === "ce_recovered")?.value ?? null;
  // one real /inspect call feeds the tour, the live card and the inspect tile (top_n is capped at 20)
  const demo = useApi((sig) => api.inspect({ text: DEMO_TEXT, top_n: 20 }, sig), "landing:demo");
  const demoPeak = Math.max(1e-6, ...(demo.data?.tokens ?? []).filter((t) => !t.is_outlier_position).map((t) => t.features[0]?.activation ?? 0));

  const named = useMemo(() => (names.data?.items ?? []).filter((f) => f.label).map((f) => ({ id: f.feature_id, label: f.label! })), [names.data]);
  // "one neuron, many ideas": this model's picks (lib/model-data.ts) when the database has them, otherwise
  // the first named ones
  const picks = useModelData()?.ideas;
  const ideas = useMemo(() => {
    const pref = (picks ?? []).map((id) => named.find((n) => n.id === id)).filter(Boolean) as typeof named;
    return (pref.length >= 5 ? pref : named).slice(0, 5);
  }, [named, picks]);

  const live = useReveal<HTMLElement>();
  const split = useReveal<HTMLElement>();
  const bento = useReveal<HTMLElement>();
  const end = useReveal<HTMLElement>();

  // sizes come from /health only, never hard-coded
  const total = h ? fmtInt(h.d_sae) : null;

  return (
    <div className={s.landing}>
      {/* hero: a guided tour that plays by itself and loops */}
      <section className={s.hero}>
        <Tour h={h ?? null} named={names.data ? named : names.error ? [] : null} demo={demo.data} ceRec={ceRec} />
      </section>

      {/* the headline numbers, just below the tour */}
      <section aria-label="Key numbers">
        {h && (
          <div className={s.statBar}>
            <div className={s.stat}>
              <b>
                <CountUp to={h.d_sae} />
              </b>
              <span>features</span>
            </div>
            <div className={s.stat}>
              <b>
                <CountUp to={h.k} />
              </b>
              <span>fire per token</span>
            </div>
            <div className={s.stat}>
              <b>
                L<CountUp to={h.hook_layer} />
              </b>
              <span>layer of {modelLabel(h.model_name)}</span>
            </div>
            <div className={s.stat}>
              <b>{ceRec !== null ? <CountUp to={ceRec * 100} decimals={1} suffix="%" /> : "–"}</b>
              <span>loss recovered</span>
            </div>
            {sample && <p className={s.simNote}>Sample data, not a trained model.</p>}
          </div>
        )}
      </section>

      {/* one sentence, live */}
      {!demo.error && (
        <section className={s.live} ref={live}>
          <div className={s.liveCopy}>
            <span className="eyebrow">01 · Live</span>
            <h2 className={s.h2}>One sentence, token by token.</h2>
            <p className={s.lead}>Each token lights the features it fires. Brighter means stronger.</p>
            <Link href={`/inspect?text=${encodeURIComponent(DEMO_TEXT)}`} className={s.more}>
              Try your own <span aria-hidden="true">→</span>
            </Link>
          </div>
          <LiveDemo res={demo} />
        </section>
      )}

      {/* one neuron, many ideas */}
      <section className={s.split} ref={split}>
        <div className={s.splitHead}>
          <span className="eyebrow">02 · Why</span>
          <h2 className={s.h2}>One neuron, many ideas.</h2>
          <p className={s.lead}>A neuron mixes many ideas. A sparse autoencoder splits the layer into {total ?? ""} features, one idea each. Five of them:</p>
        </div>
        <div className={s.splitBody}>
          <div className={s.splitArt}>
            <div className={s.cutout}>
              <div className={s.cutoutRing} aria-hidden="true" />
              <NeuronArt seed={11} />
            </div>
            <span className={s.figTag}>fig. 1 — illustration</span>
          </div>
          <div className={s.fan} aria-hidden="true">
            <svg viewBox="0 0 100 500" preserveAspectRatio="none">
              {[50, 150, 250, 350, 450].map((y, i) => (
                <g key={y}>
                  <path d={`M0,250 C55,250 45,${y} 100,${y}`} className={s.fanLine} vectorEffect="non-scaling-stroke" />
                  <path d={`M0,250 C55,250 45,${y} 100,${y}`} className={s.fanPulse} pathLength={1} style={{ animationDelay: `${i * 0.9}s` }} />
                </g>
              ))}
            </svg>
          </div>
          <ul className={s.ideas}>
            {(ideas.length ? ideas : Array.from({ length: 5 }, () => null)).map((f, i) =>
              f ? (
                <li key={f.id} style={{ animationDelay: `${i * 0.9}s` }}>
                  <Link href={`/features/${f.id}`}>
                    <span className={s.ideaId}>#{f.id}</span>
                    <span className={s.ideaName}>{f.label}</span>
                  </Link>
                </li>
              ) : (
                <li key={i} className="skel" />
              ),
            )}
          </ul>
        </div>
      </section>

      {/* every screen */}
      <section className={s.bento} ref={bento}>
        <div className={s.bentoHead}>
          <span className="eyebrow">03 · Explore</span>
          <h2 className={s.h2}>Five ways in.</h2>
        </div>
        <div className={s.bentoGrid}>
          <Link href="/microscope" className={`${s.tile} ${s.tileBig}`}>
            <div className={s.tileArt}>
              <svg viewBox="-100 -100 200 200" className={s.rings} aria-hidden="true">
                {RINGS.map((ring, ri) => (
                  <g key={ri} className={s.ring} style={{ animationDuration: `${40 + ri * 18}s`, animationDirection: ri % 2 ? "reverse" : "normal" }}>
                    <circle r={ring.r} className={s.ringLine} />
                    {Array.from({ length: ring.n }, (_, k) => {
                      const a = (k / ring.n) * Math.PI * 2;
                      return (
                        <circle
                          key={k}
                          cx={round(Math.cos(a) * ring.r)}
                          cy={round(Math.sin(a) * ring.r)}
                          r={2.6 - ri * 0.4}
                          className={s.ringDot}
                          style={{ animationDelay: `${round(((k * 7 + ri * 5) % 23) * 0.27)}s` }}
                        />
                      );
                    })}
                  </g>
                ))}
                <circle r="7" className={s.core} />
              </svg>
            </div>
            <div className={s.tileText}>
              <h3>Microscope</h3>
              <p>Fly through all {total ?? "of them"}.</p>
            </div>
            <span className={s.arrow} aria-hidden="true">↗</span>
          </Link>

          <Link href="/features" className={s.tile}>
            <div className={`${s.tileArt} ${s.listArt}`}>
              <div className={s.listTrack}>
                {[...named, ...named].slice(0, Math.max(10, named.length * 2)).map((n, i) => (
                  <span key={i}>
                    <em>#{n.id}</em> {n.label}
                  </span>
                ))}
              </div>
            </div>
            <div className={s.tileText}>
              <h3>Features</h3>
              <p>Search, sort, open.</p>
            </div>
            <span className={s.arrow} aria-hidden="true">↗</span>
          </Link>

          <Link href="/inspect" className={s.tile}>
            <div className={s.tileArt}>
              <div className={s.heatRow}>
                {/* the first tokens of the real /inspect result, shaded by their real activations */}
                {(demo.data?.tokens ?? []).slice(0, 4).map((t, i) => (
                  <span
                    key={t.index}
                    className={t.is_outlier_position ? s.heatSkip : undefined}
                    style={{ ...(t.is_outlier_position ? {} : tokenColors((t.features[0]?.activation ?? 0) / demoPeak)), animationDelay: `${i * 0.6}s` }}
                  >
                    {showToken(t.text)}
                  </span>
                ))}
              </div>
            </div>
            <div className={s.tileText}>
              <h3>Inspect</h3>
              <p>Any sentence, token by token.</p>
            </div>
            <span className={s.arrow} aria-hidden="true">↗</span>
          </Link>

          <Link href="/steer" className={`${s.tile} ${s.tileWide}`}>
            <div className={s.tileArt}>
              <div className={s.steerArt}>
                <div className={s.slider}>
                  <i />
                  <b />
                </div>
                <div className={s.lines}>
                  {[0, 1, 2].map((l) => (
                    <div key={l} className={s.line}>
                      {Array.from({ length: 7 }, (_, k) => (
                        <span key={k} className={(k + l) % 3 === 1 ? s.changes : undefined} style={{ flexGrow: 1 + ((k * 5 + l * 3) % 4) }} />
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className={s.tileText}>
              <h3>Steer</h3>
              <p>Force a feature on or off. Compare.</p>
            </div>
            <span className={s.arrow} aria-hidden="true">↗</span>
          </Link>

          <Link href="/metrics" className={s.tile}>
            <div className={`${s.tileArt} ${s.metricArt}`}>
              <b>{ceRec !== null ? <CountUp to={ceRec * 100} decimals={1} suffix="%" /> : "–"}</b>
              <span className={s.metricBar}>
                <i style={{ width: `${(ceRec ?? 0) * 100}%` }} />
              </span>
              <span className="label-mono">loss recovered</span>
            </div>
            <div className={s.tileText}>
              <h3>Metrics</h3>
              <p>How good is the SAE.</p>
            </div>
            <span className={s.arrow} aria-hidden="true">↗</span>
          </Link>
        </div>
      </section>

      {/* end */}
      <section className={s.end} ref={end}>
        <svg className={s.endLens} viewBox="-200 -200 400 400" aria-hidden="true">
          <circle r="150" className={s.endRim} />
          <circle r="120" className={s.endInner} />
          <g className={s.endTicks}>
            {Array.from({ length: 90 }, (_, i) => {
              const a = (i / 90) * Math.PI * 2;
              const long = i % 5 === 0;
              return <line key={i} x1={round(Math.cos(a) * 160)} y1={round(Math.sin(a) * 160)} x2={round(Math.cos(a) * (long ? 176 : 168))} y2={round(Math.sin(a) * (long ? 176 : 168))} />;
            })}
          </g>
        </svg>
        <h2 className={s.endTitle}>Look closer.</h2>
        <Link href="/microscope" className="btn btn-lg">
          Open microscope <span aria-hidden="true">→</span>
        </Link>
      </section>
    </div>
  );
}
