"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { CountUp } from "@/components/CountUp";
import { isSampleData, useHealth } from "@/components/HealthProvider";
import { NetworkBg } from "@/components/landing/NetworkBg";
import { NeuronArt } from "@/components/landing/NeuronArt";
import s from "@/components/landing/landing.module.css";
import { api } from "@/lib/api/client";
import { featureName, fmtInt } from "@/lib/format";
import { useApi } from "@/lib/useApi";

/** Adds data-in="1" once the element scrolls into view. Works for elements that mount late. */
function useReveal<T extends HTMLElement>() {
  const [el, setEl] = useState<T | null>(null);
  useEffect(() => {
    if (!el) return;
    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          el.dataset.in = "1";
          io.disconnect();
        }
      },
      { threshold: 0.15 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [el]);
  return setEl;
}

export default function Landing() {
  const { data: h } = useHealth();
  const sample = isSampleData(h);
  const names = useApi((sig) => api.features({ status: "named", sort: "max_activation", order: "desc", limit: 40 }, sig), "landing:names");
  const metrics = useApi((sig) => api.metrics(null, sig), "landing:metrics");
  const ceRec = metrics.data?.metrics.find((m) => m.name === "ce_recovered")?.value ?? null;

  const artRef = useRef<HTMLDivElement>(null);
  const steps = useReveal<HTMLElement>();
  const stats = useReveal<HTMLElement>();
  const end = useReveal<HTMLElement>();

  function parallax(e: React.MouseEvent) {
    const el = artRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    el.style.setProperty("--mx", ((e.clientX - r.left) / r.width - 0.5).toFixed(3));
    el.style.setProperty("--my", ((e.clientY - r.top) / r.height - 0.5).toFixed(3));
  }

  const labels = (names.data?.items ?? []).map((f) => featureName(f.label, f.feature_id));
  const rowA = labels.slice(0, 20);
  const rowB = labels.slice(20, 40).length ? labels.slice(20, 40) : labels.slice(0, 20).reverse();

  return (
    <div className={s.landing}>
      {/* hero */}
      <section className={s.hero} onMouseMove={parallax}>
        <NetworkBg />
        <div className={s.heroInner}>
          <div className={s.heroCopy}>
            <span className={`${s.kicker} rise`}>
              <i /> {h ? `${h.model_name} · layer ${h.hook_layer} · ${fmtInt(h.d_sae)} features` : "GPT-2 · layer 8"}
            </span>
            <h1 className={`${s.title} rise`} style={{ animationDelay: "0.08s" }}>
              See inside
              <br />
              the model.
            </h1>
            <p className={`${s.sub} rise`} style={{ animationDelay: "0.16s" }}>
              Type a sentence. Watch its neurons light up. Switch one off and see what changes.
            </p>
            <div className={`${s.ctas} rise`} style={{ animationDelay: "0.24s" }}>
              <Link href="/microscope" className="btn btn-lg">
                Open microscope <span aria-hidden="true">→</span>
              </Link>
              <Link href="/features" className="btn btn-ghost btn-lg">
                Browse features
              </Link>
            </div>
          </div>

          <div className={s.heroArt} ref={artRef}>
            <div className={s.cutout}>
              <div className={s.cutoutRing} aria-hidden="true" />
              <NeuronArt seed={11} />
            </div>
            <span className={s.figTag}>fig. 1 — a neuron</span>
          </div>
        </div>
        <a href="#how" className={s.scrollCue} aria-label="Scroll down">
          <span />
        </a>
      </section>

      {/* real feature names */}
      {labels.length > 0 && (
        <section className={s.ticker} aria-label="Some named features">
          <div className={s.tickRow}>
            <div className={s.tickTrack}>
              {[...rowA, ...rowA].map((l, i) => (
                <span key={i}>{l}</span>
              ))}
            </div>
          </div>
          <div className={`${s.tickRow} ${s.tickReverse}`}>
            <div className={s.tickTrack}>
              {[...rowB, ...rowB].map((l, i) => (
                <span key={i}>{l}</span>
              ))}
            </div>
          </div>
        </section>
      )}

      {/* how it works */}
      <section id="how" className={s.steps} ref={steps}>
        <h2 className={s.h2}>Three steps.</h2>
        <div className={s.stepGrid}>
          <article className={s.step}>
            <div className={s.stepArt}>
              <div className={s.typing}>
                <span>The court ruled</span>
                <i />
              </div>
            </div>
            <span className={s.stepNum}>01</span>
            <h3>Type</h3>
            <p>Any sentence.</p>
          </article>
          <article className={s.step}>
            <div className={s.stepArt}>
              <div className={s.dots}>
                {Array.from({ length: 35 }, (_, i) => (
                  <i key={i} style={{ animationDelay: `${((i * 7) % 35) * 0.09}s` }} />
                ))}
              </div>
            </div>
            <span className={s.stepNum}>02</span>
            <h3>Watch</h3>
            <p>Neurons fire, token by token.</p>
          </article>
          <article className={s.step}>
            <div className={s.stepArt}>
              <div className={s.toggle}>
                <span className={s.knob} />
              </div>
            </div>
            <span className={s.stepNum}>03</span>
            <h3>Steer</h3>
            <p>Force one on or off. Compare.</p>
          </article>
        </div>
      </section>

      {/* numbers from the server */}
      {h && (
        <section className={s.stats} ref={stats}>
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
            <span>active per token</span>
          </div>
          <div className={s.stat}>
            <b>
              L<CountUp to={h.hook_layer} />
            </b>
            <span>layer of {h.model_name}</span>
          </div>
          {ceRec !== null && (
            <div className={s.stat}>
              <b>
                <CountUp to={ceRec * 100} decimals={1} suffix="%" />
              </b>
              <span>loss recovered</span>
            </div>
          )}
          {sample && <p className={s.simNote}>Sample data, not a trained model.</p>}
        </section>
      )}

      {/* end */}
      <section className={s.end} ref={end}>
        <div className={s.endArt} aria-hidden="true">
          <NeuronArt seed={29} />
        </div>
        <h2 className={s.h2}>Look closer.</h2>
        <Link href="/microscope" className="btn btn-lg">
          Open microscope <span aria-hidden="true">→</span>
        </Link>
      </section>
    </div>
  );
}
