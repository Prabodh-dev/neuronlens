"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import type { InspectResponse } from "@/lib/api/types";
import { featureName, showToken } from "@/lib/format";
import { tokenColors } from "@/lib/heat";
import type { ApiState } from "@/lib/useApi";
import s from "./landing.module.css";

export const DEMO_TEXT = "The Supreme Court ruled that the decree was void.";

/** The landing page's one real /inspect result. Steps through the tokens by itself; hover or click to take over. */
export function LiveDemo({ res }: { res: ApiState<InspectResponse> }) {
  const tokens = useMemo(() => res.data?.tokens ?? [], [res.data]);
  const [sel, setSel] = useState(1);
  const [hold, setHold] = useState(false);

  // tokens worth stepping to: not the first (outlier) position, with features
  const steps = useMemo(() => tokens.filter((t) => !t.is_outlier_position && t.features.length > 0).map((t) => t.index), [tokens]);
  const peak = useMemo(() => Math.max(1e-6, ...tokens.filter((t) => !t.is_outlier_position).map((t) => t.features[0]?.activation ?? 0)), [tokens]);

  useEffect(() => {
    if (hold || steps.length < 2 || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const id = setInterval(() => setSel((cur) => steps[(steps.indexOf(cur) + 1) % steps.length] ?? steps[0]), 1700);
    return () => clearInterval(id);
  }, [hold, steps]);

  const tok = tokens.find((t) => t.index === sel) ?? tokens.find((t) => t.index === steps[0]);
  const top = tok?.features[0]?.activation ?? 1;

  if (res.error) return null;

  return (
    <div className={s.demoCard} onPointerEnter={() => setHold(true)} onPointerLeave={() => setHold(false)}>
      <div className={s.demoHead}>
        <span className={s.liveDot} aria-hidden="true" />
        <span>/inspect</span>
        <span className={s.demoHint}>{hold ? "paused" : "auto"}</span>
      </div>

      <p className={s.demoText} aria-label="Tokens of the example sentence">
        {tokens.length === 0
          ? DEMO_TEXT.split(" ").map((w, i) => (
              <span key={i} className={`${s.ghostTok} skel`}>
                {w}{" "}
              </span>
            ))
          : tokens.map((t) => {
              const a = t.features[0]?.activation ?? 0;
              const skip = t.is_outlier_position;
              return (
                <button
                  key={t.index}
                  type="button"
                  className={`tok ${skip ? "skip" : ""} ${t.text === "" ? "empty" : ""} ${tok?.index === t.index ? "selected" : ""}`}
                  style={skip ? undefined : tokenColors(a / peak)}
                  onClick={() => !skip && setSel(t.index)}
                  aria-pressed={tok?.index === t.index}
                  aria-label={skip ? `token "${t.text}", first position, not scored` : `token "${t.text}", strongest feature ${a.toFixed(1)}`}
                >
                  {showToken(t.text)}
                </button>
              );
            })}
      </p>

      <div className={s.demoFeats} aria-live="polite">
        <div className={s.demoTok}>
          <span className="label-mono">Token {tok?.index ?? "–"}</span>
          <b>{tok ? `“${showToken(tok.text)}”` : "…"}</b>
          <span className="label-mono">top 5 features</span>
        </div>
        {(tok?.features ?? Array.from({ length: 5 }, () => null)).slice(0, 5).map((f, i) =>
          f ? (
            <Link key={`${tok!.index}-${f.feature_id}`} href={`/features/${f.feature_id}`} className={s.featRow} style={{ animationDelay: `${i * 60}ms` }}>
              <span className={s.featId}>#{f.feature_id}</span>
              <span className={`${s.featName} ${f.label ? "" : s.unl}`}>{featureName(f.label ?? null, f.feature_id)}</span>
              <span className={s.featBar}>
                <i style={{ width: `${Math.max(4, (f.activation / top) * 100)}%` }} />
              </span>
              <span className={s.featVal}>{f.activation.toFixed(1)}</span>
            </Link>
          ) : (
            <span key={i} className={`${s.featGhost} skel`} />
          ),
        )}
      </div>
    </div>
  );
}
