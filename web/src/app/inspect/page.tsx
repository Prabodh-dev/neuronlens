"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useState } from "react";
import { HeatText } from "@/components/HeatText";
import { isSimulated, useHealth } from "@/components/HealthProvider";
import { EmptyState, ErrorState, Skeleton } from "@/components/States";
import { api } from "@/lib/api/client";
import type { InspectResponse, InspectToken } from "@/lib/api/types";
import { featureName, fmtNum, showToken } from "@/lib/format";
import { useAction } from "@/lib/useApi";
import s from "./inspect.module.css";

const TRY = [
  "The court ruled that the contract was void.",
  "Whisk the eggs, then fold in the flour slowly.",
  "She opened the door and walked into the rain.",
  "def main():\n    return parse(json.loads(body))",
];

export default function InspectPage() {
  return (
    <Suspense fallback={<div className="page" />}>
      <Inspector />
    </Suspense>
  );
}

type Pin = { feature_id: number; label: string | null };

function Inspector() {
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const { data: h } = useHealth();
  const initial = params.get("text") ?? "";
  const [text, setText] = useState(initial);
  const [topN, setTopN] = useState(5);
  const [sel, setSel] = useState<number | null>(null);
  const [pin, setPin] = useState<Pin | null>(null);
  const { data, error, loading, run } = useAction((body: { text: string; top_n: number }, sig) => api.inspect(body, sig));

  async function go(t = text) {
    const v = t;
    if (!v.trim()) return;
    setSel(null);
    setPin(null);
    router.replace(`${pathname}?text=${encodeURIComponent(v)}`, { scroll: false });
    const res = await run({ text: v, top_n: topN });
    if (res) setSel(firstActive(res));
  }

  // Run once on arrival when the URL carries text (e.g. from the home page).
  useEffect(() => {
    if (initial.trim()) go(initial);
  }, []);

  const tokens = data?.tokens ?? [];
  const docMax = useMemo(() => {
    let m = 0;
    for (const t of tokens) for (const f of t.features) m = Math.max(m, f.activation);
    return m || 1;
  }, [tokens]);
  const pinMax = useMemo(() => {
    if (!pin) return 1;
    let m = 0;
    for (const t of tokens) for (const f of t.features) if (f.feature_id === pin.feature_id) m = Math.max(m, f.activation);
    return m || 1;
  }, [tokens, pin]);

  const heat = (t: InspectToken) => {
    if (t.is_outlier_position) return 0;
    if (pin) return (t.features.find((f) => f.feature_id === pin.feature_id)?.activation ?? 0) / pinMax;
    return (t.features[0]?.activation ?? 0) / docMax;
  };

  function onKey(e: React.KeyboardEvent) {
    if (!tokens.length) return;
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      const d = e.key === "ArrowRight" ? 1 : -1;
      setSel((cur) => Math.max(0, Math.min(tokens.length - 1, (cur ?? -1) + d)));
    }
  }

  const selTok = sel !== null ? tokens[sel] : null;

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Inspect</h1>
          <p className="lede">Paste text. See what fires on each token.</p>
        </div>
      </div>

      <form
        className={`card ${s.input}`}
        onSubmit={(e) => {
          e.preventDefault();
          go();
        }}
      >
        <label htmlFor="ins-text" className="visually-hidden">
          Text to inspect
        </label>
        <textarea
          id="ins-text"
          className={s.textarea}
          placeholder="Type or paste text…"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
              e.preventDefault();
              go();
            }
          }}
          rows={3}
        />
        <div className={s.inputBar}>
          <div className={s.tries}>
            {TRY.map((t) => (
              <button
                key={t}
                type="button"
                className={s.try}
                onClick={() => {
                  setText(t);
                  go(t);
                }}
              >
                {t.split("\n")[0]}
              </button>
            ))}
          </div>
          <div className="row" style={{ gap: 10, flexShrink: 0 }}>
            <label className="small muted" htmlFor="ins-topn">
              Per token
            </label>
            <select id="ins-topn" className="select" style={{ width: 70 }} value={topN} onChange={(e) => setTopN(Number(e.target.value))}>
              {[3, 5, 10, 20].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
            <button className="btn" type="submit" disabled={!text.trim() || loading}>
              {loading ? "Reading…" : "Inspect"}
            </button>
          </div>
        </div>

      </form>

      {error && (
        <div style={{ marginTop: 16 }}>
          <ErrorState error={error} onRetry={() => go()} />
        </div>
      )}

      {!data && !loading && !error && (
        <div className="card" style={{ marginTop: 16 }}>
          <EmptyState title="Nothing yet" icon="spark">
            Type a sentence or pick an example.
          </EmptyState>
        </div>
      )}

      {loading && !data && (
        <div className={s.layout}>
          <div className="card card-pad stack">
            <Skeleton h={22} />
            <Skeleton h={22} w="80%" />
            <Skeleton h={22} w="60%" />
          </div>
          <div className="card card-pad stack">
            {[0, 1, 2, 3].map((i) => (
              <Skeleton key={i} h={34} />
            ))}
          </div>
        </div>
      )}

      {data && renderResult(data)}
    </div>
  );

  function renderResult(res: InspectResponse) {
    return (
      <div className={`${s.layout} ${loading ? s.dim : ""}`}>
        <section className="card" aria-labelledby="doc-h">
          <div className="card-head">
            <h2 id="doc-h">
              {res.tokens.length} tokens{h && isSimulated(h) ? " · simulated" : ""}
            </h2>
            {pin ? (
              <span className={s.pin}>
                Shading by <b>#{pin.feature_id}</b> {featureName(pin.label, pin.feature_id)}
                <button onClick={() => setPin(null)} aria-label="Stop shading by this feature">
                  ×
                </button>
              </span>
            ) : (
              <span className="heat-legend">
                weak <i /> strong
              </span>
            )}
          </div>
          <div className={s.docWrap} onKeyDown={onKey}>
            <HeatText
              className={s.doc}
              label="Inspected text. Use left and right arrows to move between tokens."
              selected={sel}
              onSelect={setSel}
              tokens={res.tokens.map((t) => {
                const top = t.features[0];
                return {
                  text: t.text,
                  heat: heat(t),
                  skip: t.is_outlier_position,
                  tip: t.is_outlier_position ? (
                    "Position 0 is skipped"
                  ) : top ? (
                    <>
                      <b>{featureName(top.label, top.feature_id)}</b>
                      <br />
                      <span className="mono">
                        #{top.feature_id} · {fmtNum(top.activation)}
                      </span>
                    </>
                  ) : (
                    "No active features"
                  ),
                };
              })}
            />
          </div>

        </section>

        <aside className={`card ${s.panel}`} aria-live="polite">
          {!selTok && <EmptyState title="Pick a token" icon="search">Click any token.</EmptyState>}
          {selTok && (
            <>
              <div className={s.panelHead}>
                <span className="eyebrow">Token {selTok.index}</span>
                <span className={s.tokBig}>{showToken(selTok.text).replace(/^ /, "·")}</span>
              </div>
              {selTok.is_outlier_position ? (
                <p className={s.skipNote}>The first token is skipped.</p>
              ) : selTok.features.length === 0 ? (
                <p className={s.skipNote}>No features were active on this token.</p>
              ) : (
                <ol className={s.feats}>
                  {selTok.features.map((f, fi) => {
                    const share = f.activation / (selTok.features[0].activation || 1);
                    const on = pin?.feature_id === f.feature_id;
                    return (
                      <li key={`${f.feature_id}-${fi}`} className={on ? `${s.feat} ${s.featOn}` : s.feat}>
                        <button className={s.featMain} onClick={() => setPin(on ? null : { feature_id: f.feature_id, label: f.label })} title="Shade the whole text by this feature">
                          <span className={s.featName}>{featureName(f.label, f.feature_id)}</span>
                          <span className={`mono ${s.featVal}`}>{fmtNum(f.activation)}</span>
                          <span className="bar heat" style={{ display: "block", gridColumn: "1 / -1" }}>
                            <i style={{ width: `${Math.max(3, share * 100)}%` }} />
                          </span>
                        </button>
                        <Link href={`/features/${f.feature_id}`} className={s.featLink} aria-label={`Open feature ${f.feature_id}`}>
                          #{f.feature_id} →
                        </Link>
                      </li>
                    );
                  })}
                </ol>
              )}
              <p className={s.panelHint}>Click a feature to shade the text by it.</p>
            </>
          )}
        </aside>
      </div>
    );
  }
}

function firstActive(res: InspectResponse): number | null {
  let best: number | null = null;
  let bv = -1;
  res.tokens.forEach((t, i) => {
    const v = t.is_outlier_position ? -1 : (t.features[0]?.activation ?? -1);
    if (v > bv) {
      bv = v;
      best = i;
    }
  });
  return best;
}
