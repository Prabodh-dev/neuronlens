"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useState } from "react";
import { HeatText } from "@/components/HeatText";
import { isMockModel, useHealth } from "@/components/HealthProvider";
import { EmptyState, ErrorState, Skeleton } from "@/components/States";
import { api } from "@/lib/api/client";
import type { InspectResponse, InspectToken, ReportRequest, TextCompareRequest } from "@/lib/api/types";
import { featureName, fmtNum, showToken } from "@/lib/format";
import { t as tr } from "@/lib/copy";
import { useAction } from "@/lib/useApi";
import s from "./inspect.module.css";
import { CompareView, ReportView } from "./views";

const TRY = [
  "The band walked on stage and played a new song. The Supreme Court ruled that the decree was void.",
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
type View = "tokens" | "report" | "compare";
const VIEWS: { v: View; label: string; run: string }[] = [
  { v: "tokens", label: "Each token", run: "Inspect" },
  { v: "report", label: "By sentence", run: "Report" },
  { v: "compare", label: "Two texts", run: "Compare" },
];
// a pair for the Two texts view
const PAIR = { a: "The band walked on stage and played a new song.", b: "The judge walked into court and read the new ruling." };
// "Hide very common features": max_density for X6 and X7. It hides the features that fire on more than 1% of
// tokens, some named ones too (e.g. "the – noun phrase start").
const COMMON = 0.01;
// "Rank by" for X6 and X7: raw activation (the server's default) or closest to its own peak (relative_to_max)
type Rank = "activation" | "relative";

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
  const [view, setView] = useState<View>("tokens");
  const [textB, setTextB] = useState("");
  // X6 and X7
  const report = useAction((body: ReportRequest, sig) => api.inspectReport(body, sig));
  const compare = useAction((body: TextCompareRequest, sig) => api.inspectCompare(body, sig));
  const busy = loading || report.loading || compare.loading;
  const [hideCommon, setHideCommon] = useState(false);
  const [rank, setRank] = useState<Rank>("activation");
  // the settings each shown result was made with
  const [repOpt, setRepOpt] = useState<{ hide: boolean; rank: Rank }>({ hide: false, rank: "activation" });
  const [cmpOpt, setCmpOpt] = useState<{ hide: boolean; rank: Rank }>({ hide: false, rank: "activation" });

  function runReport(v: string, hide = hideCommon, by = rank) {
    if (!v.trim()) return;
    setRepOpt({ hide, rank: by });
    report.run({ text: v, top_n: topN, sort: by, ...(hide ? { max_density: COMMON } : {}) });
  }
  function runCompare(a: string, b: string, hide = hideCommon, by = rank) {
    if (!a.trim() || !b.trim()) return;
    setCmpOpt({ hide, rank: by });
    compare.run({ text_a: a, text_b: b, top_n: 10, sort: by, ...(hide ? { max_density: COMMON } : {}) });
  }
  function submit() {
    if (view === "report") runReport(text);
    else if (view === "compare") runCompare(text, textB);
    else go();
  }
  // re-run the shown result with the new setting
  function toggleCommon(on: boolean) {
    setHideCommon(on);
    if (view === "report" && report.data) runReport(text, on);
    if (view === "compare" && compare.data) runCompare(text, textB, on);
  }
  function rankBy(by: Rank) {
    setRank(by);
    if (view === "report" && report.data) runReport(text, hideCommon, by);
    if (view === "compare" && compare.data) runCompare(text, textB, hideCommon, by);
  }

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
          <p className="lede">Paste text. See what fires on each token, in each sentence, or between two texts.</p>
        </div>
      </div>

      <div className="seg" role="group" aria-label="View" style={{ marginBottom: 12 }}>
        {VIEWS.map((x) => (
          <button key={x.v} type="button" aria-pressed={view === x.v} onClick={() => setView(x.v)}>
            {x.label}
          </button>
        ))}
      </div>

      <form
        className={`card ${s.input}`}
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        {view === "compare" ? (
          <label htmlFor="ins-text" className={s.textLabel}>
            Text A
          </label>
        ) : (
          <label htmlFor="ins-text" className="visually-hidden">
            Text to inspect
          </label>
        )}
        <textarea
          id="ins-text"
          className={s.textarea}
          placeholder={tr("inspect.placeholder")}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
              e.preventDefault();
              submit();
            }
          }}
          rows={3}
        />
        {view === "compare" && (
          <>
            <label htmlFor="ins-text-b" className={s.textLabel}>
              Text B
            </label>
            <textarea id="ins-text-b" className={`${s.textarea} ${s.textareaB}`} placeholder="A second text to compare with A" value={textB} onChange={(e) => setTextB(e.target.value)} rows={3} />
          </>
        )}
        <div className={s.inputBar}>
          <div className={s.tries}>
            {view === "compare" && (
              <button
                type="button"
                className={s.try}
                onClick={() => {
                  setText(PAIR.a);
                  setTextB(PAIR.b);
                  runCompare(PAIR.a, PAIR.b);
                }}
              >
                Try a pair: band vs. court
              </button>
            )}
            {view !== "compare" && TRY.map((t) => (
              <button
                key={t}
                type="button"
                className={s.try}
                onClick={() => {
                  setText(t);
                  if (view === "report") runReport(t);
                  else go(t);
                }}
              >
                {t.split("\n")[0]}
              </button>
            ))}
          </div>
          <div className="row" style={{ gap: 10, flexShrink: 0 }}>
            {view !== "tokens" && (
              <>
                <label className="small muted" htmlFor="ins-rank">
                  Rank by
                </label>
                <select id="ins-rank" className="select" style={{ width: 190 }} value={rank} onChange={(e) => rankBy(e.target.value as Rank)}>
                  <option value="activation">Strongest</option>
                  <option value="relative">Closest to its own peak</option>
                </select>
              </>
            )}
            {view !== "tokens" && (
              <label className="check" title={`Hide features that fire on more than ${COMMON * 100}% of all tokens`}>
                <input type="checkbox" checked={hideCommon} onChange={(e) => toggleCommon(e.target.checked)} />
                Hide very common features
              </label>
            )}
            {view !== "compare" && (
              <>
                <label className="small muted" htmlFor="ins-topn">
                  {view === "tokens" ? "Per token" : "Per sentence"}
                </label>
                <select id="ins-topn" className="select" style={{ width: 70 }} value={topN} onChange={(e) => setTopN(Number(e.target.value))}>
                  {[3, 5, 10, 20].map((n) => (
                    <option key={n} value={n}>
                      {n}
                    </option>
                  ))}
                </select>
              </>
            )}
            <button className="btn" type="submit" disabled={!text.trim() || (view === "compare" && !textB.trim()) || busy}>
              {busy ? "Reading…" : VIEWS.find((x) => x.v === view)!.run}
            </button>
          </div>
        </div>

      </form>

      {view === "report" && (
        <>
          {report.error && (
            <div style={{ marginTop: 16 }}>
              <ErrorState error={report.error} onRetry={submit} />
            </div>
          )}
          {report.loading && !report.data && <ListSkeleton />}
          {report.data && <ReportView res={report.data} busy={report.loading} hidCommon={repOpt.hide ? COMMON : null} rank={repOpt.rank} />}
          {!report.data && !report.loading && !report.error && (
            <div className="card" style={{ marginTop: 16 }}>
              <EmptyState title="By sentence" icon="spark">
                Splits the text into sentences and lists the most active features in each.
              </EmptyState>
            </div>
          )}
        </>
      )}

      {view === "compare" && (
        <>
          {compare.error && (
            <div style={{ marginTop: 16 }}>
              <ErrorState error={compare.error} onRetry={submit} />
            </div>
          )}
          {compare.loading && !compare.data && <ListSkeleton />}
          {compare.data && <CompareView res={compare.data} busy={compare.loading} hidCommon={cmpOpt.hide ? COMMON : null} rank={cmpOpt.rank} />}
          {!compare.data && !compare.loading && !compare.error && (
            <div className="card" style={{ marginTop: 16 }}>
              <EmptyState title="Two texts" icon="spark">
                Write two texts. See which features fire more in one than in the other.
              </EmptyState>
            </div>
          )}
        </>
      )}

      {view === "tokens" && error && (
        <div style={{ marginTop: 16 }}>
          <ErrorState error={error} onRetry={() => go()} />
        </div>
      )}

      {view === "tokens" && !data && !loading && !error && (
        <div className="card" style={{ marginTop: 16 }}>
          <EmptyState title="Nothing yet" icon="spark">
            {tr("inspect.hint")}
          </EmptyState>
        </div>
      )}

      {view === "tokens" && loading && !data && (
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

      {view === "tokens" && data && renderResult(data)}
    </div>
  );

  function renderResult(res: InspectResponse) {
    return (
      <div className={`${s.layout} ${loading ? s.dim : ""}`}>
        <section className="card" aria-labelledby="doc-h">
          <div className="card-head">
            <h2 id="doc-h">
              {res.tokens.length} tokens{isMockModel(h) ? " · simulated" : ""}
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
                    tr("inspect.outlier")
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
                <span className={s.tokBig}>{showToken(selTok.text)}</span>
              </div>
              {selTok.is_outlier_position ? (
                <p className={s.skipNote}>{tr("inspect.outlier")}</p>
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

function ListSkeleton() {
  return (
    <div className="card card-pad stack" style={{ marginTop: 16 }}>
      {[0, 1, 2, 3, 4].map((i) => (
        <Skeleton key={i} h={26} w={`${92 - i * 9}%`} />
      ))}
    </div>
  );
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
