"use client";

import Link from "next/link";
import { FeatureTitle } from "@/components/FeatureBits";
import { EmptyState, ErrorState, Skeleton } from "@/components/States";
import { api } from "@/lib/api/client";
import { showToken } from "@/lib/format";
import { useApi } from "@/lib/useApi";
import s from "./explorer.module.css";

/* W8: the features that fired most on the words typed in the search box (POST /features/search).
   max_density 0.01 is always sent, with no toggle: without it a few very common features fill every
   result (server/README.md). It shows which features fired on the words, not what they mean. */
const COMMON = 0.01;

export function WordsResults({ q }: { q: string }) {
  const query = q.trim();
  const { data, error, loading, reload } = useApi(
    (sig) => api.featureSearch({ query, top_n: 10, max_density: COMMON, sort: "activation" }, sig),
    query ? `w8:${query}` : null,
  );

  if (!query)
    return (
      <div className="card" style={{ marginTop: 16 }}>
        <EmptyState title="Search by your words" icon="search">
          Type a few words, like “court ruling”, and press Enter. It lists the features that fired most on them.
        </EmptyState>
      </div>
    );

  return (
    <section className="card" aria-labelledby="w8-h" style={{ marginTop: 16 }}>
      <div className="card-head">
        <h2 id="w8-h">Features that fired most on your words</h2>
        {data?.simulated && <span className="pill pill-sim">Simulated</span>}
      </div>
      {error && (
        <div className={s.pad}>
          <ErrorState error={error} onRetry={reload} />
        </div>
      )}
      {loading && !data && (
        <div className="card-pad stack">
          {[0, 1, 2, 3, 4].map((i) => (
            <Skeleton key={i} h={26} w={`${92 - i * 9}%`} />
          ))}
        </div>
      )}
      {data && (
        <div className={loading ? s.dim : undefined}>
          <div className={s.wordToks}>
            <span className="small muted">As the model split them:</span>
            {data.tokens.map((t, i) => (
              <span key={i} className={`mono ${s.wordTok}`}>
                {showToken(t)}
              </span>
            ))}
          </div>
          {data.features.length === 0 ? (
            <p className={`${s.pad} small muted`}>No feature fired on these words.</p>
          ) : (
            <ol className={s.wordList}>
              <li className={`${s.wordRow} ${s.wordHead}`} aria-hidden="true">
                <span />
                <span />
                <span>of its peak</span>
                <span>strongest on</span>
              </li>
              {data.features.map((f) => {
                const r = f.relative_to_max;
                return (
                  <li key={f.feature_id}>
                    <Link href={`/features/${f.feature_id}`} className={`${s.wordRow} ${s.item}`}>
                      <span className={`mono ${s.id}`}>{f.feature_id}</span>
                      <span className={s.name}>
                        <FeatureTitle label={f.label} id={f.feature_id} />
                      </span>
                      <span className={s.metric} title={r === null ? "No peak on record" : "This activation as a share of the feature's own peak"}>
                        <span className="bar heat" style={{ display: "block" }}>
                          <i style={{ width: `${r === null ? 0 : Math.max(2, Math.min(1, Math.max(0, r)) * 100)}%` }} />
                        </span>
                        <span className="mono">{r === null ? "—" : `${Math.round(r * 100)}%`}</span>
                      </span>
                      <span className={`mono small ${s.wordPeak}`}>{showToken(data.tokens[f.peak_token] ?? "")}</span>
                    </Link>
                  </li>
                );
              })}
            </ol>
          )}
          <div className={s.wordFoot}>
            <span className="small muted">
              Shows which features fired on your words, not what the words mean. Features that fire on more than 1% of all tokens are left out. Ranked by
              raw activation.
            </span>
          </div>
        </div>
      )}
    </section>
  );
}
