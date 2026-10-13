"use client";

import Link from "next/link";
import { ErrorState, Skeleton } from "@/components/States";
import { api } from "@/lib/api/client";
import { featureName, fmtInt } from "@/lib/format";
import { useApi } from "@/lib/useApi";
import s from "./detail.module.css";

/** W4: the features that fired on the same tokens as this one in a token sample, with the counts and
 *  the sample size. Shared tokens only, never a claim about meaning (docs/features/co-fire-partners.md). The panel is
 *  hidden when the server has no co-fire data for this feature. */
export function Cofire({ id }: { id: number }) {
  const { data, error, loading, reload } = useApi((sig) => api.cofire(id, sig), `cofire:${id}`);
  // no co-fire data on this server, or an older server without the endpoint: leave the panel out
  if ((data && !data.available) || error?.code === "NOT_FOUND" || error?.code === "HTTP") return null;
  return (
    <section className={s.panel} aria-labelledby="cf-h">
      <div className={s.panelHead}>
        <h2 id="cf-h">Fired on the same tokens</h2>
        {data?.simulated && <span className="pill pill-sim">Simulated</span>}
      </div>
      <div className={s.panelBody}>
        {loading && !data && (
          <div className="stack" style={{ gap: 10 }}>
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} h={18} />
            ))}
          </div>
        )}
        {error && <ErrorState error={error} onRetry={reload} compact />}
        {data?.available && (
          <>
            <p className={s.note}>
              In a {data.tokens_scanned ? `${fmtInt(data.tokens_scanned)}-token` : "token"} sample
              {data.min_count ? `, pairs with at least ${fmtInt(data.min_count)} shared tokens` : ""}.
            </p>
            {data.partners.length === 0 ? (
              <p className={s.note}>No feature fired on enough of the same tokens.</p>
            ) : (
              <ol className={s.cofire}>
                <li className={`${s.cofireRow} ${s.cofireHead}`} aria-hidden="true">
                  <span />
                  <span />
                  <span>shared</span>
                  <span>share</span>
                </li>
                {data.partners.map((p) => (
                  <li key={p.feature_id} className={s.cofireRow}>
                    <span className={`mono ${s.cofireId}`}>#{p.feature_id}</span>
                    {/* link a partner only when this server has it */}
                    {p.in_dictionary ? (
                      <Link href={`/features/${p.feature_id}`} className={s.cofireName}>
                        {featureName(p.label, p.feature_id)}
                      </Link>
                    ) : (
                      <span className={s.cofireName}>{featureName(p.label, p.feature_id)}</span>
                    )}
                    <span className="mono" title="Tokens in the sample where both fired">
                      {fmtInt(p.co_count)}
                    </span>
                    <span className="mono muted" title="Shared tokens out of the tokens where either fired (Jaccard)">
                      {p.jaccard.toFixed(2)}
                    </span>
                  </li>
                ))}
              </ol>
            )}
            <p className={s.note}>
              Shared: tokens where both fired. Share: those tokens out of all where either fired. Firing on the same tokens does not mean they share a
              meaning, and the evidence is thinner for rare features.
            </p>
          </>
        )}
      </div>
    </section>
  );
}
