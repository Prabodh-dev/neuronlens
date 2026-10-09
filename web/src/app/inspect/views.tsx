"use client";

import Link from "next/link";
import type { ReportResponse, TextCompareResponse } from "@/lib/api/types";
import { featureName, fmtNum } from "@/lib/format";
import s from "./inspect.module.css";

/* Phase 4 views on the Inspect page: X6 the most active features in each sentence, X7 the features that
   differ most between two texts. Strength bars are relative_to_max (activation / the feature's peak in
   the dictionary); the order is the server's: raw activation, or share of its peak with "Rank by". */

const peakShare = (v: number | null | undefined) => (v === null || v === undefined ? null : Math.max(0, v));
const pctOfPeak = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${Math.round(v * 100)}%`);

type Rank = "activation" | "relative";

/** The ranking note under a result: how it is ranked (sort) and what was hidden (max_density). */
function rankNote(cut: number | null, rank: Rank, compare = false) {
  const by =
    rank === "relative"
      ? compare
        ? "Ranked by the change in share of each feature's own peak; the difference shown is raw activation."
        : "Ranked by share of each feature's own peak."
      : cut === null
        ? "Ranked by raw activation, so word-piece and common-word features often come first."
        : "Ranked by raw activation.";
  return cut === null ? by : `${by} Features that fire on more than ${+(cut * 100).toFixed(2)}% of all tokens are hidden.`;
}

function Bar({ v }: { v: number | null | undefined }) {
  const x = peakShare(v);
  return (
    <span className={`bar heat ${s.peakBar}`} title={x === null ? "No peak on record" : `${pctOfPeak(v)} of its peak`}>
      <i style={{ width: `${x === null ? 0 : Math.max(2, Math.min(1, x) * 100)}%` }} />
    </span>
  );
}

/** X6: sentence by sentence, the most active features in it. */
export function ReportView({ res, busy, hidCommon, rank }: { res: ReportResponse; busy: boolean; hidCommon: number | null; rank: Rank }) {
  return (
    <section className={`card ${busy ? s.dim : ""}`} aria-labelledby="rep-h" style={{ marginTop: 16 }}>
      <div className="card-head">
        <h2 id="rep-h">
          {res.sentences.length} sentence{res.sentences.length === 1 ? "" : "s"} · most active features in each
        </h2>
        {res.simulated && <span className="pill pill-sim">Simulated</span>}
      </div>
      <ol className={s.sentences}>
        {res.sentences.map((sn) => (
          <li key={sn.index} className={s.sentence}>
            <div className={s.sentHead}>
              <span className="mono muted small">{String(sn.index + 1).padStart(2, "0")}</span>
              <p className={s.sentText}>{sn.text}</p>
            </div>
            {sn.features.length === 0 ? (
              <p className="small muted">No features were active in this sentence.</p>
            ) : (
              <ol className={s.repFeats}>
                <li className={s.repLegend} aria-hidden="true">
                  <span />
                  <span>of its peak</span>
                  <span>tokens</span>
                </li>
                {sn.features.map((f) => (
                  <li key={f.feature_id} className={s.repFeat}>
                    <Link href={`/features/${f.feature_id}`} className={s.repName}>
                      <span className="mono muted">#{f.feature_id}</span> {featureName(f.label, f.feature_id)}
                    </Link>
                    <span className={s.repBar}>
                      <Bar v={f.relative_to_max} />
                      <span className="mono small">{pctOfPeak(f.relative_to_max)}</span>
                    </span>
                    <span className="mono small muted">{f.tokens_active}</span>
                  </li>
                ))}
              </ol>
            )}
          </li>
        ))}
      </ol>
      <div className={s.foot}>
        <span className="small muted">
          {rankNote(hidCommon, rank)} Sentences split at . ! ? and new lines; abbreviations such as “U.S.” can split early. It shows which features were active,
          not what the sentence means.
        </span>
      </div>
    </section>
  );
}

/** X7: the features whose activation differs most between text A and text B. */
export function CompareView({ res, busy, hidCommon, rank }: { res: TextCompareResponse; busy: boolean; hidCommon: number | null; rank: Rank }) {
  return (
    <section className={`card ${busy ? s.dim : ""}`} aria-labelledby="tc-h" style={{ marginTop: 16 }}>
      <div className="card-head">
        <h2 id="tc-h">Features that differ most between A and B</h2>
        <div className="row" style={{ gap: 8 }}>
          <span className="mono muted small">
            A {res.tokens_a} tokens · B {res.tokens_b} tokens
          </span>
          {res.simulated && <span className="pill pill-sim">Simulated</span>}
        </div>
      </div>
      {res.features.length === 0 ? (
        <p className={s.skipNote} style={{ padding: "16px 20px" }}>
          No feature differs between the two texts.
        </p>
      ) : (
        <ol className={s.cmpList}>
          <li className={s.cmpLegend} aria-hidden="true">
            <span />
            <span>in A</span>
            <span>in B</span>
            <span>difference</span>
          </li>
          {res.features.map((f) => (
            <li key={f.feature_id} className={s.cmpRow}>
              <Link href={`/features/${f.feature_id}`} className={s.repName}>
                <span className="mono muted">#{f.feature_id}</span> {featureName(f.label, f.feature_id)}
              </Link>
              <span className={s.repBar}>
                <Bar v={f.relative_to_max_a} />
                <span className="mono small">{pctOfPeak(f.relative_to_max_a)}</span>
              </span>
              <span className={s.repBar}>
                <Bar v={f.relative_to_max_b} />
                <span className="mono small">{pctOfPeak(f.relative_to_max_b)}</span>
              </span>
              <span className={`mono small ${s.cmpDiff}`}>
                {f.difference >= 0 ? "more in B" : "more in A"} {f.difference >= 0 ? "+" : "−"}
                {fmtNum(Math.abs(f.difference), 1)}
              </span>
            </li>
          ))}
        </ol>
      )}
      <div className={s.foot}>
        <span className="small muted">
          Compares each feature&apos;s highest activation in A and in B (bars: share of its peak), not what the texts mean. {rankNote(hidCommon, rank, true)}
        </span>
      </div>
    </section>
  );
}
