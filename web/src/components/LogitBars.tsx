import type { LogitEntry } from "@/lib/api/types";
import { fmtNum, showToken } from "@/lib/format";
import s from "./LogitBars.module.css";

const shown = (t: string) => showToken(t);

/** One diverging chart: tokens the feature pushes toward grow right, tokens it pushes away grow left. */
export function LogitBars({ promoted, suppressed }: { promoted: LogitEntry[]; suppressed: LogitEntry[] }) {
  if (!promoted.length && !suppressed.length) return <p className="muted small">None recorded.</p>;
  const max = Math.max(...promoted.map((e) => Math.abs(e.score)), ...suppressed.map((e) => Math.abs(e.score)), 1e-9);
  const rows = Math.max(promoted.length, suppressed.length);
  return (
    <div className={s.chart}>
      <div className={s.head}>
        <span>away from</span>
        <span>toward</span>
      </div>
      {Array.from({ length: rows }, (_, i) => {
        const d = suppressed[i];
        const u = promoted[i];
        return (
          <div key={i} className={s.row}>
            <span className={s.tokL} title={d ? JSON.stringify(d.token) : undefined}>
              {d ? shown(d.token) : ""}
            </span>
            <span className={s.left}>{d && <i style={{ width: `${(Math.abs(d.score) / max) * 100}%` }} title={fmtNum(d.score)} />}</span>
            <span className={s.axis} />
            <span className={s.right}>{u && <i style={{ width: `${(Math.abs(u.score) / max) * 100}%` }} title={fmtNum(u.score)} />}</span>
            <span className={s.tokR} title={u ? JSON.stringify(u.token) : undefined}>
              {u ? shown(u.token) : ""}
            </span>
          </div>
        );
      })}
    </div>
  );
}
