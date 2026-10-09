"use client";

import { api } from "@/lib/api/client";
import { t } from "@/lib/copy";
import { fmtInt, fmtPct } from "@/lib/format";
import { useApi } from "@/lib/useApi";
import { useHealth } from "./HealthProvider";
import s from "./LimitsBox.module.css";

/** X5: what NeuronLens does not do. Each line is a copy-deck note or a number read from the server;
 *  a line whose number is missing is left out rather than shown with a guess. */
export function LimitsBox({ id = "limits-h" }: { id?: string }) {
  const { data: h } = useHealth();
  const metrics = useApi((sig) => api.metrics(null, sig), "limits:metrics");
  const named = useApi((sig) => api.features({ status: "named", limit: 1 }, sig), "limits:named");
  const val = (name: string) => metrics.data?.metrics.find((m) => m.name === name)?.value ?? null;
  const fvu = val("fvu");
  const ce = val("ce_recovered");

  const items: { k: string; v: string }[] = [
    { k: "Active, not why", v: t("note.limit") },
    { k: "A test, not a fix", v: t("note.steering") },
  ];
  if (h && fvu !== null && ce !== null)
    items.push({
      k: "Not a perfect copy",
      v: `Swapping layer ${h.hook_layer} for the features keeps ${fmtPct(ce * 100)} of next-word ability, not 100%. ${fmtPct(fvu * 100)} of the layer's variance stays unexplained.`,
    });
  if (h)
    items.push({
      k: "One layer, one small model",
      v: `Only layer ${h.hook_layer} of ${h.model_name}. Other layers and larger models are not covered.`,
    });
  if (h && named.data)
    items.push({
      k: "Most features have no name",
      v: `${fmtInt(named.data.total)} of ${fmtInt(h.d_sae)} have one. A name sums up a feature's top examples and can be wrong.`,
    });

  return (
    <section className={s.box} aria-labelledby={id}>
      <div className={s.head}>
        <span className="eyebrow">Limits</span>
        <h2 id={id} className={s.title}>
          What it does not do.
        </h2>
      </div>
      <ol className={s.list}>
        {items.map((it, i) => (
          <li key={it.k} className={s.item}>
            <span className={s.n}>{String(i + 1).padStart(2, "0")}</span>
            <b>{it.k}</b>
            <p>{it.v}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}
