"use client";

import { fmtInt } from "@/lib/format";
import { modelInfo } from "@/lib/model-info";
import { modelLabel } from "@/lib/models";
import { isMockModel, isSampleData } from "../HealthProvider";
import { ExperimentalTag } from "../ModelChooser";
import { pickModel, useModels, type Probe } from "../ModelProvider";
import s from "./modelPicker.module.css";

/** The first screen when more than one model answers: one card per model with a description and the
 *  details its server reports. Choosing one opens the landing page on that model (?model=<id>). */
export function ModelPicker() {
  const { probes } = useModels();
  return (
    <section className={s.wrap} aria-labelledby="pick-title">
      <header className={s.head}>
        <span className="eyebrow">NeuronLens</span>
        <h1 id="pick-title" className={s.title}>
          Pick a model.
        </h1>
        <p className={s.lead}>Every page runs on the model you pick. You can switch later from the status pill in the top bar.</p>
      </header>
      <div className={s.grid}>
        {probes.map((p, i) => (
          <Card key={p.entry.id} probe={p} index={i} />
        ))}
      </div>
    </section>
  );
}

function Card({ probe, index }: { probe: Probe; index: number }) {
  const { entry, health: h } = probe;
  const info = modelInfo(entry.id);
  const rows: [string, string][] = [];
  if (h) {
    if (h.hook_layer != null) rows.push(["Layer", String(h.hook_layer)]);
    if (h.d_sae != null) rows.push(["Features", fmtInt(h.d_sae)]);
    if (h.k != null) rows.push(["Fire per token", String(h.k)]);
    if (h.d_model != null) rows.push(["Width", fmtInt(h.d_model)]);
    rows.push(["Data", isSampleData(h) ? "Sample data" : isMockModel(h) ? "Mock" : "Live"]);
  }
  return (
    <button type="button" className={s.card} style={{ animationDelay: `${0.08 * index}s` }} disabled={!h} onClick={() => pickModel(entry)}>
      <span className={s.tagRow}>
        <span className="eyebrow">Model {String(index + 1).padStart(2, "0")}</span>
        {entry.experimental && <ExperimentalTag />}
      </span>
      <span className={s.name}>{h ? modelLabel(h.model_name) : entry.id}</span>
      <span className={s.summary}>{info.summary}</span>
      <ul className={s.points}>
        {info.points.map((t) => (
          <li key={t}>{t}</li>
        ))}
      </ul>
      {h ? (
        <dl className={s.facts}>
          {rows.map(([k, v]) => (
            <div key={k}>
              <dt>{k}</dt>
              <dd>{v}</dd>
            </div>
          ))}
        </dl>
      ) : (
        <span className={s.off}>not responding</span>
      )}
      <span className={s.open}>
        Open <span aria-hidden="true">→</span>
      </span>
    </button>
  );
}
