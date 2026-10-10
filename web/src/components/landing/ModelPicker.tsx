"use client";

import { useState } from "react";
import { fmtInt } from "@/lib/format";
import { modelInfo } from "@/lib/model-info";
import { modelLabel } from "@/lib/models";
import { isMockModel, isSampleData } from "../HealthProvider";
import { ExperimentalTag } from "../ModelChooser";
import { pickModel, useModels, type Probe } from "../ModelProvider";
import { ModelSphere } from "./ModelSphere";
import s from "./modelPicker.module.css";

/** The model selector (the home page's first screen when two or more models answer, and /models):
 *  one turning sphere per model, each with a description and the details its server reports.
 *  Choosing one opens the landing page on that model (?model=<id>). `currentId` marks the model the
 *  page is running on, when the selector is opened from inside one. */
export function ModelPicker({ currentId }: { currentId?: string }) {
  const { probes } = useModels();
  return (
    <section className={s.wrap} aria-labelledby="pick-title">
      <header className={s.head}>
        <span className="eyebrow">NeuronLens</span>
        <h1 id="pick-title" className={s.title}>
          Pick a model.
        </h1>
        <p className={s.lead}>Every page runs on the model you pick. You can come back here from Models in the top bar.</p>
      </header>
      <div className={s.grid}>
        {probes.map((p, i) => (
          <Choice key={p.entry.id} probe={p} index={i} current={p.entry.id === currentId} />
        ))}
      </div>
    </section>
  );
}

function Choice({ probe, index, current }: { probe: Probe; index: number; current: boolean }) {
  const { entry, health: h } = probe;
  const info = modelInfo(entry.id);
  const [hot, setHot] = useState(false);
  const rows: [string, string][] = [];
  if (h) {
    if (h.hook_layer != null) rows.push(["Layer", String(h.hook_layer)]);
    if (h.d_sae != null) rows.push(["Features", fmtInt(h.d_sae)]);
    if (h.k != null) rows.push(["Fire per token", String(h.k)]);
    if (h.d_model != null) rows.push(["Width", fmtInt(h.d_model)]);
    rows.push(["Data", isSampleData(h) ? "Sample data" : isMockModel(h) ? "Mock" : "Live"]);
  }
  return (
    <button
      type="button"
      className={s.choice}
      style={{ animationDelay: `${0.1 * index}s` }}
      disabled={!h}
      aria-current={current ? "true" : undefined}
      onClick={() => pickModel(entry)}
      onPointerEnter={() => setHot(true)}
      onPointerLeave={() => setHot(false)}
      onFocus={() => setHot(true)}
      onBlur={() => setHot(false)}
    >
      <span className={s.sphere}>
        <ModelSphere dashed={entry.experimental} fast={hot} />
      </span>
      <span className={s.tagRow}>
        <span className="eyebrow">Model {String(index + 1).padStart(2, "0")}</span>
        {entry.experimental && <ExperimentalTag />}
        {current && <span className={s.here}>You are here</span>}
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
        {current ? "Open its landing page" : "Open"} <span aria-hidden="true">→</span>
      </span>
    </button>
  );
}
