"use client";

import { useEffect, useRef } from "react";
import { fmtInt } from "@/lib/format";
import { modelLabel } from "@/lib/models";
import { chooseModel, useChoices, useModels } from "./ModelProvider";
import s from "./ModelChooser.module.css";

/** The "experimental" tag next to a model's name. */
export function ExperimentalTag() {
  return <span className={s.exp}>experimental</span>;
}

/** Whether there is a model to choose between: more than one model's server answers /health (or the
 *  current one does not, and another does). With one server the app looks as it did before. */
export function useCanChoose(): boolean {
  return useChoices().length >= 2;
}

/** The model chooser (docs/features/ModelChooser.md), opened from the status pill in the top bar. Each
 *  model's name, layer and size come from its own /health. Choosing one reloads the page on it
 *  (?model=<id>). */
export function ModelMenu({ onClose }: { onClose: () => void }) {
  const { current } = useModels();
  const choices = useChoices();
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    const down = (e: PointerEvent) => {
      // a click on the pill itself toggles the menu there; anywhere else closes it
      if (el && !el.parentElement?.contains(e.target as Node)) onClose();
    };
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("pointerdown", down);
    document.addEventListener("keydown", key);
    el?.querySelector<HTMLButtonElement>('[aria-checked="true"]')?.focus();
    return () => {
      document.removeEventListener("pointerdown", down);
      document.removeEventListener("keydown", key);
    };
  }, [onClose]);

  return (
    <div ref={ref} className={s.menu} role="menu" aria-label="Run every page on">
      <span className={s.head}>Run every page on</span>
      {choices.map(({ entry, health }) => {
        const on = entry.id === current?.id;
        return (
          <button
            key={entry.id}
            type="button"
            role="menuitemradio"
            aria-checked={on}
            className={s.item}
            disabled={!on && !health}
            onClick={() => (on ? onClose() : chooseModel(entry))}
          >
            <span className={s.check} aria-hidden="true">
              {on ? "✓" : ""}
            </span>
            <span className={s.main}>
              <span className={s.nameRow}>
                <span className={s.name}>{health ? modelLabel(health.model_name) : entry.id}</span>
                {entry.experimental && <ExperimentalTag />}
              </span>
              <span className={s.sub}>
                {health
                  ? `layer ${health.hook_layer} · ${fmtInt(health.d_sae)} features · ${health.is_synthetic ? "sample data" : health.mode === "mock" ? "mock" : "live"}`
                  : "not responding"}
              </span>
            </span>
          </button>
        );
      })}
    </div>
  );
}
