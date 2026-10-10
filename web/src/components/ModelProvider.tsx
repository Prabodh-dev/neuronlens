"use client";

import { usePathname, useSearchParams } from "next/navigation";
import { createContext, Suspense, useContext, useEffect, useState } from "react";
import { probeHealth } from "@/lib/api/client";
import type { Health } from "@/lib/api/types";
import { DEFAULT_MODEL, MODELS, currentModel, withModel, type ModelEntry } from "@/lib/models";

/** One model of lib/models.ts and its /health: undefined while asking, null when it did not answer. */
export type Probe = { entry: ModelEntry; health: Health | null | undefined };
type Models = { current: ModelEntry | null; probes: Probe[] };

const ModelContext = createContext<Models | null>(null);

/** Which model this page runs on, and which models answer. `current` is null until the page has loaded
 *  in the browser: pages are prerendered without the address, so anything that depends on the model
 *  (its presets and recorded examples) waits for it rather than showing the default model's first. */
export function ModelProvider({ children }: { children: React.ReactNode }) {
  const [current, setCurrent] = useState<ModelEntry | null>(null);
  const [probes, setProbes] = useState<Probe[]>(() => MODELS.map((entry) => ({ entry, health: undefined })));

  useEffect(() => {
    setCurrent(currentModel());
    let live = true;
    MODELS.forEach((m) =>
      probeHealth(m).then((health) => live && setProbes((ps) => ps.map((p) => (p.entry.id === m.id ? { ...p, health } : p)))),
    );
    return () => {
      live = false;
    };
  }, []);

  return (
    <ModelContext.Provider value={{ current, probes }}>
      {children}
      <Suspense fallback={null}>
        <KeepModelInUrl />
      </Suspense>
    </ModelContext.Provider>
  );
}

/** Links inside the app do not carry ?model=, and some pages rewrite their own query. For a model
 *  other than the default this puts ?model= back into the address after every change, so the address
 *  is always one that reopens the same model. */
function KeepModelInUrl() {
  const path = usePathname();
  const params = useSearchParams();
  useEffect(() => {
    const m = currentModel();
    if (m.id === DEFAULT_MODEL.id || params.get("model") === m.id) return;
    window.history.replaceState(window.history.state, "", withModel(new URL(window.location.href), m).toString());
  }, [path, params]);
  return null;
}

export function useModels(): Models {
  const ctx = useContext(ModelContext);
  if (!ctx) throw new Error("useModels needs <ModelProvider>");
  return ctx;
}

/** The models to offer: the ones that answered, plus the current one even when it did not (so there is
 *  a way back). Empty while the current model is unknown. */
export function useChoices(): Probe[] {
  const { current, probes } = useModels();
  if (!current) return [];
  return probes.filter((p) => p.health || p.entry.id === current.id);
}

/** Opens the same page on another model. The page reloads, so nothing from the old model stays in
 *  memory; page settings are cleared because feature numbers mean different things on each model. */
export function chooseModel(m: ModelEntry) {
  const path = window.location.pathname.startsWith("/features/") ? "/features" : window.location.pathname;
  window.location.assign(withModel(new URL(path, window.location.origin), m).toString());
}
