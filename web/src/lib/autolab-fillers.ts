import { currentModel, MODELS } from "./models";

/* Filler features: they come back from the search for many unrelated words (several fire on the first
   query word, whatever it is), so Auto-Lab and the Features page ("Your words") drop them. Each model has
   its own list, measured on its server with server/scripts/autolab_check.py --find-fillers and kept in
   demo/autolab_fillers.json (the default model) or demo/autolab_fillers_<name>.json, where <name> is the
   model's `fillers` entry in lib/models.ts (else its id); scripts/copy-fillers.mjs copies them unchanged
   to public/autolab-fillers/<name>.json, which is what this loads. A model without a
   list gets null: nothing is dropped, and the screen says it has no filler list yet. */

export type Fillers = ReadonlySet<number>;

const cache = new Map<string, Promise<Fillers | null>>();

/** The filler ids of a model (default: this page's), or null when it has no filler list yet. */
export function loadFillers(id: string = currentModel().id): Promise<Fillers | null> {
  let p = cache.get(id);
  if (!p) {
    const name = MODELS.find((m) => m.id === id)?.fillers ?? id;
    p = fetch(`/autolab-fillers/${encodeURIComponent(name)}.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d: { fillers?: { feature_id: number }[] } | null) => (Array.isArray(d?.fillers) ? new Set(d.fillers.map((f) => f.feature_id)) : null))
      .catch(() => null);
    cache.set(id, p);
  }
  return p;
}

export function dropFillers<T extends { feature_id: number }>(features: T[], fillers: Fillers | null): T[] {
  return fillers ? features.filter((f) => !fillers.has(f.feature_id)) : features;
}
