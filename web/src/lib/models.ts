/* The models NeuronLens can run on (docs/features/ModelChooser.md). Each model is its own server with
   its own database, and they all answer the same endpoints, so a page only changes which server it
   calls. The name shown on screen always comes from that server's /health (model_name, hook_layer),
   never from this list: `label` is only for people reading this file. */

export type ModelEntry = {
  /** The value of ?model= in the address. */
  id: string;
  label: string;
  /** The server's origin, without /api/v1. */
  apiBase: string;
  /** Shown with an "experimental" tag next to its name. */
  experimental: boolean;
};

// NEXT_PUBLIC_API_BASE, when set, still points the first model somewhere else, as before the chooser
const envBase = process.env.NEXT_PUBLIC_API_BASE?.replace(/\/+$/, "").replace(/\/api\/v\d+$/, "");

export const MODELS: readonly ModelEntry[] = [
  { id: "gpt2", label: "GPT-2 small", apiBase: envBase || "http://127.0.0.1:8000", experimental: false },
  { id: "neo", label: "Second model", apiBase: "http://127.0.0.1:8001", experimental: true },
];

/** The model a link without ?model= opens, so existing links and the demo stay as they were. */
export const DEFAULT_MODEL: ModelEntry = MODELS[0];

let chosen: ModelEntry | null = null;

/** The model this page load runs on: ?model=<id> in the address it was opened with, else the default
 *  (an unknown id gets the default too). It stays fixed until the page reloads, and choosing another
 *  model reloads the page, so one page never mixes two servers' answers. On the server (prerender) it
 *  is the default; nothing there calls the API. */
export function currentModel(): ModelEntry {
  if (chosen) return chosen;
  if (typeof window === "undefined") return DEFAULT_MODEL;
  const id = new URLSearchParams(window.location.search).get("model");
  chosen = MODELS.find((m) => m.id === id) ?? DEFAULT_MODEL;
  return chosen;
}

/** The base every API call of this model uses. */
export const apiV1 = (m: ModelEntry): string => `${m.apiBase}/api/v1`;

/** Sets ?model= on an app URL for a model other than the default, and removes it for the default. */
export function withModel(url: URL, m: ModelEntry = currentModel()): URL {
  if (m.id === DEFAULT_MODEL.id) url.searchParams.delete("model");
  else url.searchParams.set("model", m.id);
  return url;
}
