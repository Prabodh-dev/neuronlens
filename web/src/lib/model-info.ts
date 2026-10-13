/* What the model picker says about each model (docs/features/model-chooser.md): a short description and
   what the pages do on it. Keyed by model id (lib/models.ts), like lib/model-data.ts. No number is written
   here: names, layers and sizes come from the server's /health. Nothing here ranks or compares models. */

export type ModelInfo = { summary: string; points: string[] };

const INFO: Record<string, ModelInfo> = {
  gpt2: {
    summary: "A small language model with a sparse autoencoder on one layer. Browse its features, inspect text and test what a feature does by steering it.",
    points: ["Features are named and reviewed.", "Steer, Combine and Auto-Lab have examples recorded on it."],
  },
  neo: {
    summary: "A second language model with its own sparse autoencoder and its own database. The same pages run on it.",
    points: ["Experimental.", "No examples are recorded on it yet, so Inspect, Steer and Auto-Lab run live."],
  },
};

const OTHER: ModelInfo = {
  summary: "A language model with its own sparse autoencoder and its own database. The same pages run on it.",
  points: ["No examples are recorded on it yet, so Inspect, Steer and Auto-Lab run live."],
};

export function modelInfo(id: string): ModelInfo {
  return INFO[id] ?? OTHER;
}
