"use client";

import { useHealth } from "./HealthProvider";
import { ExperimentalTag } from "./ModelChooser";
import { useModels } from "./ModelProvider";

/** The model's name and layer, read from /health so neither is hard-coded, and the "experimental" tag
 *  for a model marked so in lib/models.ts. */
export function ModelTag() {
  const { data: h } = useHealth();
  const { current } = useModels();
  if (!h) return null;
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
      {h.model_name} · layer {h.hook_layer}
      {current?.experimental && <ExperimentalTag />}
    </span>
  );
}
