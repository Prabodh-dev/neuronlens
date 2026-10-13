"use client";

import { fmtInt } from "@/lib/format";
import { modelLabel } from "@/lib/models";
import { useHealth } from "./HealthProvider";
import { ExperimentalTag } from "./ModelChooser";
import { useModels } from "./ModelProvider";

/** The one line shown for an experimental model (docs/features/model-chooser.md). */
export const EXPERIMENTAL_NOTE = "Smaller training run than GPT-2, so expect weaker features.";

/** "Model: <model_name> · layer <layer> · <d_sae> features · k=<k>", every part read from /health so
 *  nothing is hard-coded; a part the server did not send is left out. A model marked experimental in
 *  lib/models.ts also gets the "experimental" tag and one line under it. */
export function ModelTag() {
  const { data: h } = useHealth();
  const { current } = useModels();
  if (!h) return null;
  const parts = [
    modelLabel(h.model_name),
    h.hook_layer != null && `layer ${h.hook_layer}`,
    h.d_sae != null && `${fmtInt(h.d_sae)} features`,
    h.k != null && `k=${h.k}`,
  ].filter(Boolean);
  return (
    <span style={{ display: "inline-flex", flexDirection: "column", alignItems: "flex-end", gap: 4 }}>
      <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
        Model: {parts.join(" · ")}
        {current?.experimental && <ExperimentalTag />}
      </span>
      {current?.experimental && <span>{EXPERIMENTAL_NOTE}</span>}
    </span>
  );
}
