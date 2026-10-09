"use client";

import { useHealth } from "./HealthProvider";

/** "gpt2 · layer 8", read from /health so no model size is hard-coded. */
export function ModelTag() {
  const { data: h } = useHealth();
  if (!h) return null;
  return (
    <span>
      {h.model_name} · layer {h.hook_layer}
    </span>
  );
}
