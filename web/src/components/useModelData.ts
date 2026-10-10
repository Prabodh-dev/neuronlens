"use client";

import { autolabPresets, type AutoLabPresets } from "@/lib/autolab-presets";
import { modelData, type ModelData } from "@/lib/model-data";
import { useModels } from "./ModelProvider";

/** The current model's picks and recordings, or null until the page knows its model (see ModelProvider). */
export function useModelData(): ModelData | null {
  const { current } = useModels();
  return current ? modelData(current.id) : null;
}

/** The current model's Auto-Lab prompts and goals, or null until the page knows its model. */
export function useAutoLabPresets(): AutoLabPresets | null {
  const { current } = useModels();
  return current ? autolabPresets(current.id) : null;
}
