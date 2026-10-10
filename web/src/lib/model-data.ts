import type { SteerMode } from "./api/types";

/* Everything the app shows that was picked or recorded on one model, keyed by model id (lib/models.ts).
   Feature numbers mean different things on each model, so a model without an entry here gets none of
   another model's picks: the pages fall back to live runs or an empty state. Auto-Lab's prompts and
   goals are in autolab-presets.ts, its filler lists in public/autolab-fillers/<id>.json. */

export type SteerDemo = { id: number; prompt: string; mode: SteerMode; strength: number };
export type SteerRun = { feature_id: number; mode: SteerMode; strength: number; prompt: string; seed: number; max_new_tokens: number };
/** `note`: shown under the result of this pair only. */
export type CombinePair = { ids: number[]; strength: number; prompt: string; seed: number; maxTokens: number; note: string };

export type ModelData = {
  /** Home page "one neuron, many ideas": features shown when the database has them. */
  ideas: number[];
  /** The landing tour: features it visits first, and the steering run of its "Steering" chapter. */
  tourStops: number[];
  tourSteer: SteerRun | null;
  /** The Steer page's one-click runs. */
  steerDemos: SteerDemo[];
  /** The Combine tab's one-click pair, found by an automated search. */
  combinePair: CombinePair | null;
};

const DATA: Record<string, ModelData> = {
  gpt2: {
    ideas: [1863, 3587, 3446, 9558, 4289],
    tourStops: [1863, 3587, 3446],
    // checked on the GPU (demo/phase4_examples.json): #3446 "music" up 0.5 on this prompt reads 0.00
    // to 12.44; at 1.0 it drifts, so the tour uses 0.5
    tourSteer: { feature_id: 3446, mode: "amplify", strength: 0.5, prompt: "The band walked on stage and", seed: 42, max_new_tokens: 60 },
    // the demo runs checked on the GPU (demo/phase4_examples.json, seed 42); a click reruns them live.
    // #1863 stays a recorded example only: it depends on the prompt and touches a hot-button topic.
    steerDemos: [
      { id: 3446, prompt: "The band walked on stage and", mode: "amplify", strength: 0.5 },
      { id: 13, prompt: "The acrobat hung", mode: "amplify", strength: 0.5 },
    ],
    // the recorded pair (demo/phase4_examples.json, "multi")
    combinePair: {
      ids: [4280, 9530],
      strength: 0.5,
      prompt: "The man walked into the room and",
      seed: 42,
      maxTokens: 60,
      note: "This pair was picked by a search from 100 pairs, and it depends on the prompt: on “The woman looked at him and” only one of the two went up. ",
    },
  },
};

const NONE: ModelData = { ideas: [], tourStops: [], tourSteer: null, steerDemos: [], combinePair: null };

/** The picks and recordings for a model; empty for a model nothing was recorded on. */
export function modelData(id: string): ModelData {
  return DATA[id] ?? NONE;
}
