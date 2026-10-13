/* Auto-Lab presets, per model (lib/models.ts): the prompt and goal buttons on the Auto-Lab page, and
   what the progress screen says a full run takes. Each goal is a label, the words sent to the search
   (POST /features/search) and a single-colour symbol for its button (text style, never colour emoji, to
   match the site). They were tested on one model's server, so a model without an entry gets no buttons:
   the page asks for a typed prompt and goal and runs them live. */

export type AutoLabGoal = { label: string; words: string; icon?: string };
export type AutoLabPresets = {
  prompts: string[];
  goals: AutoLabGoal[];
  /** "about N seconds" on the progress screen, from timed runs; null when this model was not timed. Never
   *  lower than real runs take; lower it only after timing again. */
  etaSeconds: number | null;
  /** Whether the ✓ thresholds (WORKED in autolab.ts) were tuned on this model's runs. */
  tuned: boolean;
};

// the three prompts the second model shares with the first (they are inputs, not recordings)
const PROMPTS = ["The band walked on stage and", "The man walked into the room and", "She opened the door and"];

const PRESETS: Record<string, AutoLabPresets> = {
  gpt2: {
    prompts: PROMPTS,
    // all 6 goals ship; the words are the ones recorded in
    // demo/autolab_examples.json, so the mock replays the real runs for them
    goals: [
      { label: "Music", words: "music", icon: "♪" },
      { label: "Courts and law", words: "court law", icon: "⚖︎" },
      { label: "Space", words: "space planet", icon: "✦" },
      { label: "Food", words: "cooking food", icon: "♨︎" },
      { label: "Weather", words: "weather rain", icon: "☂︎" },
      { label: "Sports", words: "sports game", icon: "⚽︎" },
    ],
    // timed UI runs took 20 to 25 s (the check script, about 14 s)
    etaSeconds: 20,
    tuned: true,
  },
  // the second model (experimental): Auto-Lab was checked on six goals there and three are offered. No
  // runs are recorded for it, so every run is live; not yet timed in the app, so the estimate is a round
  // figure above the checked runs. The ✓ thresholds were tuned on GPT-2 runs.
  neo: {
    prompts: PROMPTS,
    goals: [
      { label: "Music", words: "music", icon: "♪" },
      { label: "Courts and law", words: "court law", icon: "⚖︎" },
      { label: "Weather", words: "weather rain", icon: "☂︎" },
    ],
    etaSeconds: 30,
    tuned: false,
  },
};

const NONE: AutoLabPresets = { prompts: [], goals: [], etaSeconds: null, tuned: false };

export function autolabPresets(id: string): AutoLabPresets {
  return PRESETS[id] ?? NONE;
}
