/* Auto-Lab presets: the prompt and goal buttons on the Auto-Lab page.
   Each goal is a label, the words sent to the search (POST /features/search) and a single-colour
   symbol for its button (text style, never colour emoji, to match the site). Trim either list to what ships; the screen shows whatever is here. */

export type AutoLabGoal = { label: string; words: string; icon?: string };

export const AUTOLAB_PROMPTS: string[] = ["The band walked on stage and", "The man walked into the room and", "She opened the door and"];

// all 6 goals ship after the laptop test (Prabodh); the words are the ones recorded in
// demo/autolab_examples.json, so the mock replays the real runs for them
export const AUTOLAB_GOALS: AutoLabGoal[] = [
  { label: "Music", words: "music", icon: "♪" },
  { label: "Courts and law", words: "court law", icon: "⚖\uFE0E" },
  { label: "Space", words: "space planet", icon: "✦" },
  { label: "Food", words: "cooking food", icon: "♨\uFE0E" },
  { label: "Weather", words: "weather rain", icon: "☂\uFE0E" },
  { label: "Sports", words: "sports game", icon: "⚽\uFE0E" },
];
