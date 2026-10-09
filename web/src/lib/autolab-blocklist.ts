/* Words a typed Auto-Lab goal may not contain. A match shows "Try another word" and runs nothing.
   WORDS block that exact word; STEMS also block every word that starts with them ("fuck" blocks
   "fucking"). Stems are only for words with no harmless longer forms ("cock" is a word, not a stem,
   so "cocktail" is fine). Keep it small; add words here only. */

const WORDS = [
  // swearing
  "ass", "arse", "asshole", "bastard", "bollocks", "crap", "dick", "cock", "piss", "pussy", "twat",
  // sexual
  "sex", "sexy", "nude", "nudes", "naked", "boob", "boobs", "tits", "penis", "vagina", "orgasm", "horny", "rape", "raped",
  // slurs
  "fag", "spic", "chink", "kike", "tranny", "dyke", "paki", "wetback",
  // violence, self-harm, hate
  "kill", "kills", "bomb", "bombs",
];

const STEMS = [
  "fuck", "shit", "bitch", "cunt", "wank", "porn",
  "nigg", "fagg", "retard",
  "rapist", "murder", "suicid", "genocid", "massacr", "tortur", "terroris", "nazi", "hitler",
];

/** True when any word of `text` is blocked. */
export function isBlockedGoal(text: string): boolean {
  const words = text.toLowerCase().split(/\s+/).map((w) => w.replace(/[^a-z]/g, "")).filter(Boolean);
  return words.some((w) => WORDS.includes(w) || STEMS.some((s) => w.startsWith(s)));
}
