import data from "./autolab-fillers.json";

/* Features that come back from the search for many unrelated words (several fire on the first query
   word, whatever it is), so they are dropped from the search results in Auto-Lab and on the Features
   page ("Your words"). autolab-fillers.json is an unchanged copy of demo/autolab_fillers.json, which
   Mirdul's server/scripts/autolab_check.py --find-fillers measured on Prabodh's laptop (200 common
   words, kept if they came back for more than 10% of them). Copy it again whenever it is regenerated;
   the script and the mock read the same ids. */
export const FILLERS: ReadonlySet<number> = new Set(data.fillers.map((f) => f.feature_id));

export function dropFillers<T extends { feature_id: number }>(features: T[]): T[] {
  return features.filter((f) => !FILLERS.has(f.feature_id));
}
