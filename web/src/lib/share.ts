import type { SteerMode } from "./api/types";
import { withModel } from "./models";

/* S5: shareable links. Everything a result needs travels in the URL, with nothing stored on the server.
   Generation is seeded, so the same feature, prompt, strength, seed, mode and length on the same
   server and checkpoint give the same result; `run` records which run the link was made on, and
   `model` which model (lib/models.ts), left out for the default model so older links read the same. */

export type SteerLink = { feature_id: number; mode: SteerMode; strength: number; seed: number; prompt: string; max_new_tokens: number };

export const LENGTHS = [30, 60, 90, 120];
const SEED_MAX = 2147483647;

function withRun(u: URL, run: string | null | undefined) {
  if (run) u.searchParams.set("run", run);
  return withModel(u).toString();
}

/** A link that reopens one feature. */
export function featureLink(origin: string, id: number, run: string | null | undefined) {
  return withRun(new URL(`/features/${id}`, origin), run);
}

/** A link that reruns one steering result exactly. */
export function steerLink(origin: string, l: SteerLink, run: string | null | undefined) {
  const u = new URL("/steer", origin);
  u.searchParams.set("feature", String(l.feature_id));
  u.searchParams.set("mode", l.mode);
  if (l.mode === "amplify") u.searchParams.set("strength", String(+l.strength.toFixed(2)));
  u.searchParams.set("seed", String(l.seed));
  u.searchParams.set("len", String(l.max_new_tokens));
  u.searchParams.set("prompt", l.prompt);
  return withRun(u, run);
}

/** The steering result a /steer URL asks for, or null when it carries no complete one. */
export function readSteerLink(p: URLSearchParams): SteerLink | null {
  const id = p.get("feature");
  const prompt = p.get("prompt");
  if (!id || !/^\d+$/.test(id) || !prompt?.trim()) return null;
  const mode: SteerMode = p.get("mode") === "ablate" ? "ablate" : "amplify";
  const strength = Number(p.get("strength") ?? 0.5);
  const seed = Number(p.get("seed") ?? 42);
  const len = Number(p.get("len") ?? 60);
  return {
    feature_id: Number(id),
    mode,
    strength: mode === "ablate" ? 0 : Number.isFinite(strength) ? Math.max(0, Math.min(3, +strength.toFixed(2))) : 0.5,
    seed: Number.isInteger(seed) && seed >= 0 && seed <= SEED_MAX ? seed : 42,
    prompt,
    max_new_tokens: LENGTHS.includes(len) ? len : 60,
  };
}
