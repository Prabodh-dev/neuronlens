import { copy, t, type CopyKey } from "../copy";
import type {
  ErrorBody,
  FeatureDetail,
  FeatureList,
  Health,
  InspectRequest,
  InspectResponse,
  LabelStatus,
  MetricsResponse,
  PromptOnlyRequest,
  PromptOnlyResponse,
  RunList,
  SteerCheckResponse,
  SteerCompareResponse,
  SteerRequest,
  SteerResponse,
  SteerSweepRequest,
  SteerSweepResponse,
  SteeringExampleList,
} from "./types";

export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000/api/v1").replace(/\/+$/, "");

/** Server error codes from the contract, plus the ones the browser can hit on its own. */
export type ErrorCode = ErrorBody["code"] | "NETWORK" | "HTTP";

export class ApiError extends Error {
  constructor(
    public code: ErrorCode,
    message: string,
    public status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/** The copy-deck message for an error (docs/APP_BUILD_GUIDE.md Appendix A). */
export function errorMessage(e: ApiError): string {
  const key = `error.${e.code}`;
  return key in copy ? t(key as CopyKey, { message: e.message }) : t("error.generic");
}

// Reads give up after 10 s, steering after 60 s; inspect runs the model too, so it gets 30 s.
// A strength sweep runs one generation per strength plus the baseline, so it gets 120 s.
const READ_MS = 10_000;
const INSPECT_MS = 30_000;
const STEER_MS = 60_000;
const SWEEP_MS = 120_000;

async function request<T>(path: string, init: RequestInit = {}, signal?: AbortSignal, timeoutMs = READ_MS): Promise<T> {
  const ctl = new AbortController();
  const onAbort = () => ctl.abort(signal?.reason);
  signal?.addEventListener("abort", onAbort);
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    ctl.abort();
  }, timeoutMs);
  let res: Response;
  try {
    res = await fetch(API_BASE + path, {
      ...init,
      signal: ctl.signal,
      headers: { "Content-Type": "application/json", ...init.headers },
    });
  } catch (e) {
    if (timedOut) throw new ApiError("NETWORK", `The server did not answer within ${timeoutMs / 1000} s.`, 0);
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError("NETWORK", `Cannot reach ${API_BASE}.`, 0);
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
  }
  if (!res.ok) {
    let body: { error?: ErrorBody } | null = null;
    try {
      body = await res.json();
    } catch {
      /* not JSON */
    }
    if (body?.error) throw new ApiError(body.error.code, body.error.message, res.status);
    throw new ApiError("HTTP", `The server answered ${res.status}.`, res.status);
  }
  return res.json() as Promise<T>;
}

function qs(params: Record<string, string | number | boolean | null | undefined>): string {
  const u = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== "") u.set(k, String(v));
  const s = u.toString();
  return s ? `?${s}` : "";
}

export type FeatureQuery = {
  query?: string;
  status?: LabelStatus | null;
  include_dead?: boolean;
  sort?: "feature_id" | "density" | "max_activation";
  order?: "asc" | "desc";
  limit?: number;
  offset?: number;
};

export const api = {
  health: (signal?: AbortSignal) => request<Health>("/health", {}, signal),
  features: (q: FeatureQuery, signal?: AbortSignal) => request<FeatureList>(`/features${qs(q)}`, {}, signal),
  feature: (id: number, signal?: AbortSignal) => request<FeatureDetail>(`/features/${id}`, {}, signal),
  randomFeature: (signal?: AbortSignal) => request<FeatureDetail>("/features/random", {}, signal),
  inspect: (body: InspectRequest, signal?: AbortSignal) =>
    request<InspectResponse>("/inspect", { method: "POST", body: JSON.stringify(body) }, signal, INSPECT_MS),
  steer: (body: SteerRequest, signal?: AbortSignal) =>
    request<SteerResponse>("/steer", { method: "POST", body: JSON.stringify(body) }, signal, STEER_MS),
  // Phase 4: X1 activation check, X3 compare, X2 strength sweep, X4 prompt-only baseline
  steerCheck: (body: SteerRequest, signal?: AbortSignal) =>
    request<SteerCheckResponse>("/steer/check", { method: "POST", body: JSON.stringify(body) }, signal, STEER_MS),
  steerCompare: (body: SteerRequest, signal?: AbortSignal) =>
    request<SteerCompareResponse>("/steer/compare", { method: "POST", body: JSON.stringify(body) }, signal, STEER_MS),
  steerSweep: (body: SteerSweepRequest, signal?: AbortSignal) =>
    request<SteerSweepResponse>("/steer/sweep", { method: "POST", body: JSON.stringify(body) }, signal, SWEEP_MS),
  promptOnly: (body: PromptOnlyRequest, signal?: AbortSignal) =>
    request<PromptOnlyResponse>("/steer/prompt-only", { method: "POST", body: JSON.stringify(body) }, signal, STEER_MS),
  steeringExamples: (featureId: number | null, signal?: AbortSignal) =>
    request<SteeringExampleList>(`/steering-examples${qs({ feature_id: featureId })}`, {}, signal),
  runs: (signal?: AbortSignal) => request<RunList>("/runs", {}, signal),
  metrics: (runId: string | null, signal?: AbortSignal) =>
    request<MetricsResponse>(`/metrics${qs({ run_id: runId })}`, {}, signal),
};
