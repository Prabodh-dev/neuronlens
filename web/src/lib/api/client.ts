import { copy, t, type CopyKey } from "../copy";
import type {
  CofireResponse,
  ErrorBody,
  FeatureDetail,
  FeatureList,
  Health,
  InspectRequest,
  InspectResponse,
  LabelStatus,
  MetricsResponse,
  MultiSteerRequest,
  MultiSteerResponse,
  PromptOnlyRequest,
  PromptOnlyResponse,
  ReportRequest,
  ReportResponse,
  RunList,
  SearchRequest,
  SearchResponse,
  SteerCheckResponse,
  SteerCompareResponse,
  SteerRequest,
  SteerResponse,
  SteerSweepRequest,
  SteerSweepResponse,
  SteeringExampleList,
  TextCompareRequest,
  TextCompareResponse,
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

/** Whether the running server has an endpoint, from its /openapi.json: a screen whose endpoint is
 *  not on every server yet (W5) stays hidden where it is missing. */
export async function serverHas(path: string, signal?: AbortSignal): Promise<boolean> {
  try {
    const r = await fetch(`${API_BASE.replace(/\/api\/v\d+$/, "")}/openapi.json`, { signal });
    if (!r.ok) return false;
    const spec = (await r.json()) as { paths?: Record<string, unknown> };
    return Object.keys(spec.paths ?? {}).some((p) => p.endsWith(path));
  } catch {
    return false;
  }
}

// Reads give up after 10 s, steering after 60 s; inspect runs the model too, so it gets 30 s.
// A strength sweep runs one generation per strength plus the baseline, so it gets 120 s; so does
// combining features (W5), which runs each feature alone as well.
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
  // X6: per sentence, the most active features; X7: the features that differ most between two texts
  inspectReport: (body: ReportRequest, signal?: AbortSignal) =>
    request<ReportResponse>("/inspect/report", { method: "POST", body: JSON.stringify(body) }, signal, INSPECT_MS),
  inspectCompare: (body: TextCompareRequest, signal?: AbortSignal) =>
    request<TextCompareResponse>("/inspect/compare", { method: "POST", body: JSON.stringify(body) }, signal, INSPECT_MS),
  // W8: the features that fired most on the words of a query (runs the model, like inspect)
  featureSearch: (body: SearchRequest, signal?: AbortSignal) =>
    request<SearchResponse>("/features/search", { method: "POST", body: JSON.stringify(body) }, signal, INSPECT_MS),
  // W4: features that fired on the same tokens in a sample
  cofire: (id: number, signal?: AbortSignal) => request<CofireResponse>(`/features/${id}/cofire`, {}, signal),
  steer: (body: SteerRequest, signal?: AbortSignal) =>
    request<SteerResponse>("/steer", { method: "POST", body: JSON.stringify(body) }, signal, STEER_MS),
  // Phase 4: X1 activation check, X3 compare, X2 strength sweep, X4 prompt-only baseline
  steerCheck: (body: SteerRequest, signal?: AbortSignal) =>
    request<SteerCheckResponse>("/steer/check", { method: "POST", body: JSON.stringify(body) }, signal, STEER_MS),
  steerCompare: (body: SteerRequest, signal?: AbortSignal) =>
    request<SteerCompareResponse>("/steer/compare", { method: "POST", body: JSON.stringify(body) }, signal, STEER_MS),
  steerSweep: (body: SteerSweepRequest, signal?: AbortSignal) =>
    request<SteerSweepResponse>("/steer/sweep", { method: "POST", body: JSON.stringify(body) }, signal, SWEEP_MS),
  // W5: two or three features steered together, with each one alone beside it
  steerMulti: (body: MultiSteerRequest, signal?: AbortSignal) =>
    request<MultiSteerResponse>("/steer/multi", { method: "POST", body: JSON.stringify(body) }, signal, SWEEP_MS),
  promptOnly: (body: PromptOnlyRequest, signal?: AbortSignal) =>
    request<PromptOnlyResponse>("/steer/prompt-only", { method: "POST", body: JSON.stringify(body) }, signal, STEER_MS),
  steeringExamples: (featureId: number | null, signal?: AbortSignal) =>
    request<SteeringExampleList>(`/steering-examples${qs({ feature_id: featureId })}`, {}, signal),
  runs: (signal?: AbortSignal) => request<RunList>("/runs", {}, signal),
  metrics: (runId: string | null, signal?: AbortSignal) =>
    request<MetricsResponse>(`/metrics${qs({ run_id: runId })}`, {}, signal),
};
