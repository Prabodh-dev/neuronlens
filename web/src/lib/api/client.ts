import type {
  ErrorBody,
  FeatureDetail,
  FeatureList,
  Health,
  InspectRequest,
  InspectResponse,
  LabelStatus,
  MetricsResponse,
  RunList,
  SteerRequest,
  SteerResponse,
  SteeringExampleList,
} from "./types";

export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000/api/v1").replace(/\/+$/, "");

/** Server error codes from the contract, plus two the browser can hit on its own. */
export type ErrorCode = ErrorBody["code"] | "UNREACHABLE" | "HTTP";

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

async function request<T>(path: string, init: RequestInit = {}, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(API_BASE + path, {
      ...init,
      signal,
      headers: { "Content-Type": "application/json", ...init.headers },
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError("UNREACHABLE", `Can't reach the API at ${API_BASE}.`, 0);
  }
  if (!res.ok) {
    let body: { error?: ErrorBody } | null = null;
    try {
      body = await res.json();
    } catch {
      /* not JSON */
    }
    if (body?.error) throw new ApiError(body.error.code, body.error.message, res.status);
    throw new ApiError("HTTP", `The API answered ${res.status}.`, res.status);
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
  inspect: (body: InspectRequest, signal?: AbortSignal) =>
    request<InspectResponse>("/inspect", { method: "POST", body: JSON.stringify(body) }, signal),
  steer: (body: SteerRequest, signal?: AbortSignal) =>
    request<SteerResponse>("/steer", { method: "POST", body: JSON.stringify(body) }, signal),
  steeringExamples: (featureId: number | null, signal?: AbortSignal) =>
    request<SteeringExampleList>(`/steering-examples${qs({ feature_id: featureId })}`, {}, signal),
  runs: (signal?: AbortSignal) => request<RunList>("/runs", {}, signal),
  metrics: (runId: string | null, signal?: AbortSignal) =>
    request<MetricsResponse>(`/metrics${qs({ run_id: runId })}`, {}, signal),
};
