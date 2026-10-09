"use client";

import { createContext, useContext } from "react";
import { api } from "@/lib/api/client";
import type { Health } from "@/lib/api/types";
import { useApi, type ApiState } from "@/lib/useApi";

const HealthContext = createContext<ApiState<Health> | null>(null);

export function HealthProvider({ children }: { children: React.ReactNode }) {
  const state = useApi((s) => api.health(s), "health");
  return <HealthContext.Provider value={state}>{children}</HealthContext.Provider>;
}

export function useHealth(): ApiState<Health> {
  const ctx = useContext(HealthContext);
  if (!ctx) throw new Error("useHealth needs <HealthProvider>");
  return ctx;
}

/** The database is the synthetic fixture: features, examples and metrics are fake. */
export function isSampleData(h: Health | null): boolean {
  return !!h?.is_synthetic;
}

/** The server runs the mock runtime: Inspect and Steer results are simulated. */
export function isMockModel(h: Health | null): boolean {
  return h?.mode === "mock";
}
