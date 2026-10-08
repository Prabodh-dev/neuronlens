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

/** Mock mode or a synthetic database: anything on screen is simulated. */
export function isSimulated(h: Health | null): boolean {
  return !!h && (h.mode === "mock" || h.is_synthetic);
}
