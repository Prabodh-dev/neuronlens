"use client";

import { useEffect, useState } from "react";
import { ApiError, api } from "@/lib/api/client";
import { useHealth } from "./HealthProvider";
import { EmptyState } from "./States";

/** NEXT_PUBLIC_MODEL_ONLY=1 is for the demo machine: pages show only when the server runs the trained
 *  model on the real database, so no simulated or sample result can reach the screen. */
export const MODEL_ONLY = process.env.NEXT_PUBLIC_MODEL_ONLY === "1";

export function ModelGate({ children }: { children: React.ReactNode }) {
  const { data: h, error, reload } = useHealth();
  const [notLoaded, setNotLoaded] = useState<string | null>(null);
  const onModel = !!h && h.mode === "real" && !h.is_synthetic;

  // /health says real mode even when the checkpoint failed to load, so ask the model for one token
  useEffect(() => {
    if (!MODEL_ONLY || !onModel) return;
    const ctl = new AbortController();
    api
      .inspect({ text: "Hello", top_n: 1 }, ctl.signal)
      .then(() => setNotLoaded(null))
      .catch((e) => {
        if (e instanceof ApiError && e.code === "MODEL_NOT_LOADED") setNotLoaded(e.message);
      });
    return () => ctl.abort();
  }, [onModel, h]);

  // offline is shown by the pages themselves; nothing simulated can come from a server that is down
  if (!MODEL_ONLY || error) return <>{children}</>;
  // wait for /health, so nothing from a mock server flashes up first
  if (!h) return null;
  const why =
    h.mode !== "real"
      ? "The server is in mock mode, so Inspect and Steer would be simulated."
      : h.is_synthetic
        ? "The server is reading the sample database, not the trained model's."
        : notLoaded
          ? `The model did not load. ${notLoaded.replace(/[.\s]*$/, ".")}`
          : null;
  if (!why) return <>{children}</>;
  return (
    <div className="page">
      <div className="card">
        <EmptyState
          icon="spark"
          title="Not running on the model"
          action={
            <button className="btn" onClick={reload}>
              Check again
            </button>
          }
        >
          {why} Restart it with NEURONLENS_MODE=real, NEURONLENS_DB and NEURONLENS_CKPT.
        </EmptyState>
      </div>
    </div>
  );
}
