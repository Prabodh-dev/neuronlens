"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError } from "./api/client";

export type ApiState<T> = {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
  reload: () => void;
};

/** Runs a GET-style loader whenever `key` changes; aborts the previous call. */
export function useApi<T>(load: (signal: AbortSignal) => Promise<T>, key: string | null): ApiState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(key !== null);
  const [tick, setTick] = useState(0);
  const loadRef = useRef(load);
  loadRef.current = load;

  useEffect(() => {
    if (key === null) {
      setLoading(false);
      return;
    }
    const ctl = new AbortController();
    setLoading(true);
    setError(null);
    loadRef
      .current(ctl.signal)
      .then((d) => {
        if (!ctl.signal.aborted) setData(d);
      })
      .catch((e) => {
        if (ctl.signal.aborted || e?.name === "AbortError") return;
        setError(e instanceof ApiError ? e : new ApiError("INTERNAL", String(e?.message ?? e), 0));
      })
      .finally(() => {
        if (!ctl.signal.aborted) setLoading(false);
      });
    return () => ctl.abort();
  }, [key, tick]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { data, error, loading, reload };
}

/** For POST actions started by the user (inspect, steer). */
export function useAction<A, T>(run: (arg: A, signal: AbortSignal) => Promise<T>) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(false);
  const ctlRef = useRef<AbortController | null>(null);
  const runRef = useRef(run);
  runRef.current = run;

  const go = useCallback(async (arg: A) => {
    ctlRef.current?.abort();
    const ctl = new AbortController();
    ctlRef.current = ctl;
    setLoading(true);
    setError(null);
    try {
      const d = await runRef.current(arg, ctl.signal);
      if (!ctl.signal.aborted) setData(d);
      return d;
    } catch (e) {
      const err = e as Error;
      if (ctl.signal.aborted || err?.name === "AbortError") return null;
      setError(err instanceof ApiError ? err : new ApiError("INTERNAL", String(err?.message ?? err), 0));
      return null;
    } finally {
      if (!ctl.signal.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => () => ctlRef.current?.abort(), []);
  return { data, error, loading, run: go, setData };
}
