"use client";

import { useEffect, useState } from "react";
import { useHealth } from "./HealthProvider";

/** S5: a shared link carries the run it was made on (`?run=`). When this server serves another run,
 *  the same feature number or settings can give a different result, so say so. */
export function RunNote() {
  const { data: h } = useHealth();
  const [linked, setLinked] = useState<string | null>(null);
  useEffect(() => setLinked(new URLSearchParams(window.location.search).get("run")), []);
  if (!linked || !h || linked === h.run_id) return null;
  return (
    <p className="small muted" role="note" style={{ margin: "0 0 12px" }}>
      This link was made on run <span className="mono">{linked}</span>; this server runs <span className="mono">{h.run_id}</span>, so the result can
      differ.
    </p>
  );
}
