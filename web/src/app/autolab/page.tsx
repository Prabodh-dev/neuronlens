"use client";

import { isMockModel, isSampleData, useHealth } from "@/components/HealthProvider";
import { AutoLabPanel } from "./panel";

/** Auto-Lab, on its own page in the navbar. */
export default function AutoLabPage() {
  const { data: h } = useHealth();
  return (
    <div className="page">
      <AutoLabPanel simulatedServer={!h || isMockModel(h) || isSampleData(h)} run={h?.run_id ?? null} />
    </div>
  );
}
