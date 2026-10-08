import type { LabelStatus } from "@/lib/api/types";
import { t } from "@/lib/copy";
import { densityScale } from "@/lib/format";

const STATUS_TEXT: Record<LabelStatus, string> = {
  named: "Named",
  ambiguous: "Ambiguous",
  unlabelled: "Unlabelled",
};
const STATUS_TIP: Record<LabelStatus, string> = {
  named: "Has a name drafted from its top examples. Check the examples before trusting it.",
  ambiguous: "Its examples don't share one clear concept, so the name may not fit.",
  unlabelled: "No name yet. Look at its examples to judge what it responds to.",
};

export function StatusPill({ status, dead }: { status: LabelStatus; dead?: boolean }) {
  if (dead)
    return (
      <span className="pill pill-dead pill-dot" title="Never fired in the scan, so it has no examples.">
        Dead
      </span>
    );
  return (
    <span className={`pill pill-${status} pill-dot`} title={STATUS_TIP[status]}>
      {STATUS_TEXT[status]}
    </span>
  );
}

export function FeatureTitle({ label, id }: { label: string | null; id: number }) {
  if (label && label.trim()) return <>{label}</>;
  void id;
  return <span style={{ color: "var(--muted)" }}>{t("feature.unlabelled")}</span>;
}

export function DensityBar({ density }: { density: number }) {
  return (
    <span className="bar" style={{ display: "block", width: "100%" }} title="Log scale, from 1 in a million tokens to every token">
      <i style={{ width: `${densityScale(density) * 100}%` }} />
    </span>
  );
}
