"use client";

import { API_BASE } from "@/lib/api/client";
import { t } from "@/lib/copy";
import { useHealth } from "./HealthProvider";
import s from "./ModeBanner.module.css";

/** Under the header on every page: sample data, mock mode, or offline. Nothing in real mode. */
export function ModeBanner() {
  const { data: h, error, reload } = useHealth();
  let text: string | null = null;
  let kind: "warn" | "off" = "warn";
  if (error) {
    text = t("banner.offline", { url: API_BASE });
    kind = "off";
  } else if (h?.is_synthetic) text = t("banner.synthetic");
  else if (h?.mode === "mock") text = t("banner.mock");
  if (!text) return null;
  return (
    <div className={`${s.banner} ${kind === "off" ? s.off : ""}`} role="status" aria-live="polite">
      <span className={s.mark} aria-hidden="true" />
      <span className={s.text}>{text}</span>
      {error && (
        <button className="btn btn-ghost btn-sm" onClick={reload}>
          {t("button.tryAgain")}
        </button>
      )}
    </div>
  );
}
