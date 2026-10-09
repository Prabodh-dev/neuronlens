import { errorMessage, type ApiError } from "@/lib/api/client";
import { t } from "@/lib/copy";
import s from "./States.module.css";

export function Skeleton({ w = "100%", h = 14, r }: { w?: number | string; h?: number; r?: number }) {
  return <span className="skel" style={{ display: "block", width: w, height: h, borderRadius: r }} />;
}

export function LoadingBlock({ lines = 4, label = "Loading" }: { lines?: number; label?: string }) {
  return (
    <div className={s.loading} role="status" aria-label={label}>
      {Array.from({ length: lines }, (_, i) => (
        <Skeleton key={i} w={`${92 - ((i * 17) % 40)}%`} h={14} />
      ))}
    </div>
  );
}

export function EmptyState({
  title,
  children,
  action,
  icon = "search",
}: {
  title: string;
  children?: React.ReactNode;
  action?: React.ReactNode;
  icon?: "search" | "spark" | "chart";
}) {
  return (
    <div className={s.empty}>
      <div className={s.glyph} aria-hidden="true">
        {icon === "search" && (
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <circle cx="11" cy="11" r="6.5" />
            <path d="m20 20-4.2-4.2" />
          </svg>
        )}
        {icon === "spark" && (
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M5.6 18.4l2.8-2.8M15.6 8.4l2.8-2.8" />
          </svg>
        )}
        {icon === "chart" && (
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M4 20V10M10 20V4M16 20v-7M22 20H2" />
          </svg>
        )}
      </div>
      <h3>{title}</h3>
      {children && <p>{children}</p>}
      {action && <div className={s.action}>{action}</div>}
    </div>
  );
}

export function ErrorState({ error, onRetry, compact }: { error: ApiError; onRetry?: () => void; compact?: boolean }) {
  const msg = errorMessage(error);
  return (
    <div className={compact ? `${s.error} ${s.compact}` : s.error} role="alert">
      <div className={s.errIcon} aria-hidden="true">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
          <path d="M12 8v5M12 16.5v.5" />
          <circle cx="12" cy="12" r="9.5" />
        </svg>
      </div>
      <div className={s.errBody}>
        <strong>{t("error.title")}</strong>
        <p>{msg}</p>
        {error.message && !msg.includes(error.message) && <p className={s.detail}>{error.message}</p>}
      </div>
      {onRetry && error.code !== "FEATURE_NOT_FOUND" && (
        <button className="btn btn-ghost btn-sm" onClick={onRetry}>
          {t("button.tryAgain")}
        </button>
      )}
    </div>
  );
}
