import type { ApiError } from "@/lib/api/client";
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

const FRIENDLY: Partial<Record<string, { title: string; body: string }>> = {
  UNREACHABLE: {
    title: "Can't reach the API",
    body: "The server isn't running or the address is wrong. Start it from server/ and try again.",
  },
  MODEL_NOT_LOADED: {
    title: "The model isn't loaded",
    body: "Live runs need the server in real mode with a checkpoint. In mock mode, saved runs are shown instead.",
  },
  TEXT_TOO_LONG: { title: "That text is too long", body: "Shorten it and run again." },
  FEATURE_NOT_FOUND: { title: "No feature with that number", body: "Check the number, or go back to the list." },
  NOT_FOUND: {
    title: "This isn't on the server yet",
    body: "The endpoint behind this view hasn't been added to the API. It will work once the server ships it.",
  },
  SCHEMA_MISMATCH: {
    title: "The database doesn't match the app",
    body: "The server's database uses a schema version this app doesn't support.",
  },
};

export function ErrorState({ error, onRetry, compact }: { error: ApiError; onRetry?: () => void; compact?: boolean }) {
  const f = FRIENDLY[error.code];
  return (
    <div className={compact ? `${s.error} ${s.compact}` : s.error} role="alert">
      <div className={s.errIcon} aria-hidden="true">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
          <path d="M12 8v5M12 16.5v.5" />
          <circle cx="12" cy="12" r="9.5" />
        </svg>
      </div>
      <div className={s.errBody}>
        <strong>{f?.title ?? "Something went wrong"}</strong>
        <p>{f?.body ?? error.message}</p>
        {f && error.message && <p className={s.detail}>{error.code}: {error.message}</p>}
      </div>
      {onRetry && error.code !== "FEATURE_NOT_FOUND" && (
        <button className="btn btn-ghost btn-sm" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}
