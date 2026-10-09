"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useState } from "react";
import { DensityBar, FeatureTitle, StatusPill } from "@/components/FeatureBits";
import { EmptyState, ErrorState, Skeleton } from "@/components/States";
import { ApiError, api, errorMessage, type FeatureQuery } from "@/lib/api/client";
import { t } from "@/lib/copy";
import type { LabelStatus } from "@/lib/api/types";
import { fmtDensity, fmtInt, fmtNum } from "@/lib/format";
import { useApi } from "@/lib/useApi";
import s from "./explorer.module.css";

const PAGE = 50;
type Sort = NonNullable<FeatureQuery["sort"]>;
const STATUSES: { v: LabelStatus | ""; label: string }[] = [
  { v: "", label: "All" },
  { v: "named", label: "Named" },
  { v: "ambiguous", label: "Ambiguous" },
  { v: "unlabelled", label: "Unlabelled" },
];

export default function ExplorerPage() {
  return (
    <Suspense fallback={<div className="page" />}>
      <Explorer />
    </Suspense>
  );
}

function Explorer() {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();

  const q = params.get("q") ?? "";
  const status = (params.get("status") ?? "") as LabelStatus | "";
  const dead = params.get("dead") === "1";
  const sort = (params.get("sort") ?? "feature_id") as Sort;
  const order = (params.get("order") ?? (sort === "feature_id" ? "asc" : "desc")) as "asc" | "desc";
  const page = Math.max(0, Number(params.get("page") ?? 0) || 0);

  const [draft, setDraft] = useState(q);
  useEffect(() => setDraft(q), [q]);

  function setParams(next: Record<string, string | null>, resetPage = true) {
    const u = new URLSearchParams(params.toString());
    for (const [k, v] of Object.entries(next)) (v === null || v === "" ? u.delete(k) : u.set(k, v));
    if (resetPage) u.delete("page");
    const str = u.toString();
    router.replace(str ? `${pathname}?${str}` : pathname, { scroll: false });
  }

  useEffect(() => {
    if (draft === q) return;
    const t = setTimeout(() => setParams({ q: draft.trim() || null }), 250);
    return () => clearTimeout(t);
  }, [draft]);

  const query: FeatureQuery = { query: q, status: status || null, include_dead: dead, sort, order, limit: PAGE, offset: page * PAGE };
  const key = JSON.stringify(query);
  const { data, error, loading, reload } = useApi((sig) => api.features(query, sig), key);

  const peak = useApi((sig) => api.features({ sort: "max_activation", order: "desc", limit: 1, include_dead: true }, sig), "peak");
  const maxAct = peak.data?.items[0]?.max_activation ?? 1;

  function sortBy(col: Sort) {
    if (col === sort) setParams({ sort: col, order: order === "asc" ? "desc" : "asc" });
    else setParams({ sort: col, order: col === "feature_id" ? "asc" : "desc" });
  }

  const total = data?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / PAGE));
  const from = total ? page * PAGE + 1 : 0;
  const to = Math.min(total, (page + 1) * PAGE);
  const filtered = !!(q || status || dead);

  const skeletonRows = useMemo(() => Array.from({ length: 10 }, (_, i) => i), []);

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Features</h1>
          <p className="lede">Every concept the model learned.</p>
        </div>
        <RandomButton />
      </div>

      <div className={`card ${s.toolbar}`}>
        <div className={s.search}>
          <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
            <circle cx="11" cy="11" r="6.5" />
            <path d="m20 20-4.2-4.2" />
          </svg>
          <label htmlFor="ex-q" className="visually-hidden">
            Search features
          </label>
          <input
            id="ex-q"
            className={s.searchInput}
            placeholder="Search by name or #"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            autoComplete="off"
          />
          {draft && (
            <button className={s.clear} onClick={() => setDraft("")} aria-label="Clear search">
              ×
            </button>
          )}
        </div>
        <div className="seg" role="group" aria-label="Name status">
          {STATUSES.map((st) => (
            <button key={st.v || "all"} aria-pressed={status === st.v} onClick={() => setParams({ status: st.v || null })}>
              {st.label}
            </button>
          ))}
        </div>
        <label className="check">
          <input type="checkbox" checked={dead} onChange={(e) => setParams({ dead: e.target.checked ? "1" : null })} />
          Include dead
        </label>
      </div>

      <div className={s.meta}>
        <span className="muted small" aria-live="polite">
          {data ? (
            total ? (
              <>
                Showing <b className="mono">{fmtInt(from)}–{fmtInt(to)}</b> of <b className="mono">{fmtInt(total)}</b> features
              </>
            ) : (
              "No matches"
            )
          ) : (
            " "
          )}
        </span>
        {filtered && (
          <button className={s.reset} onClick={() => router.replace(pathname, { scroll: false })}>
            Clear filters
          </button>
        )}
      </div>

      <div className={`card ${s.table}`}>
        <div className={`${s.row} ${s.head}`} role="row">
          <SortHead label="#" col="feature_id" sort={sort} order={order} onSort={sortBy} />
          <span className={s.hName}>Name</span>
          <span className={s.hStatus}>Status</span>
          <SortHead label="Density" col="density" sort={sort} order={order} onSort={sortBy} className={s.hDensity} />
          <SortHead label="Peak" col="max_activation" sort={sort} order={order} onSort={sortBy} className={s.hPeak} />
        </div>

        {error && (
          <div className={s.pad}>
            <ErrorState error={error} onRetry={reload} />
          </div>
        )}

        {!error && loading && !data &&
          skeletonRows.map((i) => (
            <div key={i} className={s.row}>
              <Skeleton w={36} />
              <Skeleton w={`${60 - (i % 4) * 9}%`} />
              <Skeleton w={70} h={20} r={10} />
              <Skeleton />
              <Skeleton />
            </div>
          ))}

        {!error && data && !data.items.length && (
          <EmptyState
            title={t("empty.features")}
            action={
              <button className="btn btn-ghost" onClick={() => router.replace(pathname, { scroll: false })}>
                Clear filters
              </button>
            }
          >
            {q ? `Nothing matches “${q}”.` : "Try another filter."}
          </EmptyState>
        )}

        {!error && data && (
          <div className={loading ? s.dim : undefined}>
            {data.items.map((f) => (
              <Link key={f.feature_id} href={`/features/${f.feature_id}`} className={`${s.row} ${s.item}`}>
                <span className={`mono ${s.id}`}>{f.feature_id}</span>
                <span className={s.name}>
                  <FeatureTitle label={f.label} id={f.feature_id} />
                </span>
                <span className={s.status}>
                  <StatusPill status={f.label_status} dead={f.is_dead} />
                </span>
                <span className={s.metric}>
                  <DensityBar density={f.density} />
                  <span className="mono">{fmtDensity(f.density)}</span>
                </span>
                <span className={`${s.metric} ${s.peak}`}>
                  <span className="bar heat" style={{ display: "block" }}>
                    <i style={{ width: `${Math.max(2, (f.max_activation / maxAct) * 100)}%` }} />
                  </span>
                  <span className="mono">{fmtNum(f.max_activation, 1)}</span>
                </span>
              </Link>
            ))}
          </div>
        )}
      </div>

      {data && total > PAGE && (
        <nav className={s.pager} aria-label="Pages">
          <button className="btn btn-ghost btn-sm" disabled={page === 0} onClick={() => setParams({ page: String(page - 1) }, false)}>
            ← Previous
          </button>
          <span className="mono small muted">
            Page {page + 1} of {pages}
          </span>
          <button className="btn btn-ghost btn-sm" disabled={page + 1 >= pages} onClick={() => setParams({ page: String(page + 1) }, false)}>
            Next →
          </button>
        </nav>
      )}
    </div>
  );
}

function RandomButton() {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  return (
    <div className="row" style={{ gap: 8 }}>
      {err && <span className="small muted">{err}</span>}
      <button
        className="btn btn-ghost"
        disabled={busy}
        onClick={async () => {
          setBusy(true);
          setErr(null);
          try {
            const f = await api.randomFeature();
            router.push(`/features/${f.feature_id}`);
          } catch (e) {
            setErr(e instanceof ApiError ? errorMessage(e) : t("error.generic"));
            setBusy(false);
          }
        }}
      >
        {busy ? t("loading.default") : t("button.random")}
      </button>
    </div>
  );
}

function SortHead({
  label,
  col,
  sort,
  order,
  onSort,
  className,
}: {
  label: string;
  col: Sort;
  sort: Sort;
  order: "asc" | "desc";
  onSort: (c: Sort) => void;
  className?: string;
}) {
  const on = sort === col;
  return (
    <button
      className={`${s.sortBtn} ${on ? s.sortOn : ""} ${className ?? ""}`}
      onClick={() => onSort(col)}
      aria-label={`Sort by ${label}${on ? (order === "asc" ? ", ascending" : ", descending") : ""}`}
    >
      {label}
      <span aria-hidden="true" className={s.arrow}>
        {on ? (order === "asc" ? "↑" : "↓") : "↕"}
      </span>
    </button>
  );
}
