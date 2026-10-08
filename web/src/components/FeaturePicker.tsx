"use client";

import { useEffect, useId, useRef, useState } from "react";
import { api } from "@/lib/api/client";
import type { FeatureListItem } from "@/lib/api/types";
import { featureName, fmtDensity } from "@/lib/format";
import { useApi } from "@/lib/useApi";
import { StatusPill } from "./FeatureBits";
import s from "./FeaturePicker.module.css";

export type PickedFeature = { feature_id: number; label: string | null };

export function FeaturePicker({ value, onChange }: { value: PickedFeature | null; onChange: (f: PickedFeature) => void }) {
  const [q, setQ] = useState("");
  const [debounced, setDebounced] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const boxRef = useRef<HTMLDivElement>(null);
  const listId = useId();

  useEffect(() => {
    const t = setTimeout(() => setDebounced(q.trim()), 200);
    return () => clearTimeout(t);
  }, [q]);

  const { data, loading, error } = useApi(
    (sig) => api.features({ query: debounced, limit: 8, sort: debounced ? "feature_id" : "max_activation", order: debounced ? "asc" : "desc" }, sig),
    open ? `pick:${debounced}` : null,
  );
  const items = data?.items ?? [];

  useEffect(() => {
    function onDoc(e: MouseEvent) {
      if (!boxRef.current?.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  function pick(f: FeatureListItem) {
    onChange({ feature_id: f.feature_id, label: f.label });
    setQ("");
    setOpen(false);
  }

  function onKey(e: React.KeyboardEvent) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setOpen(true);
      setActive((a) => Math.min(items.length - 1, a + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((a) => Math.max(0, a - 1));
    } else if (e.key === "Enter" && open && items[active]) {
      e.preventDefault();
      pick(items[active]);
    } else if (e.key === "Escape") setOpen(false);
  }

  return (
    <div className={s.box} ref={boxRef}>
      <div className={s.control}>
        {value && !open ? (
          <button type="button" className={s.current} onClick={() => setOpen(true)}>
            <span className={`mono ${s.id}`}>#{value.feature_id}</span>
            <span className={s.name}>{featureName(value.label, value.feature_id)}</span>
            <span className={s.change}>Change</span>
          </button>
        ) : (
          <input
            className="input"
            placeholder="Search by name or number"
            value={q}
            autoFocus={open && !!value}
            onFocus={() => setOpen(true)}
            onChange={(e) => {
              setQ(e.target.value);
              setOpen(true);
              setActive(0);
            }}
            onKeyDown={onKey}
            role="combobox"
            aria-expanded={open}
            aria-controls={listId}
            aria-autocomplete="list"
          />
        )}
      </div>
      {open && (
        <div className={s.pop} id={listId} role="listbox">
          {!debounced && <div className={s.hint}>Strongest features</div>}
          {loading && !data && <div className={s.hint}>Searching…</div>}
          {error && <div className={s.hint}>Search failed: {error.message}</div>}
          {data && !items.length && <div className={s.hint}>No feature matches “{debounced}”.</div>}
          {items.map((f, i) => (
            <button
              type="button"
              key={f.feature_id}
              role="option"
              aria-selected={i === active}
              className={i === active ? `${s.opt} ${s.optOn}` : s.opt}
              onMouseEnter={() => setActive(i)}
              onClick={() => pick(f)}
            >
              <span className={`mono ${s.id}`}>#{f.feature_id}</span>
              <span className={s.name}>{featureName(f.label, f.feature_id)}</span>
              <span className={`mono ${s.meta}`}>{fmtDensity(f.density)}</span>
              <StatusPill status={f.label_status} dead={f.is_dead} />
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
