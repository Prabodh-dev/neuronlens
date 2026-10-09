"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { FeatureTitle } from "@/components/FeatureBits";
import { HeatText } from "@/components/HeatText";
import { isMockModel, useHealth } from "@/components/HealthProvider";
import { FeatureGrid, type GridPin } from "@/components/FeatureGrid";
import { ErrorState, Skeleton } from "@/components/States";
import { ApiError, api } from "@/lib/api/client";
import type { FeatureListItem, InspectResponse, SteerMode, SteerRequest, SteerResponse } from "@/lib/api/types";
import { diffWords } from "@/lib/diff";
import { EMPTY_TOKEN_TIP, featureName, fmtDensity, fmtInt, fmtNum, showToken } from "@/lib/format";
import { t } from "@/lib/copy";
import { useAction, useApi } from "@/lib/useApi";
import s from "./microscope.module.css";

const DEFAULT_TEXT = "The court ruled that the contract was void, so the judge walked home in the rain.";
const STEP_MS = 1100;
const PAGE = 200;

export default function Microscope() {
  const { data: h } = useHealth();
  const sim = isMockModel(h);

  /* every neuron, most frequent first so they sit in the centre */
  const [neurons, setNeurons] = useState<FeatureListItem[] | null>(null);
  const [loadErr, setLoadErr] = useState<ApiError | null>(null);
  const [progress, setProgress] = useState<[number, number] | null>(null);
  const [loadTick, setLoadTick] = useState(0);
  useEffect(() => {
    const ctl = new AbortController();
    (async () => {
      setLoadErr(null);
      try {
        const q = { include_dead: true, sort: "density" as const, order: "desc" as const, limit: PAGE };
        const first = await api.features({ ...q, offset: 0 }, ctl.signal);
        const pages = Math.ceil(first.total / PAGE);
        const all: FeatureListItem[][] = [first.items];
        setProgress([first.items.length, first.total]);
        for (let p = 1; p < pages; p += 6) {
          const batch = await Promise.all(
            Array.from({ length: Math.min(6, pages - p) }, (_, k) => api.features({ ...q, offset: (p + k) * PAGE }, ctl.signal)),
          );
          batch.forEach((b) => all.push(b.items));
          setProgress([all.reduce((n, a) => n + a.length, 0), first.total]);
        }
        setNeurons(all.flat());
      } catch (e) {
        if ((e as Error).name !== "AbortError") setLoadErr(e instanceof ApiError ? e : new ApiError("INTERNAL", String(e), 0));
      }
    })();
    return () => ctl.abort();
  }, [loadTick]);
  const byId = useMemo(() => new Map((neurons ?? []).map((f) => [f.feature_id, f])), [neurons]);

  /* sentence → per-token activity */
  const [text, setText] = useState(DEFAULT_TEXT);
  const inspect = useAction((body: { text: string; top_n: number }, sig) => api.inspect(body, sig));
  const res: InspectResponse | null = inspect.data;
  const [ti, setTi] = useState(1);
  const [playing, setPlaying] = useState(true);
  async function read(t = text) {
    if (!t.trim()) return;
    const r = await inspect.run({ text: t, top_n: 20 });
    if (r) {
      setTi(r.tokens.length > 1 ? 1 : 0);
      setPlaying(true);
    }
  }
  useEffect(() => {
    read(DEFAULT_TEXT);
  }, []);
  const tokens = res?.tokens ?? [];
  useEffect(() => {
    if (!playing || tokens.length < 2) return;
    const id = setInterval(() => setTi((i) => (i + 1 >= tokens.length ? 1 : i + 1)), STEP_MS);
    return () => clearInterval(id);
  }, [playing, tokens.length]);
  const cur = tokens[ti] ?? null;
  const activity = useMemo(() => {
    const m = new Map<number, number>();
    if (!cur || cur.is_outlier_position) return m;
    const tokMax = Math.max(...cur.features.map((f) => f.activation), 1e-9);
    for (const f of cur.features) {
      const peak = byId.get(f.feature_id)?.max_activation;
      m.set(f.feature_id, Math.min(1, peak ? f.activation / peak : f.activation / tokMax));
    }
    return m;
  }, [cur, byId]);

  /* opening a feature: the map flies in, then the tile opens into its panel; back reverses it */
  const [open, setOpen] = useState<number | null>(null);
  const [arrived, setArrived] = useState<number | null>(null);
  const [closing, setClosing] = useState(false);
  const [pin, setPin] = useState<GridPin>(null);
  const shown = useMemo(() => {
    if (!pin) return activity;
    const m = new Map(activity);
    if (pin.mode === "amplify") m.set(pin.id, Math.max(m.get(pin.id) ?? 0, 0.95));
    else m.delete(pin.id);
    return m;
  }, [activity, pin]);
  const pick = useCallback((id: number) => {
    setClosing(false);
    setArrived(null);
    setOpen(id);
  }, []);
  const back = useCallback(() => {
    setClosing(true);
    // let the panel fold back into its tile, then the map zooms out
    setTimeout(() => {
      setArrived(null);
      setOpen(null);
      setClosing(false);
    }, 260);
  }, []);
  const step = useCallback(
    (d: number) => {
      if (!neurons || open === null) return;
      const i = neurons.findIndex((f) => f.feature_id === open);
      const next = neurons[(i + d + neurons.length) % neurons.length];
      if (next) pick(next.feature_id);
    },
    [neurons, open, pick],
  );
  useEffect(() => {
    if (!neurons) return;
    const want = new URLSearchParams(window.location.search).get("neuron");
    if (want && /^\d+$/.test(want) && byId.has(Number(want))) pick(Number(want));
  }, [neurons, byId, pick]);
  const [find, setFind] = useState("");
  const matches = useMemo(() => {
    const q = find.trim().toLowerCase();
    if (!q || !neurons) return [];
    const asId = /^#?\d+$/.test(q) ? Number(q.replace("#", "")) : null;
    return neurons.filter((f) => f.feature_id === asId || (f.label ?? "").toLowerCase().includes(q));
  }, [find, neurons]);
  const highlight = useMemo(() => (matches.length ? new Set(matches.map((m) => m.feature_id)) : null), [matches]);

  /* keyboard */
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.key === " ") {
        e.preventDefault();
        setPlaying((p) => !p);
      } else if (e.key === "ArrowRight" && tokens.length) {
        setPlaying(false);
        setTi((i) => Math.min(tokens.length - 1, i + 1));
      } else if (e.key === "ArrowLeft" && tokens.length) {
        setPlaying(false);
        setTi((i) => Math.max(0, i - 1));
      } else if (e.key === "Escape" && open !== null) back();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [tokens.length, open, back]);

  const stepBy = (d: number) => {
    setPlaying(false);
    setTi((i) => Math.max(0, Math.min(tokens.length - 1, i + d)));
  };

  return (
    <div className={s.page}>
      {/* sentence bar */}
      <div className={s.bar}>
        <form
          className={s.sentence}
          onSubmit={(e) => {
            e.preventDefault();
            read();
          }}
        >
          <label htmlFor="mic-text" className="visually-hidden">
            Sentence
          </label>
          <input id="mic-text" value={text} onChange={(e) => setText(e.target.value)} placeholder="Type a sentence…" autoComplete="off" />
          <button className="btn btn-sm" type="submit" disabled={!text.trim() || inspect.loading}>
            {inspect.loading ? "Reading…" : "Read"}
          </button>
        </form>
        <div className={s.transport}>
          <button className="icon-btn" aria-label="Previous token" onClick={() => stepBy(-1)} disabled={!tokens.length}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M6 5h2.5v14H6zM20 5v14L9.5 12z" />
            </svg>
          </button>
          <button className="icon-btn" aria-pressed={playing} aria-label={playing ? "Pause" : "Play"} onClick={() => setPlaying((p) => !p)} disabled={tokens.length < 2}>
            {playing ? (
              <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                <path d="M6 5h4v14H6zM14 5h4v14h-4z" />
              </svg>
            ) : (
              <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                <path d="M7 4.5v15L19.5 12z" />
              </svg>
            )}
          </button>
          <button className="icon-btn" aria-label="Next token" onClick={() => stepBy(1)} disabled={!tokens.length}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M15.5 5H18v14h-2.5zM4 5v14l10.5-7z" />
            </svg>
          </button>
        </div>
      </div>

      <div className={s.tokens} role="list" aria-label="Tokens">
        {inspect.loading && !res && <Skeleton w={360} h={20} />}
        {tokens.map((t, i) => (
          <button
            key={i}
            role="listitem"
            className={`${s.tok} ${i === ti ? s.tokOn : ""} ${i < ti ? s.tokPast : ""} ${t.text === "" ? s.tokEmpty : ""}`}
            onClick={() => (setPlaying(false), setTi(i))}
            title={t.text === "" ? EMPTY_TOKEN_TIP : undefined}
          >
            {showToken(t.text)}
          </button>
        ))}
        {inspect.error && <span className={s.inlineErr}>{inspect.error.code === "NOT_FOUND" ? "Live firing needs /inspect on the server." : inspect.error.message}</span>}
      </div>

      {/* the map */}
      <div className={s.stage}>
        <div className={s.map} data-dim={arrived !== null && arrived === open && !closing ? "1" : "0"}>
          {loadErr ? (
            <div className={s.mapMsg}>
              <ErrorState error={loadErr} onRetry={() => setLoadTick((t) => t + 1)} />
            </div>
          ) : !neurons ? (
            <div className={s.mapMsg}>
              <div className={s.loadSquare} aria-hidden="true" />
              <span className="mono small muted">{progress ? `${fmtInt(progress[0])} / ${fmtInt(progress[1])}` : "Loading"}</span>
            </div>
          ) : (
            <FeatureGrid neurons={neurons} activity={shown} open={open} pin={pin} highlight={highlight} onOpen={pick} onArrive={setArrived} />
          )}
        </div>

        {/* top left: how many fire, and find one by number or name */}
        <div className={s.hud} data-away={open !== null ? "1" : "0"}>
          <div className={s.count}>
            <span>top</span>
            <b className="mono">{fmtInt(activity.size)}</b>
            <span>firing</span>
          </div>
          <div className={s.countSub}>
            of {fmtInt(neurons?.length ?? h?.d_sae ?? 0)} features · {sim ? "simulated" : "real data"}
          </div>
          <div className={s.find}>
            <input
              className={s.findInput}
              placeholder="Go to # or name"
              value={find}
              onChange={(e) => setFind(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && matches[0]) {
                  pick(matches[0].feature_id);
                  setFind("");
                }
              }}
              aria-label="Go to a feature by number or name"
              autoComplete="off"
            />
            {find.trim() && (
              <ul className={s.findList}>
                {matches.slice(0, 6).map((m) => (
                  <li key={m.feature_id}>
                    <button
                      onClick={() => {
                        pick(m.feature_id);
                        setFind("");
                      }}
                    >
                      <span className="mono muted">#{m.feature_id}</span> {featureName(m.label, m.feature_id)}
                    </button>
                  </li>
                ))}
                {!matches.length && <li className={s.findNone}>No match</li>}
                {matches.length > 6 && <li className={s.findNone}>{fmtInt(matches.length)} lit on the map</li>}
              </ul>
            )}
          </div>
        </div>

        {/* top right: what the current word fires */}
        <aside className={s.drawer} data-away={open !== null ? "1" : "0"} aria-live="polite">
          <FiringNow token={cur} byId={byId} onPick={pick} />
        </aside>

        {/* the opened tile */}
        {open !== null && arrived === open && (
          <div className={s.specWrap}>
          <div className={s.specimen} data-closing={closing ? "1" : "0"} role="dialog" aria-label={`Feature ${open}`}>
            <Specimen
              key={open}
              id={open}
              list={byId.get(open) ?? null}
              tokens={tokens}
              ti={ti}
              now={shown.get(open) ?? 0}
              onToken={(i) => (setPlaying(false), setTi(i))}
              onBack={back}
              onStep={step}
              pin={pin?.id === open ? pin : null}
              setPin={setPin}
              prompt={text}
              sim={sim}
            />
          </div>
          </div>
        )}

        <p className={s.hint} data-away={open !== null ? "1" : "0"}>
          <kbd>scroll</kbd> zoom · <kbd>drag</kbd> move · <kbd>click</kbd> a tile to open · <kbd>esc</kbd> back · <kbd>space</kbd> play
        </p>
      </div>
    </div>
  );
}

function FiringNow({ token, byId, onPick }: { token: InspectResponse["tokens"][number] | null; byId: Map<number, FeatureListItem>; onPick: (id: number) => void }) {
  const list = token && !token.is_outlier_position ? token.features.slice(0, 6) : [];
  return (
    <div className={s.card}>
      <span className="eyebrow">Now reading</span>
      <div className={s.bigTok} key={token?.index ?? -1}>
        {token ? showToken(token.text) : "—"}
      </div>
      {list.length ? (
        <ol className={s.fireList}>
          {list.map((f, i) => {
            const peak = byId.get(f.feature_id)?.max_activation ?? f.activation;
            const share = Math.min(1, f.activation / (peak || 1));
            return (
              <li key={`${f.feature_id}-${i}`} style={{ animationDelay: `${i * 40}ms` }}>
                <button onClick={() => onPick(f.feature_id)}>
                  <span className={s.fireName}>{featureName(f.label, f.feature_id)}</span>
                  <span className="mono muted">{Math.round(share * 100)}%</span>
                  <span className="bar" style={{ gridColumn: "1 / -1" }}>
                    <i style={{ width: `${Math.max(3, share * 100)}%` }} />
                  </span>
                </button>
              </li>
            );
          })}
        </ol>
      ) : (
        <p className="muted small">{token?.is_outlier_position ? t("inspect.outlier") : "Nothing firing."}</p>
      )}
    </div>
  );
}

function Specimen(props: {
  id: number;
  list: FeatureListItem | null;
  tokens: InspectResponse["tokens"];
  ti: number;
  now: number;
  onToken: (i: number) => void;
  onBack: () => void;
  onStep: (d: number) => void;
  pin: GridPin;
  setPin: (p: GridPin) => void;
  prompt: string;
  sim: boolean;
}) {
  const { id, list, tokens, ti, now } = props;
  const detail = useApi((sig) => api.feature(id, sig), `mic:${id}`);
  const f = detail.data?.feature_id === id ? detail.data : null;
  const peak = f?.max_activation ?? list?.max_activation ?? 1;
  const ex = f?.examples[0];
  const [mode, setMode] = useState<SteerMode>("amplify");
  const [strength, setStrength] = useState(0.5);
  const steer = useAction((body: SteerRequest, sig) => api.steer(body, sig));

  const across = tokens.map((t) => {
    const hit = t.features.find((x) => x.feature_id === id);
    return { text: t.text, heat: hit ? Math.min(1, hit.activation / (peak || 1)) : 0, skip: t.is_outlier_position };
  });
  const fired = across.some((a) => a.heat > 0);

  async function run() {
    props.setPin({ id, mode });
    await steer.run({ feature_id: id, mode, strength: mode === "amplify" ? strength : 0, prompt: props.prompt, seed: 42, max_new_tokens: 40 });
  }

  return (
    <div className={s.spec}>
      <div className={s.specHead}>
        <button className={s.backBtn} onClick={props.onBack} autoFocus>
          <span aria-hidden="true">←</span> All features
        </button>
        <span className={`eyebrow ${s.specId}`}>#{id}</span>
        <div className={s.stepper}>
          <button className="icon-btn" onClick={() => props.onStep(-1)} aria-label="Previous feature on the map">
            <span aria-hidden="true">‹</span>
          </button>
          <button className="icon-btn" onClick={() => props.onStep(1)} aria-label="Next feature on the map">
            <span aria-hidden="true">›</span>
          </button>
        </div>
      </div>

      <div className={s.specBody}>
      <div className={s.specCol}>
      <h2 className={s.specName}>
        <FeatureTitle label={list?.label ?? f?.label ?? null} id={id} />
      </h2>
      <p className={s.specMeta}>
        {list ? `${fmtDensity(list.density)} of tokens` : ""} · peak {fmtNum(peak, 1)}
        {list?.is_dead ? " · dead" : ""}
      </p>

      <div className={s.meter} aria-label={`Firing at ${Math.round(now * 100)} percent of peak`}>
        <div className={s.meterTrack}>
          <i style={{ width: `${Math.round(now * 100)}%` }} />
        </div>
        <span className="mono">{Math.round(now * 100)}%</span>
      </div>

      <section className={s.sec}>
        <span className="eyebrow">Top example</span>
        {detail.loading && !f && <Skeleton h={36} />}
        {detail.error && <ErrorState error={detail.error} compact />}
        {f && ex && <HeatText className={s.mini} tokens={ex.tokens.map((t, i) => ({ text: t, heat: (ex.activations[i] ?? 0) / (peak || 1), peak: i === ex.peak_index }))} />}
        {f && !ex && <p className="muted small">None stored.</p>}
      </section>
      </div>

      <div className={s.specCol}>
      <section className={s.sec}>
        <span className="eyebrow">In your sentence</span>
        {fired ? (
          <HeatText className={s.mini} selected={ti} onSelect={props.onToken} tokens={across.map((a) => ({ text: a.text, heat: a.heat, skip: a.skip }))} />
        ) : (
          <p className="muted small">Not active here.</p>
        )}
      </section>

      <section className={s.sec}>
        <span className="eyebrow">Steer</span>
        <div className="seg" role="group" aria-label="Mode" style={{ alignSelf: "flex-start" }}>
          <button aria-pressed={mode === "amplify"} onClick={() => setMode("amplify")}>
            Force on
          </button>
          <button aria-pressed={mode === "ablate"} onClick={() => setMode("ablate")}>
            Switch off
          </button>
        </div>
        {mode === "amplify" && (
          <label className={s.strength}>
            <span>Strength</span>
            <input type="range" min={0.25} max={3} step={0.25} value={strength} onChange={(e) => setStrength(Number(e.target.value))} />
            <span className="mono">{fmtNum(strength, 2)}×</span>
          </label>
        )}
        <div className="row" style={{ gap: 8 }}>
          <button className="btn btn-sm" onClick={run} disabled={steer.loading || !!list?.is_dead || !props.prompt.trim()}>
            {steer.loading ? "Running…" : "Run"}
          </button>
          {props.pin && (
            <button
              className="btn btn-ghost btn-sm"
              onClick={() => {
                props.setPin(null);
                steer.setData(null);
              }}
            >
              Reset
            </button>
          )}
        </div>
        {steer.error && <ErrorState error={steer.error} compact />}
        {steer.data && <Compare r={steer.data} sim={props.sim} />}
      </section>

      <Link href={`/features/${id}`} className="link small">
        Full details →
      </Link>
      </div>
      </div>
    </div>
  );
}

function Compare({ r, sim }: { r: SteerResponse; sim: boolean }) {
  const segs = useMemo(() => diffWords(r.baseline_text, r.steered_text), [r]);
  return (
    <div className={s.compare}>
      <div>
        <span className="eyebrow">Before · seed {r.seed}</span>
        <p>{r.baseline_text}</p>
      </div>
      <div className={s.after}>
        <span className="eyebrow">
          After · {r.mode === "amplify" ? `up ${+r.strength.toFixed(2)}×` : "off"} · seed {r.seed}
          {sim ? " · simulated" : ""}
        </span>
        <p>
          {segs.map((sg, i) =>
            sg.changed ? (
              <mark key={i} className={s.changed}>
                {sg.text}
              </mark>
            ) : (
              <span key={i}>{sg.text}</span>
            ),
          )}
        </p>
      </div>
    </div>
  );
}
