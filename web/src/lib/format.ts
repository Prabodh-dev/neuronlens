/** density is a fraction 0..1 (contracts/README.md). */
export function fmtDensity(d: number): string {
  if (d <= 0) return "0%";
  const p = d * 100;
  if (p >= 10) return `${p.toFixed(1)}%`;
  if (p >= 0.1) return `${p.toFixed(2)}%`;
  if (p >= 0.001) return `${p.toFixed(3)}%`;
  return `${p.toExponential(1)}%`;
}

/** "about 1 in 16,000 tokens" for a density fraction. */
export function oneIn(d: number): string | null {
  if (d <= 0) return null;
  const n = 1 / d;
  if (n < 2) return "most tokens";
  const rounded = n < 100 ? Math.round(n) : Number(n.toPrecision(2));
  return `about 1 in ${rounded.toLocaleString("en-US")} tokens`;
}

export function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  return v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function fmtInt(v: number | null | undefined): string {
  if (v === null || v === undefined) return "–";
  return v.toLocaleString("en-US");
}

/** Width 0..1 for a density on a log scale from 1e-6 to 1. */
export function densityScale(d: number): number {
  if (d <= 0) return 0;
  const lo = -6;
  return Math.max(0.02, Math.min(1, (Math.log10(d) - lo) / -lo));
}

/** Show a token exactly as tokenised, with invisible characters made visible (never trimmed). */
export function showToken(t: string): string {
  if (t === "") return "∅";
  if (/^ +$/.test(t)) return "␠";
  return t.replace(/\n/g, "↵").replace(/\t/g, "⇥");
}

/** A feature's label, or "Unlabelled feature #id" where no number is shown next to it. */
export function featureName(label: string | null | undefined, id: number): string {
  return label && label.trim() ? label : `Unlabelled feature #${id}`;
}

/** A value that is already a percent. Small values keep enough digits not to read as zero. */
export function fmtPct(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${v.toFixed(pctDigits(v))}%`;
}
export function pctDigits(v: number): number {
  const a = Math.abs(v);
  return a === 0 || a >= 1 ? 1 : a >= 0.1 ? 2 : 3;
}

export function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}
