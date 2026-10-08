/** Black and white only: activation is drawn as light, from dark grey to pure white. */

export function heatRGB(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t));
  const v = Math.round(28 + (255 - 28) * Math.pow(x, 0.85));
  return [v, v, v];
}

export function heatCss(t: number, alpha = 1): string {
  const [r, g, b] = heatRGB(t);
  return alpha >= 1 ? `rgb(${r} ${g} ${b})` : `rgb(${r} ${g} ${b} / ${alpha.toFixed(3)})`;
}

/** Background and text colour for a token shaded by activation share 0..1. */
export function tokenColors(share: number): { background?: string; color?: string; boxShadow?: string } {
  const s = Math.max(0, Math.min(1, share));
  if (s < 0.05) return {};
  const a = 0.08 + s * 0.92;
  return {
    background: `rgb(255 255 255 / ${a.toFixed(3)})`,
    color: a > 0.55 ? "#000" : "#fafafa",
    boxShadow: s > 0.75 ? `0 0 ${Math.round(8 + s * 14)}px rgb(255 255 255 / ${(s * 0.45).toFixed(2)})` : undefined,
  };
}
