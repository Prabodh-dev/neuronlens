/* Neuron shapes for the landing tour: a cell body, forking dendrites with spines, and an axon with
   terminals, grown from a seeded random walk, plus the signals that run along them. The shapes are an
   illustration; the tour decides which cells fire from real data. */

export const TAU = Math.PI * 2;

// s0: distance from the cell body, along the cell, to where this branch starts
export type Branch = { pts: Float32Array; cum: Float32Array; len: number; w0: number; w1: number; kids: Branch[]; s0: number };
export type Cell = {
  x: number;
  y: number;
  r: number;
  size: number;
  layer: number;
  roots: number[];
  dend: Branch[];
  paths: Branch[]; // soma to a dendrite tip, for the incoming pulses
  axon: Branch[]; // [0] is the axon, the rest its terminals
  feature: number | null;
  label: string | null;
  e: number;
};
export function rngFrom(seed: number) {
  let x = seed >>> 0 || 1;
  return () => {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    return (x >>> 0) / 4294967296;
  };
}

export function branch(p: number[], w0: number, w1: number): Branch {
  const pts = Float32Array.from(p);
  const n = pts.length / 2;
  const cum = new Float32Array(n);
  for (let i = 1; i < n; i++) cum[i] = cum[i - 1] + Math.hypot(pts[2 * i] - pts[2 * i - 2], pts[2 * i + 1] - pts[2 * i - 1]);
  return { pts, cum, len: cum[n - 1], w0, w1, kids: [], s0: 0 };
}

/** A dendrite: a wandering, tapering line that forks a few times. */
export function grow(rng: () => number, x: number, y: number, a: number, len: number, w: number, depth: number, size: number, out: Branch[], s0 = 0): Branch {
  const step = 3 * size;
  const n = Math.max(2, Math.round(len / step));
  const p = [x, y];
  let ang = a;
  const bend = (rng() - 0.5) * 0.05;
  for (let i = 0; i < n; i++) {
    ang += bend + (rng() - 0.5) * 0.24;
    x += Math.cos(ang) * step;
    y += Math.sin(ang) * step;
    p.push(x, y);
  }
  const w1 = Math.max(0.32 * size, w * 0.56);
  const b = branch(p, w, w1);
  b.s0 = s0;
  out.push(b);
  if (depth < 3 && len > 16 * size) {
    const spread = 0.3 + rng() * 0.42;
    const two = rng() < 0.8;
    b.kids.push(grow(rng, x, y, ang - (two ? spread : 0.15), len * (0.55 + rng() * 0.25), w1, depth + 1, size, out, s0 + b.len));
    if (two) b.kids.push(grow(rng, x, y, ang + spread, len * (0.5 + rng() * 0.25), w1 * 0.9, depth + 1, size, out, s0 + b.len));
    if (rng() < 0.4) {
      const m = 1 + Math.floor((n - 1) * (0.3 + rng() * 0.4));
      const da = Math.atan2(p[2 * m + 1] - p[2 * m - 1], p[2 * m] - p[2 * m - 2]);
      grow(rng, p[2 * m], p[2 * m + 1], da + (rng() < 0.5 ? -1 : 1) * (0.7 + rng() * 0.5), len * 0.4, w * 0.55, depth + 2, size, out, s0 + b.cum[m]);
    }
  }
  return b;
}

export function makeCell(rng: () => number, x: number, y: number, size: number, layer: number, reach = 1): Cell {
  const r = (5.5 + rng() * 3) * size;
  const c: Cell = { x, y, r, size, layer, roots: [], dend: [], paths: [], axon: [], feature: null, label: null, e: 0 };
  const n = 4 + Math.floor(rng() * 4);
  const a0 = rng() * TAU;
  for (let i = 0; i < n; i++) {
    const a = a0 + (i / n) * TAU + (rng() - 0.5) * 0.6;
    c.roots.push(a);
    const prim = grow(rng, x + Math.cos(a) * r * 0.85, y + Math.sin(a) * r * 0.85, a, (40 + rng() * 85) * size * reach, (1.7 + rng() * 1.1) * size, 0, size, c.dend);
    // follow the first fork out to a tip
    const p: number[] = [];
    for (let b: Branch | undefined = prim; b; b = b.kids[0]) for (let k = p.length ? 2 : 0; k < b.pts.length; k++) p.push(b.pts[k]);
    c.paths.push(branch(p, 1, 1));
  }
  return c;
}

/** The axon heads for a neighbour (or off the edge) and ends in a few short terminals. */
export function growAxon(rng: () => number, c: Cell, tx: number, ty: number) {
  const size = c.size;
  const step = 3.4 * size;
  let ang = Math.atan2(ty - c.y, tx - c.x) + (rng() - 0.5) * 0.9;
  let x = c.x + Math.cos(ang) * c.r;
  let y = c.y + Math.sin(ang) * c.r;
  const p = [x, y];
  const maxLen = Math.hypot(tx - c.x, ty - c.y) * 1.35 + 40 * size;
  for (let len = 0; len < maxLen; len += step) {
    let d = Math.atan2(ty - y, tx - x) - ang;
    d = Math.atan2(Math.sin(d), Math.cos(d));
    ang += d * 0.07 + (rng() - 0.5) * 0.2;
    x += Math.cos(ang) * step;
    y += Math.sin(ang) * step;
    p.push(x, y);
    if (Math.hypot(tx - x, ty - y) < 26 * size) break;
  }
  c.roots.push(Math.atan2(p[3] - c.y, p[2] - c.x));
  const ax = branch(p, 0.95 * size, 0.7 * size);
  c.axon.push(ax);
  const k = 3 + Math.floor(rng() * 3);
  for (let j = 0; j < k; j++) {
    let a = ang + (rng() - 0.5) * 2.4;
    let px = x;
    let py = y;
    const q = [px, py];
    const m = 3 + Math.floor(rng() * 6);
    for (let i = 0; i < m; i++) {
      a += (rng() - 0.5) * 0.5;
      px += Math.cos(a) * 3 * size;
      py += Math.sin(a) * 3 * size;
      q.push(px, py);
    }
    const tb = branch(q, 0.6 * size, 0.45 * size);
    tb.s0 = ax.len;
    c.axon.push(tb);
  }
}

export const grey = (v: number) => {
  const k = Math.round(255 * Math.max(0, Math.min(1, v)));
  return `rgb(${k},${k},${k})`;
};

export function strokeTapered(g: CanvasRenderingContext2D, b: Branch, tone: number) {
  const n = b.pts.length / 2;
  for (let i = 0; i < n - 1; i += 3) {
    const j = Math.min(n - 1, i + 3);
    const t = (i + j) / 2 / Math.max(1, n - 1);
    g.lineWidth = b.w0 + (b.w1 - b.w0) * t;
    g.strokeStyle = grey(tone * (0.8 - 0.3 * t));
    g.beginPath();
    g.moveTo(b.pts[2 * i], b.pts[2 * i + 1]);
    for (let k = i + 1; k <= j; k++) g.lineTo(b.pts[2 * k], b.pts[2 * k + 1]);
    g.stroke();
  }
}

export function paintCell(g: CanvasRenderingContext2D, c: Cell, tone: number, detail: boolean, seed: number) {
  const rng = rngFrom(seed);
  const halo = g.createRadialGradient(c.x, c.y, 0, c.x, c.y, c.r * 6);
  halo.addColorStop(0, `rgba(255,255,255,${0.16 * tone})`);
  halo.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = halo;
  g.fillRect(c.x - c.r * 6, c.y - c.r * 6, c.r * 12, c.r * 12);

  for (const b of c.axon) strokeTapered(g, b, tone * 0.85);
  for (const b of c.dend) strokeTapered(g, b, tone);
  if (detail) {
    // dendritic spines: tiny side ticks
    g.lineWidth = 0.55 * c.size;
    g.strokeStyle = grey(tone * 0.5);
    g.beginPath();
    for (const b of c.dend) {
      if (b.w0 > 2.2 * c.size) continue;
      for (let i = 1; i < b.pts.length / 2; i++) {
        if (rng() > 0.55) continue;
        const a = Math.atan2(b.pts[2 * i + 1] - b.pts[2 * i - 1], b.pts[2 * i] - b.pts[2 * i - 2]) + (rng() < 0.5 ? 1 : -1) * (Math.PI / 2 + (rng() - 0.5) * 0.8);
        const l = (1 + rng() * 1.8) * c.size;
        g.moveTo(b.pts[2 * i], b.pts[2 * i + 1]);
        g.lineTo(b.pts[2 * i] + Math.cos(a) * l, b.pts[2 * i + 1] + Math.sin(a) * l);
      }
    }
    g.stroke();
  }
  // boutons at the axon terminals
  g.fillStyle = grey(tone * 0.85);
  for (let i = 1; i < c.axon.length; i++) {
    const b = c.axon[i];
    const m = b.pts.length;
    g.beginPath();
    g.arc(b.pts[m - 2], b.pts[m - 1], 1.15 * c.size, 0, TAU);
    g.fill();
  }
  // cell body: round, pulled out where each process leaves it
  const ph = rng() * TAU;
  g.beginPath();
  for (let i = 0; i <= 32; i++) {
    const th = (i / 32) * TAU;
    let k = 1 + 0.05 * Math.sin(3 * th + ph);
    for (const a of c.roots) k += 0.34 * Math.max(0, Math.cos(th - a)) ** 8;
    const px = c.x + Math.cos(th) * c.r * k;
    const py = c.y + Math.sin(th) * c.r * k;
    if (i) g.lineTo(px, py);
    else g.moveTo(px, py);
  }
  const body = g.createRadialGradient(c.x - c.r * 0.3, c.y - c.r * 0.3, 0, c.x, c.y, c.r * 1.25);
  body.addColorStop(0, grey(tone));
  body.addColorStop(1, grey(tone * 0.5));
  g.fillStyle = body;
  g.fill();
  // nucleus and nucleolus
  g.fillStyle = "rgba(0,0,0,0.3)";
  g.beginPath();
  g.arc(c.x + c.r * 0.12, c.y - c.r * 0.08, c.r * 0.42, 0, TAU);
  g.fill();
  g.fillStyle = grey(tone * 0.92);
  g.beginPath();
  g.arc(c.x + c.r * 0.2, c.y - c.r * 0.14, c.r * 0.12, 0, TAU);
  g.fill();
}

/** The point at distance d along a branch; returns the index of the segment's end point. */
export function at(b: Branch, d: number, out: number[]) {
  const n = b.cum.length;
  d = Math.max(0, Math.min(b.len, d));
  let i = 1;
  while (i < n - 1 && b.cum[i] < d) i++;
  const s0 = b.cum[i - 1];
  const s1 = b.cum[i];
  const f = s1 > s0 ? (d - s0) / (s1 - s0) : 0;
  out[0] = b.pts[2 * i - 2] + (b.pts[2 * i] - b.pts[2 * i - 2]) * f;
  out[1] = b.pts[2 * i - 1] + (b.pts[2 * i + 1] - b.pts[2 * i - 1]) * f;
  return i;
}

const P = [0, 0];
const Q = [0, 0];
export function strokeSpan(ctx: CanvasRenderingContext2D, b: Branch, d0: number, d1: number) {
  const i0 = at(b, d0, P);
  const i1 = at(b, d1, Q);
  ctx.beginPath();
  ctx.moveTo(P[0], P[1]);
  for (let i = i0; i < i1; i++) ctx.lineTo(b.pts[2 * i], b.pts[2 * i + 1]);
  ctx.lineTo(Q[0], Q[1]);
  ctx.stroke();
}
/** A bright signal with a short tail; `inward` runs from the tip back to the cell body. */
export function pulse(ctx: CanvasRenderingContext2D, glow: HTMLCanvasElement, b: Branch, u: number, inward: boolean, amp: number, size: number) {
  const d = u * b.len;
  const tail = Math.min(b.len * 0.35, 46 * size);
  ctx.globalAlpha = 0.75 * amp;
  ctx.lineWidth = 1.5 * size;
  if (inward) strokeSpan(ctx, b, d, d + tail);
  else strokeSpan(ctx, b, d - tail, d);
  at(b, d, P);
  const gs = (12 + 12 * amp) * size;
  ctx.globalAlpha = Math.min(1, amp);
  ctx.drawImage(glow, P[0] - gs / 2, P[1] - gs / 2, gs, gs);
}


/** A set of small neuron pictures, each a whole cell centred in a square, for drawing many cells fast.
 *  Each comes with its shape, so signals can run along it when it is drawn large. */
export const SPRITE = 224;
export type NeuronSprite = { img: CanvasImageSource; cell: Cell; size: number; reach: number };
export function neuronSprites(count: number, seed: number, S = SPRITE): NeuronSprite[] {
  const rng = rngFrom(seed);
  const out: NeuronSprite[] = [];
  for (let v = 0; v < count; v++) {
    const cv = document.createElement("canvas");
    cv.width = cv.height = S;
    // CPU-backed: thousands of small strokes are cheap there
    const g = cv.getContext("2d", { willReadFrequently: true })!;
    g.lineCap = "round";
    g.lineJoin = "round";
    const k = S / SPRITE;
    const c = makeCell(rng, S / 2, S / 2, 1.1 * k, 2, 0.4);
    const a = rng() * TAU;
    growAxon(rng, c, S / 2 + Math.cos(a) * S * 0.3, S / 2 + Math.sin(a) * S * 0.3);
    paintCell(g, c, 1, true, 4517 + v * 61);
    let reach = 0;
    for (const b of [...c.dend, ...c.axon]) reach = Math.max(reach, b.s0 + b.len);
    const item: NeuronSprite = { img: cv, cell: c, size: S, reach };
    out.push(item);
    // a bitmap is uploaded once and then drawn cheaply
    createImageBitmap(cv)
      .then((bm) => (item.img = bm))
      .catch(() => {});
  }
  return out;
}

/** A cell drawn as it grows out from its body: g runs 0 to 1. Each branch shows the part the growth
 *  front has passed, and the tips that are still growing glow. Drawn in the cell's own coordinates. */
export function growCell(ctx: CanvasRenderingContext2D, glow: HTMLCanvasElement, c: Cell, reach: number, g: number, tips: boolean) {
  const front = g * reach * 1.02;
  ctx.strokeStyle = "#fff";
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  const lit: number[] = [];
  for (const b of c.axon.concat(c.dend)) {
    const vis = Math.min(b.len, front - b.s0);
    if (vis <= 0) continue;
    ctx.lineWidth = (b.w0 + b.w1) / 2;
    strokeSpan(ctx, b, 0, vis);
    if (tips && vis < b.len && b.w0 > 0.9 * c.size) {
      at(b, vis, P);
      lit.push(P[0], P[1]);
    }
  }
  // the cell body swells in first, a little past full size and back
  const k = Math.min(1, g * 4);
  const pop = 1 + 0.25 * Math.sin(Math.PI * k) * (1 - k * 0.5);
  ctx.fillStyle = "#fff";
  ctx.beginPath();
  ctx.arc(c.x, c.y, c.r * k * pop, 0, TAU);
  ctx.fill();
  const a = ctx.globalAlpha;
  for (let i = 0; i < lit.length; i += 2) {
    const gs = 9 * c.size;
    ctx.drawImage(glow, lit[i] - gs / 2, lit[i + 1] - gs / 2, gs, gs);
  }
  ctx.globalAlpha = a;
}
