export type Segment = { text: string; changed: boolean };

/** Word-level diff: marks the words in `b` that are not in the longest common sequence with `a`. */
export function diffWords(a: string, b: string): Segment[] {
  const A = a.split(/(\s+)/).filter(Boolean);
  const B = b.split(/(\s+)/).filter(Boolean);
  const n = A.length;
  const m = B.length;
  if (n * m > 400_000) return [{ text: b, changed: a !== b }];
  const dp: Uint16Array[] = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--) dp[i][j] = A[i] === B[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const out: Segment[] = [];
  const push = (text: string, changed: boolean) => {
    const last = out[out.length - 1];
    if (last && last.changed === changed) last.text += text;
    else out.push({ text, changed });
  };
  let i = 0;
  let j = 0;
  while (j < m) {
    if (i < n && A[i] === B[j]) {
      push(B[j], false);
      i++;
      j++;
    } else if (i < n && dp[i + 1][j] >= dp[i][j + 1]) i++;
    else {
      push(B[j], !/^\s+$/.test(B[j]));
      j++;
    }
  }
  return out;
}

/** Share of b's words that differ from a, 0..1. */
export function changedShare(segs: Segment[]): number {
  let all = 0;
  let ch = 0;
  for (const s of segs) {
    const w = s.text.split(/\s+/).filter(Boolean).length;
    all += w;
    if (s.changed) ch += w;
  }
  return all ? ch / all : 0;
}
