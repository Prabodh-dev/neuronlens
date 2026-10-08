"use client";

import { useState } from "react";
import { showToken } from "@/lib/format";
import { tokenColors } from "@/lib/heat";

export type HeatToken = {
  text: string;
  /** Raw activation, for the accessible label (meaning never relies on shading alone). */
  value?: number;
  /** 0..1 share of the reference activation. */
  heat: number;
  peak?: boolean;
  skip?: boolean;
  tip?: React.ReactNode;
};

type Props = {
  tokens: HeatToken[];
  selected?: number | null;
  onSelect?: (i: number) => void;
  className?: string;
  label?: string;
};

/** Text as the model saw it, each token shaded by activation. Whitespace is kept. */
export function HeatText({ tokens, selected, onSelect, className, label }: Props) {
  const [hover, setHover] = useState<{ i: number; x: number; y: number } | null>(null);
  const tipNode = hover !== null ? tokens[hover.i]?.tip : null;

  return (
    <div className={className} aria-label={label} onMouseLeave={() => setHover(null)}>
      {tokens.map((t, i) => {
        const a = Math.max(0, Math.min(1, t.heat));
        const cls = ["tok", t.peak ? "peak" : "", t.skip ? "skip" : "", selected === i ? "selected" : ""].join(" ").trim();
        const common = {
          className: cls,
          title: t.value !== undefined ? `token "${t.text}", activation ${t.value.toFixed(2)}` : undefined,
          "aria-label": t.value !== undefined ? `token "${t.text}", activation ${t.value.toFixed(2)}` : undefined,
          style: t.skip ? undefined : tokenColors(a),
          onMouseMove: (e: React.MouseEvent) => setHover({ i, x: e.clientX, y: e.clientY }),
        };
        return onSelect ? (
          <button key={i} type="button" {...common} onClick={() => onSelect(i)} aria-pressed={selected === i}>
            {showToken(t.text)}
          </button>
        ) : (
          <span key={i} {...common}>
            {showToken(t.text)}
          </span>
        );
      })}
      {hover && tipNode && (
        <div className="tip" style={{ left: Math.min(hover.x + 14, window.innerWidth - 300), top: hover.y + 16 }}>
          {tipNode}
        </div>
      )}
    </div>
  );
}
