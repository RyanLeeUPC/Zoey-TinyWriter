import { useState } from "react";
import { useTheme, type Theme } from "../lib/theme";

/*
  A token's embedding drawn as a grid: one square per number. Diverging
  colors: red for negative, blue for positive, gray near zero - the same
  scale for every token, so fingerprints can be compared fairly.

  Dark mode uses a steeper curve, so middling values read
  as clear reds and blues instead of fading into the dark background.
*/
const POLES: Record<Theme, { neg: string; pos: string; mid: string; curve: number }> = {
  light: { neg: "#e34948", pos: "#2a78d6", mid: "#f0efec", curve: 0.8 },
  dark: { neg: "#e66767", pos: "#3987e5", mid: "#383835", curve: 0.55 },
};

const hex = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

export function divergingColor(theme: Theme, v: number): string {
  const { neg, pos, mid, curve } = POLES[theme];
  const t = Math.min(1, Math.abs(v)) ** curve;
  const a = hex(mid);
  const b = hex(v < 0 ? neg : pos);
  const c = a.map((x, i) => Math.round(x + (b[i] - x) * t));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

export function Fingerprint({
  vector,
  scale,
  order,
  columns = 32,
  label,
}: {
  vector: Float32Array;
  scale: number;
  /** Draw the numbers in this order (e.g. sorted by another token's numbers). */
  order?: number[];
  columns?: number;
  label: string;
}) {
  const { theme } = useTheme();
  const [hover, setHover] = useState<number | null>(null);
  const idx = order ?? Array.from(vector, (_, i) => i);
  const rows = Array.from({ length: Math.ceil(idx.length / columns) }, (_, r) => idx.slice(r * columns, (r + 1) * columns));
  // When sorted, each row's average shows the overall trend at a glance.
  const showAverages = order !== undefined;

  return (
    <div>
      <div className="flex gap-1.5">
        <div
          className="grid min-w-0 flex-1 gap-px overflow-hidden rounded-lg border border-line bg-line"
          style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
          onMouseLeave={() => setHover(null)}
          role="img"
          aria-label={`The ${vector.length} learned numbers for ${label}`}
        >
          {idx.map((i) => (
            <div
              key={i}
              onMouseEnter={() => setHover(i)}
              className="aspect-square"
              style={{ background: divergingColor(theme, vector[i] / scale) }}
            />
          ))}
        </div>
        {showAverages && (
          <div
            className="grid w-4 shrink-0 gap-px overflow-hidden rounded-md border border-line bg-line"
            style={{ gridTemplateRows: `repeat(${rows.length}, minmax(0, 1fr))` }}
            title="Average of each row"
          >
            {rows.map((row, r) => {
              const mean = row.reduce((sum, i) => sum + vector[i], 0) / row.length;
              // Averages are smaller than single numbers, so they get a tighter scale.
              return <div key={r} style={{ background: divergingColor(theme, mean / (scale * 0.35)) }} />;
            })}
          </div>
        )}
      </div>
      <div className="tabular mt-1 h-4 text-xs text-muted">
        {hover !== null ? `Number #${hover + 1} of ${vector.length}: ${vector[hover].toFixed(3)}` : " "}
      </div>
    </div>
  );
}

export function FingerprintLegend() {
  const { theme } = useTheme();
  const stops = Array.from({ length: 11 }, (_, i) => `${divergingColor(theme, -1 + i / 5)} ${i * 10}%`).join(",");
  return (
    <div className="flex items-center gap-2 text-xs text-ink-2">
      <span>negative</span>
      <div className="h-2.5 w-40 rounded-full border border-line" style={{ background: `linear-gradient(90deg, ${stops})` }} />
      <span>positive</span>
      <span className="text-muted">· gray ≈ 0</span>
    </div>
  );
}
