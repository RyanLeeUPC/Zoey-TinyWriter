import { probToT, rampColor, useTheme } from "../lib/theme";

/** The key for every probability color in the app. */
export function RampLegend({ left, right, className = "" }: { left: string; right: string; className?: string }) {
  const { theme } = useTheme();
  const ticks = [0, 0.05, 0.25, 0.5, 1];
  const stops = Array.from({ length: 11 }, (_, i) => `${rampColor(theme, i / 10)} ${i * 10}%`).join(",");
  return (
    <div className={`flex items-center gap-3 text-xs text-ink-2 ${className}`}>
      <span>{left}</span>
      <div className="relative mx-2 w-48">
        <div className="h-2.5 rounded-full border border-line" style={{ background: `linear-gradient(90deg, ${stops})` }} />
        <div className="relative h-3.5">
          {ticks.map((p) => (
            <span
              key={p}
              className="tabular absolute top-0.5 -translate-x-1/2 text-[10px] text-muted"
              style={{ left: `${probToT(p) * 100}%` }}
            >
              {p === 0 ? "0" : `${Math.round(p * 100)}%`}
            </span>
          ))}
        </div>
      </div>
      <span>{right}</span>
    </div>
  );
}
