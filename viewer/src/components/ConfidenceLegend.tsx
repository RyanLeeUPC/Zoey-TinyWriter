import { BANDS, bandIndex, bandSwatch } from "../lib/colors";
import { useTheme } from "../lib/theme";

/** The key for confidence colors: four named bands. */
export function ConfidenceLegend({ className = "" }: { className?: string }) {
  const { theme } = useTheme();
  return (
    <div className={`flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-ink-2 ${className}`}>
      {BANDS.map((b, i) => (
        <span key={b.label} className="inline-flex items-center gap-1.5" title={b.range}>
          <span className="size-3 rounded-[3px]" style={{ background: bandSwatch(theme, i) }} />
          {b.label}
          <span className="text-muted">{b.range}</span>
        </span>
      ))}
    </div>
  );
}

/** A small "40% · likely" chip. */
export function ConfidenceChip({ p }: { p: number }) {
  const { theme } = useTheme();
  const i = bandIndex(p);
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border border-line bg-surface-2 px-2 py-0.5 text-xs font-medium text-ink">
      <span className="size-2.5 rounded-full" style={{ background: bandSwatch(theme, i) }} />
      {BANDS[i].label}
    </span>
  );
}
