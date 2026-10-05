import { Check } from "lucide-react";
import { bandColor } from "../lib/colors";
import { useTheme } from "../lib/theme";
import { nameToken, pct, showToken } from "../lib/format";

/**
 * Horizontal bars of TinyWriter's top guesses for the next token.
 * If we know the right answer, it gets a check mark.
 */
export function GuessBars({
  ids,
  probs,
  tokens,
  correct,
  barColor,
}: {
  ids: number[];
  probs: number[];
  tokens: string[];
  correct?: number;
  /** Color each bar by its probability (defaults to the confidence bands). */
  barColor?: (p: number) => string;
}) {
  const { theme } = useTheme();
  const color = barColor ?? ((p: number) => bandColor(theme, p));
  return (
    <ol className="grid grid-cols-[minmax(2.25rem,max-content)_1fr_3.5rem] items-center gap-x-2 gap-y-1.5">
      {ids.map((id, i) => {
        const p = probs[i];
        const right = id === correct;
        return (
          <li key={i} className="contents" title={`${nameToken(tokens[id])}: ${pct(p)}`}>
            <span
              className={`grid h-7 place-items-center whitespace-pre rounded-md border px-1.5 font-mono text-sm ${
                right ? "border-good bg-good/10 font-bold text-good-ink" : "border-line bg-surface-2 text-ink"
              }`}
            >
              {showToken(tokens[id])}
            </span>
            <div className="relative h-5">
              <div
                className="absolute inset-y-0 left-0 rounded-r-[4px] transition-[width] duration-300"
                style={{ width: `max(2px, ${p * 100}%)`, background: color(p) }}
              />
            </div>
            <span className="tabular flex items-center justify-end gap-1 text-sm text-ink-2">
              {right && <Check size={14} className="text-good-ink" aria-label="correct" />}
              {pct(p)}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
