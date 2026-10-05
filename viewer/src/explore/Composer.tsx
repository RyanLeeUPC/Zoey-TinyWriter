import { Dices, Sparkles, Square } from "lucide-react";
import { Segmented } from "../components/Panel";

export const STARTERS = [
  "Once upon a time, there was a little girl named Lily.",
  "Tom and his dog Max went to the park.",
  "The little bird was sad because",
  "One day, a big red dragon",
  'Mia looked at her mom and said, "',
];

export const LENGTHS = { short: 80, medium: 160, long: 320 } as const;
export type Length = keyof typeof LENGTHS;

export function Composer({
  prompt,
  onPrompt,
  temperature,
  onTemperature,
  length,
  onLength,
  busy,
  disabled,
  onWrite,
  onReroll,
  onStop,
}: {
  prompt: string;
  onPrompt: (p: string) => void;
  temperature: number;
  onTemperature: (t: number) => void;
  length: Length;
  onLength: (l: Length) => void;
  busy: boolean;
  disabled: boolean;
  onWrite: () => void;
  onReroll: () => void;
  onStop: () => void;
}) {
  return (
    <div className="rounded-2xl border border-line bg-surface p-4 shadow-sm sm:p-5">
      <label htmlFor="prompt" className="text-sm font-semibold">
        Start a story, and TinyWriter will continue it
      </label>
      <div className="mt-2 flex flex-col gap-3 sm:flex-row">
        <input
          id="prompt"
          value={prompt}
          onChange={(e) => onPrompt(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !busy && !disabled && onWrite()}
          placeholder="Once upon a time…"
          maxLength={300}
          className="min-w-0 flex-1 rounded-xl border border-line bg-surface-2 px-4 py-3 text-[16px] outline-none focus:border-accent focus:ring-2 focus:ring-accent-wash"
        />
        {busy ? (
          <button
            onClick={onStop}
            className="inline-flex items-center justify-center gap-2 rounded-xl border border-line bg-surface px-5 py-3 font-semibold hover:bg-surface-2"
          >
            <Square size={16} fill="currentColor" /> Stop
          </button>
        ) : (
          <div className="flex gap-2">
            <button
              onClick={onWrite}
              disabled={disabled || !prompt.trim()}
              className="inline-flex flex-1 items-center justify-center gap-2 rounded-xl bg-accent px-5 py-3 font-semibold text-white shadow-md transition-transform hover:scale-[1.02] disabled:opacity-50 disabled:hover:scale-100"
            >
              <Sparkles size={18} /> Write
            </button>
            <button
              onClick={onReroll}
              disabled={disabled || !prompt.trim()}
              title="Same start, different dice rolls"
              aria-label="Write another version"
              className="grid place-items-center rounded-xl border border-line bg-surface px-3.5 hover:bg-surface-2 disabled:opacity-50"
            >
              <Dices size={20} />
            </button>
          </div>
        )}
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        {STARTERS.map((s) => (
          <button
            key={s}
            onClick={() => onPrompt(s)}
            className={`rounded-full border px-3 py-1 text-sm transition-colors ${
              s === prompt ? "border-accent bg-accent-wash text-accent-ink" : "border-line text-ink-2 hover:bg-surface-2"
            }`}
          >
            {s.length > 34 ? `${s.slice(0, 32)}…` : s}
          </button>
        ))}
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-x-8 gap-y-3 border-t border-line pt-4 text-sm">
        <label className="flex items-center gap-3">
          <span className="font-medium" title="Technically: the sampling temperature">
            Creativity
          </span>
          <span className="text-xs text-muted">safe</span>
          <input
            type="range"
            min={0.3}
            max={1.3}
            step={0.1}
            value={temperature}
            onChange={(e) => onTemperature(+e.target.value)}
            className="w-32 accent-[var(--accent)]"
          />
          <span className="text-xs text-muted">wild</span>
          <span className="tabular w-8 text-ink-2">{temperature.toFixed(1)}</span>
        </label>
        <div className="flex items-center gap-3">
          <span className="font-medium">Length</span>
          <Segmented
            value={length}
            onChange={onLength}
            options={[
              { value: "short", label: "Short" },
              { value: "medium", label: "Medium" },
              { value: "long", label: "Long" },
            ]}
          />
        </div>
      </div>
    </div>
  );
}
