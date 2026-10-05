import { Pause, Play, RotateCcw, SkipBack, SkipForward } from "lucide-react";
import type { Manifest } from "../lib/types";
import { compact, int } from "../lib/format";

/**
 * The time machine. Drag through training, or press play to watch TinyWriter learn.
 * The faint curve behind the slider is the loss at each snapshot, so you can
 * see where the big changes happen.
 */
export function Timeline({
  manifest,
  index,
  onIndex,
  playing,
  onPlaying,
  hasPlayed,
}: {
  manifest: Manifest;
  index: number;
  onIndex: (i: number) => void;
  playing: boolean;
  onPlaying: (p: boolean) => void;
  hasPlayed: boolean;
}) {
  const cps = manifest.checkpoints;
  const cp = cps[index];
  const last = cps.length - 1;
  const atEnd = index === last;
  const tokensRead = cp.step * manifest.train.tokens_per_step;

  return (
    <div className="sticky top-0 z-20 -mx-4 border-b border-line bg-page/85 px-4 py-3 backdrop-blur-md sm:-mx-6 sm:px-6">
      <div className="mx-auto flex max-w-7xl items-center gap-3 sm:gap-5">
        <div className="flex items-center gap-1">
          <IconButton label="Previous snapshot" onClick={() => onIndex(Math.max(0, index - 1))} disabled={index === 0}>
            <SkipBack size={18} />
          </IconButton>
          <button
            onClick={() => (atEnd && !playing ? (onIndex(0), onPlaying(true)) : onPlaying(!playing))}
            aria-label={playing ? "Pause" : atEnd ? "Replay" : "Play"}
            className={`grid size-12 shrink-0 place-items-center rounded-full bg-accent text-white shadow-md transition-transform hover:scale-105 active:scale-95 ${
              hasPlayed ? "" : "nudge"
            }`}
          >
            {playing ? <Pause size={22} fill="currentColor" /> : atEnd ? <RotateCcw size={20} /> : <Play size={22} fill="currentColor" className="ml-0.5" />}
          </button>
          <IconButton label="Next snapshot" onClick={() => onIndex(Math.min(last, index + 1))} disabled={atEnd}>
            <SkipForward size={18} />
          </IconButton>
        </div>

        <div className="hidden w-40 shrink-0 sm:block">
          <div className="text-xs font-medium uppercase tracking-wide text-muted">Training step</div>
          <div className="tabular text-xl font-semibold">
            {int(cp.step)} <span className="text-sm font-normal text-muted">/ {int(manifest.train.max_steps)}</span>
          </div>
          <div className="tabular text-xs text-ink-2">{compact(tokensRead)} tokens read</div>
        </div>

        <div className="relative h-14 min-w-0 flex-1">
          <LossBackdrop manifest={manifest} index={index} />
          <input
            type="range"
            className="scrubber absolute inset-0 h-14"
            min={0}
            max={last}
            value={index}
            onChange={(e) => {
              onPlaying(false);
              onIndex(+e.target.value);
            }}
            aria-label="Training timeline"
            aria-valuetext={`Step ${cp.step}`}
          />
          <div className="pointer-events-none absolute -bottom-0.5 left-0 right-0 flex justify-between text-[11px] text-muted">
            <span>start</span>
            {!hasPlayed && <span className="font-medium text-accent-ink">◀ drag me, or press play ▶</span>}
            <span>end</span>
          </div>
        </div>

        <div className="hidden w-24 shrink-0 text-right md:block">
          <div className="text-xs font-medium uppercase tracking-wide text-muted">Loss</div>
          <div className="tabular text-xl font-semibold">{cp.val.toFixed(2)}</div>
          <div className="text-xs text-ink-2">lower is better</div>
        </div>
      </div>
      {/* compact readout for phones */}
      <div className="tabular mt-1 flex justify-between text-xs text-ink-2 sm:hidden">
        <span>
          Step {int(cp.step)} / {int(manifest.train.max_steps)}
        </span>
        <span>Loss {cp.val.toFixed(2)}</span>
      </div>
    </div>
  );
}

function IconButton({
  label,
  onClick,
  disabled,
  children,
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={label}
      className="grid size-9 place-items-center rounded-full text-ink-2 hover:bg-surface-2 disabled:opacity-30"
    >
      {children}
    </button>
  );
}

/** Loss per snapshot, drawn behind the slider. Past = solid, future = faint. */
function LossBackdrop({ manifest, index }: { manifest: Manifest; index: number }) {
  const cps = manifest.checkpoints;
  const vals = cps.map((c) => c.val);
  const hi = Math.max(...vals);
  const lo = Math.min(...vals);
  const W = 1000;
  const H = 48;
  const x = (i: number) => (i / (cps.length - 1)) * W;
  const y = (v: number) => 4 + (1 - (v - lo) / (hi - lo || 1)) * (H - 8);
  const line = vals.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(v)}`).join("");
  const area = `${line}L${W},${H}L0,${H}Z`;
  const split = x(index);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="absolute inset-x-0 top-0 h-12 w-full" aria-hidden>
      <defs>
        <clipPath id="tl-past">
          <rect x={0} y={0} width={split} height={H} />
        </clipPath>
      </defs>
      <line x1={0} x2={W} y1={H - 0.5} y2={H - 0.5} stroke="var(--axis)" vectorEffect="non-scaling-stroke" />
      <path d={area} fill="var(--series-1)" opacity={0.05} />
      <path d={line} fill="none" stroke="var(--axis)" strokeWidth={2} vectorEffect="non-scaling-stroke" />
      <g clipPath="url(#tl-past)">
        <path d={area} fill="var(--series-1)" opacity={0.12} />
        <path d={line} fill="none" stroke="var(--series-1)" strokeWidth={2} vectorEffect="non-scaling-stroke" />
      </g>
    </svg>
  );
}
