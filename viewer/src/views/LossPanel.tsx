import { useMemo, useRef, useState } from "react";
import { Panel } from "../components/Panel";
import type { Manifest } from "../lib/types";
import { int, unitOf } from "../lib/format";

const W = 720;
const H = 260;
const M = { top: 16, right: 70, bottom: 34, left: 40 };

/**
 * Loss over training. The x-axis is logarithmic (like the timeline), so the
 * fast early learning isn't squashed into a sliver on the left.
 */
export function LossPanel({ manifest, step }: { manifest: Manifest; step: number }) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoverStep, setHoverStep] = useState<number | null>(null);

  const vocab = manifest.tokenizer.tokens.length;
  const unit = unitOf(manifest.tokenizer.name);
  const randomLoss = Math.log(vocab);
  const train = manifest.loss;
  const val = manifest.checkpoints;
  const maxStep = manifest.train.max_steps;

  const { x, y, yTicks, xTicks } = useMemo(() => {
    const lx = (s: number) => Math.log10(s + 1);
    const xMax = lx(maxStep);
    const x = (s: number) => M.left + (lx(s) / xMax) * (W - M.left - M.right);
    const all = [...train.map((p) => p.train), ...val.map((p) => p.val), randomLoss];
    const yMax = Math.ceil(Math.max(...all));
    const yMin = Math.floor(Math.min(...all));
    const y = (v: number) => M.top + (1 - (v - yMin) / (yMax - yMin)) * (H - M.top - M.bottom);
    const yTicks = Array.from({ length: yMax - yMin + 1 }, (_, i) => yMin + i);
    const xTicks = [0, 1, 10, 100, 1000, 10000, 100000].filter((s) => s <= maxStep);
    return { x, y, yTicks, xTicks };
  }, [train, val, randomLoss, maxStep]);

  const path = (pts: [number, number][]) => pts.map(([s, v], i) => `${i ? "L" : "M"}${x(s)},${y(v)}`).join("");
  const trainPath = path(train.map((p) => [p.step, p.train]));
  const valPath = path(val.map((p) => [p.step, p.val]));
  const lastVal = val[val.length - 1];
  const lastTrain = train[train.length - 1];

  // Hover: snap to the nearest snapshot.
  const nearest = (s: number) => val.reduce((a, b) => (Math.abs(b.step - s) < Math.abs(a.step - s) ? b : a));
  const hovered = hoverStep === null ? null : nearest(hoverStep);
  const onMove = (e: React.MouseEvent) => {
    const rect = svgRef.current!.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const frac = Math.min(1, Math.max(0, (px - M.left) / (W - M.left - M.right)));
    setHoverStep(10 ** (frac * Math.log10(maxStep + 1)) - 1);
  };

  return (
    <Panel
      title="Loss: how surprised is TinyWriter?"
      subtitle="The single number training tries to push down."
      help={
        <>
          <p>
            <b>Loss</b> measures how surprised TinyWriter is, on average, by the real next {unit.one}. If TinyWriter gives the right
            {unit.one} 100% probability, loss is 0. The less probability it gave, the higher the loss.
          </p>
          <p>
            The dashed line is <b>random guessing</b>: picking uniformly from all {int(vocab)} {unit.vocab} gives a loss of
            ln({vocab}) ≈ {randomLoss.toFixed(2)}. TinyWriter actually starts <i>worse</i> than that, because its random
            starting table is confidently wrong in random directions.
          </p>
          <p>
            <b>Training loss</b> is measured on text TinyWriter learns from; <b>validation loss</b> on text it never sees.
            When they stay together, TinyWriter is learning general patterns rather than memorizing. When the curve goes
            flat, the model has learned all it <i>can</i>
            {manifest.model.name === "bigram"
              ? ". For a bigram model, that's about 2.3. Beating it needs a model that can see more than one letter back."
              : " at this size."}
          </p>
        </>
      }
    >
      <div className="mb-2 flex flex-wrap gap-4 text-sm text-ink-2">
        <LegendKey color="var(--series-1)" label="Training loss" />
        <LegendKey color="var(--series-2)" label="Validation loss" dot />
        <LegendKey color="var(--muted)" label="Random guessing" dashed />
      </div>
      <div className="relative">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="w-full"
          onMouseMove={onMove}
          onMouseLeave={() => setHoverStep(null)}
          role="img"
          aria-label={`Loss chart. Final validation loss ${lastVal.val.toFixed(2)} after ${int(maxStep)} steps.`}
        >
          {yTicks.map((t) => (
            <g key={t}>
              <line x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
              <text x={M.left - 8} y={y(t)} textAnchor="end" dominantBaseline="middle" className="tabular fill-muted text-[11px]">
                {t}
              </text>
            </g>
          ))}
          {xTicks.map((s) => (
            <text key={s} x={x(s)} y={H - M.bottom + 18} textAnchor="middle" className="tabular fill-muted text-[11px]">
              {int(s)}
            </text>
          ))}
          <text x={(W - M.right + M.left) / 2} y={H - 2} textAnchor="middle" className="fill-muted text-[11px]">
            training step (log scale)
          </text>

          <line
            x1={M.left}
            x2={W - M.right}
            y1={y(randomLoss)}
            y2={y(randomLoss)}
            stroke="var(--muted)"
            strokeDasharray="5 4"
          />

          {/* playhead: where the timeline is */}
          <line x1={x(step)} x2={x(step)} y1={M.top} y2={H - M.bottom} stroke="var(--ink)" strokeWidth={2} opacity={0.5} />

          <path d={trainPath} fill="none" stroke="var(--series-1)" strokeWidth={2} strokeLinejoin="round" />
          <path d={valPath} fill="none" stroke="var(--series-2)" strokeWidth={2} strokeLinejoin="round" />
          {val.map((p) => (
            <circle key={p.step} cx={x(p.step)} cy={y(p.val)} r={2.5} fill="var(--series-2)" />
          ))}

          {/* direct labels at the end */}
          <text x={W - M.right + 8} y={y(lastVal.val) - 7} dominantBaseline="middle" className="tabular fill-ink-2 text-[12px]">
            val {lastVal.val.toFixed(2)}
          </text>
          <text x={W - M.right + 8} y={y(lastTrain.train) + 9} dominantBaseline="middle" className="tabular fill-ink-2 text-[12px]">
            train {lastTrain.train.toFixed(2)}
          </text>

          {hovered && (
            <g>
              <line x1={x(hovered.step)} x2={x(hovered.step)} y1={M.top} y2={H - M.bottom} stroke="var(--axis)" />
              <circle cx={x(hovered.step)} cy={y(hovered.val)} r={5} fill="var(--series-2)" stroke="var(--surface)" strokeWidth={2} />
            </g>
          )}
        </svg>
        {hovered && (
          <div
            className="pointer-events-none absolute top-2 rounded-lg border border-line bg-surface px-3 py-2 text-sm shadow-lg"
            style={{ left: `clamp(0px, calc(${(x(hovered.step) / W) * 100}% + 12px), calc(100% - 160px))` }}
          >
            <div className="tabular font-semibold">Step {int(hovered.step)}</div>
            <div className="tabular text-ink-2">Train {hovered.train.toFixed(3)}</div>
            <div className="tabular text-ink-2">Validation {hovered.val.toFixed(3)}</div>
          </div>
        )}
      </div>
    </Panel>
  );
}

function LegendKey({ color, label, dashed, dot }: { color: string; label: string; dashed?: boolean; dot?: boolean }) {
  return (
    <span className="inline-flex items-center gap-2">
      <svg width="22" height="10" aria-hidden>
        <line x1="1" x2="21" y1="5" y2="5" stroke={color} strokeWidth="2" strokeDasharray={dashed ? "4 3" : undefined} />
        {dot && <circle cx="11" cy="5" r="3" fill={color} />}
      </svg>
      {label}
    </span>
  );
}
