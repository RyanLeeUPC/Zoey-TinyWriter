import { useEffect, useRef, useState } from "react";
import { probToT, rampColor, useTheme } from "../lib/theme";

export interface HeatmapHover {
  row: number;
  col: number;
  x: number;
  y: number;
}

/**
 * A grid of probabilities drawn on a canvas (fast enough to redraw on every
 * frame of the time-lapse). Rows and columns are indices into `rowIds`/`colIds`.
 */
export function Heatmap({
  matrix,
  rowIds,
  colIds,
  rowLabels,
  colLabels,
  selectedRow,
  onSelectRow,
  renderTooltip,
}: {
  matrix: number[][];
  rowIds: number[];
  colIds: number[];
  rowLabels: string[];
  colLabels: string[];
  selectedRow?: number;
  onSelectRow?: (rowId: number) => void;
  renderTooltip?: (rowId: number, colId: number) => React.ReactNode;
}) {
  const { theme } = useTheme();
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(600);
  const [hover, setHover] = useState<HeatmapHover | null>(null);

  const label = 22; // space for row/column labels
  const cell = Math.max(6, Math.floor((width - label) / colIds.length));
  const gap = cell >= 12 ? 2 : 1;
  const W = label + cell * colIds.length;
  const H = label + cell * rowIds.length;

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(e.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    const ctx = canvas.getContext("2d")!;
    ctx.scale(dpr, dpr);
    const css = getComputedStyle(document.documentElement);
    const ink = css.getPropertyValue("--ink-2").trim();
    const muted = css.getPropertyValue("--muted").trim();
    const surface = css.getPropertyValue("--surface").trim();

    ctx.fillStyle = surface;
    ctx.fillRect(0, 0, W, H);
    ctx.font = `${Math.min(13, cell - 1)}px ui-monospace, "Cascadia Code", Menlo, Consolas, monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    colIds.forEach((_, c) => {
      ctx.fillStyle = muted;
      ctx.fillText(colLabels[c], label + c * cell + cell / 2, label / 2);
    });
    rowIds.forEach((rid, r) => {
      const selected = rid === selectedRow;
      ctx.fillStyle = selected ? ink : muted;
      ctx.font = `${selected ? "bold " : ""}${Math.min(13, cell - 1)}px ui-monospace, "Cascadia Code", Menlo, Consolas, monospace`;
      ctx.fillText(rowLabels[r], label / 2, label + r * cell + cell / 2);
      const row = matrix[rid];
      colIds.forEach((cid, c) => {
        ctx.fillStyle = rampColor(theme, probToT(row[cid]));
        ctx.fillRect(label + c * cell, label + r * cell, cell - gap, cell - gap);
      });
    });

    if (selectedRow !== undefined) {
      const r = rowIds.indexOf(selectedRow);
      if (r >= 0) {
        ctx.strokeStyle = css.getPropertyValue("--ink").trim();
        ctx.lineWidth = 2;
        ctx.strokeRect(label - 1, label + r * cell - 1, cell * colIds.length + 1, cell + 1 - gap + 1);
      }
    }
  }, [matrix, rowIds, colIds, rowLabels, colLabels, selectedRow, theme, cell, gap, W, H]);

  const cellAt = (e: React.MouseEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const c = Math.floor((x - label) / cell);
    const r = Math.floor((y - label) / cell);
    if (c < 0 || r < 0 || c >= colIds.length || r >= rowIds.length) return null;
    return { row: r, col: c, x, y };
  };

  return (
    <div ref={wrapRef} className="relative w-full">
      <canvas
        ref={canvasRef}
        style={{ width: W, height: H }}
        className="cursor-pointer"
        onMouseMove={(e) => setHover(cellAt(e))}
        onMouseLeave={() => setHover(null)}
        onClick={(e) => {
          const h = cellAt(e);
          if (h && onSelectRow) onSelectRow(rowIds[h.row]);
        }}
      />
      {hover && renderTooltip && (
        <div
          className="pointer-events-none absolute z-10 rounded-lg border border-line bg-surface px-3 py-2 text-sm shadow-lg"
          style={{
            left: Math.min(hover.x + 14, W - 200),
            top: hover.y + 14,
          }}
        >
          {renderTooltip(rowIds[hover.row], colIds[hover.col])}
        </div>
      )}
    </div>
  );
}
