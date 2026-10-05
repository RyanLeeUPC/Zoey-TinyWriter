import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Maximize2, Minus, Plus } from "lucide-react";
import { useTheme } from "../lib/theme";
import { compact } from "../lib/format";
import { isWordLike, wordOf, type DictionaryMap } from "./data";

interface View {
  cx: number; // map coordinates at the center of the canvas
  cy: number;
  zoom: number;
}

const HOME: View = { cx: 0, cy: 0, zoom: 1 };
const LABEL_CANDIDATES = 1500; // only the most common word tokens get labels

/**
 * Every token TinyWriter knows, laid out so that tokens with similar embeddings sit
 * near each other. Scroll to zoom, drag to pan, click a dot to look it up.
 */
export function VocabMap({
  map,
  vocab,
  selected,
  neighbors,
  onPick,
}: {
  map: DictionaryMap;
  vocab: string[];
  selected: number;
  neighbors: number[];
  onPick: (id: number) => void;
}) {
  const { theme } = useTheme();
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ w: 600, h: 420 });
  const [view, setView] = useState<View>(HOME);
  const [hover, setHover] = useState<{ id: number; x: number; y: number } | null>(null);
  const drag = useRef<{ x: number; y: number; view: View; moved: boolean } | null>(null);

  // Word tokens, most common first: the ones that get labels.
  const labelOrder = useMemo(
    () =>
      vocab
        .map((_, id) => id)
        .filter((id) => isWordLike(vocab[id]) && vocab[id].startsWith(" "))
        .sort((a, b) => map.count[b] - map.count[a])
        .slice(0, LABEL_CANDIDATES),
    [vocab, map],
  );
  const maxLog = useMemo(() => Math.log(Math.max(...map.count) + 1), [map]);

  useLayoutEffect(() => {
    const el = wrapRef.current!;
    const fit = (w: number) => setSize({ w, h: Math.max(320, Math.min(560, w * 0.7)) });
    fit(el.clientWidth); // measure right away, then follow any resizes
    const ro = new ResizeObserver(([e]) => fit(e.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Zoom in on the selected word whenever it changes.
  useEffect(() => {
    setView({ cx: map.xy[selected * 2], cy: map.xy[selected * 2 + 1], zoom: 3.5 });
  }, [selected, map]);

  const base = Math.min(size.w, size.h) / 2.1;
  const toScreen = useCallback(
    (id: number) => ({
      x: size.w / 2 + (map.xy[id * 2] - view.cx) * base * view.zoom,
      y: size.h / 2 + (map.xy[id * 2 + 1] - view.cy) * base * view.zoom,
    }),
    [size, view, base, map],
  );

  useEffect(() => {
    const canvas = canvasRef.current!;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size.w * dpr;
    canvas.height = size.h * dpr;
    const ctx = canvas.getContext("2d")!;
    ctx.scale(dpr, dpr);
    const css = getComputedStyle(document.documentElement);
    const v = (name: string) => css.getPropertyValue(name).trim();
    ctx.fillStyle = v("--surface");
    ctx.fillRect(0, 0, size.w, size.h);
    // Text and dots grow as you zoom in: 1x for the whole map, up to 2x up close.
    const grow = Math.min(2, Math.max(1, 1 + 0.3 * Math.log2(view.zoom)));

    // Every token: a small dot, stronger for common tokens.
    const muted = v("--muted");
    for (let id = 0; id < vocab.length; id++) {
      const { x, y } = toScreen(id);
      if (x < -5 || y < -5 || x > size.w + 5 || y > size.h + 5) continue;
      ctx.globalAlpha = 0.15 + 0.6 * (Math.log(map.count[id] + 1) / maxLog);
      ctx.fillStyle = muted;
      ctx.beginPath();
      ctx.arc(x, y, 1.8 * Math.sqrt(grow), 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    const sans = "system-ui, -apple-system, 'Segoe UI', sans-serif";
    // Labels: selected word, then its neighbors, then common words - skipping any that would overlap.
    // Keep the zoom buttons' corner free of labels.
    const placed: { x: number; y: number; w: number; h: number }[] = [{ x: size.w - 52, y: 0, w: 52, h: 124 }];
    const overlaps = (r: { x: number; y: number; w: number; h: number }) =>
      placed.some((p) => r.x < p.x + p.w && r.x + r.w > p.x && r.y < p.y + p.h && r.y + r.h > p.y);
    const label = (id: number, weight: string, px: number, color: string, force = false) => {
      const { x, y } = toScreen(id);
      if (x < 0 || y < 0 || x > size.w || y > size.h) return;
      const fontPx = px * grow;
      ctx.font = `${weight} ${fontPx}px ${sans}`;
      const text = wordOf(vocab[id]);
      const w = ctx.measureText(text).width;
      const r = { x: x + 5, y: y - fontPx * 0.6, w: w + 4, h: fontPx * 1.2 };
      if (!force && overlaps(r)) return;
      placed.push(r);
      ctx.fillStyle = color;
      ctx.fillText(text, x + 6, y + fontPx * 0.35);
    };

    const accent = v("--accent");
    const ink = v("--ink");
    const ink2 = v("--ink-2");
    // Highlighted dots first (so labels can be placed around them).
    for (const id of neighbors) {
      const { x, y } = toScreen(id);
      ctx.fillStyle = accent;
      ctx.beginPath();
      ctx.arc(x, y, 4 * Math.sqrt(grow), 0, Math.PI * 2);
      ctx.fill();
    }
    const s = toScreen(selected);
    ctx.fillStyle = v("--surface");
    ctx.beginPath();
    ctx.arc(s.x, s.y, 9 * Math.sqrt(grow), 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = ink;
    ctx.beginPath();
    ctx.arc(s.x, s.y, 7 * Math.sqrt(grow), 0, Math.PI * 2);
    ctx.fill();

    label(selected, "bold", 16, ink, true);
    for (const id of neighbors) label(id, "600", 13, ink);
    for (const id of labelOrder) label(id, "400", 12, ink2);
  }, [size, view, theme, map, vocab, selected, neighbors, labelOrder, maxLog, toScreen]);

  // Zoom with the mouse wheel, around the cursor.
  useEffect(() => {
    const canvas = canvasRef.current!;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left - size.w / 2;
      const my = e.clientY - rect.top - size.h / 2;
      setView((vw) => {
        const zoom = Math.min(40, Math.max(0.8, vw.zoom * Math.exp(-e.deltaY * 0.0015)));
        const k = base * vw.zoom;
        const k2 = base * zoom;
        return { zoom, cx: vw.cx + mx / k - mx / k2, cy: vw.cy + my / k - my / k2 };
      });
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, [size, base]);

  const nearest = (px: number, py: number) => {
    let best = -1;
    let bestD = 64; // within 8px
    for (let id = 0; id < vocab.length; id++) {
      const { x, y } = toScreen(id);
      const d = (x - px) ** 2 + (y - py) ** 2;
      if (d < bestD) {
        bestD = d;
        best = id;
      }
    }
    return best;
  };

  const local = (e: React.PointerEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  };

  const zoomBy = (f: number) => setView((vw) => ({ ...vw, zoom: Math.min(40, Math.max(0.8, vw.zoom * f)) }));

  return (
    <div ref={wrapRef} className="relative w-full overflow-hidden rounded-xl border border-line">
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: size.h, touchAction: "none", display: "block" }}
        className={hover ? "cursor-pointer" : "cursor-grab"}
        onPointerDown={(e) => {
          drag.current = { ...local(e), view, moved: false };
          (e.target as HTMLElement).setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          const p = local(e);
          if (drag.current) {
            const dx = p.x - drag.current.x;
            const dy = p.y - drag.current.y;
            if (Math.abs(dx) + Math.abs(dy) > 3) drag.current.moved = true;
            const k = base * drag.current.view.zoom;
            setView({ ...drag.current.view, cx: drag.current.view.cx - dx / k, cy: drag.current.view.cy - dy / k });
            setHover(null);
          } else {
            const id = nearest(p.x, p.y);
            setHover(id >= 0 ? { id, ...p } : null);
          }
        }}
        onPointerUp={(e) => {
          const wasDrag = drag.current?.moved;
          drag.current = null;
          if (!wasDrag) {
            const p = local(e);
            const id = nearest(p.x, p.y);
            if (id >= 0) onPick(id);
          }
        }}
        onPointerLeave={() => setHover(null)}
      />
      {hover && (
        <div
          className="pointer-events-none absolute z-10 rounded-lg border border-line bg-surface px-3 py-1.5 text-sm shadow-lg"
          style={{ left: Math.min(hover.x + 12, size.w - 170), top: hover.y + 12 }}
        >
          <b className="font-mono">{wordOf(vocab[hover.id])}</b>
          <span className="ml-2 text-xs text-muted">used {compact(map.count[hover.id])}×</span>
        </div>
      )}
      <div className="absolute top-2 right-2 flex flex-col gap-1">
        {[
          { icon: <Plus size={16} />, label: "Zoom in", on: () => zoomBy(1.5) },
          { icon: <Minus size={16} />, label: "Zoom out", on: () => zoomBy(1 / 1.5) },
          { icon: <Maximize2 size={14} />, label: "Show all words", on: () => setView(HOME) },
        ].map((b) => (
          <button
            key={b.label}
            onClick={b.on}
            aria-label={b.label}
            title={b.label}
            className="grid size-8 place-items-center rounded-lg border border-line bg-surface text-ink-2 shadow-sm hover:text-ink"
          >
            {b.icon}
          </button>
        ))}
      </div>
    </div>
  );
}
