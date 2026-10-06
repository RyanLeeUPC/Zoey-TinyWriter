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
  // Every finger (or the mouse) currently pressed on the map, and the gesture they make.
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const gesture = useRef<{ view: View; start: Map<number, { x: number; y: number }>; moved: boolean; multi: boolean } | null>(null);
  const viewRef = useRef(view);
  viewRef.current = view;

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

  const clampZoom = (z: number) => Math.min(40, Math.max(0.8, z));

  /** (Re)start the gesture from the current view, e.g. when a finger is added or lifted. */
  const restartGesture = (keep?: { moved: boolean; multi: boolean }) => {
    gesture.current = {
      view: viewRef.current,
      start: new Map(pointers.current),
      moved: keep?.moved ?? false,
      multi: (keep?.multi ?? false) || pointers.current.size > 1,
    };
  };

  /** One finger pans; two fingers pan and pinch-zoom around their midpoint. */
  const applyGesture = () => {
    const g = gesture.current;
    if (!g) return;
    const ids = [...pointers.current.keys()].filter((id) => g.start.has(id));
    if (!ids.length) return;
    const now = ids.map((id) => pointers.current.get(id)!);
    const was = ids.map((id) => g.start.get(id)!);
    const mid = (pts: { x: number; y: number }[]) => ({
      x: pts.reduce((a, p) => a + p.x, 0) / pts.length,
      y: pts.reduce((a, p) => a + p.y, 0) / pts.length,
    });
    const m0 = mid(was);
    const m1 = mid(now);
    let zoom = g.view.zoom;
    if (now.length >= 2) {
      const d0 = Math.hypot(was[0].x - was[1].x, was[0].y - was[1].y) || 1;
      const d1 = Math.hypot(now[0].x - now[1].x, now[0].y - now[1].y);
      zoom = clampZoom(g.view.zoom * (d1 / d0));
    }
    if (Math.abs(m1.x - m0.x) + Math.abs(m1.y - m0.y) > 3 || zoom !== g.view.zoom) g.moved = true;
    // Keep the map point that was under the fingers' midpoint under it now.
    const wx = g.view.cx + (m0.x - size.w / 2) / (base * g.view.zoom);
    const wy = g.view.cy + (m0.y - size.h / 2) / (base * g.view.zoom);
    setView({ zoom, cx: wx - (m1.x - size.w / 2) / (base * zoom), cy: wy - (m1.y - size.h / 2) / (base * zoom) });
  };

  const release = (e: React.PointerEvent, tap: boolean) => {
    pointers.current.delete(e.pointerId);
    const g = gesture.current;
    if (pointers.current.size > 0) {
      // A finger lifted mid-gesture: carry on smoothly with the ones left.
      restartGesture(g ? { moved: g.moved, multi: g.multi } : undefined);
      return;
    }
    gesture.current = null;
    // Only a quick single-finger tap (or click) opens a word.
    if (tap && g && !g.moved && !g.multi) {
      const p = local(e);
      const id = nearest(p.x, p.y);
      if (id >= 0) onPick(id);
    }
  };

  const zoomBy = (f: number) => setView((vw) => ({ ...vw, zoom: clampZoom(vw.zoom * f) }));

  return (
    <div ref={wrapRef} className="relative w-full overflow-hidden rounded-xl border border-line">
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: size.h, touchAction: "none", display: "block" }}
        className={hover ? "cursor-pointer" : "cursor-grab"}
        onPointerDown={(e) => {
          pointers.current.set(e.pointerId, local(e));
          try {
            (e.target as HTMLElement).setPointerCapture(e.pointerId);
          } catch {
            // some pointers (e.g. synthetic ones) can't be captured; the gesture still works
          }
          const g = gesture.current;
          restartGesture(g ? { moved: g.moved, multi: true } : undefined);
          setHover(null);
        }}
        onPointerMove={(e) => {
          const p = local(e);
          if (pointers.current.has(e.pointerId)) {
            pointers.current.set(e.pointerId, p);
            applyGesture();
          } else if (e.pointerType === "mouse") {
            const id = nearest(p.x, p.y);
            setHover(id >= 0 ? { id, ...p } : null);
          }
        }}
        onPointerUp={(e) => release(e, true)}
        onPointerCancel={(e) => release(e, false)}
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
