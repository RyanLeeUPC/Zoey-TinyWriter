import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router";
import { Cpu, Lightbulb, Play } from "lucide-react";
import { useZoey, zoey } from "../engine/client";
import { Composer, LENGTHS, STARTERS, type Length } from "../explore/Composer";
import { Story, type Paint } from "../explore/Story";
import { Inspector } from "../explore/Inspector";
import { attentionMap, attentionRow, autoPick, type HeadChoice } from "../explore/analysis";
import { Segmented } from "../components/Panel";
import { ConfidenceLegend } from "../components/ConfidenceLegend";
import { bandColor, TARGET_SLOTS, targetColor, withAlpha } from "../lib/colors";
import { useTheme } from "../lib/theme";
import { compact } from "../lib/format";
import { MODEL_ID } from "../config";

// A seed that gives a nice first story for the default prompt.
const FIRST_SEED = 7;

/*
  The story lives in the model's store for the whole visit, so the page's own
  choices (prompt, settings, which word is selected) live outside the page too.
  Visit the Dictionary, come back, and everything is where you left it.
*/
const sticky = new Map<string, unknown>();
function useSticky<T>(key: string, initial: T): [T, (value: T) => void] {
  const [value, setValue] = useState<T>(() => (sticky.has(key) ? (sticky.get(key) as T) : initial));
  const set = useCallback(
    (v: T) => {
      sticky.set(key, v);
      setValue(v);
    },
    [key],
  );
  return [value, set];
}
let welcomed = false; // has the first story been written this visit?
let lastPickedRun = 0; // which story the inspector was last auto-opened on

export function Explore() {
  const z = useZoey();
  const [prompt, setPrompt] = useSticky("prompt", STARTERS[0]);
  const [temperature, setTemperature] = useSticky("temperature", 0.7);
  const [length, setLength] = useSticky<Length>("length", "medium");
  const [seed, setSeed] = useSticky("seed", FIRST_SEED);
  const [selected, setSelected] = useSticky<number | null>("selected", null);
  const [head, setHead] = useSticky<HeadChoice>("head", "all");
  const [touched, setTouched] = useSticky("touched", false); // has the person clicked anything yet?
  const [findsName, setFindsName] = useSticky("findsName", false); // does the auto-picked head look at a name?
  const [pickTarget, setPickTarget] = useSticky<number | null>("pickTarget", null); // the word that head looks at
  const [mode, setMode] = useSticky<"attention" | "confidence">("mode", "attention");
  const [hoverTarget, setHoverTarget] = useState<number | null>(null);
  const { theme } = useTheme();

  const shape = z.info ? { layers: z.info.config.n_layer, heads: z.info.config.n_head } : { layers: 1, heads: 1 };
  const busy = z.status === "generating";

  useEffect(() => zoey.load(), []);

  const write = (s = seed) => {
    setSeed(s);
    setSelected(null);
    setHead("all");
    zoey.generate({ prompt, maxTokens: LENGTHS[length], temperature, seed: s });
  };

  // Write a first story as soon as the model is ready, so there's something to look at.
  useEffect(() => {
    if (z.status === "ready" && !welcomed) {
      welcomed = true;
      write(FIRST_SEED);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [z.status]);

  // When a story finishes, open the inspector on an interesting token.
  useEffect(() => {
    if (z.status !== "ready" || !z.tokens.length || lastPickedRun === z.run) return;
    lastPickedRun = z.run;
    const pick = autoPick(z.tokens, shape);
    if (pick) {
      // Open on the colored overview of all heads; the hint names the word the
      // most interesting head was looking at.
      setSelected(pick.pos);
      setHead("all");
      setFindsName(pick.findsName);
      const focus = pick.head === "all" ? null : attentionMap(z.tokens[pick.pos], shape, TARGET_SLOTS, z.tokens).cells[pick.head.layer][pick.head.head];
      setPickTarget(focus?.pos ?? null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [z.status, z.run]);

  const map = useMemo(() => {
    if (selected === null || selected < 1 || !z.tokens[selected]) return null;
    return attentionMap(z.tokens[selected], shape, TARGET_SLOTS, z.tokens);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, z.tokens]);

  // Hovering a word or head outlines every place that word appears.
  const outlined = useMemo(() => {
    if (hoverTarget === null) return null;
    return new Set(map?.groups.find((g) => g.positions.includes(hoverTarget))?.positions ?? [hoverTarget]);
  }, [hoverTarget, map]);

  // What color each word in the story gets.
  const paint = useMemo(() => {
    const out = new Map<number, Paint>();
    if (mode === "confidence") {
      // Every word underlined by how likely TinyWriter thought it was.
      for (const t of z.tokens) {
        const p = z.tokens[t.pos - 1]?.nextProb;
        if (t.source !== "start" && p !== undefined) out.set(t.pos, { mark: bandColor(theme, p) });
      }
    } else if (map && selected !== null) {
      if (head === "all") {
        // Each word the heads focus on, in its group's color.
        for (const g of map.groups) {
          const color = targetColor(theme, g.slot);
          for (const pos of g.positions)
            if (pos !== selected) out.set(pos, g.slot === null ? { bg: withAlpha(color, 0.18) } : { bg: withAlpha(color, 0.28), underline: color });
        }
      } else {
        // One head: everything it looked at, stronger tint = more attention.
        const row = attentionRow(z.tokens[selected], shape, head);
        const color = targetColor(theme, map.cells[head.layer][head.head].group.slot);
        let peak = 0.05;
        row.forEach((w, j) => j !== selected && (peak = Math.max(peak, w)));
        row.forEach((w, j) => {
          const r = w / peak;
          if (j !== selected && r > 0.04) out.set(j, { bg: withAlpha(color, 0.08 + 0.5 * Math.sqrt(r)), underline: r > 0.5 ? color : undefined });
        });
      }
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, map, head, selected, z.tokens, theme]);

  // Compute the layer-by-layer view for whichever token is selected.
  useEffect(() => {
    if (selected !== null && !busy) zoey.requestLens(selected);
  }, [selected, busy, z.run]);

  const select = (pos: number) => {
    setTouched(true);
    setSelected(pos);
    setHoverTarget(null);
    // On narrow screens the inspector sits below the story: bring it into view.
    if (window.innerWidth < 1024) document.getElementById("inspector")?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div className="pb-8">
      <section className="pt-10 pb-6">
        <p className="text-sm font-semibold uppercase tracking-wide text-accent-ink">A glass-box language model</p>
        <h1 className="mt-2 text-4xl font-bold tracking-tight sm:text-5xl">Meet TinyWriter. Look inside.</h1>
        <p className="mt-3 max-w-3xl text-lg leading-relaxed text-ink-2">
          TinyWriter is a small language model that writes children's stories. It works like the AI chatbots you know, just{" "}
          <b className="text-ink">thousands of times smaller</b>, so you can see every step. Give it a start, then click
          any word to look inside TinyWriter at that moment.
        </p>
        <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2 text-sm text-ink-2">
          <span className="inline-flex items-center gap-1.5">
            <Cpu size={15} /> Running entirely in your browser
          </span>
          {z.info && (
            <span>
              {compact(z.info.params)} parameters · {z.info.config.n_layer} layers · {z.info.config.n_layer * z.info.config.n_head}{" "}
              attention heads
            </span>
          )}
          <Link to={`/run/${MODEL_ID}`} className="inline-flex items-center gap-1 font-medium text-accent-ink hover:underline">
            <Play size={13} fill="currentColor" /> Watch how TinyWriter learned
          </Link>
        </div>
      </section>

      {z.status === "loading" || z.status === "idle" ? (
        <Loading progress={z.progress} />
      ) : z.status === "error" ? (
        <div className="rounded-2xl border border-bad/40 bg-surface p-6">
          <p className="font-semibold">TinyWriter couldn't start.</p>
          <p className="mt-1 text-sm text-ink-2">{z.error}</p>
        </div>
      ) : (
        <>
          <Composer
            prompt={prompt}
            onPrompt={setPrompt}
            temperature={temperature}
            onTemperature={setTemperature}
            length={length}
            onLength={setLength}
            busy={busy}
            disabled={z.status !== "ready"}
            onWrite={() => write()}
            onReroll={() => write(Math.floor(Math.random() * 1e9))}
            onStop={() => zoey.stop()}
          />

          {z.trimmed > 0 && (
            <p className="mt-3 rounded-xl bg-surface-2 px-4 py-2 text-sm text-ink-2">
              Your start was too long to fit along with the story, so TinyWriter read only its last part (the first{" "}
              {z.trimmed} tokens were left out).
            </p>
          )}

          {!touched && selected !== null && !busy && map && (
            <Hint tokens={z.tokens} selected={selected} map={map} target={pickTarget} vocab={z.vocab} findsName={findsName} />
          )}

          <div className="mt-5 grid items-start gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,27rem)]">
            <div className="rounded-2xl border border-line bg-surface p-5 shadow-sm sm:p-6">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2 text-sm text-muted">
                <span>
                  {busy ? "TinyWriter is writing…" : "TinyWriter's story"}
                  {z.tokens.length > 1 && <span className="tabular"> · {z.tokens.length - 1} tokens</span>}
                </span>
                <div className="flex items-center gap-2">
                  <span className="hidden text-xs sm:inline">Color by</span>
                  <Segmented
                    value={mode}
                    onChange={(m) => {
                      setTouched(true);
                      setHoverTarget(null);
                      setMode(m);
                    }}
                    options={[
                      { value: "attention", label: "Attention" },
                      { value: "confidence", label: "Confidence" },
                    ]}
                  />
                </div>
              </div>
              {mode === "confidence" && (
                <div className="mb-3 rounded-lg bg-surface-2 px-3 py-2">
                  <p className="mb-1 text-xs text-ink-2">How likely TinyWriter thought each word was, before writing it:</p>
                  <ConfidenceLegend />
                </div>
              )}
              <Story
                tokens={z.tokens}
                selected={selected}
                paint={paint}
                outlined={mode === "attention" ? outlined : null}
                busy={busy}
                onSelect={select}
                onHover={(pos) => setHoverTarget(mode === "attention" && map ? pos : null)}
              />
            </div>
            <div id="inspector" className="scroll-mt-4 lg:sticky lg:top-4">
              <Inspector
                tokens={z.tokens}
                selected={selected}
                vocab={z.vocab}
                shape={shape}
                map={map}
                lens={selected === null ? undefined : z.lens.get(selected)}
                head={head}
                onHead={(h) => {
                  setTouched(true);
                  setMode("attention");
                  setHead(h);
                }}
                hoverTarget={hoverTarget}
                onHoverTarget={setHoverTarget}
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function Loading({ progress }: { progress: number }) {
  return (
    <div className="mx-auto max-w-md rounded-2xl border border-line bg-surface p-8 text-center shadow-sm">
      <p className="font-semibold">Downloading TinyWriter's brain…</p>
      <p className="mt-1 text-sm text-ink-2">28 MB, one time. After that, everything runs on your own computer.</p>
      <div className="mt-5 h-2 overflow-hidden rounded-full bg-surface-2">
        <div className="h-full rounded-full bg-accent transition-[width]" style={{ width: `${Math.round(progress * 100)}%` }} />
      </div>
      <p className="tabular mt-2 text-xs text-muted">{Math.round(progress * 100)}%</p>
    </div>
  );
}

/** First-visit callout explaining what's highlighted. */
function Hint({
  tokens,
  selected,
  map,
  target,
  vocab,
  findsName,
}: {
  tokens: ReturnType<typeof useZoey>["tokens"];
  selected: number;
  map: ReturnType<typeof attentionMap>;
  target: number | null;
  vocab: string[];
  findsName: boolean;
}) {
  const word = tokens[selected]?.text.trim();
  const group = map.groups.find((g) => target !== null && g.positions.includes(target));
  const targetWord = target !== null && target > 0 ? vocab[tokens[target].id].trim() : null;
  return (
    <div className="mt-5 flex items-start gap-3 rounded-2xl border border-accent/30 bg-accent-wash p-4 text-[15px]">
      <Lightbulb size={20} className="mt-0.5 shrink-0 text-accent-ink" />
      <p>
        <b>Look:</b> when TinyWriter got to <b className="font-mono">“{word}”</b>
        {group && targetWord ? (
          <>
            , {group.heads === 1 ? "one of its attention heads was" : `${group.heads} of its attention heads were`} looking back at{" "}
            <b className="font-mono">“{targetWord}”</b>
            {findsName ? <>. That's how TinyWriter keeps track of who “{word}” is.</> : "."} Each color matches a word in the story
            to the heads looking at it.
          </>
        ) : (
          <>, each color matches a word in the story to the attention heads looking at it.</>
        )}{" "}
        Click any other word to look inside TinyWriter at that moment.
      </p>
    </div>
  );
}
