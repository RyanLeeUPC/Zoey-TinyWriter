import { useState, type ReactNode } from "react";
import { BookOpen, ChevronDown, CircleHelp, MousePointerClick } from "lucide-react";
import { Link } from "react-router";
import type { Lens, TokenTrace } from "../engine/protocol";
import { GuessBars } from "../components/GuessBars";
import { ConfidenceChip, ConfidenceLegend } from "../components/ConfidenceLegend";
import type { attentionMap, HeadChoice, Shape } from "./analysis";
import { useTheme } from "../lib/theme";
import { bandColor, targetColor, withAlpha } from "../lib/colors";
import { pct, showToken } from "../lib/format";

type AttentionMap = ReturnType<typeof attentionMap>;

/**
 * Inside TinyWriter at one moment: having just read the selected token, where is
 * it looking, and what does it expect to come next?
 */
export function Inspector({
  tokens,
  selected,
  vocab,
  shape,
  map,
  lens,
  head,
  onHead,
  hoverTarget,
  onHoverTarget,
}: {
  tokens: TokenTrace[];
  selected: number | null;
  vocab: string[];
  shape: Shape;
  map: AttentionMap | null;
  lens: Lens | undefined;
  head: HeadChoice;
  onHead: (h: HeadChoice) => void;
  hoverTarget: number | null;
  onHoverTarget: (pos: number | null) => void;
}) {
  const { theme } = useTheme();
  if (selected === null || selected < 1 || !tokens[selected] || !map) {
    return (
      <div className="grid h-full min-h-64 place-items-center rounded-2xl border border-dashed border-line p-8 text-center text-ink-2">
        <div>
          <MousePointerClick className="mx-auto mb-3 text-muted" size={28} />
          <p className="font-medium">Click any word in the story</p>
          <p className="mt-1 text-sm">to see what was going on inside TinyWriter at that moment.</p>
        </div>
      </div>
    );
  }

  const at = tokens[selected];
  const word = at.text.trim() || showToken(at.text);
  const next = at.next === undefined ? null : vocab[at.next];
  const nextWord = next === null ? null : next.trim() || showToken(next);
  const nextByZoey = tokens[selected + 1] ? tokens[selected + 1].source === "zoey" : next !== null;
  const bar = (p: number) => bandColor(theme, p);

  return (
    <div className="space-y-4">
      <div className="rounded-2xl border border-line bg-surface p-5 shadow-sm">
        <div className="text-xs font-semibold uppercase tracking-wide text-muted">Inside TinyWriter, right after reading</div>
        <div className="mt-1 flex flex-wrap items-baseline justify-between gap-x-3">
          <span className="font-mono text-2xl font-bold">“{word}”</span>
          <Link
            to={`/dictionary/${at.id}`}
            className="inline-flex items-center gap-1 text-xs font-medium text-accent-ink hover:underline"
            title={`What TinyWriter knows about “${word}” on its own`}
          >
            <BookOpen size={13} /> In TinyWriter's dictionary
          </Link>
        </div>
        {nextWord !== null ? (
          <p className="mt-1 text-sm leading-6 text-ink-2">
            Next, {nextByZoey ? "TinyWriter wrote" : "your prompt continued with"}{" "}
            <b className="font-mono text-ink">“{nextWord}”</b>
            {nextByZoey ? ", which it gave a " : ". TinyWriter had given that a "}
            <b className="tabular text-ink">{pct(at.nextProb ?? 0)}</b> chance <ConfidenceChip p={at.nextProb ?? 0} />
          </p>
        ) : (
          <p className="mt-1 text-sm text-ink-2">This is where the story stopped.</p>
        )}

        <Section
          title="What TinyWriter expected next"
          help={
            <p>
              After reading each token, TinyWriter gives <b>every one of its 4,096 tokens</b> a probability of coming next.
              These are the top ones. TinyWriter then picks one at random, weighted by these chances. That's why the same
              start can lead to different stories.
            </p>
          }
        >
          <GuessBars ids={at.top.ids.slice(0, 5)} probs={at.top.probs.slice(0, 5)} tokens={vocab} correct={at.next} barColor={bar} />
          {at.next !== undefined && !at.top.ids.slice(0, 5).includes(at.next) && (
            <p className="mt-2 text-sm text-ink-2">
              “{nextWord}” wasn't even in the top 5: a {pct(at.nextProb ?? 0)} long shot that came up.
            </p>
          )}
          <ConfidenceLegend className="mt-3" />
        </Section>
      </div>

      <div className="rounded-2xl border border-line bg-surface p-5 shadow-sm">
        <Section
          title="Where TinyWriter was looking"
          first
          help={
            <>
              <p>
                TinyWriter has <b>{shape.layers} layers × {shape.heads} attention heads</b>. Each head looks back over the story
                and decides which earlier words matter right now. Each square is one head, labeled with the word it
                focused on most.
              </p>
              <p>
                <b>Heads looking at the same word share a color</b>, and that word has the same color in the story. Hover a
                square or a word to see the connection. Click a square to see everything that one head was looking at.
              </p>
              <p>
                Heads specialize: some track the previous word, some find names, some find the start of the sentence.
                Heads with nothing useful to do often rest their attention on the <b>START</b> marker.
              </p>
            </>
          }
        >
          <HeadGrid
            map={map}
            shape={shape}
            vocab={vocab}
            tokens={tokens}
            head={head}
            onHead={onHead}
            hoverTarget={hoverTarget}
            onHoverTarget={onHoverTarget}
          />
        </Section>
      </div>

      <div className="rounded-2xl border border-line bg-surface p-5 shadow-sm">
        <Section
          title="How the answer formed, layer by layer"
          first
          help={
            <>
              <p>
                Each token's information flows up through TinyWriter's {shape.layers} layers, and each layer adds to it. Here we
                peek after every layer and ask: <b>"if you had to answer now, what would you say?"</b>
              </p>
              <p>
                Early layers usually guess something generic. The bars show how much probability the token that
                really came next had at each depth. You can watch the answer take shape as the information climbs.
              </p>
            </>
          }
        >
          {at.next === undefined ? (
            <p className="text-sm text-ink-2">Nothing came after this token.</p>
          ) : lens ? (
            <LayerLens lens={lens} next={at.next} vocab={vocab} barColor={bar} />
          ) : (
            <p className="text-sm text-muted">Working it out…</p>
          )}
        </Section>
      </div>
    </div>
  );
}

function Section({ title, help, first, children }: { title: string; help?: ReactNode; first?: boolean; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className={first ? "" : "mt-5 border-t border-line pt-4"}>
      <div className="mb-3 flex items-center justify-between gap-2">
        <h3 className="font-semibold">{title}</h3>
        {help && (
          <button
            onClick={() => setOpen((o) => !o)}
            className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium text-accent-ink hover:bg-accent-wash"
            aria-expanded={open}
          >
            <CircleHelp size={14} /> What's this?
            <ChevronDown size={13} className={`transition-transform ${open ? "rotate-180" : ""}`} />
          </button>
        )}
      </div>
      {open && <div className="mb-3 space-y-2 rounded-xl bg-surface-2 p-3 text-sm leading-relaxed text-ink-2 [&_b]:text-ink">{help}</div>}
      {children}
    </div>
  );
}

function HeadGrid({
  map,
  shape,
  vocab,
  tokens,
  head,
  onHead,
  hoverTarget,
  onHoverTarget,
}: {
  map: AttentionMap;
  shape: Shape;
  vocab: string[];
  tokens: TokenTrace[];
  head: HeadChoice;
  onHead: (h: HeadChoice) => void;
  hoverTarget: number | null;
  onHoverTarget: (pos: number | null) => void;
}) {
  const { theme } = useTheme();
  const label = (pos: number) => (pos === 0 ? "START" : vocab[tokens[pos].id].trim() || showToken(vocab[tokens[pos].id]));
  const sel = head === "all" ? null : head;
  const selFocus = sel ? map.cells[sel.layer][sel.head] : null;

  return (
    <div onMouseLeave={() => onHoverTarget(null)}>
      <div className="grid gap-1" style={{ gridTemplateColumns: `auto repeat(${shape.heads}, minmax(0, 1fr))` }}>
        <span />
        {Array.from({ length: shape.heads }, (_, h) => (
          <span key={h} className="text-center text-[10px] text-muted">
            head {h + 1}
          </span>
        ))}
        {map.cells.map((row, layer) => (
          <Row key={layer}>
            <span className="pr-1 text-right text-[10px] leading-8 text-muted">layer {layer + 1}</span>
            {row.map((cell, h) => {
              const active = sel?.layer === layer && sel?.head === h;
              const dimmed =
                (hoverTarget !== null && !cell.group.positions.includes(hoverTarget)) || (sel !== null && !active && hoverTarget === null);
              const color = targetColor(theme, cell.group.slot);
              const start = cell.pos === 0;
              return (
                <button
                  key={h}
                  onClick={() => onHead(active ? "all" : { layer, head: h })}
                  onMouseEnter={() => onHoverTarget(cell.pos)}
                  title={`Layer ${layer + 1}, head ${h + 1}: ${pct(cell.weight)} on “${label(cell.pos)}”`}
                  style={{
                    // Fill strength follows how focused the head is; the hue says which word.
                    background: withAlpha(color, start ? 0.14 : 0.2 + 0.65 * cell.weight),
                    boxShadow: start ? undefined : `inset 0 -3px 0 ${color}`,
                  }}
                  className={`h-8 min-w-0 truncate rounded-md px-1 font-mono text-[11px] text-ink transition ${
                    active ? "outline-2 outline-offset-1 outline-ink" : ""
                  } ${dimmed ? "opacity-25" : ""} ${start ? "text-muted" : ""}`}
                >
                  {label(cell.pos)}
                </button>
              );
            })}
          </Row>
        ))}
      </div>

      <ul className="mt-3 flex flex-wrap gap-1.5" aria-label="Words the heads are looking at">
        {map.groups
          .filter((g) => g.slot !== null || g.pos === 0)
          .map((g) => (
            <li key={g.pos}>
              <span
                onMouseEnter={() => onHoverTarget(g.pos)}
                className={`inline-flex cursor-default items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs transition-opacity ${
                  hoverTarget !== null && !g.positions.includes(hoverTarget) ? "opacity-40" : ""
                } ${g.pos === 0 ? "border-dashed border-line text-muted" : "border-line text-ink"}`}
              >
                <span className="size-2.5 rounded-full" style={{ background: targetColor(theme, g.slot) }} />
                <b className="font-mono font-semibold">{label(g.pos)}</b>
                <span className="text-muted">
                  {g.heads} head{g.heads === 1 ? "" : "s"}
                  {g.pos === 0 ? " resting" : ""}
                </span>
              </span>
            </li>
          ))}
      </ul>

      <div className="mt-3 flex items-center justify-between gap-3 text-sm text-ink-2">
        <p>
          {sel && selFocus ? (
            <>
              <b className="text-ink">
                Layer {sel.layer + 1}, head {sel.head + 1}
              </b>{" "}
              put {pct(selFocus.weight)} of its attention on <b className="font-mono text-ink">“{label(selFocus.pos)}”</b>. The
              story shows everything it looked at.
            </>
          ) : (
            <>Hover a square or a word to connect them. Click a square to focus on one head.</>
          )}
        </p>
        {sel && (
          <button onClick={() => onHead("all")} className="shrink-0 rounded-lg px-2 py-1 text-xs font-medium text-accent-ink hover:bg-accent-wash">
            Show all
          </button>
        )}
      </div>
    </div>
  );
}

const Row = ({ children }: { children: ReactNode }) => <>{children}</>;

function LayerLens({ lens, next, vocab, barColor }: { lens: Lens; next: number; vocab: string[]; barColor: (p: number) => string }) {
  const rows = lens.ids.map((ids, l) => ({
    name: l === 0 ? "Embedding" : `Layer ${l}`,
    guess: vocab[ids[0]],
    guessP: lens.probs[l][0],
    answerP: lens.next[l],
  }));
  return (
    <div>
      <div className="mb-1 grid grid-cols-[5.5rem_minmax(0,7rem)_1fr_3.5rem] gap-2 text-[11px] font-medium uppercase tracking-wide text-muted">
        <span>after</span>
        <span>top guess</span>
        <span>chance of what came next</span>
        <span />
      </div>
      <ol className="space-y-1">
        {rows.map((r) => {
          const hit = r.guess === vocab[next];
          return (
            <li key={r.name} className="grid grid-cols-[5.5rem_minmax(0,7rem)_1fr_3.5rem] items-center gap-2 text-sm">
              <span className="text-ink-2">{r.name}</span>
              <span
                className={`truncate rounded-md border px-1.5 py-0.5 font-mono text-[13px] whitespace-pre ${
                  hit ? "border-good bg-good/10 font-bold text-good-ink" : "border-line bg-surface-2"
                }`}
                title={`${pct(r.guessP)}`}
              >
                {showToken(r.guess)}
              </span>
              <div className="relative h-4">
                <div className="absolute inset-0 rounded-r-[4px] bg-surface-2" />
                <div
                  className="absolute inset-y-0 left-0 rounded-r-[4px] transition-[width] duration-300"
                  style={{ width: `max(2px, ${r.answerP * 100}%)`, background: barColor(r.answerP) }}
                />
              </div>
              <span className="tabular text-right text-ink-2">{pct(r.answerP)}</span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
