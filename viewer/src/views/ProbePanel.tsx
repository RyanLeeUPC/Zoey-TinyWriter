import { useState } from "react";
import { Panel } from "../components/Panel";
import { GuessBars } from "../components/GuessBars";
import { ConfidenceLegend } from "../components/ConfidenceLegend";
import type { Manifest, Snapshot } from "../lib/types";
import { useTheme } from "../lib/theme";
import { bandColor, underline } from "../lib/colors";
import { nameToken, pct, showToken } from "../lib/format";

/**
 * A fixed test sentence. For every token we show how much probability
 * TinyWriter gave it before seeing it. Click a token to see the full set of guesses.
 */
export function ProbePanel({
  manifest,
  snap,
}: {
  manifest: Manifest;
  snap: Snapshot;
}) {
  const { theme } = useTheme();
  const tokens = manifest.tokenizer.tokens;
  const ids = manifest.probe.ids;
  const [sel, setSel] = useState(() => initialSelection(ids, tokens));

  const pos = sel - 1; // the prediction made after reading tokens [0..pos]
  const target = ids[sel];
  const p = snap.probe.target_probs[pos];

  return (
    <Panel
      title={`Guess the next token`}
      subtitle={
        <>
          A test sentence TinyWriter never trained on. Each token is underlined by how strongly TinyWriter predicted it.{" "}
          <b className="text-ink">Click any token.</b>
        </>
      }
      help={
        <>
          <p>
            Before each token, we ask TinyWriter: <b>"what comes next?"</b> TinyWriter answers with a probability for every
            possible token. The underline under each token shows how much probability TinyWriter gave to the token that{" "}
            <i>actually</i> came next.
          </p>
          <p>
            This is exactly what the <b>loss</b> measures. Loss is the average "surprise" across tokens like these:
            a confident correct guess costs almost nothing, a confident wrong guess costs a lot.
          </p>
        </>
      }
    >
      <div className="flex flex-wrap gap-y-1.5 font-mono text-[17px]" role="listbox" aria-label="Test sentence">
        {words(ids, tokens).map((word, w) => (
          // Keep each word (plus its trailing space) together so lines never break mid-word.
          <span key={w} className="inline-flex">
            {word.map((i) => {
              const id = ids[i];
              const tp = i === 0 ? null : snap.probe.target_probs[i - 1];
              const active = i === sel;
              return (
                <button
                  key={i}
                  role="option"
                  aria-selected={active}
                  disabled={i === 0}
                  onClick={() => setSel(i)}
                  title={tp === null ? `First token: nothing to predict from` : `${pct(tp)} chance of ${nameToken(tokens[id])}`}
                  className={`h-10 min-w-[1.35rem] whitespace-pre rounded-[4px] px-0.5 transition-colors ${
                    active ? "relative z-10 bg-ink text-page" : ""
                  } ${tp === null ? "text-muted" : "cursor-pointer hover:bg-surface-2"}`}
                >
                  <TokenText text={tokens[id]} style={tp === null ? undefined : underline(bandColor(theme, tp))} />
                </button>
              );
            })}
          </span>
        ))}
      </div>
      <ConfidenceLegend className="mt-3" />

      <div className="mt-5 grid gap-5 rounded-xl bg-surface-2 p-4 md:grid-cols-[1fr_1.2fr]">
        <div>
          <div className="text-xs font-medium uppercase tracking-wide text-muted">TinyWriter has read</div>
          <div className="mt-1 break-words font-mono text-[15px] leading-7">
            {ids.slice(0, sel).map((id, i) => (
              <span
                key={i}
                className="rounded-[3px] bg-accent-wash font-bold text-accent-ink"
              >
                {tokens[id]}
              </span>
            ))}
            <span className="ml-0.5 inline-block h-5 w-0.5 translate-y-1 animate-pulse bg-ink" />
          </div>
          <p className="mt-3 text-sm text-ink-2">
            The real next token was <b className="font-mono text-ink">{showToken(tokens[target])}</b>. TinyWriter gave it{" "}
            <b className="tabular text-ink">{pct(p)}</b>.
          </p>
        </div>
        <div>
          <div className="mb-2 text-xs font-medium uppercase tracking-wide text-muted">TinyWriter's top 5 guesses</div>
          <GuessBars ids={snap.probe.top_ids[pos]} probs={snap.probe.top_probs[pos]} tokens={tokens} correct={target} />
        </div>
      </div>
    </Panel>
  );
}

/**
 * Group token positions into words so lines never break mid-word. A new word
 * starts after a whitespace token, or at a token with a leading space (BPE).
 */
function words(ids: number[], tokens: string[]): number[][] {
  const out: number[][] = [[]];
  ids.forEach((id, i) => {
    const t = tokens[id];
    if (i > 0 && /^\s\S/.test(t)) out.push([]);
    out[out.length - 1].push(i);
    if (/^\s+$/.test(t)) out.push([]);
  });
  return out.filter((w) => w.length);
}

/** Start on an instructive spot: the second "Lily", which has to be copied from earlier. */
function initialSelection(ids: number[], tokens: string[]): number {
  const text = ids.map((id) => tokens[id]);
  const lily = text.map((t, i) => (t.trim() === "Lily" ? i : -1)).filter((i) => i > 0);
  return lily[1] ?? lily[0] ?? 1;
}

/**
 * A token with its confidence underline on the word only, not its leading
 * space. A token that is *only* whitespace shows as a visible symbol (␣ or ↵),
 * so its underline has something to sit under.
 */
function TokenText({ text, style }: { text: string; style?: React.CSSProperties }) {
  const [, space, word] = text.match(/^(\s*)([\s\S]*)$/)!;
  if (!word) return <span style={style}>{showToken(text)}</span>;
  return (
    <>
      {space}
      <span style={style}>{word}</span>
    </>
  );
}
