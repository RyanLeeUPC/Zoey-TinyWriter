import { useMemo, useState } from "react";
import { Panel, Segmented } from "../components/Panel";
import { Heatmap } from "../components/Heatmap";
import { GuessBars } from "../components/GuessBars";
import { RampLegend } from "../components/RampLegend";
import type { Manifest, Snapshot } from "../lib/types";
import { nameToken, pct, showToken } from "../lib/format";

// The characters that make up nearly all of TinyStories. Showing only these
// keeps the table readable; "All" shows every character in the vocabulary.
const COMMON = [" ", ..."abcdefghijklmnopqrstuvwxyz", ".", ",", "!", "?", "'", '"'];

export function BigramPanel({ manifest, snap }: { manifest: Manifest; snap: Snapshot }) {
  const tokens = manifest.tokenizer.tokens;
  const matrix = snap.views.bigram!;
  const [scope, setScope] = useState<"common" | "all">("common");
  const [row, setRow] = useState(() => tokens.indexOf("q"));

  const ids = useMemo(
    () => (scope === "all" ? tokens.map((_, i) => i) : COMMON.map((c) => tokens.indexOf(c)).filter((i) => i >= 0)),
    [scope, tokens],
  );
  const labels = useMemo(() => ids.map((i) => showToken(tokens[i])), [ids, tokens]);

  const top = useMemo(() => {
    const r = matrix[row];
    const order = r.map((_, i) => i).sort((a, b) => r[b] - r[a]).slice(0, 6);
    return { ids: order, probs: order.map((i) => r[i]) };
  }, [matrix, row]);

  return (
    <Panel
      title="Inside TinyWriter: the bigram table"
      subtitle="This table is the entire model. Each row is a letter; the colors show which letters TinyWriter expects next."
      actions={
        <Segmented
          value={scope}
          onChange={setScope}
          options={[
            { value: "common", label: "Common letters" },
            { value: "all", label: `All ${tokens.length}` },
          ]}
        />
      }
      help={
        <>
          <p>
            Read it row by row. Pick a row on the left (say <b>"q"</b>). The colored squares across that row show what
            TinyWriter thinks comes <b>after</b> a "q". By the end of training, the "u" column lights up, because in English
            "q" is almost always followed by "u".
          </p>
          <p>
            At step 0 the table is <b>random noise</b>. Press play and watch the patterns appear: vowels after
            consonants, spaces after punctuation, "h" after "t".
          </p>
          <p>
            Each row adds up to 100%. These {(tokens.length ** 2).toLocaleString()} numbers are{" "}
            <b>all of TinyWriter's parameters</b>. Real LLMs have billions, but they're trained the same way.
          </p>
        </>
      }
    >
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_260px]">
        <div className="min-w-0">
          <div className="mb-1 flex justify-between pl-6 text-xs text-muted">
            <span>next letter →</span>
          </div>
          <Heatmap
            matrix={matrix}
            rowIds={ids}
            colIds={ids}
            rowLabels={labels}
            colLabels={labels}
            selectedRow={row}
            onSelectRow={setRow}
            renderTooltip={(r, c) => (
              <>
                After <b className="font-mono">{showToken(tokens[r])}</b>, TinyWriter expects{" "}
                <b className="font-mono">{showToken(tokens[c])}</b>
                <div className="tabular text-base font-semibold">{pct(matrix[r][c])}</div>
              </>
            )}
          />
          <RampLegend className="mt-3" left="unlikely" right="likely" />
        </div>
        <div className="rounded-xl bg-surface-2 p-4">
          <div className="text-xs font-medium uppercase tracking-wide text-muted">Selected row</div>
          <p className="mt-1 mb-3 text-sm text-ink-2">
            After {nameToken(tokens[row])}, TinyWriter's favorite next letters:
          </p>
          <GuessBars ids={top.ids} probs={top.probs} tokens={tokens} />
          <p className="mt-4 text-xs text-muted">Click any row in the table to inspect it.</p>
        </div>
      </div>
    </Panel>
  );
}
