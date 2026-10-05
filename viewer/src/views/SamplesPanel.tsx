import { useState } from "react";
import { Panel, Segmented } from "../components/Panel";
import { ConfidenceLegend } from "../components/ConfidenceLegend";
import type { Manifest, Snapshot } from "../lib/types";
import { useTheme } from "../lib/theme";
import { bandColor, underline } from "../lib/colors";
import { nameToken, pct, unitOf } from "../lib/format";

export function SamplesPanel({ manifest, snap }: { manifest: Manifest; snap: Snapshot }) {
  const { theme } = useTheme();
  const [which, setWhich] = useState(0);
  const [mode, setMode] = useState<"plain" | "confidence">("plain");
  const sample = snap.samples[which];
  const tokens = manifest.tokenizer.tokens;
  const unit = unitOf(manifest.tokenizer.name);

  return (
    <Panel
      title="What TinyWriter writes"
      subtitle={`We give TinyWriter the start of a story and let it continue, one ${unit.one} at a time.`}
      actions={
        <Segmented
          value={mode}
          onChange={setMode}
          options={[
            { value: "plain", label: "Plain" },
            { value: "confidence", label: "Confidence" },
          ]}
        />
      }
      help={
        <>
          <p>
            The <b>highlighted text</b> at the start is the prompt we typed in. Everything after it was written by TinyWriter, picking each
            {unit.one} randomly according to the probabilities it has learned.
          </p>
          <p>
            We use the <b>same random dice rolls</b> at every point in training, so when the text changes as you move
            the timeline, it's because TinyWriter changed, not luck.
          </p>
          <p>
            Switch to <b>Confidence</b> to underline each {unit.one} by how likely TinyWriter thought it was, using the
            same four bands as the rest of the site: from pale (a long shot) to dark (confident).
          </p>
        </>
      }
    >
      <div className="mb-3 flex flex-wrap gap-2">
        {manifest.prompts.map((p, i) => (
          <button
            key={i}
            onClick={() => setWhich(i)}
            className={`rounded-full border px-3 py-1 font-mono text-sm transition-colors ${
              i === which ? "border-accent bg-accent-wash text-accent-ink" : "border-line text-ink-2 hover:bg-surface-2"
            }`}
          >
            {p.trim() || "(blank)"}…
          </button>
        ))}
      </div>

      <div className="min-h-44 whitespace-pre-wrap break-words rounded-xl bg-surface-2 p-4 font-mono text-[15px] leading-8 text-ink-2">
        <span className="rounded-[3px] bg-accent-wash font-bold text-accent-ink">{sample.prompt}</span>
        {sample.ids.map((id, i) => {
          const t = tokens[id];
          const text = t === "<eot>" ? " ⏹\n" : t;
          const p = sample.probs[i];
          if (mode === "plain") return <span key={i}>{text}</span>;
          // Keep the leading space outside the underline, so it hugs the word.
          const [, space, word] = text.match(/^(\s*)([\s\S]*)$/)!;
          return (
            <span key={i}>
              {space}
              <span title={`TinyWriter gave ${nameToken(t)} a ${pct(p)} chance`} style={underline(bandColor(theme, p))}>
                {word}
              </span>
            </span>
          );
        })}
      </div>
      {mode === "confidence" && <ConfidenceLegend className="mt-3" />}
    </Panel>
  );
}
