import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router";
import { ArrowLeft, BookOpen, Lightbulb } from "lucide-react";
import { useRun, type LoadedRun } from "../lib/data";
import { chapterForRun } from "../content/chapters";
import { Timeline } from "../components/Timeline";
import { SamplesPanel } from "../views/SamplesPanel";
import { ProbePanel } from "../views/ProbePanel";
import { BigramPanel } from "../views/BigramPanel";
import { LossPanel } from "../views/LossPanel";
import { compact, int, unitOf } from "../lib/format";

const FRAME_MS = 450;

export function RunPage() {
  const { runId = "" } = useParams();
  const { run, error } = useRun(runId);

  if (error) {
    return (
      <div className="py-24 text-center">
        <p className="text-lg font-semibold">Couldn't load this run.</p>
        <p className="mt-2 text-ink-2">{error}</p>
        <Link to="/" className="mt-6 inline-block text-accent-ink underline">
          Back to TinyWriter
        </Link>
      </div>
    );
  }
  if (!run) {
    return (
      <div className="mx-auto max-w-sm py-32 text-center">
        <p className="text-ink-2">Loading TinyWriter's training history…</p>
        <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-surface-2">
          <div className="h-full w-1/3 animate-pulse rounded-full bg-accent" />
        </div>
      </div>
    );
  }
  return <Lab run={run} />;
}

function Lab({ run }: { run: LoadedRun }) {
  const { manifest, snapshots, loaded } = run;
  const last = snapshots.length - 1;
  const chapter = chapterForRun(manifest.id);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [hasPlayed, setHasPlayed] = useState(false);

  const play = useCallback((p: boolean) => {
    setPlaying(p);
    if (p) setHasPlayed(true);
  }, []);

  // Advance one snapshot per frame while playing (waiting for downloads if
  // playback catches up with them); stop at the end.
  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => {
      setIndex((i) => {
        if (i >= last) {
          setPlaying(false);
          return i;
        }
        return snapshots[i + 1] ? i + 1 : i;
      });
    }, FRAME_MS);
    return () => clearInterval(t);
  }, [playing, last, snapshots]);

  // Keyboard: space = play/pause, arrows = step.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest("input, textarea, button")) return;
      if (e.key === " ") {
        e.preventDefault();
        if (index >= last && !playing) setIndex(0);
        play(!playing);
      } else if (e.key === "ArrowRight") {
        play(false);
        setIndex((i) => Math.min(last, i + 1));
      } else if (e.key === "ArrowLeft") {
        play(false);
        setIndex((i) => Math.max(0, i - 1));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [index, last, playing, play]);

  // Show the requested snapshot, or the closest earlier one while it downloads.
  let shown = index;
  while (!snapshots[shown]) shown--;
  const snap = snapshots[shown]!;
  const step = manifest.checkpoints[index].step;

  return (
    <div className="pb-8">
      <div className="pt-6 pb-5">
        <Link to="/" className="inline-flex items-center gap-1 text-sm text-ink-2 hover:text-ink">
          <ArrowLeft size={15} /> Back to TinyWriter
        </Link>
        <div className="mt-3 flex flex-wrap items-end justify-between gap-4">
          <div>
            {manifest.chapter !== undefined && (
              <div className="text-sm font-semibold uppercase tracking-wide text-accent-ink">Chapter {manifest.chapter}</div>
            )}
            <h1 className="mt-1 text-3xl font-bold tracking-tight sm:text-4xl">{manifest.title}</h1>
            {(chapter?.tagline ?? manifest.description) && (
              <p className="mt-2 max-w-2xl text-lg text-ink-2">{chapter?.tagline ?? manifest.description}</p>
            )}
          </div>
          {chapter && (
            <a
              href={chapter.doc}
              className="inline-flex items-center gap-2 rounded-xl border border-line bg-surface px-4 py-2 text-sm font-medium hover:bg-surface-2"
            >
              <BookOpen size={16} /> Read the chapter
            </a>
          )}
        </div>
        <RunFacts run={run} />
        {loaded < snapshots.length && (
          <p className="mt-3 text-xs text-muted">
            Downloading training history… {Math.round((loaded / snapshots.length) * 100)}%
          </p>
        )}
      </div>

      <Timeline
        manifest={manifest}
        index={index}
        onIndex={setIndex}
        playing={playing}
        onPlaying={play}
        hasPlayed={hasPlayed}
      />

      {!hasPlayed && (
        <div className="mt-5 flex items-start gap-3 rounded-2xl border border-accent/30 bg-accent-wash p-4 text-[15px]">
          <Lightbulb size={20} className="mt-0.5 shrink-0 text-accent-ink" />
          <p>
            <b>New here?</b> Right now you're looking at TinyWriter <b>before any training</b>: pure random noise. Press the{" "}
            <b>play button</b> above to watch it learn, or drag the timeline yourself. You can also use the{" "}
            <kbd className="rounded border border-line bg-surface px-1.5 font-mono text-xs">space</kbd> and{" "}
            <kbd className="rounded border border-line bg-surface px-1.5 font-mono text-xs">← →</kbd> keys.
          </p>
        </div>
      )}

      <div className="mt-5 grid gap-5">
        <SamplesPanel manifest={manifest} snap={snap} />
        <ProbePanel manifest={manifest} snap={snap} contextSize={manifest.model.name === "bigram" ? 1 : undefined} />
        {snap.views.bigram && <BigramPanel manifest={manifest} snap={snap} />}
        <LossPanel manifest={manifest} step={step} />
      </div>
    </div>
  );
}

function RunFacts({ run }: { run: LoadedRun }) {
  const { manifest } = run;
  const unit = unitOf(manifest.tokenizer.name);
  const facts = [
    { label: "Parameters", value: int(manifest.model.params) },
    { label: "Vocabulary", value: `${int(manifest.tokenizer.tokens.length)} ${unit.vocab}` },
    { label: "Training steps", value: int(manifest.train.max_steps) },
    { label: `${unit.many[0].toUpperCase()}${unit.many.slice(1)} read`, value: compact(manifest.train.max_steps * manifest.train.tokens_per_step) },
    manifest.train.seconds !== undefined && {
      label: "Training time",
      value: `${duration(manifest.train.seconds)} on ${manifest.train.device === "cuda" ? "a GPU" : manifest.train.device}`,
    },
  ].filter(Boolean) as { label: string; value: string }[];

  return (
    <dl className="mt-5 flex flex-wrap gap-x-8 gap-y-3">
      {facts.map((f) => (
        <div key={f.label}>
          <dt className="text-xs font-medium uppercase tracking-wide text-muted">{f.label}</dt>
          <dd className="font-semibold">{f.value}</dd>
        </div>
      ))}
    </dl>
  );
}

function duration(seconds: number): string {
  if (seconds < 120) return `${Math.round(seconds)}s`;
  if (seconds < 7200) return `${Math.round(seconds / 60)} min`;
  return `${(seconds / 3600).toFixed(1)} h`;
}
