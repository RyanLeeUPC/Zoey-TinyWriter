import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Link, useNavigate, useParams } from "react-router";
import { ArrowLeftRight, ChevronDown, CircleHelp, X } from "lucide-react";
import { useZoey, zoey } from "../engine/client";
import type { TokenInfo } from "../engine/protocol";
import { int } from "../lib/format";
import { Fingerprint, FingerprintLegend } from "../dictionary/Fingerprint";
import { TokenSearch } from "../dictionary/TokenSearch";
import { VocabMap } from "../dictionary/VocabMap";
import { cosine, describeSimilarity, useDictionaryMap, wordOf } from "../dictionary/data";
import { Segmented } from "../components/Panel";

const DEFAULT_WORD = " dog";
const SHOWN_NEIGHBORS = 12;
const QUICK_PICKS = [" dog", " happy", " red", " Lily", " ran", " three", " cake", " mom"];

export function Dictionary() {
  const z = useZoey();
  const { map, error: mapError } = useDictionaryMap();
  const navigate = useNavigate();
  const { tokenId } = useParams();
  const [compareId, setCompareId] = useState<number | null>(null);

  useEffect(() => zoey.load(), []);

  const ready = z.vocab.length > 0 && map !== null;
  const selected = useMemo(() => {
    if (!ready) return null;
    const fromUrl = tokenId === undefined ? NaN : +tokenId;
    return Number.isInteger(fromUrl) && fromUrl >= 0 && fromUrl < z.vocab.length ? fromUrl : z.vocab.indexOf(DEFAULT_WORD);
  }, [ready, tokenId, z.vocab]);

  useEffect(() => {
    if (selected !== null) zoey.requestToken(selected);
    if (compareId !== null) zoey.requestToken(compareId);
  }, [selected, compareId, z.status]);

  const pick = (id: number) => navigate(`/dictionary/${id}`);
  const info = selected === null ? undefined : z.tokenInfo.get(selected);
  const other = compareId === null ? undefined : z.tokenInfo.get(compareId);

  // Neighbors with duplicates folded together (" Max" and "Max" both read as "Max").
  const neighbors = useMemo(() => {
    if (!info) return [];
    const seen = new Set([wordOf(z.vocab[info.id])]);
    const out: { id: number; sim: number }[] = [];
    info.neighbors.ids.forEach((id, i) => {
      const w = wordOf(z.vocab[id]);
      if (!seen.has(w) && out.length < SHOWN_NEIGHBORS) {
        seen.add(w);
        out.push({ id, sim: info.neighbors.sims[i] });
      }
    });
    return out;
  }, [info, z.vocab]);

  return (
    <div className="pb-8">
      <section className="pt-10 pb-6">
        <p className="text-sm font-semibold uppercase tracking-wide text-accent-ink">TinyWriter's dictionary</p>
        <h1 className="mt-2 text-4xl font-bold tracking-tight sm:text-5xl">What TinyWriter knows about a word</h1>
        <p className="mt-3 max-w-3xl text-lg leading-relaxed text-ink-2">
          TinyWriter knows {int(4096)} tokens. Each one has <b className="text-ink">512 numbers that TinyWriter learned</b> while reading
          stories: its <i>embedding</i>. Nobody told TinyWriter what any word means. Words that get used in similar ways simply
          ended up with similar numbers.
        </p>
      </section>

      {z.status === "error" || mapError ? (
        <div className="rounded-2xl border border-bad/40 bg-surface p-6">
          <p className="font-semibold">TinyWriter's dictionary couldn't open.</p>
          <p className="mt-1 text-sm text-ink-2">{z.error ?? mapError}. Reloading the page usually fixes it.</p>
        </div>
      ) : !ready || selected === null ? (
        <div className="mx-auto max-w-md rounded-2xl border border-line bg-surface p-8 text-center shadow-sm">
          <p className="font-semibold">Opening TinyWriter's dictionary…</p>
          <div className="mt-5 h-2 overflow-hidden rounded-full bg-surface-2">
            <div className="h-full rounded-full bg-accent transition-[width]" style={{ width: `${Math.round(z.progress * 100)}%` }} />
          </div>
        </div>
      ) : (
        <>
          <div className="rounded-2xl border border-line bg-surface p-4 shadow-sm sm:p-5">
            <TokenSearch vocab={z.vocab} count={map.count} onPick={pick} />
            <div className="mt-3 flex flex-wrap items-center gap-2 text-sm">
              <span className="text-muted">Try:</span>
              {QUICK_PICKS.map((w) => {
                const id = z.vocab.indexOf(w);
                return (
                  id >= 0 && (
                    <button
                      key={w}
                      onClick={() => pick(id)}
                      className={`rounded-full border px-3 py-0.5 font-mono ${
                        id === selected ? "border-accent bg-accent-wash text-accent-ink" : "border-line text-ink-2 hover:bg-surface-2"
                      }`}
                    >
                      {w.trim()}
                    </button>
                  )
                );
              })}
            </div>
          </div>

          <div className="mt-5 grid items-start gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,24rem)]">
            <div className="min-w-0 space-y-5">
              <Card>
                <TokenHeader id={selected} vocab={z.vocab} count={map.count} />
                <Section
                  title="Its fingerprint: 512 learned numbers"
                  help={
                    <>
                      <p>
                        Each square is one of the 512 numbers TinyWriter learned for this token. <b>Red</b> is negative,{" "}
                        <b>blue</b> is positive, gray is close to zero.
                      </p>
                      <p>
                        No single number means anything on its own, like "is an animal". The meaning is spread across the
                        whole pattern. To see that, compare two words below.
                      </p>
                    </>
                  }
                >
                  {info ? (
                    <>
                      <Fingerprint vector={info.vector} scale={info.scale} label={wordOf(z.vocab[selected])} />
                      <FingerprintLegend />
                    </>
                  ) : (
                    <Pending />
                  )}
                </Section>
              </Card>

              <Card>
                <Compare
                  vocab={z.vocab}
                  count={map.count}
                  a={info}
                  b={other}
                  compareId={compareId}
                  onCompare={setCompareId}
                  suggestions={neighbors.slice(0, 3).map((n) => n.id)}
                />
              </Card>

              <Card>
                <Section
                  title="Map of every word TinyWriter knows"
                  help={
                    <>
                      <p>
                        All {int(z.vocab.length)} tokens, arranged so that tokens with similar numbers sit near each other.
                        Look for neighborhoods: names, colors, numbers, feelings, things you eat.
                      </p>
                      <p>
                        Squashing 512 numbers onto a flat page loses detail, so distances are only a rough guide. The
                        "most similar" list is the exact answer. Scroll or pinch to zoom, drag to move around, click a dot
                        to look it up.
                      </p>
                    </>
                  }
                >
                  <VocabMap map={map} vocab={z.vocab} selected={selected} neighbors={neighbors.map((n) => n.id)} onPick={pick} />
                </Section>
              </Card>
            </div>

            <div className="lg:sticky lg:top-4">
              <Card>
                <Section
                  title={`Most similar to “${wordOf(z.vocab[selected])}”`}
                  help={
                    <>
                      <p>
                        Similarity compares two tokens' 512 numbers: 1 means identical direction, 0 means unrelated.
                      </p>
                      <p>
                        "Similar" means <b>used in the same kinds of places</b>, not "means the same thing". That's why
                        opposites like <i>happy</i> and <i>sad</i> come out somewhat alike: they fit the same spots in a
                        sentence.
                      </p>
                    </>
                  }
                >
                  {info ? (
                    <ol className="space-y-1">
                      {neighbors.map((n) => (
                        <li key={n.id} className="group grid grid-cols-[minmax(0,7rem)_1fr_2.5rem_1.75rem] items-center gap-2">
                          <button onClick={() => pick(n.id)} className="truncate rounded-md px-1 text-left font-mono hover:bg-surface-2" title={`Look up “${wordOf(z.vocab[n.id])}”`}>
                            {wordOf(z.vocab[n.id])}
                          </button>
                          <div className="relative h-3.5">
                            <div className="absolute inset-0 rounded-r-[4px] bg-surface-2" />
                            <div className="absolute inset-y-0 left-0 rounded-r-[4px] bg-accent" style={{ width: `${Math.max(0, n.sim) * 100}%` }} />
                          </div>
                          <span className="tabular text-right text-sm text-ink-2">{n.sim.toFixed(2)}</span>
                          <button
                            onClick={() => setCompareId(n.id)}
                            title={`Compare with “${wordOf(z.vocab[n.id])}”`}
                            aria-label={`Compare with ${wordOf(z.vocab[n.id])}`}
                            className="grid size-7 place-items-center rounded-md text-muted opacity-60 hover:bg-surface-2 hover:text-ink group-hover:opacity-100"
                          >
                            <ArrowLeftRight size={14} />
                          </button>
                        </li>
                      ))}
                    </ol>
                  ) : (
                    <Pending />
                  )}
                </Section>
                <p className="mt-4 border-t border-line pt-3 text-xs text-muted">
                  Click a word to look it up, or <ArrowLeftRight size={11} className="inline" /> to compare.{" "}
                  <Link to="/" className="text-accent-ink hover:underline">
                    Back to Explore
                  </Link>
                </p>
              </Card>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function TokenHeader({ id, vocab, count }: { id: number; vocab: string[]; count: number[] }) {
  const t = vocab[id];
  const whole = t.startsWith(" ");
  return (
    <div className="mb-5">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <h2 className="font-mono text-4xl font-bold">{wordOf(t)}</h2>
        <span className="tabular text-sm text-muted">token #{id}</span>
      </div>
      <p className="mt-1 text-sm text-ink-2">
        Used <b className="tabular text-ink">{int(count[id])}</b> times in the training stories ·{" "}
        {whole ? "a whole word (it starts with a space)" : "a word piece (it attaches to the token before it)"}
      </p>
    </div>
  );
}

function Compare({
  vocab,
  count,
  a,
  b,
  compareId,
  onCompare,
  suggestions,
}: {
  vocab: string[];
  count: number[];
  a?: TokenInfo;
  b?: TokenInfo;
  compareId: number | null;
  onCompare: (id: number | null) => void;
  suggestions: number[];
}) {
  // "Line up" view: put both fingerprints in the order of the first word's
  // numbers, lowest to highest. The first word becomes a smooth red-to-blue
  // sweep; a similar word roughly follows it, an unrelated one looks like static.
  const [linedUp, setLinedUp] = useState(false);
  const sorted = useMemo(() => (a ? Array.from(a.vector, (_, i) => i).sort((i, j) => a.vector[i] - a.vector[j]) : undefined), [a]);
  const order = linedUp ? sorted : undefined;
  const nameA = a ? wordOf(vocab[a.id]) : "";
  const nameB = compareId === null ? "" : wordOf(vocab[compareId]);
  const sim = a && b ? cosine(a.vector, b.vector) : null;
  const unrelated = useMemo(() => {
    const w = vocab.indexOf(" the");
    return w >= 0 && a && w !== a.id ? w : vocab.indexOf(" was");
  }, [vocab, a]);

  return (
    <Section
      title="Compare two words"
      help={
        <>
          <p>
            <b>As they are</b> shows both fingerprints exactly like the one above. With 512 numbers each, it's hard to spot
            a resemblance by eye.
          </p>
          <p>
            <b>Line up</b> rearranges both grids the same way: in order of the first word's numbers, from its most negative
            (top left) to its most positive (bottom right). The first word turns into a smooth red-to-blue sweep. If the
            second word is similar, it roughly follows that sweep; if it's unrelated, it looks like static. The thin strip
            beside each grid is the average of each row, which makes the trend easy to see. Try a close neighbor, then a
            word like “the”.
          </p>
        </>
      }
    >
      {!a ? (
        <Pending />
      ) : compareId === null ? (
        <div>
          <p className="mb-3 text-sm text-ink-2">Pick a second word to compare with “{wordOf(vocab[a.id])}”.</p>
          <TokenSearch vocab={vocab} count={count} onPick={onCompare} placeholder="Compare with…" />
          <div className="mt-3 flex flex-wrap items-center gap-2 text-sm">
            <span className="text-muted">Try:</span>
            {[...suggestions, unrelated].map((id) => (
              <button key={id} onClick={() => onCompare(id)} className="rounded-full border border-line px-3 py-0.5 font-mono text-ink-2 hover:bg-surface-2">
                {wordOf(vocab[id])}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div>
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3 rounded-xl bg-surface-2 px-4 py-3">
            <p>
              <b className="font-mono">{wordOf(vocab[a.id])}</b> and <b className="font-mono">{wordOf(vocab[compareId])}</b>:{" "}
              {sim === null ? "…" : (
                <>
                  <b className="tabular">{sim.toFixed(2)}</b> <span className="text-ink-2">· {describeSimilarity(sim)}</span>
                </>
              )}
            </p>
            <button onClick={() => onCompare(null)} className="inline-flex items-center gap-1 rounded-lg px-2 py-1 text-xs font-medium text-accent-ink hover:bg-accent-wash">
              <X size={13} /> Clear
            </button>
          </div>
          <div className="mb-3 flex flex-wrap items-center gap-3">
            <Segmented
              value={linedUp ? "lined" : "plain"}
              onChange={(v) => setLinedUp(v === "lined")}
              options={[
                { value: "plain", label: "As they are" },
                { value: "lined", label: `Line up by “${nameA}”` },
              ]}
            />
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <div className="mb-1 text-sm">
                <b className="font-mono">{nameA}</b>
                {linedUp && <span className="text-muted"> · rearranged, lowest to highest</span>}
              </div>
              <Fingerprint vector={a.vector} scale={a.scale} order={order} label={nameA} />
            </div>
            <div>
              <div className="mb-1 text-sm">
                <b className="font-mono">{nameB}</b>
                {linedUp && <span className="text-muted"> · in {nameA}'s order</span>}
              </div>
              {b ? <Fingerprint vector={b.vector} scale={b.scale} order={order} label={nameB} /> : <Pending />}
            </div>
          </div>
          <p className="mt-1 text-xs leading-relaxed text-muted">
            {linedUp ? (
              <>
                Both grids are rearranged by {nameA}'s numbers, so {nameA} becomes a smooth sweep. The thin strip is each
                row's average: if {nameB}'s strip also runs red to blue, the two words point the same way.
              </>
            ) : (
              <>
                Same layout as the fingerprint above. Hard to compare by eye? Try <b>Line up by “{nameA}”</b>.
              </>
            )}
          </p>
        </div>
      )}
    </Section>
  );
}

function Card({ children }: { children: ReactNode }) {
  return <section className="rounded-2xl border border-line bg-surface p-5 shadow-sm sm:p-6">{children}</section>;
}

function Section({ title, help, children }: { title: string; help?: ReactNode; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <div className="mb-3 flex items-center justify-between gap-2">
        <h3 className="font-semibold">{title}</h3>
        {help && (
          <button
            onClick={() => setOpen((o) => !o)}
            className="inline-flex shrink-0 items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium text-accent-ink hover:bg-accent-wash"
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

const Pending = () => <p className="text-sm text-muted">Looking it up…</p>;

