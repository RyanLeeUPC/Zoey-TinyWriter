import { useMemo, useState } from "react";
import { Search } from "lucide-react";
import { compact } from "../lib/format";
import { searchTokens, wordOf } from "./data";

/** A search box that finds tokens by their text. */
export function TokenSearch({
  vocab,
  count,
  onPick,
  placeholder = "Search for a word…",
  autoFocus,
}: {
  vocab: string[];
  count: number[];
  onPick: (id: number) => void;
  placeholder?: string;
  autoFocus?: boolean;
}) {
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const hits = useMemo(() => searchTokens(query, vocab, count), [query, vocab, count]);

  const pick = (id: number) => {
    onPick(id);
    setQuery("");
    setActive(0);
  };

  return (
    <div className="relative">
      <Search size={18} className="pointer-events-none absolute top-1/2 left-3.5 -translate-y-1/2 text-muted" />
      <input
        value={query}
        autoFocus={autoFocus}
        onChange={(e) => {
          setQuery(e.target.value);
          setActive(0);
        }}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") setActive((a) => Math.min(a + 1, hits.length - 1));
          else if (e.key === "ArrowUp") setActive((a) => Math.max(a - 1, 0));
          else if (e.key === "Enter" && hits[active] !== undefined) pick(hits[active]);
          else if (e.key === "Escape") setQuery("");
        }}
        placeholder={placeholder}
        aria-label={placeholder}
        className="w-full rounded-xl border border-line bg-surface-2 py-2.5 pr-3 pl-10 text-[16px] outline-none focus:border-accent focus:ring-2 focus:ring-accent-wash"
      />
      {query.trim() && (
        <ul className="absolute z-30 mt-1 w-full overflow-hidden rounded-xl border border-line bg-surface shadow-lg" role="listbox">
          {hits.length === 0 && <li className="px-3 py-2 text-sm text-muted">TinyWriter has no token starting with “{query.trim()}”.</li>}
          {hits.map((id, i) => (
            <li key={id} role="option" aria-selected={i === active}>
              <button
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => pick(id)}
                onMouseEnter={() => setActive(i)}
                className={`flex w-full items-center justify-between gap-3 px-3 py-2 text-left ${i === active ? "bg-surface-2" : ""}`}
              >
                <span className="font-mono">
                  {wordOf(vocab[id])}
                  {!vocab[id].startsWith(" ") && <span className="ml-2 font-sans text-xs text-muted">word piece</span>}
                </span>
                <span className="tabular text-xs text-muted">used {compact(count[id])}×</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
