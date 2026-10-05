import type { TokenTrace } from "../engine/protocol";
import { pct } from "../lib/format";
import { underline } from "../lib/colors";

/** How to paint one token: a background tint, ink, a thin inset underline, or a thick underline. */
export interface Paint {
  bg?: string;
  ink?: string;
  underline?: string;
  /** A thick underline below the text, leaving the word itself plain. */
  mark?: string;
}

/**
 * The story, as clickable tokens. What the colors mean is decided by the
 * page (attention targets, or confidence); the story just paints them.
 */
export function Story({
  tokens,
  selected,
  paint,
  outlined,
  busy,
  onSelect,
  onHover,
}: {
  tokens: TokenTrace[];
  selected: number | null;
  paint: Map<number, Paint>;
  /** Tokens to ring, e.g. the word the hovered head is looking at. */
  outlined: Set<number> | null;
  busy: boolean;
  onSelect: (pos: number) => void;
  onHover: (pos: number | null) => void;
}) {
  return (
    <div className="font-mono text-[16px] leading-[2.1rem] whitespace-pre-wrap break-words" onMouseLeave={() => onHover(null)}>
      {tokens.map((t) => {
        const p = paint.get(t.pos);
        const isSel = t.pos === selected;
        const isNext = selected !== null && t.pos === selected + 1;
        const ring = outlined?.has(t.pos) ? "outline-2 outline-offset-1 outline-ink" : "";
        const style =
          p && !isSel
            ? {
                background: p.bg,
                color: p.ink,
                boxShadow: p.underline ? `inset 0 -2px 0 ${p.underline}` : undefined,
                ...(p.mark && underline(p.mark)),
              }
            : undefined;
        const prev = tokens[t.pos - 1];
        const title =
          t.source === "start"
            ? "Start-of-story marker. Every story TinyWriter trained on began with one."
            : prev?.nextProb !== undefined
              ? `TinyWriter gave this a ${pct(prev.nextProb)} chance`
              : undefined;

        if (t.source === "start") {
          return (
            <span
              key={t.pos}
              title={title}
              style={style}
              onMouseEnter={() => onHover(t.pos)}
              className={`mr-2 inline-block rounded-md border border-line px-1.5 align-middle font-sans text-[11px] leading-5 font-semibold tracking-wide text-muted uppercase ${ring}`}
            >
              start
            </span>
          );
        }

        // Keep a token's leading space outside the colored box, so highlights hug the word.
        const [, space, word] = t.text.match(/^(\s*)([\s\S]*)$/)!;
        return (
          <span key={t.pos}>
            {space}
            {word && (
              <span
                role="button"
                tabIndex={0}
                title={title}
                onClick={() => onSelect(t.pos)}
                onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onSelect(t.pos)}
                onMouseEnter={() => onHover(t.pos)}
                style={style}
                className={`cursor-pointer rounded-[4px] px-[1px] transition-colors ${ring} ${
                  isSel
                    ? "bg-ink text-page"
                    : p
                      ? ""
                      : t.source === "prompt"
                        ? "text-accent-ink hover:bg-accent-wash"
                        : "hover:bg-surface-2"
                } ${t.source === "prompt" ? "font-bold" : ""} ${isNext && !p ? "underline decoration-accent decoration-2 underline-offset-4" : ""}`}
              >
                {word}
              </span>
            )}
          </span>
        );
      })}
      {busy && <span className="ml-0.5 inline-block h-5 w-2 translate-y-1 animate-pulse rounded-sm bg-accent" />}
    </div>
  );
}
