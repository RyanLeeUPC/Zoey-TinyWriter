/** How a token is drawn on screen. Invisible characters get visible stand-ins. */
export function showToken(t: string): string {
  if (t === " ") return "␣";
  if (t === "\n") return "↵";
  if (t === "<eot>") return "⏹";
  return t;
}

/** Plain-English name for a token, for tooltips and sentences. */
export function nameToken(t: string): string {
  if (t === " ") return "a space";
  if (t === "\n") return "a new line";
  if (t === "<eot>") return "end of story";
  return `"${t}"`;
}

export function pct(p: number): string {
  if (p >= 0.995) return "100%";
  if (p >= 0.1) return `${Math.round(p * 100)}%`;
  if (p >= 0.001) return `${(p * 100).toFixed(1)}%`;
  return "<0.1%";
}

export function compact(n: number): string {
  if (n >= 1e9) return `${+(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${+(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${+(n / 1e3).toFixed(1)}K`;
  return `${n}`;
}

export function int(n: number): string {
  return n.toLocaleString("en-US");
}
