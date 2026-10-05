import type { TokenTrace } from "../engine/protocol";

export interface Shape {
  layers: number;
  heads: number;
}

/** "All heads" averaged, or one specific head. */
export type HeadChoice = { layer: number; head: number } | "all";

/**
 * Attention weights from the trace's position back over every earlier
 * position, for one head or averaged over all of them.
 */
export function attentionRow(trace: TokenTrace, shape: Shape, choice: HeadChoice): Float32Array {
  const T = trace.pos + 1;
  if (choice !== "all") {
    const base = (choice.layer * shape.heads + choice.head) * T;
    return trace.attention.subarray(base, base + T);
  }
  const out = new Float32Array(T);
  const n = shape.layers * shape.heads;
  for (let k = 0; k < n; k++) for (let j = 0; j < T; j++) out[j] += trace.attention[k * T + j] / n;
  return out;
}

/** For one head: the earlier position it attended to most, and how much. */
export function strongestTarget(row: Float32Array, skip: Set<number> = new Set()) {
  let best = -1;
  let weight = 0;
  row.forEach((w, j) => {
    if (!skip.has(j) && w > weight) {
      best = j;
      weight = w;
    }
  });
  return { pos: best, weight };
}

const SINGULAR = new Set(["she", "he", "her", "his", "him"]);
const PRONOUNS = new Set([...SINGULAR, "they", "them", "their", "it", "its"]);
// Capitalized words that aren't names.
const NOT_NAMES = new Set(
  "The Once One Then When But And So As After Before Every A An In On At It I We You They She He His Her Their Mom Dad Yes No OK Oh".split(" "),
);

export const isPronoun = (text: string) => PRONOUNS.has(text.trim().toLowerCase());
const isNameLike = (text: string) => /^\s?[A-Z][a-z]+$/.test(text) && !NOT_NAMES.has(text.trim());

export interface Pick {
  pos: number;
  head: HeadChoice;
  /** True when the chosen head is looking at a name: the "who is she?" moment. */
  findsName: boolean;
}

/**
 * Pick something interesting to show first. Best case: a pronoun TinyWriter wrote
 * ("she", "he") where some head is clearly looking back at a name - that's
 * TinyWriter working out who "she" is. Otherwise, a token and its most decisive head.
 */
export function autoPick(tokens: TokenTrace[], shape: Shape): Pick | null {
  const written = tokens.filter((t) => t.source === "zoey" && t.next !== undefined);
  if (!written.length) return null;
  const pronouns = [
    ...written.filter((t) => SINGULAR.has(t.text.trim().toLowerCase())),
    ...written.filter((t) => isPronoun(t.text) && !SINGULAR.has(t.text.trim().toLowerCase())),
  ];
  for (const t of pronouns) {
    const found = nameHead(t, tokens, shape);
    if (found) return { pos: t.pos, head: found, findsName: true };
  }
  const target = pronouns[0] ?? written[Math.min(8, written.length - 1)];
  return { pos: target.pos, head: bestHead(target, shape), findsName: false };
}

/** The head putting the most weight (at least 30%) on a name-like earlier token. */
function nameHead(trace: TokenTrace, tokens: TokenTrace[], shape: Shape): HeadChoice | null {
  let best: HeadChoice | null = null;
  let bestW = 0.3;
  for (let layer = 0; layer < shape.layers; layer++)
    for (let head = 0; head < shape.heads; head++) {
      const { pos, weight } = strongestTarget(attentionRow(trace, shape, { layer, head }), new Set([0, trace.pos]));
      if (pos > 0 && weight > bestW && isNameLike(tokens[pos].text)) {
        bestW = weight;
        best = { layer, head };
      }
    }
  return best;
}

/**
 * The head that put the most weight on a single earlier token - ignoring the
 * start marker (where idle heads park their attention), the token itself, and
 * the one right before it (which many heads simply copy from).
 */
export function bestHead(trace: TokenTrace, shape: Shape): HeadChoice {
  const q = trace.pos;
  const skip = new Set([0, q, q - 1]);
  let best: HeadChoice = "all";
  let bestW = 0;
  for (let layer = 0; layer < shape.layers; layer++)
    for (let head = 0; head < shape.heads; head++) {
      const { weight } = strongestTarget(attentionRow(trace, shape, { layer, head }), skip);
      if (weight > bestW) {
        bestW = weight;
        best = { layer, head };
      }
    }
  return best;
}

// ------------------------------------------------- color by attention target

export interface TargetGroup {
  /** The first place in the story these heads focus on. */
  pos: number;
  /** Every place a head looked at this word (e.g. "Rob" both times it appears). */
  positions: number[];
  heads: number;
  weight: number;
  /** Color slot, or null for neutral (START, and minor targets beyond the palette). */
  slot: number | null;
}

export interface HeadFocus {
  pos: number;
  weight: number;
  group: TargetGroup;
}

/**
 * For every head, the earlier token it focuses on most - then group heads by
 * that *word*, so all heads looking at "Lily" (in any of the places it
 * appears) share a color with "Lily" in the story. The biggest groups get
 * colors; START (where idle heads rest) and anything past the palette stay
 * neutral.
 */
export function attentionMap(trace: TokenTrace, shape: Shape, slots: number, tokens: TokenTrace[]) {
  const byWord = new Map<number, TargetGroup>(); // keyed by token id
  const cells: HeadFocus[][] = [];
  for (let layer = 0; layer < shape.layers; layer++) {
    const row: HeadFocus[] = [];
    for (let head = 0; head < shape.heads; head++) {
      const { pos, weight } = strongestTarget(attentionRow(trace, shape, { layer, head }));
      const id = tokens[pos].id;
      let group = byWord.get(id);
      if (!group) byWord.set(id, (group = { pos, positions: [], heads: 0, weight: 0, slot: null }));
      if (pos < group.pos) group.pos = pos;
      if (!group.positions.includes(pos)) group.positions.push(pos);
      group.heads++;
      group.weight += weight;
      row.push({ pos, weight, group });
    }
    cells.push(row);
  }
  const groups = [...byWord.values()].sort((a, b) => b.weight - a.weight);
  let next = 0;
  for (const g of groups) if (g.pos !== 0 && next < slots) g.slot = next++;
  // Colored groups first, then the neutral ones, START last.
  groups.sort((a, b) => (a.slot ?? 99) - (b.slot ?? 99) || (a.pos === 0 ? 1 : 0) - (b.pos === 0 ? 1 : 0) || b.weight - a.weight);
  return { cells, groups };
}
