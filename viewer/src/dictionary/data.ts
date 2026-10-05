import { useEffect, useState } from "react";
import { MODEL_ID } from "../config";
import { showToken } from "../lib/format";

/** The precomputed map of every token, and how often each one appears in the training stories. */
export interface DictionaryMap {
  xy: Float32Array; // [x0, y0, x1, y1, ...], each in [-1, 1]
  count: number[];
}

let cached: Promise<DictionaryMap> | null = null;

export function useDictionaryMap() {
  const [map, setMap] = useState<DictionaryMap | null>(null);
  useEffect(() => {
    cached ??= fetch(`./models/${MODEL_ID}/dictionary.json`)
      .then((r) => r.json())
      .then((d) => ({ xy: Float32Array.from(d.xy), count: d.count }));
    cached.then(setMap);
  }, []);
  return map;
}

/** How a token reads as a word: its text without the leading space. */
export const wordOf = (token: string) => token.trim() || showToken(token);

/** Is this token a whole word (a letter run, usually with a leading space)? */
export const isWordLike = (token: string) => /^ ?[A-Za-z]+$/.test(token);

/** Plain-English strength of a cosine similarity between two embeddings. */
export function describeSimilarity(s: number): string {
  if (s >= 0.6) return "very similar";
  if (s >= 0.4) return "similar";
  if (s >= 0.25) return "a bit alike";
  if (s >= 0.1) return "barely alike";
  return "unrelated";
}

export function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / Math.sqrt(na * nb || 1);
}

/**
 * Find tokens matching a search: exact word first, whole words (leading
 * space) before word pieces, then the most common.
 */
export function searchTokens(query: string, vocab: string[], count: number[], limit = 8): number[] {
  const q = query.trim().toLowerCase();
  if (!q) return [];
  const hits: number[] = [];
  vocab.forEach((t, id) => {
    if (t !== "<eot>" && t.trim().toLowerCase().startsWith(q)) hits.push(id);
  });
  const rank = (id: number) => {
    const w = vocab[id].trim();
    return (w.toLowerCase() === q ? 0 : 4) + (w === query.trim() ? 0 : 1) + (vocab[id].startsWith(" ") ? 0 : 2);
  };
  return hits.sort((a, b) => rank(a) - rank(b) || count[b] - count[a]).slice(0, limit);
}
