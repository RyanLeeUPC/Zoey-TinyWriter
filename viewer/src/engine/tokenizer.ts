/**
 * TinyWriter's BPE tokenizer, in the browser. A line-by-line port of zoey/bpe.py:
 * split text into chunks, then replay the learned merges in the order they
 * were learned.
 */

import { printable } from "../lib/format";

export interface TokenizerMeta {
  name: string;
  pattern: string;
  merges: [number, number][];
  eot_id: number;
}

const encoder = new TextEncoder();
// ignoreBOM: keep a leading byte-order mark, exactly like Python's decode does.
const decoder = new TextDecoder("utf-8", { ignoreBOM: true });

export class BPETokenizer {
  readonly eotId: number;
  readonly vocabSize: number;
  private readonly ranks = new Map<number, number>(); // pair key -> new token id
  private readonly vocabBytes: Uint8Array[] = [];
  private readonly pattern: RegExp;
  private readonly cache = new Map<string, number[]>();

  constructor(meta: TokenizerMeta) {
    this.pattern = new RegExp(meta.pattern, "gu");
    for (let i = 0; i < 256; i++) this.vocabBytes.push(Uint8Array.of(i));
    meta.merges.forEach(([a, b], i) => {
      this.ranks.set(pairKey(a, b), 256 + i);
      this.vocabBytes.push(concat(this.vocabBytes[a], this.vocabBytes[b]));
    });
    this.eotId = meta.eot_id;
    this.vocabSize = this.vocabBytes.length + 1;
  }

  encode(text: string): number[] {
    const ids: number[] = [];
    for (const [chunk] of text.matchAll(this.pattern)) {
      let cached = this.cache.get(chunk);
      if (!cached) {
        cached = this.encodeChunk(encoder.encode(chunk));
        this.cache.set(chunk, cached);
      }
      ids.push(...cached);
    }
    return ids;
  }

  private encodeChunk(bytes: Uint8Array): number[] {
    let ids = Array.from(bytes);
    while (ids.length >= 2) {
      // Of all neighbouring pairs, find the one that was learned first.
      let best = -1;
      let bestRank = Infinity;
      for (let i = 0; i < ids.length - 1; i++) {
        const rank = this.ranks.get(pairKey(ids[i], ids[i + 1]));
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          best = i;
        }
      }
      if (best < 0) break;
      const [a, b] = [ids[best], ids[best + 1]];
      const merged: number[] = [];
      for (let i = 0; i < ids.length; i++) {
        if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) {
          merged.push(bestRank);
          i++;
        } else merged.push(ids[i]);
      }
      ids = merged;
    }
    return ids;
  }

  decode(ids: number[]): string {
    const parts = ids.map((i) => (i === this.eotId ? encoder.encode("\n\n") : this.vocabBytes[i]));
    return decoder.decode(concat(...parts));
  }

  /** One token as text. Half of a multi-byte character shows as its hex bytes. */
  tokenText(id: number): string {
    if (id === this.eotId) return "<eot>";
    const bytes = this.vocabBytes[id];
    try {
      return printable(new TextDecoder("utf-8", { fatal: true, ignoreBOM: true }).decode(bytes));
    } catch {
      return `<${Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join(" ")}>`;
    }
  }
}

const pairKey = (a: number, b: number) => a * 65536 + b;

function concat(...arrays: Uint8Array[]): Uint8Array {
  const out = new Uint8Array(arrays.reduce((n, a) => n + a.length, 0));
  let o = 0;
  for (const a of arrays) {
    out.set(a, o);
    o += a.length;
  }
  return out;
}
