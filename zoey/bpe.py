"""
Chapter 2: Byte-pair encoding (BPE) - the tokenizer real LLMs use.

The character tokenizer from chapter 1 makes the model spell everything out
letter by letter. BPE fixes that by *learning* a vocabulary of useful chunks.

The idea is beautifully simple:

  1. Start with text as raw bytes (256 possible values - so ANY text works).
  2. Find the pair of neighbouring tokens that appears most often.
  3. Give that pair a new token id, and replace it everywhere.
  4. Repeat until the vocabulary is as big as you want.

    "the cat the hat"
    t h e _ c a t _ t h e _ h a t        start: bytes
    (t h) e _ c a t _ (t h) e _ h a t    most common pair: "t"+"h" -> new token "th"
    (th e) _ c a t _ (th e) _ h a t      next: "th"+"e" -> "the"
    ...

After a few thousand merges, common words like " the", " Lily" and " happy"
are single tokens, while rare words get built from smaller pieces.

Two details from GPT-2 that make it work well:

  - We first split text into "chunks" (words, numbers, punctuation) with a
    regex, and never merge across chunks. Otherwise we'd learn silly tokens
    like "dog." and "dog!" and "dog?" separately.
  - The space goes at the *front* of a word: " dog" is one token. That's why
    you'll see tokens with a leading space everywhere in the viewer.

Train one:
    uv run python -m zoey.bpe --vocab-size 4096
"""

from __future__ import annotations

import argparse
import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

TOKENIZER_DIR = Path(__file__).resolve().parent.parent / "tokenizers"

# Split text into chunks: contractions, words (with their leading space),
# numbers, runs of punctuation, and whitespace. Merges never cross chunks.
SPLIT_PATTERN = r"""'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"""

EOT = "<|endoftext|>"


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]], name: str):
        self.name = name
        self.merges = merges
        self.pattern = re.compile(SPLIT_PATTERN)

        # merge (a, b) -> new token id. Lower id = learned earlier = applied first.
        self.ranks = {pair: 256 + i for i, pair in enumerate(merges)}

        # The bytes behind every token, built up from the merges.
        self.vocab_bytes = [bytes([i]) for i in range(256)]
        for a, b in merges:
            self.vocab_bytes.append(self.vocab_bytes[a] + self.vocab_bytes[b])

        # One special token at the very end, marking the boundary between stories.
        self.eot_id = len(self.vocab_bytes)
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab_bytes) + 1  # +1 for <|endoftext|>

    # ------------------------------------------------------------------ encode

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in self.pattern.findall(text):
            # Stories reuse the same words constantly, so caching chunk -> ids
            # makes encoding a whole dataset fast.
            cached = self._cache.get(chunk)
            if cached is None:
                cached = self._encode_chunk(chunk.encode("utf-8"))
                if len(self._cache) < 500_000:
                    self._cache[chunk] = cached
            ids.extend(cached)
        return ids

    def _encode_chunk(self, data: bytes) -> list[int]:
        """Apply the learned merges to one chunk, earliest-learned first."""
        ids = list(data)
        while len(ids) >= 2:
            # Of all neighbouring pairs, find the one that was learned first.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, 1 << 30))
            if pair not in self.ranks:
                break  # nothing left to merge
            ids = _merge(ids, pair, self.ranks[pair])
        return ids

    # ------------------------------------------------------------------ decode

    def decode(self, ids) -> str:
        out = b"".join(b"\n\n" if i == self.eot_id else self.vocab_bytes[i] for i in ids)
        return out.decode("utf-8", errors="replace")

    def token_strings(self) -> list[str]:
        """How each token is displayed in the viewer."""
        out = []
        for b in self.vocab_bytes:
            try:
                out.append(b.decode("utf-8"))
            except UnicodeDecodeError:
                out.append(b.hex(" ").join(["<", ">"]))  # a partial UTF-8 character
        return out + ["<eot>"]

    # ------------------------------------------------------------ save / load

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "name": self.name,
                    "type": "byte-bpe",
                    "pattern": SPLIT_PATTERN,
                    "special": {EOT: self.eot_id},
                    "merges": self.merges,
                }
            )
        )

    @classmethod
    def load(cls, name: str) -> "BPETokenizer":
        data = json.loads((TOKENIZER_DIR / f"{name}.json").read_text())
        return cls([tuple(m) for m in data["merges"]], data["name"])

    # ------------------------------------------------------------------ train

    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = True) -> "BPETokenizer":
        """
        Learn `vocab_size - 257` merges from `text`.

        Doing this naively (re-counting every pair after every merge) takes
        hours. The trick: we work on *unique chunks* with their counts, and
        after each merge only update the chunks that actually contained the
        merged pair.
        """
        n_merges = vocab_size - 256 - 1  # 256 bytes + merges + <|endoftext|>

        # 1. Split into chunks and count them. "the" appears millions of
        #    times, but we only need to store it once with its count.
        chunk_counts = Counter(re.findall(SPLIT_PATTERN, text))
        words = [list(c.encode("utf-8")) for c in chunk_counts]
        freqs = list(chunk_counts.values())
        if verbose:
            print(f"{sum(freqs):,} chunks, {len(words):,} unique")

        # 2. Count every neighbouring pair, and remember which words contain it.
        pair_counts: Counter = Counter()
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, (w, f) in enumerate(zip(words, freqs)):
            for pair in zip(w, w[1:]):
                pair_counts[pair] += f
                where[pair].add(wi)

        # A heap lets us find the most frequent pair quickly. Entries can go
        # stale as counts change; we just skip those when we pop them.
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        merges: list[tuple[int, int]] = []
        while len(merges) < n_merges and heap:
            neg, pair = heapq.heappop(heap)
            if pair_counts.get(pair, 0) != -neg or -neg < 2:
                continue  # stale entry
            new_id = 256 + len(merges)
            merges.append(pair)

            # 3. Rewrite only the words that contain this pair.
            touched: set[tuple[int, int]] = set()
            for wi in where.pop(pair):
                w, f = words[wi], freqs[wi]
                for p in zip(w, w[1:]):
                    pair_counts[p] -= f
                    touched.add(p)
                w = _merge(w, pair, new_id)
                words[wi] = w
                for p in zip(w, w[1:]):
                    pair_counts[p] += f
                    where[p].add(wi)
                    touched.add(p)
            pair_counts.pop(pair, None)
            for p in touched:
                if pair_counts.get(p, 0) > 0:
                    heapq.heappush(heap, (-pair_counts[p], p))

            if verbose and (len(merges) % 500 == 0 or len(merges) <= 10):
                tok = _bytes_of(merges, new_id).decode("utf-8", errors="replace")
                print(f"merge {len(merges):5d}: {tok!r} ({-neg:,} times)")

        return cls(merges, f"bpe-{vocab_size}")


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def _bytes_of(merges: list[tuple[int, int]], token: int) -> bytes:
    if token < 256:
        return bytes([token])
    a, b = merges[token - 256]
    return _bytes_of(merges, a) + _bytes_of(merges, b)


def main():
    from .data import TINYSTORIES_FILES, download, STORY_SEPARATOR

    ap = argparse.ArgumentParser(description="Train a BPE tokenizer on TinyStories.")
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--sample-mb", type=int, default=100, help="how much training text to learn from")
    args = ap.parse_args()

    # Learn from the start of the plain stories plus some of the instruction
    # format, so words like "Summary:" get tokens too.
    sample = []
    for key, mb in (("full", args.sample_mb), ("instruct-small", 10)):
        with open(download(TINYSTORIES_FILES[key]), encoding="utf-8") as f:
            sample.append(f.read(mb * 1_000_000))
    text = "\n".join(sample).replace(STORY_SEPARATOR, "\n")

    tok = BPETokenizer.train(text, args.vocab_size)
    path = TOKENIZER_DIR / f"{tok.name}.json"
    tok.save(path)
    print(f"Saved {path}")

    demo = "Once upon a time, there was a little girl named Lily."
    ids = tok.encode(demo)
    print(f"\n{demo!r}\n-> {len(ids)} tokens: {[tok.vocab_bytes[i].decode() for i in ids]}")


if __name__ == "__main__":
    main()
