# Chapter 2: Tokenization

> **Goal:** stop making TinyWriter read letter by letter. Build the tokenizer that real LLMs use, byte-pair encoding (BPE), from scratch.
>
> **Code:** [`zoey/bpe.py`](../zoey/bpe.py) · **Trained tokenizer:** [`tokenizers/bpe-4096.json`](../tokenizers/bpe-4096.json)

## Why letters are a bad unit

In chapter 1, every character was a token. That made the bigram model easy to understand, but it's wasteful for a real model:

- **Sequences get long.** "Once upon a time, there was a little girl named Lily." is 53 characters. A model that can look back 512 tokens can only see a few sentences.
- **The model has to learn spelling before meaning.** It spends effort figuring out that `t`, `h`, `e` travel together before it can learn anything about *the*.

Going the other way, one token per *word*, has its own problems. Words like *dragon*, *dragons* and *dragonfly* would be unrelated tokens. Any word not in the dictionary (a name, a typo, an emoji) can't be represented at all.

BPE sits in between: **common words become single tokens, and rare words get built from smaller pieces.** Here's TinyWriter's tokenizer on a few inputs (`·` marks a space):

| Text | Tokens |
|---|---|
| `Once upon a time, there was a little girl named Lily.` | `Once` `·upon` `·a` `·time` `,` `·there` `·was` `·a` `·little` `·girl` `·named` `·Lily` `.` (53 characters → **13 tokens**) |
| `The dragon was unhappy.` | `The` `·dragon` `·was` `·un` `h` `appy` `.` |
| `Supercalifragilistic!` | `S` `uper` `c` `al` `if` `rag` `il` `ist` `ic` `!` |
| `Zoey-TinyWriter` | `Z` `o` `ey` `-` `T` `iny` `W` `r` `it` `er` |

Yes, TinyWriter doesn't even have a token for its own name: it never appears in children's stories.

## How BPE learns its vocabulary

Start with text as raw **bytes**. There are only 256 possible bytes, and any text in any language (including emoji) is made of them, so nothing is ever "unknown." Then repeat:

1. Count every pair of neighboring tokens in the training text.
2. Take the most common pair and give it a new token id.
3. Replace that pair everywhere with the new token.

These are the first merges TinyWriter's tokenizer learned, from 100 MB of stories:

```
merge     1: ' t'       (3,115,876 times)
merge     2: 'he'       (3,109,166 times)
merge     3: ' a'       (2,343,417 times)
merge     7: ' the'     (1,414,046 times)   <- ' t' + 'he'
merge    10: ' to'      (1,025,405 times)
merge  1000: ' follow'      (6,471 times)
merge  3000: ' houses'        (898 times)
```

Notice how ` the` is built from two earlier merges, ` t` and `he`. Merges stack on top of each other, so later tokens are longer and rarer. We stop after 3,839 merges, which gives 256 bytes + 3,839 merges + 1 special token = **4,096 tokens**.

### Two details borrowed from GPT-2

**Chunking first.** Before merging, a regular expression splits text into chunks (words, numbers, punctuation, whitespace), and merges never cross a chunk boundary. Without this, the tokenizer would waste vocabulary on things like `dog.` `dog!` `dog,` as separate tokens.

**Spaces belong to the next word.** ` dog` (with a leading space) is one token. That's why the tokenizer handles `The dragon was unhappy.` the way it does: ` happy` is a token, but `happy` *without* a space in front isn't. Inside ` unhappy` it has to fall back to smaller pieces. You'll see leading spaces on tokens everywhere in the viewer.

### Making training fast

The obvious implementation (recount every pair in the whole dataset after each merge) would take hours. [`bpe.py`](../zoey/bpe.py) uses two tricks that are worth reading:

1. **Count unique chunks once.** 100 MB of stories is 26 million chunks, but only 22,538 *unique* ones. The word ` the` is stored once, with its count.
2. **Only update what changed.** After merging a pair, only the chunks that contained it need recounting. A heap keeps the most frequent pair at the top.

Training takes a few seconds.

## Encoding new text

To tokenize new text, split it into chunks and then replay the merges **in the order they were learned**: always apply the earliest-learned merge available first. That reproduces exactly the tokens the training process would have made.

Decoding is the easy direction. Look up each token's bytes, join them, and turn the bytes back into text.

## Try it

```bash
uv run python -m zoey.bpe --vocab-size 4096
```

Then in Python:

```python
from zoey.bpe import BPETokenizer
tok = BPETokenizer.load("bpe-4096")
ids = tok.encode("The dog was happy.")
print(ids, [tok.decode([i]) for i in ids])
```

## Exercises

1. **Vocabulary size.** Train with `--vocab-size 512` and `--vocab-size 16384`. How many tokens does the Lily sentence take with each? What does a bigger vocabulary cost the model? (Hint: every token needs its own row in the embedding table.)
2. **Your own name.** How many tokens is your name? Why are names from TinyStories (Lily, Tim, Sue) single tokens?
3. **Other languages.** Tokenize a sentence in another language. Why does it take so many more tokens? (This is a real fairness problem for large LLMs, whose tokenizers are mostly trained on English.)

## What's next

With text turned into compact tokens, chapter 3 gives each token a **vector**, a list of numbers the model can learn, and shows similar words drifting together during training.
