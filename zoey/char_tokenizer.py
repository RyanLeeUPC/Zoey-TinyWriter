"""
The simplest possible tokenizer: one token per character.

A neural network can't read text, only numbers. A tokenizer is the dictionary
that turns text into a list of integers (encode) and back again (decode).

Here every character gets its own id:

    "Hi!"  ->  [ 61, 35, 82 ]

It's easy to understand, but wasteful: the model has to spend effort learning
that "t", "h", "e" usually travel together. Chapter 2 builds a smarter
tokenizer (BPE) that learns common chunks like "the" and "ing" as single tokens.
"""

from __future__ import annotations

import string

# Typographic characters that TinyStories uses, mapped to plain ASCII so the
# vocabulary stays small.
NORMALIZE = str.maketrans(
    {
        "‘": "'",  # left single quote
        "’": "'",  # right single quote / apostrophe
        "“": '"',  # left double quote
        "”": '"',  # right double quote
        "–": "-",  # en dash
        "—": "-",  # em dash
        "…": "...",  # ellipsis
        " ": " ",  # non-breaking space
    }
)

EOT = "<eot>"  # "end of text": marks where one story ends and the next begins

# The vocabulary, in an order chosen so that the viewer's heatmaps group
# similar characters together: whitespace, lowercase, uppercase, digits, punctuation.
VOCAB: list[str] = (
    [EOT, "\n", " "]
    + list(string.ascii_lowercase)
    + list(string.ascii_uppercase)
    + list(string.digits)
    + [c for c in string.punctuation]
)


class CharTokenizer:
    name = "char"

    def __init__(self):
        self.vocab = VOCAB
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.eot_id = self.stoi[EOT]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        text = text.translate(NORMALIZE)
        # Characters outside the vocabulary (emoji, accents, ...) are rare in
        # TinyStories, so we just drop them.
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join("\n\n" if i == self.eot_id else self.vocab[i] for i in ids)

    def token_strings(self) -> list[str]:
        """How each token should be displayed in the viewer."""
        return list(self.vocab)
