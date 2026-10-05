"""
Chapter 1: the bigram model - the smallest thing that deserves the name
"language model".

A bigram model predicts the next character by looking at exactly ONE
character: the current one. That's it. No memory of anything earlier.

Its entire "brain" is a single table with one row per character:

                   next char ->
                   a     b     c    ...
    current   a  [ 0.01  0.03  0.04 ... ]
    char      b  [ 0.09  0.02  0.00 ... ]
              q  [ 0.00  0.00  0.00 ... u: 0.97 ]

Row "q" says: after a "q", the next character is almost always "u".

We don't fill in this table by counting (although we could!). We start it
full of random numbers and let gradient descent nudge it, step by step, until
its guesses match the training text. That's the same training process that
every later chapter uses for much bigger models - here it's just small enough
to watch every single number change.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BigramConfig:
    vocab_size: int
    block_size: int = 256  # how many characters per training window


class BigramModel(nn.Module):
    def __init__(self, config: BigramConfig):
        super().__init__()
        self.config = config
        # The whole model. Row i holds the scores ("logits") for which
        # character follows character i.
        self.table = nn.Embedding(config.vocab_size, config.vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        # idx: (batch, time) token ids.
        # Looking up each token's row gives us scores for every possible next token.
        logits = self.table(idx)  # (batch, time, vocab)

        loss = None
        if targets is not None:
            # Cross-entropy measures how surprised the model was by the real
            # next character. Lower = better guesses.
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    @torch.no_grad()
    def inspect(self, idx: torch.Tensor) -> dict:
        """
        Everything the viewer needs to see inside the model.
        For the bigram model, the inside IS the table, so we export all of it
        as probabilities (each row sums to 1).
        """
        probs = F.softmax(self.table.weight, dim=-1)
        return {"bigram": probs}
