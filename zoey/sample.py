"""
Generating text, one token at a time.

The model never writes a story. It only ever answers: "given this text so far,
how likely is each possible next token?" To write, we:

  1. ask the model for next-token probabilities
  2. pick one (randomly, weighted by those probabilities)
  3. stick it on the end of the text
  4. repeat

Picking randomly instead of always taking the top choice matters - always
taking the most likely token makes text that loops forever ("the the the").

Temperature controls how adventurous the picks are:
  - low (0.5)  -> sharpen the probabilities: safer, more repetitive
  - 1.0        -> use the model's probabilities as-is
  - high (1.5) -> flatten them: more surprising, more nonsense
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
):
    """
    Extend `idx` (shape (1, time)) by `max_new_tokens` tokens.

    Returns the new token ids and, for each one, the probability the model
    gave it - the viewer uses that to show how confident TinyWriter was.
    """
    block_size = model.config.block_size
    new_ids, new_probs = [], []
    for _ in range(max_new_tokens):
        context = idx[:, -block_size:]  # the model can only see so far back
        logits, _ = model(context)
        logits = logits[:, -1, :] / temperature  # only the last position matters
        if top_k is not None:
            # Forbid everything outside the k most likely tokens.
            kth = torch.topk(logits, top_k).values[:, -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1, generator=generator)
        idx = torch.cat([idx, next_id], dim=1)
        new_ids.append(next_id.item())
        new_probs.append(probs[0, next_id.item()].item())
    return new_ids, new_probs
