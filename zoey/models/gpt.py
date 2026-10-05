"""
The real thing: a GPT-style transformer.

The bigram model could only see one token back. A transformer can look at
*every* earlier token and decide which ones matter. Here's the whole flow
for one forward pass:

    token ids            [ Once, upon, a, time ]
        |
    embeddings           each id -> a vector of d_model numbers (what the token is)
      + positions        each slot -> a vector too (where the token is)
        |
    block 1..n_layer     each block does two things, and ADDS its result back
        |                onto the running vector (the "residual stream"):
        |                  attention: look back at earlier tokens, gather info
        |                  MLP:       think about what was gathered, per token
        |
    final LayerNorm
        |
    unembed              vector -> one score per vocabulary entry (the logits)
        |
    softmax              scores -> next-token probabilities

The residual stream is the key mental model: every token has a vector that
flows up through the layers, and each layer reads from it and writes a small
update into it. The viewer's "layer by layer" view decodes that vector at
every layer to show what the model "would say" if it stopped there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 512  # longest context the model can see
    n_layer: int = 8  # how many transformer blocks
    n_head: int = 6  # attention heads per block
    d_model: int = 384  # size of each token's vector
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    """
    Attention: every token gets to look back at earlier tokens and pull in
    information from them.

    Each token makes three vectors from its own state:
      query (q): "what am I looking for?"
      key   (k): "what do I contain?"
      value (v): "what will I hand over if you pick me?"

    A token compares its query against every earlier token's key (a dot
    product). High scores = "that's relevant to me". Softmax turns the scores
    into weights that sum to 1, and the token takes that weighted mix of the
    earlier tokens' values.

    "Causal" means no peeking: a token can only look at itself and tokens
    before it, never after. (Otherwise predicting the next token would be
    cheating - the answer would be right there.)

    "Multi-head" means we do this n_head times in parallel with smaller
    vectors, so different heads can specialize: one might track the
    previous token, another the subject of the sentence, another quotes.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, record: list | None = None) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # Split each vector into n_head smaller ones: (B, T, C) -> (B, heads, T, head_dim)
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))

        if record is None:
            # Fast path for training: PyTorch's fused kernel computes exactly
            # the same thing as the step-by-step version below.
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            # Step by step, so we can record the attention pattern.
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, heads, T, T)
            future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            scores = scores.masked_fill(future, float("-inf"))  # no peeking ahead
            weights = F.softmax(scores, dim=-1)  # each row sums to 1
            record.append(weights)
            y = weights @ v  # weighted mix of values

        # Glue the heads back together and mix them.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    """
    After attention gathers information, the MLP processes it - separately
    for each token. It widens the vector 4x, applies a non-linearity (GELU),
    and narrows it back. Much of a model's "knowledge" is thought to live in
    these weights.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x), approximate="tanh")))


class Block(nn.Module):
    """
    One transformer block: attention, then MLP, each with a LayerNorm in
    front and a residual connection around it (`x = x + ...`).

    The residual connection means each block only has to learn a small
    *change* to the token's vector, not rebuild it from scratch - that's
    what makes deep stacks of these trainable.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, record: list | None = None):
        x = x + self.attn(self.ln1(x), record)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.config = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)  # token embeddings
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)  # position embeddings
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Weight tying: the same matrix turns ids into vectors (wte) and
        # vectors back into scores (lm_head). Saves parameters and helps.
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # Scale down the layers that write into the residual stream, so the
        # stream doesn't grow with depth at the start of training.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.shape[1]
        pos = torch.arange(T, device=idx.device)
        return self.drop(self.wte(idx) + self.wpe(pos))

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
        return logits, loss

    # ------------------------------------------------------------------
    # The glass box. Everything below exists only to show what's inside.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inspect(self, idx: torch.Tensor, top_k: int = 5) -> dict:
        """
        Run `idx` (shape (1, T)) through the model and capture:

        attention:   (layers, heads, T, T) - who looked at whom
        lens_*:      the "logit lens" - decode the residual stream after every
                     layer as if it were the last one. Shows at which depth
                     the model works out its answer.
        induction:   (layers, heads) - how strongly each head does "copy what
                     came after this token last time", the classic in-context
                     learning circuit.
        """
        record: list = []
        x = self.embed(idx)
        stream = [x]  # the residual stream after embedding and after each block
        for block in self.blocks:
            x = block(x, record)
            stream.append(x)

        lens = torch.stack([F.softmax(self.lm_head(self.ln_f(s[0])).float(), dim=-1) for s in stream])
        top_p, top_i = lens.topk(top_k, dim=-1)  # (layers+1, T, k)
        T = idx.shape[1]
        target = lens[:, torch.arange(T - 1), idx[0, 1:]]  # (layers+1, T-1)

        return {
            "attention": torch.stack([w[0] for w in record]),
            "lens_top_ids": top_i,
            "lens_top_probs": top_p,
            "lens_target": target,
            "induction": self.induction_scores(),
        }

    @torch.no_grad()
    def induction_scores(self, length: int = 48, batch: int = 8) -> torch.Tensor:
        """
        Feed random tokens repeated twice: [A B C D ... A B C D ...].

        In the second half, a head that has learned induction looks back at
        the token that came *after* the current token's first appearance -
        because that's what will probably come next. We measure how much
        attention each head puts exactly there.

        Before training, every head scores near 0. At some point during
        training, a few heads suddenly jump towards 1. That jump is one of the
        best-known moments in a transformer's training.
        """
        length = min(length, self.config.block_size // 2)
        g = torch.Generator(device="cpu").manual_seed(0)
        device = self.wte.weight.device
        rand = torch.randint(256, self.config.vocab_size - 1, (batch, length), generator=g).to(device)
        seq = torch.cat([rand, rand], dim=1)
        record: list = []
        x = self.embed(seq)
        for block in self.blocks:
            x = block(x, record)
        att = torch.stack(record)  # (layers, batch, heads, 2L, 2L)
        q = torch.arange(length, 2 * length, device=device)  # positions in the second copy
        k = q - length + 1  # the token after the earlier copy of the current token
        return att[:, :, :, q, k].mean(dim=(1, 3))  # (layers, heads)
