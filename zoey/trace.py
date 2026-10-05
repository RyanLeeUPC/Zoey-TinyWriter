"""
The glass box: recording what the model is doing while it learns.

Every few steps during training we take a "snapshot" of the model and write
it to disk as JSON. The viewer (viewer/) loads those snapshots and plays them
back as a time-lapse.

Each snapshot answers three questions:

  1. What does TinyWriter write right now?        -> samples
  2. What does TinyWriter predict, letter by letter, for a fixed sentence?  -> probe
  3. What do TinyWriter's insides look like?      -> views (model-specific)

Samples use the SAME random seed at every snapshot. That way, when the text
changes between snapshots, it's because the model changed - not because the
dice rolled differently.

The format is documented in docs/trace-format.md.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .sample import generate

SCHEMA_VERSION = 1
RUNS_DIR = Path(__file__).resolve().parent.parent / "viewer" / "public" / "runs"


def checkpoint_schedule(max_steps: int, count: int) -> list[int]:
    """
    Which steps to snapshot. Models change fastest at the very start of
    training, so snapshots are spaced logarithmically: lots early, fewer late.
    """
    steps = np.geomspace(1, max_steps, count - 1).round().astype(int)
    return sorted({0, *steps.tolist(), max_steps})


def _round(x, digits: int = 4):
    """Round tensors/arrays to keep the JSON files small. Integers pass through."""
    if isinstance(x, torch.Tensor):
        if not x.is_floating_point():
            return x.cpu().tolist()
        x = x.detach().float().cpu().numpy()
    return np.round(np.asarray(x, dtype=np.float64), digits).tolist()


# Words for the embedding map, by group. Only words that are a single token
# get used. Watch the groups pull apart into clusters as TinyWriter trains.
EMBEDDING_WORDS = {
    "animals": "dog cat bird fish bear rabbit frog duck cow horse lion mouse pig monkey bunny puppy",
    "colors": "red blue green yellow pink purple orange black white brown",
    "people": "mom dad girl boy friend teacher baby sister brother grandma man lady",
    "names": "Lily Tim Tom Sue Ben Max Sam Anna Mia Lucy Jack Sara",
    "feelings": "happy sad angry scared excited proud sorry surprised curious tired",
    "actions": "ran jumped played looked smiled walked laughed cried found saw said went",
    "places": "park house school garden forest store beach room kitchen sky",
    "food": "cake apple cookie candy pie bread soup carrot banana milk",
}


class TraceWriter:
    def __init__(self, run: dict, model, tokenizer, trace_cfg: dict, device: str):
        self.dir = RUNS_DIR / run["id"]
        self.dir.mkdir(parents=True, exist_ok=True)
        for old in self.dir.glob("step_*.json"):
            old.unlink()

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cfg = trace_cfg
        self.probe_ids = tokenizer.encode(trace_cfg["probe"])
        self.embedding_words = _single_token_words(tokenizer) if hasattr(model, "wte") else []
        self._prev_coords: torch.Tensor | None = None

        self.manifest = {
            "schema": SCHEMA_VERSION,
            **run,
            "tokenizer": {
                "name": tokenizer.name,
                "tokens": tokenizer.token_strings(),
            },
            "probe": {"text": trace_cfg["probe"], "ids": self.probe_ids},
            "prompts": trace_cfg["prompts"],
            "embedding_words": self.embedding_words,
            "loss": [],
            "checkpoints": [],
        }

    def log_loss(self, step: int, train: float):
        """One point on the training-loss curve (the loss of a single batch)."""
        self.manifest["loss"].append({"step": step, "train": round(train, 4)})

    @torch.no_grad()
    def snapshot(self, step: int, train_loss: float, val_loss: float):
        self.model.eval()
        snap = {
            "step": step,
            "samples": self._samples(),
            "probe": self._probe(),
            "views": {
                k: _round(v, 3 if k == "attention" else 4)
                for k, v in self.model.inspect(self._probe_tensor()).items()
            },
        }
        if self.embedding_words:
            snap["views"]["embedding_2d"] = _round(self._embedding_map(), 3)
        self.model.train()

        name = f"step_{step:07d}.json"
        (self.dir / name).write_text(json.dumps(snap, separators=(",", ":")))
        self.manifest["checkpoints"].append(
            {"step": step, "file": name, "train": round(train_loss, 4), "val": round(val_loss, 4)}
        )
        self.save_manifest()

    def save_manifest(self):
        (self.dir / "manifest.json").write_text(json.dumps(self.manifest, indent=1))

    def _embedding_map(self) -> torch.Tensor:
        """
        Squash each word's d_model-dimensional embedding down to 2D (with PCA)
        so we can plot it. Each snapshot is rotated to line up with the
        previous one, so the map doesn't spin around between frames.
        """
        ids = torch.tensor([w["id"] for w in self.embedding_words])
        X = self.model.wte.weight[ids.to(self.device)].float().cpu()
        X = X - X.mean(0)
        _, _, V = torch.linalg.svd(X, full_matrices=False)
        coords = X @ V[:2].T
        coords = coords / coords.pow(2).sum(1).mean().sqrt()  # fixed overall size
        if self._prev_coords is not None:
            # Procrustes: the rotation/flip that best matches the last frame.
            U, _, Wt = torch.linalg.svd(coords.T @ self._prev_coords)
            coords = coords @ (U @ Wt)
        self._prev_coords = coords
        return coords

    def _probe_tensor(self) -> torch.Tensor:
        return torch.tensor([self.probe_ids], device=self.device)

    def _probe(self) -> dict:
        """
        Feed the probe sentence in and record, at every position, what the
        model expected to come next - its top guesses, and how much
        probability it gave to the token that actually came next.
        """
        logits, _ = self.model(self._probe_tensor())
        probs = F.softmax(logits[0].float(), dim=-1)  # (time, vocab)
        top_p, top_i = probs.topk(self.cfg.get("top_k_shown", 5), dim=-1)

        targets = self.probe_ids[1:]
        target_p = [probs[t, tok].item() for t, tok in enumerate(targets)]
        return {
            "top_ids": top_i.tolist(),
            "top_probs": _round(top_p),
            "target_probs": _round(target_p),
            # Average surprise on this sentence (same units as the loss).
            "loss": round(-sum(math.log(max(p, 1e-9)) for p in target_p) / len(target_p), 4),
        }

    def _samples(self) -> list[dict]:
        out = []
        for i, prompt in enumerate(self.cfg["prompts"]):
            gen = torch.Generator(device=self.device).manual_seed(1000 + i)
            prompt_ids = self.tokenizer.encode(prompt) or [self.tokenizer.eot_id]
            idx = torch.tensor([prompt_ids], device=self.device)
            ids, probs = generate(
                self.model,
                idx,
                self.cfg["sample_tokens"],
                temperature=self.cfg.get("temperature", 1.0),
                generator=gen,
            )
            out.append({"prompt": prompt, "ids": ids, "probs": _round(probs, 3)})
        return out


def _single_token_words(tokenizer) -> list[dict]:
    out = []
    for group, words in EMBEDDING_WORDS.items():
        for w in words.split():
            ids = tokenizer.encode(" " + w)
            if len(ids) == 1:
                out.append({"word": w, "id": ids[0], "group": group})
    return out
