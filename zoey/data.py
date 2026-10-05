"""
Getting text in, getting batches out.

An LLM only ever learns one thing: given some tokens, guess the next one.
So "data" is just one very long list of token ids, and a training example is
any window of it:

    tokens:  [ O, n, c, e, _, u, p, o, n ]
    input:   [ O, n, c, e, _, u, p, o ]      <- what the model sees
    target:  [ n, c, e, _, u, p, o, n ]      <- what it should have guessed

Every position is its own little prediction problem, so one window of 512
tokens is really 512 training examples at once.

Dataset: TinyStories (Eldan & Li, 2023) - short stories written in the
vocabulary of a 3-4 year old. It's small enough to learn from on one GPU and
simple enough that a tiny model can produce something that reads like English.
"""

from __future__ import annotations

import os
import urllib.request
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

HF = "https://huggingface.co/datasets/roneneldan"
TINYSTORIES_FILES = {
    # ~22 MB. Plenty for the bigram chapter.
    "small": f"{HF}/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
    # ~2.2 GB, ~2.7M stories. What the real TinyWriter learns from.
    "full": f"{HF}/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    # Stories with instructions ("Words: ... Features: ... Story: ...").
    "instruct-small": f"{HF}/TinyStoriesInstruct/resolve/main/TinyStories-Instruct-valid.txt",
    "instruct-full": f"{HF}/TinyStoriesInstruct/resolve/main/TinyStories-Instruct-train.txt",
}

# TinyStories marks the boundary between stories with this string.
STORY_SEPARATOR = "<|endoftext|>"


def download(url: str) -> Path:
    """Download a file once and cache it under data/."""
    path = DATA_DIR / url.rsplit("/", 1)[-1]
    if path.exists():
        return path
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {path.name} ...")
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp, reporthook=_progress)
    tmp.rename(path)
    print()
    return path


def _progress(blocks: int, block_size: int, total: int) -> None:
    if total > 0 and blocks % 256 == 0:
        done = min(blocks * block_size, total)
        print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)


class TokenStream:
    """
    A dataset that has been turned into one long array of token ids, with a
    training part and a validation part.

    The validation part is text the model never trains on. If training loss
    keeps falling but validation loss doesn't, the model is memorizing
    instead of learning - that's called overfitting.

    The arrays are memory-mapped: they stay on disk and the OS pages in
    whatever we touch, so a 1 GB dataset doesn't need 1 GB of RAM.
    """

    def __init__(self, train: np.ndarray, val: np.ndarray):
        self.train = train
        self.val = val

    def batch(self, split: str, batch_size: int, block_size: int, device: str):
        """Grab `batch_size` random windows of length `block_size`."""
        data = self.train if split == "train" else self.val
        starts = torch.randint(len(data) - block_size - 1, (batch_size,)).tolist()
        xy = torch.from_numpy(np.stack([data[s : s + block_size + 1] for s in starts]).astype(np.int64))
        x, y = xy[:, :-1], xy[:, 1:]
        if device == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)


def build_stream(tokenizer, size: str = "small") -> TokenStream:
    """
    Encode the dataset once (cached on disk), and return it ready for batching.

    "small": the 22 MB file, split 90% train / 10% validation.
    "full":  the 2.2 GB train file for training, the 22 MB file for validation.
    "instruct-full": the instruction-formatted stories, likewise.
    """
    if size == "small":
        ids = encode_file(tokenizer, "small")
        split = int(len(ids) * 0.9)
        return TokenStream(ids[:split], ids[split:])
    val_key = size.replace("full", "small")
    return TokenStream(encode_file(tokenizer, size), encode_file(tokenizer, val_key))


def encode_file(tokenizer, key: str) -> np.ndarray:
    cache = DATA_DIR / f"{Path(TINYSTORIES_FILES[key]).stem}.{tokenizer.name}.bin"
    if not cache.exists():
        src = download(TINYSTORIES_FILES[key])
        print(f"Encoding {src.name} with the {tokenizer.name} tokenizer ...")
        tmp = cache.with_suffix(".part")
        n = 0
        # Encode blocks of stories in parallel on every CPU core.
        with open(tmp, "wb") as out, Pool(os.cpu_count(), _init_worker, (tokenizer,)) as pool:
            for arr in pool.imap(_encode_block, _read_blocks(src)):
                out.write(arr.tobytes())
                n += len(arr)
                print(f"\r  {n / 1e6:,.1f}M tokens", end="", flush=True)
        print()
        tmp.rename(cache)
    return np.memmap(cache, dtype=np.uint16, mode="r")


def _read_blocks(path: Path, block_chars: int = 4_000_000):
    """Yield the file in big pieces, always cut on a story boundary."""
    with open(path, encoding="utf-8") as f:
        leftover = ""
        while True:
            chunk = f.read(block_chars)
            if not chunk:
                break
            text = leftover + chunk
            cut = text.rfind(STORY_SEPARATOR)
            if cut == -1:
                leftover = text
                continue
            yield text[:cut]
            leftover = text[cut + len(STORY_SEPARATOR) :]
        if leftover.strip():
            yield leftover


_worker_tokenizer = None


def _init_worker(tokenizer):
    global _worker_tokenizer
    _worker_tokenizer = tokenizer


def _encode_block(text: str) -> np.ndarray:
    tok = _worker_tokenizer
    ids: list[int] = []
    for story in text.split(STORY_SEPARATOR):
        story = story.strip()
        if story:
            ids.extend(tok.encode(story))
            ids.append(tok.eot_id)
    return np.array(ids, dtype=np.uint16)
