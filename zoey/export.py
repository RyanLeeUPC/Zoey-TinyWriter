"""
Package a trained model for the browser.

    uv run python -m zoey.export zoey

Writes viewer/public/models/<name>/:
  model.json   config, tokenizer, and where each tensor lives in weights.bin
  weights.bin  the parameters

Big weight matrices are stored as 8-bit integers (one scale per row) instead
of 32-bit floats - a quarter of the download, for a barely measurable loss in
quality. The script measures that loss, so you don't have to take it on faith.
It also writes a small reference file the browser engine's tests compare
against, to prove the TypeScript forward pass matches PyTorch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from .bpe import SPLIT_PATTERN, BPETokenizer
from .data import build_stream
from .models import build_model
from .train import CHECKPOINT_DIR, load_tokenizer

MODELS_DIR = Path(__file__).resolve().parent.parent / "viewer" / "public" / "models"

# The tokenizer's chunking regex, rewritten for JavaScript so the browser
# splits text *exactly* like Python does. The two languages disagree on what
# counts as a letter, a number, or whitespace, so each class is spelled out:
#   Python [^\W\d_] (letters) = Unicode letters + letter-like numbers (Ⅳ, ², ½)
#   Python \d       (digits)  = decimal digits only
#   Python \s       (spaces)  = str.isspace(), which (unlike JS) includes
#                               \x1c-\x1f and \x85, and excludes U+FEFF
_WS = r"\t-\r\x1c-\x20\x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000"
JS_SPLIT_PATTERN = (
    r"'(?:s|t|re|ve|m|ll|d)"
    r"| ?[\p{L}\p{Nl}\p{No}]+"
    r"| ?\p{Nd}+"
    rf"| ?[^{_WS}\p{{L}}\p{{N}}]+"
    rf"|[{_WS}]+(?![^{_WS}])"
    rf"|[{_WS}]+"
)

# Text that is easy to tokenize differently in Python and JavaScript. The
# browser engine's tests check every one of these splits identically.
TOKENIZER_CASES = [
    "unseen wörds 🐶 and   spaces\n\nOK?",
    "It's 3 o'clock_now!",
    "x² 3½ Ⅳ ① ٣",
    "\ufeffOnce upon a time",
    "a\x1cb\x85c\u3000d\u00a0e",
    "under_score __init__ ...!?",
]


def quantize_rows(w: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """int8 per row: each row is scaled so its largest value maps to 127."""
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return q.numpy(), scale.squeeze(1).float().numpy()


def dequantized_state(state: dict) -> dict:
    """What the browser will actually compute with."""
    out = {}
    for k, v in state.items():
        if v.dim() == 2:
            q, s = quantize_rows(v.float().cpu())
            out[k] = torch.from_numpy(q.astype(np.float32) * s[:, None])
        else:
            out[k] = v.float().cpu()
    return out


@torch.no_grad()
def val_loss(model, stream, device, batches: int = 40) -> float:
    torch.manual_seed(0)
    losses = [model(*stream.batch("val", 32, model.config.block_size, device))[1].item() for _ in range(batches)]
    return sum(losses) / len(losses)


def main(name: str):
    ckpt = torch.load(CHECKPOINT_DIR / f"{name}.pt", map_location="cpu")
    model = build_model(ckpt["model"], **ckpt["config"]).eval()
    model.load_state_dict(ckpt["state"])
    cfg = model.config
    # The tokenizer the model was trained with (checkpoints saved before this
    # was recorded all used bpe-4096).
    tokenizer: BPETokenizer = load_tokenizer(ckpt.get("tokenizer", "bpe-4096"))
    assert tokenizer.vocab_size == cfg.vocab_size

    out_dir = MODELS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # lm_head shares its weights with wte, so it's skipped.
    state = {k: v for k, v in model.state_dict().items() if k != "lm_head.weight"}
    tensors, chunks, offset = [], [], 0

    def add(arr: np.ndarray):
        nonlocal offset
        data = arr.tobytes()
        pad = (-len(data)) % 4  # keep every tensor 4-byte aligned for Float32Array
        chunks.append(data + b"\0" * pad)
        start = offset
        offset += len(data) + pad
        return start

    for key, v in state.items():
        v = v.float().cpu()
        entry = {"name": key, "shape": list(v.shape)}
        if v.dim() == 2:
            q, s = quantize_rows(v)
            entry.update(dtype="int8", offset=add(q), scales=add(s))
        else:
            entry.update(dtype="float32", offset=add(v.numpy().astype(np.float32)))
        tensors.append(entry)

    (out_dir / "weights.bin").write_bytes(b"".join(chunks))
    meta = {
        "name": name,
        "config": {k: v for k, v in cfg.__dict__.items() if k != "dropout"},
        "params": sum(p.numel() for p in model.parameters()),
        "tokenizer": {
            "name": tokenizer.name,
            "pattern": JS_SPLIT_PATTERN,
            "python_pattern": SPLIT_PATTERN,
            "merges": tokenizer.merges,
            "eot_id": tokenizer.eot_id,
        },
        "tensors": tensors,
    }
    (out_dir / "model.json").write_text(json.dumps(meta))
    print(f"Wrote {out_dir} ({offset / 1e6:.1f} MB of weights)")

    # How much does 8-bit storage cost? Compare validation loss.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stream = build_stream(tokenizer, "full")
    model.to(device)
    full = val_loss(model, stream, device)
    model.load_state_dict({k: v.to(device) for k, v in dequantized_state(model.state_dict()).items()})
    quant = val_loss(model, stream, device)
    print(f"Validation loss: {full:.4f} full precision, {quant:.4f} 8-bit ({quant - full:+.4f})")

    export_dictionary(model, stream, out_dir)

    # Reference outputs for the browser engine's parity test, computed with
    # the 8-bit weights the browser uses.
    model.cpu()
    prompt = "Once upon a time, there was a little girl named Lily. She"
    ids = tokenizer.encode(prompt)
    idx = torch.tensor([ids])
    logits, _ = model(idx)
    views = model.inspect(idx)
    ref = {
        "prompt": prompt,
        "ids": ids,
        "last_logits": logits[0, -1].tolist(),
        "last_attention": views["attention"][:, :, -1, :].tolist(),  # (layers, heads, T)
        "last_lens_top_ids": views["lens_top_ids"][:, -1].tolist(),
        "tokenizer_cases": {s: tokenizer.encode(s) for s in [prompt, *TOKENIZER_CASES]},
    }
    ref_dir = Path(__file__).resolve().parent.parent / "viewer" / "test"
    ref_dir.mkdir(exist_ok=True)
    (ref_dir / f"{name}.reference.json").write_text(json.dumps(ref))
    print("Wrote parity reference for the browser engine tests")


def export_dictionary(model, stream, out_dir: Path):
    """
    Data for the viewer's Dictionary page:
      xy:    a 2D map of every token, laid out with t-SNE so tokens with similar
             embeddings land near each other (similarity itself is computed live
             in the browser from the weights)
      count: how often each token appears in the training stories
    """
    from sklearn.manifold import TSNE

    emb = model.wte.weight.detach().float().cpu().numpy()
    xy = TSNE(n_components=2, metric="cosine", perplexity=30, init="pca", random_state=0).fit_transform(emb)
    xy = (xy - xy.mean(0)) / np.abs(xy - xy.mean(0)).max()  # centered, within [-1, 1]
    count = np.bincount(np.asarray(stream.train), minlength=emb.shape[0])
    data = {"xy": np.round(xy, 4).flatten().tolist(), "count": count.tolist()}
    (out_dir / "dictionary.json").write_text(json.dumps(data, separators=(",", ":")))
    print(f"Wrote the dictionary map ({len(count)} tokens)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m zoey.export <checkpoint name>")
    main(sys.argv[1])
