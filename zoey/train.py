"""
The training loop - the same for every model in Zoey-TinyWriter.

    python -m zoey.train configs/zoey.toml

Training is a loop of four steps, repeated thousands of times:

  1. grab a random batch of text
  2. ask the model to predict every next token  (forward pass)
  3. measure how wrong it was                    (the loss)
  4. nudge every parameter to be a bit less wrong (backward pass + optimizer)

Along the way, the TraceWriter takes snapshots so the viewer can replay
the whole thing.
"""

from __future__ import annotations

import math
import sys
import time
import tomllib
from contextlib import nullcontext
from pathlib import Path

import torch

from .bpe import BPETokenizer
from .data import build_stream
from .models import build_model
from .trace import TraceWriter, checkpoint_schedule

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def load_tokenizer(name: str):
    return BPETokenizer.load(name)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def learning_rate(step: int, cfg: dict) -> float:
    """
    Warm up linearly, then decay along a cosine curve.

    Big steps early (once warmed up) make fast progress; small steps late let
    the model settle into a good spot instead of bouncing around it.
    """
    lr, warmup, max_steps = cfg["learning_rate"], cfg.get("warmup_steps", 0), cfg["max_steps"]
    min_lr = lr * cfg.get("min_lr_ratio", 1.0)
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def make_optimizer(model, cfg: dict, device: str):
    """
    AdamW, with weight decay only on the big weight matrices. Weight decay
    gently pulls weights towards zero, which helps generalization; it isn't
    useful on biases and LayerNorm scales, so those are left alone.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": cfg.get("weight_decay", 0.0)},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=cfg["learning_rate"],
        betas=tuple(cfg.get("betas", (0.9, 0.95))),
        fused=device == "cuda",
    )


@torch.no_grad()
def estimate_loss(model, stream, cfg: dict, device: str, autocast) -> dict:
    """Average the loss over several batches - a single batch is too noisy."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(cfg["eval_iters"]):
            x, y = stream.batch(split, cfg["batch_size"], model.config.block_size, device)
            with autocast:
                losses.append(model(x, y)[1].item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def save_checkpoint(path: Path, model, model_name: str, step: int, optimizer=None):
    path.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "model": model_name,
            "config": model.config.__dict__,
            "state": model.state_dict(),
            "step": step,
            "optimizer": optimizer.state_dict() if optimizer else None,
        },
        path,
    )


def main(config_path: str):
    cfg = tomllib.loads(Path(config_path).read_text())
    run, data_cfg, model_cfg, train_cfg = cfg["run"], cfg["data"], cfg["model"], cfg["train"]

    torch.manual_seed(run.get("seed", 1337))
    device = pick_device()
    print(f"Device: {device}")
    # Let the GPU use faster (slightly less precise) maths where it's safe.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Mixed precision: do the heavy matrix maths in bfloat16 (half the bits,
    # roughly twice the speed) while keeping the weights in full precision.
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()

    tokenizer = load_tokenizer(data_cfg["tokenizer"])
    stream = build_stream(tokenizer, data_cfg["size"])
    print(f"Tokens: {len(stream.train):,} train / {len(stream.val):,} val")

    model_name = model_cfg.pop("name")
    model = build_model(model_name, vocab_size=tokenizer.vocab_size, **model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name}, {n_params:,} parameters")
    if "init_from" in train_cfg:
        # Fine-tuning: start from an already-trained model instead of random weights.
        ckpt = torch.load(CHECKPOINT_DIR / f"{train_cfg['init_from']}.pt", map_location=device)
        model.load_state_dict(ckpt["state"])
        print(f"Starting from checkpoints/{train_cfg['init_from']}.pt")

    optimizer = make_optimizer(model, train_cfg, device)
    tokens_per_step = train_cfg["batch_size"] * model.config.block_size
    max_steps = train_cfg["max_steps"]

    trace = TraceWriter(
        run={
            **run,
            "model": {"name": model_name, "config": model.config.__dict__, "params": n_params},
            "train": {**train_cfg, "tokens_per_step": tokens_per_step},
        },
        model=model,
        tokenizer=tokenizer,
        trace_cfg=cfg["trace"],
        device=device,
    )
    snapshot_steps = set(checkpoint_schedule(max_steps, cfg["trace"]["checkpoints"]))
    save_every = train_cfg.get("save_every", 0)
    ckpt_path = CHECKPOINT_DIR / f"{run['id']}.pt"

    started = time.time()
    last_print, last_step = started, 0
    for step in range(max_steps + 1):
        if step in snapshot_steps:
            losses = estimate_loss(model, stream, train_cfg, device, autocast)
            trace.snapshot(step, losses["train"], losses["val"])
            print(f"step {step:6d} | train {losses['train']:.4f} | val {losses['val']:.4f} | snapshot")
        if step == max_steps:
            break

        lr = learning_rate(step, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = stream.batch("train", train_cfg["batch_size"], model.config.block_size, device)
        with autocast:
            _, loss = model(x, y)  # 2 + 3: predict, and measure how wrong
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # 4: work out which way to nudge each parameter...
        if "grad_clip" in train_cfg:
            # Cap the size of any one update, so a single weird batch can't
            # knock the model off course.
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
        optimizer.step()  # ...and nudge

        # The training-loss curve for the viewer. Every step early on (when
        # things change fast), then every `log_interval` steps.
        if step < 100 or step % train_cfg.get("log_interval", 10) == 0:
            trace.log_loss(step, loss.item())

        if time.time() - last_print > 30:
            now = time.time()
            tok_s = (step - last_step) * tokens_per_step / (now - last_print)
            eta = (max_steps - step) * (now - last_print) / max(1, step - last_step)
            print(f"step {step:6d} | loss {loss.item():.4f} | lr {lr:.2e} | {tok_s / 1e3:,.0f}K tok/s | eta {eta / 60:.0f} min")
            last_print, last_step = now, step

        if save_every and step and step % save_every == 0:
            save_checkpoint(ckpt_path, model, model_name, step, optimizer)

    trace.manifest["train"]["seconds"] = round(time.time() - started, 1)
    trace.manifest["train"]["device"] = torch.cuda.get_device_name() if device == "cuda" else device
    trace.save_manifest()
    save_checkpoint(ckpt_path, model, model_name, max_steps)
    print(f"Done in {(time.time() - started) / 60:.1f} min. Trace written to {trace.dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m zoey.train configs/<config>.toml")
    main(sys.argv[1])
