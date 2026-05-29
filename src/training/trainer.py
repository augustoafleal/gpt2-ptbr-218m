from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn

from src.data.dataloader import get_batch


@dataclass
class TrainConfig:
    batch_size: int = 32
    block_size: int = 256
    learning_rate: float = 3e-4
    max_iters: int = 10000
    eval_interval: int = 500
    eval_iters: int = 100
    warmup_iters: int = 100
    lr_decay_iters: int = 10000
    min_lr: float = 3e-5
    grad_clip: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1


def get_gpu_memory(device: torch.device) -> str:
    if device.type != "cuda":
        return ""
    gb = torch.cuda.memory_allocated(device) / 1e9
    return f"gpu {gb:.1f} GB"


def get_lr(it: int, config: TrainConfig) -> float:
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    data_dir: Path,
    split: str,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(split, data_dir, batch_size, block_size, device)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def train(
    model: nn.Module,
    data_dir: Path,
    run_dir: Path,
    config: TrainConfig,
    model_config: dict,
    device: torch.device,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    optimizer = model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=config.betas,
        device=device,
    )

    train_metrics_file = run_dir / "train_metrics.csv"
    eval_metrics_file = run_dir / "eval_metrics.csv"

    gpu_col = device.type == "cuda"
    with open(train_metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["step", "loss", "tokens_per_sec", "lr"]
        if gpu_col:
            header.append("gpu_mem_gb")
        writer.writerow(header)

    with open(eval_metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "val_loss", "perplexity"])

    best_val_loss = float("inf")

    print(f"Training on {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Run dir: {run_dir}")
    print()

    model.train()
    tokens_processed = 0
    tick = time.time()

    for step in range(1, config.max_iters + 1):
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        X, Y = get_batch("train", data_dir, config.batch_size, config.block_size, device)
        _, loss = model(X, Y)
        optimizer.zero_grad()
        loss.backward()

        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()

        tokens_processed += config.batch_size * config.block_size
        tock = time.time()
        tokens_per_sec = tokens_processed / (tock - tick)

        row = [step, f"{loss.item():.6f}", f"{tokens_per_sec:.0f}", f"{lr:.2e}"]
        gpu_mem = ""
        if gpu_col:
            gpu_mem = get_gpu_memory(device)
            row.append(gpu_mem)

        with open(train_metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        if step % 50 == 0 or step == 1:
            gpu_str = f" | {gpu_mem}" if gpu_mem else ""
            print(f"step {step:>6d} | loss {loss.item():.4f} | "
                  f"tok/s {tokens_per_sec:.0f} | lr {lr:.2e}{gpu_str}")

        if step % config.eval_interval == 0 or step == 1:
            val_loss = estimate_loss(
                model, data_dir, "val", config.batch_size,
                config.block_size, config.eval_iters, device,
            )
            perplexity = math.exp(val_loss)
            print(f"  └─ val loss {val_loss:.4f} | perplexity {perplexity:.2f}")

            with open(eval_metrics_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([step, f"{val_loss:.6f}", f"{perplexity:.6f}"])

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(),
                     "step": step,
                     "val_loss": val_loss,
                     "train_config": asdict(config),
                     "model_config": model_config},
                    run_dir / "best.pt",
                )

    torch.save(
        {"model_state_dict": model.state_dict(),
         "optimizer_state_dict": optimizer.state_dict(),
         "step": config.max_iters,
         "val_loss": val_loss if "val_loss" in dir() else None,
         "train_config": asdict(config),
         "model_config": model_config},
        run_dir / "last.pt",
    )

    print()
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {run_dir}")
