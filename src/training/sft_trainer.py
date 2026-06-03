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
from src.training.trainer import estimate_loss, get_gpu_memory, get_lr


@dataclass
class SFTConfig:
    batch_size: int = 16
    block_size: int = 256
    learning_rate: float = 5e-5
    max_iters: int = 1000
    eval_interval: int = 50
    eval_iters: int = 20
    warmup_iters: int = 100
    lr_decay_iters: int = 1000
    min_lr: float = 5e-6
    grad_clip: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1


def load_pretrained_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict, dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=True)

    model_config = None
    if "model_config" in ckpt:
        model_config = ckpt["model_config"]
    elif "config" in ckpt and isinstance(ckpt["config"], dict):
        possible_keys = ["vocab_size", "block_size", "n_embd", "n_head", "n_layer", "dropout"]
        if all(k in ckpt["config"] for k in possible_keys):
            model_config = ckpt["config"]

    if model_config is None:
        raise ValueError(
            "Could not find model configuration in checkpoint. "
            "Expected key 'model_config' or 'config' with "
            "vocab_size, block_size, n_embd, n_head, n_layer, dropout."
        )

    from src.model.gpt import GPT
    model = GPT(**model_config)
    model.to(device)

    state_dict = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
    if "model_state_dict" not in ckpt and "model" not in ckpt:
        state_dict = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}

    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load state dict: {e}")

    if "model_config" not in ckpt:
        ckpt["model_config"] = model_config

    return model, model_config, ckpt


def train_sft(
    model: nn.Module,
    data_dir: Path,
    run_dir: Path,
    config: SFTConfig,
    model_config: dict,
    pretrained_checkpoint: str,
    pretrained_run_id: str,
    device: torch.device,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    full_config = {
        "training_type": "sft_full",
        "pretrained_run_id": pretrained_run_id,
        "pretrained_checkpoint": pretrained_checkpoint,
        "data_dir": str(data_dir.resolve()),
        "dataset_metadata": str((data_dir / "metadata.json").resolve()),
        **asdict(config),
        "device": str(device),
        "model_config": model_config,
    }

    with open(run_dir / "config.json", "w") as f:
        json.dump(full_config, f, indent=2)

    run_metadata = {
        "training_type": "sft_full",
        "base_model_type": "pretrained_gpt",
        "pretrained_checkpoint": pretrained_checkpoint,
        "dataset_path": str(data_dir.resolve()),
        "dataset_metadata": str((data_dir / "metadata.json").resolve()),
        "output_run_dir": str(run_dir.resolve()),
        "notes": "Full supervised fine-tuning on Alpaca PT-BR. Base checkpoint is not modified.",
    }

    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(run_metadata, f, indent=2)

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

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Training type: SFT Full Fine-Tuning")
    print(f"Parameters: {n_params:,}")
    print(f"Data: {data_dir}")
    print(f"Run dir: {run_dir}")
    print(f"Pretrained: {pretrained_checkpoint}")
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
                     "model_config": model_config,
                     "training_type": "sft_full",
                     "pretrained_checkpoint": pretrained_checkpoint},
                    run_dir / "best.pt",
                )

    torch.save(
        {"model_state_dict": model.state_dict(),
         "optimizer_state_dict": optimizer.state_dict(),
         "step": config.max_iters,
         "val_loss": val_loss if "val_loss" in dir() else None,
         "train_config": asdict(config),
         "model_config": model_config,
         "training_type": "sft_full",
         "pretrained_checkpoint": pretrained_checkpoint},
        run_dir / "last.pt",
    )

    print()
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {run_dir}")
