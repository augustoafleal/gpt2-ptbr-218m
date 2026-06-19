from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.gpt import GPT
from src.training.sft_trainer import SFTConfig, load_pretrained_checkpoint, train_sft


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Supervised Fine-Tuning (SFT) for GPT model"
    )

    parser.add_argument("--pretrained-run-id", type=str, required=True,
                        help="Run ID of the pretrained model (e.g. 20260531_232031)")
    parser.add_argument("--checkpoint-name", type=str, default="best.pt",
                        help="Checkpoint file name (default: best.pt)")

    parser.add_argument("--data-dir", type=str, default=None,
                        help="SFT data directory (default: data/sft/alpaca_ptbr/processed, or processed_response_only with --response-only)")
    parser.add_argument("--out-dir", type=str, default="runs",
                        help="Output base directory (default: runs)")

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-iters", type=int, default=20)

    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5)")
    parser.add_argument("--min-lr", type=float, default=5e-6,
                        help="Minimum learning rate (default: 5e-6)")
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--lr-decay-iters", type=int, default=None,
                        help="LR decay iterations (default: max_iters)")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--device", type=str, default="",
                        help="Device (default: auto: cuda if available else cpu)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--response-only", action="store_true",
                        help="Train only on response tokens (ignore instruction/input loss)")

    args = parser.parse_args(argv)

    if args.seed >= 0:
        torch.manual_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = PROJECT_ROOT / "runs" / args.pretrained_run_id / args.checkpoint_name
    if args.data_dir is not None:
        data_dir = PROJECT_ROOT / args.data_dir
    elif args.response_only:
        data_dir = PROJECT_ROOT / "data/sft/alpaca_ptbr/processed_response_only"
    else:
        data_dir = PROJECT_ROOT / "data/sft/alpaca_ptbr/processed"
    out_dir = PROJECT_ROOT / args.out_dir

    if not checkpoint_path.exists():
        print(f"Error: pretrained checkpoint not found at {checkpoint_path}")
        return 1

    if not data_dir.exists():
        print(f"Error: data directory not found at {data_dir}")
        return 1

    train_bin = data_dir / "train.bin"
    val_bin = data_dir / "val.bin"
    meta_path = data_dir / "metadata.json"

    required_files = [(train_bin, "train.bin"), (val_bin, "val.bin"), (meta_path, "metadata.json")]
    if args.response_only:
        required_files.append((data_dir / "train_loss_mask.bin", "train_loss_mask.bin"))
        required_files.append((data_dir / "val_loss_mask.bin", "val_loss_mask.bin"))

    for f, name in required_files:
        if not f.exists():
            print(f"Error: {name} not found in {data_dir}")
            return 1

    with open(meta_path) as f:
        dataset_meta = json.load(f)
    ds_vocab_size = dataset_meta.get("vocab_size")

    print(f"Loading pretrained checkpoint from {checkpoint_path} ...")
    print(f"Device: {device}")
    model, model_config, ckpt = load_pretrained_checkpoint(checkpoint_path, device)

    if ds_vocab_size is not None and model_config.get("vocab_size") != ds_vocab_size:
        print(f"Warning: dataset vocab_size ({ds_vocab_size}) differs from "
              f"model vocab_size ({model_config['vocab_size']})")

    model_block_size = model_config.get("block_size", 256)
    if args.block_size > model_block_size:
        print(f"Error: requested block_size ({args.block_size}) exceeds "
              f"model block_size ({model_block_size})")
        return 1

    for p in model.parameters():
        p.requires_grad = True

    lr_decay_iters = args.lr_decay_iters if args.lr_decay_iters is not None else args.max_iters
    if lr_decay_iters <= args.warmup_iters:
        lr_decay_iters = args.warmup_iters + 1

    training_mode = "response_only" if args.response_only else "full_loss"

    sft_config = SFTConfig(
        batch_size=args.batch_size,
        block_size=args.block_size,
        learning_rate=args.lr,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        warmup_iters=args.warmup_iters,
        lr_decay_iters=lr_decay_iters,
        min_lr=args.min_lr,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        training_mode=training_mode,
    )

    timestamp = datetime.now().strftime("sft_%Y%m%d_%H%M%S")
    run_dir = out_dir / timestamp

    train_sft(
        model=model,
        data_dir=data_dir,
        run_dir=run_dir,
        config=sft_config,
        model_config=model_config,
        pretrained_checkpoint=str(checkpoint_path.resolve()),
        pretrained_run_id=args.pretrained_run_id,
        device=device,
        vocab_size=model_config.get("vocab_size"),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
