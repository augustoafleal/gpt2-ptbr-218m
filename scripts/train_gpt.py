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
from src.training.trainer import TrainConfig, train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Mini GPT-like model")

    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to train on (default: cpu)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--block-size", type=int, default=256,
                        help="Block/context size (default: 256)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--max-iters", type=int, default=10000,
                        help="Maximum training iterations (default: 10000)")
    parser.add_argument("--eval-interval", type=int, default=500,
                        help="Evaluation interval (default: 500)")
    parser.add_argument("--eval-iters", type=int, default=100,
                        help="Evaluation iterations (default: 100)")
    parser.add_argument("--n-embd", type=int, default=384,
                        help="Embedding dimension (default: 384)")
    parser.add_argument("--n-head", type=int, default=6,
                        help="Number of attention heads (default: 6)")
    parser.add_argument("--n-layer", type=int, default=6,
                        help="Number of transformer layers (default: 6)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate (default: 0.1)")

    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data_dir = PROJECT_ROOT / "data" / "tokenized"
    meta_path = data_dir / "metadata.json"

    if not meta_path.exists():
        print(f"Metadata not found at {meta_path}")
        return 1

    with open(meta_path) as f:
        meta = json.load(f)

    model_config = dict(
        vocab_size=meta["vocab_size"],
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    )

    model = GPT(**model_config)
    model.to(device)

    train_config = TrainConfig(
        batch_size=args.batch_size,
        block_size=args.block_size,
        learning_rate=args.lr,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
    )

    run_dir = PROJECT_ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")

    train(model, data_dir, run_dir, train_config, model_config, device)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
