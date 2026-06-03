from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot training and validation metrics from a run"
    )
    parser.add_argument("run_id", type=str,
                        help="Run ID (subdirectory name under runs/)")
    parser.add_argument("--no-grid", action="store_true",
                        help="Disable grid on plots")
    args = parser.parse_args(argv)

    run_dir = PROJECT_ROOT / "runs" / args.run_id

    train_csv = run_dir / "train_metrics.csv"
    eval_csv = run_dir / "eval_metrics.csv"

    missing = []
    if not train_csv.exists():
        missing.append(str(train_csv))
    if not eval_csv.exists():
        missing.append(str(eval_csv))
    if missing:
        print("Missing files:")
        for f in missing:
            print(f"  {f}")
        return 1

    train_df = pd.read_csv(train_csv)
    eval_df = pd.read_csv(eval_csv)

    grid = not args.no_grid

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_df["step"], train_df["loss"], color="blue")
    ax.set_title(f"Training Loss - {args.run_id}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    if grid:
        ax.grid(True)
    fig.tight_layout()
    loss_path = run_dir / "train_loss.png"
    fig.savefig(loss_path)
    plt.close(fig)
    print(f"Saved: {loss_path}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(eval_df["step"], eval_df["perplexity"], color="orange")
    ax.set_title(f"Validation Perplexity - {args.run_id}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Perplexity")
    if grid:
        ax.grid(True)
    fig.tight_layout()
    perp_path = run_dir / "val_perplexity.png"
    fig.savefig(perp_path)
    plt.close(fig)
    print(f"Saved: {perp_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
