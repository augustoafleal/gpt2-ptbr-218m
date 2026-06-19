from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"

SHORT_NAMES = {
    "20260531_232031": "Pretrain",
    "20260603_224519": "Pretrain+",
    "sft_20260607_230617": "SFT Alpaca",
    "sft_20260610_172000": "SFT Canarim",
}


def _save(run_dir: Path, stem: str) -> list[Path]:
    files = []
    for ext in [".png", ".pdf"]:
        path = run_dir / f"paper_{stem}{ext}"
        plt.savefig(str(path))
        files.append(path)
    return files


def _annotate_best(ax, steps, values, fmt=".4f"):
    idx = values.idxmin()
    best_step = steps[idx]
    best_val = values[idx]
    ax.scatter(best_step, best_val, color="red", s=40, zorder=5, label=f"Best: {best_val:{fmt}}  |  Step: {best_step}")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)


def plot_training_loss(run_id: str) -> list[Path]:
    run_dir = RUNS_DIR / run_id
    csv_path = run_dir / "train_metrics.csv"
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["step"], df["loss"], color="blue", linewidth=0.8)
    _annotate_best(ax, df["step"], df["loss"])
    ax.set_title("Training Loss During Pre-training")
    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = _save(run_dir, "train_loss")
    plt.close(fig)
    return out


def plot_validation_loss(run_id: str) -> list[Path]:
    run_dir = RUNS_DIR / run_id
    csv_path = run_dir / "eval_metrics.csv"
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["step"], df["val_loss"], color="blue", linewidth=0.8)
    _annotate_best(ax, df["step"], df["val_loss"])
    ax.set_title("Validation Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = _save(run_dir, "val_loss")
    plt.close(fig)
    return out


def plot_validation_perplexity(run_id: str) -> list[Path]:
    run_dir = RUNS_DIR / run_id
    csv_path = run_dir / "eval_metrics.csv"
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["step"], df["perplexity"], color="blue", linewidth=0.8)
    _annotate_best(ax, df["step"], df["perplexity"])
    ax.set_title("Validation Perplexity")
    ax.set_xlabel("Step")
    ax.set_ylabel("Perplexity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = _save(run_dir, "val_perplexity")
    plt.close(fig)
    return out


def plot_perplexity_comparison(run_ids: list[str]) -> list[Path]:
    ORDER = ["20260531_232031", "20260603_224519", "sft_20260607_230617", "sft_20260610_172000"]
    labels = []
    perps = []
    for rid in ORDER:
        if rid not in run_ids:
            continue
        short = SHORT_NAMES.get(rid, rid)
        csv_path = RUNS_DIR / rid / "eval_metrics.csv"
        df = pd.read_csv(csv_path)
        best_perp = df["perplexity"].min()
        labels.append(short)
        perps.append(best_perp)

    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, perps, color=colors[: len(labels)], width=0.55, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, perps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title("Best Validation Perplexity")
    ax.set_ylabel("Perplexity")
    ax.grid(True, axis="y", alpha=0.3)
    ymax = max(perps) * 1.15 if perps else 1
    ax.set_ylim(0, ymax)
    fig.tight_layout()

    comparison_dir = RUNS_DIR
    files = []
    for ext in [".png", ".pdf"]:
        path = comparison_dir / f"paper_perplexity_comparison{ext}"
        plt.savefig(str(path))
        files.append(path)
    plt.close(fig)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures for paper section 4.1"
    )
    parser.add_argument(
        "run_ids",
        type=str,
        nargs="+",
        help="One or more run IDs (order-independent)",
    )
    args = parser.parse_args(argv)

    seen: set[str] = set()
    generated: list[Path] = []

    for rid in args.run_ids:
        run_dir = RUNS_DIR / rid
        if not run_dir.is_dir():
            print(f"Warning: run directory not found, skipping: {run_dir}")
            continue
        if rid in seen:
            continue
        seen.add(rid)

        has_train = (run_dir / "train_metrics.csv").exists()
        has_eval = (run_dir / "eval_metrics.csv").exists()

        print(f"\n--- {rid} ---")

        if has_train:
            print("  Training loss ... ", end="", flush=True)
            out = plot_training_loss(rid)
            for p in out:
                print(f"\n    {p}")
                generated.append(p)
        else:
            print("  Training loss ... SKIPPED (no train_metrics.csv)")

        if has_eval:
            print("  Validation loss ... ", end="", flush=True)
            out = plot_validation_loss(rid)
            for p in out:
                print(f"\n    {p}")
                generated.append(p)

            print("  Validation perplexity ... ", end="", flush=True)
            out = plot_validation_perplexity(rid)
            for p in out:
                print(f"\n    {p}")
                generated.append(p)
        else:
            print("  Validation metrics ... SKIPPED (no eval_metrics.csv)")

    if len(seen) >= 2:
        print("\n--- Comparison ---")
        print("  Perplexity comparison bar chart ... ", end="", flush=True)
        out = plot_perplexity_comparison(list(seen))
        for p in out:
            print(f"\n    {p}")
            generated.append(p)

    print("\n=== Generated files ===")
    for p in generated:
        print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
