from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.sft_dataset import (
    format_example,
    load_alpaca_json,
    normalize_example,
    save_bin,
    save_metadata,
    save_txt,
    train_val_split,
)


def format_prefix(example: dict) -> str:
    parts = ["### Instrução:", example["instruction"]]
    if example["input"]:
        parts.append("")
        parts.append("### Entrada:")
        parts.append(example["input"])
    parts.append("")
    parts.append("### Resposta:")
    return "\n".join(parts)


def format_and_tokenize_with_mask(
    examples: list[dict],
    sp: spm.SentencePieceProcessor,
    eos_id: int,
) -> tuple[list[int], list[int]]:
    all_tokens: list[int] = []
    all_masks: list[int] = []
    total_response_tokens = 0

    for ex in examples:
        text = format_example(ex)
        prefix_text = format_prefix(ex)

        ids = sp.encode(text)
        prefix_ids = sp.encode(prefix_text)

        response_start = len(prefix_ids)
        n_response = len(ids) - response_start

        mask = [0] * response_start + [1] * n_response

        all_tokens.extend(ids)
        all_tokens.append(eos_id)

        all_masks.extend(mask)
        all_masks.append(1)

        total_response_tokens += n_response + 1

    return all_tokens, all_masks, total_response_tokens


def save_mask(mask: list[int], path: Path) -> None:
    np.array(mask, dtype=np.uint8).tofile(str(path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare Response-Only SFT dataset with loss masks"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/sft/alpaca_ptbr/raw/alpaca_data_ptbr.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sft/alpaca_ptbr/processed_response_only",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="artifacts/tokenizer/tokenizer.model",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args(argv)

    input_path = PROJECT_ROOT / args.input_path
    output_dir = PROJECT_ROOT / args.output_dir
    tokenizer_path = PROJECT_ROOT / args.tokenizer_path

    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return 1
    if not tokenizer_path.exists():
        print(f"Tokenizer not found: {tokenizer_path}")
        return 1

    print(f"Loading dataset from: {input_path}")
    raw = load_alpaca_json(input_path)
    print(f"Loaded {len(raw)} raw examples")

    print("Normalizing examples...")
    examples = []
    for ex in raw:
        normalized = normalize_example(ex)
        if normalized is not None:
            examples.append(normalized)
    print(f"Valid examples after normalization: {len(examples)}")

    if args.max_examples is not None and args.max_examples < len(examples):
        print(f"Limiting to max_examples={args.max_examples}")
        rng = __import__("random").Random(args.seed)
        examples = rng.sample(examples, args.max_examples)

    print(f"Splitting train/val (val_ratio={args.val_ratio}, seed={args.seed})...")
    train_examples, val_examples = train_val_split(examples, args.val_ratio, args.seed)
    print(f"Train examples: {len(train_examples)}")
    print(f"Val examples:   {len(val_examples)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Saving text files...")
    train_txt = output_dir / "sft_train.txt"
    val_txt = output_dir / "sft_val.txt"
    save_txt(train_examples, train_txt)
    save_txt(val_examples, val_txt)
    print(f"  {train_txt}")
    print(f"  {val_txt}")

    print("Loading tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load(str(tokenizer_path))
    eos_id = sp.piece_to_id("<eos>")
    vocab_size = sp.get_piece_size()

    if vocab_size > 65535:
        print(f"Error: vocab_size {vocab_size} exceeds uint16 limit (65535)")
        return 1

    print(f"vocab_size={vocab_size}, eos_id={eos_id}")
    print("Tokenizing with loss masks...")

    train_tokens, train_masks, train_response_tokens = format_and_tokenize_with_mask(
        train_examples, sp, eos_id,
    )
    val_tokens, val_masks, val_response_tokens = format_and_tokenize_with_mask(
        val_examples, sp, eos_id,
    )

    train_loss_tokens = sum(train_masks)
    val_loss_tokens = sum(val_masks)

    print(f"Train tokens: {len(train_tokens)} (loss: {train_loss_tokens}, response: {train_response_tokens})")
    print(f"Val tokens:   {len(val_tokens)} (loss: {val_loss_tokens}, response: {val_response_tokens})")

    loss_pct_train = 100.0 * train_loss_tokens / len(train_tokens) if train_tokens else 0
    loss_pct_val = 100.0 * val_loss_tokens / len(val_tokens) if val_tokens else 0
    print(f"Loss participation: train {loss_pct_train:.1f}%, val {loss_pct_val:.1f}%")

    print("Saving binary files...")
    train_bin = output_dir / "train.bin"
    val_bin = output_dir / "val.bin"
    save_bin(train_tokens, train_bin)
    save_bin(val_tokens, val_bin)
    print(f"  {train_bin}")
    print(f"  {val_bin}")

    print("Saving loss mask files...")
    train_mask_path = output_dir / "train_loss_mask.bin"
    val_mask_path = output_dir / "val_loss_mask.bin"
    save_mask(train_masks, train_mask_path)
    save_mask(val_masks, val_mask_path)
    print(f"  {train_mask_path}")
    print(f"  {val_mask_path}")

    metadata = {
        "dataset_name": "dominguesm/alpaca-data-pt-br",
        "source_json": str(input_path.resolve()),
        "tokenizer_path": str(tokenizer_path.resolve()),
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "mask_dtype": "uint8",
        "eos_id": eos_id,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "loss_tokens_train": train_loss_tokens,
        "loss_tokens_val": val_loss_tokens,
        "response_tokens_train": train_response_tokens,
        "response_tokens_val": val_response_tokens,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "max_examples": args.max_examples,
        "format": "alpaca_ptbr_instruction_response_v1",
        "training_format": "response_only",
    }

    meta_path = output_dir / "metadata.json"
    save_metadata(metadata, meta_path)
    print(f"  {meta_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
