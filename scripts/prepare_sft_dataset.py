from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sentencepiece as spm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.sft_dataset import (
    format_and_tokenize,
    load_alpaca_json,
    normalize_example,
    save_bin,
    save_metadata,
    save_txt,
    train_val_split,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare Alpaca PT-BR dataset for SFT causal LM training"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/sft/alpaca_ptbr/raw/alpaca_data_ptbr.json",
        help="Path to input JSON (default: data/sft/alpaca_ptbr/raw/alpaca_data_ptbr.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sft/alpaca_ptbr/processed",
        help="Output directory (default: data/sft/alpaca_ptbr/processed)",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="artifacts/tokenizer/tokenizer.model",
        help="Path to SentencePiece tokenizer (default: artifacts/tokenizer/tokenizer.model)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to use (default: all)",
    )
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
    print("Tokenizing...")
    train_tokens = format_and_tokenize(train_examples, sp, eos_id)
    val_tokens = format_and_tokenize(val_examples, sp, eos_id)
    print(f"Train tokens: {len(train_tokens)}")
    print(f"Val tokens:   {len(val_tokens)}")

    print("Saving binary files...")
    train_bin = output_dir / "train.bin"
    val_bin = output_dir / "val.bin"
    save_bin(train_tokens, train_bin)
    save_bin(val_tokens, val_bin)
    print(f"  {train_bin}")
    print(f"  {val_bin}")

    metadata = {
        "dataset_name": "dominguesm/alpaca-data-pt-br",
        "source_json": str(input_path.resolve()),
        "tokenizer_path": str(tokenizer_path.resolve()),
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "eos_id": eos_id,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "max_examples": args.max_examples,
        "format": "alpaca_ptbr_instruction_response_v1",
    }

    meta_path = output_dir / "metadata.json"
    save_metadata(metadata, meta_path)
    print(f"  {meta_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
