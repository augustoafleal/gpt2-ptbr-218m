from __future__ import annotations

import json
from array import array
from pathlib import Path

import numpy as np
import sentencepiece as spm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "training" / "dataset_general.txt"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "tokenizer" / "tokenizer.model"
OUTPUT_DIR = PROJECT_ROOT / "data" / "tokenized"

TRAIN_SPLIT = 0.9


def count_articles(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "<eos>":
                count += 1
    return count


def tokenize_dataset() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sp = spm.SentencePieceProcessor()
    sp.load(str(MODEL_PATH))
    eos_id = sp.piece_to_id("<eos>")

    print(f"Tokenizer: {MODEL_PATH}")
    print(f"EOS ID:    {eos_id}")
    print(f"Dataset:   {DATASET_PATH}")
    print()

    print("Counting articles...")
    total_articles = count_articles(DATASET_PATH)
    train_cutoff = int(total_articles * TRAIN_SPLIT)

    print(f"Total articles: {total_articles}")
    print(f"Train articles: {train_cutoff}")
    print(f"Val articles:   {total_articles - train_cutoff}")
    print()

    print("Tokenizing...")
    train_tokens: array = array("H")
    val_tokens: array = array("H")

    article_idx = 0
    current_lines: list[str] = []

    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "<eos>":
                if current_lines:
                    article_text = "\n".join(current_lines)
                    ids = array("H", sp.encode(article_text))
                    ids.append(eos_id)

                    if article_idx < train_cutoff:
                        train_tokens.extend(ids)
                    else:
                        val_tokens.extend(ids)

                    article_idx += 1
                    if article_idx % 50000 == 0:
                        print(f"  processed {article_idx}/{total_articles} articles")
                current_lines = []
            elif stripped:
                current_lines.append(stripped)

    if current_lines:
        article_text = "\n".join(current_lines)
        ids = array("H", sp.encode(article_text))
        ids.append(eos_id)
        if article_idx < train_cutoff:
            train_tokens.extend(ids)
        else:
            val_tokens.extend(ids)

    total_tokens = len(train_tokens) + len(val_tokens)
    print()
    print(f"Total tokens:  {total_tokens:>12}")
    print(f"Train tokens:  {len(train_tokens):>12}")
    print(f"Val tokens:    {len(val_tokens):>12}")
    print()

    print("Saving binary files...")

    train_path = OUTPUT_DIR / "train.bin"
    np.array(train_tokens, dtype=np.uint16).tofile(str(train_path))

    val_path = OUTPUT_DIR / "val.bin"
    np.array(val_tokens, dtype=np.uint16).tofile(str(val_path))

    metadata = {
        "vocab_size": sp.get_piece_size(),
        "dtype": "uint16",
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "eos_id": eos_id,
        "tokenizer_path": str(MODEL_PATH.resolve()),
    }

    meta_path = OUTPUT_DIR / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Train:    {train_path}")
    print(f"Val:      {val_path}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    tokenize_dataset()
