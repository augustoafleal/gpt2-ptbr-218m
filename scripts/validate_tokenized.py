from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sentencepiece as spm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "tokenizer" / "tokenizer.model"
TOKENIZED_DIR = PROJECT_ROOT / "data" / "tokenized"
TRAIN_PATH = TOKENIZED_DIR / "train.bin"
VAL_PATH = TOKENIZED_DIR / "val.bin"
META_PATH = TOKENIZED_DIR / "metadata.json"
SAMPLES = 3


def validate() -> None:
    sp = spm.SentencePieceProcessor()
    sp.load(str(MODEL_PATH))
    eos_id = sp.piece_to_id("<eos>")

    print(f"Tokenizer: {MODEL_PATH}")
    print(f"Vocab size: {sp.get_piece_size()}")
    print(f"EOS ID:     {eos_id}")
    print()

    print("=" * 60)
    print("1. Metadata")
    print("=" * 60)

    with open(META_PATH) as f:
        meta = json.load(f)

    required_keys = {"vocab_size", "dtype", "train_tokens", "val_tokens", "eos_id", "tokenizer_path"}
    present = set(meta.keys())
    assert required_keys.issubset(present), f"Missing keys: {required_keys - present}"
    print(f"Keys: {len(present)}/{len(required_keys)} present")
    assert meta["vocab_size"] == sp.get_piece_size(), "vocab_size mismatch"
    assert meta["dtype"] == "uint16", f"Expected uint16, got {meta['dtype']}"
    assert meta["eos_id"] == eos_id, f"eos_id mismatch: {meta['eos_id']} vs {eos_id}"

    for k, v in meta.items():
        print(f"{k}: {v}")
    print()

    print("=" * 60)
    print("2. Binary integrity")
    print("=" * 60)

    train = np.fromfile(str(TRAIN_PATH), dtype=np.uint16)
    val = np.fromfile(str(VAL_PATH), dtype=np.uint16)

    print(f"train shape: {train.shape}")
    print(f"val shape:   {val.shape}")
    assert train.dtype == np.uint16, f"train dtype: {train.dtype}"
    assert val.dtype == np.uint16, f"val dtype: {val.dtype}"
    assert len(train) == meta["train_tokens"], f"train_tokens mismatch: {len(train)} vs {meta['train_tokens']}"
    assert len(val) == meta["val_tokens"], f"val_tokens mismatch: {len(val)} vs {meta['val_tokens']}"
    print("Metadata counts match binary arrays.")
    print()

    print("=" * 60)
    print("3. Token range")
    print("=" * 60)

    train_max = int(train.max())
    val_max = int(val.max())
    print(f"train max id: {train_max} (vocab_size={sp.get_piece_size()})")
    print(f"val max id:   {val_max}")
    assert train_max < sp.get_piece_size(), f"train id {train_max} >= vocab_size {sp.get_piece_size()}"
    assert val_max < sp.get_piece_size(), f"val id {val_max} >= vocab_size {sp.get_piece_size()}"
    assert train_max < 65535, "train id overflow uint16"
    assert val_max < 65535, "val id overflow uint16"
    print("All IDs valid.")
    print()

    print("=" * 60)
    print("4. EOS markers")
    print("=" * 60)

    train_eos = np.where(train == eos_id)[0]
    val_eos = np.where(val == eos_id)[0]
    print(f"train EOS count: {len(train_eos)}")
    print(f"val EOS count:   {len(val_eos)}")
    assert len(train_eos) > 0, "No EOS found in train"
    assert len(val_eos) > 0, "No EOS found in val"
    assert train[-1] == eos_id, "train does not end with EOS"
    assert val[-1] == eos_id, "val does not end with EOS"
    print("Last token is EOS in both splits.")
    print()

    print("=" * 60)
    print("5. Roundtrip (encode -> decode)")
    print("=" * 60)

    for split_name, arr in [("train", train), ("val", val)]:
        for i in range(SAMPLES):
            offset = i * 1000 if i * 1000 < len(arr) else 0
            chunk = arr[offset:offset + 50].tolist()
            decoded = sp.decode(chunk)
            re_encoded = sp.encode(decoded)
            match = re_encoded == chunk
            status = "OK" if match else "MISMATCH"
            print(f"[{split_name}] offset={offset}: {status} -> {decoded[:60]}...")
    print()

    print("=" * 60)
    print("6. Article decode (first and last)")
    print("=" * 60)

    for split_name, arr, eos_pos in [("train", train, train_eos), ("val", val, val_eos)]:
        first_end = eos_pos[0]
        first_article = sp.decode(arr[:first_end].tolist())
        print(f"[{split_name}] first article:")
        print(f"{first_article[:120]}...")
        print()

        last_start = eos_pos[-1] + 1
        if last_start < len(arr):
            last_article = sp.decode(arr[last_start:].tolist())
        else:
            last_article = sp.decode(arr[eos_pos[-2] + 1:eos_pos[-1]].tolist())
        print(f"[{split_name}] last article:")
        print(f"{last_article[:120]}...")
        print()

    print("=" * 60)
    print("7. Stream format")
    print("=" * 60)

    for split_name, arr, eos_pos in [("train", train, train_eos), ("val", val, val_eos)]:
        assert eos_pos[0] > 0, "First token cannot be EOS"
        for j in range(1, len(eos_pos)):
            gap = eos_pos[j] - eos_pos[j - 1]
            assert gap > 1, f"Consecutive EOS at positions {eos_pos[j-1]}, {eos_pos[j]}"
        print(f"[{split_name}] {len(eos_pos)} articles, all non-empty.")
    print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"train tokens: {len(train):<12} val tokens: {len(val)}")
    print(f"train EOS:    {len(train_eos):<12} val EOS:    {len(val_eos)}")
    print(f"max id:       {train_max:<12} max id:     {val_max}")
    print("dtype:        uint16")
    print()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    validate()
