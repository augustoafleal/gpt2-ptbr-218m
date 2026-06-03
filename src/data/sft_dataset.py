from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import sentencepiece as spm


def load_alpaca_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list or dict with 'data' key, got {type(data)}")
    return data


def normalize_example(example: dict) -> dict | None:
    instruction = example.get("instruction", "").strip()
    output = example.get("output", "").strip()
    if not instruction or not output:
        return None
    inp = example.get("input", "").strip()
    return {"instruction": instruction, "input": inp, "output": output}


def format_example(example: dict) -> str:
    parts = ["### Instrução:", example["instruction"]]
    if example["input"]:
        parts.append("")
        parts.append("### Entrada:")
        parts.append(example["input"])
    parts.append("")
    parts.append("### Resposta:")
    parts.append(example["output"])
    return "\n".join(parts)


def train_val_split(
    examples: list[dict],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    split = int(len(indices) * (1 - val_ratio))
    train_idx = indices[:split]
    val_idx = indices[split:]
    return [examples[i] for i in train_idx], [examples[i] for i in val_idx]


def format_and_tokenize(
    examples: list[dict],
    sp: spm.SentencePieceProcessor,
    eos_id: int,
) -> list[int]:
    tokens: list[int] = []
    for ex in examples:
        text = format_example(ex)
        ids = sp.encode(text)
        tokens.extend(ids)
        tokens.append(eos_id)
    return tokens


def save_txt(examples: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(format_example(ex))
            f.write("\n<eos>\n")


def save_bin(tokens: list[int], path: Path) -> None:
    np.array(tokens, dtype=np.uint16).tofile(str(path))


def save_metadata(metadata: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
