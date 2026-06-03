from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download Alpaca PT-BR dataset from Hugging Face"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="dominguesm/alpaca-data-pt-br",
        help="Hugging Face dataset name (default: dominguesm/alpaca-data-pt-br)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/sft/alpaca_ptbr/raw/alpaca_data_ptbr.json",
        help="Output JSON path (default: data/sft/alpaca_ptbr/raw/alpaca_data_ptbr.json)",
    )
    args = parser.parse_args(argv)

    output_path = PROJECT_ROOT / args.output_path

    if output_path.exists():
        print(f"File already exists: {output_path}")
        print("Skipping download.")
        return 0

    from datasets import load_dataset

    print(f"Downloading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")

    required_fields = {"instruction", "input", "output"}
    missing = required_fields - set(dataset.features)
    if missing:
        print(f"Dataset missing required fields: {missing}")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for row in dataset:
        records.append({
            "instruction": row["instruction"],
            "input": row.get("input", ""),
            "output": row["output"],
        })

    import json
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(records)} examples to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
