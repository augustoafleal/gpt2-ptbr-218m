#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Extract a Wikimedia XML dump (.bz2) into JSON using WikiExtractor."
    )
    parser.add_argument(
        "--dump",
        default=None,
        help="Path to the .bz2 dump (default: data/raw/wiki_dump.xml.bz2)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: data/extracted)",
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=200,
        help="Minimum text length for WikiExtractor (default: 200).",
    )
    parser.add_argument(
        "--keep-templates",
        action="store_true",
        help="Keep templates (by default --no_templates is used).",
    )
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    default_dump = project_root / "data" / "raw" / "wiki_dump.xml.bz2"
    dump_path = Path(args.dump) if args.dump else default_dump
    if not dump_path.is_absolute():
        dump_path = project_root / dump_path

    if not dump_path.exists():
        raise SystemExit(
            f"Dump not found: {dump_path}\n"
            "Run scripts/download_wiki.py first or place the .bz2 file in data/raw/."
        )

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "data" / "extracted"
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "wikiextractor.WikiExtractor",
        "--json",
        "--min_text_length",
        str(args.min_text_length),
        "-o",
        str(output_dir),
    ]
    if not args.keep_templates:
        cmd.append("--no_templates")
    cmd.append(str(dump_path))

    print(f"Extracting {dump_path}")
    print(f"To {output_dir}")
    subprocess.run(cmd, check=True)
    print(f"Extraction finished in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
