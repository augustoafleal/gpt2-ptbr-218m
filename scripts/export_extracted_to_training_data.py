from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util

_ingest_dir = str(PROJECT_ROOT / "ingest")
sys.path.insert(0, _ingest_dir)
import config as _ingest_config  # noqa: F401
import db as _ingest_db  # noqa: F401
_spec = importlib.util.spec_from_file_location(
    "_ingest_mod", PROJECT_ROOT / "ingest" / "ingest.py"
)
_ingest_mod = importlib.util.module_from_spec(_spec)
sys.modules["_ingest_mod"] = _ingest_mod
_spec.loader.exec_module(_ingest_mod)
sys.path.remove(_ingest_dir)

normalize_article = _ingest_mod.normalize_article
iter_articles_from_file = _ingest_mod.iter_articles_from_file

from scripts.export_training_data import clean_field


def iter_wiki_files(input_dir: Path):
    for path in sorted(input_dir.rglob("wiki_*")):
        if path.is_file():
            yield path


def build_metadata(
    input_dir: str,
    output_file: str,
    total_files: int,
    total_articles_seen: int,
    total_articles_written: int,
    duplicates: int,
    skipped_empty: int,
    skipped_short: int,
    total_characters: int,
    min_length: int,
    elapsed_seconds: float,
) -> dict:
    return {
        "input_dir": input_dir,
        "output_file": output_file,
        "total_files": total_files,
        "total_articles_seen": total_articles_seen,
        "total_articles_written": total_articles_written,
        "duplicates": duplicates,
        "skipped_empty": skipped_empty,
        "skipped_short": skipped_short,
        "total_characters": total_characters,
        "min_length": min_length,
        "elapsed_seconds": round(elapsed_seconds, 2),
    }


def export(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir).resolve()
    output_path = Path(args.output_file).resolve()
    min_length = args.min_length

    if not input_dir.is_dir():
        print(f"Erro: diretório de entrada não encontrado: {input_dir}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = list(iter_wiki_files(input_dir))
    if not files:
        print(f"Erro: nenhum arquivo wiki_* encontrado em {input_dir}")
        sys.exit(1)

    print(f"Diretório de entrada: {input_dir}")
    print(f"Arquivos encontrados: {len(files)}")
    print(f"Arquivo de saída:     {output_path}")
    print(f"Comprimento mínimo:   {min_length}")
    print()

    seen_ids: set[str] = set()
    articles: list[tuple[str, str, str]] = []
    stats = {
        "total_articles_seen": 0,
        "duplicates": 0,
        "skipped_empty": 0,
        "skipped_short": 0,
    }

    start = time.time()

    print("Lendo artigos...")
    for file_idx, file_path in enumerate(files, start=1):
        for article_id, title, text, length in iter_articles_from_file(file_path):
            stats["total_articles_seen"] += 1
            if article_id in seen_ids:
                stats["duplicates"] += 1
                continue
            if not title:
                stats["skipped_empty"] += 1
                continue
            if length < min_length:
                stats["skipped_short"] += 1
                continue
            seen_ids.add(article_id)
            articles.append((article_id, title, text))

        if file_idx % 100 == 0 or file_idx == len(files):
            print(
                f"  [{file_idx}/{len(files)}] "
                f"{stats['total_articles_seen']} vistos, "
                f"{len(articles)} acumulados, "
                f"{stats['duplicates']} duplicatas"
            )

    print(f"\nOrdenando {len(articles)} artigos por id...")
    articles.sort(key=lambda x: int(x[0]) if x[0].isdigit() else x[0])

    print("Escrevendo corpus...")
    total_chars = 0
    with output_path.open("w", encoding="utf-8") as f:
        for article_id, title, text in articles:
            title = clean_field(title)
            text = clean_field(text)
            f.write(f"{title}\n{text}\n\n<eos>\n")
            total_chars += len(title) + len(text)

    elapsed = time.time() - start

    meta = build_metadata(
        input_dir=str(input_dir),
        output_file=str(output_path),
        total_files=len(files),
        total_articles_seen=stats["total_articles_seen"],
        total_articles_written=len(articles),
        duplicates=stats["duplicates"],
        skipped_empty=stats["skipped_empty"],
        skipped_short=stats["skipped_short"],
        total_characters=total_chars,
        min_length=min_length,
        elapsed_seconds=elapsed,
    )

    meta_path = output_path.with_suffix(".metadata.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print()
    print(f"Total de artigos vistos:    {stats['total_articles_seen']:>8}")
    print(f"Artigos escritos:           {len(articles):>8}")
    print(f"Duplicatas descartadas:     {stats['duplicates']:>8}")
    print(f"Skip (vazio):               {stats['skipped_empty']:>8}")
    print(f"Skip (curto < {min_length}):       {stats['skipped_short']:>8}")
    print(f"Caracteres totais:          {total_chars:>12}")
    print(f"Tempo total:                {elapsed:.1f}s")
    print()
    print(f"Corpus:  {output_path}")
    print(f"Metadados: {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Exporta corpus de treino diretamente dos JSON lines do "
            "WikiExtractor, pulando PostgreSQL."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/extracted",
        help="Diretório com arquivos wiki_* do WikiExtractor (default: data/extracted)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/training/dataset_general_full.txt",
        help="Arquivo de saída no formato TITLE\\nTEXT\\n\\n<eos>\\n",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=int(os.getenv("MIN_TEXT_LENGTH", "200")),
        help=(
            "Comprimento mínimo do texto (default: MIN_TEXT_LENGTH do .env, "
            "ou 200)"
        ),
    )
    args = parser.parse_args()
    export(args)


if __name__ == "__main__":
    main()
