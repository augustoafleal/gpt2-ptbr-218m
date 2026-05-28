from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from ingest.db import get_connection

BATCH_SIZE = 1000
OUTPUT_PATH = PROJECT_ROOT / "data" / "training" / "dataset_general.txt"

QUERY = "SELECT title, text FROM wiki_articles WHERE length > 200 ORDER BY id;"


def clean_field(value: str) -> str:
    value = value.replace("\n", " ")
    value = " ".join(value.split())
    return value


def export() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(QUERY)

    exported = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        while True:
            rows = cursor.fetchmany(BATCH_SIZE)
            if not rows:
                break
            for title, text in rows:
                title = clean_field(title)
                text = clean_field(text)
                f.write(f"{title}\n{text}\n\n<eos>\n")
                exported += 1

    cursor.close()
    conn.close()

    print(f"Artigos exportados: {exported}")
    print(f"Arquivo gerado: {OUTPUT_PATH}")


if __name__ == "__main__":
    export()
