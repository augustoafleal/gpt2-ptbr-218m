from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import psycopg2
from psycopg2.extras import execute_values

from config import BATCH_SIZE, EXTRACTED_DATA_DIR, JSON_GLOB, MIN_TEXT_LENGTH, ensure_data_directories
from db import get_connection

ArticleRow = Tuple[str, str, str, int]

INSERT_SQL = """
    INSERT INTO wiki_articles (id, title, text, length)
    VALUES %s
    ON CONFLICT (id) DO NOTHING
"""


def iter_json_files(base_dir: Path) -> Iterator[Path]:
    for path in sorted(base_dir.glob(JSON_GLOB)):
        if path.is_file():
            yield path


def normalize_article(payload: dict) -> ArticleRow | None:
    article_id = str(payload.get("id", "")).strip()
    title = str(payload.get("title", "")).strip()
    text = str(payload.get("text", ""))
    length = len(text)

    if not article_id or not title or length < MIN_TEXT_LENGTH:
        return None

    return article_id, title, text, length


def iter_articles_from_file(file_path: Path) -> Iterator[ArticleRow]:
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line {line_number} in {file_path}")
                continue
            if not isinstance(payload, dict):
                continue

            article = normalize_article(payload)
            if article is not None:
                yield article


def batch_iterable(items: Iterable[ArticleRow], batch_size: int) -> Iterator[List[ArticleRow]]:
    batch: List[ArticleRow] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def safe_rollback(connection) -> bool:
    try:
        connection.rollback()
        return True
    except Exception as exc:
        print(f"Rollback failed: {exc}")
        return False


def safe_close_cursor(cursor) -> None:
    try:
        cursor.close()
    except Exception:
        pass


def safe_close_connection(connection) -> None:
    try:
        connection.close()
    except Exception:
        pass


def reconnect() -> tuple:
    connection = get_connection()
    cursor = connection.cursor()
    print("Database connection re-established")
    return connection, cursor


def insert_batch(cursor, batch: List[ArticleRow]) -> int:
    execute_values(cursor, INSERT_SQL, batch, page_size=BATCH_SIZE)
    return cursor.rowcount if cursor.rowcount > 0 else 0


def ingest_files(files: Sequence[Path]) -> int:
    total_inserted = 0
    total_batches = 0
    failed_batches = 0
    connection, cursor = reconnect()

    try:
        for file_path in files:
            file_inserted = 0
            file_batches = 0
            for batch_index, batch in enumerate(
                batch_iterable(iter_articles_from_file(file_path), BATCH_SIZE),
                start=1,
            ):
                file_batches += 1
                total_batches += 1

                try:
                    inserted_rows = insert_batch(cursor, batch)
                    connection.commit()
                    file_inserted += inserted_rows
                    print(
                        f"Processed batch {batch_index} from {file_path.name}: "
                        f"{inserted_rows} rows inserted"
                    )
                except (psycopg2.OperationalError, psycopg2.InterfaceError) as exc:
                    print(
                        f"Connection error on batch {batch_index} from {file_path.name}: {exc}"
                    )
                    safe_rollback(connection)
                    safe_close_cursor(cursor)
                    safe_close_connection(connection)

                    try:
                        connection, cursor = reconnect()
                        inserted_rows = insert_batch(cursor, batch)
                        connection.commit()
                        file_inserted += inserted_rows
                        print(
                            f"Retried batch {batch_index} from {file_path.name}: "
                            f"{inserted_rows} rows inserted"
                        )
                    except Exception as retry_exc:
                        safe_rollback(connection)
                        safe_close_cursor(cursor)
                        safe_close_connection(connection)
                        failed_batches += 1
                        print(
                            f"Retry failed for batch {batch_index} from {file_path.name}: "
                            f"{retry_exc}"
                        )
                        connection, cursor = reconnect()
                        continue
                except Exception as exc:
                    rollback_ok = safe_rollback(connection)
                    if not rollback_ok:
                        safe_close_cursor(cursor)
                        safe_close_connection(connection)
                        connection, cursor = reconnect()

                    failed_batches += 1
                    print(
                        f"Error processing batch {batch_index} from {file_path.name}: {exc}"
                    )
                    continue

            total_inserted += file_inserted
            print(
                f"Finished file {file_path}: "
                f"{file_inserted} rows inserted across {file_batches} batches"
            )
    finally:
        safe_close_cursor(cursor)
        safe_close_connection(connection)

    print(
        f"Ingestion summary: {total_inserted} rows inserted, "
        f"{total_batches} batches processed, {failed_batches} batches failed"
    )
    return total_inserted


def main() -> None:
    ensure_data_directories()
    files = list(iter_json_files(EXTRACTED_DATA_DIR))

    if not files:
        print(f"No extracted files found in {EXTRACTED_DATA_DIR}")
        return

    print(f"Using extracted data from {EXTRACTED_DATA_DIR}")
    print(f"Found {len(files)} files to ingest")
    total_inserted = ingest_files(files)
    print(f"Finished ingestion. Total inserted rows: {total_inserted}")


if __name__ == "__main__":
    main()
