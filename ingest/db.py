from __future__ import annotations

import os
from typing import Tuple

import psycopg2
from psycopg2.extensions import connection as PGConnection
from psycopg2.extensions import cursor as PGCursor


def get_connection() -> PGConnection:
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "wikidb"),
        user=os.getenv("POSTGRES_USER", "wikiuser"),
        password=os.getenv("POSTGRES_PASSWORD", "wikipass"),
    )


def get_connection_and_cursor() -> Tuple[PGConnection, PGCursor]:
    connection = get_connection()
    cursor = connection.cursor()
    return connection, cursor
