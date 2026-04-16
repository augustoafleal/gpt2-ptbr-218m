from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
MIN_TEXT_LENGTH = int(os.getenv("MIN_TEXT_LENGTH", "200"))
JSON_GLOB = os.getenv("WIKI_JSON_GLOB", "**/*")
_configured_extracted_dir = os.getenv("WIKI_EXTRACTED_DIR", "").strip()
if not _configured_extracted_dir:
    raise RuntimeError(
        "WIKI_EXTRACTED_DIR is required (path to extracted WikiExtractor output). "
        "Set it in .env or your environment."
    )

EXTRACTED_DATA_DIR = Path(_configured_extracted_dir).expanduser()
if not EXTRACTED_DATA_DIR.is_absolute():
    EXTRACTED_DATA_DIR = PROJECT_ROOT / EXTRACTED_DATA_DIR


def ensure_data_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
