from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _human_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def download_file(url: str, output_file: Path, user_agent: str, timeout_s: float) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    temp_file = output_file.with_suffix(output_file.suffix + ".part")
    existing_size = temp_file.stat().st_size if temp_file.exists() else 0

    headers = {"User-Agent": user_agent}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"

    request = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status = getattr(response, "status", None)
            if existing_size > 0 and status != 206:
                existing_size = 0
                temp_file.unlink(missing_ok=True)
                headers.pop("Range", None)
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=timeout_s) as response2:
                    _stream_response(response2, temp_file, resume_from=0)
            else:
                _stream_response(response, temp_file, resume_from=existing_size)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Download failed (HTTP {exc.code}): {url}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Download failed: {exc}") from exc

    temp_file.replace(output_file)


def _stream_response(response, temp_file: Path, resume_from: int) -> None:
    chunk_size = 8 * 1024 * 1024
    downloaded = resume_from
    last_log_t = time.monotonic()
    last_log_bytes = downloaded

    mode = "ab" if resume_from > 0 else "wb"
    with temp_file.open(mode) as f:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)

            now = time.monotonic()
            if now - last_log_t >= 5:
                delta_bytes = downloaded - last_log_bytes
                speed = delta_bytes / (now - last_log_t) if now > last_log_t else 0.0
                print(
                    f"Downloaded {_human_bytes(downloaded)} "
                    f"({_human_bytes(int(speed))}/s)",
                    flush=True,
                )
                last_log_t = now
                last_log_bytes = downloaded


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Download a Wikimedia 'pages-articles-multistream' dump (XML .bz2)."
    )
    parser.add_argument(
        "lang_code",
        nargs="?",
        default="ptwiki",
        help="Wikimedia project code (default: ptwiki). Example: enwiki",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: data/raw/wiki_dump.xml.bz2)",
    )
    parser.add_argument(
        "--user-agent",
        default=os.environ.get("WIKI_DUMP_USER_AGENT", "llm-project/0.1 (dump downloader)"),
        help="HTTP User-Agent header (or env WIKI_DUMP_USER_AGENT).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Network timeout in seconds (default: 60).",
    )
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"
    output_file = Path(args.output) if args.output else (raw_dir / "wiki_dump.xml.bz2")
    if not output_file.is_absolute():
        output_file = project_root / output_file

    url = (
        f"https://dumps.wikimedia.org/{args.lang_code}/latest/"
        f"{args.lang_code}-latest-pages-articles-multistream.xml.bz2"
    )

    print(f"Downloading {url}")
    print(f"To {output_file}")
    if output_file.with_suffix(output_file.suffix + ".part").exists():
        print("Resuming from existing .part file (if server supports HTTP Range).")

    download_file(url, output_file, user_agent=args.user_agent, timeout_s=args.timeout)
    print(f"Dump saved to {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

