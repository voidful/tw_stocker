#!/usr/bin/env python3
"""Build and validate the deterministic manifest for the CSV data archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"
TICKER_PATTERN = re.compile(r"^[0-9A-Z]+$")
SUPPORTED_HEADERS = {
    (
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    ): "daily",
    (
        "Datetime",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Dividends",
        "Stock Splits",
    ): "intraday",
    (
        "Datetime",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Dividends",
        "Stock Splits",
        "Capital Gains",
    ): "intraday",
}


class DataError(ValueError):
    """Raised when an input CSV does not match the documented archive schema."""


def parse_row(raw_line: bytes, path: Path) -> list[str]:
    try:
        return next(csv.reader([raw_line.decode("utf-8").rstrip("\r\n")]))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise DataError(f"{path}: invalid UTF-8 CSV row: {exc}") from exc


def inspect_csv(path: Path) -> dict[str, object]:
    ticker = path.stem
    if not TICKER_PATTERN.fullmatch(ticker):
        raise DataError(f"{path}: filename must contain only uppercase letters and digits")

    digest = hashlib.sha256()
    newline_count = 0
    last_byte = b""
    with path.open("rb") as file:
        header_line = file.readline()
        first_line = file.readline()
        second_line = file.readline()
        if not header_line:
            raise DataError(f"{path}: CSV must contain a header")
        file.seek(0)
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
            newline_count += chunk.count(b"\n")
            last_byte = chunk[-1:]

    line_count = newline_count + (0 if last_byte == b"\n" else 1)
    records = line_count - 1
    header = tuple(parse_row(header_line, path))
    frequency = SUPPORTED_HEADERS.get(header)
    if frequency is None:
        raise DataError(f"{path}: unsupported columns: {','.join(header)}")

    first_row: list[str] | None = None
    last_row: list[str] | None = None
    if records:
        first_row = parse_row(first_line, path)
        second_row = parse_row(second_line, path) if records > 1 else first_row
        with path.open("rb") as file:
            file.seek(-min(path.stat().st_size, 64 * 1024), 2)
            tail = file.read().splitlines()
        if not tail:
            raise DataError(f"{path}: unable to read final row")
        last_row = parse_row(tail[-1], path)

        for label, row in (("first", first_row), ("second", second_row), ("last", last_row)):
            if len(row) != len(header):
                raise DataError(
                    f"{path}: {label} row has {len(row)} columns; expected {len(header)}"
                )
            try:
                for column in ("Open", "High", "Low", "Close", "Volume"):
                    float(row[header.index(column)])
            except ValueError as exc:
                raise DataError(f"{path}: {label} row contains a non-numeric OHLCV value") from exc

    return {
        "ticker": ticker,
        "path": path.relative_to(ROOT).as_posix(),
        "frequency": frequency,
        "columns": list(header),
        "records": records,
        "bytes": path.stat().st_size,
        "start": first_row[0] if first_row else None,
        "end": last_row[0] if last_row else None,
        "sha256": digest.hexdigest(),
    }


def build_manifest() -> dict[str, object]:
    csv_paths = sorted(DATA_DIR.glob("*.csv"))
    if not csv_paths:
        raise DataError(f"No CSV files found in {DATA_DIR}")

    files: list[dict[str, object]] = []
    errors: list[str] = []
    for path in csv_paths:
        try:
            files.append(inspect_csv(path))
        except DataError as exc:
            errors.append(str(exc))
    if errors:
        details = "\n  - ".join(errors)
        raise DataError(f"{len(errors)} invalid CSV file(s):\n  - {details}")
    frequencies = Counter(str(item["frequency"]) for item in files)
    populated = [item for item in files if item["records"]]
    if not populated:
        raise DataError("All CSV files are empty")
    start_date = min(str(item["start"])[:10] for item in populated)
    end_date = max(str(item["end"])[:10] for item in populated)
    return {
        "schema_version": 1,
        "dataset": "Taiwan market OHLCV archive",
        "file_count": len(files),
        "record_count": sum(int(item["records"]) for item in files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "date_range": {"start": start_date, "end": end_date},
        "frequencies": dict(sorted(frequencies.items())),
        "files": files,
    }


def serialize(manifest: dict[str, object]) -> str:
    return json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate CSV files and fail if data/manifest.json is stale",
    )
    args = parser.parse_args()

    try:
        rendered = serialize(build_manifest())
    except DataError as exc:
        print(f"data validation failed: {exc}", file=sys.stderr)
        return 1

    if args.check:
        if not MANIFEST_PATH.exists():
            print(f"missing manifest: {MANIFEST_PATH.relative_to(ROOT)}", file=sys.stderr)
            return 1
        if MANIFEST_PATH.read_text(encoding="utf-8") != rendered:
            print("data/manifest.json is stale; run scripts/build_catalog.py", file=sys.stderr)
            return 1
        print("data archive and manifest are valid")
        return 0

    MANIFEST_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {MANIFEST_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
