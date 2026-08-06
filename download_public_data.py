#!/usr/bin/env python3
"""Download the fixed public-data snapshots used by the experiment protocol.

The script uses only the Python standard library. It verifies every archive
against the MD5 published on the corresponding Zenodo record and can extract
ZIP files into one directory per dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


CHUNK = 1024 * 1024


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"dataset_id", "tier", "filename", "md5", "url"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain columns: {sorted(required)}")
    return rows


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def select_rows(rows: list[dict[str, str]], names: set[str], tiers: set[str]) -> list[dict[str, str]]:
    selected = []
    for row in rows:
        if names and row["dataset_id"] not in names:
            continue
        if tiers and row["tier"] not in tiers:
            continue
        selected.append(row)
    known = {row["dataset_id"] for row in rows}
    unknown = names - known
    if unknown:
        raise ValueError(f"Unknown dataset_id values: {', '.join(sorted(unknown))}")
    return selected


def progress_download(url: str, destination: Path) -> None:
    part = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "SCI-routing-study-data-downloader/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as handle:
        total = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        while True:
            chunk = response.read(CHUNK)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = 100.0 * downloaded / total
                print(f"\r  {downloaded / 1e6:8.1f}/{total / 1e6:.1f} MB  {percent:5.1f}%", end="", flush=True)
            else:
                print(f"\r  {downloaded / 1e6:8.1f} MB", end="", flush=True)
    print()
    os.replace(part, destination)


def safe_extract(archive: Path, destination: Path, overwrite: bool) -> None:
    if destination.exists() and overwrite:
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        zipped.extractall(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("data_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--names", default="", help="Comma-separated dataset_id values.")
    parser.add_argument("--tier", action="append", choices=["core", "extended"], default=[])
    parser.add_argument("--list", action="store_true", help="List matching datasets without downloading.")
    parser.add_argument("--extract", action="store_true", help="Extract verified ZIP archives.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing archives/extracted directories.")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    names = {item.strip() for item in args.names.split(",") if item.strip()}
    selected = select_rows(rows, names, set(args.tier))
    if not selected:
        print("No datasets matched the requested filters.", file=sys.stderr)
        return 2

    if args.list:
        for row in selected:
            print(
                f"{row['dataset_id']:<22} tier={row['tier']:<8} "
                f"series={row.get('series_count', '?'):<5} doi={row.get('doi', '')}"
            )
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt: list[dict[str, str]] = []
    for row in selected:
        archive = args.output_dir / row["filename"]
        expected = row["md5"].lower()
        print(f"[{row['dataset_id']}] {row['filename']}")

        if archive.exists() and not args.overwrite:
            actual = md5sum(archive)
            if actual != expected:
                raise RuntimeError(
                    f"Existing file has the wrong MD5: {archive}\n"
                    f"expected={expected}\nactual={actual}\n"
                    "Delete it or rerun with --overwrite."
                )
            print("  archive already present; MD5 verified")
        else:
            if archive.exists():
                archive.unlink()
            progress_download(row["url"], archive)
            actual = md5sum(archive)
            if actual != expected:
                archive.unlink(missing_ok=True)
                raise RuntimeError(
                    f"MD5 mismatch for {row['dataset_id']}: expected {expected}, got {actual}"
                )
            print("  download complete; MD5 verified")

        if args.extract:
            extract_dir = args.output_dir / row["dataset_id"]
            if extract_dir.exists() and not args.overwrite:
                print(f"  extraction directory already exists: {extract_dir}")
            else:
                safe_extract(archive, extract_dir, args.overwrite)
                print(f"  extracted to {extract_dir}")

        receipt.append(
            {
                "dataset_id": row["dataset_id"],
                "doi": row.get("doi", ""),
                "archive": str(archive.resolve()),
                "md5": expected,
                "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    receipt_path = args.output_dir / "download_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Receipt written to {receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
