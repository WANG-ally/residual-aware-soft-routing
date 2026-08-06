#!/usr/bin/env python3
"""Encoding-compatible reader for Monash .tsf archives."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


DATE_WITH_HYPHEN_TIME = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}$"
)


def detect_text_encoding(path: str | Path) -> str:
    """Prefer UTF-8 and fall back to Windows-1252 when required."""
    path = Path(path)
    for encoding in ("utf-8", "cp1252"):
        try:
            with path.open("r", encoding=encoding) as handle:
                while handle.read(1024 * 1024):
                    pass
            return encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Unable to decode {path} as UTF-8 or cp1252")


def parse_attribute(value: str, kind: str) -> Any:
    if kind == "numeric":
        return float(value)
    if kind == "date":
        if DATE_WITH_HYPHEN_TIME.fullmatch(value):
            return pd.to_datetime(value, format="%Y-%m-%d %H-%M-%S")
        return pd.to_datetime(value, errors="raise")
    return value


def read_tsf(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(path)
    encoding = detect_text_encoding(path)
    attributes: list[tuple[str, str]] = []
    metadata: dict[str, Any] = {"source_encoding": encoding}
    rows: list[dict[str, Any]] = []
    in_data = False

    with path.open("r", encoding=encoding) as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            lower = line.lower()
            if not in_data:
                if lower.startswith("@attribute"):
                    _, name, kind = line.split(maxsplit=2)
                    attributes.append((name, kind.lower()))
                elif lower == "@data":
                    in_data = True
                elif line.startswith("@"):
                    key, *rest = line[1:].split(maxsplit=1)
                    value: Any = rest[0] if rest else True
                    if isinstance(value, str) and value.lower() in {"true", "false"}:
                        value = value.lower() == "true"
                    elif key.lower() == "horizon" and isinstance(value, str):
                        value = int(value)
                    metadata[key.lower()] = value
                continue

            parts = line.split(":")
            if len(parts) != len(attributes) + 1:
                raise ValueError(
                    f"Malformed row {line_number} in {path.name}: expected "
                    f"{len(attributes) + 1} fields, got {len(parts)}"
                )
            record = {
                name: parse_attribute(value, kind)
                for (name, kind), value in zip(attributes, parts[:-1])
            }
            values = np.asarray(
                [np.nan if item == "?" else float(item) for item in parts[-1].split(",")],
                dtype=float,
            )
            if len(values) == 0 or np.isnan(values).all():
                raise ValueError(f"Empty/all-missing row {line_number} in {path.name}")
            record["series_value"] = values
            rows.append(record)

    if not in_data or not rows:
        raise ValueError(f"No @data section or no series found in {path}")
    metadata["attributes"] = attributes
    return pd.DataFrame(rows), metadata


def to_long(data: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    id_column = "series_name" if "series_name" in data.columns else None
    frames = []
    for row_number, row in data.iterrows():
        values = np.asarray(row["series_value"], dtype=float)
        series_id = str(row[id_column]) if id_column else f"series_{row_number:06d}"
        frame = pd.DataFrame(
            {
                "dataset_id": dataset_id,
                "series_id": series_id,
                "time_index": np.arange(len(values), dtype=np.int64),
                "value": values,
            }
        )
        if "start_timestamp" in data.columns:
            frame["timestamp"] = row["start_timestamp"] + pd.to_timedelta(
                frame["time_index"], unit="h"
            )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
