#!/usr/bin/env python3
"""Minimal reader for the Monash Time Series Forecasting .tsf format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _parse_attribute(value: str, kind: str) -> Any:
    if kind == "numeric":
        return float(value)
    if kind == "date":
        return pd.to_datetime(value)
    return value


def read_tsf(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return one row per series plus TSF metadata.

    The returned DataFrame contains the declared attribute columns and a
    ``series_value`` column whose entries are NumPy arrays. Missing values are
    represented by ``np.nan``.
    """

    path = Path(path)
    attributes: list[tuple[str, str]] = []
    metadata: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    in_data = False

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
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
                    value = rest[0] if rest else True
                    if isinstance(value, str) and value.lower() in {"true", "false"}:
                        value = value.lower() == "true"
                    elif key.lower() == "horizon" and isinstance(value, str):
                        value = int(value)
                    metadata[key.lower()] = value
                continue

            parts = line.split(":")
            if len(parts) != len(attributes) + 1:
                raise ValueError(
                    f"Malformed TSF data row in {path.name}: expected "
                    f"{len(attributes) + 1} colon-delimited fields, got {len(parts)}"
                )
            record = {
                name: _parse_attribute(value, kind)
                for (name, kind), value in zip(attributes, parts[:-1])
            }
            values = [np.nan if item == "?" else float(item) for item in parts[-1].split(",")]
            if not values or np.isnan(values).all():
                raise ValueError(f"Empty/all-missing series in {path.name}")
            record["series_value"] = np.asarray(values, dtype=float)
            rows.append(record)

    if not in_data or not rows:
        raise ValueError(f"No @data section or no series found in {path}")
    metadata["attributes"] = attributes
    return pd.DataFrame(rows), metadata


def to_long(data: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    """Convert a TSF DataFrame to dataset_id/series_id/time_index/value rows."""

    id_column = "series_name" if "series_name" in data.columns else None
    frames = []
    for row_number, row in data.iterrows():
        values = row["series_value"]
        series_id = str(row[id_column]) if id_column else f"series_{row_number:06d}"
        frames.append(
            pd.DataFrame(
                {
                    "dataset_id": dataset_id,
                    "series_id": series_id,
                    "time_index": np.arange(len(values), dtype=np.int64),
                    "value": values,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
