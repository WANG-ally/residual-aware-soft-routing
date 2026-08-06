#!/usr/bin/env python3
"""Audit the official Pedestrian hourly dataset before modelling."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(
    os.environ.get("SCI_ROUTING_ROOT", Path(__file__).resolve().parents[3])
).resolve()
OUTPUT_ROOT = Path(os.environ.get("SCI_ROUTING_OUTPUT_ROOT", PROJECT_ROOT)).resolve()

# During verification the compatibility reader is beside this scratch file.
# After installation it is in the project root.
for candidate in (PROJECT_ROOT, SCRIPT_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tsf_reader_compat import read_tsf  # noqa: E402


DATASET_ID = "pedestrian_hourly"
TSF_PATH = (
    PROJECT_ROOT
    / "data/raw/pedestrian_hourly_staging/pedestrian_hourly"
    / "pedestrian_counts_dataset.tsf"
)
ARCHIVE_PATH = (
    PROJECT_ROOT
    / "data/raw/pedestrian_hourly_staging/pedestrian_counts_dataset.zip"
)
RECEIPT_PATH = (
    PROJECT_ROOT / "data/raw/pedestrian_hourly_staging/download_receipt.json"
)
MANIFEST_PATH = PROJECT_ROOT / "data_manifest.csv"
CONFIG_PATH = PROJECT_ROOT / "experiment_config.yaml"

RESULTS_DIR = OUTPUT_ROOT / "results"
FIGURES_DIR = OUTPUT_ROOT / "figures"
LOGS_DIR = OUTPUT_ROOT / "logs"

AUDIT_CSV = RESULTS_DIR / "pedestrian_series_audit.csv"
CHECKS_CSV = RESULTS_DIR / "pedestrian_audit_checks.csv"
SUMMARY_YAML = RESULTS_DIR / "pedestrian_data_quality_summary.yaml"
FIGURE_PATH = FIGURES_DIR / "pedestrian_data_audit.png"
REPORT_JSON = LOGS_DIR / "pedestrian_data_audit_report.json"


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def plain_timestamp(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def robust_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    iqr = float(q75 - q25)
    if not np.isfinite(iqr) or iqr <= 0:
        iqr = float(np.std(values))
    if not np.isfinite(iqr) or iqr <= 0:
        iqr = 1.0
    return (values - median) / iqr


def main() -> None:
    for path in (TSF_PATH, ARCHIVE_PATH, RECEIPT_PATH, MANIFEST_PATH, CONFIG_PATH):
        require_file(path)
    for directory in (RESULTS_DIR, FIGURES_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    minimum_length = int(config["data"]["minimum_series_length_after_cleaning"])
    hourly_windows = [
        int(value) for value in config["preprocessing"]["window_by_frequency"]["hourly"]
    ]
    split_ratios = {
        key: float(value)
        for key, value in config["split"]["chronological_ratios"].items()
    }

    manifest = pd.read_csv(MANIFEST_PATH, dtype=str)
    manifest_row = manifest.loc[manifest["dataset_id"] == DATASET_ID]
    if len(manifest_row) != 1:
        raise AssertionError(
            f"Expected exactly one manifest row for {DATASET_ID}, got {len(manifest_row)}"
        )
    manifest_row = manifest_row.iloc[0]
    expected_md5 = str(manifest_row["md5"]).lower()
    archive_md5 = file_md5(ARCHIVE_PATH)

    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    if not isinstance(receipt, list) or len(receipt) != 1:
        raise AssertionError("Pedestrian receipt must contain exactly one record")
    receipt_row = receipt[0]

    data, metadata = read_tsf(TSF_PATH)
    required_columns = {"series_name", "start_timestamp", "series_value"}
    if not required_columns.issubset(data.columns):
        raise AssertionError(
            f"TSF columns are incomplete: expected {sorted(required_columns)}"
        )

    audit_rows: list[dict[str, object]] = []
    total_missing = 0
    total_zeros = 0
    total_nonfinite = 0
    all_values_integer = True

    for row in data.itertuples(index=False):
        values = np.asarray(row.series_value, dtype=float)
        start = pd.Timestamp(row.start_timestamp)
        length = int(len(values))
        finite = np.isfinite(values)
        missing_count = int(np.isnan(values).sum())
        nonfinite_count = int((~finite).sum())
        finite_values = values[finite]
        zero_count = int(np.count_nonzero(finite_values == 0))
        end = start + pd.to_timedelta(length - 1, unit="h")

        total_missing += missing_count
        total_nonfinite += nonfinite_count
        total_zeros += zero_count
        all_values_integer = all_values_integer and bool(
            np.all(np.equal(finite_values, np.floor(finite_values)))
        )

        audit_rows.append(
            {
                "dataset_id": DATASET_ID,
                "series_id": str(row.series_name),
                "start_timestamp": plain_timestamp(start),
                "end_timestamp": plain_timestamp(end),
                "length": length,
                "missing_count": missing_count,
                "nonfinite_count": nonfinite_count,
                "zero_count": zero_count,
                "minimum": float(np.min(finite_values)),
                "maximum": float(np.max(finite_values)),
                "mean": float(np.mean(finite_values)),
                "median": float(np.median(finite_values)),
                "nonnegative": bool(np.all(finite_values >= 0)),
                "integer_valued": bool(
                    np.all(np.equal(finite_values, np.floor(finite_values)))
                ),
                "meets_minimum_length": length >= minimum_length,
            }
        )

    audit = pd.DataFrame(audit_rows).sort_values("series_id").reset_index(drop=True)
    starts = pd.to_datetime(audit["start_timestamp"], format="%Y-%m-%d %H:%M:%S")
    ends = pd.to_datetime(audit["end_timestamp"], format="%Y-%m-%d %H:%M:%S")
    lengths = audit["length"].to_numpy(dtype=int)

    check_items: list[tuple[str, bool, str]] = [
        ("manifest_has_one_row", True, DATASET_ID),
        (
            "archive_md5_matches_manifest",
            archive_md5 == expected_md5,
            f"actual={archive_md5}; expected={expected_md5}",
        ),
        (
            "receipt_dataset_matches",
            str(receipt_row.get("dataset_id")) == DATASET_ID,
            str(receipt_row.get("dataset_id")),
        ),
        (
            "receipt_md5_matches",
            str(receipt_row.get("md5", "")).lower() == expected_md5,
            str(receipt_row.get("md5")),
        ),
        (
            "supported_source_encoding",
            str(metadata.get("source_encoding")) in {"utf-8", "cp1252"},
            str(metadata.get("source_encoding")),
        ),
        (
            "frequency_is_hourly",
            str(metadata.get("frequency", "")).lower() == "hourly",
            str(metadata.get("frequency")),
        ),
        (
            "metadata_missing_is_false",
            metadata.get("missing") is False,
            str(metadata.get("missing")),
        ),
        (
            "metadata_equal_length_is_false",
            metadata.get("equallength") is False,
            str(metadata.get("equallength")),
        ),
        ("series_count_is_66", len(audit) == 66, str(len(audit))),
        (
            "series_ids_are_T1_to_T66",
            set(audit["series_id"]) == {f"T{i}" for i in range(1, 67)},
            f"unique={audit['series_id'].nunique()}",
        ),
        (
            "series_ids_are_unique",
            audit["series_id"].is_unique,
            f"unique={audit['series_id'].nunique()}",
        ),
        ("timestamps_parse_exactly", not starts.isna().any(), "format=%Y-%m-%d %H-%M-%S"),
        (
            "timestamp_seconds_are_archive_offsets",
            bool(starts.dt.second.isin([0, 1]).all()),
            f"seconds={sorted(starts.dt.second.unique().tolist())}",
        ),
        ("no_missing_values", total_missing == 0, str(total_missing)),
        ("no_nonfinite_values", total_nonfinite == 0, str(total_nonfinite)),
        (
            "all_values_nonnegative",
            bool(audit["nonnegative"].all()),
            f"global_min={audit['minimum'].min()}",
        ),
        ("all_values_integer_valued", all_values_integer, str(all_values_integer)),
        (
            "all_series_meet_minimum_length",
            bool(audit["meets_minimum_length"].all()),
            f"minimum_observed={lengths.min()}; required={minimum_length}",
        ),
        (
            "series_lengths_are_variable",
            int(audit["length"].nunique()) > 1,
            f"unique_lengths={audit['length'].nunique()}",
        ),
    ]
    checks = pd.DataFrame(check_items, columns=["check", "passed", "detail"])
    failed = checks.loc[~checks["passed"]]
    if not failed.empty:
        failed_text = "; ".join(
            f"{row.check}: {row.detail}" for row in failed.itertuples(index=False)
        )
        raise AssertionError(f"Pedestrian audit failed: {failed_text}")

    audit.to_csv(AUDIT_CSV, index=False)
    checks.to_csv(CHECKS_CSV, index=False)

    total_points = int(lengths.sum())
    summary = {
        "dataset_id": DATASET_ID,
        "audit_passed": True,
        "source": {
            "doi": str(manifest_row["doi"]),
            "license": str(manifest_row["license"]),
            "archive_filename": str(manifest_row["filename"]),
            "archive_md5": archive_md5,
            "source_encoding": str(metadata.get("source_encoding")),
            "timestamp_text_format": "%Y-%m-%d %H-%M-%S",
            "frequency": str(metadata.get("frequency")),
            "archive_snapshot_end_note": str(manifest_row["notes"]),
        },
        "series": {
            "count": int(len(audit)),
            "total_observations": total_points,
            "length_minimum": int(lengths.min()),
            "length_first_quartile": float(np.quantile(lengths, 0.25)),
            "length_median": float(np.median(lengths)),
            "length_third_quartile": float(np.quantile(lengths, 0.75)),
            "length_maximum": int(lengths.max()),
            "unique_lengths": int(audit["length"].nunique()),
            "earliest_start": plain_timestamp(starts.min()),
            "latest_start": plain_timestamp(starts.max()),
            "earliest_end": plain_timestamp(ends.min()),
            "latest_end": plain_timestamp(ends.max()),
        },
        "values": {
            "missing_count": int(total_missing),
            "nonfinite_count": int(total_nonfinite),
            "zero_count": int(total_zeros),
            "global_minimum": float(audit["minimum"].min()),
            "global_maximum": float(audit["maximum"].max()),
            "all_nonnegative": bool(audit["nonnegative"].all()),
            "all_integer_valued": bool(all_values_integer),
            "imputation_applied": False,
        },
        "registered_protocol": {
            "minimum_series_length": minimum_length,
            "all_series_eligible_by_length": bool(
                audit["meets_minimum_length"].all()
            ),
            "chronological_split_ratios": split_ratios,
            "candidate_windows": hourly_windows,
            "seasonal_period_hours": 24,
            "test_values_accessed_for_modelling": False,
            "models_fitted": False,
            "parameters_tuned": False,
        },
        "warnings": [
            "The official TSF is Windows-1252 encoded; do not replace the frozen NN5 reader.",
            "Series lengths and start dates differ, so later chronological boundaries must be computed per series.",
            "The archive contains start-second offsets of 00 and 01; they are preserved and do not change within-series hourly spacing.",
        ],
    }
    SUMMARY_YAML.write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    source_order_lengths = data["series_value"].map(len).to_numpy(dtype=int)
    sorted_index = np.argsort(source_order_lengths)
    sample_positions = [sorted_index[0], sorted_index[len(sorted_index) // 2], sorted_index[-1]]
    sample_labels = ["shortest", "median-length", "longest"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].hist(lengths, bins=12, color="#4C78A8", edgecolor="white")
    axes[0].axvline(np.median(lengths), color="#E45756", linestyle="--", linewidth=2)
    axes[0].set_title("Series length distribution")
    axes[0].set_xlabel("Hourly observations per series")
    axes[0].set_ylabel("Number of series")

    timeline_order = np.argsort(starts.to_numpy())
    for display_y, row_index in enumerate(timeline_order):
        axes[1].plot(
            [starts.iloc[row_index], ends.iloc[row_index]],
            [display_y, display_y],
            color="#59A14F",
            linewidth=1.5,
        )
    axes[1].set_title("Observed time spans")
    axes[1].set_xlabel("Calendar time")
    axes[1].set_ylabel("Series (ordered by start)")
    axes[1].tick_params(axis="x", rotation=30)

    for row_index, label in zip(sample_positions, sample_labels):
        values = np.asarray(data.iloc[int(row_index)]["series_value"], dtype=float)
        tail = values[-min(168, len(values)) :]
        series_id = str(data.iloc[int(row_index)]["series_name"])
        axes[2].plot(
            np.arange(-len(tail) + 1, 1),
            robust_scale(tail),
            linewidth=1.2,
            label=f"{label}: {series_id} (n={len(values)})",
        )
    axes[2].set_title("Last week of three example series")
    axes[2].set_xlabel("Hours before series end")
    axes[2].set_ylabel("Robust-scaled pedestrian count")
    axes[2].legend(fontsize=8)

    fig.suptitle("Pedestrian hourly dataset: pre-modelling audit", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = {
        "status": "passed",
        "dataset_id": DATASET_ID,
        "check_count": int(len(checks)),
        "failed_check_count": 0,
        "series_count": int(len(audit)),
        "total_observations": total_points,
        "source_encoding": str(metadata.get("source_encoding")),
        "archive_md5": archive_md5,
        "outputs": {
            "series_audit": str(AUDIT_CSV),
            "audit_checks": str(CHECKS_CSV),
            "quality_summary": str(SUMMARY_YAML),
            "audit_figure": str(FIGURE_PATH),
        },
    }
    REPORT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("Pedestrian 小时数据质量检查全部通过")
    print("读取编码：", metadata.get("source_encoding"))
    print("时间文本格式：%Y-%m-%d %H-%M-%S")
    print("序列数量：", len(audit))
    print("总观测数量：", total_points)
    print(
        "序列长度（最短/中位数/最长）：",
        f"{lengths.min()} / {np.median(lengths):.1f} / {lengths.max()}",
    )
    print("总缺失值数量：", total_missing)
    print("零值数量：", total_zeros)
    print(
        "全局取值范围：",
        f"{audit['minimum'].min():.0f} 至 {audit['maximum'].max():.0f}",
    )
    print(
        "起始日期范围：",
        f"{plain_timestamp(starts.min())} 至 {plain_timestamp(starts.max())}",
    )
    print(
        "结束日期范围：",
        f"{plain_timestamp(ends.min())} 至 {plain_timestamp(ends.max())}",
    )
    print(f"全部序列长度不少于 {minimum_length}：是")
    print("测试集是否用于建模：否")
    print("审计明细：", AUDIT_CSV)
    print("检查清单：", CHECKS_CSV)
    print("质量摘要：", SUMMARY_YAML)
    print("审计图片：", FIGURE_PATH)


if __name__ == "__main__":
    main()
