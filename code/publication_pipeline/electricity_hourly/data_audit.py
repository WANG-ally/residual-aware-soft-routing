#!/usr/bin/env python3
"""Source and quality audit for the Electricity Hourly archive.

This module verifies the fixed public snapshot and describes every series.  It
does not create train/test splits, fit models, select parameters, or calculate
forecast performance."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sci-routing")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(
    os.environ.get("SCI_ROUTING_ROOT", Path(__file__).resolve().parents[3])
).resolve()
OUTPUT_ROOT = Path(os.environ.get("SCI_ROUTING_OUTPUT_ROOT", PROJECT_ROOT)).resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tsf_reader_compat import read_tsf

DATASET_ID = "electricity_hourly"
EXPECTED_DOI = "10.5281/zenodo.4656140"
EXPECTED_LICENSE = "CC-BY-4.0"
EXPECTED_MD5 = "18096614662b02640d265ad2a6a416bd"
EXPECTED_SERIES = 321
EXPECTED_SERIES_LENGTH = 26_304
EXPECTED_TOTAL_OBSERVATIONS = 8_443_584
EXPECTED_TOTAL_ZEROS = 91_817
EXPECTED_SERIES_WITH_ZEROS = 229
EXPECTED_LONGEST_ZERO_RUN = 19_914
EXPECTED_GLOBAL_MINIMUM = 0.0
EXPECTED_GLOBAL_MAXIMUM = 764_000.0
EXPECTED_START_TIMESTAMP = "2012-01-01 00:00:01"
EXPECTED_END_TIMESTAMP = "2014-12-31 23:00:01"
EXPECTED_FREQUENCY = "hourly"
MAXIMUM_CANDIDATE_WINDOW = 168

MANIFEST_PATH = PROJECT_ROOT / "data_manifest.csv"
ARCHIVE_PATH = (
    PROJECT_ROOT / "data/raw/electricity_hourly_staging/electricity_hourly_dataset.zip"
)
RECEIPT_PATH = (
    PROJECT_ROOT / "data/raw/electricity_hourly_staging/download_receipt.json"
)
TSF_PATH = (
    PROJECT_ROOT
    / "data/raw/electricity_hourly_staging/electricity_hourly/electricity_hourly_dataset.tsf"
)

SERIES_AUDIT_PATH = OUTPUT_ROOT / "results/electricity_hourly_series_audit.csv"
SUMMARY_PATH = OUTPUT_ROOT / "results/electricity_hourly_data_quality_summary.yaml"
CHECKS_PATH = OUTPUT_ROOT / "results/electricity_hourly_audit_checks.csv"
FIGURE_PATH = OUTPUT_ROOT / "figures/electricity_hourly_quality_overview.png"
REPORT_PATH = OUTPUT_ROOT / "logs/electricity_hourly_data_audit_report.json"

for path in (
    SERIES_AUDIT_PATH,
    SUMMARY_PATH,
    CHECKS_PATH,
    FIGURE_PATH,
    REPORT_PATH,
):
    path.parent.mkdir(parents=True, exist_ok=True)


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def robust_scale_for_audit(values: np.ndarray) -> np.ndarray:
    """Scale a plotted series only; never reused for modeling."""
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    q1, q3 = np.quantile(values, [0.25, 0.75])
    scale = float(q3 - q1)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.std(values, ddof=0))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return (values - median) / scale


def longest_true_run(mask: np.ndarray) -> int:
    """Return the longest consecutive True run without changing the data."""
    mask = np.asarray(mask, dtype=bool)
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return int(np.max(ends - starts)) if len(starts) else 0


def main() -> None:
    required = [MANIFEST_PATH, ARCHIVE_PATH, RECEIPT_PATH, TSF_PATH]
    missing_files = [str(path) for path in required if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Required Electricity Hourly files are missing: {missing_files}")

    checks: list[dict[str, object]] = []

    def record_check(name: str, passed: bool, detail: object) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    manifest = pd.read_csv(MANIFEST_PATH)
    manifest_rows = manifest.loc[manifest["dataset_id"] == DATASET_ID]
    manifest_unique = len(manifest_rows) == 1
    record_check(
        "data_manifest_has_one_electricity_hourly_row",
        manifest_unique,
        f"matching_rows={len(manifest_rows)}",
    )
    if not manifest_unique:
        raise AssertionError("Electricity Hourly manifest row is not unique")
    manifest_row = manifest_rows.iloc[0]
    manifest_metadata_valid = bool(
        str(manifest_row["doi"]) == EXPECTED_DOI
        and str(manifest_row["license"]) == EXPECTED_LICENSE
        and int(manifest_row["series_count"]) == EXPECTED_SERIES
        and str(manifest_row["md5"]) == EXPECTED_MD5
        and str(manifest_row["frequency"]) == EXPECTED_FREQUENCY
        and str(manifest_row["tier"]) == "core"
    )
    record_check(
        "manifest_source_metadata_valid",
        manifest_metadata_valid,
        (
            f"doi={manifest_row['doi']}; license={manifest_row['license']}; "
            f"series={manifest_row['series_count']}"
        ),
    )

    with RECEIPT_PATH.open("r", encoding="utf-8") as handle:
        receipt = json.load(handle)
    receipt_valid = bool(
        isinstance(receipt, list)
        and len(receipt) == 1
        and receipt[0].get("dataset_id") == DATASET_ID
        and receipt[0].get("doi") == EXPECTED_DOI
        and receipt[0].get("md5") == EXPECTED_MD5
        and Path(receipt[0].get("archive", "")).resolve() == ARCHIVE_PATH.resolve()
    )
    record_check(
        "independent_download_receipt_valid",
        receipt_valid,
        f"entries={len(receipt) if isinstance(receipt, list) else 'invalid'}",
    )

    observed_md5 = md5_file(ARCHIVE_PATH)
    archive_hash_valid = observed_md5 == EXPECTED_MD5
    record_check(
        "archive_md5_matches_zenodo_manifest",
        archive_hash_valid,
        f"expected={EXPECTED_MD5}; observed={observed_md5}",
    )

    data, metadata = read_tsf(TSF_PATH)
    metadata_valid = bool(
        metadata.get("source_encoding") == "utf-8"
        and metadata.get("relation") == "Electricity"
        and metadata.get("frequency") == EXPECTED_FREQUENCY
        and metadata.get("horizon") is None
        and metadata.get("missing") is False
        and metadata.get("equallength") is True
        and metadata.get("attributes")
        == [("series_name", "string"), ("start_timestamp", "date")]
    )
    record_check(
        "tsf_metadata_valid",
        metadata_valid,
        (
            f"encoding={metadata.get('source_encoding')}; frequency="
            f"{metadata.get('frequency')}; archive_horizon="
            f"{metadata.get('horizon', 'not declared')}"
        ),
    )

    series_ids = data["series_name"].astype(str).tolist()
    expected_ids = [f"T{index}" for index in range(1, EXPECTED_SERIES + 1)]
    series_identity_valid = bool(
        len(data) == EXPECTED_SERIES
        and data["series_name"].nunique() == EXPECTED_SERIES
        and series_ids == expected_ids
    )
    record_check(
        "all_321_series_ids_are_unique_and_ordered",
        series_identity_valid,
        (
            f"rows={len(data)}; unique={data['series_name'].nunique()}; "
            f"first={series_ids[0]}; last={series_ids[-1]}"
        ),
    )

    audit_records: list[dict[str, object]] = []
    all_values: list[np.ndarray] = []
    all_timestamps_valid = True
    for source_order, row in enumerate(data.itertuples(index=False), start=1):
        values = np.asarray(row.series_value, dtype=np.float64)
        start = pd.Timestamp(row.start_timestamp)
        end = start + pd.to_timedelta(len(values) - 1, unit="h")
        missing_count = int(np.isnan(values).sum())
        nonfinite_count = int((~np.isfinite(values)).sum())
        zero_count = int(np.sum(values == 0.0))
        negative_count = int(np.sum(values < 0.0))
        longest_zero_run = longest_true_run(values == 0.0)
        fractional_count = int(np.sum(np.abs(values - np.round(values)) > 1e-12))
        q1, q3 = np.quantile(values, [0.25, 0.75])
        standard_deviation = float(np.std(values, ddof=0))
        mean_squared_first_difference = float(np.mean(np.diff(values) ** 2))
        all_timestamps_valid = bool(
            all_timestamps_valid
            and start == pd.Timestamp(EXPECTED_START_TIMESTAMP)
            and end == pd.Timestamp(EXPECTED_END_TIMESTAMP)
            and start.tzinfo is None
            and end.tzinfo is None
        )
        audit_records.append(
            {
                "dataset_id": DATASET_ID,
                "series_id": str(row.series_name),
                "source_order": source_order,
                "start_timestamp": start,
                "end_timestamp": end,
                "length": int(len(values)),
                "missing_count": missing_count,
                "nonfinite_count": nonfinite_count,
                "zero_count": zero_count,
                "zero_fraction": float(zero_count / len(values)),
                "longest_zero_run": longest_zero_run,
                "negative_count": negative_count,
                "fractional_value_count": fractional_count,
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(q3 - q1),
                "standard_deviation": standard_deviation,
                "mean_squared_first_difference": mean_squared_first_difference,
            }
        )
        all_values.append(values)
    audit = pd.DataFrame(audit_records)
    concatenated = np.concatenate(all_values)

    observed_length_counts = {
        int(length): int(count)
        for length, count in audit["length"].value_counts().sort_index().items()
    }
    length_distribution_valid = bool(
        observed_length_counts == {EXPECTED_SERIES_LENGTH: EXPECTED_SERIES}
        and int(audit["length"].sum()) == EXPECTED_TOTAL_OBSERVATIONS
        and audit["length"].eq(EXPECTED_SERIES_LENGTH).all()
    )
    record_check(
        "length_distribution_and_total_observations_valid",
        length_distribution_valid,
        (
            f"length_counts={observed_length_counts}; total="
            f"{int(audit['length'].sum())}"
        ),
    )

    no_missing_or_nonfinite = bool(
        int(audit["missing_count"].sum()) == 0
        and int(audit["nonfinite_count"].sum()) == 0
        and np.isfinite(concatenated).all()
    )
    record_check(
        "no_missing_or_nonfinite_values",
        no_missing_or_nonfinite,
        (
            f"missing={int(audit['missing_count'].sum())}; nonfinite="
            f"{int(audit['nonfinite_count'].sum())}"
        ),
    )

    value_range_valid = bool(
        float(np.min(concatenated)) == EXPECTED_GLOBAL_MINIMUM
        and float(np.max(concatenated)) == EXPECTED_GLOBAL_MAXIMUM
        and int(audit["negative_count"].sum()) == 0
        and int(audit["fractional_value_count"].sum()) == 0
    )
    record_check(
        "nonnegative_integer_value_range_matches_snapshot",
        value_range_valid,
        (
            f"range={float(np.min(concatenated))}.."
            f"{float(np.max(concatenated))}; zeros="
            f"{int(audit['zero_count'].sum())}"
        ),
    )

    zero_structure_valid = bool(
        int(audit["zero_count"].sum()) == EXPECTED_TOTAL_ZEROS
        and int(audit["zero_count"].gt(0).sum()) == EXPECTED_SERIES_WITH_ZEROS
        and int(audit["longest_zero_run"].max()) == EXPECTED_LONGEST_ZERO_RUN
        and str(
            audit.loc[audit["longest_zero_run"].idxmax(), "series_id"]
        )
        == "T183"
    )
    record_check(
        "observed_zero_structure_is_documented_not_treated_as_missing",
        zero_structure_valid,
        (
            f"zeros={int(audit['zero_count'].sum())}; series_with_zeros="
            f"{int(audit['zero_count'].gt(0).sum())}; longest_run="
            f"{int(audit['longest_zero_run'].max())}"
        ),
    )

    record_check(
        "timestamps_use_strict_hyphen_time_parser_and_cover_three_years",
        all_timestamps_valid,
        (
            f"start_range={audit['start_timestamp'].min()}.."
            f"{audit['start_timestamp'].max()}; end_range="
            f"{audit['end_timestamp'].min()}..{audit['end_timestamp'].max()}"
        ),
    )

    scale_safeguards_valid = bool(
        int(audit["standard_deviation"].le(0.0).sum()) == 0
        and int(audit["mean_squared_first_difference"].le(0.0).sum()) == 0
        and int(audit["iqr"].le(0.0).sum()) == 1
        and audit.loc[audit["iqr"].le(0.0), "series_id"].tolist() == ["T183"]
    )
    record_check(
        "all_series_have_valid_rmsse_scale_and_one_iqr_fallback_is_flagged",
        scale_safeguards_valid,
        (
            f"constant_series={int(audit['standard_deviation'].le(0.0).sum())}; "
            f"zero_first_difference_scale="
            f"{int(audit['mean_squared_first_difference'].le(0.0).sum())}; "
            f"zero_iqr_series={audit.loc[audit['iqr'].le(0.0), 'series_id'].tolist()}"
        ),
    )

    protocol_capacity_valid = bool(
        metadata.get("horizon") is None
        and audit["length"].ge(4 * MAXIMUM_CANDIDATE_WINDOW).all()
    )
    record_check(
        "archive_horizon_absent_and_history_supports_registered_window",
        protocol_capacity_valid,
        (
            "archive_horizon=not declared; maximum_candidate_window="
            f"{MAXIMUM_CANDIDATE_WINDOW}; series_length={EXPECTED_SERIES_LENGTH}"
        ),
    )

    record_check(
        "audit_does_not_fit_models_or_measure_forecast_performance",
        True,
        "raw values parsed for source/quality audit only; no split, fit, tuning, or metric",
    )

    failed_checks = [item for item in checks if not item["passed"]]
    if failed_checks:
        raise AssertionError(f"Electricity Hourly data audit failed: {failed_checks}")

    audit.to_csv(SERIES_AUDIT_PATH, index=False)
    pd.DataFrame(checks).to_csv(CHECKS_PATH, index=False)

    summary = {
        "dataset_id": DATASET_ID,
        "data_quality_audit_passed": True,
        "source": {
            "doi": EXPECTED_DOI,
            "license": EXPECTED_LICENSE,
            "archive_file": str(ARCHIVE_PATH.relative_to(PROJECT_ROOT)),
            "archive_md5": observed_md5,
            "tsf_file": str(TSF_PATH.relative_to(PROJECT_ROOT)),
            "tsf_sha256": sha256_file(TSF_PATH),
            "source_encoding": metadata["source_encoding"],
        },
        "metadata": {
            "relation": metadata["relation"],
            "frequency": metadata["frequency"],
            "archive_declared_forecast_horizon_hours": None,
            "archive_horizon_is_absent": True,
            "archive_missing_flag": bool(metadata["missing"]),
            "archive_equal_length_flag": bool(metadata["equallength"]),
        },
        "series_count": int(len(audit)),
        "total_observations": int(audit["length"].sum()),
        "length_counts": {
            str(key): value for key, value in observed_length_counts.items()
        },
        "length_minimum": int(audit["length"].min()),
        "length_median": float(audit["length"].median()),
        "length_maximum": int(audit["length"].max()),
        "total_missing_values": int(audit["missing_count"].sum()),
        "total_nonfinite_values": int(audit["nonfinite_count"].sum()),
        "total_zero_values": int(audit["zero_count"].sum()),
        "series_with_zero_values": int(audit["zero_count"].gt(0).sum()),
        "maximum_consecutive_zero_run": int(audit["longest_zero_run"].max()),
        "maximum_zero_run_series_id": str(
            audit.loc[audit["longest_zero_run"].idxmax(), "series_id"]
        ),
        "zero_iqr_series_ids_requiring_scaler_fallback": audit.loc[
            audit["iqr"].le(0.0), "series_id"
        ].tolist(),
        "global_minimum": float(np.min(concatenated)),
        "global_maximum": float(np.max(concatenated)),
        "start_timestamp_minimum": str(audit["start_timestamp"].min()),
        "start_timestamp_maximum": str(audit["start_timestamp"].max()),
        "end_timestamp_minimum": str(audit["end_timestamp"].min()),
        "end_timestamp_maximum": str(audit["end_timestamp"].max()),
        "project_time_split_created_now": False,
        "formal_test_performance_calculated": False,
        "test_values_used_for_model_fitting_or_tuning": False,
    }
    SUMMARY_PATH.write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    length_counts_frame = audit["length"].value_counts().sort_index()
    axes[0, 0].bar(
        length_counts_frame.index.astype(str),
        length_counts_frame.values,
        color="#4c78a8",
    )
    for position, value in enumerate(length_counts_frame.values):
        axes[0, 0].text(position, value + 5, str(value), ha="center")
    axes[0, 0].set_xlabel("Series length (hours)")
    axes[0, 0].set_ylabel("Series count")
    axes[0, 0].set_title("All 321 series have equal length")

    for index in range(3):
        values = np.asarray(data.iloc[index]["series_value"], dtype=np.float64)
        tail_count = min(672, len(values))
        relative_hours = np.arange(-tail_count, 0)
        axes[0, 1].plot(
            relative_hours,
            robust_scale_for_audit(values[-tail_count:]),
            label=str(data.iloc[index]["series_name"]),
            linewidth=1.1,
        )
    axes[0, 1].axvline(
        -MAXIMUM_CANDIDATE_WINDOW,
        color="black",
        linestyle="--",
        label="168-hour candidate window",
    )
    axes[0, 1].set_xlabel("Hours relative to series end")
    axes[0, 1].set_ylabel("Audit-only robust-scaled value")
    axes[0, 1].set_title("First three series; candidate window marked")
    axes[0, 1].legend()

    largest_zero_counts = audit.nlargest(12, "zero_count").sort_values(
        "zero_count"
    )
    axes[1, 0].barh(
        largest_zero_counts["series_id"],
        largest_zero_counts["zero_count"],
        color="#59a14f",
    )
    axes[1, 0].set_xlabel("Observed zero count")
    axes[1, 0].set_ylabel("Series ID")
    axes[1, 0].set_title("Series with the most observed zeros")

    scatter = axes[1, 1].scatter(
        audit["median"],
        audit["standard_deviation"],
        c=np.log1p(audit["zero_count"]),
        cmap="viridis",
        alpha=0.75,
        s=24,
    )
    axes[1, 1].set_xscale("symlog", linthresh=1.0)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Per-series median (symmetric log scale)")
    axes[1, 1].set_ylabel("Per-series standard deviation (log scale)")
    axes[1, 1].set_title("Scale and zero heterogeneity across 321 series")
    colorbar = figure.colorbar(scatter, ax=axes[1, 1])
    colorbar.set_label("log(1 + zero count)")

    figure.suptitle("Electricity Hourly source and data-quality audit", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)

    report = {
        "status": "passed",
        "dataset_id": DATASET_ID,
        "check_count": len(checks),
        "failed_check_count": 0,
        "series_count": int(len(audit)),
        "total_observations": int(audit["length"].sum()),
        "archive_declared_horizon_hours": None,
        "formal_test_performance_calculated": False,
        "models_fit_or_tuned": False,
        "outputs": {
            "series_audit": str(SERIES_AUDIT_PATH),
            "summary": str(SUMMARY_PATH),
            "checks": str(CHECKS_PATH),
            "figure": str(FIGURE_PATH),
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("Electricity Hourly 数据质量检查全部通过")
    print("读取编码：", metadata["source_encoding"])
    print("序列数量：", len(audit))
    print("总观测数量：", int(audit["length"].sum()))
    print("每条序列长度：", EXPECTED_SERIES_LENGTH, "小时")
    print(
        "时间范围：",
        f"{audit['start_timestamp'].min()} 至 {audit['end_timestamp'].max()}",
    )
    print("总缺失值数量：", int(audit["missing_count"].sum()))
    print(
        "总零值数量：",
        int(audit["zero_count"].sum()),
        f"（涉及{int(audit['zero_count'].gt(0).sum())}条序列）",
    )
    print(
        "最长连续零值：",
        int(audit["longest_zero_run"].max()),
        "小时，序列",
        str(audit.loc[audit["longest_zero_run"].idxmax(), "series_id"]),
    )
    print("零IQR回退序列：", audit.loc[audit["iqr"].le(0.0), "series_id"].tolist())
    print(
        "全局取值范围：",
        f"{float(np.min(concatenated)):g} 至 {float(np.max(concatenated)):g}",
    )
    print("档案是否声明官方预测范围：否")
    print("本步骤是否训练或调参：否")
    print("本步骤是否计算预测性能：否")
    print("质量检查表：", CHECKS_PATH)
    print("逐序列审计：", SERIES_AUDIT_PATH)
    print("质量摘要：", SUMMARY_PATH)
    print("质量检查图片：", FIGURE_PATH)
    print("审计报告：", REPORT_PATH)


if __name__ == "__main__":
    main()
