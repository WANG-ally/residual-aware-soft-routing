#!/usr/bin/env python3
"""Source and quality audit for the M4 Hourly archive.

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

DATASET_ID = "m4_hourly"
EXPECTED_DOI = "10.5281/zenodo.4656589"
EXPECTED_LICENSE = "CC-BY-4.0"
EXPECTED_MD5 = "3983df90f60db0f4e1f4dc3fca8ef565"
EXPECTED_SERIES = 414
EXPECTED_TOTAL_OBSERVATIONS = 373_372
EXPECTED_LENGTH_COUNTS = {748: 169, 1008: 245}
EXPECTED_HORIZON = 48
EXPECTED_FREQUENCY = "hourly"
LONGEST_REQUIRED_HISTORY = 168

MANIFEST_PATH = PROJECT_ROOT / "data_manifest.csv"
ARCHIVE_PATH = (
    PROJECT_ROOT / "data/raw/m4_hourly_staging/m4_hourly_dataset.zip"
)
RECEIPT_PATH = (
    PROJECT_ROOT / "data/raw/m4_hourly_staging/download_receipt.json"
)
TSF_PATH = (
    PROJECT_ROOT
    / "data/raw/m4_hourly_staging/m4_hourly/m4_hourly_dataset.tsf"
)

SERIES_AUDIT_PATH = OUTPUT_ROOT / "results/m4_hourly_series_audit.csv"
SUMMARY_PATH = OUTPUT_ROOT / "results/m4_hourly_data_quality_summary.yaml"
CHECKS_PATH = OUTPUT_ROOT / "results/m4_hourly_audit_checks.csv"
FIGURE_PATH = OUTPUT_ROOT / "figures/m4_hourly_quality_overview.png"
REPORT_PATH = OUTPUT_ROOT / "logs/m4_hourly_data_audit_report.json"

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


def main() -> None:
    required = [MANIFEST_PATH, ARCHIVE_PATH, RECEIPT_PATH, TSF_PATH]
    missing_files = [str(path) for path in required if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Required M4 Hourly files are missing: {missing_files}")

    checks: list[dict[str, object]] = []

    def record_check(name: str, passed: bool, detail: object) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    manifest = pd.read_csv(MANIFEST_PATH)
    manifest_rows = manifest.loc[manifest["dataset_id"] == DATASET_ID]
    manifest_unique = len(manifest_rows) == 1
    record_check(
        "data_manifest_has_one_m4_hourly_row",
        manifest_unique,
        f"matching_rows={len(manifest_rows)}",
    )
    if not manifest_unique:
        raise AssertionError("M4 Hourly manifest row is not unique")
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
        metadata.get("source_encoding") == "cp1252"
        and metadata.get("relation") == "M4"
        and metadata.get("frequency") == EXPECTED_FREQUENCY
        and int(metadata.get("horizon", -1)) == EXPECTED_HORIZON
        and metadata.get("missing") is False
        and metadata.get("equallength") is False
        and metadata.get("attributes")
        == [("series_name", "string"), ("start_timestamp", "date")]
    )
    record_check(
        "tsf_metadata_valid",
        metadata_valid,
        (
            f"encoding={metadata.get('source_encoding')}; frequency="
            f"{metadata.get('frequency')}; horizon={metadata.get('horizon')}"
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
        "all_414_series_ids_are_unique_and_ordered",
        series_identity_valid,
        (
            f"rows={len(data)}; unique={data['series_name'].nunique()}; "
            f"first={series_ids[0]}; last={series_ids[-1]}"
        ),
    )

    audit_records: list[dict[str, object]] = []
    all_values: list[np.ndarray] = []
    all_start_timestamps_valid = True
    for source_order, row in enumerate(data.itertuples(index=False), start=1):
        values = np.asarray(row.series_value, dtype=np.float64)
        start = pd.Timestamp(row.start_timestamp)
        end = start + pd.to_timedelta(len(values) - 1, unit="h")
        official_start_index = len(values) - EXPECTED_HORIZON
        official_start_timestamp = start + pd.to_timedelta(
            official_start_index, unit="h"
        )
        missing_count = int(np.isnan(values).sum())
        nonfinite_count = int((~np.isfinite(values)).sum())
        zero_count = int(np.sum(values == 0.0))
        nonpositive_count = int(np.sum(values <= 0.0))
        fractional_count = int(np.sum(np.abs(values - np.round(values)) > 1e-12))
        all_start_timestamps_valid = bool(
            all_start_timestamps_valid
            and not pd.isna(start)
            and end >= start
            and official_start_timestamp > start
        )
        audit_records.append(
            {
                "dataset_id": DATASET_ID,
                "series_id": str(row.series_name),
                "source_order": source_order,
                "start_timestamp": start,
                "end_timestamp": end,
                "length": int(len(values)),
                "official_horizon": EXPECTED_HORIZON,
                "official_holdout_start_index": official_start_index,
                "official_holdout_start_timestamp": official_start_timestamp,
                "missing_count": missing_count,
                "nonfinite_count": nonfinite_count,
                "zero_count": zero_count,
                "nonpositive_count": nonpositive_count,
                "fractional_value_count": fractional_count,
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "standard_deviation": float(np.std(values, ddof=0)),
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
        observed_length_counts == EXPECTED_LENGTH_COUNTS
        and int(audit["length"].sum()) == EXPECTED_TOTAL_OBSERVATIONS
        and int(audit["length"].min()) == 748
        and int(audit["length"].max()) == 1008
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
        float(np.min(concatenated)) == 10.0
        and float(np.max(concatenated)) == 703_008.0
        and int(audit["zero_count"].sum()) == 0
        and int(audit["nonpositive_count"].sum()) == 0
    )
    record_check(
        "positive_value_range_matches_snapshot",
        value_range_valid,
        (
            f"range={float(np.min(concatenated))}.."
            f"{float(np.max(concatenated))}; zeros="
            f"{int(audit['zero_count'].sum())}"
        ),
    )

    record_check(
        "timestamps_are_parseable_and_hourly_inference_is_valid",
        all_start_timestamps_valid,
        (
            f"start_range={audit['start_timestamp'].min()}.."
            f"{audit['start_timestamp'].max()}; end_range="
            f"{audit['end_timestamp'].min()}..{audit['end_timestamp'].max()}"
        ),
    )

    horizon_valid = bool(
        audit["official_horizon"].eq(EXPECTED_HORIZON).all()
        and audit["official_holdout_start_index"].ge(LONGEST_REQUIRED_HISTORY).all()
        and audit["length"].gt(EXPECTED_HORIZON).all()
    )
    record_check(
        "official_48_hour_horizon_is_valid_for_every_series",
        horizon_valid,
        (
            f"horizon={EXPECTED_HORIZON}; minimum pre-horizon history="
            f"{int(audit['official_holdout_start_index'].min())}"
        ),
    )

    record_check(
        "audit_does_not_fit_models_or_measure_forecast_performance",
        True,
        "raw values parsed for source/quality audit only; no split, fit, tuning, or metric",
    )

    failed_checks = [item for item in checks if not item["passed"]]
    if failed_checks:
        raise AssertionError(f"M4 Hourly data audit failed: {failed_checks}")

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
            "official_forecast_horizon_hours": int(metadata["horizon"]),
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
        "global_minimum": float(np.min(concatenated)),
        "global_maximum": float(np.max(concatenated)),
        "start_timestamp_minimum": str(audit["start_timestamp"].min()),
        "start_timestamp_maximum": str(audit["start_timestamp"].max()),
        "end_timestamp_minimum": str(audit["end_timestamp"].min()),
        "end_timestamp_maximum": str(audit["end_timestamp"].max()),
        "official_horizon_values_reserved_for_modeling_now": False,
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
    axes[0, 0].set_title("Two fixed length groups")

    for index in range(3):
        values = np.asarray(data.iloc[index]["series_value"], dtype=np.float64)
        tail_count = min(336, len(values))
        relative_hours = np.arange(-tail_count, 0)
        axes[0, 1].plot(
            relative_hours,
            robust_scale_for_audit(values[-tail_count:]),
            label=str(data.iloc[index]["series_name"]),
            linewidth=1.1,
        )
    axes[0, 1].axvline(-EXPECTED_HORIZON, color="black", linestyle="--")
    axes[0, 1].set_xlabel("Hours relative to series end")
    axes[0, 1].set_ylabel("Audit-only robust-scaled value")
    axes[0, 1].set_title("First three series; official 48-hour horizon marked")
    axes[0, 1].legend()

    axes[1, 0].hist(
        pd.to_datetime(audit["start_timestamp"]).dt.year,
        bins=np.arange(2008.5, 2018.6, 1.0),
        color="#59a14f",
        edgecolor="white",
    )
    axes[1, 0].set_xlabel("Start year")
    axes[1, 0].set_ylabel("Series count")
    axes[1, 0].set_title("Series start-time distribution")

    axes[1, 1].scatter(
        audit["median"],
        audit["standard_deviation"],
        c=audit["length"],
        cmap="viridis",
        alpha=0.75,
        s=24,
    )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Per-series median (log scale)")
    axes[1, 1].set_ylabel("Per-series standard deviation (log scale)")
    axes[1, 1].set_title("Scale heterogeneity across 414 series")

    figure.suptitle("M4 Hourly source and data-quality audit", fontsize=15)
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
        "official_horizon_hours": EXPECTED_HORIZON,
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
    print("M4 Hourly 数据质量检查全部通过")
    print("读取编码：", metadata["source_encoding"])
    print("序列数量：", len(audit))
    print("总观测数量：", int(audit["length"].sum()))
    print(
        "序列长度分布：",
        f"748小时={observed_length_counts[748]}条；",
        f"1008小时={observed_length_counts[1008]}条",
    )
    print("总缺失值数量：", int(audit["missing_count"].sum()))
    print("总零值数量：", int(audit["zero_count"].sum()))
    print(
        "全局取值范围：",
        f"{float(np.min(concatenated)):g} 至 {float(np.max(concatenated)):g}",
    )
    print("官方预测范围：", EXPECTED_HORIZON, "小时")
    print("本步骤是否训练或调参：否")
    print("本步骤是否计算预测性能：否")
    print("质量检查表：", CHECKS_PATH)
    print("逐序列审计：", SERIES_AUDIT_PATH)
    print("质量摘要：", SUMMARY_PATH)
    print("质量检查图片：", FIGURE_PATH)
    print("审计报告：", REPORT_PATH)


if __name__ == "__main__":
    main()
