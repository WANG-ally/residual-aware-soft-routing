#!/usr/bin/env python3
"""Source and quality audit for the Weather Daily archive.

This module audits all 3,010 archived series before any fixed 500-series
subsample is registered. It does not split time, fit models, select parameters,
open a formal test set, or calculate forecast performance."""

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


DATASET_ID = "weather_daily"
EXPECTED_DOI = "10.5281/zenodo.4654822"
EXPECTED_LICENSE = "CC-BY-4.0"
EXPECTED_MD5 = "57155594af0883ccd5e63a5948976796"
EXPECTED_SERIES = 3_010
EXPECTED_TOTAL_OBSERVATIONS = 43_032_000
EXPECTED_TYPE_COUNTS = {
    "maxtemp": 746,
    "mintemp": 748,
    "rain": 729,
    "solar": 787,
}
EXPECTED_MINIMUM_LENGTH = 1_332
EXPECTED_MEDIAN_LENGTH = 10_828.0
EXPECTED_MAXIMUM_LENGTH = 65_981
EXPECTED_TOTAL_ZEROS = 9_648_982
EXPECTED_SERIES_WITH_ZEROS = 1_279
EXPECTED_TOTAL_NEGATIVES = 373_016
EXPECTED_GLOBAL_MINIMUM = -41.8
EXPECTED_GLOBAL_MAXIMUM = 707.8
EXPECTED_LONGEST_ZERO_RUN = 336
EXPECTED_LONGEST_ZERO_RUN_SERIES = "T709"
EXPECTED_ZERO_IQR_SERIES = 182
EXPECTED_FREQUENCY = "daily"
MAXIMUM_CANDIDATE_WINDOW = 56
FUTURE_REGISTERED_SAMPLE_SIZE = 500

MANIFEST_PATH = PROJECT_ROOT / "data_manifest.csv"
ARCHIVE_PATH = PROJECT_ROOT / "data/raw/weather_daily_staging/weather_dataset.zip"
RECEIPT_PATH = PROJECT_ROOT / "data/raw/weather_daily_staging/download_receipt.json"
TSF_PATH = (
    PROJECT_ROOT
    / "data/raw/weather_daily_staging/weather_daily/weather_dataset.tsf"
)

SERIES_AUDIT_PATH = OUTPUT_ROOT / "results/weather_daily_series_audit.csv"
SUMMARY_PATH = OUTPUT_ROOT / "results/weather_daily_data_quality_summary.yaml"
CHECKS_PATH = OUTPUT_ROOT / "results/weather_daily_audit_checks.csv"
FIGURE_PATH = OUTPUT_ROOT / "figures/weather_daily_quality_overview.png"
REPORT_PATH = OUTPUT_ROOT / "logs/weather_daily_data_audit_report.json"

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


def longest_true_run(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return int(np.max(ends - starts)) if len(starts) else 0


def robust_scale_for_audit(values: np.ndarray) -> np.ndarray:
    """Scale plotted values only; these parameters are never used for modeling."""
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
        raise FileNotFoundError(f"Required Weather Daily files are missing: {missing_files}")

    checks: list[dict[str, object]] = []

    def record_check(name: str, passed: bool, detail: object) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    manifest = pd.read_csv(MANIFEST_PATH)
    manifest_rows = manifest.loc[manifest["dataset_id"] == DATASET_ID]
    manifest_unique = len(manifest_rows) == 1
    record_check(
        "data_manifest_has_one_weather_daily_row",
        manifest_unique,
        f"matching_rows={len(manifest_rows)}",
    )
    if not manifest_unique:
        raise AssertionError("Weather Daily manifest row is not unique")
    manifest_row = manifest_rows.iloc[0]
    manifest_metadata_valid = bool(
        str(manifest_row["doi"]) == EXPECTED_DOI
        and str(manifest_row["license"]) == EXPECTED_LICENSE
        and int(manifest_row["series_count"]) == EXPECTED_SERIES
        and str(manifest_row["md5"]) == EXPECTED_MD5
        and str(manifest_row["frequency"]) == EXPECTED_FREQUENCY
        and str(manifest_row["tier"]) == "core"
        and "500-series sample" in str(manifest_row["notes"])
    )
    record_check(
        "manifest_source_and_compute_protocol_valid",
        manifest_metadata_valid,
        (
            f"doi={manifest_row['doi']}; series={manifest_row['series_count']}; "
            f"notes={manifest_row['notes']}"
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
    record_check(
        "archive_md5_matches_zenodo_manifest",
        observed_md5 == EXPECTED_MD5,
        f"expected={EXPECTED_MD5}; observed={observed_md5}",
    )

    data, metadata = read_tsf(TSF_PATH)
    metadata_valid = bool(
        metadata.get("source_encoding") == "utf-8"
        and metadata.get("relation") == "Weather"
        and metadata.get("frequency") == EXPECTED_FREQUENCY
        and metadata.get("horizon") is None
        and metadata.get("missing") is False
        and metadata.get("equallength") is False
        and metadata.get("attributes")
        == [("series_name", "string"), ("series_type", "string")]
    )
    record_check(
        "tsf_metadata_valid",
        metadata_valid,
        (
            f"encoding={metadata.get('source_encoding')}; frequency="
            f"{metadata.get('frequency')}; horizon={metadata.get('horizon', 'not declared')}; "
            f"equal_length={metadata.get('equallength')}"
        ),
    )

    series_ids = data["series_name"].astype(str).tolist()
    expected_ids = [f"T{index}" for index in range(1, EXPECTED_SERIES + 1)]
    identity_valid = bool(
        len(data) == EXPECTED_SERIES
        and data["series_name"].nunique() == EXPECTED_SERIES
        and series_ids == expected_ids
    )
    record_check(
        "all_3010_series_ids_are_unique_and_ordered",
        identity_valid,
        f"rows={len(data)}; unique={data['series_name'].nunique()}",
    )

    observed_type_counts = {
        str(key): int(value)
        for key, value in data["series_type"].value_counts().sort_index().items()
    }
    record_check(
        "four_weather_variable_type_counts_match_snapshot",
        observed_type_counts == EXPECTED_TYPE_COUNTS,
        observed_type_counts,
    )

    audit_records: list[dict[str, object]] = []
    global_minimum = np.inf
    global_maximum = -np.inf
    for source_order, row in enumerate(data.itertuples(index=False), start=1):
        values = np.asarray(row.series_value, dtype=np.float64)
        finite = values[np.isfinite(values)]
        missing_count = int(np.isnan(values).sum())
        nonfinite_count = int((~np.isfinite(values)).sum())
        zero_count = int(np.sum(values == 0.0))
        negative_count = int(np.sum(values < 0.0))
        fractional_count = int(
            np.sum(np.abs(values - np.round(values)) > 1e-12)
        )
        q1, q3 = np.quantile(finite, [0.25, 0.75])
        standard_deviation = float(np.std(finite, ddof=0))
        mean_squared_first_difference = float(np.mean(np.diff(finite) ** 2))
        series_minimum = float(np.min(finite))
        series_maximum = float(np.max(finite))
        global_minimum = min(global_minimum, series_minimum)
        global_maximum = max(global_maximum, series_maximum)
        audit_records.append(
            {
                "dataset_id": DATASET_ID,
                "series_id": str(row.series_name),
                "series_type": str(row.series_type),
                "source_order": source_order,
                "length": int(len(values)),
                "missing_count": missing_count,
                "nonfinite_count": nonfinite_count,
                "zero_count": zero_count,
                "zero_fraction": float(zero_count / len(values)),
                "longest_zero_run": longest_true_run(values == 0.0),
                "negative_count": negative_count,
                "fractional_value_count": fractional_count,
                "minimum": series_minimum,
                "maximum": series_maximum,
                "mean": float(np.mean(finite)),
                "median": float(np.median(finite)),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(q3 - q1),
                "standard_deviation": standard_deviation,
                "mean_squared_first_difference": mean_squared_first_difference,
            }
        )
    audit = pd.DataFrame(audit_records)

    length_valid = bool(
        int(audit["length"].sum()) == EXPECTED_TOTAL_OBSERVATIONS
        and int(audit["length"].min()) == EXPECTED_MINIMUM_LENGTH
        and float(audit["length"].median()) == EXPECTED_MEDIAN_LENGTH
        and int(audit["length"].max()) == EXPECTED_MAXIMUM_LENGTH
    )
    record_check(
        "unequal_length_distribution_and_total_observations_valid",
        length_valid,
        (
            f"total={int(audit['length'].sum())}; min={int(audit['length'].min())}; "
            f"median={audit['length'].median():g}; max={int(audit['length'].max())}"
        ),
    )

    no_missing_or_nonfinite = bool(
        int(audit["missing_count"].sum()) == 0
        and int(audit["nonfinite_count"].sum()) == 0
    )
    record_check(
        "no_missing_or_nonfinite_values",
        no_missing_or_nonfinite,
        (
            f"missing={int(audit['missing_count'].sum())}; "
            f"nonfinite={int(audit['nonfinite_count'].sum())}"
        ),
    )

    value_structure_valid = bool(
        np.isclose(global_minimum, EXPECTED_GLOBAL_MINIMUM)
        and np.isclose(global_maximum, EXPECTED_GLOBAL_MAXIMUM)
        and int(audit["zero_count"].sum()) == EXPECTED_TOTAL_ZEROS
        and int(audit["zero_count"].gt(0).sum()) == EXPECTED_SERIES_WITH_ZEROS
        and int(audit["negative_count"].sum()) == EXPECTED_TOTAL_NEGATIVES
    )
    record_check(
        "weather_value_range_zeros_and_negative_temperatures_documented",
        value_structure_valid,
        (
            f"range={global_minimum:g}..{global_maximum:g}; "
            f"zeros={int(audit['zero_count'].sum())}; "
            f"negatives={int(audit['negative_count'].sum())}"
        ),
    )

    longest_row = audit.loc[audit["longest_zero_run"].idxmax()]
    zero_run_valid = bool(
        int(longest_row["longest_zero_run"]) == EXPECTED_LONGEST_ZERO_RUN
        and str(longest_row["series_id"]) == EXPECTED_LONGEST_ZERO_RUN_SERIES
        and str(longest_row["series_type"]) == "rain"
    )
    record_check(
        "longest_observed_zero_run_is_documented_not_treated_as_missing",
        zero_run_valid,
        (
            f"series={longest_row['series_id']}; type={longest_row['series_type']}; "
            f"days={int(longest_row['longest_zero_run'])}"
        ),
    )

    scale_safeguards_valid = bool(
        int(audit["standard_deviation"].le(0.0).sum()) == 0
        and int(audit["mean_squared_first_difference"].le(0.0).sum()) == 0
        and int(audit["iqr"].le(0.0).sum()) == EXPECTED_ZERO_IQR_SERIES
    )
    record_check(
        "all_series_have_valid_rmsse_scale_and_iqr_fallbacks_are_flagged",
        scale_safeguards_valid,
        (
            f"constant={int(audit['standard_deviation'].le(0.0).sum())}; "
            f"zero_difference_scale={int(audit['mean_squared_first_difference'].le(0.0).sum())}; "
            f"zero_iqr={int(audit['iqr'].le(0.0).sum())}"
        ),
    )

    protocol_capacity_valid = bool(
        metadata.get("horizon") is None
        and audit["length"].ge(4 * MAXIMUM_CANDIDATE_WINDOW).all()
    )
    record_check(
        "archive_horizon_absent_and_all_histories_support_registered_window",
        protocol_capacity_valid,
        (
            f"archive_horizon=not declared; maximum_candidate_window="
            f"{MAXIMUM_CANDIDATE_WINDOW}; minimum_length={int(audit['length'].min())}"
        ),
    )

    record_check(
        "full_archive_audited_before_fixed_500_series_registration",
        len(audit) == EXPECTED_SERIES,
        (
            f"audited={len(audit)}; future_registered_sample="
            f"{FUTURE_REGISTERED_SAMPLE_SIZE}; sample_created_now=False"
        ),
    )
    record_check(
        "audit_does_not_split_fit_tune_or_measure_forecast_performance",
        True,
        "source/quality audit only; no sampling, split, fit, tuning, test, or metric",
    )

    failed_checks = [item for item in checks if not item["passed"]]
    if failed_checks:
        raise AssertionError(f"Weather Daily data audit failed: {failed_checks}")

    audit.to_csv(SERIES_AUDIT_PATH, index=False)
    pd.DataFrame(checks).to_csv(CHECKS_PATH, index=False)

    zero_iqr_by_type = {
        str(key): int(value)
        for key, value in (
            audit.loc[audit["iqr"].le(0.0), "series_type"]
            .value_counts()
            .sort_index()
            .items()
        )
    }
    length_by_type = {}
    for series_type, group in audit.groupby("series_type", sort=True):
        length_by_type[str(series_type)] = {
            "series_count": int(len(group)),
            "total_observations": int(group["length"].sum()),
            "minimum_length": int(group["length"].min()),
            "median_length": float(group["length"].median()),
            "maximum_length": int(group["length"].max()),
        }

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
            "archive_declared_forecast_horizon_days": None,
            "archive_horizon_is_absent": True,
            "archive_missing_flag": bool(metadata["missing"]),
            "archive_equal_length_flag": bool(metadata["equallength"]),
            "series_types": list(EXPECTED_TYPE_COUNTS),
        },
        "series_count": int(len(audit)),
        "series_type_counts": observed_type_counts,
        "total_observations": int(audit["length"].sum()),
        "length_minimum": int(audit["length"].min()),
        "length_median": float(audit["length"].median()),
        "length_maximum": int(audit["length"].max()),
        "length_summary_by_series_type": length_by_type,
        "total_missing_values": int(audit["missing_count"].sum()),
        "total_nonfinite_values": int(audit["nonfinite_count"].sum()),
        "total_zero_values": int(audit["zero_count"].sum()),
        "series_with_zero_values": int(audit["zero_count"].gt(0).sum()),
        "total_negative_values": int(audit["negative_count"].sum()),
        "maximum_consecutive_zero_run_days": int(longest_row["longest_zero_run"]),
        "maximum_zero_run_series_id": str(longest_row["series_id"]),
        "zero_iqr_series_count_requiring_scaler_fallback": int(
            audit["iqr"].le(0.0).sum()
        ),
        "zero_iqr_series_count_by_type": zero_iqr_by_type,
        "global_minimum": float(global_minimum),
        "global_maximum": float(global_maximum),
        "all_series_have_positive_rmsse_difference_scale": True,
        "full_archive_audited": True,
        "registered_500_series_sample_created_now": False,
        "project_time_split_created_now": False,
        "formal_test_performance_calculated": False,
        "test_values_used_for_model_fitting_or_tuning": False,
    }
    SUMMARY_PATH.write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    colors = {
        "rain": "#4c78a8",
        "mintemp": "#59a14f",
        "maxtemp": "#e15759",
        "solar": "#f28e2b",
    }
    log_bins = np.geomspace(
        audit["length"].min(), audit["length"].max(), num=35
    )
    for series_type in ("rain", "mintemp", "maxtemp", "solar"):
        values = audit.loc[audit["series_type"] == series_type, "length"]
        axes[0, 0].hist(
            values,
            bins=log_bins,
            alpha=0.45,
            label=series_type,
            color=colors[series_type],
        )
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("Series length (days; log scale)")
    axes[0, 0].set_ylabel("Series count")
    axes[0, 0].set_title("Unequal series-length distribution")
    axes[0, 0].legend()

    representative_types = ("rain", "mintemp", "maxtemp", "solar")
    representative_offsets = []
    for type_index, series_type in enumerate(representative_types):
        row = data.loc[data["series_type"] == series_type].iloc[0]
        values = np.asarray(row["series_value"], dtype=np.float64)
        tail_count = min(365, len(values))
        vertical_offset = float(type_index * 10)
        representative_offsets.append(vertical_offset)
        scaled_values = np.clip(
            robust_scale_for_audit(values[-tail_count:]), -4.0, 4.0
        )
        axes[0, 1].plot(
            np.arange(-tail_count, 0),
            scaled_values + vertical_offset,
            label=f"{row['series_name']} ({series_type})",
            color=colors[series_type],
            linewidth=1.0,
        )
    axes[0, 1].axvline(
        -MAXIMUM_CANDIDATE_WINDOW,
        color="black",
        linestyle="--",
        label="56-day candidate window",
    )
    axes[0, 1].set_xlabel("Days relative to series end")
    axes[0, 1].set_yticks(representative_offsets, representative_types)
    axes[0, 1].set_ylabel("Weather variable (vertically offset)")
    axes[0, 1].set_title("One representative series per weather variable")
    axes[0, 1].legend(fontsize=8)

    type_frame = audit["series_type"].value_counts().reindex(
        ["rain", "mintemp", "maxtemp", "solar"]
    )
    axes[1, 0].bar(
        type_frame.index,
        type_frame.values,
        color=[colors[item] for item in type_frame.index],
    )
    for position, value in enumerate(type_frame.values):
        axes[1, 0].text(position, value + 8, str(value), ha="center")
    axes[1, 0].set_xlabel("Weather variable")
    axes[1, 0].set_ylabel("Series count")
    axes[1, 0].set_title("All 3,010 archived series by variable")

    for series_type in ("rain", "mintemp", "maxtemp", "solar"):
        group = audit.loc[audit["series_type"] == series_type]
        axes[1, 1].scatter(
            group["length"],
            group["standard_deviation"],
            alpha=0.45,
            s=13,
            label=series_type,
            color=colors[series_type],
        )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Series length (days; log scale)")
    axes[1, 1].set_ylabel("Per-series standard deviation (log scale)")
    axes[1, 1].set_title("Length and scale heterogeneity")
    axes[1, 1].legend()

    figure.suptitle("Weather Daily source and data-quality audit", fontsize=15)
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
        "archive_declared_horizon_days": None,
        "full_archive_audited": True,
        "registered_500_series_sample_created_now": False,
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
    print("Weather Daily 数据质量检查全部通过")
    print("读取编码：", metadata["source_encoding"])
    print("序列数量：", len(audit))
    print("变量类型数量：", observed_type_counts)
    print("总观测数量：", int(audit["length"].sum()))
    print(
        "序列长度（最短/中位数/最长）：",
        f"{int(audit['length'].min())} / {audit['length'].median():g} / "
        f"{int(audit['length'].max())}",
    )
    print("总缺失值数量：", int(audit["missing_count"].sum()))
    print(
        "总零值数量：",
        int(audit["zero_count"].sum()),
        f"（涉及{int(audit['zero_count'].gt(0).sum())}条序列）",
    )
    print("负值数量：", int(audit["negative_count"].sum()), "（来自温度变量）")
    print(
        "最长连续零值：",
        int(longest_row["longest_zero_run"]),
        "天，序列",
        str(longest_row["series_id"]),
    )
    print("零IQR回退序列数量：", int(audit["iqr"].le(0.0).sum()))
    print("全局取值范围：", f"{global_minimum:g} 至 {global_maximum:g}")
    print("档案是否声明官方预测范围：否")
    print("本步骤是否创建500条固定样本：否")
    print("本步骤是否训练或调参：否")
    print("本步骤是否计算预测性能：否")
    print("质量检查表：", CHECKS_PATH)
    print("逐序列审计：", SERIES_AUDIT_PATH)
    print("质量摘要：", SUMMARY_PATH)
    print("质量检查图片：", FIGURE_PATH)
    print("审计报告：", REPORT_PATH)


if __name__ == "__main__":
    main()
