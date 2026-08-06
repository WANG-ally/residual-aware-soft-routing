#!/usr/bin/env python3
"""Leakage-safe four-stage split for Electricity Hourly.

This module applies the already frozen project-wide 60/15/10/15 chronological
protocol independently to every series.  It constructs partitions and checks
their integrity; it does not fit a model, tune a parameter, or calculate a
forecasting metric."""

from __future__ import annotations

import json
import os
import sys
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sci-routing")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pyarrow import parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(
    os.environ.get("SCI_ROUTING_ROOT", Path(__file__).resolve().parents[3])
).resolve()
OUTPUT_ROOT = Path(os.environ.get("SCI_ROUTING_OUTPUT_ROOT", PROJECT_ROOT)).resolve()

for candidate in (PROJECT_ROOT, SCRIPT_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tsf_reader_compat import read_tsf  # noqa: E402


DATASET_ID = "electricity_hourly"
SPLIT_NAMES = ["base_train", "router_train", "calibration", "test"]
COLORS = {
    "base_train": "#4C78A8",
    "router_train": "#59A14F",
    "calibration": "#F28E2B",
    "test": "#E15759",
}
EXPECTED_SERIES = 321
EXPECTED_SERIES_LENGTH = 26_304
EXPECTED_TOTAL_OBSERVATIONS = 8_443_584
EXPECTED_PER_SERIES_COUNTS = {
    "base_train": 15_782,
    "router_train": 3_946,
    "calibration": 2_630,
    "test": 3_946,
}
EXPECTED_FREQUENCY = "hourly"

DATA_PATH = (
    PROJECT_ROOT
    / "data/raw/electricity_hourly_staging/electricity_hourly/electricity_hourly_dataset.tsf"
)
CONFIG_PATH = PROJECT_ROOT / "experiment_config.yaml"
AUDIT_SUMMARY_PATH = PROJECT_ROOT / "results/electricity_hourly_data_quality_summary.yaml"
AUDIT_CHECKS_PATH = PROJECT_ROOT / "results/electricity_hourly_audit_checks.csv"

PROCESSED_PATH = OUTPUT_ROOT / "data/processed/electricity_hourly_long.parquet"
MANIFEST_PATH = OUTPUT_ROOT / "results/electricity_hourly_split_manifest.csv"
CHECKS_PATH = OUTPUT_ROOT / "results/electricity_hourly_split_checks.csv"
SUMMARY_PATH = OUTPUT_ROOT / "results/electricity_hourly_split_summary.yaml"
FIGURE_PATH = OUTPUT_ROOT / "figures/electricity_hourly_four_stage_split.png"
REPORT_PATH = OUTPUT_ROOT / "logs/electricity_hourly_split_report.json"


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def fraction_from_config(value: object) -> Fraction:
    """Convert through text so decimal ratios remain exact."""
    return Fraction(str(value))


def timestamp_text(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def all_passed(values: pd.Series) -> bool:
    normalized = values.astype(str).str.strip().str.lower()
    return bool(normalized.isin({"true", "1"}).all())


def robust_scale_from_base(values: np.ndarray, base_end: int) -> np.ndarray:
    base = np.asarray(values[:base_end], dtype=np.float64)
    median = float(np.median(base))
    q25, q75 = np.quantile(base, [0.25, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.std(base, ddof=0))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return (np.asarray(values, dtype=np.float64) - median) / scale


def plot_example(
    ax: plt.Axes,
    data_row: pd.Series,
    manifest_row: pd.Series,
) -> None:
    values = np.asarray(data_row["series_value"], dtype=np.float64)
    n = len(values)
    scaled = robust_scale_from_base(values, int(manifest_row["base_train_count"]))
    boundaries = [
        0,
        int(manifest_row["base_train_count"]),
        int(
            manifest_row["base_train_count"]
            + manifest_row["router_train_count"]
        ),
        int(
            manifest_row["base_train_count"]
            + manifest_row["router_train_count"]
            + manifest_row["calibration_count"]
        ),
        n,
    ]
    for split_number, split_name in enumerate(SPLIT_NAMES):
        left, right = boundaries[split_number], boundaries[split_number + 1]
        indices = np.arange(left, right, dtype=np.int32)
        ax.plot(
            indices,
            scaled[indices],
            color=COLORS[split_name],
            linewidth=0.85,
            label=split_name,
        )
    for boundary in boundaries[1:-1]:
        ax.axvline(boundary - 0.5, color="black", linestyle="--", linewidth=0.6)
    ax.set_title(f"{data_row['series_name']} (n={n})")
    ax.set_xlabel("Hourly time index")
    ax.set_ylabel("Base-train robust-scaled value")
    ax.grid(alpha=0.15)


def main() -> None:
    for path in (DATA_PATH, CONFIG_PATH, AUDIT_SUMMARY_PATH, AUDIT_CHECKS_PATH):
        require_file(path)
    for output_path in (
        PROCESSED_PATH,
        MANIFEST_PATH,
        CHECKS_PATH,
        SUMMARY_PATH,
        FIGURE_PATH,
        REPORT_PATH,
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    audit_summary = yaml.safe_load(AUDIT_SUMMARY_PATH.read_text(encoding="utf-8"))
    audit_checks = pd.read_csv(AUDIT_CHECKS_PATH)

    ratio_config = config["split"]["chronological_ratios"]
    if list(ratio_config.keys()) != SPLIT_NAMES:
        raise AssertionError(
            f"Split order must be {SPLIT_NAMES}, got {list(ratio_config.keys())}"
        )
    ratios = {name: fraction_from_config(ratio_config[name]) for name in SPLIT_NAMES}
    ratio_sum = sum(ratios.values(), Fraction(0, 1))
    if ratio_sum != Fraction(1, 1):
        raise AssertionError(f"Four split ratios must sum to 1, got {ratio_sum}")

    cumulative = {
        "base_train": ratios["base_train"],
        "router_train": ratios["base_train"] + ratios["router_train"],
        "calibration": (
            ratios["base_train"]
            + ratios["router_train"]
            + ratios["calibration"]
        ),
    }
    largest_window = max(
        int(value)
        for value in config["preprocessing"]["window_by_frequency"]["hourly"]
    )

    data, metadata = read_tsf(DATA_PATH)
    series_names = data["series_name"].astype(str).tolist()
    lengths = data["series_value"].map(len).to_numpy(dtype=np.int32)
    total_rows = int(lengths.sum())
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))

    value_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    timestamp_parts: list[np.ndarray] = []
    manifest_records: list[dict[str, object]] = []
    split_codes = np.empty(total_rows, dtype=np.int8)

    for series_number, row in data.iterrows():
        values = np.asarray(row["series_value"], dtype=np.float64)
        n = len(values)
        start = pd.Timestamp(row["start_timestamp"])
        cut_base = n * cumulative["base_train"].numerator // cumulative["base_train"].denominator
        cut_router = n * cumulative["router_train"].numerator // cumulative["router_train"].denominator
        cut_calibration = n * cumulative["calibration"].numerator // cumulative["calibration"].denominator
        cuts = [0, cut_base, cut_router, cut_calibration, n]
        counts = np.diff(cuts).astype(int)
        if np.any(counts <= 0):
            raise AssertionError(f"{row['series_name']} has an empty split: {counts}")

        global_left = int(offsets[series_number])
        for split_number in range(4):
            local_left, local_right = cuts[split_number], cuts[split_number + 1]
            split_codes[global_left + local_left : global_left + local_right] = split_number

        time_index = np.arange(n, dtype=np.int32)
        timestamps = (
            start.to_datetime64() + time_index.astype("timedelta64[h]")
        ).astype("datetime64[ns]")
        value_parts.append(values)
        index_parts.append(time_index)
        timestamp_parts.append(timestamps)

        record: dict[str, object] = {
            "dataset_id": DATASET_ID,
            "series_id": str(row["series_name"]),
            "total_length": n,
            "source_start_timestamp": timestamp_text(start),
            "source_end_timestamp": timestamp_text(timestamps[-1]),
        }
        for split_number, split_name in enumerate(SPLIT_NAMES):
            local_left, local_right = cuts[split_number], cuts[split_number + 1]
            count = local_right - local_left
            record[f"{split_name}_start"] = local_left
            record[f"{split_name}_end"] = local_right - 1
            record[f"{split_name}_count"] = count
            record[f"{split_name}_start_timestamp"] = timestamp_text(
                timestamps[local_left]
            )
            record[f"{split_name}_end_timestamp"] = timestamp_text(
                timestamps[local_right - 1]
            )
            record[f"{split_name}_actual_ratio"] = count / n
        manifest_records.append(record)

    values_all = np.concatenate(value_parts)
    time_index_all = np.concatenate(index_parts)
    timestamps_all = np.concatenate(timestamp_parts)
    series_codes = np.repeat(np.arange(len(data), dtype=np.int16), lengths)

    long_data = pd.DataFrame(
        {
            "dataset_id": pd.Categorical.from_codes(
                np.zeros(total_rows, dtype=np.int8), categories=[DATASET_ID]
            ),
            "series_id": pd.Categorical.from_codes(
                series_codes, categories=series_names
            ),
            "time_index": time_index_all,
            "timestamp": timestamps_all,
            "value": values_all,
            "split": pd.Categorical.from_codes(
                split_codes, categories=SPLIT_NAMES, ordered=True
            ),
        }
    )
    manifest = pd.DataFrame(manifest_records)

    per_series_index_ok = True
    per_series_hourly_ok = True
    per_series_order_ok = True
    per_series_counts_ok = True
    values_match_source = True
    for series_number, manifest_row in manifest.iterrows():
        left, right = int(offsets[series_number]), int(offsets[series_number + 1])
        n = right - left
        local_codes = split_codes[left:right]
        expected_counts = np.asarray(
            [manifest_row[f"{name}_count"] for name in SPLIT_NAMES], dtype=int
        )
        per_series_index_ok &= np.array_equal(
            time_index_all[left:right], np.arange(n, dtype=np.int32)
        )
        per_series_hourly_ok &= bool(
            np.all(np.diff(timestamps_all[left:right]) == np.timedelta64(1, "h"))
        )
        per_series_order_ok &= bool(
            np.all(np.diff(local_codes.astype(np.int16)) >= 0)
        )
        per_series_counts_ok &= bool(
            np.array_equal(np.bincount(local_codes, minlength=4), expected_counts)
            and expected_counts.sum() == n
        )
        values_match_source &= np.array_equal(
            values_all[left:right],
            np.asarray(data.iloc[series_number]["series_value"], dtype=np.float64),
        )

    aggregate_counts = {
        name: int(np.count_nonzero(split_codes == split_number))
        for split_number, name in enumerate(SPLIT_NAMES)
    }
    count_columns = [f"{name}_count" for name in SPLIT_NAMES]
    minimum_counts = {
        name: int(manifest[f"{name}_count"].min()) for name in SPLIT_NAMES
    }
    maximum_counts = {
        name: int(manifest[f"{name}_count"].max()) for name in SPLIT_NAMES
    }

    check_items: list[tuple[str, bool, str]] = [
        (
            "previous_data_audit_passed",
            bool(audit_summary.get("data_quality_audit_passed")),
            str(audit_summary.get("data_quality_audit_passed")),
        ),
        (
            "previous_audit_checks_all_passed",
            all_passed(audit_checks["passed"]),
            f"checks={len(audit_checks)}",
        ),
        ("split_ratio_sum_is_exactly_one", ratio_sum == 1, str(ratio_sum)),
        (
            "source_metadata_is_electricity_hourly_without_declared_horizon",
            str(metadata.get("frequency")).lower() == EXPECTED_FREQUENCY
            and str(metadata.get("relation")) == "Electricity"
            and metadata.get("equallength") is True
            and metadata.get("horizon") is None
            and audit_summary["metadata"]["archive_horizon_is_absent"] is True,
            f"frequency={metadata.get('frequency')}; horizon={metadata.get('horizon')}",
        ),
        (
            "source_size_matches_completed_audit",
            len(data) == EXPECTED_SERIES
            and total_rows == EXPECTED_TOTAL_OBSERVATIONS
            and bool(np.all(lengths == EXPECTED_SERIES_LENGTH))
            and int(audit_summary["series_count"]) == EXPECTED_SERIES
            and int(audit_summary["total_observations"]) == EXPECTED_TOTAL_OBSERVATIONS,
            f"series={len(data)}; rows={total_rows}",
        ),
        (
            "manifest_has_one_unique_row_per_series",
            len(manifest) == EXPECTED_SERIES and manifest["series_id"].is_unique,
            f"rows={len(manifest)}; unique={manifest['series_id'].nunique()}",
        ),
        (
            "all_four_splits_are_nonempty",
            bool((manifest[count_columns] > 0).all().all()),
            str(minimum_counts),
        ),
        (
            "equal_length_series_have_registered_exact_split_counts",
            all(
                manifest[f"{name}_count"].eq(expected).all()
                for name, expected in EXPECTED_PER_SERIES_COUNTS.items()
            ),
            str(EXPECTED_PER_SERIES_COUNTS),
        ),
        (
            "base_train_supports_largest_candidate_window",
            minimum_counts["base_train"] > largest_window,
            f"minimum_base={minimum_counts['base_train']}; largest_window={largest_window}",
        ),
        (
            "each_project_test_partition_exceeds_largest_candidate_window",
            minimum_counts["test"] > largest_window,
            f"minimum_test={minimum_counts['test']}; largest_window={largest_window}",
        ),
        ("per_series_time_index_is_contiguous", per_series_index_ok, str(per_series_index_ok)),
        ("per_series_timestamps_are_hourly", per_series_hourly_ok, str(per_series_hourly_ok)),
        ("per_series_split_order_is_monotone", per_series_order_ok, str(per_series_order_ok)),
        ("per_series_split_counts_match_manifest", per_series_counts_ok, str(per_series_counts_ok)),
        ("long_values_exactly_match_source_order", values_match_source, str(values_match_source)),
        (
            "aggregate_counts_cover_every_observation_once",
            sum(aggregate_counts.values()) == total_rows,
            f"aggregate={sum(aggregate_counts.values())}; source={total_rows}",
        ),
        (
            "long_table_has_no_missing_values",
            not long_data.isna().any().any(),
            f"missing_cells={int(long_data.isna().sum().sum())}",
        ),
        (
            "split_step_does_not_fit_or_evaluate_models",
            True,
            "partition construction only; no fit, tuning, prediction, or metric",
        ),
    ]
    checks = pd.DataFrame(check_items, columns=["check", "passed", "detail"])
    failed = checks.loc[~checks["passed"]]
    if not failed.empty:
        message = "; ".join(
            f"{row.check}: {row.detail}" for row in failed.itertuples(index=False)
        )
        raise AssertionError(f"Electricity Hourly split audit failed: {message}")

    long_data.to_parquet(PROCESSED_PATH, index=False, compression="snappy")
    parquet_rows = int(pq.ParquetFile(PROCESSED_PATH).metadata.num_rows)
    if parquet_rows != total_rows:
        raise AssertionError(
            f"Parquet row mismatch: expected {total_rows}, got {parquet_rows}"
        )
    manifest.to_csv(MANIFEST_PATH, index=False)
    checks.to_csv(CHECKS_PATH, index=False)

    summary = {
        "dataset_id": DATASET_ID,
        "split_passed": True,
        "rounding_rule": "floor each cumulative boundary; test receives remainder",
        "ratios": {name: float(ratios[name]) for name in SPLIT_NAMES},
        "series_count": int(len(manifest)),
        "total_observations": total_rows,
        "aggregate_counts": aggregate_counts,
        "per_series_count_minimum": minimum_counts,
        "per_series_count_median": {
            name: float(manifest[f"{name}_count"].median()) for name in SPLIT_NAMES
        },
        "per_series_count_maximum": maximum_counts,
        "largest_candidate_window": largest_window,
        "minimum_base_train_count": minimum_counts["base_train"],
        "archive_declared_forecast_horizon_hours": None,
        "project_test_definition": "final 15% under frozen cross-dataset protocol",
        "archive_horizon_note": (
            "The TSF archive does not declare a forecast horizon. The final 15 percent "
            "test partition follows the unchanged cross-dataset protocol and is not "
            "selected from observed forecasting performance."
        ),
        "time_leakage_checks_passed": True,
        "models_fitted": False,
        "parameters_tuned": False,
        "forecast_metrics_calculated": False,
        "test_values_used_for_training": False,
        "test_values_used_for_tuning": False,
    }
    SUMMARY_PATH.write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    length_group_rows = []
    for length in sorted(manifest["total_length"].unique()):
        row: dict[str, object] = {"length": int(length)}
        group = manifest.loc[manifest["total_length"] == length]
        for split_name in SPLIT_NAMES:
            row[split_name] = int(group[f"{split_name}_count"].iloc[0])
        length_group_rows.append(row)
    length_groups = pd.DataFrame(length_group_rows).set_index("length")

    fig, axes = plt.subplots(2, 2, figsize=(16, 9.5))
    bottom = np.zeros(len(length_groups), dtype=float)
    x = np.arange(len(length_groups))
    for split_name in SPLIT_NAMES:
        heights = length_groups[split_name].to_numpy(dtype=float)
        axes[0, 0].bar(
            x,
            heights,
            bottom=bottom,
            color=COLORS[split_name],
            label=split_name,
        )
        bottom += heights
    axes[0, 0].set_xticks(x, [str(value) for value in length_groups.index])
    axes[0, 0].set_xlabel("Original series length (hours)")
    axes[0, 0].set_ylabel("Hours per series")
    axes[0, 0].set_title("Exact split counts for all equal-length series")
    axes[0, 0].grid(axis="y", alpha=0.15)

    aggregate_percent = np.asarray(
        [aggregate_counts[name] / total_rows * 100.0 for name in SPLIT_NAMES]
    )
    bars = axes[0, 1].bar(
        SPLIT_NAMES,
        aggregate_percent,
        color=[COLORS[name] for name in SPLIT_NAMES],
    )
    axes[0, 1].bar_label(bars, fmt="%.2f%%", padding=2)
    axes[0, 1].set_ylim(0, 66)
    axes[0, 1].set_ylabel("Share of all observations (%)")
    axes[0, 1].set_title("Aggregate chronological allocation")
    axes[0, 1].tick_params(axis="x", rotation=15)
    axes[0, 1].grid(axis="y", alpha=0.15)

    series_to_index = {
        str(series_id): int(index)
        for index, series_id in enumerate(data["series_name"].astype(str))
    }
    example_indices = [series_to_index["T1"], series_to_index["T183"]]
    manifest_lookup = manifest.set_index("series_id")
    for ax, data_index in zip(axes[1], example_indices):
        data_row = data.iloc[data_index]
        series_id = str(data_row["series_name"])
        plot_example(ax, data_row, manifest_lookup.loc[series_id])

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(
        "Electricity Hourly: leakage-safe 60/15/10/15 chronological split",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = {
        "status": "passed",
        "dataset_id": DATASET_ID,
        "check_count": int(len(checks)),
        "failed_check_count": 0,
        "series_count": int(len(manifest)),
        "parquet_rows": parquet_rows,
        "aggregate_counts": aggregate_counts,
        "models_fitted": False,
        "forecast_metrics_calculated": False,
        "test_used_for_training_or_tuning": False,
        "outputs": {
            "processed_long_table": str(PROCESSED_PATH),
            "split_manifest": str(MANIFEST_PATH),
            "split_checks": str(CHECKS_PATH),
            "split_summary": str(SUMMARY_PATH),
            "split_figure": str(FIGURE_PATH),
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Electricity Hourly 四阶段时间切分全部通过")
    print("序列数量：", len(manifest))
    print("长表总行数：", total_rows)
    print("各数据段总观测数量：")
    for name in SPLIT_NAMES:
        print(f"  {name}: {aggregate_counts[name]}")
    print("各序列数据段计数范围（最少至最多）：")
    for name in SPLIT_NAMES:
        print(f"  {name}: {minimum_counts[name]} 至 {maximum_counts[name]}")
    print("最大候选窗口：", largest_window, "小时")
    print("最短序列的 base_train：", minimum_counts["base_train"], "小时")
    print("档案是否声明官方预测范围：否")
    print("项目测试定义：冻结协议中的最后15%，不是重新定义参数")
    print("防止时间泄漏检查：通过")
    print("测试值是否用于训练：否")
    print("测试值是否用于调参：否")
    print("处理后数据：", PROCESSED_PATH)
    print("切分清单：", MANIFEST_PATH)
    print("切分检查：", CHECKS_PATH)
    print("切分图片：", FIGURE_PATH)


if __name__ == "__main__":
    main()
