#!/usr/bin/env python3
"""Leakage-safe four-stage split for variable-length pedestrian series."""

from __future__ import annotations

import json
import os
import sys
from fractions import Fraction
from pathlib import Path

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


DATASET_ID = "pedestrian_hourly"
SPLIT_NAMES = ["base_train", "router_train", "calibration", "test"]
COLORS = {
    "base_train": "#4C78A8",
    "router_train": "#59A14F",
    "calibration": "#F28E2B",
    "test": "#E15759",
}

DATA_PATH = (
    PROJECT_ROOT
    / "data/raw/pedestrian_hourly_staging/pedestrian_hourly"
    / "pedestrian_counts_dataset.tsf"
)
CONFIG_PATH = PROJECT_ROOT / "experiment_config.yaml"
AUDIT_SUMMARY_PATH = PROJECT_ROOT / "results/pedestrian_data_quality_summary.yaml"
AUDIT_CHECKS_PATH = PROJECT_ROOT / "results/pedestrian_audit_checks.csv"

PROCESSED_PATH = OUTPUT_ROOT / "data/processed/pedestrian_hourly_long.parquet"
MANIFEST_PATH = OUTPUT_ROOT / "results/pedestrian_split_manifest.csv"
CHECKS_PATH = OUTPUT_ROOT / "results/pedestrian_split_checks.csv"
SUMMARY_PATH = OUTPUT_ROOT / "results/pedestrian_split_summary.yaml"
FIGURE_PATH = OUTPUT_ROOT / "figures/pedestrian_four_stage_split.png"
REPORT_PATH = OUTPUT_ROOT / "logs/pedestrian_split_report.json"


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def fraction_from_config(value: object) -> Fraction:
    """Convert through text so decimal ratios remain exact and reproducible."""
    return Fraction(str(value))


def timestamp_text(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def robust_scale_from_base(values: np.ndarray, base_end: int) -> np.ndarray:
    base = np.asarray(values[:base_end], dtype=float)
    median = float(np.median(base))
    q25, q75 = np.quantile(base, [0.25, 0.75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(base))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return (np.asarray(values, dtype=float) - median) / scale


def add_example_plot(
    ax: plt.Axes,
    data_row: pd.Series,
    manifest_row: pd.Series,
) -> None:
    values = np.asarray(data_row["series_value"], dtype=float)
    n = len(values)
    scaled = robust_scale_from_base(values, int(manifest_row["base_train_count"]))
    boundaries = [
        0,
        int(manifest_row["base_train_count"]),
        int(manifest_row["base_train_count"] + manifest_row["router_train_count"]),
        int(
            manifest_row["base_train_count"]
            + manifest_row["router_train_count"]
            + manifest_row["calibration_count"]
        ),
        n,
    ]

    for split_number, split_name in enumerate(SPLIT_NAMES):
        left, right = boundaries[split_number], boundaries[split_number + 1]
        local_length = right - left
        step = max(1, local_length // 1200)
        indices = np.arange(left, right, step, dtype=np.int64)
        relative_percent = 100.0 * indices / max(n - 1, 1)
        ax.plot(
            relative_percent,
            scaled[indices],
            color=COLORS[split_name],
            linewidth=0.8,
            label=split_name,
        )

    for fraction in (60, 75, 85):
        ax.axvline(fraction, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_title(f"{data_row['series_name']} (n={n:,})")
    ax.set_xlabel("Relative position in series (%)")
    ax.set_ylabel("Base-train robust-scaled count")
    ax.grid(alpha=0.15)


def main() -> None:
    for path in (DATA_PATH, CONFIG_PATH, AUDIT_SUMMARY_PATH, AUDIT_CHECKS_PATH):
        require_file(path)
    for directory in (
        PROCESSED_PATH.parent,
        MANIFEST_PATH.parent,
        FIGURE_PATH.parent,
        REPORT_PATH.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    audit_summary = yaml.safe_load(AUDIT_SUMMARY_PATH.read_text(encoding="utf-8"))
    audit_checks = pd.read_csv(AUDIT_CHECKS_PATH)

    ratio_config = config["split"]["chronological_ratios"]
    if list(ratio_config.keys()) != SPLIT_NAMES:
        raise AssertionError(
            f"Split order must be {SPLIT_NAMES}, got {list(ratio_config.keys())}"
        )
    ratios = {name: fraction_from_config(ratio_config[name]) for name in SPLIT_NAMES}
    if sum(ratios.values(), Fraction(0, 1)) != Fraction(1, 1):
        raise AssertionError("The four split ratios must sum exactly to 1")

    cumulative = {
        "base_train": ratios["base_train"],
        "router_train": ratios["base_train"] + ratios["router_train"],
        "calibration": (
            ratios["base_train"]
            + ratios["router_train"]
            + ratios["calibration"]
        ),
    }
    maximum_window = max(
        int(value)
        for value in config["preprocessing"]["window_by_frequency"]["hourly"]
    )

    data, metadata = read_tsf(DATA_PATH)
    series_names = data["series_name"].astype(str).tolist()
    lengths = data["series_value"].map(len).to_numpy(dtype=np.int64)
    total_rows = int(lengths.sum())
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))

    value_parts: list[np.ndarray] = []
    time_index_parts: list[np.ndarray] = []
    timestamp_parts: list[np.ndarray] = []
    manifest_records: list[dict[str, object]] = []
    split_codes = np.empty(total_rows, dtype=np.int8)

    for series_number, row in data.iterrows():
        values = np.asarray(row["series_value"], dtype=np.float64)
        n = int(len(values))
        start = pd.Timestamp(row["start_timestamp"])

        cut_base = n * cumulative["base_train"].numerator // cumulative["base_train"].denominator
        cut_router = n * cumulative["router_train"].numerator // cumulative["router_train"].denominator
        cut_calibration = (
            n
            * cumulative["calibration"].numerator
            // cumulative["calibration"].denominator
        )
        cuts = [0, cut_base, cut_router, cut_calibration, n]
        counts = np.diff(cuts).astype(int)
        if np.any(counts <= 0):
            raise AssertionError(f"{row['series_name']} has an empty split: {counts}")

        global_left = int(offsets[series_number])
        for split_number in range(4):
            local_left, local_right = cuts[split_number], cuts[split_number + 1]
            split_codes[
                global_left + local_left : global_left + local_right
            ] = split_number

        time_index = np.arange(n, dtype=np.int32)
        timestamps = (
            start.to_datetime64()
            + time_index.astype("timedelta64[h]")
        ).astype("datetime64[ns]")
        value_parts.append(values)
        time_index_parts.append(time_index)
        timestamp_parts.append(timestamps)

        record: dict[str, object] = {
            "dataset_id": DATASET_ID,
            "series_id": str(row["series_name"]),
            "total_length": n,
            "source_start_timestamp": timestamp_text(start),
            "source_end_timestamp": timestamp_text(pd.Timestamp(timestamps[-1])),
        }
        for split_number, split_name in enumerate(SPLIT_NAMES):
            local_left, local_right = cuts[split_number], cuts[split_number + 1]
            count = int(local_right - local_left)
            record[f"{split_name}_start"] = int(local_left)
            record[f"{split_name}_end"] = int(local_right - 1)
            record[f"{split_name}_count"] = count
            record[f"{split_name}_start_timestamp"] = timestamp_text(
                pd.Timestamp(timestamps[local_left])
            )
            record[f"{split_name}_end_timestamp"] = timestamp_text(
                pd.Timestamp(timestamps[local_right - 1])
            )
            record[f"{split_name}_actual_ratio"] = count / n
        manifest_records.append(record)

    values_all = np.concatenate(value_parts)
    time_index_all = np.concatenate(time_index_parts)
    timestamps_all = np.concatenate(timestamp_parts)
    series_codes = np.repeat(np.arange(len(data), dtype=np.int16), lengths)

    long_data = pd.DataFrame(
        {
            "dataset_id": pd.Categorical.from_codes(
                np.zeros(total_rows, dtype=np.int8), categories=[DATASET_ID]
            ),
            "series_id": pd.Categorical.from_codes(
                series_codes, categories=series_names, ordered=False
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

    # Leakage and integrity checks are made per series because boundaries differ.
    per_series_order_ok = True
    per_series_index_ok = True
    per_series_hourly_ok = True
    per_series_counts_ok = True
    for series_number, manifest_row in manifest.iterrows():
        left, right = int(offsets[series_number]), int(offsets[series_number + 1])
        n = right - left
        local_codes = split_codes[left:right]
        local_index = time_index_all[left:right]
        local_times = timestamps_all[left:right]
        expected_counts = np.asarray(
            [manifest_row[f"{name}_count"] for name in SPLIT_NAMES], dtype=int
        )
        observed_counts = np.bincount(local_codes, minlength=4)

        per_series_order_ok = per_series_order_ok and bool(
            np.all(np.diff(local_codes.astype(np.int16)) >= 0)
        )
        per_series_index_ok = per_series_index_ok and bool(
            np.array_equal(local_index, np.arange(n, dtype=np.int32))
        )
        per_series_hourly_ok = per_series_hourly_ok and bool(
            np.all(np.diff(local_times) == np.timedelta64(1, "h"))
        )
        per_series_counts_ok = per_series_counts_ok and bool(
            np.array_equal(observed_counts, expected_counts)
            and int(expected_counts.sum()) == n
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
            bool(audit_summary.get("audit_passed")),
            str(audit_summary.get("audit_passed")),
        ),
        (
            "previous_audit_checks_all_passed",
            bool(audit_checks["passed"].astype(bool).all()),
            f"checks={len(audit_checks)}",
        ),
        ("split_ratio_sum_is_one", True, str(sum(ratios.values(), Fraction(0, 1)))),
        (
            "source_frequency_is_hourly",
            str(metadata.get("frequency")).lower() == "hourly",
            str(metadata.get("frequency")),
        ),
        (
            "source_series_count_matches_audit",
            len(data) == int(audit_summary["series"]["count"]),
            f"source={len(data)}; audit={audit_summary['series']['count']}",
        ),
        (
            "source_row_count_matches_audit",
            total_rows == int(audit_summary["series"]["total_observations"]),
            f"source={total_rows}; audit={audit_summary['series']['total_observations']}",
        ),
        ("manifest_has_one_row_per_series", len(manifest) == len(data), str(len(manifest))),
        ("series_ids_are_unique", manifest["series_id"].is_unique, str(manifest["series_id"].nunique())),
        ("all_four_splits_are_nonempty", bool((manifest[count_columns] > 0).all().all()), str(minimum_counts)),
        (
            "base_train_supports_largest_window",
            int(manifest["base_train_count"].min()) > maximum_window,
            f"minimum_base={manifest['base_train_count'].min()}; maximum_window={maximum_window}",
        ),
        ("per_series_time_index_is_contiguous", per_series_index_ok, str(per_series_index_ok)),
        ("per_series_timestamps_are_hourly", per_series_hourly_ok, str(per_series_hourly_ok)),
        ("per_series_split_order_is_monotone", per_series_order_ok, str(per_series_order_ok)),
        ("per_series_split_counts_match_manifest", per_series_counts_ok, str(per_series_counts_ok)),
        (
            "aggregate_split_counts_cover_all_rows",
            sum(aggregate_counts.values()) == total_rows,
            f"aggregate={sum(aggregate_counts.values())}; rows={total_rows}",
        ),
        ("long_table_has_no_missing_values", not long_data.isna().any().any(), str(long_data.isna().sum().sum())),
        ("values_are_nonnegative", bool((values_all >= 0).all()), f"minimum={values_all.min()}"),
    ]
    checks = pd.DataFrame(check_items, columns=["check", "passed", "detail"])
    failed = checks.loc[~checks["passed"]]
    if not failed.empty:
        message = "; ".join(
            f"{row.check}: {row.detail}" for row in failed.itertuples(index=False)
        )
        raise AssertionError(f"Pedestrian split audit failed: {message}")

    long_data.to_parquet(PROCESSED_PATH, index=False, compression="snappy")
    parquet_rows = int(pq.ParquetFile(PROCESSED_PATH).metadata.num_rows)
    if parquet_rows != total_rows:
        raise AssertionError(
            f"Parquet row count mismatch: expected {total_rows}, got {parquet_rows}"
        )
    manifest.to_csv(MANIFEST_PATH, index=False)
    checks.to_csv(CHECKS_PATH, index=False)

    summary = {
        "dataset_id": DATASET_ID,
        "split_passed": True,
        "rounding_rule": "floor each cumulative boundary; test receives the remainder",
        "ratios": {name: float(ratios[name]) for name in SPLIT_NAMES},
        "series_count": int(len(manifest)),
        "total_observations": total_rows,
        "aggregate_counts": aggregate_counts,
        "per_series_count_minimum": minimum_counts,
        "per_series_count_median": {
            name: float(manifest[f"{name}_count"].median()) for name in SPLIT_NAMES
        },
        "per_series_count_maximum": maximum_counts,
        "largest_candidate_window": maximum_window,
        "minimum_base_train_count": int(manifest["base_train_count"].min()),
        "time_leakage_checks_passed": True,
        "models_fitted": False,
        "parameters_tuned": False,
        "test_values_used_for_training": False,
        "test_values_used_for_tuning": False,
        "note": (
            "Raw values were read only to construct and verify chronological partitions; "
            "no forecast model or router was fitted."
        ),
    }
    SUMMARY_PATH.write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    natural_order = np.argsort(lengths)
    example_indices = [int(natural_order[0]), int(natural_order[-1])]
    example_rows = [data.iloc[index] for index in example_indices]
    manifest_lookup = manifest.set_index("series_id")

    fig, axes = plt.subplots(2, 2, figsize=(17, 10))
    axes[0, 0].boxplot(
        [manifest[f"{name}_count"].to_numpy() for name in SPLIT_NAMES],
        tick_labels=SPLIT_NAMES,
        patch_artist=True,
        boxprops={"facecolor": "#D9E8F5"},
        medianprops={"color": "#E15759", "linewidth": 1.8},
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Per-series split sizes (log scale)")
    axes[0, 0].set_ylabel("Hourly observations")
    axes[0, 0].tick_params(axis="x", rotation=18)
    axes[0, 0].grid(axis="y", alpha=0.2)

    timeline_order = np.argsort(
        pd.to_datetime(manifest["source_start_timestamp"]).to_numpy()
    )
    for display_y, row_index in enumerate(timeline_order):
        row = manifest.iloc[int(row_index)]
        for split_name in SPLIT_NAMES:
            axes[0, 1].plot(
                pd.to_datetime(
                    [
                        row[f"{split_name}_start_timestamp"],
                        row[f"{split_name}_end_timestamp"],
                    ]
                ),
                [display_y, display_y],
                color=COLORS[split_name],
                linewidth=1.5,
            )
    axes[0, 1].set_title("Calendar-time partitions for all 66 series")
    axes[0, 1].set_xlabel("Calendar time")
    axes[0, 1].set_ylabel("Series (ordered by start)")
    axes[0, 1].tick_params(axis="x", rotation=25)

    for ax, example_row in zip(axes[1], example_rows):
        series_id = str(example_row["series_name"])
        add_example_plot(ax, example_row, manifest_lookup.loc[series_id])
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(
        "Pedestrian hourly: leakage-safe 60/15/10/15 chronological split",
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
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("Pedestrian 四阶段时间切分全部通过")
    print("序列数量：", len(manifest))
    print("长表总行数：", total_rows)
    print("各数据段总观测数量：")
    for name in SPLIT_NAMES:
        print(f"  {name}: {aggregate_counts[name]}")
    print("各序列数据段计数范围（最少至最多）：")
    for name in SPLIT_NAMES:
        print(f"  {name}: {minimum_counts[name]} 至 {maximum_counts[name]}")
    print("最大候选窗口：", maximum_window, "小时")
    print("最短序列的 base_train：", int(manifest["base_train_count"].min()), "小时")
    print("防止时间泄漏检查：通过")
    print("测试值是否用于训练：否")
    print("测试值是否用于调参：否")
    print("处理后数据：", PROCESSED_PATH)
    print("切分清单：", MANIFEST_PATH)
    print("切分检查：", CHECKS_PATH)
    print("切分图片：", FIGURE_PATH)


if __name__ == "__main__":
    main()
