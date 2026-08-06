#!/usr/bin/env python3
"""Leakage-safe four-stage split for Weather Daily.

The already frozen 500-series sample is split independently with the registered
60/15/10/15 chronological protocol. This module creates partitions only; it
does not fit models, tune parameters, or calculate forecast performance."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sci-routing")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import pandas as pd
import yaml
from pyarrow import parquet as pq


PROJECT_ROOT = Path(
    os.environ.get("SCI_ROUTING_ROOT", Path(__file__).resolve().parents[3])
).resolve()
OUTPUT_ROOT = Path(os.environ.get("SCI_ROUTING_OUTPUT_ROOT", PROJECT_ROOT)).resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tsf_reader_compat import read_tsf


DATASET_ID = "weather_daily"
SPLIT_NAMES = ["base_train", "router_train", "calibration", "test"]
COLORS = {
    "base_train": "#4C78A8",
    "router_train": "#59A14F",
    "calibration": "#F28E2B",
    "test": "#E15759",
}
TYPE_COLORS = {
    "rain": "#4C78A8",
    "mintemp": "#59A14F",
    "maxtemp": "#E15759",
    "solar": "#F28E2B",
}
EXPECTED_SAMPLE_ID = (
    "5518ada55d6adb8a8df20f5a30d52ec904362ce57df9adeeceb0bc3d22854044"
)
EXPECTED_SAMPLE_SERIES = 500
EXPECTED_SAMPLE_OBSERVATIONS = 7_370_025
EXPECTED_SAMPLE_TYPE_COUNTS = {
    "maxtemp": 124,
    "mintemp": 124,
    "rain": 121,
    "solar": 131,
}
EXPECTED_FREQUENCY = "daily"

DATA_PATH = (
    PROJECT_ROOT
    / "data/raw/weather_daily_staging/weather_daily/weather_dataset.tsf"
)
CONFIG_PATH = PROJECT_ROOT / "experiment_config.yaml"
SAMPLE_MANIFEST_PATH = PROJECT_ROOT / "results/weather_daily_sample_manifest.csv"
SAMPLE_REGISTRATION_PATH = (
    PROJECT_ROOT / "results/weather_daily_sample_registration.yaml"
)
SAMPLE_CHECKS_PATH = (
    PROJECT_ROOT / "results/weather_daily_sample_registration_checks.csv"
)

PROCESSED_PATH = OUTPUT_ROOT / "data/processed/weather_daily_long.parquet"
SPLIT_MANIFEST_PATH = OUTPUT_ROOT / "results/weather_daily_split_manifest.csv"
CHECKS_PATH = OUTPUT_ROOT / "results/weather_daily_split_checks.csv"
SUMMARY_PATH = OUTPUT_ROOT / "results/weather_daily_split_summary.yaml"
FIGURE_PATH = OUTPUT_ROOT / "figures/weather_daily_four_stage_split.png"
REPORT_PATH = OUTPUT_ROOT / "logs/weather_daily_split_report.json"

for path in (
    PROCESSED_PATH,
    SPLIT_MANIFEST_PATH,
    CHECKS_PATH,
    SUMMARY_PATH,
    FIGURE_PATH,
    REPORT_PATH,
):
    path.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fraction_from_config(value: object) -> Fraction:
    return Fraction(str(value))


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1"})


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
    base_count = int(manifest_row["base_train_count"])
    scaled = np.clip(robust_scale_from_base(values, base_count), -8.0, 8.0)
    boundaries = [
        0,
        base_count,
        base_count + int(manifest_row["router_train_count"]),
        base_count
        + int(manifest_row["router_train_count"])
        + int(manifest_row["calibration_count"]),
        n,
    ]
    for split_number, split_name in enumerate(SPLIT_NAMES):
        left, right = boundaries[split_number], boundaries[split_number + 1]
        count = right - left
        step = max(1, count // 700)
        indices = np.arange(left, right, step, dtype=np.int32)
        if len(indices) == 0 or indices[-1] != right - 1:
            indices = np.append(indices, right - 1)
        ax.plot(
            indices,
            scaled[indices],
            color=COLORS[split_name],
            linewidth=0.75,
            label=split_name,
        )
    for boundary in boundaries[1:-1]:
        ax.axvline(boundary - 0.5, color="black", linestyle="--", linewidth=0.55)
    ax.set_title(
        f"{data_row['series_name']} ({data_row['series_type']}, n={n})",
        fontsize=10,
    )
    ax.set_xlabel("Daily time index")
    ax.set_ylabel("Base-scaled value (clipped)")
    ax.grid(alpha=0.12)


def main() -> None:
    required = [
        DATA_PATH,
        CONFIG_PATH,
        SAMPLE_MANIFEST_PATH,
        SAMPLE_REGISTRATION_PATH,
        SAMPLE_CHECKS_PATH,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required Weather Daily files are missing: {missing}")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    registration = yaml.safe_load(
        SAMPLE_REGISTRATION_PATH.read_text(encoding="utf-8")
    )
    sample_manifest = pd.read_csv(SAMPLE_MANIFEST_PATH)
    sample_checks = pd.read_csv(SAMPLE_CHECKS_PATH)

    ratio_config = config["split"]["chronological_ratios"]
    if list(ratio_config.keys()) != SPLIT_NAMES:
        raise AssertionError(
            f"Split order must be {SPLIT_NAMES}, got {list(ratio_config.keys())}"
        )
    ratios = {
        name: fraction_from_config(ratio_config[name]) for name in SPLIT_NAMES
    }
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
        for value in config["preprocessing"]["window_by_frequency"]["daily"]
    )

    prechecks: list[tuple[str, bool, str]] = []
    sample_hash_valid = bool(
        registration.get("dataset_id") == DATASET_ID
        and registration.get("status")
        == "REGISTERED_AND_FROZEN_BEFORE_TIME_SPLIT_OR_MODELING"
        and registration.get("sample_id") == EXPECTED_SAMPLE_ID
        and int(registration.get("registered_sample_size", -1))
        == EXPECTED_SAMPLE_SERIES
        and registration.get("redraw_or_replacement_after_registration_allowed")
        is False
        and registration.get("performance_values_used_for_selection") is False
        and registration.get("formal_test_accessed") is False
        and registration.get("sample_manifest_sha256")
        == sha256_file(SAMPLE_MANIFEST_PATH)
    )
    prechecks.append(
        (
            "frozen_sample_registration_and_manifest_hash_valid",
            sample_hash_valid,
            f"sample_id={registration.get('sample_id')}",
        )
    )
    prechecks.append(
        (
            "all_sample_registration_checks_passed",
            len(sample_checks) == 12 and bool_series(sample_checks["passed"]).all(),
            f"passed={int(bool_series(sample_checks['passed']).sum())}/{len(sample_checks)}",
        )
    )
    observed_sample_types = {
        str(key): int(value)
        for key, value in sample_manifest["series_type"]
        .value_counts()
        .sort_index()
        .items()
    }
    sample_frame_valid = bool(
        len(sample_manifest) == EXPECTED_SAMPLE_SERIES
        and sample_manifest["series_id"].nunique() == EXPECTED_SAMPLE_SERIES
        and sample_manifest["sample_order"].tolist()
        == list(range(1, EXPECTED_SAMPLE_SERIES + 1))
        and observed_sample_types == EXPECTED_SAMPLE_TYPE_COUNTS
        and int(sample_manifest["length"].sum()) == EXPECTED_SAMPLE_OBSERVATIONS
    )
    prechecks.append(
        (
            "frozen_sample_frame_is_complete_and_unique",
            sample_frame_valid,
            (
                f"series={len(sample_manifest)}; rows="
                f"{int(sample_manifest['length'].sum())}; types={observed_sample_types}"
            ),
        )
    )
    prechecks.append(
        ("split_ratios_sum_exactly_to_one", ratio_sum == 1, str(ratio_sum))
    )
    failed_prechecks = [item for item in prechecks if not item[1]]
    if failed_prechecks:
        raise AssertionError(f"Weather Daily split prechecks failed: {failed_prechecks}")

    data_all, metadata = read_tsf(DATA_PATH)
    source_lookup = {
        str(series_id): index
        for index, series_id in enumerate(data_all["series_name"].astype(str))
    }
    selected_ids = sample_manifest["series_id"].astype(str).tolist()
    missing_selected_ids = [item for item in selected_ids if item not in source_lookup]
    if missing_selected_ids:
        raise KeyError(f"Frozen selected IDs missing from source: {missing_selected_ids}")
    selected_indices = [source_lookup[item] for item in selected_ids]
    data = data_all.iloc[selected_indices].reset_index(drop=True)

    lengths = data["series_value"].map(len).to_numpy(dtype=np.int32)
    total_rows = int(lengths.sum())
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    values_all = np.empty(total_rows, dtype=np.float64)
    time_index_all = np.empty(total_rows, dtype=np.int32)
    split_codes = np.empty(total_rows, dtype=np.int8)
    manifest_records: list[dict[str, object]] = []

    for series_number, row in data.iterrows():
        values = np.asarray(row["series_value"], dtype=np.float64)
        n = len(values)
        cut_base = (
            n
            * cumulative["base_train"].numerator
            // cumulative["base_train"].denominator
        )
        cut_router = (
            n
            * cumulative["router_train"].numerator
            // cumulative["router_train"].denominator
        )
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
        global_right = int(offsets[series_number + 1])
        values_all[global_left:global_right] = values
        time_index_all[global_left:global_right] = np.arange(n, dtype=np.int32)
        for split_number in range(4):
            local_left, local_right = cuts[split_number], cuts[split_number + 1]
            split_codes[
                global_left + local_left : global_left + local_right
            ] = split_number

        sample_row = sample_manifest.iloc[series_number]
        record: dict[str, object] = {
            "dataset_id": DATASET_ID,
            "sample_id": EXPECTED_SAMPLE_ID,
            "sample_order": int(sample_row["sample_order"]),
            "series_id": str(row["series_name"]),
            "series_type": str(row["series_type"]),
            "source_order": int(sample_row["source_order"]),
            "total_length": n,
        }
        for split_number, split_name in enumerate(SPLIT_NAMES):
            local_left, local_right = cuts[split_number], cuts[split_number + 1]
            count = local_right - local_left
            record[f"{split_name}_start"] = local_left
            record[f"{split_name}_end"] = local_right - 1
            record[f"{split_name}_count"] = count
            record[f"{split_name}_actual_ratio"] = count / n
        manifest_records.append(record)

    manifest = pd.DataFrame(manifest_records)
    series_codes = np.repeat(
        np.arange(EXPECTED_SAMPLE_SERIES, dtype=np.int16), lengths
    )
    type_order = ["rain", "mintemp", "maxtemp", "solar"]
    type_to_code = {name: index for index, name in enumerate(type_order)}
    per_series_type_codes = np.asarray(
        [type_to_code[str(value)] for value in data["series_type"]], dtype=np.int8
    )
    type_codes = np.repeat(per_series_type_codes, lengths)

    long_data = pd.DataFrame(
        {
            "dataset_id": pd.Categorical.from_codes(
                np.zeros(total_rows, dtype=np.int8), categories=[DATASET_ID]
            ),
            "series_id": pd.Categorical.from_codes(
                series_codes, categories=selected_ids
            ),
            "series_type": pd.Categorical.from_codes(
                type_codes, categories=type_order
            ),
            "time_index": time_index_all,
            "value": values_all,
            "split": pd.Categorical.from_codes(
                split_codes, categories=SPLIT_NAMES, ordered=True
            ),
        }
    )

    per_series_index_ok = True
    per_series_order_ok = True
    per_series_counts_ok = True
    values_match_source = True
    identities_match_frozen_sample = True
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
        identities_match_frozen_sample &= bool(
            str(manifest_row["series_id"])
            == str(sample_manifest.iloc[series_number]["series_id"])
            and str(manifest_row["series_type"])
            == str(sample_manifest.iloc[series_number]["series_type"])
            and int(manifest_row["total_length"])
            == int(sample_manifest.iloc[series_number]["length"])
        )

    aggregate_counts = {
        name: int(np.count_nonzero(split_codes == split_number))
        for split_number, name in enumerate(SPLIT_NAMES)
    }
    count_columns = [f"{name}_count" for name in SPLIT_NAMES]
    minimum_counts = {
        name: int(manifest[f"{name}_count"].min()) for name in SPLIT_NAMES
    }
    median_counts = {
        name: float(manifest[f"{name}_count"].median()) for name in SPLIT_NAMES
    }
    maximum_counts = {
        name: int(manifest[f"{name}_count"].max()) for name in SPLIT_NAMES
    }

    check_items = prechecks + [
        (
            "source_metadata_is_unequal_length_weather_daily_without_horizon",
            str(metadata.get("frequency")).lower() == EXPECTED_FREQUENCY
            and str(metadata.get("relation")) == "Weather"
            and metadata.get("equallength") is False
            and metadata.get("missing") is False
            and metadata.get("horizon") is None,
            (
                f"frequency={metadata.get('frequency')}; "
                f"equal_length={metadata.get('equallength')}; "
                f"horizon={metadata.get('horizon')}"
            ),
        ),
        (
            "only_frozen_selected_ids_were_extracted_in_canonical_order",
            identities_match_frozen_sample and not missing_selected_ids,
            f"selected={len(data)}; missing={len(missing_selected_ids)}",
        ),
        (
            "selected_source_size_matches_frozen_manifest",
            len(data) == EXPECTED_SAMPLE_SERIES
            and total_rows == EXPECTED_SAMPLE_OBSERVATIONS
            and np.array_equal(
                lengths, sample_manifest["length"].to_numpy(dtype=np.int32)
            ),
            f"series={len(data)}; rows={total_rows}",
        ),
        (
            "split_manifest_has_one_unique_row_per_selected_series",
            len(manifest) == EXPECTED_SAMPLE_SERIES
            and manifest["series_id"].is_unique,
            f"rows={len(manifest)}; unique={manifest['series_id'].nunique()}",
        ),
        (
            "all_four_splits_are_nonempty",
            bool((manifest[count_columns] > 0).all().all()),
            str(minimum_counts),
        ),
        (
            "base_train_supports_largest_candidate_window",
            minimum_counts["base_train"] > largest_window,
            f"minimum_base={minimum_counts['base_train']}; window={largest_window}",
        ),
        (
            "every_test_partition_exceeds_largest_candidate_window",
            minimum_counts["test"] > largest_window,
            f"minimum_test={minimum_counts['test']}; window={largest_window}",
        ),
        (
            "per_series_time_index_is_contiguous",
            per_series_index_ok,
            str(per_series_index_ok),
        ),
        (
            "per_series_split_order_is_monotone",
            per_series_order_ok,
            str(per_series_order_ok),
        ),
        (
            "per_series_split_counts_match_cumulative_floor_rule",
            per_series_counts_ok,
            str(per_series_counts_ok),
        ),
        (
            "long_values_exactly_match_selected_source_order",
            values_match_source,
            str(values_match_source),
        ),
        (
            "aggregate_counts_cover_every_selected_observation_once",
            sum(aggregate_counts.values()) == total_rows,
            f"aggregate={sum(aggregate_counts.values())}; source={total_rows}",
        ),
        (
            "long_table_has_no_missing_values",
            not long_data.isna().any().any(),
            f"missing_cells={int(long_data.isna().sum().sum())}",
        ),
        (
            "split_step_does_not_fit_tune_predict_or_calculate_metrics",
            True,
            "partition construction only; no modeling or forecast performance",
        ),
        (
            "test_values_are_partitioned_but_not_used_for_training_or_tuning",
            True,
            "formal test boundaries created under frozen protocol; no test use",
        ),
    ]
    checks = pd.DataFrame(check_items, columns=["check", "passed", "detail"])
    failed = checks.loc[~checks["passed"]]
    if not failed.empty:
        message = "; ".join(
            f"{row.check}: {row.detail}" for row in failed.itertuples(index=False)
        )
        raise AssertionError(f"Weather Daily split audit failed: {message}")

    long_data.to_parquet(PROCESSED_PATH, index=False, compression="snappy")
    parquet_rows = int(pq.ParquetFile(PROCESSED_PATH).metadata.num_rows)
    if parquet_rows != total_rows:
        raise AssertionError(
            f"Parquet row mismatch: expected {total_rows}, got {parquet_rows}"
        )
    manifest.to_csv(SPLIT_MANIFEST_PATH, index=False)
    checks.to_csv(CHECKS_PATH, index=False)

    summary = {
        "dataset_id": DATASET_ID,
        "sample_id": EXPECTED_SAMPLE_ID,
        "split_passed": True,
        "rounding_rule": "floor each cumulative boundary; test receives remainder",
        "ratios": {name: float(ratios[name]) for name in SPLIT_NAMES},
        "series_count": int(len(manifest)),
        "series_type_counts": observed_sample_types,
        "total_observations": total_rows,
        "aggregate_counts": aggregate_counts,
        "per_series_count_minimum": minimum_counts,
        "per_series_count_median": median_counts,
        "per_series_count_maximum": maximum_counts,
        "largest_candidate_window_days": largest_window,
        "minimum_base_train_count": minimum_counts["base_train"],
        "archive_declared_forecast_horizon_days": None,
        "project_test_definition": "final 15% under frozen cross-dataset protocol",
        "archive_horizon_note": (
            "The archive does not declare a forecast horizon. The final 15 percent "
            "test partition follows the unchanged project protocol and was not "
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

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for series_type in ["rain", "mintemp", "maxtemp", "solar"]:
        group = manifest.loc[manifest["series_type"] == series_type]
        axes[0, 0].scatter(
            group["total_length"],
            group["base_train_count"],
            s=16,
            alpha=0.55,
            color=TYPE_COLORS[series_type],
            label=series_type,
        )
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].xaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
    axes[0, 0].xaxis.set_minor_formatter(NullFormatter())
    axes[0, 0].yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
    axes[0, 0].yaxis.set_minor_formatter(NullFormatter())
    axes[0, 0].set_xlabel("Total series length (days)")
    axes[0, 0].set_ylabel("Base-train count (days)")
    axes[0, 0].set_title("Per-series chronological allocation")
    axes[0, 0].legend(fontsize=8)

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
    axes[0, 1].set_ylabel("Share of selected observations (%)")
    axes[0, 1].set_title("Aggregate 60/15/10/15 allocation")
    axes[0, 1].tick_params(axis="x", rotation=15)

    axes[0, 2].axis("off")
    table_rows = [
        [
            name,
            f"{minimum_counts[name]:,}",
            f"{median_counts[name]:,.1f}",
            f"{maximum_counts[name]:,}",
            f"{aggregate_counts[name]:,}",
        ]
        for name in SPLIT_NAMES
    ]
    table = axes[0, 2].table(
        cellText=table_rows,
        colLabels=["split", "min", "median", "max", "total"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.05, 1.55)
    axes[0, 2].set_title("Split-count audit", pad=12)

    manifest_lookup = manifest.set_index("series_id")
    example_types = ["rain", "mintemp", "maxtemp"]
    example_axes = list(axes[1])
    for ax, series_type in zip(example_axes, example_types):
        data_index = int(data.index[data["series_type"] == series_type][0])
        row = data.iloc[data_index]
        plot_example(ax, row, manifest_lookup.loc[str(row["series_name"])])

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(
        "Weather Daily: frozen-sample leakage-safe chronological split",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = {
        "status": "passed",
        "dataset_id": DATASET_ID,
        "sample_id": EXPECTED_SAMPLE_ID,
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
            "split_manifest": str(SPLIT_MANIFEST_PATH),
            "split_checks": str(CHECKS_PATH),
            "split_summary": str(SUMMARY_PATH),
            "split_figure": str(FIGURE_PATH),
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("Weather Daily 四阶段时间切分全部通过")
    print("固定样本编号：", EXPECTED_SAMPLE_ID)
    print("序列数量：", len(manifest))
    print("长表总行数：", total_rows)
    print("各数据段总观测数量：")
    for name in SPLIT_NAMES:
        print(f"  {name}: {aggregate_counts[name]}")
    print("各序列数据段计数范围（最少至最多）：")
    for name in SPLIT_NAMES:
        print(f"  {name}: {minimum_counts[name]} 至 {maximum_counts[name]}")
    print("最大候选窗口：", largest_window, "天")
    print("最短序列的 base_train：", minimum_counts["base_train"], "天")
    print("档案是否声明官方预测范围：否")
    print("项目测试定义：冻结协议中的最后15%，不是重新定义参数")
    print("防止时间泄漏检查：通过")
    print("测试值是否用于训练：否")
    print("测试值是否用于调参：否")
    print("处理后数据：", PROCESSED_PATH)
    print("切分清单：", SPLIT_MANIFEST_PATH)
    print("切分检查：", CHECKS_PATH)
    print("切分摘要：", SUMMARY_PATH)
    print("切分图片：", FIGURE_PATH)


if __name__ == "__main__":
    main()
