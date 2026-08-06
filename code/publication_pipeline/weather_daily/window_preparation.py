#!/usr/bin/env python3
"""Base-only scaling and causal-window audit for Weather Daily.

Only base_train values are loaded. Candidate-window counts for later segments
are derived from frozen split boundaries, without loading router, calibration,
or test values. No dense window matrices, models, or metrics are produced."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sci-routing")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pyarrow import parquet as pq


PROJECT_ROOT = Path(
    os.environ.get("SCI_ROUTING_ROOT", Path(__file__).resolve().parents[3])
).resolve()
OUTPUT_ROOT = Path(os.environ.get("SCI_ROUTING_OUTPUT_ROOT", PROJECT_ROOT)).resolve()

DATASET_ID = "weather_daily"
EXPECTED_SAMPLE_ID = (
    "5518ada55d6adb8a8df20f5a30d52ec904362ce57df9adeeceb0bc3d22854044"
)
SPLIT_NAMES = ["base_train", "router_train", "calibration", "test"]
EXPECTED_WINDOWS = [7, 14, 28, 56]
EXPECTED_SERIES = 500
EXPECTED_TOTAL_ROWS = 7_370_025
EXPECTED_BASE_ROWS = 4_421_760
EXPECTED_FALLBACK_COUNT = 31
EXPECTED_FALLBACK_BY_TYPE = {"maxtemp": 2, "mintemp": 4, "rain": 25}
EXPECTED_WINDOW_COUNTS = {
    7: {
        "base_train": 4_418_260,
        "router_train": 1_105_616,
        "calibration": 736_875,
        "test": 1_105_774,
        "total": 7_366_525,
    },
    14: {
        "base_train": 4_414_760,
        "router_train": 1_105_616,
        "calibration": 736_875,
        "test": 1_105_774,
        "total": 7_363_025,
    },
    28: {
        "base_train": 4_407_760,
        "router_train": 1_105_616,
        "calibration": 736_875,
        "test": 1_105_774,
        "total": 7_356_025,
    },
    56: {
        "base_train": 4_393_760,
        "router_train": 1_105_616,
        "calibration": 736_875,
        "test": 1_105_774,
        "total": 7_342_025,
    },
}
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

DATA_PATH = PROJECT_ROOT / "data/processed/weather_daily_long.parquet"
CONFIG_PATH = PROJECT_ROOT / "experiment_config.yaml"
SPLIT_MANIFEST_PATH = PROJECT_ROOT / "results/weather_daily_split_manifest.csv"
SPLIT_CHECKS_PATH = PROJECT_ROOT / "results/weather_daily_split_checks.csv"
SPLIT_SUMMARY_PATH = PROJECT_ROOT / "results/weather_daily_split_summary.yaml"

SCALER_PATH = OUTPUT_ROOT / "results/weather_daily_scaler_parameters.csv"
COUNT_PATH = OUTPUT_ROOT / "results/weather_daily_window_candidate_counts.csv"
SERIES_COUNT_PATH = OUTPUT_ROOT / "results/weather_daily_window_counts_by_series.csv"
PREVIEW_PATH = OUTPUT_ROOT / "results/weather_daily_window_preview.csv"
CHECKS_PATH = OUTPUT_ROOT / "results/weather_daily_window_preparation_checks.csv"
SUMMARY_PATH = OUTPUT_ROOT / "results/weather_daily_window_preparation_summary.yaml"
FIGURE_PATH = OUTPUT_ROOT / "figures/weather_daily_window_example.png"
REPORT_PATH = OUTPUT_ROOT / "logs/weather_daily_window_preparation_report.json"

for path in (
    SCALER_PATH,
    COUNT_PATH,
    SERIES_COUNT_PATH,
    PREVIEW_PATH,
    CHECKS_PATH,
    SUMMARY_PATH,
    FIGURE_PATH,
    REPORT_PATH,
):
    path.parent.mkdir(parents=True, exist_ok=True)


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1"})


def scale_values(values: np.ndarray, median: float, scale: float) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - median) / scale


def main() -> None:
    required = [
        DATA_PATH,
        CONFIG_PATH,
        SPLIT_MANIFEST_PATH,
        SPLIT_CHECKS_PATH,
        SPLIT_SUMMARY_PATH,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required Weather Daily files are missing: {missing}")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    split_summary = yaml.safe_load(SPLIT_SUMMARY_PATH.read_text(encoding="utf-8"))
    split_checks = pd.read_csv(SPLIT_CHECKS_PATH)
    manifest = pd.read_csv(SPLIT_MANIFEST_PATH)

    candidate_windows = [
        int(value)
        for value in config["preprocessing"]["window_by_frequency"]["daily"]
    ]
    if candidate_windows != EXPECTED_WINDOWS:
        raise AssertionError(
            f"Daily windows must remain {EXPECTED_WINDOWS}, got {candidate_windows}"
        )
    if config["preprocessing"]["scale_from"] != "base_train":
        raise AssertionError("Scaler source must remain base_train")
    if config["preprocessing"]["scaler"] != "median_iqr":
        raise AssertionError("Scaler must remain median_iqr")
    if config["base_models"]["prediction_mode"] != "frozen_parameters_causal_one_step":
        raise AssertionError("Prediction mode must remain causal one-step")

    parquet_file = pq.ParquetFile(DATA_PATH)
    parquet_rows = int(parquet_file.metadata.num_rows)
    base_data = pd.read_parquet(
        DATA_PATH,
        columns=[
            "dataset_id",
            "series_id",
            "series_type",
            "time_index",
            "value",
            "split",
        ],
        filters=[("split", "==", "base_train")],
    )
    required_columns = {
        "dataset_id",
        "series_id",
        "series_type",
        "time_index",
        "value",
        "split",
    }
    if required_columns.difference(base_data.columns):
        raise AssertionError("Filtered processed data lacks required columns")
    loaded_splits = set(base_data["split"].astype(str).unique())
    if loaded_splits != {"base_train"}:
        raise AssertionError(f"Unexpected values loaded from splits: {loaded_splits}")
    if set(base_data["dataset_id"].astype(str).unique()) != {DATASET_ID}:
        raise AssertionError("Processed table contains an unexpected dataset_id")

    sample_order_lookup = manifest.set_index("series_id")["sample_order"].to_dict()
    base_data["_sample_order"] = base_data["series_id"].astype(str).map(
        sample_order_lookup
    )
    if base_data["_sample_order"].isna().any():
        raise AssertionError("A base_train series is absent from the split manifest")
    base_data = base_data.sort_values(
        ["_sample_order", "time_index"], kind="stable"
    ).reset_index(drop=True)
    manifest_lookup = manifest.set_index("series_id")

    scaler_records: list[dict[str, object]] = []
    scaler_lookup: dict[str, tuple[float, float]] = {}
    group_lookup: dict[str, pd.DataFrame] = {}
    contiguous_base_index_ok = True
    series_type_match = True
    for series_id_raw, group in base_data.groupby(
        "series_id", sort=False, observed=True
    ):
        series_id = str(series_id_raw)
        group = group.sort_values("time_index", kind="stable").reset_index(drop=True)
        group_lookup[series_id] = group
        indices = group["time_index"].to_numpy(dtype=np.int64)
        contiguous_base_index_ok &= np.array_equal(
            indices, np.arange(len(group), dtype=np.int64)
        )
        expected = manifest_lookup.loc[series_id]
        series_type = str(group["series_type"].iloc[0])
        series_type_match &= series_type == str(expected["series_type"])

        values = group["value"].to_numpy(dtype=np.float64)
        median = float(np.median(values))
        q1, q3 = np.percentile(values, [25, 75], method="linear")
        q1 = float(q1)
        q3 = float(q3)
        iqr = float(q3 - q1)
        used_fallback = bool((not np.isfinite(iqr)) or iqr <= 1e-12)
        scale = 1.0 if used_fallback else iqr
        scaler_lookup[series_id] = (median, scale)
        scaler_records.append(
            {
                "dataset_id": DATASET_ID,
                "sample_id": EXPECTED_SAMPLE_ID,
                "sample_order": int(expected["sample_order"]),
                "series_id": series_id,
                "series_type": series_type,
                "source_split": "base_train",
                "source_start": int(group["time_index"].iloc[0]),
                "source_end": int(group["time_index"].iloc[-1]),
                "source_count": int(len(group)),
                "median": median,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "scale_used": scale,
                "zero_iqr_fallback": used_fallback,
            }
        )
    scalers = pd.DataFrame(scaler_records).sort_values("sample_order").reset_index(
        drop=True
    )

    scaler_manifest_match = True
    independent_recalculation_ok = True
    for row in scalers.itertuples(index=False):
        expected = manifest_lookup.loc[str(row.series_id)]
        scaler_manifest_match &= bool(
            row.source_start == int(expected["base_train_start"])
            and row.source_end == int(expected["base_train_end"])
            and row.source_count == int(expected["base_train_count"])
        )
        values = group_lookup[str(row.series_id)]["value"].to_numpy(dtype=np.float64)
        median = float(np.median(values))
        q1, q3 = np.percentile(values, [25, 75], method="linear")
        independent_recalculation_ok &= bool(
            np.isclose(median, row.median, rtol=0.0, atol=1e-12)
            and np.isclose(q1, row.q1, rtol=0.0, atol=1e-12)
            and np.isclose(q3, row.q3, rtol=0.0, atol=1e-12)
        )

    fallback_by_type = {
        str(key): int(value)
        for key, value in scalers.loc[
            scalers["zero_iqr_fallback"], "series_type"
        ].value_counts().sort_index().items()
    }

    per_series_records: list[dict[str, object]] = []
    causal_boundary_ok = True
    count_formula_ok = True
    for row in manifest.itertuples(index=False):
        series_id = str(row.series_id)
        total_length = int(row.total_length)
        split_bounds = {
            name: (
                int(getattr(row, f"{name}_start")),
                int(getattr(row, f"{name}_end")) + 1,
            )
            for name in SPLIT_NAMES
        }
        for window in candidate_windows:
            record: dict[str, object] = {
                "dataset_id": DATASET_ID,
                "sample_id": EXPECTED_SAMPLE_ID,
                "series_id": series_id,
                "series_type": str(row.series_type),
                "window": window,
                "total_length": total_length,
            }
            total_count = 0
            for split_name in SPLIT_NAMES:
                split_start, split_stop = split_bounds[split_name]
                first_target = max(split_start, window)
                count = max(0, split_stop - first_target)
                record[f"{split_name}_first_target"] = (
                    int(first_target) if count > 0 else np.nan
                )
                record[f"{split_name}_count"] = int(count)
                total_count += int(count)
                if count > 0:
                    first_history_start = first_target - window
                    first_history_end = first_target - 1
                    last_target = split_stop - 1
                    last_history_end = last_target - 1
                    causal_boundary_ok &= bool(
                        first_history_start >= 0
                        and first_history_end < first_target
                        and last_history_end < last_target
                    )
            record["total"] = total_count
            count_formula_ok &= total_count == total_length - window
            per_series_records.append(record)
    per_series_counts = pd.DataFrame(per_series_records)

    count_records: list[dict[str, int]] = []
    for window in candidate_windows:
        subset = per_series_counts.loc[per_series_counts["window"] == window]
        counts = {
            name: int(subset[f"{name}_count"].sum()) for name in SPLIT_NAMES
        }
        count_records.append(
            {"window": window, **counts, "total": int(sum(counts.values()))}
        )
    window_counts = pd.DataFrame(count_records)
    observed_counts = {
        int(row.window): {
            name: int(getattr(row, name)) for name in [*SPLIT_NAMES, "total"]
        }
        for row in window_counts.itertuples(index=False)
    }

    preview_series = str(
        manifest.sort_values(["total_length", "sample_order"]).iloc[0]["series_id"]
    )
    preview_window = candidate_windows[0]
    example = group_lookup[preview_series]
    median, scale = scaler_lookup[preview_series]
    raw_values = example["value"].to_numpy(dtype=np.float64)
    scaled_values = scale_values(raw_values, median, scale)
    preview_records: list[dict[str, object]] = []
    for target_position in range(preview_window, preview_window + 5):
        record: dict[str, object] = {
            "series_id": preview_series,
            "window": preview_window,
            "history_start_index": target_position - preview_window,
            "history_end_index": target_position - 1,
            "target_time_index": target_position,
            "target_split": "base_train",
            "target_raw": float(raw_values[target_position]),
            "target_scaled": float(scaled_values[target_position]),
        }
        for lag in range(1, preview_window + 1):
            record[f"lag_{lag}"] = float(scaled_values[target_position - lag])
        preview_records.append(record)
    preview = pd.DataFrame(preview_records)

    estimated_dense_bytes = {
        window: int(
            EXPECTED_WINDOW_COUNTS[window]["total"]
            * window
            * np.dtype(np.float64).itemsize
        )
        for window in candidate_windows
    }
    check_items: list[tuple[str, bool, str]] = [
        (
            "previous_split_and_sample_id_are_valid",
            bool(split_summary.get("split_passed"))
            and split_summary.get("sample_id") == EXPECTED_SAMPLE_ID,
            f"sample_id={split_summary.get('sample_id')}",
        ),
        (
            "all_previous_split_checks_passed",
            len(split_checks) == 19 and bool_series(split_checks["passed"]).all(),
            f"passed={int(bool_series(split_checks['passed']).sum())}/{len(split_checks)}",
        ),
        (
            "parquet_metadata_matches_frozen_split",
            parquet_rows == EXPECTED_TOTAL_ROWS
            and parquet_rows == int(split_summary["total_observations"]),
            f"parquet_rows={parquet_rows}",
        ),
        (
            "only_base_train_values_are_loaded",
            loaded_splits == {"base_train"} and len(base_data) == EXPECTED_BASE_ROWS,
            f"loaded_splits={loaded_splits}; rows={len(base_data)}",
        ),
        (
            "candidate_windows_match_frozen_configuration",
            candidate_windows == EXPECTED_WINDOWS,
            str(candidate_windows),
        ),
        (
            "one_unique_base_only_scaler_per_selected_series",
            len(scalers) == EXPECTED_SERIES
            and scalers["series_id"].nunique() == EXPECTED_SERIES
            and scalers["source_split"].eq("base_train").all(),
            f"scalers={len(scalers)}",
        ),
        (
            "scaler_series_types_match_frozen_manifest",
            series_type_match,
            str(series_type_match),
        ),
        (
            "base_time_indices_are_contiguous_from_zero",
            contiguous_base_index_ok,
            str(contiguous_base_index_ok),
        ),
        (
            "scaler_ranges_match_split_manifest",
            scaler_manifest_match,
            str(scaler_manifest_match),
        ),
        (
            "scaler_parameters_recalculate_from_loaded_base_rows",
            independent_recalculation_ok,
            str(independent_recalculation_ok),
        ),
        (
            "scaler_parameters_are_finite_and_scales_positive",
            np.isfinite(
                scalers[["median", "q1", "q3", "iqr", "scale_used"]].to_numpy(
                    dtype=float
                )
            ).all()
            and scalers["scale_used"].gt(0.0).all(),
            f"minimum_scale={float(scalers['scale_used'].min())}",
        ),
        (
            "base_only_zero_iqr_fallback_count_matches_snapshot",
            int(scalers["zero_iqr_fallback"].sum()) == EXPECTED_FALLBACK_COUNT
            and fallback_by_type == EXPECTED_FALLBACK_BY_TYPE,
            (
                f"count={int(scalers['zero_iqr_fallback'].sum())}; "
                f"by_type={fallback_by_type}"
            ),
        ),
        (
            "every_window_history_strictly_precedes_target",
            causal_boundary_ok,
            str(causal_boundary_ok),
        ),
        (
            "per_series_window_count_formula_holds",
            count_formula_ok,
            str(count_formula_ok),
        ),
        (
            "aggregate_window_counts_match_frozen_boundaries",
            observed_counts == EXPECTED_WINDOW_COUNTS,
            f"observed={observed_counts}",
        ),
        (
            "preview_contains_base_train_causal_histories_only",
            preview["target_split"].eq("base_train").all()
            and (preview["history_end_index"] < preview["target_time_index"]).all(),
            f"preview_series={preview_series}; rows={len(preview)}",
        ),
        (
            "router_calibration_and_test_values_were_not_loaded",
            loaded_splits.isdisjoint({"router_train", "calibration", "test"}),
            str(loaded_splits),
        ),
        (
            "preparation_step_does_not_fit_tune_predict_or_calculate_metrics",
            True,
            "scaling/window audit only; no model or forecast performance",
        ),
    ]
    checks = pd.DataFrame(check_items, columns=["check", "passed", "detail"])
    failed = checks.loc[~checks["passed"]]
    if not failed.empty:
        message = "; ".join(
            f"{row.check}: {row.detail}" for row in failed.itertuples(index=False)
        )
        raise AssertionError(f"Weather Daily window preparation failed: {message}")

    scalers.to_csv(SCALER_PATH, index=False)
    window_counts.to_csv(COUNT_PATH, index=False)
    per_series_counts.to_csv(SERIES_COUNT_PATH, index=False)
    preview.to_csv(PREVIEW_PATH, index=False)
    checks.to_csv(CHECKS_PATH, index=False)

    summary = {
        "dataset_id": DATASET_ID,
        "sample_id": EXPECTED_SAMPLE_ID,
        "window_preparation_passed": True,
        "candidate_windows_days": candidate_windows,
        "prediction_mode": "frozen_parameters_causal_one_step",
        "scaler": {
            "type": "per-series median and IQR",
            "source_split": "base_train only",
            "quantile_method": "NumPy linear percentile",
            "zero_iqr_fallback_rule": "scale_used = 1.0",
            "zero_iqr_fallback_count": int(
                scalers["zero_iqr_fallback"].sum()
            ),
            "zero_iqr_fallback_count_by_type": fallback_by_type,
        },
        "aggregate_window_counts": observed_counts,
        "window_storage": {
            "strategy": "lazy/on-demand construction from processed long table",
            "full_dense_matrices_saved": False,
            "estimated_float64_input_gib": {
                window: float(estimated_dense_bytes[window] / 1024**3)
                for window in candidate_windows
            },
        },
        "preview_series": preview_series,
        "future_information_check_passed": True,
        "loaded_value_splits": ["base_train"],
        "router_train_values_loaded": False,
        "calibration_values_loaded": False,
        "test_values_loaded": False,
        "test_values_used_for_scaler": False,
        "models_fitted": False,
        "parameters_tuned": False,
        "forecast_metrics_calculated": False,
    }
    SUMMARY_PATH.write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for series_type in ["rain", "mintemp", "maxtemp", "solar"]:
        group = scalers.loc[scalers["series_type"] == series_type]
        axes[0, 0].scatter(
            group["median"],
            group["iqr"],
            color=TYPE_COLORS[series_type],
            alpha=0.55,
            s=22,
            label=series_type,
        )
    axes[0, 0].set_xscale("symlog", linthresh=0.1)
    axes[0, 0].set_yscale("symlog", linthresh=0.05)
    axes[0, 0].set_title("Base-train scaler parameters across 500 series")
    axes[0, 0].set_xlabel("Base-train median (symmetric log scale)")
    axes[0, 0].set_ylabel("Base-train IQR (symmetric log scale)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.15)

    bottom = np.zeros(len(window_counts), dtype=float)
    x = np.arange(len(window_counts))
    for split_name in SPLIT_NAMES:
        heights = window_counts[split_name].to_numpy(dtype=float)
        axes[0, 1].bar(
            x,
            heights,
            bottom=bottom,
            label=split_name,
            color=COLORS[split_name],
        )
        bottom += heights
    axes[0, 1].set_xticks(x, [str(value) for value in candidate_windows])
    axes[0, 1].set_title("Available causal targets by history length")
    axes[0, 1].set_xlabel("History window (days)")
    axes[0, 1].set_ylabel("Target count")
    axes[0, 1].ticklabel_format(axis="y", style="plain")
    axes[0, 1].legend(frameon=False, fontsize=8)

    input_indices = np.arange(preview_window)
    target_index = preview_window
    axes[1, 0].plot(
        input_indices,
        raw_values[:preview_window],
        color=COLORS["base_train"],
        marker="o",
        linewidth=1.2,
        label="7 historical inputs",
    )
    axes[1, 0].scatter(
        [target_index],
        [raw_values[target_index]],
        color="#7B2CBF",
        marker="*",
        s=130,
        label="base_train target",
        zorder=4,
    )
    axes[1, 0].axvline(target_index - 0.5, color="black", linestyle="--")
    axes[1, 0].set_title(f"{preview_series}: one causal 7-day sample")
    axes[1, 0].set_xlabel("Daily time index")
    axes[1, 0].set_ylabel("Raw value")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(alpha=0.15)

    long_window = candidate_windows[-1]
    router_start = int(manifest_lookup.loc[preview_series, "router_train_start"])
    history_indices = np.arange(router_start - long_window, router_start, dtype=int)
    axes[1, 1].plot(
        history_indices,
        np.clip(scaled_values[history_indices], -8.0, 8.0),
        color=COLORS["base_train"],
        marker="o",
        markersize=2.6,
        linewidth=0.9,
        label="56 base_train historical inputs",
    )
    axes[1, 1].scatter(
        [router_start],
        [0.0],
        color=COLORS["router_train"],
        marker="*",
        s=145,
        edgecolor="black",
        linewidth=0.5,
        label="first router target (value not loaded)",
        zorder=4,
    )
    axes[1, 1].axvline(router_start - 0.5, color="black", linestyle="--")
    axes[1, 1].set_title(
        f"{preview_series}: history before first router target"
    )
    axes[1, 1].set_xlabel("Daily time index")
    axes[1, 1].set_ylabel("Base-scaled history (clipped)")
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].grid(alpha=0.15)

    fig.suptitle(
        "Weather Daily: base-only scaling and causal window preparation",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = {
        "status": "passed",
        "dataset_id": DATASET_ID,
        "sample_id": EXPECTED_SAMPLE_ID,
        "check_count": int(len(checks)),
        "failed_check_count": 0,
        "candidate_windows": candidate_windows,
        "scaler_count": int(len(scalers)),
        "zero_iqr_fallback_count": int(scalers["zero_iqr_fallback"].sum()),
        "loaded_value_splits": ["base_train"],
        "future_information_check": "passed",
        "models_fitted": False,
        "forecast_metrics_calculated": False,
        "test_values_loaded_or_used": False,
        "outputs": {
            "scalers": str(SCALER_PATH),
            "aggregate_counts": str(COUNT_PATH),
            "per_series_counts": str(SERIES_COUNT_PATH),
            "preview": str(PREVIEW_PATH),
            "checks": str(CHECKS_PATH),
            "summary": str(SUMMARY_PATH),
            "figure": str(FIGURE_PATH),
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("Weather Daily 滑动窗口准备全部通过")
    print("固定样本编号：", EXPECTED_SAMPLE_ID)
    print("候选窗口：", candidate_windows)
    print("缩放参数来源：仅 base_train")
    print("缩放序列数量：", len(scalers))
    print("零 IQR 回退数量：", int(scalers["zero_iqr_fallback"].sum()))
    print("零 IQR 回退按变量类型：", fallback_by_type)
    print("加载到内存的数值段：仅 base_train")
    print("router_train 数值是否加载：否")
    print("calibration 数值是否加载：否")
    print("test 数值是否加载：否")
    print("完整窗口矩阵保存：否（后续按滚动验证折即时构造）")
    print(
        "56天窗口若完整展开约：",
        f"{estimated_dense_bytes[56] / 1024**3:.2f} GiB",
    )
    print("因果预测方式：冻结参数、逐时点一步预测")
    print("未来信息检查：通过")
    print("各窗口样本数量：")
    print(window_counts.to_string(index=False))
    print("缩放参数：", SCALER_PATH)
    print("窗口计数：", COUNT_PATH)
    print("逐序列计数：", SERIES_COUNT_PATH)
    print("窗口示例表：", PREVIEW_PATH)
    print("窗口示意图：", FIGURE_PATH)


if __name__ == "__main__":
    main()
