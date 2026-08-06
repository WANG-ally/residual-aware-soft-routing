#!/usr/bin/env python3
"""Base-only scaling and causal window audit for M4 Hourly.

The script records per-series median/IQR parameters using base_train only and
audits candidate histories of 24, 48, and 168 hours.  It deliberately stores
counts and examples instead of materializing every dense window matrix.  No
model is fitted and no forecasting metric is calculated."""

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


PROJECT_ROOT = Path(
    os.environ.get("SCI_ROUTING_ROOT", Path(__file__).resolve().parents[3])
).resolve()
OUTPUT_ROOT = Path(os.environ.get("SCI_ROUTING_OUTPUT_ROOT", PROJECT_ROOT)).resolve()

DATASET_ID = "m4_hourly"
SPLIT_NAMES = ["base_train", "router_train", "calibration", "test"]
EXPECTED_WINDOWS = [24, 48, 168]
COLORS = {
    "base_train": "#4C78A8",
    "router_train": "#59A14F",
    "calibration": "#F28E2B",
    "test": "#E15759",
}

DATA_PATH = PROJECT_ROOT / "data/processed/m4_hourly_long.parquet"
CONFIG_PATH = PROJECT_ROOT / "experiment_config.yaml"
SPLIT_MANIFEST_PATH = PROJECT_ROOT / "results/m4_hourly_split_manifest.csv"
SPLIT_CHECKS_PATH = PROJECT_ROOT / "results/m4_hourly_split_checks.csv"
SPLIT_SUMMARY_PATH = PROJECT_ROOT / "results/m4_hourly_split_summary.yaml"

SCALER_PATH = OUTPUT_ROOT / "results/m4_hourly_scaler_parameters.csv"
COUNT_PATH = OUTPUT_ROOT / "results/m4_hourly_window_candidate_counts.csv"
SERIES_COUNT_PATH = OUTPUT_ROOT / "results/m4_hourly_window_counts_by_series.csv"
PREVIEW_PATH = OUTPUT_ROOT / "results/m4_hourly_window_preview.csv"
CHECKS_PATH = OUTPUT_ROOT / "results/m4_hourly_window_preparation_checks.csv"
SUMMARY_PATH = OUTPUT_ROOT / "results/m4_hourly_window_preparation_summary.yaml"
FIGURE_PATH = OUTPUT_ROOT / "figures/m4_hourly_window_example.png"
REPORT_PATH = OUTPUT_ROOT / "logs/m4_hourly_window_preparation_report.json"


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def passed_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1"})


def scale_values(values: np.ndarray, median: float, scale: float) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - median) / scale


def main() -> None:
    for path in (
        DATA_PATH,
        CONFIG_PATH,
        SPLIT_MANIFEST_PATH,
        SPLIT_CHECKS_PATH,
        SPLIT_SUMMARY_PATH,
    ):
        require_file(path)
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

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    split_summary = yaml.safe_load(SPLIT_SUMMARY_PATH.read_text(encoding="utf-8"))
    split_checks = pd.read_csv(SPLIT_CHECKS_PATH)
    manifest = pd.read_csv(SPLIT_MANIFEST_PATH)

    candidate_windows = [
        int(value)
        for value in config["preprocessing"]["window_by_frequency"]["hourly"]
    ]
    if candidate_windows != EXPECTED_WINDOWS:
        raise AssertionError(
            f"Hourly windows must remain {EXPECTED_WINDOWS}, got {candidate_windows}"
        )
    if config["preprocessing"]["scale_from"] != "base_train":
        raise AssertionError("Scaler source must remain base_train")
    if config["preprocessing"]["scaler"] != "median_iqr":
        raise AssertionError("Scaler must remain median_iqr")
    if config["base_models"]["prediction_mode"] != "frozen_parameters_causal_one_step":
        raise AssertionError("Prediction mode must remain causal one-step")

    data = pd.read_parquet(DATA_PATH)
    required_columns = {
        "dataset_id",
        "series_id",
        "time_index",
        "timestamp",
        "value",
        "split",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise AssertionError(f"Processed data lacks columns: {sorted(missing_columns)}")
    if set(data["dataset_id"].astype(str).unique()) != {DATASET_ID}:
        raise AssertionError("Processed table contains an unexpected dataset_id")

    data = data.sort_values(
        ["series_id", "time_index"], kind="stable"
    ).reset_index(drop=True)
    manifest_lookup = manifest.set_index("series_id")

    scaler_records: list[dict[str, object]] = []
    scaler_lookup: dict[str, tuple[float, float]] = {}
    group_lookup: dict[str, pd.DataFrame] = {}
    contiguous_index_ok = True
    hourly_timestamp_ok = True

    for series_id_raw, group in data.groupby(
        "series_id", sort=False, observed=True
    ):
        series_id = str(series_id_raw)
        group = group.sort_values("time_index", kind="stable").reset_index(drop=True)
        group_lookup[series_id] = group
        indices = group["time_index"].to_numpy(dtype=np.int64)
        timestamps = group["timestamp"].to_numpy(dtype="datetime64[ns]")
        contiguous_index_ok &= np.array_equal(
            indices, np.arange(len(group), dtype=np.int64)
        )
        hourly_timestamp_ok &= bool(
            np.all(np.diff(timestamps) == np.timedelta64(1, "h"))
        )

        base = group.loc[group["split"] == "base_train"]
        if base.empty:
            raise AssertionError(f"{series_id} has no base_train observations")
        base_values = base["value"].to_numpy(dtype=np.float64)
        median = float(np.median(base_values))
        q1, q3 = np.percentile(base_values, [25, 75], method="linear")
        q1 = float(q1)
        q3 = float(q3)
        iqr = float(q3 - q1)
        used_fallback = bool((not np.isfinite(iqr)) or iqr <= 1e-12)
        scale = 1.0 if used_fallback else iqr
        scaler_lookup[series_id] = (median, scale)
        scaler_records.append(
            {
                "dataset_id": DATASET_ID,
                "series_id": series_id,
                "source_split": "base_train",
                "source_start": int(base["time_index"].iloc[0]),
                "source_end": int(base["time_index"].iloc[-1]),
                "source_count": int(len(base)),
                "median": median,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "scale_used": scale,
                "zero_iqr_fallback": used_fallback,
            }
        )

    scalers = pd.DataFrame(scaler_records)
    scaler_manifest_match = True
    independent_scaler_recalculation_ok = True
    for row in scalers.itertuples(index=False):
        expected = manifest_lookup.loc[str(row.series_id)]
        scaler_manifest_match &= bool(
            row.source_start == int(expected["base_train_start"])
            and row.source_end == int(expected["base_train_end"])
            and row.source_count == int(expected["base_train_count"])
        )
        group = group_lookup[str(row.series_id)]
        allowed = group.loc[
            group["time_index"].between(row.source_start, row.source_end), "value"
        ].to_numpy(dtype=np.float64)
        recalc_median = float(np.median(allowed))
        recalc_q1, recalc_q3 = np.percentile(
            allowed, [25, 75], method="linear"
        )
        independent_scaler_recalculation_ok &= bool(
            np.isclose(recalc_median, row.median, rtol=0.0, atol=1e-12)
            and np.isclose(recalc_q1, row.q1, rtol=0.0, atol=1e-12)
            and np.isclose(recalc_q3, row.q3, rtol=0.0, atol=1e-12)
        )

    per_series_count_records: list[dict[str, object]] = []
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
                "series_id": series_id,
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
            per_series_count_records.append(record)

    per_series_counts = pd.DataFrame(per_series_count_records)
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

    preview_series = str(manifest.loc[manifest["total_length"].idxmin(), "series_id"])
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
            "target_timestamp": str(example.loc[target_position, "timestamp"]),
            "target_split": str(example.loc[target_position, "split"]),
            "target_raw": float(raw_values[target_position]),
            "target_scaled": float(scaled_values[target_position]),
        }
        for lag in range(1, preview_window + 1):
            record[f"lag_{lag}"] = float(scaled_values[target_position - lag])
        preview_records.append(record)
    preview = pd.DataFrame(preview_records)

    expected_totals = {
        window: int(len(data) - len(manifest) * window)
        for window in candidate_windows
    }
    observed_totals = {
        int(row.window): int(row.total) for row in window_counts.itertuples(index=False)
    }
    estimated_dense_bytes = {
        window: int(observed_totals[window] * window * np.dtype(np.float64).itemsize)
        for window in candidate_windows
    }

    check_items: list[tuple[str, bool, str]] = [
        (
            "previous_split_passed",
            bool(split_summary.get("split_passed")),
            str(split_summary.get("split_passed")),
        ),
        (
            "previous_split_checks_all_passed",
            bool(passed_column(split_checks["passed"]).all()),
            f"checks={len(split_checks)}",
        ),
        (
            "processed_row_count_matches_split_summary",
            len(data) == int(split_summary["total_observations"]),
            f"data={len(data)}; expected={split_summary['total_observations']}",
        ),
        (
            "candidate_windows_match_frozen_configuration",
            candidate_windows == EXPECTED_WINDOWS,
            str(candidate_windows),
        ),
        (
            "one_unique_scaler_per_series",
            len(scalers) == len(manifest) and scalers["series_id"].is_unique,
            f"scalers={len(scalers)}; series={len(manifest)}",
        ),
        (
            "scalers_use_only_base_train",
            bool(scalers["source_split"].eq("base_train").all()),
            str(scalers["source_split"].unique().tolist()),
        ),
        (
            "scaler_ranges_match_split_manifest",
            scaler_manifest_match,
            str(scaler_manifest_match),
        ),
        (
            "scaler_parameters_recalculate_from_allowed_rows",
            independent_scaler_recalculation_ok,
            str(independent_scaler_recalculation_ok),
        ),
        (
            "scaler_parameters_are_finite",
            bool(
                np.isfinite(
                    scalers[["median", "q1", "q3", "iqr", "scale_used"]]
                    .to_numpy(dtype=float)
                ).all()
            ),
            "median/q1/q3/iqr/scale_used",
        ),
        (
            "all_scales_are_positive",
            bool((scalers["scale_used"] > 0).all()),
            f"minimum={scalers['scale_used'].min()}",
        ),
        ("time_indices_are_contiguous", contiguous_index_ok, str(contiguous_index_ok)),
        ("timestamps_are_hourly", hourly_timestamp_ok, str(hourly_timestamp_ok)),
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
            "aggregate_window_counts_match_closed_form",
            observed_totals == expected_totals,
            f"observed={observed_totals}; expected={expected_totals}",
        ),
        (
            "preview_histories_end_before_targets",
            bool((preview["history_end_index"] < preview["target_time_index"]).all()),
            f"preview_rows={len(preview)}",
        ),
        (
            "preparation_step_does_not_fit_or_evaluate_models",
            True,
            "scaling/window audit only; no fit, tuning, prediction, or metric",
        ),
    ]
    checks = pd.DataFrame(check_items, columns=["check", "passed", "detail"])
    failed = checks.loc[~checks["passed"]]
    if not failed.empty:
        message = "; ".join(
            f"{row.check}: {row.detail}" for row in failed.itertuples(index=False)
        )
        raise AssertionError(f"M4 Hourly window preparation failed: {message}")

    scalers.to_csv(SCALER_PATH, index=False)
    window_counts.to_csv(COUNT_PATH, index=False)
    per_series_counts.to_csv(SERIES_COUNT_PATH, index=False)
    preview.to_csv(PREVIEW_PATH, index=False)
    checks.to_csv(CHECKS_PATH, index=False)

    summary = {
        "dataset_id": DATASET_ID,
        "window_preparation_passed": True,
        "candidate_windows_hours": candidate_windows,
        "prediction_mode": "frozen_parameters_causal_one_step",
        "scaler": {
            "type": "per-series median and IQR",
            "source_split": "base_train only",
            "quantile_method": "NumPy linear percentile",
            "zero_iqr_fallback_rule": "scale_used = 1.0",
            "zero_iqr_fallback_count": int(scalers["zero_iqr_fallback"].sum()),
        },
        "aggregate_window_counts": {
            int(row.window): {
                name: int(getattr(row, name)) for name in [*SPLIT_NAMES, "total"]
            }
            for row in window_counts.itertuples(index=False)
        },
        "window_storage": {
            "strategy": "lazy/on-demand construction from processed long table",
            "full_dense_matrices_saved": False,
            "estimated_float64_input_gib": {
                window: float(estimated_dense_bytes[window] / 1024**3)
                for window in candidate_windows
            },
            "reason": (
                "Avoid duplicated matrices and construct only the chronology-allowed "
                "fold or prediction block needed by the next pipeline stage."
            ),
        },
        "preview_series": preview_series,
        "future_information_check_passed": True,
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
    axes[0, 0].scatter(
        scalers["median"],
        scalers["iqr"],
        color="#4C78A8",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.35,
    )
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Base-train scaler parameters across 414 series")
    axes[0, 0].set_xlabel("Base-train median (log scale)")
    axes[0, 0].set_ylabel("Base-train IQR (log scale)")
    axes[0, 0].grid(alpha=0.2)

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
    axes[0, 1].set_xlabel("History window (hours)")
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
        markersize=3,
        linewidth=1.2,
        label="24 historical inputs",
    )
    axes[1, 0].scatter(
        [target_index],
        [raw_values[target_index]],
        color=COLORS["test"],
        s=65,
        zorder=3,
        label="prediction target",
    )
    axes[1, 0].axvline(target_index - 0.5, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_title(f"{preview_series}: one causal 24-hour sample")
    axes[1, 0].set_xlabel("Time index (hour)")
    axes[1, 0].set_ylabel("Raw value")
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend(frameon=False)

    long_window = candidate_windows[-1]
    test_start = int(manifest_lookup.loc[preview_series, "test_start"])
    history_indices = np.arange(test_start - long_window, test_start, dtype=int)
    example_splits = example["split"].astype(str).to_numpy()
    for split_name in SPLIT_NAMES:
        mask = example_splits[history_indices] == split_name
        if mask.any():
            axes[1, 1].scatter(
                history_indices[mask],
                scaled_values[history_indices[mask]],
                color=COLORS[split_name],
                s=11,
                label=f"history from {split_name}",
            )
    axes[1, 1].scatter(
        [test_start],
        [scaled_values[test_start]],
        color=COLORS["test"],
        marker="*",
        s=140,
        edgecolor="black",
        linewidth=0.5,
        label="first test target",
        zorder=4,
    )
    axes[1, 1].axvline(test_start - 0.5, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_title(
        f"{preview_series}: 168-hour history before first test target"
    )
    axes[1, 1].set_xlabel("Time index (hour)")
    axes[1, 1].set_ylabel("Base-train robust-scaled value")
    axes[1, 1].grid(alpha=0.2)
    axes[1, 1].legend(frameon=False, fontsize=8)

    fig.suptitle(
        "M4 Hourly: base-only scaling and causal window preparation",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = {
        "status": "passed",
        "dataset_id": DATASET_ID,
        "check_count": int(len(checks)),
        "failed_check_count": 0,
        "candidate_windows": candidate_windows,
        "scaler_count": int(len(scalers)),
        "zero_iqr_fallback_count": int(scalers["zero_iqr_fallback"].sum()),
        "future_information_check": "passed",
        "models_fitted": False,
        "forecast_metrics_calculated": False,
        "test_used_for_scaler_or_tuning": False,
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
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("M4 Hourly 滑动窗口准备全部通过")
    print("候选窗口：", candidate_windows)
    print("缩放参数来源：仅 base_train")
    print("缩放序列数量：", len(scalers))
    print("零 IQR 回退数量：", int(scalers["zero_iqr_fallback"].sum()))
    print("完整窗口矩阵保存：否（后续按滚动验证折即时构造）")
    print(
        "168小时窗口若完整展开约：",
        f"{estimated_dense_bytes[168] / 1024**3:.2f} GiB",
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
