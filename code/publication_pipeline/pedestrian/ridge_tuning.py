#!/usr/bin/env python3
"""Base-only rolling validation for Pedestrian Ridge AR."""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import Ridge
import yaml


PROJECT_ROOT = Path(
    os.environ.get("SCI_ROUTING_ROOT", Path(__file__).resolve().parents[3])
).resolve()
OUTPUT_ROOT = Path(os.environ.get("SCI_ROUTING_OUTPUT_ROOT", PROJECT_ROOT)).resolve()

DATA_PATH = PROJECT_ROOT / "data/processed/pedestrian_hourly_long.parquet"
CONFIG_PATH = PROJECT_ROOT / "experiment_config.yaml"
WINDOW_SUMMARY_PATH = PROJECT_ROOT / "results/pedestrian_window_preparation_summary.yaml"
WINDOW_CHECKS_PATH = PROJECT_ROOT / "results/pedestrian_window_preparation_checks.csv"
SPLIT_SUMMARY_PATH = PROJECT_ROOT / "results/pedestrian_split_summary.yaml"

DETAIL_PATH = OUTPUT_ROOT / "results/pedestrian_ridge_rolling_validation.csv"
SUMMARY_PATH = OUTPUT_ROOT / "results/pedestrian_ridge_tuning_summary.csv"
SELECTED_PATH = OUTPUT_ROOT / "results/pedestrian_selected_ridge_params.yaml"
FOLD_PATH = OUTPUT_ROOT / "results/pedestrian_ridge_fold_manifest.csv"
FOLD_SCALER_PATH = OUTPUT_ROOT / "results/pedestrian_ridge_fold_scalers.csv"
CHECKS_PATH = OUTPUT_ROOT / "results/pedestrian_ridge_tuning_checks.csv"
FIGURE_PATH = OUTPUT_ROOT / "figures/pedestrian_ridge_tuning.png"
REPORT_PATH = OUTPUT_ROOT / "logs/pedestrian_ridge_tuning_report.json"

DATASET_ID = "pedestrian_hourly"
N_FOLDS = 3
SEASONAL_PERIOD = 24
MAX_VALIDATION_SIZE = 168
VALIDATION_BLOCK_UNIT = 24
MIN_INITIAL_TARGETS_AT_MAX_WINDOW = 24


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def passed_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.lower().eq("true")


def robust_parameters(values: np.ndarray) -> tuple[float, float, float, float, bool]:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    q1, q3 = np.percentile(values, [25, 75], method="linear")
    q1, q3 = float(q1), float(q3)
    iqr = float(q3 - q1)
    fallback = bool((not np.isfinite(iqr)) or iqr <= 1e-12)
    scale = 1.0 if fallback else iqr
    return median, q1, q3, scale, fallback


def build_fold_schedule(
    series_values: dict[str, np.ndarray], maximum_window: int
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for series_id, values in series_values.items():
        n = int(len(values))
        available_for_validation = (
            n - maximum_window - MIN_INITIAL_TARGETS_AT_MAX_WINDOW
        )
        blocks = available_for_validation // (
            N_FOLDS * VALIDATION_BLOCK_UNIT
        )
        validation_size = min(
            MAX_VALIDATION_SIZE,
            int(blocks * VALIDATION_BLOCK_UNIT),
        )
        if validation_size < VALIDATION_BLOCK_UNIT:
            raise AssertionError(
                f"{series_id} cannot support {N_FOLDS} rolling folds after "
                f"the {maximum_window}-hour window"
            )
        first_validation_start = n - N_FOLDS * validation_size
        if first_validation_start - maximum_window < MIN_INITIAL_TARGETS_AT_MAX_WINDOW:
            raise AssertionError(f"{series_id} has too few initial training targets")

        for fold in range(1, N_FOLDS + 1):
            validation_start = first_validation_start + (fold - 1) * validation_size
            validation_end = validation_start + validation_size
            records.append(
                {
                    "dataset_id": DATASET_ID,
                    "series_id": series_id,
                    "fold": fold,
                    "base_length": n,
                    "train_start": 0,
                    "train_end": validation_start - 1,
                    "train_count": validation_start,
                    "validation_start": validation_start,
                    "validation_end": validation_end - 1,
                    "validation_count": validation_size,
                    "validation_is_final_base_block": bool(
                        fold == N_FOLDS and validation_end == n
                    ),
                }
            )
    return pd.DataFrame(records)


def sufficient_statistics_for_windows(
    scaled_values: np.ndarray,
    train_end_exclusive: int,
    windows: list[int],
) -> dict[int, dict[str, np.ndarray | float | int]]:
    """Compute exact X'X/X'y statistics without materialising dense lag matrices."""
    z = np.asarray(scaled_values[:train_end_exclusive], dtype=np.float64)
    length = int(len(z))
    prefix = np.empty(length + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(z, out=prefix[1:])

    output: dict[int, dict[str, np.ndarray | float | int]] = {}
    for window in windows:
        if length <= window:
            raise AssertionError(
                f"Training length {length} cannot support window {window}"
            )
        sample_count = length - window
        sum_x = np.asarray(
            [
                prefix[length - lag_index - 1]
                - prefix[window - lag_index - 1]
                for lag_index in range(window)
            ],
            dtype=np.float64,
        )
        output[window] = {
            "n": sample_count,
            "sum_x": sum_x,
            "sum_y": float(prefix[length] - prefix[window]),
            "xtx": np.zeros((window, window), dtype=np.float64),
            "xty": np.zeros(window, dtype=np.float64),
        }

    maximum_window = max(windows)
    for difference in range(maximum_window + 1):
        if difference == 0:
            product = z * z
        else:
            product = z[difference:] * z[:-difference]
        product_prefix = np.empty(len(product) + 1, dtype=np.float64)
        product_prefix[0] = 0.0
        np.cumsum(product, out=product_prefix[1:])

        for window in windows:
            statistics = output[window]
            xtx = statistics["xtx"]
            xty = statistics["xty"]
            assert isinstance(xtx, np.ndarray)
            assert isinstance(xty, np.ndarray)

            if 1 <= difference <= window:
                xty[difference - 1] = (
                    product_prefix[length - difference]
                    - product_prefix[window - difference]
                )

            if difference < window:
                for first_lag_index in range(window - difference):
                    second_lag_index = first_lag_index + difference
                    value = (
                        product_prefix[length - second_lag_index - 1]
                        - product_prefix[window - second_lag_index - 1]
                    )
                    xtx[first_lag_index, second_lag_index] = value
                    xtx[second_lag_index, first_lag_index] = value

    return output


def add_statistics(
    destination: dict[str, np.ndarray | float | int],
    source: dict[str, np.ndarray | float | int],
) -> None:
    destination["n"] = int(destination["n"]) + int(source["n"])
    destination["sum_y"] = float(destination["sum_y"]) + float(source["sum_y"])
    for key in ("sum_x", "xtx", "xty"):
        destination_array = destination[key]
        source_array = source[key]
        assert isinstance(destination_array, np.ndarray)
        assert isinstance(source_array, np.ndarray)
        destination_array += source_array


def solve_ridge_from_statistics(
    statistics: dict[str, np.ndarray | float | int], alpha: float
) -> tuple[np.ndarray, float]:
    n = int(statistics["n"])
    sum_x = statistics["sum_x"]
    sum_y = float(statistics["sum_y"])
    xtx = statistics["xtx"]
    xty = statistics["xty"]
    assert isinstance(sum_x, np.ndarray)
    assert isinstance(xtx, np.ndarray)
    assert isinstance(xty, np.ndarray)

    mean_x = sum_x / n
    mean_y = sum_y / n
    centered_xtx = xtx - np.outer(sum_x, sum_x) / n
    centered_xty = xty - sum_x * sum_y / n
    centered_xtx = 0.5 * (centered_xtx + centered_xtx.T)
    penalized = centered_xtx + float(alpha) * np.eye(len(sum_x))
    coefficient = np.linalg.solve(penalized, centered_xty)
    intercept = float(mean_y - mean_x @ coefficient)
    return coefficient, intercept


def explicit_window_matrix(values: np.ndarray, window: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values[:end], dtype=float)
    matrix = sliding_window_view(values, window)[:-1, ::-1].copy()
    target = values[window:end].copy()
    return matrix, target


def main() -> None:
    for path in (
        DATA_PATH,
        CONFIG_PATH,
        WINDOW_SUMMARY_PATH,
        WINDOW_CHECKS_PATH,
        SPLIT_SUMMARY_PATH,
    ):
        require_file(path)
    for path in (
        DETAIL_PATH,
        SUMMARY_PATH,
        SELECTED_PATH,
        FOLD_PATH,
        FOLD_SCALER_PATH,
        CHECKS_PATH,
        FIGURE_PATH,
        REPORT_PATH,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    window_summary = yaml.safe_load(WINDOW_SUMMARY_PATH.read_text(encoding="utf-8"))
    window_checks = pd.read_csv(WINDOW_CHECKS_PATH)
    split_summary = yaml.safe_load(SPLIT_SUMMARY_PATH.read_text(encoding="utf-8"))

    seed = int(config["study"]["seed"])
    windows = [
        int(value)
        for value in config["preprocessing"]["window_by_frequency"]["hourly"]
    ]
    alphas = [
        float(value) for value in config["base_models"]["simple"]["alpha_grid"]
    ]
    if windows != [24, 48, 168]:
        raise AssertionError(f"Unexpected hourly windows: {windows}")
    if len(alphas) != 9 or len(set(alphas)) != len(alphas) or min(alphas) <= 0:
        raise AssertionError(f"Unexpected Ridge alpha grid: {alphas}")

    base_data = pd.read_parquet(
        DATA_PATH,
        columns=["series_id", "time_index", "value", "split"],
        filters=[("split", "==", "base_train")],
    )
    observed_splits = set(base_data["split"].astype(str).unique().tolist())
    if base_data.empty or observed_splits != {"base_train"}:
        raise AssertionError(f"Tuning data must contain only base_train: {observed_splits}")

    series_values: dict[str, np.ndarray] = {}
    contiguous_indices_ok = True
    for series_id_raw, group in base_data.groupby("series_id", sort=False, observed=True):
        series_id = str(series_id_raw)
        group = group.sort_values("time_index", kind="stable")
        indices = group["time_index"].to_numpy(dtype=np.int64)
        contiguous_indices_ok = contiguous_indices_ok and bool(
            np.array_equal(indices, np.arange(len(group), dtype=np.int64))
        )
        series_values[series_id] = group["value"].to_numpy(dtype=np.float64)

    fold_manifest = build_fold_schedule(series_values, max(windows))
    schedule_lookup = fold_manifest.set_index(["series_id", "fold"])
    schedule_chronology_ok = True
    schedule_complete_ok = True
    for series_id, group in fold_manifest.groupby("series_id", sort=False):
        group = group.sort_values("fold")
        schedule_chronology_ok = schedule_chronology_ok and bool(
            (group["train_end"] < group["validation_start"]).all()
            and (group["validation_start"] <= group["validation_end"]).all()
            and group["validation_start"].is_monotonic_increasing
        )
        schedule_complete_ok = schedule_complete_ok and bool(
            int(group.iloc[-1]["validation_end"]) + 1
            == int(group.iloc[-1]["base_length"])
        )

    fold_scaler_records: list[dict[str, object]] = []
    fold_info: dict[tuple[str, int], dict[str, float | int | bool]] = {}
    denominator_positive = True
    for row in fold_manifest.itertuples(index=False):
        series_id = str(row.series_id)
        fold = int(row.fold)
        values = series_values[series_id]
        train_end = int(row.train_count)
        training = values[:train_end]
        median, q1, q3, scale, fallback = robust_parameters(training)
        differences = training[SEASONAL_PERIOD:] - training[:-SEASONAL_PERIOD]
        denominator = float(np.mean(differences**2))
        denominator_positive = denominator_positive and bool(
            np.isfinite(denominator) and denominator > 1e-12
        )
        fold_info[(series_id, fold)] = {
            "train_end": train_end,
            "validation_start": int(row.validation_start),
            "validation_end": int(row.validation_end) + 1,
            "median": median,
            "scale": scale,
            "denominator": denominator,
            "fallback": fallback,
        }
        fold_scaler_records.append(
            {
                "dataset_id": DATASET_ID,
                "series_id": series_id,
                "fold": fold,
                "source_split": "base_train",
                "source_start": 0,
                "source_end": train_end - 1,
                "source_count": train_end,
                "median": median,
                "q1": q1,
                "q3": q3,
                "iqr": q3 - q1,
                "scale_used": scale,
                "zero_iqr_fallback": fallback,
                "RMSSE_seasonal_period": SEASONAL_PERIOD,
                "RMSSE_denominator": denominator,
            }
        )
    fold_scalers = pd.DataFrame(fold_scaler_records)

    # Independent exactness audit on a small explicit matrix.
    audit_series = min(series_values, key=lambda key: len(series_values[key]))
    audit_fold = 1
    audit_window = windows[0]
    audit_info = fold_info[(audit_series, audit_fold)]
    audit_training_end = int(audit_info["train_end"])
    audit_raw = series_values[audit_series]
    audit_scaled = (audit_raw - float(audit_info["median"])) / float(audit_info["scale"])
    audit_statistics = sufficient_statistics_for_windows(
        audit_scaled, audit_training_end, [audit_window]
    )[audit_window]
    explicit_x, explicit_y = explicit_window_matrix(
        audit_scaled, audit_window, audit_training_end
    )
    stats_xtx = audit_statistics["xtx"]
    stats_xty = audit_statistics["xty"]
    stats_sum_x = audit_statistics["sum_x"]
    assert isinstance(stats_xtx, np.ndarray)
    assert isinstance(stats_xty, np.ndarray)
    assert isinstance(stats_sum_x, np.ndarray)
    sufficient_statistic_max_difference = float(
        max(
            np.max(np.abs(stats_xtx - explicit_x.T @ explicit_x)),
            np.max(np.abs(stats_xty - explicit_x.T @ explicit_y)),
            np.max(np.abs(stats_sum_x - explicit_x.sum(axis=0))),
            abs(float(audit_statistics["sum_y"]) - float(explicit_y.sum())),
        )
    )
    audit_alpha = 1.0
    audit_coefficient, audit_intercept = solve_ridge_from_statistics(
        audit_statistics, audit_alpha
    )
    sklearn_audit = Ridge(alpha=audit_alpha, fit_intercept=True, solver="cholesky")
    sklearn_audit.fit(explicit_x, explicit_y)
    ridge_coefficient_max_difference = float(
        np.max(np.abs(audit_coefficient - sklearn_audit.coef_))
    )
    ridge_intercept_difference = float(abs(audit_intercept - sklearn_audit.intercept_))

    records: list[dict[str, object]] = []
    start_time = perf_counter()

    for fold in range(1, N_FOLDS + 1):
        aggregate_statistics = {
            window: {
                "n": 0,
                "sum_x": np.zeros(window, dtype=np.float64),
                "sum_y": 0.0,
                "xtx": np.zeros((window, window), dtype=np.float64),
                "xty": np.zeros(window, dtype=np.float64),
            }
            for window in windows
        }
        validation_buffers = {
            window: {
                "x": [],
                "y_raw": [],
                "median": [],
                "scale": [],
                "lengths": [],
                "denominators": [],
            }
            for window in windows
        }

        for series_id, raw_values in series_values.items():
            info = fold_info[(series_id, fold)]
            train_end = int(info["train_end"])
            validation_start = int(info["validation_start"])
            validation_end = int(info["validation_end"])
            median = float(info["median"])
            scale = float(info["scale"])
            denominator = float(info["denominator"])
            scaled_values = (raw_values - median) / scale

            series_statistics = sufficient_statistics_for_windows(
                scaled_values, train_end, windows
            )
            for window in windows:
                add_statistics(aggregate_statistics[window], series_statistics[window])
                window_view = sliding_window_view(scaled_values, window)
                validation_x = window_view[
                    validation_start - window : validation_end - window, ::-1
                ].copy()
                validation_y_raw = raw_values[validation_start:validation_end].copy()
                validation_count = validation_end - validation_start
                if len(validation_x) != validation_count:
                    raise AssertionError("Validation window count mismatch")

                buffer = validation_buffers[window]
                buffer["x"].append(validation_x)
                buffer["y_raw"].append(validation_y_raw)
                buffer["median"].append(
                    np.full(validation_count, median, dtype=np.float64)
                )
                buffer["scale"].append(
                    np.full(validation_count, scale, dtype=np.float64)
                )
                buffer["lengths"].append(validation_count)
                buffer["denominators"].append(denominator)

        for window in windows:
            statistics = aggregate_statistics[window]
            buffer = validation_buffers[window]
            validation_x = np.vstack(buffer["x"])
            validation_y_raw = np.concatenate(buffer["y_raw"])
            validation_medians = np.concatenate(buffer["median"])
            validation_scales = np.concatenate(buffer["scale"])
            validation_lengths = np.asarray(buffer["lengths"], dtype=int)
            validation_denominators = np.asarray(
                buffer["denominators"], dtype=float
            )
            validation_offsets = np.concatenate(
                ([0], np.cumsum(validation_lengths, dtype=np.int64))
            )

            for alpha in alphas:
                coefficient, intercept = solve_ridge_from_statistics(
                    statistics, alpha
                )
                prediction_scaled = validation_x @ coefficient + intercept
                prediction_raw = (
                    prediction_scaled * validation_scales + validation_medians
                )
                squared_error = (prediction_raw - validation_y_raw) ** 2
                series_rmsse = np.asarray(
                    [
                        np.sqrt(
                            np.mean(
                                squared_error[
                                    validation_offsets[index] : validation_offsets[index + 1]
                                ]
                            )
                            / validation_denominators[index]
                        )
                        for index in range(len(validation_lengths))
                    ],
                    dtype=float,
                )

                fold_rows = fold_manifest.loc[fold_manifest["fold"] == fold]
                records.append(
                    {
                        "dataset_id": DATASET_ID,
                        "model": "ridge_autoregression",
                        "fold": fold,
                        "train_scope": "base_train_prefix_only",
                        "validation_scope": "later_base_train_block_only",
                        "train_end_min": int(fold_rows["train_end"].min()),
                        "train_end_max": int(fold_rows["train_end"].max()),
                        "validation_size_min": int(validation_lengths.min()),
                        "validation_size_max": int(validation_lengths.max()),
                        "window": window,
                        "alpha": alpha,
                        "train_samples": int(statistics["n"]),
                        "validation_samples": int(len(validation_y_raw)),
                        "series_count": int(len(validation_lengths)),
                        "mean_series_RMSSE": float(series_rmsse.mean()),
                        "median_series_RMSSE": float(np.median(series_rmsse)),
                        "raw_RMSE": float(np.sqrt(np.mean(squared_error))),
                        "coefficient_l2_norm": float(np.linalg.norm(coefficient)),
                        "intercept": float(intercept),
                        "seed": seed,
                    }
                )
        print(f"[完成] fold={fold}/{N_FOLDS}, windows={len(windows)}, alphas={len(alphas)}")

    elapsed = perf_counter() - start_time
    details = pd.DataFrame(records)
    tuning_summary = (
        details.groupby(["window", "alpha"], as_index=False)
        .agg(
            mean_RMSSE=("mean_series_RMSSE", "mean"),
            std_RMSSE=("mean_series_RMSSE", "std"),
            mean_raw_RMSE=("raw_RMSE", "mean"),
            folds=("fold", "nunique"),
        )
        .sort_values(["mean_RMSSE", "window", "alpha"], kind="stable")
        .reset_index(drop=True)
    )
    best = tuning_summary.iloc[0]

    expected_records = N_FOLDS * len(windows) * len(alphas)
    fold_scaler_ranges_match = True
    for row in fold_scalers.itertuples(index=False):
        schedule = schedule_lookup.loc[(str(row.series_id), int(row.fold))]
        fold_scaler_ranges_match = fold_scaler_ranges_match and bool(
            row.source_split == "base_train"
            and row.source_start == 0
            and row.source_end == int(schedule["train_end"])
            and row.source_count == int(schedule["train_count"])
        )

    check_items: list[tuple[str, bool, str]] = [
        (
            "previous_window_preparation_passed",
            bool(window_summary.get("window_preparation_passed")),
            str(window_summary.get("window_preparation_passed")),
        ),
        (
            "previous_window_checks_all_passed",
            bool(passed_column(window_checks["passed"]).all()),
            f"checks={len(window_checks)}",
        ),
        (
            "tuning_input_contains_only_base_train",
            observed_splits == {"base_train"},
            str(observed_splits),
        ),
        (
            "base_row_count_matches_split_summary",
            len(base_data) == int(split_summary["aggregate_counts"]["base_train"]),
            f"data={len(base_data)}; expected={split_summary['aggregate_counts']['base_train']}",
        ),
        (
            "all_series_time_indices_are_contiguous",
            contiguous_indices_ok,
            str(contiguous_indices_ok),
        ),
        (
            "fold_schedule_is_chronological",
            schedule_chronology_ok,
            str(schedule_chronology_ok),
        ),
        (
            "final_validation_fold_ends_at_base_end",
            schedule_complete_ok,
            str(schedule_complete_ok),
        ),
        (
            "all_fold_scalers_use_training_prefix_only",
            fold_scaler_ranges_match,
            str(fold_scaler_ranges_match),
        ),
        (
            "all_RMSSE_denominators_are_positive",
            denominator_positive,
            str(denominator_positive),
        ),
        (
            "sufficient_statistics_match_explicit_windows",
            sufficient_statistic_max_difference < 1e-7,
            f"max_abs_difference={sufficient_statistic_max_difference:.3e}",
        ),
        (
            "ridge_coefficients_match_sklearn",
            ridge_coefficient_max_difference < 1e-8,
            f"max_abs_difference={ridge_coefficient_max_difference:.3e}",
        ),
        (
            "ridge_intercept_matches_sklearn",
            ridge_intercept_difference < 1e-8,
            f"abs_difference={ridge_intercept_difference:.3e}",
        ),
        (
            "expected_number_of_evaluations",
            len(details) == expected_records,
            f"actual={len(details)}; expected={expected_records}",
        ),
        (
            "every_parameter_combination_has_three_folds",
            bool((tuning_summary["folds"] == N_FOLDS).all()),
            f"combinations={len(tuning_summary)}",
        ),
        (
            "all_tuning_metrics_are_finite",
            bool(
                np.isfinite(
                    details[
                        [
                            "mean_series_RMSSE",
                            "median_series_RMSSE",
                            "raw_RMSE",
                        ]
                    ].to_numpy(dtype=float)
                ).all()
            ),
            f"records={len(details)}",
        ),
    ]
    checks = pd.DataFrame(check_items, columns=["check", "passed", "detail"])
    failed = checks.loc[~checks["passed"]]
    if not failed.empty:
        message = "; ".join(
            f"{row.check}: {row.detail}" for row in failed.itertuples(index=False)
        )
        raise AssertionError(f"Pedestrian Ridge tuning failed: {message}")

    details.to_csv(DETAIL_PATH, index=False)
    tuning_summary.to_csv(SUMMARY_PATH, index=False)
    fold_manifest.to_csv(FOLD_PATH, index=False)
    fold_scalers.to_csv(FOLD_SCALER_PATH, index=False)
    checks.to_csv(CHECKS_PATH, index=False)

    selected = {
        "dataset_id": DATASET_ID,
        "model": "ridge_autoregression",
        "selection_scope": "base_train_only",
        "selection_metric": "equal-weight mean series RMSSE",
        "seasonal_period_hours": SEASONAL_PERIOD,
        "rolling_folds": N_FOLDS,
        "validation_schedule": {
            "maximum_size_per_fold_hours": MAX_VALIDATION_SIZE,
            "block_unit_hours": VALIDATION_BLOCK_UNIT,
            "minimum_initial_targets_at_max_window": MIN_INITIAL_TARGETS_AT_MAX_WINDOW,
            "per_series_validation_size_minimum": int(
                fold_manifest["validation_count"].min()
            ),
            "per_series_validation_size_maximum": int(
                fold_manifest["validation_count"].max()
            ),
            "rule": (
                "Use the largest whole-day block up to 168 hours that permits "
                "three expanding folds and at least 24 initial targets after the "
                "largest candidate window."
            ),
        },
        "parameter_grid": {"windows": windows, "alphas": alphas},
        "selected_window": int(best["window"]),
        "selected_alpha": float(best["alpha"]),
        "mean_validation_RMSSE": float(best["mean_RMSSE"]),
        "std_validation_RMSSE": float(best["std_RMSSE"]),
        "tie_break_order": ["mean_RMSSE", "smaller_window", "smaller_alpha"],
        "fit_method": "exact Ridge sufficient statistics with unpenalized intercept",
        "sklearn_equivalence_audit_passed": True,
        "test_values_accessed": False,
        "seed": seed,
    }
    SELECTED_PATH.write_text(
        yaml.safe_dump(selected, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for window in windows:
        part = tuning_summary.loc[tuning_summary["window"] == window].sort_values("alpha")
        axes[0].plot(
            part["alpha"],
            part["mean_RMSSE"],
            marker="o",
            linewidth=1.5,
            label=f"window={window}",
        )
        axes[0].fill_between(
            part["alpha"].to_numpy(dtype=float),
            (part["mean_RMSSE"] - part["std_RMSSE"]).to_numpy(dtype=float),
            (part["mean_RMSSE"] + part["std_RMSSE"]).to_numpy(dtype=float),
            alpha=0.08,
        )
    axes[0].set_xscale("log")
    axes[0].set_title("Base-only Ridge tuning curves")
    axes[0].set_xlabel("Ridge alpha (log scale)")
    axes[0].set_ylabel("Mean series RMSSE")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)

    horizon_counts = (
        fold_manifest[["series_id", "validation_count"]]
        .drop_duplicates()
        .groupby("validation_count", as_index=False)
        .size()
    )
    axes[1].bar(
        horizon_counts["validation_count"].astype(str),
        horizon_counts["size"],
        color="#59A14F",
    )
    axes[1].set_title("Per-series validation horizon")
    axes[1].set_xlabel("Hours per fold")
    axes[1].set_ylabel("Number of series")
    axes[1].grid(axis="y", alpha=0.2)

    selected_folds = details.loc[
        (details["window"] == int(best["window"]))
        & np.isclose(details["alpha"], float(best["alpha"]))
    ].sort_values("fold")
    axes[2].bar(
        selected_folds["fold"].astype(str),
        selected_folds["mean_series_RMSSE"],
        color=["#4C78A8", "#F28E2B", "#E15759"],
    )
    axes[2].axhline(
        float(best["mean_RMSSE"]),
        color="black",
        linestyle="--",
        linewidth=1,
        label="three-fold mean",
    )
    axes[2].set_title(
        f"Selected: window={int(best['window'])}, alpha={float(best['alpha']):g}"
    )
    axes[2].set_xlabel("Rolling fold")
    axes[2].set_ylabel("Mean series RMSSE")
    axes[2].grid(axis="y", alpha=0.2)
    axes[2].legend(frameon=False)

    fig.suptitle("Pedestrian Ridge tuning within base_train only", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = {
        "status": "passed",
        "dataset_id": DATASET_ID,
        "check_count": int(len(checks)),
        "failed_check_count": 0,
        "rolling_folds": N_FOLDS,
        "parameter_combinations": int(len(tuning_summary)),
        "model_evaluations": int(len(details)),
        "selected_window": int(best["window"]),
        "selected_alpha": float(best["alpha"]),
        "selected_mean_RMSSE": float(best["mean_RMSSE"]),
        "runtime_seconds": float(elapsed),
        "tuning_scope": "base_train_only",
        "test_values_accessed": False,
        "outputs": {
            "details": str(DETAIL_PATH),
            "summary": str(SUMMARY_PATH),
            "selected": str(SELECTED_PATH),
            "fold_manifest": str(FOLD_PATH),
            "fold_scalers": str(FOLD_SCALER_PATH),
            "checks": str(CHECKS_PATH),
            "figure": str(FIGURE_PATH),
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print()
    print("Pedestrian Ridge 滚动调参全部通过")
    print("调参数据范围：仅 base_train")
    print("滚动验证折数：", N_FOLDS)
    print(
        "每条序列每折验证长度范围：",
        f"{fold_manifest['validation_count'].min()} 至 "
        f"{fold_manifest['validation_count'].max()} 小时",
    )
    print("参数组合数量：", len(tuning_summary))
    print("实际模型评价次数：", len(details))
    print("最佳窗口：", int(best["window"]))
    print("最佳 alpha：", float(best["alpha"]))
    print("最佳平均 RMSSE：", f"{float(best['mean_RMSSE']):.6f}")
    print("充分统计量与显式窗口核对最大差异：", f"{sufficient_statistic_max_difference:.3e}")
    print("与 sklearn Ridge 系数核对最大差异：", f"{ridge_coefficient_max_difference:.3e}")
    print("运行秒数：", f"{elapsed:.2f}")
    print("逐折结果：", DETAIL_PATH)
    print("汇总结果：", SUMMARY_PATH)
    print("选定参数：", SELECTED_PATH)
    print("调参图片：", FIGURE_PATH)


if __name__ == "__main__":
    main()
