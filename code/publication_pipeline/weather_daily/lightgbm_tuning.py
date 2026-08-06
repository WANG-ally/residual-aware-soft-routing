#!/usr/bin/env python3
"""Base-only rolling tuning for Weather Daily LightGBM."""

from __future__ import annotations

import json
import os
from itertools import product
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sci-routing")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from numpy.lib.stride_tricks import sliding_window_view
import yaml


PROJECT_ROOT = Path(
    os.environ.get("SCI_ROUTING_ROOT", Path(__file__).resolve().parents[3])
).resolve()
OUTPUT_ROOT = Path(os.environ.get("SCI_ROUTING_OUTPUT_ROOT", PROJECT_ROOT)).resolve()

DATASET_ID = "weather_daily"
EXPECTED_SAMPLE_ID = (
    "5518ada55d6adb8a8df20f5a30d52ec904362ce57df9adeeceb0bc3d22854044"
)
EXPECTED_WINDOWS = [7, 14, 28, 56]
EXPECTED_SERIES = 500
EXPECTED_BASE_ROWS = 4_421_760
EXPECTED_VALIDATION_COUNT_PER_SERIES = 56
EXPECTED_VALIDATION_SAMPLES = 28_000
EXPECTED_SAMPLED_ROWS_PER_FOLD_WINDOW = 100_000
N_FOLDS = 3
SEASONAL_PERIOD = 7
MAX_TUNING_TRAIN_TARGETS = 100_000
SAMPLING_METHOD = (
    "global midpoint systematic sampling over concatenated series targets"
)

DATA_PATH = PROJECT_ROOT / "data/processed/weather_daily_long.parquet"
CONFIG_PATH = PROJECT_ROOT / "experiment_config.yaml"
RIDGE_FOLD_PATH = PROJECT_ROOT / "results/weather_daily_ridge_fold_manifest.csv"
RIDGE_FOLD_SCALER_PATH = (
    PROJECT_ROOT / "results/weather_daily_ridge_fold_scalers.csv"
)
RIDGE_CHECKS_PATH = PROJECT_ROOT / "results/weather_daily_ridge_tuning_checks.csv"

DETAIL_PATH = OUTPUT_ROOT / "results/weather_daily_lightgbm_rolling_validation.csv"
SUMMARY_PATH = OUTPUT_ROOT / "results/weather_daily_lightgbm_tuning_summary.csv"
SELECTED_PATH = OUTPUT_ROOT / "results/weather_daily_selected_lightgbm_params.yaml"
SAMPLING_MANIFEST_PATH = (
    OUTPUT_ROOT / "results/weather_daily_lightgbm_sampling_manifest.csv"
)
SAMPLING_POSITIONS_PATH = (
    OUTPUT_ROOT / "results/weather_daily_lightgbm_sample_positions.parquet"
)
CHECKS_PATH = OUTPUT_ROOT / "results/weather_daily_lightgbm_tuning_checks.csv"
FIGURE_PATH = OUTPUT_ROOT / "figures/weather_daily_lightgbm_tuning.png"
REPORT_PATH = OUTPUT_ROOT / "logs/weather_daily_lightgbm_tuning_report.json"

for path in (
    DETAIL_PATH,
    SUMMARY_PATH,
    SELECTED_PATH,
    SAMPLING_MANIFEST_PATH,
    SAMPLING_POSITIONS_PATH,
    CHECKS_PATH,
    FIGURE_PATH,
    REPORT_PATH,
):
    path.parent.mkdir(parents=True, exist_ok=True)


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1"})


def systematic_sample_by_series(
    series_order: list[str],
    available_counts: dict[str, int],
    window: int,
    sample_cap: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    counts = np.asarray(
        [available_counts[series_id] for series_id in series_order], dtype=np.int64
    )
    if np.any(counts <= 0):
        bad = [series_order[index] for index in np.flatnonzero(counts <= 0)]
        raise AssertionError(f"No training target for series: {bad}")
    boundaries = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    total = int(boundaries[-1])
    sample_size = min(total, int(sample_cap))
    global_positions = np.floor(
        (np.arange(sample_size, dtype=np.float64) + 0.5)
        * total
        / sample_size
    ).astype(np.int64)
    if len(np.unique(global_positions)) != sample_size:
        raise AssertionError("Systematic sample contains duplicate positions")
    target_positions: dict[str, np.ndarray] = {}
    for series_number, series_id in enumerate(series_order):
        mask = (
            (global_positions >= boundaries[series_number])
            & (global_positions < boundaries[series_number + 1])
        )
        local_positions = global_positions[mask] - boundaries[series_number]
        target_positions[series_id] = (window + local_positions).astype(np.int64)
    return target_positions, global_positions


def main() -> None:
    required = [
        DATA_PATH,
        CONFIG_PATH,
        RIDGE_FOLD_PATH,
        RIDGE_FOLD_SCALER_PATH,
        RIDGE_CHECKS_PATH,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required Weather Daily files are missing: {missing}")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    fold_manifest = pd.read_csv(RIDGE_FOLD_PATH)
    fold_scalers = pd.read_csv(RIDGE_FOLD_SCALER_PATH)
    ridge_checks = pd.read_csv(RIDGE_CHECKS_PATH)
    seed = int(config["study"]["seed"])
    windows = [
        int(value)
        for value in config["preprocessing"]["window_by_frequency"]["daily"]
    ]
    black_box = config["base_models"]["black_box"]
    num_leaves_grid = [int(value) for value in black_box["num_leaves"]]
    learning_rate_grid = [float(value) for value in black_box["learning_rate"]]
    n_estimators_grid = [int(value) for value in black_box["n_estimators"]]
    feature_fraction_grid = [
        float(value) for value in black_box["feature_fraction"]
    ]
    parameter_grid = list(
        product(
            num_leaves_grid,
            learning_rate_grid,
            n_estimators_grid,
            feature_fraction_grid,
        )
    )
    if windows != EXPECTED_WINDOWS:
        raise AssertionError(f"Unexpected daily windows: {windows}")
    if len(parameter_grid) != 16:
        raise AssertionError(f"Expected 16 tree configurations, got {len(parameter_grid)}")

    base_data = pd.read_parquet(
        DATA_PATH,
        columns=["series_id", "time_index", "value", "split"],
        filters=[("split", "==", "base_train")],
    )
    observed_splits = set(base_data["split"].astype(str).unique())
    if base_data.empty or observed_splits != {"base_train"}:
        raise AssertionError(f"Tuning input must contain only base_train: {observed_splits}")
    if len(base_data) != EXPECTED_BASE_ROWS:
        raise AssertionError(
            f"Expected {EXPECTED_BASE_ROWS} base rows, got {len(base_data)}"
        )

    series_values: dict[str, np.ndarray] = {}
    contiguous_indices_ok = True
    for series_id_raw, group in base_data.groupby(
        "series_id", sort=False, observed=True
    ):
        series_id = str(series_id_raw)
        group = group.sort_values("time_index", kind="stable")
        indices = group["time_index"].to_numpy(dtype=np.int64)
        contiguous_indices_ok &= np.array_equal(
            indices, np.arange(len(group), dtype=np.int64)
        )
        series_values[series_id] = group["value"].to_numpy(dtype=np.float64)
    series_order = list(series_values)
    if len(series_values) != EXPECTED_SERIES:
        raise AssertionError(f"Expected {EXPECTED_SERIES} series")

    expected_artifact_rows = EXPECTED_SERIES * N_FOLDS
    fold_artifact_identity_ok = bool(
        len(fold_manifest) == expected_artifact_rows
        and len(fold_scalers) == expected_artifact_rows
        and set(fold_manifest["dataset_id"].astype(str)) == {DATASET_ID}
        and set(fold_scalers["dataset_id"].astype(str)) == {DATASET_ID}
        and set(fold_manifest["sample_id"].astype(str)) == {EXPECTED_SAMPLE_ID}
        and set(fold_scalers["sample_id"].astype(str)) == {EXPECTED_SAMPLE_ID}
        and set(fold_manifest["series_id"].astype(str)) == set(series_order)
        and set(fold_scalers["series_id"].astype(str)) == set(series_order)
        and set(fold_manifest["fold"].astype(int)) == {1, 2, 3}
        and set(fold_manifest["validation_count"].astype(int))
        == {EXPECTED_VALIDATION_COUNT_PER_SERIES}
    )
    if not fold_artifact_identity_ok:
        raise AssertionError("Ridge fold artifacts do not match Weather Daily")
    fold_lookup = fold_manifest.set_index(["series_id", "fold"])
    scaler_lookup = fold_scalers.set_index(["series_id", "fold"])
    shared_fold_artifacts_ok = True
    for series_id, values in series_values.items():
        for fold in range(1, N_FOLDS + 1):
            fold_row = fold_lookup.loc[(series_id, fold)]
            scaler_row = scaler_lookup.loc[(series_id, fold)]
            shared_fold_artifacts_ok &= bool(
                int(fold_row["base_length"]) == len(values)
                and int(scaler_row["source_count"]) == int(fold_row["train_count"])
                and str(scaler_row["source_split"]) == "base_train"
                and int(scaler_row["RMSSE_seasonal_period"]) == SEASONAL_PERIOD
                and float(scaler_row["RMSSE_denominator"]) > 0.0
            )

    records: list[dict[str, object]] = []
    sampling_manifest_records: list[dict[str, object]] = []
    sample_position_frames: list[pd.DataFrame] = []
    expected_samples_by_fold_window: dict[tuple[int, int], int] = {}
    all_samples_causal = True
    all_series_represented = True
    full_validation_used = True
    overall_start = perf_counter()

    for fold in range(1, N_FOLDS + 1):
        fold_rows = fold_manifest.loc[fold_manifest["fold"] == fold]
        expected_validation_samples = int(fold_rows["validation_count"].sum())
        for window in windows:
            available_counts = {
                series_id: int(fold_lookup.loc[(series_id, fold), "train_count"])
                - window
                for series_id in series_order
            }
            target_positions, global_positions = systematic_sample_by_series(
                series_order,
                available_counts,
                window,
                MAX_TUNING_TRAIN_TARGETS,
            )
            total_available = int(sum(available_counts.values()))
            expected_train_samples = min(total_available, MAX_TUNING_TRAIN_TARGETS)
            if expected_train_samples != EXPECTED_SAMPLED_ROWS_PER_FOLD_WINDOW:
                raise AssertionError("Every fold-window group must reach 100000 rows")
            expected_samples_by_fold_window[(fold, window)] = expected_train_samples
            if len(global_positions) != expected_train_samples:
                raise AssertionError("Global sample size mismatch")

            train_x_parts: list[np.ndarray] = []
            train_y_parts: list[np.ndarray] = []
            validation_x_parts: list[np.ndarray] = []
            validation_y_parts: list[np.ndarray] = []
            validation_median_parts: list[np.ndarray] = []
            validation_scale_parts: list[np.ndarray] = []
            validation_lengths: list[int] = []
            validation_denominators: list[float] = []
            sample_order_cursor = 0

            for series_id in series_order:
                raw_values = series_values[series_id]
                fold_row = fold_lookup.loc[(series_id, fold)]
                scaler_row = scaler_lookup.loc[(series_id, fold)]
                train_end = int(fold_row["train_count"])
                validation_start = int(fold_row["validation_start"])
                validation_end = int(fold_row["validation_end"]) + 1
                median = float(scaler_row["median"])
                scale = float(scaler_row["scale_used"])
                denominator = float(scaler_row["RMSSE_denominator"])
                scaled_values = (raw_values - median) / scale
                window_view = sliding_window_view(scaled_values, window)

                targets = target_positions[series_id]
                sampled_count = len(targets)
                all_series_represented &= sampled_count > 0
                all_samples_causal &= bool(
                    sampled_count > 0
                    and targets.min() >= window
                    and targets.max() < train_end
                )
                train_x_parts.append(
                    window_view[targets - window, ::-1].astype(np.float32, copy=True)
                )
                train_y_parts.append(
                    scaled_values[targets].astype(np.float32, copy=True)
                )

                validation_targets = np.arange(
                    validation_start, validation_end, dtype=np.int64
                )
                validation_x_parts.append(
                    window_view[
                        validation_targets - window, ::-1
                    ].astype(np.float32, copy=True)
                )
                validation_y_parts.append(
                    raw_values[validation_start:validation_end].astype(
                        np.float64, copy=True
                    )
                )
                validation_count = validation_end - validation_start
                validation_median_parts.append(
                    np.full(validation_count, median, dtype=np.float64)
                )
                validation_scale_parts.append(
                    np.full(validation_count, scale, dtype=np.float64)
                )
                validation_lengths.append(validation_count)
                validation_denominators.append(denominator)

                sampling_manifest_records.append(
                    {
                        "dataset_id": DATASET_ID,
                        "sample_id": EXPECTED_SAMPLE_ID,
                        "fold": fold,
                        "window": window,
                        "series_id": series_id,
                        "available_target_count": int(available_counts[series_id]),
                        "sampled_target_count": int(sampled_count),
                        "first_sampled_target": int(targets.min()),
                        "last_sampled_target": int(targets.max()),
                        "train_end_exclusive": train_end,
                        "sampling_method": SAMPLING_METHOD,
                        "sample_cap_per_fold_window": MAX_TUNING_TRAIN_TARGETS,
                    }
                )
                sample_position_frames.append(
                    pd.DataFrame(
                        {
                            "dataset_id": DATASET_ID,
                            "sample_id": EXPECTED_SAMPLE_ID,
                            "fold": np.full(sampled_count, fold, dtype=np.int8),
                            "window": np.full(sampled_count, window, dtype=np.int16),
                            "series_id": series_id,
                            "target_time_index": targets.astype(np.int32),
                            "sample_order_within_fold_window": np.arange(
                                sample_order_cursor,
                                sample_order_cursor + sampled_count,
                                dtype=np.int32,
                            ),
                        }
                    )
                )
                sample_order_cursor += sampled_count

            train_x = np.vstack(train_x_parts)
            train_y = np.concatenate(train_y_parts)
            validation_x = np.vstack(validation_x_parts)
            validation_y_raw = np.concatenate(validation_y_parts)
            validation_medians = np.concatenate(validation_median_parts)
            validation_scales = np.concatenate(validation_scale_parts)
            validation_lengths_array = np.asarray(validation_lengths, dtype=np.int64)
            validation_denominators_array = np.asarray(
                validation_denominators, dtype=np.float64
            )
            validation_offsets = np.concatenate(
                ([0], np.cumsum(validation_lengths_array, dtype=np.int64))
            )
            if len(train_x) != expected_train_samples:
                raise AssertionError("Sampled training row count mismatch")
            full_validation_used &= len(validation_x) == expected_validation_samples
            full_validation_used &= len(validation_x) == EXPECTED_VALIDATION_SAMPLES

            for configuration_number, (
                num_leaves,
                learning_rate,
                n_estimators,
                feature_fraction,
            ) in enumerate(parameter_grid, start=1):
                fit_start = perf_counter()
                model = LGBMRegressor(
                    objective="regression_l2",
                    num_leaves=num_leaves,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    feature_fraction=feature_fraction,
                    random_state=seed,
                    feature_fraction_seed=seed,
                    data_random_seed=seed,
                    deterministic=True,
                    force_col_wise=True,
                    n_jobs=-1,
                    verbosity=-1,
                )
                model.fit(train_x, train_y)
                fit_seconds = perf_counter() - fit_start
                prediction_start = perf_counter()
                prediction_scaled = model.booster_.predict(validation_x)
                prediction_seconds = perf_counter() - prediction_start
                prediction_raw = prediction_scaled * validation_scales + validation_medians
                squared_error = (prediction_raw - validation_y_raw) ** 2
                series_rmsse = np.asarray(
                    [
                        np.sqrt(
                            np.mean(
                                squared_error[
                                    validation_offsets[index]
                                    : validation_offsets[index + 1]
                                ]
                            )
                            / validation_denominators_array[index]
                        )
                        for index in range(len(validation_lengths_array))
                    ],
                    dtype=np.float64,
                )
                records.append(
                    {
                        "dataset_id": DATASET_ID,
                        "sample_id": EXPECTED_SAMPLE_ID,
                        "model": "lightgbm",
                        "fold": fold,
                        "train_scope": "systematic_sample_from_base_train_prefix",
                        "validation_scope": "complete_later_base_train_block",
                        "window": window,
                        "num_leaves": num_leaves,
                        "learning_rate": learning_rate,
                        "n_estimators": n_estimators,
                        "feature_fraction": feature_fraction,
                        "available_train_samples": total_available,
                        "sampled_train_samples": int(len(train_y)),
                        "training_sampling_fraction": float(
                            len(train_y) / total_available
                        ),
                        "validation_samples": int(len(validation_y_raw)),
                        "series_count": int(len(validation_lengths_array)),
                        "mean_series_RMSSE": float(series_rmsse.mean()),
                        "median_series_RMSSE": float(np.median(series_rmsse)),
                        "fit_seconds": float(fit_seconds),
                        "validation_predict_seconds": float(prediction_seconds),
                        "seed": seed,
                    }
                )
                if configuration_number % 4 == 0:
                    print(
                        f"[进度] fold={fold}/{N_FOLDS}, window={window}, "
                        f"config={configuration_number}/{len(parameter_grid)}",
                        flush=True,
                    )
            print(
                f"[完成] fold={fold}/{N_FOLDS}, window={window}, "
                f"configurations={len(parameter_grid)}",
                flush=True,
            )
            del train_x, train_y, validation_x

    elapsed = perf_counter() - overall_start
    details = pd.DataFrame(records)
    group_columns = [
        "window",
        "num_leaves",
        "learning_rate",
        "n_estimators",
        "feature_fraction",
    ]
    tuning_summary = (
        details.groupby(group_columns, as_index=False)
        .agg(
            mean_RMSSE=("mean_series_RMSSE", "mean"),
            std_RMSSE=("mean_series_RMSSE", "std"),
            mean_fit_seconds=("fit_seconds", "mean"),
            folds=("fold", "nunique"),
        )
        .sort_values(["mean_RMSSE", *group_columns], kind="stable")
        .reset_index(drop=True)
    )
    best = tuning_summary.iloc[0]
    sampling_manifest = pd.DataFrame(sampling_manifest_records)
    sample_positions = pd.concat(sample_position_frames, ignore_index=True)
    sample_positions["dataset_id"] = sample_positions["dataset_id"].astype("category")
    sample_positions["sample_id"] = sample_positions["sample_id"].astype("category")
    sample_positions["series_id"] = sample_positions["series_id"].astype("category")

    expected_evaluations = N_FOLDS * len(windows) * len(parameter_grid)
    expected_position_rows = int(sum(expected_samples_by_fold_window.values()))
    sample_group_counts = (
        sample_positions.groupby(["fold", "window"], observed=True).size().to_dict()
    )
    exact_sample_counts_ok = all(
        int(sample_group_counts.get(key, -1)) == expected
        for key, expected in expected_samples_by_fold_window.items()
    )
    check_items: list[tuple[str, bool, str]] = [
        (
            "ridge_fold_audit_passed",
            len(ridge_checks) == 20 and bool_series(ridge_checks["passed"]).all(),
            f"passed={int(bool_series(ridge_checks['passed']).sum())}/{len(ridge_checks)}",
        ),
        (
            "tuning_input_contains_only_base_train",
            observed_splits == {"base_train"},
            str(observed_splits),
        ),
        (
            "all_500_series_and_base_rows_are_present",
            len(series_values) == EXPECTED_SERIES
            and len(base_data) == EXPECTED_BASE_ROWS,
            f"series={len(series_values)}; rows={len(base_data)}",
        ),
        (
            "all_series_time_indices_are_contiguous",
            contiguous_indices_ok,
            str(contiguous_indices_ok),
        ),
        (
            "ridge_fold_boundaries_denominators_and_scalers_reused",
            shared_fold_artifacts_ok and fold_artifact_identity_ok,
            str(shared_fold_artifacts_ok),
        ),
        (
            "parameter_grid_has_16_tree_configurations",
            len(parameter_grid) == 16,
            str(len(parameter_grid)),
        ),
        (
            "all_sampled_targets_are_causal_training_targets",
            all_samples_causal,
            str(all_samples_causal),
        ),
        (
            "every_series_is_represented_in_every_fold_window",
            all_series_represented,
            str(all_series_represented),
        ),
        (
            "sample_count_is_minimum_of_available_and_cap",
            len(sample_positions) == expected_position_rows
            and exact_sample_counts_ok,
            f"actual={len(sample_positions)}; expected={expected_position_rows}",
        ),
        (
            "every_fold_window_uses_exactly_100000_training_rows",
            set(sample_group_counts.values())
            == {EXPECTED_SAMPLED_ROWS_PER_FOLD_WINDOW}
            and len(sample_group_counts) == N_FOLDS * len(windows),
            str(sample_group_counts),
        ),
        (
            "sampling_manifest_has_every_series_fold_window",
            len(sampling_manifest) == EXPECTED_SERIES * N_FOLDS * len(windows),
            f"rows={len(sampling_manifest)}",
        ),
        (
            "sample_positions_are_unique_within_each_fold_window",
            not sample_positions.duplicated(
                ["fold", "window", "series_id", "target_time_index"]
            ).any(),
            f"rows={len(sample_positions)}",
        ),
        (
            "complete_validation_blocks_are_used",
            full_validation_used,
            f"expected_per_fit={EXPECTED_VALIDATION_SAMPLES}",
        ),
        (
            "expected_number_of_model_fits",
            len(details) == expected_evaluations,
            f"actual={len(details)}; expected={expected_evaluations}",
        ),
        (
            "every_configuration_has_three_folds",
            len(tuning_summary) == len(windows) * len(parameter_grid)
            and tuning_summary["folds"].eq(N_FOLDS).all(),
            f"combinations={len(tuning_summary)}",
        ),
        (
            "all_metrics_and_timings_are_finite",
            np.isfinite(
                details[
                    [
                        "mean_series_RMSSE",
                        "median_series_RMSSE",
                        "fit_seconds",
                        "validation_predict_seconds",
                    ]
                ].to_numpy(dtype=float)
            ).all(),
            f"records={len(details)}",
        ),
        (
            "router_calibration_and_test_values_were_not_accessed",
            True,
            "Parquet predicate returned base_train rows only",
        ),
    ]
    checks = pd.DataFrame(check_items, columns=["check", "passed", "detail"])
    failed = checks.loc[~checks["passed"]]
    if not failed.empty:
        message = "; ".join(
            f"{row.check}: {row.detail}" for row in failed.itertuples(index=False)
        )
        raise AssertionError(f"Weather Daily LightGBM tuning failed: {message}")

    details.to_csv(DETAIL_PATH, index=False)
    tuning_summary.to_csv(SUMMARY_PATH, index=False)
    sampling_manifest.to_csv(SAMPLING_MANIFEST_PATH, index=False)
    sample_positions.to_parquet(
        SAMPLING_POSITIONS_PATH, index=False, compression="snappy"
    )
    checks.to_csv(CHECKS_PATH, index=False)

    selected = {
        "dataset_id": DATASET_ID,
        "sample_id": EXPECTED_SAMPLE_ID,
        "model": "lightgbm",
        "selection_scope": "base_train_only",
        "selection_metric": "equal-weight mean series RMSSE",
        "seasonal_period_days": SEASONAL_PERIOD,
        "rolling_folds": N_FOLDS,
        "fold_boundaries": "identical to weather_daily_ridge_fold_manifest.csv",
        "training_sample": {
            "maximum_targets_per_fold_window": MAX_TUNING_TRAIN_TARGETS,
            "method": SAMPLING_METHOD,
            "sample_count_rule": "min(all available causal targets, 100000)",
            "all_series_represented": True,
            "exact_positions_file": str(SAMPLING_POSITIONS_PATH),
        },
        "validation_sample": "all targets in each registered validation block",
        "parameter_grid": {
            "windows": windows,
            "num_leaves": num_leaves_grid,
            "learning_rate": learning_rate_grid,
            "n_estimators": n_estimators_grid,
            "feature_fraction": feature_fraction_grid,
        },
        "selected_window": int(best["window"]),
        "num_leaves": int(best["num_leaves"]),
        "learning_rate": float(best["learning_rate"]),
        "n_estimators": int(best["n_estimators"]),
        "feature_fraction": float(best["feature_fraction"]),
        "mean_validation_RMSSE": float(best["mean_RMSSE"]),
        "std_validation_RMSSE": float(best["std_RMSSE"]),
        "tie_break_order": ["mean_RMSSE", *group_columns],
        "planned_final_training_scope": "all eligible base_train targets",
        "router_train_accessed": False,
        "calibration_accessed": False,
        "test_values_accessed": False,
        "seed": seed,
    }
    SELECTED_PATH.write_text(
        yaml.safe_dump(selected, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    top = tuning_summary.head(15).copy().iloc[::-1]
    labels = [
        (
            f"w={int(row.window)}, leaves={int(row.num_leaves)}, "
            f"lr={row.learning_rate:g}, trees={int(row.n_estimators)}, "
            f"ff={row.feature_fraction:g}"
        )
        for row in top.itertuples(index=False)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    y_positions = np.arange(len(top))
    axes[0].scatter(
        top["mean_RMSSE"], y_positions, color="#4C78A8", s=48, zorder=3
    )
    axes[0].set_yticks(y_positions, labels)
    top_min = float(top["mean_RMSSE"].min())
    top_max = float(top["mean_RMSSE"].max())
    margin = max((top_max - top_min) * 0.12, 1e-5)
    axes[0].set_xlim(top_min - margin, top_max + margin)
    axes[0].set_title("Top 15 base-only LightGBM configurations")
    axes[0].set_xlabel("Equal-weight mean series RMSSE (lower is better)")
    axes[0].grid(axis="x", alpha=0.2)

    window_effect = (
        tuning_summary.groupby("window", as_index=False)
        .agg(best_RMSSE=("mean_RMSSE", "min"), median_RMSSE=("mean_RMSSE", "median"))
        .sort_values("window")
    )
    x = np.arange(len(window_effect))
    axes[1].bar(
        x - 0.18,
        window_effect["best_RMSSE"],
        width=0.36,
        label="best configuration",
        color="#59A14F",
    )
    axes[1].bar(
        x + 0.18,
        window_effect["median_RMSSE"],
        width=0.36,
        label="median configuration",
        color="#F28E2B",
    )
    axes[1].set_xticks(x, window_effect["window"].astype(str))
    axes[1].set_title("Window-length effect across tree configurations")
    axes[1].set_xlabel("Window length (days)")
    axes[1].set_ylabel("Equal-weight mean series RMSSE")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Weather Daily LightGBM tuning within base_train only", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = {
        "status": "passed",
        "dataset_id": DATASET_ID,
        "sample_id": EXPECTED_SAMPLE_ID,
        "check_count": int(len(checks)),
        "failed_check_count": 0,
        "rolling_folds": N_FOLDS,
        "tree_configurations": len(parameter_grid),
        "parameter_combinations_including_window": int(len(tuning_summary)),
        "model_fits": int(len(details)),
        "training_sample_cap": MAX_TUNING_TRAIN_TARGETS,
        "complete_validation_used": True,
        "selected_window": int(best["window"]),
        "selected_num_leaves": int(best["num_leaves"]),
        "selected_learning_rate": float(best["learning_rate"]),
        "selected_n_estimators": int(best["n_estimators"]),
        "selected_feature_fraction": float(best["feature_fraction"]),
        "selected_mean_RMSSE": float(best["mean_RMSSE"]),
        "runtime_seconds": float(elapsed),
        "tuning_scope": "base_train_only",
        "post_base_values_accessed": False,
        "outputs": {
            "details": str(DETAIL_PATH),
            "summary": str(SUMMARY_PATH),
            "selected": str(SELECTED_PATH),
            "sampling_manifest": str(SAMPLING_MANIFEST_PATH),
            "sample_positions": str(SAMPLING_POSITIONS_PATH),
            "checks": str(CHECKS_PATH),
            "figure": str(FIGURE_PATH),
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("Weather Daily LightGBM 滚动调参全部通过")
    print("固定样本编号：", EXPECTED_SAMPLE_ID)
    print("调参数据范围：仅 base_train")
    print("滚动验证折数：", N_FOLDS)
    print("每个折×窗口的训练抽样上限：", MAX_TUNING_TRAIN_TARGETS)
    print("样本不足上限时：使用该折全部合格训练窗口")
    print("验证数据：使用每折完整验证块")
    print("参数组合数量：", len(tuning_summary))
    print("实际模型拟合次数：", len(details))
    print("最佳窗口：", int(best["window"]))
    print("最佳 num_leaves：", int(best["num_leaves"]))
    print("最佳 learning_rate：", float(best["learning_rate"]))
    print("最佳 n_estimators：", int(best["n_estimators"]))
    print("最佳 feature_fraction：", float(best["feature_fraction"]))
    print("最佳平均 RMSSE：", f"{float(best['mean_RMSSE']):.6f}")
    print("router_train 是否访问：否")
    print("calibration 是否访问：否")
    print("test 是否访问：否")
    print("运行秒数：", f"{elapsed:.2f}")
    print("逐折结果：", DETAIL_PATH)
    print("汇总结果：", SUMMARY_PATH)
    print("选定参数：", SELECTED_PATH)
    print("抽样位置：", SAMPLING_POSITIONS_PATH)
    print("调参图片：", FIGURE_PATH)


if __name__ == "__main__":
    main()
