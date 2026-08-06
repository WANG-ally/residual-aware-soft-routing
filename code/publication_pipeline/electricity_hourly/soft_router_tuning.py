#!/usr/bin/env python3
"""Router-train-only rolling tuning for Electricity Hourly soft routing."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
import yaml


PROJECT_ROOT = Path(
    os.environ.get("SCI_ROUTING_ROOT", Path(__file__).resolve().parents[3])
).resolve()
OUTPUT_ROOT = Path(os.environ.get("SCI_ROUTING_OUTPUT_ROOT", PROJECT_ROOT)).resolve()

FEATURE_PATH = PROJECT_ROOT / "results/electricity_hourly_router_features.parquet"
MANIFEST_PATH = PROJECT_ROOT / "results/electricity_hourly_router_feature_manifest.csv"
FEATURE_CHECKS_PATH = PROJECT_ROOT / "results/electricity_hourly_router_feature_checks.csv"
CONFIG_PATH = PROJECT_ROOT / "experiment_config.yaml"

DETAIL_PATH = OUTPUT_ROOT / "results/electricity_hourly_soft_router_rolling_validation.csv"
SUMMARY_PATH = OUTPUT_ROOT / "results/electricity_hourly_soft_router_tuning_summary.csv"
SELECTED_PATH = OUTPUT_ROOT / "results/electricity_hourly_selected_soft_router_params.yaml"
FOLD_PATH = OUTPUT_ROOT / "results/electricity_hourly_soft_router_fold_manifest.csv"
SAMPLING_MANIFEST_PATH = OUTPUT_ROOT / "results/electricity_hourly_soft_router_sampling_manifest.csv"
SAMPLING_POSITIONS_PATH = OUTPUT_ROOT / "results/electricity_hourly_soft_router_sample_positions.parquet"
CHECKS_PATH = OUTPUT_ROOT / "results/electricity_hourly_soft_router_tuning_checks.csv"
FIGURE_PATH = OUTPUT_ROOT / "figures/electricity_hourly_soft_router_tuning.png"
REPORT_PATH = OUTPUT_ROOT / "logs/electricity_hourly_soft_router_tuning_report.json"

DATASET_ID = "electricity_hourly"
N_FOLDS = 3
MAX_VALIDATION_SIZE = 168
VALIDATION_BLOCK_UNIT = 12
MIN_INITIAL_ROUTER_ROWS = 24
MAX_TUNING_TRAIN_ROWS = 100_000
SAMPLING_METHOD = "global midpoint systematic sampling over per-series training prefixes"
EXPECTED_SERIES = 321
EXPECTED_ROUTER_ROWS = 1_261_530
EXPECTED_ROUTER_ROWS_PER_SERIES = 3_930
EXPECTED_VALIDATION_SIZE = 168
EXPECTED_VALIDATION_ROWS_PER_FOLD = 53_928
EXPECTED_FOLD_TRAIN_COUNTS = {1: 3_426, 2: 3_594, 3: 3_762}
EXPECTED_SAMPLE_ROWS_PER_FOLD = 100_000
EXPECTED_PARAMETER_COMBINATIONS = 80
EXPECTED_MODEL_FITS = 240


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def passed_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1"})


def stable_soft_target(loss_advantage: np.ndarray, temperature: float) -> np.ndarray:
    normalized = np.clip(
        np.asarray(loss_advantage, dtype=np.float64) / float(temperature),
        -40.0,
        40.0,
    )
    return 1.0 / (1.0 + np.exp(-normalized))


def features_for_lag(
    maximum_lag: int,
    context_features: list[str],
    residual_features: list[str],
) -> list[str]:
    selected = list(context_features)
    selected.extend(
        name
        for name in residual_features
        if int(name.rsplit("_", 1)[1]) <= maximum_lag
    )
    expected_count = len(context_features) + 2 * maximum_lag
    if len(selected) != expected_count:
        raise AssertionError(
            f"Feature count mismatch for lag {maximum_lag}: "
            f"{len(selected)} vs {expected_count}"
        )
    return selected


def build_fold_schedule(router: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for series_id_raw, group in router.groupby(
        "series_id", sort=False, observed=True
    ):
        series_id = str(series_id_raw)
        group = group.sort_values("time_index", kind="stable")
        n = len(group)
        available = n - MIN_INITIAL_ROUTER_ROWS
        blocks = available // (N_FOLDS * VALIDATION_BLOCK_UNIT)
        validation_size = min(
            MAX_VALIDATION_SIZE, int(blocks * VALIDATION_BLOCK_UNIT)
        )
        if validation_size < VALIDATION_BLOCK_UNIT:
            raise AssertionError(
                f"{series_id} cannot support three router rolling folds"
            )
        first_validation_position = n - N_FOLDS * validation_size
        if first_validation_position < MIN_INITIAL_ROUTER_ROWS:
            raise AssertionError(f"{series_id} has too few initial router rows")
        time_indices = group["time_index"].to_numpy(dtype=np.int64)
        dataframe_indices = group.index.to_numpy(dtype=np.int64)
        for fold in range(1, N_FOLDS + 1):
            validation_position_start = (
                first_validation_position + (fold - 1) * validation_size
            )
            validation_position_end = validation_position_start + validation_size
            records.append(
                {
                    "dataset_id": DATASET_ID,
                    "series_id": series_id,
                    "fold": fold,
                    "router_rows": n,
                    "train_position_start": 0,
                    "train_position_end": validation_position_start - 1,
                    "train_count": validation_position_start,
                    "train_first_time_index": int(time_indices[0]),
                    "train_last_time_index": int(
                        time_indices[validation_position_start - 1]
                    ),
                    "validation_position_start": validation_position_start,
                    "validation_position_end": validation_position_end - 1,
                    "validation_count": validation_size,
                    "validation_first_time_index": int(
                        time_indices[validation_position_start]
                    ),
                    "validation_last_time_index": int(
                        time_indices[validation_position_end - 1]
                    ),
                    "train_first_dataframe_index": int(dataframe_indices[0]),
                    "train_last_dataframe_index": int(
                        dataframe_indices[validation_position_start - 1]
                    ),
                    "validation_first_dataframe_index": int(
                        dataframe_indices[validation_position_start]
                    ),
                    "validation_last_dataframe_index": int(
                        dataframe_indices[validation_position_end - 1]
                    ),
                    "validation_is_final_router_block": bool(
                        fold == N_FOLDS and validation_position_end == n
                    ),
                }
            )
    return pd.DataFrame(records)


def systematic_sample_indices(
    train_indices_by_series: list[np.ndarray], sample_cap: int
) -> tuple[np.ndarray, np.ndarray]:
    concatenated = np.concatenate(train_indices_by_series)
    sample_size = min(len(concatenated), int(sample_cap))
    positions = np.floor(
        (np.arange(sample_size, dtype=np.float64) + 0.5)
        * len(concatenated)
        / sample_size
    ).astype(np.int64)
    if len(np.unique(positions)) != sample_size:
        raise AssertionError("Systematic router sample has duplicate positions")
    return concatenated[positions], positions


def equal_series_mean(values: np.ndarray, lengths: np.ndarray) -> float:
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    means = np.asarray(
        [
            np.mean(values[offsets[index] : offsets[index + 1]])
            for index in range(len(lengths))
        ],
        dtype=np.float64,
    )
    return float(means.mean())


def main() -> None:
    for path in (FEATURE_PATH, MANIFEST_PATH, FEATURE_CHECKS_PATH, CONFIG_PATH):
        require_file(path)
    for path in (
        DETAIL_PATH,
        SUMMARY_PATH,
        SELECTED_PATH,
        FOLD_PATH,
        SAMPLING_MANIFEST_PATH,
        SAMPLING_POSITIONS_PATH,
        CHECKS_PATH,
        FIGURE_PATH,
        REPORT_PATH,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    feature_checks = pd.read_csv(FEATURE_CHECKS_PATH)
    feature_manifest = pd.read_csv(MANIFEST_PATH)
    seed = int(config["study"]["seed"])
    target_simple_coverage = float(config["study"]["primary_target_coverage"])
    temperatures = [
        float(value) for value in config["soft_router"]["temperature_grid"]
    ]
    residual_lags = [
        int(value) for value in config["soft_router"]["residual_lag_grid"]
    ]
    c_grid = [float(value) for value in config["soft_router"]["c_grid"]]
    if temperatures != [0.02, 0.05, 0.1, 0.2, 0.5]:
        raise AssertionError(f"Unexpected temperature grid: {temperatures}")
    if residual_lags != [1, 4, 8, 16]:
        raise AssertionError(f"Unexpected residual-lag grid: {residual_lags}")
    if c_grid != [0.01, 0.1, 1.0, 10.0]:
        raise AssertionError(f"Unexpected C grid: {c_grid}")
    if not passed_column(feature_checks["passed"]).all():
        raise AssertionError("Router-feature checks did not all pass")
    if not passed_column(feature_manifest["available_before_target"]).all():
        raise AssertionError("Feature manifest contains post-target features")

    context_features = feature_manifest.loc[
        feature_manifest["group"] == "context", "feature"
    ].tolist()
    residual_features = feature_manifest.loc[
        feature_manifest["group"] == "past_residual", "feature"
    ].tolist()
    if len(context_features) != 14 or len(residual_features) != 32:
        raise AssertionError(
            f"Unexpected feature groups: context={len(context_features)}, "
            f"residual={len(residual_features)}"
        )
    if len(feature_manifest) != 46 or feature_manifest["feature"].nunique() != 46:
        raise AssertionError("Router feature manifest must contain 46 unique features")

    router = pd.read_parquet(
        FEATURE_PATH, filters=[("split", "==", "router_train")]
    )
    observed_splits = set(router["split"].astype(str).unique().tolist())
    if router.empty or observed_splits != {"router_train"}:
        raise AssertionError(f"Router tuning scope is invalid: {observed_splits}")
    router = (
        router.sort_values(["series_id", "time_index"], kind="stable")
        .reset_index(drop=True)
    )
    if len(router) != EXPECTED_ROUTER_ROWS:
        raise AssertionError(
            f"Expected {EXPECTED_ROUTER_ROWS} router_train rows, got {len(router)}"
        )
    router_rows_per_series = (
        router.groupby("series_id", observed=True).size().to_numpy(dtype=np.int64)
    )
    if not (
        len(router_rows_per_series) == EXPECTED_SERIES
        and np.all(router_rows_per_series == EXPECTED_ROUTER_ROWS_PER_SERIES)
    ):
        raise AssertionError("Electricity router_train series lengths changed")

    per_series_time_contiguous = True
    for _, group in router.groupby("series_id", sort=False, observed=True):
        times = group["time_index"].to_numpy(dtype=np.int64)
        per_series_time_contiguous &= bool(np.all(np.diff(times) == 1))

    fold_manifest = build_fold_schedule(router)
    observed_fold_train_counts = (
        fold_manifest.groupby("fold", observed=True)["train_count"]
        .unique()
        .to_dict()
    )
    exact_fold_schedule_ok = all(
        len(observed_fold_train_counts.get(fold, [])) == 1
        and int(observed_fold_train_counts[fold][0]) == expected
        for fold, expected in EXPECTED_FOLD_TRAIN_COUNTS.items()
    ) and set(fold_manifest["validation_count"].astype(int).unique().tolist()) == {
        EXPECTED_VALIDATION_SIZE
    }
    if not exact_fold_schedule_ok:
        raise AssertionError("Electricity router rolling-fold schedule changed")
    fold_lookup = fold_manifest.set_index(["series_id", "fold"])
    fold_chronology_ok = True
    fold_final_block_ok = True
    for _, group in fold_manifest.groupby("series_id", sort=False):
        group = group.sort_values("fold")
        fold_chronology_ok &= bool(
            (group["train_last_time_index"] < group["validation_first_time_index"]).all()
            and group["validation_first_time_index"].is_monotonic_increasing
        )
        fold_final_block_ok &= bool(
            group.iloc[-1]["validation_is_final_router_block"]
        )

    series_index_lookup = {
        str(series_id): group.index.to_numpy(dtype=np.int64)
        for series_id, group in router.groupby(
            "series_id", sort=False, observed=True
        )
    }
    series_order = list(series_index_lookup)
    if len(series_order) != EXPECTED_SERIES:
        raise AssertionError(
            f"Expected {EXPECTED_SERIES} router series, got {len(series_order)}"
        )

    records: list[dict[str, object]] = []
    sampling_manifest_records: list[dict[str, object]] = []
    sample_position_frames: list[pd.DataFrame] = []
    expected_sample_rows_by_fold: dict[int, int] = {}
    all_series_sampled = True
    validation_complete = True
    all_models_converged = True
    overall_start = perf_counter()

    for fold in range(1, N_FOLDS + 1):
        train_indices_by_series: list[np.ndarray] = []
        validation_indices_by_series: list[np.ndarray] = []
        for series_id in series_order:
            indices = series_index_lookup[series_id]
            schedule = fold_lookup.loc[(series_id, fold)]
            train_count = int(schedule["train_count"])
            validation_count = int(schedule["validation_count"])
            train_indices_by_series.append(indices[:train_count])
            validation_indices_by_series.append(
                indices[train_count : train_count + validation_count]
            )
        full_train_indices = np.concatenate(train_indices_by_series)
        validation_indices = np.concatenate(validation_indices_by_series)
        validation_lengths = np.asarray(
            [len(values) for values in validation_indices_by_series],
            dtype=np.int64,
        )
        sampled_train_indices, global_sample_positions = systematic_sample_indices(
            train_indices_by_series, MAX_TUNING_TRAIN_ROWS
        )
        expected_sample_rows = min(
            len(full_train_indices), MAX_TUNING_TRAIN_ROWS
        )
        if expected_sample_rows != EXPECTED_SAMPLE_ROWS_PER_FOLD:
            raise AssertionError("Every Electricity router fold must reach the sample cap")
        expected_sample_rows_by_fold[fold] = expected_sample_rows
        if (
            len(sampled_train_indices) != expected_sample_rows
            or len(global_sample_positions) != expected_sample_rows
        ):
            raise AssertionError("Router training sample size mismatch")

        sampled_set = set(sampled_train_indices.tolist())
        sample_cursor = 0
        for series_id, series_train_indices in zip(
            series_order, train_indices_by_series
        ):
            sampled_for_series = np.asarray(
                [
                    index
                    for index in series_train_indices
                    if int(index) in sampled_set
                ],
                dtype=np.int64,
            )
            all_series_sampled &= len(sampled_for_series) > 0
            schedule = fold_lookup.loc[(series_id, fold)]
            sampling_manifest_records.append(
                {
                    "dataset_id": DATASET_ID,
                    "fold": fold,
                    "series_id": series_id,
                    "available_train_rows": len(series_train_indices),
                    "sampled_train_rows": len(sampled_for_series),
                    "first_sampled_time_index": int(
                        router.loc[sampled_for_series[0], "time_index"]
                    ),
                    "last_sampled_time_index": int(
                        router.loc[sampled_for_series[-1], "time_index"]
                    ),
                    "train_last_time_index": int(schedule["train_last_time_index"]),
                    "sampling_method": SAMPLING_METHOD,
                    "sample_cap_per_fold": MAX_TUNING_TRAIN_ROWS,
                }
            )
            sampled_rows = router.loc[
                sampled_for_series, ["series_id", "time_index"]
            ].copy()
            sampled_rows.insert(0, "dataset_id", DATASET_ID)
            sampled_rows.insert(1, "fold", fold)
            sampled_rows["sample_order_within_fold"] = np.arange(
                sample_cursor,
                sample_cursor + len(sampled_rows),
                dtype=np.int32,
            )
            sample_cursor += len(sampled_rows)
            sample_position_frames.append(sampled_rows)
        if sample_cursor != expected_sample_rows:
            raise AssertionError("Per-series sample reconstruction changed row count")

        validation_complete &= len(validation_indices) == int(
            fold_manifest.loc[
                fold_manifest["fold"] == fold, "validation_count"
            ].sum()
        )
        validation_complete &= len(validation_indices) == EXPECTED_VALIDATION_ROWS_PER_FOLD
        validation_y_true = router.loc[
            validation_indices, "y_true"
        ].to_numpy(dtype=np.float64)
        validation_ridge = router.loc[
            validation_indices, "ridge_prediction"
        ].to_numpy(dtype=np.float64)
        validation_lgbm = router.loc[
            validation_indices, "lightgbm_prediction"
        ].to_numpy(dtype=np.float64)
        validation_loss_scale = router.loc[
            validation_indices, "seasonal_naive_mae_scale"
        ].to_numpy(dtype=np.float64)
        validation_advantage = router.loc[
            validation_indices, "loss_advantage_black_box"
        ].to_numpy(dtype=np.float64)
        validation_hard_target = router.loc[
            validation_indices, "hard_black_box_target"
        ].to_numpy(dtype=np.int8)

        for maximum_lag in residual_lags:
            feature_names = features_for_lag(
                maximum_lag, context_features, residual_features
            )
            scaler = StandardScaler()
            scaler.fit(
                router.loc[full_train_indices, feature_names].to_numpy(
                    dtype=np.float32
                )
            )
            train_x = scaler.transform(
                router.loc[sampled_train_indices, feature_names].to_numpy(
                    dtype=np.float32
                )
            ).astype(np.float32)
            validation_x = scaler.transform(
                router.loc[validation_indices, feature_names].to_numpy(
                    dtype=np.float32
                )
            ).astype(np.float32)
            duplicated_x = np.vstack([train_x, train_x])
            duplicated_y = np.concatenate(
                [
                    np.ones(len(train_x), dtype=np.int8),
                    np.zeros(len(train_x), dtype=np.int8),
                ]
            )
            train_advantage = router.loc[
                sampled_train_indices, "loss_advantage_black_box"
            ].to_numpy(dtype=np.float64)

            for temperature, c_value in product(temperatures, c_grid):
                train_soft_target = stable_soft_target(train_advantage, temperature)
                validation_soft_target = stable_soft_target(
                    validation_advantage, temperature
                )
                duplicated_weight = np.concatenate(
                    [train_soft_target, 1.0 - train_soft_target]
                )
                model = LogisticRegression(
                    solver="liblinear",
                    l1_ratio=1.0,
                    C=c_value,
                    fit_intercept=True,
                    max_iter=3000,
                    random_state=seed,
                )
                model.fit(
                    duplicated_x,
                    duplicated_y,
                    sample_weight=duplicated_weight,
                )
                converged = bool(int(model.n_iter_[0]) < model.max_iter)
                all_models_converged &= converged
                probability = np.clip(
                    model.predict_proba(validation_x)[:, 1],
                    1e-8,
                    1.0 - 1e-8,
                )
                threshold = float(
                    np.quantile(probability, target_simple_coverage)
                )
                use_black_box = probability >= threshold
                selected_prediction = np.where(
                    use_black_box, validation_lgbm, validation_ridge
                )
                scaled_error = (
                    (validation_y_true - selected_prediction)
                    / validation_loss_scale
                ) ** 2
                series_equal_loss = equal_series_mean(
                    scaled_error, validation_lengths
                )
                soft_log_loss = float(
                    np.mean(
                        -validation_soft_target * np.log(probability)
                        - (1.0 - validation_soft_target)
                        * np.log(1.0 - probability)
                    )
                )
                records.append(
                    {
                        "dataset_id": DATASET_ID,
                        "fold": fold,
                        "training_scope": "router_train_prefixes_only",
                        "validation_scope": "complete_later_router_train_blocks",
                        "temperature": temperature,
                        "residual_lag": maximum_lag,
                        "C": c_value,
                        "feature_count": len(feature_names),
                        "available_train_rows": len(full_train_indices),
                        "sampled_train_rows": len(sampled_train_indices),
                        "scaler_fit_rows": len(full_train_indices),
                        "validation_rows": len(validation_indices),
                        "validation_size_min": int(validation_lengths.min()),
                        "validation_size_max": int(validation_lengths.max()),
                        "constrained_scaled_loss": series_equal_loss,
                        "pooled_constrained_scaled_loss": float(
                            np.mean(scaled_error)
                        ),
                        "simple_coverage": float(np.mean(~use_black_box)),
                        "soft_log_loss": soft_log_loss,
                        "soft_brier": float(
                            np.mean(
                                (probability - validation_soft_target) ** 2
                            )
                        ),
                        "hard_brier": float(
                            brier_score_loss(validation_hard_target, probability)
                        ),
                        "black_box_AUPRC": float(
                            average_precision_score(
                                validation_hard_target, probability
                            )
                        ),
                        "nonzero_coefficients": int(
                            np.count_nonzero(model.coef_)
                        ),
                        "converged": converged,
                        "seed": seed,
                    }
                )
            print(
                f"[完成] fold={fold}/{N_FOLDS}, residual_lag={maximum_lag}",
                flush=True,
            )

    elapsed = perf_counter() - overall_start
    details = pd.DataFrame(records)
    tuning_summary = (
        details.groupby(["temperature", "residual_lag", "C"], as_index=False)
        .agg(
            mean_constrained_scaled_loss=("constrained_scaled_loss", "mean"),
            std_constrained_scaled_loss=("constrained_scaled_loss", "std"),
            mean_pooled_constrained_scaled_loss=(
                "pooled_constrained_scaled_loss", "mean"
            ),
            mean_simple_coverage=("simple_coverage", "mean"),
            mean_soft_log_loss=("soft_log_loss", "mean"),
            mean_soft_brier=("soft_brier", "mean"),
            mean_hard_brier=("hard_brier", "mean"),
            mean_black_box_AUPRC=("black_box_AUPRC", "mean"),
            mean_nonzero_coefficients=("nonzero_coefficients", "mean"),
            folds=("fold", "nunique"),
        )
        .sort_values(
            [
                "mean_constrained_scaled_loss",
                "mean_soft_brier",
                "residual_lag",
                "C",
                "temperature",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    best = tuning_summary.iloc[0]
    selected_feature_names = features_for_lag(
        int(best["residual_lag"]), context_features, residual_features
    )
    sampling_manifest = pd.DataFrame(sampling_manifest_records)
    sample_positions = pd.concat(sample_position_frames, ignore_index=True)
    sample_positions["dataset_id"] = sample_positions["dataset_id"].astype(
        "category"
    )
    sample_positions["series_id"] = sample_positions["series_id"].astype(
        "category"
    )

    expected_fit_count = (
        N_FOLDS * len(residual_lags) * len(temperatures) * len(c_grid)
    )
    expected_summary_count = len(residual_lags) * len(temperatures) * len(c_grid)
    if expected_fit_count != EXPECTED_MODEL_FITS:
        raise AssertionError(f"Expected model-fit count changed: {expected_fit_count}")
    if expected_summary_count != EXPECTED_PARAMETER_COMBINATIONS:
        raise AssertionError(
            f"Expected parameter-combination count changed: {expected_summary_count}"
        )
    expected_sample_position_rows = int(sum(expected_sample_rows_by_fold.values()))
    actual_sample_rows_by_fold = (
        sample_positions.groupby("fold", observed=True).size().to_dict()
    )
    exact_sample_counts_ok = all(
        int(actual_sample_rows_by_fold.get(fold, -1)) == expected
        for fold, expected in expected_sample_rows_by_fold.items()
    )
    check_items: list[tuple[str, bool, str]] = [
        (
            "router_feature_audit_passed",
            bool(passed_column(feature_checks["passed"]).all()),
            f"checks={len(feature_checks)}",
        ),
        ("tuning_input_contains_only_router_train", observed_splits == {"router_train"}, str(observed_splits)),
        ("calibration_not_read_for_tuning", "calibration" not in observed_splits, str(observed_splits)),
        ("test_not_read_for_tuning", "test" not in observed_splits, str(observed_splits)),
        ("all_321_series_are_present", len(series_order) == EXPECTED_SERIES, str(len(series_order))),
        (
            "router_row_count_matches_registered_feature_output",
            len(router) == EXPECTED_ROUTER_ROWS,
            f"actual={len(router)}; expected={EXPECTED_ROUTER_ROWS}",
        ),
        (
            "every_series_has_registered_router_row_count",
            bool(
                len(router_rows_per_series) == EXPECTED_SERIES
                and np.all(
                    router_rows_per_series == EXPECTED_ROUTER_ROWS_PER_SERIES
                )
            ),
            f"expected_each={EXPECTED_ROUTER_ROWS_PER_SERIES}",
        ),
        ("per_series_router_times_are_contiguous", per_series_time_contiguous, str(per_series_time_contiguous)),
        ("fold_schedule_is_chronological", fold_chronology_ok, str(fold_chronology_ok)),
        ("last_fold_ends_at_router_train_end", fold_final_block_ok, str(fold_final_block_ok)),
        (
            "fold_schedule_matches_registered_electricity_snapshot",
            exact_fold_schedule_ok,
            (
                f"train_counts={EXPECTED_FOLD_TRAIN_COUNTS}; "
                f"validation_count={EXPECTED_VALIDATION_SIZE}"
            ),
        ),
        ("all_series_are_in_every_training_sample", all_series_sampled, str(all_series_sampled)),
        (
            "training_sample_count_is_minimum_of_available_and_cap",
            len(sample_positions) == expected_sample_position_rows and exact_sample_counts_ok,
            f"actual={len(sample_positions)}; expected={expected_sample_position_rows}",
        ),
        (
            "every_fold_uses_exactly_100000_training_rows",
            set(actual_sample_rows_by_fold.values())
            == {EXPECTED_SAMPLE_ROWS_PER_FOLD}
            and len(actual_sample_rows_by_fold) == N_FOLDS,
            str(actual_sample_rows_by_fold),
        ),
        (
            "sampling_manifest_has_every_series_fold",
            len(sampling_manifest) == EXPECTED_SERIES * N_FOLDS,
            f"rows={len(sampling_manifest)}",
        ),
        (
            "sample_positions_are_unique_per_fold",
            not sample_positions.duplicated(["fold", "series_id", "time_index"]).any(),
            f"rows={len(sample_positions)}",
        ),
        (
            "complete_validation_blocks_are_used",
            validation_complete,
            f"passed={validation_complete}; expected_per_fold={EXPECTED_VALIDATION_ROWS_PER_FOLD}",
        ),
        (
            "expected_number_of_model_fits",
            len(details) == expected_fit_count,
            f"actual={len(details)}; expected={expected_fit_count}",
        ),
        (
            "expected_number_of_parameter_combinations",
            len(tuning_summary) == expected_summary_count,
            f"actual={len(tuning_summary)}; expected={expected_summary_count}",
        ),
        (
            "every_configuration_has_three_folds",
            bool((tuning_summary["folds"] == N_FOLDS).all()),
            f"combinations={len(tuning_summary)}",
        ),
        (
            "all_models_converged",
            all_models_converged and bool(details["converged"].all()),
            str(all_models_converged),
        ),
        (
            "all_metrics_are_finite",
            bool(
                np.isfinite(
                    details[
                        [
                            "constrained_scaled_loss",
                            "simple_coverage",
                            "soft_brier",
                            "hard_brier",
                            "black_box_AUPRC",
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
        raise AssertionError(f"Electricity Hourly soft-router tuning failed: {message}")

    details.to_csv(DETAIL_PATH, index=False)
    tuning_summary.to_csv(SUMMARY_PATH, index=False)
    fold_manifest.to_csv(FOLD_PATH, index=False)
    sampling_manifest.to_csv(SAMPLING_MANIFEST_PATH, index=False)
    sample_positions.to_parquet(
        SAMPLING_POSITIONS_PATH, index=False, compression="snappy"
    )
    checks.to_csv(CHECKS_PATH, index=False)

    selected = {
        "dataset_id": DATASET_ID,
        "model": "l1_logistic_soft_label_duplication",
        "selection_scope": "router_train_only",
        "selection_metric": "equal-series coverage-constrained scaled loss",
        "target_simple_coverage": target_simple_coverage,
        "rolling_folds": N_FOLDS,
        "validation_schedule": {
            "maximum_size_per_fold_hours": MAX_VALIDATION_SIZE,
            "block_unit_hours": VALIDATION_BLOCK_UNIT,
            "minimum_initial_router_rows": MIN_INITIAL_ROUTER_ROWS,
            "per_series_validation_size_minimum": int(
                fold_manifest["validation_count"].min()
            ),
            "per_series_validation_size_maximum": int(
                fold_manifest["validation_count"].max()
            ),
        },
        "training_sample": {
            "maximum_rows_per_fold": MAX_TUNING_TRAIN_ROWS,
            "sample_count_rule": "min(all available prefix rows, 100000)",
            "method": SAMPLING_METHOD,
            "all_series_represented": True,
            "exact_positions_file": str(SAMPLING_POSITIONS_PATH),
            "feature_standardizer_fit_scope": "all allowed training-prefix rows",
        },
        "parameter_grid": {
            "temperatures": temperatures,
            "residual_lags": residual_lags,
            "C": c_grid,
        },
        "temperature": float(best["temperature"]),
        "residual_lag": int(best["residual_lag"]),
        "C": float(best["C"]),
        "feature_count": len(selected_feature_names),
        "mean_validation_constrained_scaled_loss": float(
            best["mean_constrained_scaled_loss"]
        ),
        "mean_validation_simple_coverage": float(
            best["mean_simple_coverage"]
        ),
        "mean_validation_black_box_AUPRC": float(
            best["mean_black_box_AUPRC"]
        ),
        "tie_break_order": [
            "mean_constrained_scaled_loss",
            "mean_soft_brier",
            "residual_lag",
            "C",
            "temperature",
        ],
        "feature_names": selected_feature_names,
        "calibration_accessed": False,
        "test_accessed": False,
        "seed": seed,
    }
    SELECTED_PATH.write_text(
        yaml.safe_dump(selected, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    top = tuning_summary.head(15).copy().iloc[::-1]
    labels = [
        f"T={row.temperature:g}, lag={int(row.residual_lag)}, C={row.C:g}"
        for row in top.itertuples(index=False)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    axes[0].barh(
        labels, top["mean_constrained_scaled_loss"], color="#4C78A8"
    )
    axes[0].set_title("Top 15 router-train configurations")
    axes[0].set_xlabel("Equal-series constrained scaled loss")
    axes[0].grid(axis="x", alpha=0.2)

    lag_effect = (
        tuning_summary.groupby("residual_lag", as_index=False)
        .agg(
            best_loss=("mean_constrained_scaled_loss", "min"),
            median_loss=("mean_constrained_scaled_loss", "median"),
            best_AUPRC=("mean_black_box_AUPRC", "max"),
        )
        .sort_values("residual_lag")
    )
    axes[1].plot(
        lag_effect["residual_lag"],
        lag_effect["best_loss"],
        marker="o",
        label="best constrained loss",
        color="#59A14F",
    )
    axes[1].plot(
        lag_effect["residual_lag"],
        lag_effect["median_loss"],
        marker="s",
        label="median constrained loss",
        color="#F28E2B",
    )
    axes[1].set_xticks(residual_lags)
    axes[1].set_title("Effect of residual-history length")
    axes[1].set_xlabel("Maximum residual lag (hours)")
    axes[1].set_ylabel("Equal-series constrained scaled loss")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)

    fig.suptitle("Electricity Hourly soft-router tuning within router_train only", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = {
        "status": "passed",
        "dataset_id": DATASET_ID,
        "check_count": int(len(checks)),
        "failed_check_count": 0,
        "tuning_scope": "router_train_only",
        "calibration_accessed": False,
        "formal_test_accessed": False,
        "rolling_folds": N_FOLDS,
        "parameter_combinations": int(len(tuning_summary)),
        "model_fits": int(len(details)),
        "training_sample_cap": MAX_TUNING_TRAIN_ROWS,
        "selected_temperature": float(best["temperature"]),
        "selected_residual_lag": int(best["residual_lag"]),
        "selected_C": float(best["C"]),
        "selected_feature_count": len(selected_feature_names),
        "selected_mean_constrained_scaled_loss": float(
            best["mean_constrained_scaled_loss"]
        ),
        "runtime_seconds": float(elapsed),
        "outputs": {
            "details": str(DETAIL_PATH),
            "summary": str(SUMMARY_PATH),
            "selected": str(SELECTED_PATH),
            "fold_manifest": str(FOLD_PATH),
            "sampling_manifest": str(SAMPLING_MANIFEST_PATH),
            "sample_positions": str(SAMPLING_POSITIONS_PATH),
            "checks": str(CHECKS_PATH),
            "figure": str(FIGURE_PATH),
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print()
    print("Electricity Hourly 软路由器滚动调参全部通过")
    print("调参数据范围：仅 router_train")
    print("calibration 是否用于调参：否")
    print("test 是否访问：否")
    print("滚动验证折数：", N_FOLDS)
    print(
        "每条序列每折验证长度范围：",
        f"{fold_manifest['validation_count'].min()} 至 "
        f"{fold_manifest['validation_count'].max()} 小时",
    )
    print("每折训练抽样上限：", MAX_TUNING_TRAIN_ROWS)
    print("样本不足上限时：使用该折全部合格训练行")
    print("参数组合数量：", len(tuning_summary))
    print("实际模型拟合次数：", len(details))
    print("最佳温度：", float(best["temperature"]))
    print("最佳残差滞后：", int(best["residual_lag"]))
    print("最佳 C：", float(best["C"]))
    print("最佳特征数量：", len(selected_feature_names))
    print(
        "平均约束缩放损失：",
        f"{float(best['mean_constrained_scaled_loss']):.6f}",
    )
    print(
        "平均简单模型覆盖率：",
        f"{float(best['mean_simple_coverage']):.6f}",
    )
    print(
        "平均 LightGBM AUPRC：",
        f"{float(best['mean_black_box_AUPRC']):.6f}",
    )
    print("运行秒数：", f"{elapsed:.2f}")
    print("逐折结果：", DETAIL_PATH)
    print("汇总结果：", SUMMARY_PATH)
    print("选定参数：", SELECTED_PATH)
    print("调参图片：", FIGURE_PATH)


if __name__ == "__main__":
    main()
