#!/usr/bin/env python3
"""Causal context and past-residual features for Weather Daily routing."""

from __future__ import annotations

import gc
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

DATA_PATH = PROJECT_ROOT / "data/processed/weather_daily_long.parquet"
PREDICTION_PATH = PROJECT_ROOT / "results/weather_daily_pretest_predictions.parquet"
SCALER_PATH = PROJECT_ROOT / "results/weather_daily_scaler_parameters.csv"
CONFIG_PATH = PROJECT_ROOT / "experiment_config.yaml"
BASE_CHECKS_PATH = PROJECT_ROOT / "results/weather_daily_base_model_fit_checks.csv"
BASE_METADATA_PATH = PROJECT_ROOT / "results/weather_daily_base_model_fit_metadata.yaml"
SPLIT_MANIFEST_PATH = PROJECT_ROOT / "results/weather_daily_split_manifest.csv"
SPLIT_SUMMARY_PATH = PROJECT_ROOT / "results/weather_daily_split_summary.yaml"
SAMPLE_REGISTRATION_PATH = PROJECT_ROOT / "results/weather_daily_sample_registration.yaml"

FEATURE_PATH = OUTPUT_ROOT / "results/weather_daily_router_features.parquet"
MANIFEST_PATH = OUTPUT_ROOT / "results/weather_daily_router_feature_manifest.csv"
COUNT_PATH = OUTPUT_ROOT / "results/weather_daily_router_feature_counts.csv"
CHECKS_PATH = OUTPUT_ROOT / "results/weather_daily_router_feature_checks.csv"
SUMMARY_PATH = OUTPUT_ROOT / "results/weather_daily_router_feature_summary.yaml"
FIGURE_PATH = OUTPUT_ROOT / "figures/weather_daily_router_target_distribution.png"
REPORT_PATH = OUTPUT_ROOT / "logs/weather_daily_router_feature_report.json"

DATASET_ID = "weather_daily"
SAMPLE_ID = "5518ada55d6adb8a8df20f5a30d52ec904362ce57df9adeeceb0bc3d22854044"
ALLOWED_RAW_SPLITS = ["base_train", "router_train", "calibration"]
ROUTING_SPLITS = ["router_train", "calibration"]
SERIES_TYPES = ["maxtemp", "mintemp", "rain", "solar"]
SEASONAL_PERIOD = 7
LONG_CONTEXT = 28
EXPECTED_SERIES = 500
EXPECTED_RAW_ROWS = 6_264_251
EXPECTED_PREDICTION_ROWS = 1_842_491
EXPECTED_ROUTER_ROWS_AFTER_WARMUP = 1_097_616
EXPECTED_CALIBRATION_ROWS = 736_875
EXPECTED_RETAINED_ROWS = 1_834_491


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def passed_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1"})


def rolling_past_mean_std(
    values: np.ndarray, targets: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Population mean/std over [target-window, target), never including target."""

    values = np.asarray(values, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    prefix = np.empty(len(values) + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(values, out=prefix[1:])
    squared_prefix = np.empty(len(values) + 1, dtype=np.float64)
    squared_prefix[0] = 0.0
    np.cumsum(values * values, out=squared_prefix[1:])
    sums = prefix[targets] - prefix[targets - window]
    squared_sums = squared_prefix[targets] - squared_prefix[targets - window]
    means = sums / window
    variances = np.maximum(squared_sums / window - means * means, 0.0)
    return means, np.sqrt(variances)


def main() -> None:
    input_paths = (
        DATA_PATH,
        PREDICTION_PATH,
        SCALER_PATH,
        CONFIG_PATH,
        BASE_CHECKS_PATH,
        BASE_METADATA_PATH,
        SPLIT_MANIFEST_PATH,
        SPLIT_SUMMARY_PATH,
        SAMPLE_REGISTRATION_PATH,
    )
    for path in input_paths:
        require_file(path)
    output_paths = (
        FEATURE_PATH,
        MANIFEST_PATH,
        COUNT_PATH,
        CHECKS_PATH,
        SUMMARY_PATH,
        FIGURE_PATH,
        REPORT_PATH,
    )
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    base_checks = pd.read_csv(BASE_CHECKS_PATH)
    base_metadata = yaml.safe_load(BASE_METADATA_PATH.read_text(encoding="utf-8"))
    split_summary = yaml.safe_load(SPLIT_SUMMARY_PATH.read_text(encoding="utf-8"))
    registration = yaml.safe_load(
        SAMPLE_REGISTRATION_PATH.read_text(encoding="utf-8")
    )
    split_manifest = pd.read_csv(SPLIT_MANIFEST_PATH).sort_values("sample_order")
    scalers = pd.read_csv(SCALER_PATH).sort_values("sample_order")
    residual_lag_candidates = [
        int(value) for value in config["soft_router"]["residual_lag_grid"]
    ]
    if residual_lag_candidates != [1, 4, 8, 16]:
        raise AssertionError(f"Unexpected residual lag grid: {residual_lag_candidates}")
    max_residual_lag = max(residual_lag_candidates)
    if not passed_column(base_checks["passed"]).all():
        raise AssertionError("Base-model fit checks did not all pass")
    if bool(base_metadata.get("formal_test_accessed", True)):
        raise AssertionError("Base-model metadata reports formal-test access")
    if base_metadata.get("training_scope") != "base_train_only":
        raise AssertionError("Base models were not trained on the frozen base scope")
    if list(base_metadata.get("prediction_splits", [])) != ROUTING_SPLITS:
        raise AssertionError("Base-model prediction scope changed")
    if scalers["source_split"].ne("base_train").any():
        raise AssertionError("Scaler source is not base_train only")

    sample_ids = {
        str(base_metadata.get("sample_id")),
        str(split_summary.get("sample_id")),
        str(registration.get("sample_id")),
        *split_manifest["sample_id"].astype(str).unique().tolist(),
        *scalers["sample_id"].astype(str).unique().tolist(),
    }
    if sample_ids != {SAMPLE_ID}:
        raise AssertionError(f"Frozen sample IDs disagree: {sample_ids}")

    # Predicate pushdown prevents formal-test values from entering this process.
    raw = pd.read_parquet(
        DATA_PATH,
        columns=["series_id", "series_type", "time_index", "value", "split"],
        filters=[[('split', '==', name)] for name in ALLOWED_RAW_SPLITS],
    )
    observed_raw_splits = set(raw["split"].astype(str).unique().tolist())
    if observed_raw_splits != set(ALLOWED_RAW_SPLITS):
        raise AssertionError(f"Unexpected raw-data splits: {observed_raw_splits}")
    expected_raw_rows = int(
        sum(split_summary["aggregate_counts"][name] for name in ALLOWED_RAW_SPLITS)
    )
    if expected_raw_rows != EXPECTED_RAW_ROWS or len(raw) != EXPECTED_RAW_ROWS:
        raise AssertionError(
            f"Raw pre-test row mismatch: actual={len(raw)}; registered={expected_raw_rows}"
        )
    observed_raw_rows = int(len(raw))

    prediction = pd.read_parquet(PREDICTION_PATH)
    required_prediction_columns = {
        "dataset_id",
        "sample_id",
        "series_id",
        "series_type",
        "time_index",
        "split",
        "y_true",
        "ridge_prediction",
        "lightgbm_prediction",
        "ridge_residual",
        "lightgbm_residual",
    }
    if not required_prediction_columns.issubset(prediction.columns):
        missing = sorted(required_prediction_columns.difference(prediction.columns))
        raise AssertionError(f"Prediction file is missing columns: {missing}")
    if set(prediction["sample_id"].astype(str).unique()) != {SAMPLE_ID}:
        raise AssertionError("Prediction file does not carry the frozen sample ID")
    sample_order_map = split_manifest.set_index("series_id")["sample_order"].to_dict()
    prediction["_sample_order"] = prediction["series_id"].astype(str).map(sample_order_map)
    if prediction["_sample_order"].isna().any():
        raise AssertionError("Prediction file contains an unregistered series")
    prediction = (
        prediction.sort_values(["_sample_order", "time_index"], kind="stable")
        .drop(columns="_sample_order")
        .reset_index(drop=True)
    )
    observed_prediction_splits = set(prediction["split"].astype(str).unique())
    if observed_prediction_splits != set(ROUTING_SPLITS):
        raise AssertionError(f"Unexpected prediction splits: {observed_prediction_splits}")
    expected_prediction_rows = int(
        sum(split_summary["aggregate_counts"][name] for name in ROUTING_SPLITS)
    )
    if (
        expected_prediction_rows != EXPECTED_PREDICTION_ROWS
        or len(prediction) != EXPECTED_PREDICTION_ROWS
    ):
        raise AssertionError(
            f"Prediction row mismatch: actual={len(prediction)}; registered={expected_prediction_rows}"
        )
    observed_prediction_rows = int(len(prediction))

    manifest_lookup = {
        str(row.series_id): row for row in split_manifest.itertuples(index=False)
    }
    median_lookup = {
        str(row.series_id): float(row.median) for row in scalers.itertuples(index=False)
    }
    scale_lookup = {
        str(row.series_id): float(row.scale_used) for row in scalers.itertuples(index=False)
    }
    raw_series: dict[str, np.ndarray] = {}
    seasonal_mae_scale: dict[str, float] = {}
    raw_type_lookup: dict[str, str] = {}
    raw_indices_contiguous = True
    exact_raw_manifest_counts = True
    series_types_match_manifest = True
    for series_id_raw, group in raw.groupby("series_id", sort=False, observed=True):
        series_id = str(series_id_raw)
        group = group.sort_values("time_index", kind="stable")
        row = manifest_lookup[series_id]
        indices = group["time_index"].to_numpy(dtype=np.int64)
        raw_indices_contiguous &= np.array_equal(indices, np.arange(len(group)))
        observed_counts = group["split"].astype(str).value_counts().to_dict()
        expected_counts = {
            "base_train": int(row.base_train_count),
            "router_train": int(row.router_train_count),
            "calibration": int(row.calibration_count),
        }
        exact_raw_manifest_counts &= observed_counts == expected_counts
        exact_raw_manifest_counts &= len(group) == sum(expected_counts.values())
        observed_types = group["series_type"].astype(str).unique().tolist()
        series_types_match_manifest &= (
            observed_types == [str(row.series_type)]
            and str(row.series_type) in SERIES_TYPES
        )
        raw_type_lookup[series_id] = str(row.series_type)
        values = group["value"].to_numpy(dtype=np.float64)
        raw_series[series_id] = values
        base_values = values[: int(row.base_train_count)]
        seasonal_difference = (
            base_values[SEASONAL_PERIOD:] - base_values[:-SEASONAL_PERIOD]
        )
        mae_scale = float(np.mean(np.abs(seasonal_difference)))
        if not np.isfinite(mae_scale) or mae_scale <= 1e-12:
            raise AssertionError(f"{series_id} has a zero seasonal MAE scale")
        seasonal_mae_scale[series_id] = mae_scale

    del raw
    gc.collect()
    if len(raw_series) != EXPECTED_SERIES or set(raw_series) != set(manifest_lookup):
        raise AssertionError("Raw data do not contain the frozen 500 Weather series")

    prediction["median_base"] = (
        prediction["series_id"].astype(str).map(median_lookup).astype(np.float64)
    )
    prediction["iqr_base"] = (
        prediction["series_id"].astype(str).map(scale_lookup).astype(np.float64)
    )
    prediction["seasonal_naive_mae_scale"] = (
        prediction["series_id"].astype(str).map(seasonal_mae_scale).astype(np.float64)
    )
    if prediction[["median_base", "iqr_base", "seasonal_naive_mae_scale"]].isna().any().any():
        raise AssertionError("A registered scale could not be mapped to predictions")

    history_context_names = [
        "last_value_scaled",
        "mean_7_scaled",
        "trend_7_scaled",
        "volatility_7_scaled",
        "trend_28_scaled",
        "volatility_28_scaled",
    ]
    context_arrays = {
        name: np.empty(len(prediction), dtype=np.float32)
        for name in history_context_names
    }
    prediction_targets_contiguous = True
    prediction_counts_match_manifest = True
    context_uses_only_past = True
    for series_id_raw, group in prediction.groupby(
        "series_id", sort=False, observed=True
    ):
        series_id = str(series_id_raw)
        row_indices = group.index.to_numpy(dtype=np.int64)
        targets = group["time_index"].to_numpy(dtype=np.int64)
        row = manifest_lookup[series_id]
        prediction_targets_contiguous &= bool(np.all(np.diff(targets) == 1))
        prediction_counts_match_manifest &= len(group) == (
            int(row.router_train_count) + int(row.calibration_count)
        )
        prediction_counts_match_manifest &= int(targets[0]) == int(row.router_train_start)
        prediction_counts_match_manifest &= int(targets[-1]) == int(row.calibration_end)
        context_uses_only_past &= bool(targets.min() >= LONG_CONTEXT + 1)
        values = raw_series[series_id]
        median = float(median_lookup[series_id])
        scale = float(scale_lookup[series_id])
        mean_7, std_7 = rolling_past_mean_std(values, targets, SEASONAL_PERIOD)
        _, std_28 = rolling_past_mean_std(values, targets, LONG_CONTEXT)
        context_arrays["last_value_scaled"][row_indices] = (
            (values[targets - 1] - median) / scale
        ).astype(np.float32)
        context_arrays["mean_7_scaled"][row_indices] = (
            (mean_7 - median) / scale
        ).astype(np.float32)
        context_arrays["trend_7_scaled"][row_indices] = (
            (values[targets - 1] - values[targets - 1 - SEASONAL_PERIOD]) / scale
        ).astype(np.float32)
        context_arrays["volatility_7_scaled"][row_indices] = (
            std_7 / scale
        ).astype(np.float32)
        context_arrays["trend_28_scaled"][row_indices] = (
            (values[targets - 1] - values[targets - 1 - LONG_CONTEXT]) / scale
        ).astype(np.float32)
        context_arrays["volatility_28_scaled"][row_indices] = (
            std_28 / scale
        ).astype(np.float32)

    prediction = pd.concat(
        [prediction, pd.DataFrame(context_arrays, index=prediction.index)], axis=1
    )
    del context_arrays
    gc.collect()

    prediction["ridge_prediction_scaled"] = (
        (prediction["ridge_prediction"] - prediction["median_base"])
        / prediction["iqr_base"]
    ).astype(np.float32)
    prediction["lightgbm_prediction_scaled"] = (
        (prediction["lightgbm_prediction"] - prediction["median_base"])
        / prediction["iqr_base"]
    ).astype(np.float32)
    prediction["prediction_difference_scaled"] = (
        (prediction["lightgbm_prediction"] - prediction["ridge_prediction"])
        / prediction["iqr_base"]
    ).astype(np.float32)
    prediction["absolute_prediction_difference_scaled"] = prediction[
        "prediction_difference_scaled"
    ].abs().astype(np.float32)

    type_feature_names = [f"series_type_{name}" for name in SERIES_TYPES]
    observed_prediction_types = set(prediction["series_type"].astype(str).unique())
    if observed_prediction_types != set(SERIES_TYPES):
        raise AssertionError(f"Unexpected prediction series types: {observed_prediction_types}")
    type_consistency_ok = True
    for series_type, feature_name in zip(SERIES_TYPES, type_feature_names):
        prediction[feature_name] = (
            prediction["series_type"].astype(str).eq(series_type).astype(np.float32)
        )
    type_sum = prediction[type_feature_names].sum(axis=1).to_numpy(dtype=np.float32)
    type_consistency_ok &= bool(np.all(type_sum == 1.0))
    for series_id_raw, group in prediction.groupby(
        "series_id", sort=False, observed=True
    ):
        series_id = str(series_id_raw)
        type_consistency_ok &= (
            group["series_type"].astype(str).nunique() == 1
            and str(group["series_type"].iloc[0]) == raw_type_lookup[series_id]
        )

    context_feature_names = [
        *history_context_names,
        "ridge_prediction_scaled",
        "lightgbm_prediction_scaled",
        "prediction_difference_scaled",
        "absolute_prediction_difference_scaled",
        *type_feature_names,
    ]
    if len(context_feature_names) != 14:
        raise AssertionError(f"Expected 14 context features, got {len(context_feature_names)}")

    grouped_prediction = prediction.groupby("series_id", sort=False, observed=True)
    residual_feature_data: dict[str, pd.Series] = {}
    residual_feature_names: list[str] = []
    for lag in range(1, max_residual_lag + 1):
        ridge_name = f"ridge_residual_lag_{lag}"
        lightgbm_name = f"lightgbm_residual_lag_{lag}"
        residual_feature_data[ridge_name] = (
            grouped_prediction["ridge_residual"].shift(lag) / prediction["iqr_base"]
        ).astype(np.float32)
        residual_feature_data[lightgbm_name] = (
            grouped_prediction["lightgbm_residual"].shift(lag) / prediction["iqr_base"]
        ).astype(np.float32)
        residual_feature_names.extend([ridge_name, lightgbm_name])
    prediction = pd.concat(
        [prediction, pd.DataFrame(residual_feature_data, index=prediction.index)],
        axis=1,
    )
    del residual_feature_data, grouped_prediction
    gc.collect()

    prediction["ridge_scaled_loss"] = (
        prediction["ridge_residual"] / prediction["seasonal_naive_mae_scale"]
    ) ** 2
    prediction["lightgbm_scaled_loss"] = (
        prediction["lightgbm_residual"] / prediction["seasonal_naive_mae_scale"]
    ) ** 2
    prediction["loss_advantage_black_box"] = (
        prediction["ridge_scaled_loss"] - prediction["lightgbm_scaled_loss"]
    )
    prediction["hard_black_box_target"] = (
        prediction["loss_advantage_black_box"] > 0
    ).astype(np.int8)

    feature_names = context_feature_names + residual_feature_names
    missing_before_warmup = prediction[residual_feature_names].isna().any(axis=1)
    retained_mask = ~missing_before_warmup
    features = prediction.loc[retained_mask].copy().reset_index(drop=True)
    all_features_finite = all(
        bool(np.isfinite(features[name].to_numpy(dtype=np.float32)).all())
        for name in feature_names
    )
    if not all_features_finite:
        raise AssertionError("Retained router features contain nonfinite values")

    removed = prediction.loc[missing_before_warmup]
    removed_counts = removed.groupby("series_id", observed=True).size()
    warmup_exact = bool(
        len(removed_counts) == EXPECTED_SERIES
        and (removed_counts == max_residual_lag).all()
        and set(removed["split"].astype(str).unique()) == {"router_train"}
    )
    removed_total = int(len(removed))
    counts = (
        features.groupby("split", as_index=False, observed=True)
        .agg(rows=("series_id", "size"), series_count=("series_id", "nunique"))
        .sort_values("split", kind="stable")
        .reset_index(drop=True)
    )
    expected_counts = {
        "router_train": int(
            split_summary["aggregate_counts"]["router_train"]
            - EXPECTED_SERIES * max_residual_lag
        ),
        "calibration": int(split_summary["aggregate_counts"]["calibration"]),
    }
    observed_counts = {
        str(row.split): int(row.rows) for row in counts.itertuples(index=False)
    }
    if expected_counts != {
        "router_train": EXPECTED_ROUTER_ROWS_AFTER_WARMUP,
        "calibration": EXPECTED_CALIBRATION_ROWS,
    }:
        raise AssertionError(f"Registered retained counts changed: {expected_counts}")
    if observed_counts != expected_counts or len(features) != EXPECTED_RETAINED_ROWS:
        raise AssertionError(
            f"Router feature counts differ: observed={observed_counts}; expected={expected_counts}"
        )

    retained_counts_match_manifest = True
    observed_retained_per_series = features.groupby("series_id", observed=True).size()
    for series_id, row in manifest_lookup.items():
        expected = int(row.router_train_count) + int(row.calibration_count) - max_residual_lag
        retained_counts_match_manifest &= int(observed_retained_per_series.loc[series_id]) == expected

    del prediction, removed, missing_before_warmup, retained_mask
    gc.collect()

    output_columns = [
        "dataset_id",
        "sample_id",
        "series_id",
        "series_type",
        "time_index",
        "split",
        "y_true",
        "ridge_prediction",
        "lightgbm_prediction",
        "ridge_residual",
        "lightgbm_residual",
        "ridge_scaled_loss",
        "lightgbm_scaled_loss",
        "loss_advantage_black_box",
        "hard_black_box_target",
        "seasonal_naive_mae_scale",
        *feature_names,
    ]
    features[output_columns].to_parquet(FEATURE_PATH, index=False, compression="snappy")
    counts.to_csv(COUNT_PATH, index=False)

    descriptions = {
        "last_value_scaled": "Most recent observed value, base-train median/IQR scaled",
        "mean_7_scaled": "Mean of the seven observations immediately before target",
        "trend_7_scaled": "Change from target-8 to target-1",
        "volatility_7_scaled": "Population standard deviation over previous seven days",
        "trend_28_scaled": "Change from target-29 to target-1",
        "volatility_28_scaled": "Population standard deviation over previous 28 days",
        "ridge_prediction_scaled": "Current causal Ridge prediction, base-train scaled",
        "lightgbm_prediction_scaled": "Current causal LightGBM prediction, base-train scaled",
        "prediction_difference_scaled": "LightGBM minus Ridge prediction, base-train scaled",
        "absolute_prediction_difference_scaled": "Absolute current model disagreement",
        "series_type_maxtemp": "Known archive variable-type indicator: maximum temperature",
        "series_type_mintemp": "Known archive variable-type indicator: minimum temperature",
        "series_type_rain": "Known archive variable-type indicator: rainfall",
        "series_type_solar": "Known archive variable-type indicator: solar exposure",
    }
    manifest_records: list[dict[str, object]] = []
    for name in context_feature_names:
        if name.startswith("series_type_"):
            relative_time: float | int = np.nan
        elif name.endswith("prediction_scaled") or "prediction_difference" in name:
            relative_time = np.nan
        else:
            relative_time = -1
        manifest_records.append(
            {
                "feature": name,
                "group": "context",
                "available_before_target": True,
                "maximum_observed_value_time_relative_to_target": relative_time,
                "description": descriptions[name],
            }
        )
    for name in residual_feature_names:
        lag = int(name.rsplit("_", 1)[1])
        model_name = "Ridge" if name.startswith("ridge") else "LightGBM"
        manifest_records.append(
            {
                "feature": name,
                "group": "past_residual",
                "available_before_target": True,
                "maximum_observed_value_time_relative_to_target": -lag,
                "description": f"{model_name} residual from {lag} day(s) before target",
            }
        )
    manifest = pd.DataFrame(manifest_records)
    manifest.to_csv(MANIFEST_PATH, index=False)

    win_rate = (
        features.groupby("split", observed=True)["hard_black_box_target"]
        .mean()
        .reindex(ROUTING_SPLITS)
    )
    type_counts = split_manifest["series_type"].astype(str).value_counts().to_dict()
    expected_type_counts = {
        str(key): int(value)
        for key, value in split_summary["series_type_counts"].items()
    }
    check_items: list[tuple[str, bool, str]] = [
        ("frozen_sample_id_is_consistent", sample_ids == {SAMPLE_ID}, SAMPLE_ID),
        (
            "base_model_fit_audit_passed",
            bool(passed_column(base_checks["passed"]).all()),
            f"checks={len(base_checks)}",
        ),
        ("raw_input_excludes_test", "test" not in observed_raw_splits, str(observed_raw_splits)),
        ("prediction_input_excludes_test", "test" not in observed_prediction_splits, str(observed_prediction_splits)),
        ("raw_row_count_matches_manifest", observed_raw_rows == expected_raw_rows, f"actual={observed_raw_rows}; expected={expected_raw_rows}"),
        ("prediction_row_count_matches_manifest", observed_prediction_rows == expected_prediction_rows, f"actual={observed_prediction_rows}; expected={expected_prediction_rows}"),
        ("all_500_registered_series_are_present", len(raw_series) == EXPECTED_SERIES, str(len(raw_series))),
        ("raw_split_counts_match_each_series_manifest", exact_raw_manifest_counts, str(exact_raw_manifest_counts)),
        ("prediction_counts_match_each_series_manifest", prediction_counts_match_manifest, str(prediction_counts_match_manifest)),
        ("series_types_match_frozen_manifest", series_types_match_manifest, str(series_types_match_manifest)),
        ("registered_type_allocation_is_unchanged", type_counts == expected_type_counts, str(type_counts)),
        ("raw_time_indices_are_contiguous", raw_indices_contiguous, str(raw_indices_contiguous)),
        ("prediction_targets_are_contiguous_per_series", prediction_targets_contiguous, str(prediction_targets_contiguous)),
        ("context_features_use_only_past_observations", context_uses_only_past, str(context_uses_only_past)),
        ("series_type_indicators_are_valid_and_preknown", type_consistency_ok, str(type_consistency_ok)),
        ("residual_shift_warmup_is_exact", warmup_exact, f"removed={removed_total}"),
        ("retained_feature_counts_match_expected", observed_counts == expected_counts, f"observed={observed_counts}; expected={expected_counts}"),
        ("variable_retained_counts_match_each_series_manifest", retained_counts_match_manifest, f"series={len(observed_retained_per_series)}"),
        (
            "feature_manifest_has_46_unique_features",
            len(manifest) == 46 and len(feature_names) == 46 and len(set(feature_names)) == 46,
            f"context={len(context_feature_names)}; residual={len(residual_feature_names)}; total={len(feature_names)}",
        ),
        (
            "all_manifest_features_are_available_before_target",
            bool(manifest["available_before_target"].all()),
            f"features={len(manifest)}",
        ),
        ("all_retained_features_are_finite", all_features_finite, str(all_features_finite)),
    ]
    checks = pd.DataFrame(check_items, columns=["check", "passed", "detail"])
    failed = checks.loc[~checks["passed"]]
    if not failed.empty:
        message = "; ".join(
            f"{row.check}: {row.detail}" for row in failed.itertuples(index=False)
        )
        raise AssertionError(f"Weather Daily router-feature audit failed: {message}")
    checks.to_csv(CHECKS_PATH, index=False)

    summary = {
        "dataset_id": DATASET_ID,
        "sample_id": SAMPLE_ID,
        "router_feature_generation_passed": True,
        "context_feature_count": len(context_feature_names),
        "residual_feature_count": len(residual_feature_names),
        "total_feature_count": len(feature_names),
        "context_protocol": {
            "past_observation_features": len(history_context_names),
            "current_causal_prediction_features": 4,
            "preknown_series_type_indicators": len(type_feature_names),
            "calendar_features": 0,
            "calendar_feature_reason": "Weather archive has no reliable timestamps; none were fabricated",
        },
        "residual_lag_candidates_days": residual_lag_candidates,
        "maximum_residual_lag_days": max_residual_lag,
        "warmup_rows_removed_per_series": max_residual_lag,
        "retained_counts": observed_counts,
        "black_box_point_win_rate": {
            split_name: float(win_rate.loc[split_name]) for split_name in ROUTING_SPLITS
        },
        "seasonal_loss_scale": "base_train mean absolute seven-day difference",
        "future_information_check_passed": True,
        "formal_test_accessed": False,
    }
    SUMMARY_PATH.write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes[0, 0].bar(win_rate.index, win_rate.values, color=["#4C78A8", "#F28E2B"])
    axes[0, 0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel("LightGBM pointwise win rate")
    axes[0, 0].set_title("Routing target balance")

    router = features.loc[features["split"] == "router_train"]
    lower, upper = router["loss_advantage_black_box"].quantile([0.01, 0.99])
    axes[0, 1].hist(
        router["loss_advantage_black_box"].clip(lower, upper),
        bins=70,
        color="#76B7B2",
        edgecolor="none",
    )
    axes[0, 1].axvline(0, color="#E15759", linestyle="--", linewidth=1)
    axes[0, 1].set_xlabel("Scaled loss advantage of LightGBM")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Router-train advantage (1st–99th percentile clipped)")

    sample_size = min(6000, len(router))
    sample_indices = np.floor(
        (np.arange(sample_size, dtype=float) + 0.5) * len(router) / sample_size
    ).astype(int)
    sample = router.iloc[sample_indices]
    axes[1, 0].scatter(
        sample["absolute_prediction_difference_scaled"],
        sample["loss_advantage_black_box"].clip(lower, upper),
        c=sample["hard_black_box_target"],
        cmap="coolwarm",
        s=8,
        alpha=0.35,
    )
    axes[1, 0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_xlabel("Absolute scaled model disagreement")
    axes[1, 0].set_ylabel("Scaled LightGBM loss advantage (clipped)")
    axes[1, 0].set_title("Model disagreement and routing advantage")

    type_win_rate = (
        router.groupby("series_type", observed=True)["hard_black_box_target"]
        .mean()
        .reindex(SERIES_TYPES)
    )
    axes[1, 1].bar(type_win_rate.index, type_win_rate.values, color="#59A14F")
    axes[1, 1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel("LightGBM pointwise win rate")
    axes[1, 1].set_title("Router-train target balance by weather variable")
    axes[1, 1].grid(axis="y", alpha=0.2)

    fig.suptitle("Weather Daily causal routing features and targets", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = {
        "status": "passed",
        "dataset_id": DATASET_ID,
        "sample_id": SAMPLE_ID,
        "check_count": int(len(checks)),
        "failed_check_count": 0,
        "context_features": len(context_feature_names),
        "residual_features": len(residual_feature_names),
        "total_features": len(feature_names),
        "retained_counts": observed_counts,
        "black_box_win_rate": {
            split_name: float(win_rate.loc[split_name]) for split_name in ROUTING_SPLITS
        },
        "future_information_check": "passed",
        "formal_test_accessed": False,
        "outputs": {
            "features": str(FEATURE_PATH),
            "manifest": str(MANIFEST_PATH),
            "counts": str(COUNT_PATH),
            "checks": str(CHECKS_PATH),
            "summary": str(SUMMARY_PATH),
            "figure": str(FIGURE_PATH),
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("Weather Daily 路由特征生成全部通过")
    print("固定样本编号：", SAMPLE_ID)
    print("最大残差滞后：", max_residual_lag, "天")
    print("上下文特征数量：", len(context_feature_names))
    print("其中变量类型指示特征：", len(type_feature_names))
    print("残差特征数量：", len(residual_feature_names))
    print("路由特征总数：", len(feature_names))
    print("每条序列删除的残差预热行数：", max_residual_lag)
    print("未来信息检查：通过")
    print("各数据段样本数量：")
    print(counts.to_string(index=False))
    print("各数据段 LightGBM 单点胜率：")
    print(win_rate.to_string())
    print("路由特征：", FEATURE_PATH)
    print("特征清单：", MANIFEST_PATH)
    print("样本计数：", COUNT_PATH)
    print("目标分布图：", FIGURE_PATH)


if __name__ == "__main__":
    main()
