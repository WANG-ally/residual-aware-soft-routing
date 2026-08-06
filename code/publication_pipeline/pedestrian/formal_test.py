#!/usr/bin/env python3
"""Freeze or execute the one-time Pedestrian formal evaluator.

``--freeze-evaluator`` verifies the final pretest lock and performs a complete
calibration-only dry run.  It does not read the test split.  The exact evaluator
SHA-256 is then bound to the final pretest freeze in an authorization receipt.

``--execute-final-test`` is deliberately separate.  It first writes a permanent
access receipt and only then reads the test split.  A second run is prohibited."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sci-routing")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
import yaml


PROJECT_ROOT = Path(
    os.environ.get("SCI_ROUTING_ROOT", Path(__file__).resolve().parents[3])
).resolve()
OUTPUT_ROOT = Path(os.environ.get("SCI_ROUTING_OUTPUT_ROOT", PROJECT_ROOT)).resolve()
EVALUATOR_PATH = Path(__file__).resolve()

DATASET_ID = "pedestrian_hourly"
EXPECTED_FINAL_FREEZE_ID = (
    "786a38ac96473411d414faac48f32af7f0d690b3915cc266349bf0a0eb86ee07"
)
EXPECTED_SERIES = 66
EXPECTED_CALIBRATION_ROWS = 313_219
EXPECTED_CALIBRATION_STEPS = 9_642
EXPECTED_TEST_ROWS = 469_884
EXPECTED_TEST_STEPS = 14_464
SEASONAL_PERIOD = 24
LONG_CONTEXT = 168
MAX_RESIDUAL_LAG = 16
ROLLING_COVERAGE_HOURS = 168

FINAL_LOCK_PATH = (
    PROJECT_ROOT / "results/pedestrian_final_pretest_lock_manifest.json"
)
AUTHORIZATION_PLAN_PATH = (
    PROJECT_ROOT / "results/pedestrian_final_test_authorization_plan.yaml"
)
DATA_PATH = PROJECT_ROOT / "data/processed/pedestrian_hourly_long.parquet"
SPLIT_MANIFEST_PATH = PROJECT_ROOT / "results/pedestrian_split_manifest.csv"
SCALER_PATH = PROJECT_ROOT / "results/pedestrian_scaler_parameters.csv"
PRETEST_PATH = PROJECT_ROOT / "results/pedestrian_pretest_predictions.parquet"
ROUTER_FEATURE_PATH = PROJECT_ROOT / "results/pedestrian_router_features.parquet"
CALIBRATION_ROUTER_PATH = (
    PROJECT_ROOT / "results/pedestrian_calibration_router_scores.parquet"
)
CALIBRATION_CONTROLLER_PATH = (
    PROJECT_ROOT / "results/pedestrian_calibration_controller_decisions.parquet"
)
BASELINE_CALIBRATION_PATH = (
    PROJECT_ROOT / "results/pedestrian_baseline_calibration_scores.parquet"
)
RIDGE_PARAM_PATH = PROJECT_ROOT / "results/pedestrian_selected_ridge_params.yaml"
LGBM_PARAM_PATH = (
    PROJECT_ROOT / "results/pedestrian_selected_lightgbm_params.yaml"
)
CONTROLLER_PATH = (
    PROJECT_ROOT / "results/pedestrian_selected_coverage_controller.yaml"
)
BASELINE_PARAM_PATH = (
    PROJECT_ROOT / "results/pedestrian_selected_baseline_params.yaml"
)
RIDGE_MODEL_PATH = PROJECT_ROOT / "models/pedestrian_ridge.joblib"
LGBM_MODEL_PATH = PROJECT_ROOT / "models/pedestrian_lightgbm.joblib"
FULL_ROUTER_MODEL_PATH = PROJECT_ROOT / "models/pedestrian_soft_router.joblib"

AUTHORIZATION_PATH = OUTPUT_ROOT / "logs/pedestrian_evaluator_authorization.json"
RECEIPT_PATH = OUTPUT_ROOT / "logs/pedestrian_formal_test_access_receipt.json"
PREDICTION_PATH = OUTPUT_ROOT / "results/pedestrian_test_predictions.parquet"
PER_SERIES_PATH = OUTPUT_ROOT / "results/pedestrian_test_per_series_metrics.csv"
AGGREGATE_PATH = OUTPUT_ROOT / "results/pedestrian_test_aggregate_metrics.csv"
COVERAGE_TRACE_PATH = OUTPUT_ROOT / "results/pedestrian_test_coverage_trace.csv"
FIGURE_PATH = OUTPUT_ROOT / "figures/pedestrian_test_method_comparison.png"

METHODS = [
    "seasonal_naive",
    "ridge_only",
    "lightgbm_only",
    "equal_weight_average",
    "hard_aalf_like_router",
    "hard_logistic_same_features",
    "hard_random_forest_same_features",
    "class_weight_only",
    "soft_targets_only",
    "residual_features_only",
    "static_full_router",
    "adaptive_full_router",
    "unconstrained_oracle",
    "coverage_constrained_oracle",
]

for path in (
    AUTHORIZATION_PATH,
    RECEIPT_PATH,
    PREDICTION_PATH,
    PER_SERIES_PATH,
    AGGREGATE_PATH,
    COVERAGE_TRACE_PATH,
    FIGURE_PATH,
):
    path.parent.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def maximum_absolute_difference(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"Array shape mismatch: {left.shape} != {right.shape}")
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left - right)))


def verify_final_lock() -> dict:
    if not FINAL_LOCK_PATH.is_file():
        raise FileNotFoundError(f"Final pretest lock is missing: {FINAL_LOCK_PATH}")
    with FINAL_LOCK_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    changed: list[str] = []
    current_records: list[dict[str, object]] = []
    for item in manifest["files"]:
        path = PROJECT_ROOT / item["path"]
        if not path.is_file():
            changed.append(item["path"])
            continue
        observed_hash = sha256_file(path)
        observed_size = int(path.stat().st_size)
        if (
            observed_hash != item["sha256"]
            or observed_size != int(item["size_bytes"])
        ):
            changed.append(item["path"])
        current_records.append(
            {
                "path": item["path"],
                "size_bytes": observed_size,
                "sha256": observed_hash,
            }
        )
    if changed:
        raise ValueError(f"Final frozen files changed: {changed}")
    canonical = json.dumps(
        current_records, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    observed_id = hashlib.sha256(canonical).hexdigest()
    if (
        manifest.get("dataset_id") != DATASET_ID
        or manifest.get("final_freeze_id") != EXPECTED_FINAL_FREEZE_ID
        or observed_id != EXPECTED_FINAL_FREEZE_ID
        or int(manifest.get("file_count", -1)) != 82
        or manifest.get("status") != "READY_FOR_EVALUATOR_FREEZE"
        or manifest.get("formal_test_authorized") is not False
        or int(manifest.get("formal_test_runs_completed", -1)) != 0
    ):
        raise ValueError(
            "Final pretest manifest identity/status is invalid: "
            f"recorded={manifest.get('final_freeze_id')}; observed={observed_id}"
        )
    return manifest


def load_artifacts() -> dict:
    required = [
        AUTHORIZATION_PLAN_PATH,
        DATA_PATH,
        SPLIT_MANIFEST_PATH,
        SCALER_PATH,
        PRETEST_PATH,
        ROUTER_FEATURE_PATH,
        CALIBRATION_ROUTER_PATH,
        CALIBRATION_CONTROLLER_PATH,
        BASELINE_CALIBRATION_PATH,
        RIDGE_PARAM_PATH,
        LGBM_PARAM_PATH,
        CONTROLLER_PATH,
        BASELINE_PARAM_PATH,
        RIDGE_MODEL_PATH,
        LGBM_MODEL_PATH,
        FULL_ROUTER_MODEL_PATH,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required evaluator artifacts are missing: {missing}")

    plan = load_yaml(AUTHORIZATION_PLAN_PATH)
    ridge_params = load_yaml(RIDGE_PARAM_PATH)
    lgbm_params = load_yaml(LGBM_PARAM_PATH)
    controller = load_yaml(CONTROLLER_PATH)
    baseline_params = load_yaml(BASELINE_PARAM_PATH)
    full_router = joblib.load(FULL_ROUTER_MODEL_PATH)
    baseline_bundles: dict[str, dict] = {}
    for method, metadata in baseline_params["methods"].items():
        if method == "hard_aalf_like_router":
            continue
        model_path = PROJECT_ROOT / metadata["model_file"]
        if sha256_file(model_path) != metadata["model_sha256"]:
            raise ValueError(f"Baseline model hash changed: {method}")
        baseline_bundles[method] = joblib.load(model_path)

    if (
        plan.get("formal_test_authorized") is not False
        or int(plan.get("formal_test_runs_completed", -1)) != 0
        or plan.get("status") != "all_pretest_models_frozen_evaluator_pending"
        or controller.get("test_accessed") is not False
        or baseline_params.get("test_accessed") is not False
        or full_router.get("test_accessed") is not False
    ):
        raise ValueError("A frozen artifact reports test access or invalid status")
    if (
        int(ridge_params["selected_window"]) != LONG_CONTEXT
        or int(lgbm_params["selected_window"]) != LONG_CONTEXT
        or int(full_router["residual_lag"]) != MAX_RESIDUAL_LAG
        or len(full_router["feature_names"]) != 46
        or float(controller["primary_target_simple_coverage"]) != 0.7
        or controller["new_segment_initialization"]["bias"] != 0.0
        or controller["new_segment_initialization"][
            "calibration_final_bias_is_not_carried_to_test"
        ]
        is not True
    ):
        raise ValueError("Frozen Pedestrian parameters differ from the test plan")

    scalers = pd.read_csv(SCALER_PATH)
    split_manifest = pd.read_csv(SPLIT_MANIFEST_PATH)
    pretest = pd.read_parquet(PRETEST_PATH)
    if (
        scalers["source_split"].ne("base_train").any()
        or set(pretest["split"].astype(str)) != {"router_train", "calibration"}
        or len(pretest) != 783_080
    ):
        raise ValueError("Scaler or pretest-history scope is invalid")

    return {
        "plan": plan,
        "ridge_params": ridge_params,
        "lgbm_params": lgbm_params,
        "controller": controller,
        "baseline_params": baseline_params,
        "ridge_model": joblib.load(RIDGE_MODEL_PATH),
        "lgbm_model": joblib.load(LGBM_MODEL_PATH),
        "full_router": full_router,
        "baseline_bundles": baseline_bundles,
        "scalers": scalers,
        "split_manifest": split_manifest,
        "pretest": pretest,
    }


def read_raw_splits(split_names: list[str]) -> pd.DataFrame:
    raw = pd.read_parquet(
        DATA_PATH,
        columns=[
            "dataset_id",
            "series_id",
            "time_index",
            "timestamp",
            "value",
            "split",
        ],
        filters=[[('split', '==', name)] for name in split_names],
    )
    observed = set(raw["split"].astype(str).unique())
    if observed != set(split_names):
        raise ValueError(f"Raw split scope is invalid: {sorted(observed)}")
    return raw


def rolling_past_mean_std(
    values: np.ndarray, targets: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    prefix = np.empty(len(values) + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(values, out=prefix[1:])
    squared_prefix = np.empty(len(values) + 1, dtype=np.float64)
    squared_prefix[0] = 0.0
    np.cumsum(values * values, out=squared_prefix[1:])
    sums = prefix[targets] - prefix[targets - window]
    squared_sums = (
        squared_prefix[targets] - squared_prefix[targets - window]
    )
    means = sums / window
    variances = np.maximum(squared_sums / window - means * means, 0.0)
    return means, np.sqrt(variances)


def build_segment(
    raw: pd.DataFrame, segment_name: str, artifacts: dict
) -> pd.DataFrame:
    """Build causal predictions/features for a variable-length segment.

    Base-model targets are vectorized within each series, but every feature at
    target ``t`` is indexed strictly from values/residuals at ``t-1`` or earlier.
    """

    allowed_segment_names = {"calibration", "test"}
    if segment_name not in allowed_segment_names:
        raise ValueError(f"Unsupported segment: {segment_name}")
    frames = {
        str(series_id): group.sort_values("time_index", kind="stable").reset_index(
            drop=True
        )
        for series_id, group in raw.groupby(
            "series_id", sort=False, observed=True
        )
    }
    if len(frames) != EXPECTED_SERIES:
        raise ValueError(f"Expected 66 series, observed {len(frames)}")

    scaler_lookup = {
        str(row.series_id): (float(row.median), float(row.scale_used))
        for row in artifacts["scalers"].itertuples(index=False)
    }
    if set(frames) != set(scaler_lookup):
        raise ValueError("Raw-data and scaler series IDs differ")
    history = artifacts["pretest"].copy()
    history["series_id"] = history["series_id"].astype(str)
    history_lookup = {
        str(series_id): group.sort_values("time_index", kind="stable")
        for series_id, group in history.groupby(
            "series_id", sort=False, observed=True
        )
    }

    ridge_window = int(artifacts["ridge_params"]["selected_window"])
    lgbm_window = int(artifacts["lgbm_params"]["selected_window"])
    ridge_model = artifacts["ridge_model"]
    lgbm_model = artifacts["lgbm_model"]
    parts: list[pd.DataFrame] = []

    for series_id, frame in frames.items():
        indices = frame["time_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(indices, np.arange(len(frame), dtype=np.int64)):
            raise ValueError(f"{series_id} has non-contiguous time indices")
        segment_mask = frame["split"].astype(str).eq(segment_name).to_numpy()
        targets = frame.loc[segment_mask, "time_index"].to_numpy(dtype=np.int64)
        if len(targets) == 0 or not np.all(np.diff(targets) == 1):
            raise ValueError(f"{series_id} has an invalid {segment_name} block")
        if targets.min() < max(ridge_window, lgbm_window, LONG_CONTEXT + 1):
            raise ValueError(f"{series_id} lacks causal history")

        values = frame["value"].to_numpy(dtype=np.float64)
        base_values = frame.loc[
            frame["split"].astype(str).eq("base_train"), "value"
        ].to_numpy(dtype=np.float64)
        seasonal_difference = (
            base_values[SEASONAL_PERIOD:] - base_values[:-SEASONAL_PERIOD]
        )
        rmsse_denominator = float(np.mean(seasonal_difference**2))
        mase_denominator = float(np.mean(np.abs(seasonal_difference)))
        if (
            not np.isfinite(rmsse_denominator)
            or not np.isfinite(mase_denominator)
            or rmsse_denominator <= 1e-12
            or mase_denominator <= 1e-12
        ):
            raise ValueError(f"{series_id} has a zero seasonal denominator")

        median, scale = scaler_lookup[series_id]
        scaled = (values - median) / scale
        ridge_view = sliding_window_view(scaled, ridge_window)
        ridge_x = ridge_view[targets - ridge_window, ::-1]
        ridge_prediction = ridge_model.predict(ridge_x) * scale + median
        if lgbm_window == ridge_window:
            lgbm_x = ridge_x.astype(np.float32, copy=True)
        else:
            lgbm_view = sliding_window_view(scaled, lgbm_window)
            lgbm_x = lgbm_view[
                targets - lgbm_window, ::-1
            ].astype(np.float32, copy=True)
        lgbm_prediction = (
            lgbm_model.booster_.predict(lgbm_x) * scale + median
        )
        y_true = values[targets]
        ridge_residual = y_true - ridge_prediction
        lgbm_residual = y_true - lgbm_prediction

        mean_24, std_24 = rolling_past_mean_std(
            values, targets, SEASONAL_PERIOD
        )
        _, std_168 = rolling_past_mean_std(values, targets, LONG_CONTEXT)
        timestamp = pd.to_datetime(frame.loc[segment_mask, "timestamp"])
        hour = timestamp.dt.hour.to_numpy(dtype=np.float64)
        weekday = timestamp.dt.dayofweek.to_numpy(dtype=np.float64)

        part = frame.loc[
            segment_mask,
            ["dataset_id", "series_id", "time_index", "timestamp", "split"],
        ].copy()
        part["series_id"] = part["series_id"].astype(str)
        part["relative_segment_step"] = np.arange(len(part), dtype=np.int64)
        part["y_true"] = y_true
        part["ridge_prediction"] = ridge_prediction
        part["lightgbm_prediction"] = lgbm_prediction
        part["seasonal_naive_prediction"] = values[targets - SEASONAL_PERIOD]
        part["RMSSE_denominator"] = rmsse_denominator
        part["MASE_denominator"] = mase_denominator
        part["seasonal_naive_mae_scale"] = mase_denominator
        part["ridge_residual"] = ridge_residual
        part["lightgbm_residual"] = lgbm_residual
        part["last_value_scaled"] = (
            (values[targets - 1] - median) / scale
        ).astype(np.float32)
        part["mean_24_scaled"] = ((mean_24 - median) / scale).astype(np.float32)
        part["trend_24_scaled"] = (
            (values[targets - 1] - values[targets - 1 - SEASONAL_PERIOD])
            / scale
        ).astype(np.float32)
        part["volatility_24_scaled"] = (std_24 / scale).astype(np.float32)
        part["trend_168_scaled"] = (
            (values[targets - 1] - values[targets - 1 - LONG_CONTEXT])
            / scale
        ).astype(np.float32)
        part["volatility_168_scaled"] = (std_168 / scale).astype(np.float32)
        part["ridge_prediction_scaled"] = (
            (ridge_prediction - median) / scale
        ).astype(np.float32)
        part["lightgbm_prediction_scaled"] = (
            (lgbm_prediction - median) / scale
        ).astype(np.float32)
        part["prediction_difference_scaled"] = (
            (lgbm_prediction - ridge_prediction) / scale
        ).astype(np.float32)
        part["absolute_prediction_difference_scaled"] = np.abs(
            part["prediction_difference_scaled"].to_numpy(dtype=np.float32)
        ).astype(np.float32)
        part["hour_of_day_sin"] = np.sin(2.0 * np.pi * hour / 24.0).astype(
            np.float32
        )
        part["hour_of_day_cos"] = np.cos(2.0 * np.pi * hour / 24.0).astype(
            np.float32
        )
        part["day_of_week_sin"] = np.sin(
            2.0 * np.pi * weekday / 7.0
        ).astype(np.float32)
        part["day_of_week_cos"] = np.cos(
            2.0 * np.pi * weekday / 7.0
        ).astype(np.float32)

        ridge_history = np.full(len(values), np.nan, dtype=np.float64)
        lgbm_history = np.full(len(values), np.nan, dtype=np.float64)
        previous = history_lookup[series_id]
        previous = previous.loc[previous["time_index"] < int(targets[0])]
        previous_indices = previous["time_index"].to_numpy(dtype=np.int64)
        ridge_history[previous_indices] = previous["ridge_residual"].to_numpy(
            dtype=np.float64
        )
        lgbm_history[previous_indices] = previous[
            "lightgbm_residual"
        ].to_numpy(dtype=np.float64)
        ridge_history[targets] = ridge_residual
        lgbm_history[targets] = lgbm_residual
        for lag in range(1, MAX_RESIDUAL_LAG + 1):
            ridge_lag = ridge_history[targets - lag]
            lgbm_lag = lgbm_history[targets - lag]
            if not np.isfinite(ridge_lag).all() or not np.isfinite(lgbm_lag).all():
                raise ValueError(
                    f"{series_id} lacks residual history for lag {lag}"
                )
            part[f"ridge_residual_lag_{lag}"] = (
                ridge_lag / scale
            ).astype(np.float32)
            part[f"lightgbm_residual_lag_{lag}"] = (
                lgbm_lag / scale
            ).astype(np.float32)
        parts.append(part)

    segment = pd.concat(parts, ignore_index=True)
    segment = segment.sort_values(
        ["relative_segment_step", "series_id"], kind="stable"
    ).reset_index(drop=True)
    if (
        segment["dataset_id"].ne(DATASET_ID).any()
        or set(segment["split"].astype(str)) != {segment_name}
        or segment["series_id"].nunique() != EXPECTED_SERIES
    ):
        raise ValueError(f"Built {segment_name} segment has an invalid identity")
    feature_names = artifacts["full_router"]["feature_names"]
    if (
        len(feature_names) != 46
        or not np.isfinite(segment[feature_names].to_numpy(dtype=float)).all()
    ):
        raise ValueError("Built router feature matrix is invalid")
    return segment


def apply_methods(frame: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    frame = frame.copy().reset_index(drop=True)
    controller = artifacts["controller"]
    target_coverage = float(controller["primary_target_simple_coverage"])
    base_threshold = float(controller["primary_base_threshold"])
    eta = float(controller["selected_eta"])
    bias_limit = float(controller["bias_limit"])

    full_bundle = artifacts["full_router"]
    full_x = full_bundle["scaler"].transform(
        frame[full_bundle["feature_names"]].to_numpy(dtype=np.float64)
    )
    frame["full_router_probability"] = np.clip(
        full_bundle["model"].predict_proba(full_x)[:, 1],
        1e-8,
        1.0 - 1e-8,
    )

    for method, bundle in artifacts["baseline_bundles"].items():
        x_value = bundle["scaler"].transform(
            frame[bundle["feature_names"]].to_numpy(dtype=np.float64)
        )
        frame[f"{method}_probability"] = np.clip(
            bundle["model"].predict_proba(x_value)[:, 1],
            1e-8,
            1.0 - 1e-8,
        )

    aalf = artifacts["baseline_params"]["methods"]["hard_aalf_like_router"]
    aalf_lag = int(aalf["residual_lag"])
    ridge_columns = [
        f"ridge_residual_lag_{lag}" for lag in range(1, aalf_lag + 1)
    ]
    lgbm_columns = [
        f"lightgbm_residual_lag_{lag}" for lag in range(1, aalf_lag + 1)
    ]
    frame["hard_aalf_like_router_score"] = (
        np.mean(frame[ridge_columns].to_numpy(dtype=np.float64) ** 2, axis=1)
        - np.mean(frame[lgbm_columns].to_numpy(dtype=np.float64) ** 2, axis=1)
    )

    for method, bundle in artifacts["baseline_bundles"].items():
        threshold = float(bundle["thresholds"][str(target_coverage)])
        use_ridge = frame[f"{method}_probability"].to_numpy() < threshold
        frame[f"use_ridge_{method}"] = use_ridge
        frame[f"prediction_{method}"] = np.where(
            use_ridge,
            frame["ridge_prediction"],
            frame["lightgbm_prediction"],
        )

    aalf_threshold = float(aalf["thresholds"][str(target_coverage)])
    frame["use_ridge_hard_aalf_like_router"] = (
        frame["hard_aalf_like_router_score"].to_numpy() < aalf_threshold
    )
    frame["prediction_hard_aalf_like_router"] = np.where(
        frame["use_ridge_hard_aalf_like_router"],
        frame["ridge_prediction"],
        frame["lightgbm_prediction"],
    )

    frame["use_ridge_static_full_router"] = (
        frame["full_router_probability"].to_numpy() < base_threshold
    )
    frame["prediction_static_full_router"] = np.where(
        frame["use_ridge_static_full_router"],
        frame["ridge_prediction"],
        frame["lightgbm_prediction"],
    )

    probability = frame["full_router_probability"].to_numpy(dtype=np.float64)
    adaptive_use_ridge = np.zeros(len(frame), dtype=bool)
    effective_thresholds = np.empty(len(frame), dtype=np.float64)
    biases_before = np.empty(len(frame), dtype=np.float64)
    biases_after = np.empty(len(frame), dtype=np.float64)
    bias = 0.0
    for _, positions in frame.groupby(
        "relative_segment_step", sort=True, observed=True
    ).indices.items():
        positions = np.asarray(positions, dtype=np.int64)
        effective_threshold = float(np.clip(base_threshold + bias, 0.0, 1.0))
        batch_use_ridge = probability[positions] < effective_threshold
        batch_coverage = float(np.mean(batch_use_ridge))
        next_bias = float(
            np.clip(
                bias + eta * (target_coverage - batch_coverage),
                -bias_limit,
                bias_limit,
            )
        )
        adaptive_use_ridge[positions] = batch_use_ridge
        effective_thresholds[positions] = effective_threshold
        biases_before[positions] = bias
        biases_after[positions] = next_bias
        bias = next_bias
    frame["use_ridge_adaptive_full_router"] = adaptive_use_ridge
    frame["adaptive_effective_threshold"] = effective_thresholds
    frame["adaptive_bias_before"] = biases_before
    frame["adaptive_bias_after"] = biases_after
    frame["prediction_adaptive_full_router"] = np.where(
        adaptive_use_ridge,
        frame["ridge_prediction"],
        frame["lightgbm_prediction"],
    )

    frame["prediction_ridge_only"] = frame["ridge_prediction"]
    frame["prediction_lightgbm_only"] = frame["lightgbm_prediction"]
    frame["prediction_seasonal_naive"] = frame["seasonal_naive_prediction"]
    frame["prediction_equal_weight_average"] = 0.5 * (
        frame["ridge_prediction"] + frame["lightgbm_prediction"]
    )
    frame["use_ridge_ridge_only"] = True
    frame["use_ridge_lightgbm_only"] = False

    ridge_squared_error = (
        frame["y_true"] - frame["ridge_prediction"]
    ) ** 2
    lgbm_squared_error = (
        frame["y_true"] - frame["lightgbm_prediction"]
    ) ** 2
    frame["hard_black_box_target"] = (
        lgbm_squared_error < ridge_squared_error
    ).astype(np.int8)
    frame["use_ridge_unconstrained_oracle"] = (
        ridge_squared_error <= lgbm_squared_error
    )
    frame["prediction_unconstrained_oracle"] = np.where(
        frame["use_ridge_unconstrained_oracle"],
        frame["ridge_prediction"],
        frame["lightgbm_prediction"],
    )

    advantage = (
        (frame["y_true"] - frame["ridge_prediction"]) ** 2
        - (frame["y_true"] - frame["lightgbm_prediction"]) ** 2
    ) / (frame["seasonal_naive_mae_scale"] ** 2)
    simple_count = int(round(target_coverage * len(frame)))
    black_box_count = len(frame) - simple_count
    order = np.argsort(-advantage.to_numpy(dtype=np.float64), kind="mergesort")
    use_black_box = np.zeros(len(frame), dtype=bool)
    use_black_box[order[:black_box_count]] = True
    frame["use_ridge_coverage_constrained_oracle"] = ~use_black_box
    frame["prediction_coverage_constrained_oracle"] = np.where(
        frame["use_ridge_coverage_constrained_oracle"],
        frame["ridge_prediction"],
        frame["lightgbm_prediction"],
    )
    return frame


def expected_calibration_differences(computed: pd.DataFrame) -> dict[str, float]:
    expected_prediction = pd.read_parquet(
        PRETEST_PATH, filters=[("split", "==", "calibration")]
    )
    expected_features = pd.read_parquet(
        ROUTER_FEATURE_PATH, filters=[("split", "==", "calibration")]
    )
    expected_router = pd.read_parquet(CALIBRATION_ROUTER_PATH)
    expected_baselines = pd.read_parquet(BASELINE_CALIBRATION_PATH)
    expected_controller = pd.read_parquet(CALIBRATION_CONTROLLER_PATH)
    keys = ["series_id", "time_index"]

    frames = [
        computed,
        expected_prediction,
        expected_features,
        expected_router,
        expected_baselines,
        expected_controller,
    ]
    sorted_frames: list[pd.DataFrame] = []
    for frame in frames:
        frame = frame.copy()
        frame["series_id"] = frame["series_id"].astype(str)
        sorted_frames.append(
            frame.sort_values(keys, kind="stable").reset_index(drop=True)
        )
    (
        calculated,
        expected_prediction,
        expected_features,
        expected_router,
        expected_baselines,
        expected_controller,
    ) = sorted_frames

    if len(calculated) != EXPECTED_CALIBRATION_ROWS:
        raise ValueError(f"Calibration dry-run row count is {len(calculated)}")
    for expected in sorted_frames[1:]:
        if not calculated[keys].equals(expected[keys]):
            raise ValueError("Calibration dry-run keys differ from frozen evidence")

    full_features = [
        "last_value_scaled",
        "mean_24_scaled",
        "trend_24_scaled",
        "volatility_24_scaled",
        "trend_168_scaled",
        "volatility_168_scaled",
        "ridge_prediction_scaled",
        "lightgbm_prediction_scaled",
        "prediction_difference_scaled",
        "absolute_prediction_difference_scaled",
        "hour_of_day_sin",
        "hour_of_day_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        *[
            name
            for lag in range(1, MAX_RESIDUAL_LAG + 1)
            for name in (
                f"ridge_residual_lag_{lag}",
                f"lightgbm_residual_lag_{lag}",
            )
        ],
    ]
    baseline_columns = [
        "hard_logistic_same_features_probability",
        "hard_random_forest_same_features_probability",
        "class_weight_only_probability",
        "soft_targets_only_probability",
        "residual_features_only_probability",
        "hard_aalf_like_router_score",
    ]
    differences = {
        "ridge_prediction": maximum_absolute_difference(
            calculated["ridge_prediction"],
            expected_prediction["ridge_prediction"],
        ),
        "lightgbm_prediction": maximum_absolute_difference(
            calculated["lightgbm_prediction"],
            expected_prediction["lightgbm_prediction"],
        ),
        "router_features": maximum_absolute_difference(
            calculated[full_features].to_numpy(dtype=np.float64),
            expected_features[full_features].to_numpy(dtype=np.float64),
        ),
        "full_router_probability": maximum_absolute_difference(
            calculated["full_router_probability"],
            expected_router["black_box_probability"],
        ),
        "baseline_scores": maximum_absolute_difference(
            calculated[baseline_columns].to_numpy(dtype=np.float64),
            expected_baselines[baseline_columns].to_numpy(dtype=np.float64),
        ),
        "controller_effective_threshold": maximum_absolute_difference(
            calculated["adaptive_effective_threshold"],
            expected_controller["effective_threshold"],
        ),
        "controller_bias_before": maximum_absolute_difference(
            calculated["adaptive_bias_before"],
            expected_controller["bias_before_decision"],
        ),
        "controller_bias_after": maximum_absolute_difference(
            calculated["adaptive_bias_after"],
            expected_controller["bias_after_update"],
        ),
        "controller_selected_prediction": maximum_absolute_difference(
            calculated["prediction_adaptive_full_router"],
            expected_controller["selected_prediction"],
        ),
    }
    decision_equal = np.array_equal(
        calculated["use_ridge_adaptive_full_router"].to_numpy(dtype=bool),
        expected_controller["use_ridge"].to_numpy(dtype=bool),
    )
    relative_step_equal = np.array_equal(
        calculated["relative_segment_step"].to_numpy(dtype=np.int64),
        expected_controller["relative_calibration_step"].to_numpy(dtype=np.int64),
    )
    differences["controller_decision_mismatch_count"] = float(
        0 if decision_equal else np.sum(
            calculated["use_ridge_adaptive_full_router"].to_numpy(dtype=bool)
            != expected_controller["use_ridge"].to_numpy(dtype=bool)
        )
    )
    differences["relative_step_mismatch_count"] = float(
        0 if relative_step_equal else 1
    )
    return differences


def dry_run_calibration(artifacts: dict) -> dict:
    raw = read_raw_splits(["base_train", "router_train", "calibration"])
    if "test" in set(raw["split"].astype(str)):
        raise ValueError("Calibration dry run unexpectedly loaded test")
    computed = build_segment(raw, "calibration", artifacts)
    if (
        len(computed) != EXPECTED_CALIBRATION_ROWS
        or computed["series_id"].nunique() != EXPECTED_SERIES
        or computed["relative_segment_step"].nunique()
        != EXPECTED_CALIBRATION_STEPS
        or set(computed["split"].astype(str)) != {"calibration"}
    ):
        raise ValueError("Calibration dry-run segment identity is invalid")
    computed = apply_methods(computed, artifacts)
    differences = expected_calibration_differences(computed)
    tolerance = 1e-7
    if any(value > tolerance for value in differences.values()):
        raise ValueError(
            "Calibration reconstruction differs from frozen evidence: "
            f"{differences}"
        )
    return {
        "rows": len(computed),
        "series_count": int(computed["series_id"].nunique()),
        "relative_steps": int(computed["relative_segment_step"].nunique()),
        "minimum_active_series": int(
            computed.groupby("relative_segment_step", observed=True).size().min()
        ),
        "maximum_active_series": int(
            computed.groupby("relative_segment_step", observed=True).size().max()
        ),
        "tolerance": tolerance,
        "maximum_differences": differences,
        "test_accessed": False,
    }


def expected_calibration_error(
    probability: np.ndarray, hard_target: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (
                (probability >= edges[index])
                & (probability <= edges[index + 1])
            )
        else:
            mask = (
                (probability >= edges[index])
                & (probability < edges[index + 1])
            )
        if np.any(mask):
            result += float(np.mean(mask)) * abs(
                float(np.mean(probability[mask]))
                - float(np.mean(hard_target[mask]))
            )
    return float(result)


def compute_metrics(
    frame: pd.DataFrame, target_coverage: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_series_records: list[dict[str, object]] = []
    for series_id, group in frame.groupby(
        "series_id", sort=False, observed=True
    ):
        y_true = group["y_true"].to_numpy(dtype=np.float64)
        rmsse_denominator = float(group["RMSSE_denominator"].iloc[0])
        mase_denominator = float(group["MASE_denominator"].iloc[0])
        for method in METHODS:
            prediction = group[f"prediction_{method}"].to_numpy(dtype=np.float64)
            error = y_true - prediction
            smape_denominator = np.abs(y_true) + np.abs(prediction)
            smape_values = np.divide(
                200.0 * np.abs(error),
                smape_denominator,
                out=np.zeros_like(error),
                where=smape_denominator > 0.0,
            )
            use_column = f"use_ridge_{method}"
            coverage = (
                float(group[use_column].mean())
                if use_column in group.columns
                else np.nan
            )
            per_series_records.append(
                {
                    "dataset_id": DATASET_ID,
                    "series_id": str(series_id),
                    "test_rows": int(len(group)),
                    "method": method,
                    "RMSSE": float(
                        np.sqrt(np.mean(error**2) / rmsse_denominator)
                    ),
                    "MASE": float(np.mean(np.abs(error)) / mase_denominator),
                    "sMAPE": float(np.mean(smape_values)),
                    "RMSE": float(np.sqrt(np.mean(error**2))),
                    "MAE": float(np.mean(np.abs(error))),
                    "simple_coverage": coverage,
                }
            )
    per_series = pd.DataFrame(per_series_records)

    coverage_records: list[dict[str, object]] = []
    for method in METHODS:
        use_column = f"use_ridge_{method}"
        if use_column not in frame.columns:
            continue
        for relative_step, group in frame.groupby(
            "relative_segment_step", sort=True, observed=True
        ):
            simple_count = int(group[use_column].sum())
            batch_rows = int(len(group))
            coverage_records.append(
                {
                    "dataset_id": DATASET_ID,
                    "method": method,
                    "relative_test_step": int(relative_step),
                    "simple_count": simple_count,
                    "batch_rows": batch_rows,
                    "simple_coverage": float(simple_count / batch_rows),
                }
            )
    coverage_trace = pd.DataFrame(coverage_records)
    coverage_trace["rolling_168h_simple_count"] = coverage_trace.groupby(
        "method", sort=False, observed=True
    )["simple_count"].transform(
        lambda values: values.rolling(
            ROLLING_COVERAGE_HOURS, min_periods=ROLLING_COVERAGE_HOURS
        ).sum()
    )
    coverage_trace["rolling_168h_rows"] = coverage_trace.groupby(
        "method", sort=False, observed=True
    )["batch_rows"].transform(
        lambda values: values.rolling(
            ROLLING_COVERAGE_HOURS, min_periods=ROLLING_COVERAGE_HOURS
        ).sum()
    )
    coverage_trace["rolling_168h_simple_coverage"] = (
        coverage_trace["rolling_168h_simple_count"]
        / coverage_trace["rolling_168h_rows"]
    )

    score_map = {
        "hard_logistic_same_features": (
            "hard_logistic_same_features_probability",
            True,
        ),
        "hard_random_forest_same_features": (
            "hard_random_forest_same_features_probability",
            True,
        ),
        "class_weight_only": ("class_weight_only_probability", True),
        "soft_targets_only": ("soft_targets_only_probability", True),
        "residual_features_only": ("residual_features_only_probability", True),
        "hard_aalf_like_router": ("hard_aalf_like_router_score", False),
        "static_full_router": ("full_router_probability", True),
        "adaptive_full_router": ("full_router_probability", True),
    }
    y_true_all = frame["y_true"].to_numpy(dtype=np.float64)
    hard_target = frame["hard_black_box_target"].to_numpy(dtype=np.int8)
    aggregate_records: list[dict[str, object]] = []
    for method in METHODS:
        method_metrics = per_series.loc[per_series["method"] == method]
        prediction = frame[f"prediction_{method}"].to_numpy(dtype=np.float64)
        use_column = f"use_ridge_{method}"
        coverage = (
            float(frame[use_column].mean())
            if use_column in frame.columns
            else np.nan
        )
        method_trace = coverage_trace.loc[coverage_trace["method"] == method]
        rolling = method_trace["rolling_168h_simple_coverage"].dropna()
        worst_rolling = (
            float(np.max(np.abs(rolling.to_numpy() - target_coverage)))
            if not rolling.empty
            else np.nan
        )
        auprc = brier = ece = recall = np.nan
        if method in score_map:
            score_column, is_probability = score_map[method]
            score = frame[score_column].to_numpy(dtype=np.float64)
            auprc = float(average_precision_score(hard_target, score))
            if is_probability:
                brier = float(brier_score_loss(hard_target, score))
                ece = expected_calibration_error(score, hard_target)
            if use_column in frame.columns and np.any(hard_target == 1):
                recall = float(
                    np.mean(~frame.loc[hard_target == 1, use_column].to_numpy(bool))
                )
        aggregate_records.append(
            {
                "dataset_id": DATASET_ID,
                "method": method,
                "mean_RMSSE": float(method_metrics["RMSSE"].mean()),
                "median_RMSSE": float(method_metrics["RMSSE"].median()),
                "std_RMSSE": float(method_metrics["RMSSE"].std(ddof=1)),
                "mean_MASE": float(method_metrics["MASE"].mean()),
                "mean_sMAPE": float(method_metrics["sMAPE"].mean()),
                "overall_RMSE": float(
                    np.sqrt(np.mean((y_true_all - prediction) ** 2))
                ),
                "simple_coverage": coverage,
                "absolute_coverage_violation": (
                    abs(coverage - target_coverage)
                    if np.isfinite(coverage)
                    else np.nan
                ),
                "worst_168h_weighted_coverage_violation": worst_rolling,
                "black_box_AUPRC": auprc,
                "black_box_Brier": brier,
                "black_box_ECE": ece,
                "black_box_recall": recall,
            }
        )
    aggregate = pd.DataFrame(aggregate_records).sort_values(
        "mean_RMSSE", kind="stable"
    ).reset_index(drop=True)
    aggregate["RMSSE_rank"] = np.arange(1, len(aggregate) + 1)
    return per_series, aggregate, coverage_trace


def save_figure(
    aggregate: pd.DataFrame,
    coverage_trace: pd.DataFrame,
    target_coverage: float,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(20, 7))
    plot_data = aggregate.sort_values("mean_RMSSE", ascending=True)
    axes[0].barh(plot_data["method"], plot_data["mean_RMSSE"], color="#4c78a8")
    axes[0].set_xlabel("Mean per-series RMSSE")
    axes[0].set_title("Pedestrian formal-test accuracy")
    coverage_data = aggregate.dropna(subset=["simple_coverage"])
    axes[1].barh(
        coverage_data["method"], coverage_data["simple_coverage"], color="#59a14f"
    )
    axes[1].axvline(target_coverage, color="black", linestyle="--")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel("Ridge coverage")
    axes[1].set_title("Global coverage")
    for method, color in (
        ("static_full_router", "#f28e2b"),
        ("adaptive_full_router", "#e15759"),
    ):
        part = coverage_trace.loc[coverage_trace["method"] == method]
        axes[2].plot(
            part["relative_test_step"],
            part["rolling_168h_simple_coverage"],
            label=method,
            color=color,
        )
    axes[2].axhline(target_coverage, color="black", linestyle="--")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_xlabel("Hours since series entered test")
    axes[2].set_ylabel("Weighted rolling 168-hour Ridge coverage")
    axes[2].set_title("Coverage through relative time")
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def freeze_evaluator() -> None:
    if RECEIPT_PATH.exists():
        raise FileExistsError("Formal-test access receipt exists; cannot authorize again")
    manifest = verify_final_lock()
    artifacts = load_artifacts()
    dry_run = dry_run_calibration(artifacts)
    evaluator_hash = sha256_file(EVALUATOR_PATH)
    authorization_material = (
        f"{manifest['final_freeze_id']}:{evaluator_hash}"
    ).encode("utf-8")
    authorization_id = hashlib.sha256(authorization_material).hexdigest()
    try:
        evaluator_display_path = str(EVALUATOR_PATH.relative_to(PROJECT_ROOT))
    except ValueError:
        evaluator_display_path = str(EVALUATOR_PATH)
    payload = {
        "dataset_id": DATASET_ID,
        "status": "AUTHORIZED",
        "authorized_at_utc": utc_now(),
        "authorization_id": authorization_id,
        "final_freeze_id": manifest["final_freeze_id"],
        "evaluator_path": evaluator_display_path,
        "evaluator_sha256": evaluator_hash,
        "formal_test_runs_allowed": 1,
        "formal_test_runs_completed": 0,
        "dry_run": dry_run,
    }
    if AUTHORIZATION_PATH.exists():
        with AUTHORIZATION_PATH.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if (
            existing.get("authorization_id") != authorization_id
            or existing.get("evaluator_sha256") != evaluator_hash
            or int(existing.get("formal_test_runs_completed", -1)) != 0
        ):
            raise ValueError("Existing authorization differs from this evaluator")
    else:
        write_json_atomic(AUTHORIZATION_PATH, payload)

    print()
    print("Pedestrian 正式测试执行器冻结与校准干运行全部通过")
    print("最终预测试冻结编号：", manifest["final_freeze_id"])
    print("执行器SHA-256：", evaluator_hash)
    print("授权编号：", authorization_id)
    print("校准干运行样本：", dry_run["rows"])
    print("校准序列数量：", dry_run["series_count"])
    print("校准相对时间点：", dry_run["relative_steps"])
    print(
        "校准重算最大差异：",
        max(dry_run["maximum_differences"].values()),
    )
    print("测试集是否访问：否")
    print("正式测试运行次数：0")
    print("状态：已授权，但尚未执行正式测试")
    print("授权文件：", AUTHORIZATION_PATH)


def execute_formal_test() -> None:
    if not AUTHORIZATION_PATH.is_file():
        raise FileNotFoundError(
            "Evaluator authorization is missing; run --freeze-evaluator first"
        )
    if RECEIPT_PATH.exists():
        raise FileExistsError("Formal-test receipt exists; a second run is prohibited")
    forbidden = [
        path
        for path in (
            PREDICTION_PATH,
            PER_SERIES_PATH,
            AGGREGATE_PATH,
            COVERAGE_TRACE_PATH,
            FIGURE_PATH,
        )
        if path.exists()
    ]
    if forbidden:
        raise FileExistsError(f"Formal-test outputs already exist: {forbidden}")

    manifest = verify_final_lock()
    with AUTHORIZATION_PATH.open("r", encoding="utf-8") as handle:
        authorization = json.load(handle)
    evaluator_hash = sha256_file(EVALUATOR_PATH)
    if (
        authorization.get("status") != "AUTHORIZED"
        or authorization.get("evaluator_sha256") != evaluator_hash
        or authorization.get("final_freeze_id") != manifest["final_freeze_id"]
        or int(authorization.get("formal_test_runs_allowed", -1)) != 1
        or int(authorization.get("formal_test_runs_completed", -1)) != 0
        or authorization.get("dry_run", {}).get("test_accessed") is not False
    ):
        raise ValueError("Evaluator authorization is invalid or no longer matches")

    started = perf_counter()
    receipt = {
        "dataset_id": DATASET_ID,
        "status": "STARTED",
        "formal_test_run_number": 1,
        "started_at_utc": utc_now(),
        "authorization_id": authorization["authorization_id"],
        "final_freeze_id": manifest["final_freeze_id"],
        "evaluator_sha256": evaluator_hash,
    }
    # This permanent one-run lock is written before the first test-split read.
    write_json_atomic(RECEIPT_PATH, receipt)
    try:
        artifacts = load_artifacts()
        raw = read_raw_splits(
            ["base_train", "router_train", "calibration", "test"]
        )
        test_frame = build_segment(raw, "test", artifacts)
        if (
            len(test_frame) != EXPECTED_TEST_ROWS
            or test_frame["series_id"].nunique() != EXPECTED_SERIES
            or test_frame["relative_segment_step"].nunique()
            != EXPECTED_TEST_STEPS
            or set(test_frame["split"].astype(str)) != {"test"}
        ):
            raise ValueError(
                "Formal test segment range is invalid: "
                f"rows={len(test_frame)}; series="
                f"{test_frame['series_id'].nunique()}; steps="
                f"{test_frame['relative_segment_step'].nunique()}"
            )
        test_frame = apply_methods(test_frame, artifacts)
        target_coverage = float(
            artifacts["controller"]["primary_target_simple_coverage"]
        )
        per_series, aggregate, coverage_trace = compute_metrics(
            test_frame, target_coverage
        )
        if (
            len(per_series) != EXPECTED_SERIES * len(METHODS)
            or set(aggregate["method"]) != set(METHODS)
            or len(aggregate) != len(METHODS)
        ):
            raise ValueError("Formal metric tables are incomplete")

        test_frame.to_parquet(PREDICTION_PATH, index=False, compression="snappy")
        per_series.to_csv(PER_SERIES_PATH, index=False)
        aggregate.to_csv(AGGREGATE_PATH, index=False)
        coverage_trace.to_csv(COVERAGE_TRACE_PATH, index=False)
        save_figure(aggregate, coverage_trace, target_coverage)
        result_paths = [
            PREDICTION_PATH,
            PER_SERIES_PATH,
            AGGREGATE_PATH,
            COVERAGE_TRACE_PATH,
            FIGURE_PATH,
        ]
        result_hashes = {
            str(path.relative_to(OUTPUT_ROOT)): sha256_file(path)
            for path in result_paths
        }
        receipt.update(
            {
                "status": "COMPLETED",
                "completed_at_utc": utc_now(),
                "formal_test_rows": len(test_frame),
                "series_count": EXPECTED_SERIES,
                "relative_time_points": EXPECTED_TEST_STEPS,
                "elapsed_seconds": perf_counter() - started,
                "result_sha256": result_hashes,
            }
        )
        write_json_atomic(RECEIPT_PATH, receipt)
    except Exception as error:
        receipt.update(
            {
                "status": "FAILED_LOCKED",
                "failed_at_utc": utc_now(),
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        write_json_atomic(RECEIPT_PATH, receipt)
        raise

    primary = aggregate.loc[
        aggregate["method"] == "adaptive_full_router"
    ].iloc[0]
    print()
    print("Pedestrian 唯一一次正式测试全部完成")
    print("正式测试行数：", len(test_frame))
    print("序列数量：", EXPECTED_SERIES)
    print("相对时间点数量：", EXPECTED_TEST_STEPS)
    print("自适应完整方法 mean RMSSE：", f"{primary['mean_RMSSE']:.6f}")
    print(
        "自适应完整方法 Ridge 覆盖率：",
        f"{primary['simple_coverage']:.6f}",
    )
    print(
        "自适应完整方法覆盖偏差：",
        f"{primary['absolute_coverage_violation']:.6f}",
    )
    print("方法排名表：")
    print(
        aggregate[
            ["RMSSE_rank", "method", "mean_RMSSE", "simple_coverage"]
        ].to_string(index=False)
    )
    print("正式测试运行次数：1（已锁定，禁止再次运行）")
    print("逐时点预测：", PREDICTION_PATH)
    print("逐序列指标：", PER_SERIES_PATH)
    print("汇总指标：", AGGREGATE_PATH)
    print("覆盖率轨迹：", COVERAGE_TRACE_PATH)
    print("结果图片：", FIGURE_PATH)
    print("访问回执：", RECEIPT_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze or execute the one-time Pedestrian formal evaluator"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--freeze-evaluator",
        action="store_true",
        help=(
            "verify frozen artifacts and authorize this exact evaluator using "
            "a calibration-only dry run"
        ),
    )
    group.add_argument(
        "--execute-final-test",
        action="store_true",
        help="perform the single authorized formal-test run",
    )
    arguments = parser.parse_args()
    if arguments.freeze_evaluator:
        freeze_evaluator()
    else:
        execute_formal_test()


if __name__ == "__main__":
    main()
