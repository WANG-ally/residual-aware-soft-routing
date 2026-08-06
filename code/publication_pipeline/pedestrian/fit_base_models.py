#!/usr/bin/env python3
"""Fit frozen Pedestrian base models and make pre-test predictions."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import Ridge
import yaml


PROJECT_ROOT = Path(
    os.environ.get("SCI_ROUTING_ROOT", Path(__file__).resolve().parents[3])
).resolve()
OUTPUT_ROOT = Path(os.environ.get("SCI_ROUTING_OUTPUT_ROOT", PROJECT_ROOT)).resolve()

DATA_PATH = PROJECT_ROOT / "data/processed/pedestrian_hourly_long.parquet"
SCALER_PATH = PROJECT_ROOT / "results/pedestrian_scaler_parameters.csv"
WINDOW_COUNT_PATH = PROJECT_ROOT / "results/pedestrian_window_candidate_counts.csv"
SPLIT_SUMMARY_PATH = PROJECT_ROOT / "results/pedestrian_split_summary.yaml"
RIDGE_PARAMS_PATH = PROJECT_ROOT / "results/pedestrian_selected_ridge_params.yaml"
LGBM_PARAMS_PATH = PROJECT_ROOT / "results/pedestrian_selected_lightgbm_params.yaml"
RIDGE_CHECKS_PATH = PROJECT_ROOT / "results/pedestrian_ridge_tuning_checks.csv"
LGBM_CHECKS_PATH = PROJECT_ROOT / "results/pedestrian_lightgbm_tuning_checks.csv"

RIDGE_MODEL_PATH = OUTPUT_ROOT / "models/pedestrian_ridge.joblib"
LGBM_MODEL_PATH = OUTPUT_ROOT / "models/pedestrian_lightgbm.joblib"
PREDICTION_PATH = OUTPUT_ROOT / "results/pedestrian_pretest_predictions.parquet"
PER_SERIES_PATH = OUTPUT_ROOT / "results/pedestrian_pretest_per_series_metrics.csv"
AGGREGATE_PATH = OUTPUT_ROOT / "results/pedestrian_pretest_aggregate_metrics.csv"
METADATA_PATH = OUTPUT_ROOT / "results/pedestrian_base_model_fit_metadata.yaml"
CHECKS_PATH = OUTPUT_ROOT / "results/pedestrian_base_model_fit_checks.csv"
FIGURE_PATH = OUTPUT_ROOT / "figures/pedestrian_pretest_predictions_T66.png"
REPORT_PATH = OUTPUT_ROOT / "logs/pedestrian_base_model_fit_report.json"

DATASET_ID = "pedestrian_hourly"
ALLOWED_SPLITS = ["base_train", "router_train", "calibration"]
PREDICTION_SPLITS = ["router_train", "calibration"]
SEASONAL_PERIOD = 24


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def passed_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.lower().eq("true")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sufficient_statistics(
    scaled_values: np.ndarray, train_end: int, window: int
) -> dict[str, np.ndarray | float | int]:
    z = np.asarray(scaled_values[:train_end], dtype=np.float64)
    length = len(z)
    prefix = np.empty(length + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(z, out=prefix[1:])
    sum_x = np.asarray(
        [
            prefix[length - lag_index - 1]
            - prefix[window - lag_index - 1]
            for lag_index in range(window)
        ],
        dtype=np.float64,
    )
    statistics: dict[str, np.ndarray | float | int] = {
        "n": length - window,
        "sum_x": sum_x,
        "sum_y": float(prefix[length] - prefix[window]),
        "xtx": np.zeros((window, window), dtype=np.float64),
        "xty": np.zeros(window, dtype=np.float64),
    }
    xtx = statistics["xtx"]
    xty = statistics["xty"]
    assert isinstance(xtx, np.ndarray)
    assert isinstance(xty, np.ndarray)

    for difference in range(window + 1):
        if difference == 0:
            product = z * z
        else:
            product = z[difference:] * z[:-difference]
        product_prefix = np.empty(len(product) + 1, dtype=np.float64)
        product_prefix[0] = 0.0
        np.cumsum(product, out=product_prefix[1:])
        if difference >= 1:
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
    return statistics


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


def solve_ridge(
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
    centered_xtx = xtx - np.outer(sum_x, sum_x) / n
    centered_xtx = 0.5 * (centered_xtx + centered_xtx.T)
    centered_xty = xty - sum_x * sum_y / n
    coefficient = np.linalg.solve(
        centered_xtx + float(alpha) * np.eye(len(sum_x)), centered_xty
    )
    intercept = float(sum_y / n - (sum_x / n) @ coefficient)
    return coefficient, intercept


def main() -> None:
    for path in (
        DATA_PATH,
        SCALER_PATH,
        WINDOW_COUNT_PATH,
        SPLIT_SUMMARY_PATH,
        RIDGE_PARAMS_PATH,
        LGBM_PARAMS_PATH,
        RIDGE_CHECKS_PATH,
        LGBM_CHECKS_PATH,
    ):
        require_file(path)
    for path in (
        RIDGE_MODEL_PATH,
        LGBM_MODEL_PATH,
        PREDICTION_PATH,
        PER_SERIES_PATH,
        AGGREGATE_PATH,
        METADATA_PATH,
        CHECKS_PATH,
        FIGURE_PATH,
        REPORT_PATH,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    ridge_params = yaml.safe_load(RIDGE_PARAMS_PATH.read_text(encoding="utf-8"))
    lgbm_params = yaml.safe_load(LGBM_PARAMS_PATH.read_text(encoding="utf-8"))
    split_summary = yaml.safe_load(SPLIT_SUMMARY_PATH.read_text(encoding="utf-8"))
    ridge_checks = pd.read_csv(RIDGE_CHECKS_PATH)
    lgbm_checks = pd.read_csv(LGBM_CHECKS_PATH)
    window_counts = pd.read_csv(WINDOW_COUNT_PATH)
    scalers = pd.read_csv(SCALER_PATH)

    for selected in (ridge_params, lgbm_params):
        if selected["selection_scope"] != "base_train_only":
            raise AssertionError("Selected parameters were not chosen from base_train only")
        if bool(selected.get("test_values_accessed", True)):
            raise AssertionError("A tuning artifact reports test access")
    if not passed_column(ridge_checks["passed"]).all():
        raise AssertionError("Ridge tuning checks did not all pass")
    if not passed_column(lgbm_checks["passed"]).all():
        raise AssertionError("LightGBM tuning checks did not all pass")
    if scalers["source_split"].ne("base_train").any():
        raise AssertionError("Scaler parameters are not base_train-only")

    data = pd.read_parquet(
        DATA_PATH,
        columns=["dataset_id", "series_id", "time_index", "timestamp", "value", "split"],
        filters=[[("split", "==", split_name)] for split_name in ALLOWED_SPLITS],
    )
    observed_splits = set(data["split"].astype(str).unique().tolist())
    if observed_splits != set(ALLOWED_SPLITS) or (data["split"] == "test").any():
        raise AssertionError(f"Unexpected pre-test data scope: {observed_splits}")
    expected_allowed_rows = int(
        sum(split_summary["aggregate_counts"][name] for name in ALLOWED_SPLITS)
    )
    if len(data) != expected_allowed_rows:
        raise AssertionError(
            f"Allowed row mismatch: data={len(data)}, expected={expected_allowed_rows}"
        )

    series_frames = {
        str(series_id): group.sort_values("time_index", kind="stable").reset_index(drop=True)
        for series_id, group in data.groupby("series_id", sort=False, observed=True)
    }
    del data
    gc.collect()
    scaler_lookup = {
        str(row.series_id): (float(row.median), float(row.scale_used))
        for row in scalers.itertuples(index=False)
    }
    if set(series_frames) != set(scaler_lookup):
        raise AssertionError("Series IDs differ between data and scaler table")

    contiguous_indices_ok = True
    split_order_ok = True
    split_codes = {name: number for number, name in enumerate(ALLOWED_SPLITS)}
    for series_id, group in series_frames.items():
        indices = group["time_index"].to_numpy(dtype=np.int64)
        codes = group["split"].astype(str).map(split_codes).to_numpy(dtype=np.int8)
        contiguous_indices_ok = contiguous_indices_ok and bool(
            np.array_equal(indices, np.arange(len(group), dtype=np.int64))
        )
        split_order_ok = split_order_ok and bool(np.all(np.diff(codes) >= 0))

    ridge_window = int(ridge_params["selected_window"])
    lgbm_window = int(lgbm_params["selected_window"])
    if ridge_window != 168 or lgbm_window != 168:
        raise AssertionError(
            f"Expected selected 168-hour windows, got Ridge={ridge_window}, LGBM={lgbm_window}"
        )
    expected_ridge_samples = int(
        window_counts.loc[window_counts["window"] == ridge_window, "base_train"].iloc[0]
    )
    expected_lgbm_samples = int(
        window_counts.loc[window_counts["window"] == lgbm_window, "base_train"].iloc[0]
    )

    # Fit exact Ridge from all eligible base_train targets.
    ridge_statistics: dict[str, np.ndarray | float | int] = {
        "n": 0,
        "sum_x": np.zeros(ridge_window, dtype=np.float64),
        "sum_y": 0.0,
        "xtx": np.zeros((ridge_window, ridge_window), dtype=np.float64),
        "xty": np.zeros(ridge_window, dtype=np.float64),
    }
    ridge_fit_start = perf_counter()
    for series_id, group in series_frames.items():
        values = group["value"].to_numpy(dtype=np.float64)
        base_count = int((group["split"] == "base_train").sum())
        median, scale = scaler_lookup[series_id]
        scaled = (values - median) / scale
        add_statistics(
            ridge_statistics,
            sufficient_statistics(scaled, base_count, ridge_window),
        )
    ridge_coefficient, ridge_intercept = solve_ridge(
        ridge_statistics, float(ridge_params["selected_alpha"])
    )
    ridge_fit_seconds = perf_counter() - ridge_fit_start
    ridge = Ridge(alpha=float(ridge_params["selected_alpha"]), fit_intercept=True)
    ridge.coef_ = ridge_coefficient
    ridge.intercept_ = ridge_intercept
    ridge.n_features_in_ = ridge_window
    joblib.dump(ridge, RIDGE_MODEL_PATH, compress=3)

    if int(ridge_statistics["n"]) != expected_ridge_samples:
        raise AssertionError("Ridge did not use every eligible base_train target")

    # Preallocate one float32 matrix, then fit LightGBM on all eligible targets.
    estimated_gib = expected_lgbm_samples * lgbm_window * 4 / 1024**3
    print(
        f"[准备] LightGBM 全量训练矩阵：{expected_lgbm_samples} × "
        f"{lgbm_window}，约 {estimated_gib:.2f} GiB",
        flush=True,
    )
    lgbm_x = np.empty((expected_lgbm_samples, lgbm_window), dtype=np.float32)
    lgbm_y = np.empty(expected_lgbm_samples, dtype=np.float32)
    cursor = 0
    for series_id, group in series_frames.items():
        values = group["value"].to_numpy(dtype=np.float64)
        base_count = int((group["split"] == "base_train").sum())
        median, scale = scaler_lookup[series_id]
        scaled = (values - median) / scale
        count = base_count - lgbm_window
        view = sliding_window_view(scaled[:base_count], lgbm_window)
        lgbm_x[cursor : cursor + count] = view[:-1, ::-1]
        lgbm_y[cursor : cursor + count] = scaled[lgbm_window:base_count]
        cursor += count
    if cursor != expected_lgbm_samples:
        raise AssertionError(
            f"LightGBM full training rows mismatch: {cursor} vs {expected_lgbm_samples}"
        )
    print("[完成] LightGBM 全量训练矩阵已构造", flush=True)

    seed = int(lgbm_params["seed"])
    lgbm = LGBMRegressor(
        objective="regression_l2",
        num_leaves=int(lgbm_params["num_leaves"]),
        learning_rate=float(lgbm_params["learning_rate"]),
        n_estimators=int(lgbm_params["n_estimators"]),
        feature_fraction=float(lgbm_params["feature_fraction"]),
        random_state=seed,
        feature_fraction_seed=seed,
        data_random_seed=seed,
        deterministic=True,
        force_col_wise=True,
        n_jobs=-1,
        verbosity=-1,
    )
    print("[开始] 使用全部 base_train 窗口拟合 LightGBM", flush=True)
    lgbm_fit_start = perf_counter()
    lgbm.fit(lgbm_x, lgbm_y)
    lgbm_fit_seconds = perf_counter() - lgbm_fit_start
    joblib.dump(lgbm, LGBM_MODEL_PATH, compress=3)
    del lgbm_x, lgbm_y
    gc.collect()
    print("[完成] LightGBM 全量拟合与模型保存", flush=True)

    # Predict router_train and calibration one series at a time.
    prediction_parts: list[pd.DataFrame] = []
    causal_windows_ok = True
    prediction_start = perf_counter()
    denominators: dict[str, dict[str, float]] = {}
    for series_id, group in series_frames.items():
        values = group["value"].to_numpy(dtype=np.float64)
        split_text = group["split"].astype(str)
        base_values = group.loc[split_text == "base_train", "value"].to_numpy(dtype=float)
        seasonal_differences = (
            base_values[SEASONAL_PERIOD:] - base_values[:-SEASONAL_PERIOD]
        )
        squared_denominator = float(np.mean(seasonal_differences**2))
        absolute_denominator = float(np.mean(np.abs(seasonal_differences)))
        if squared_denominator <= 1e-12 or absolute_denominator <= 1e-12:
            raise AssertionError(f"{series_id} has a zero error-scaling denominator")
        denominators[series_id] = {
            "squared": squared_denominator,
            "absolute": absolute_denominator,
        }

        prediction_mask = split_text.isin(PREDICTION_SPLITS).to_numpy()
        targets = group.loc[prediction_mask, "time_index"].to_numpy(dtype=np.int64)
        causal_windows_ok = causal_windows_ok and bool(
            targets.min() >= max(ridge_window, lgbm_window)
        )
        median, scale = scaler_lookup[series_id]
        scaled = (values - median) / scale
        ridge_view = sliding_window_view(scaled, ridge_window)
        ridge_x = ridge_view[targets - ridge_window, ::-1]
        ridge_prediction = ridge_x @ ridge_coefficient + ridge_intercept
        ridge_prediction = ridge_prediction * scale + median

        if lgbm_window == ridge_window:
            lgbm_x_series = ridge_x.astype(np.float32, copy=True)
        else:
            lgbm_view = sliding_window_view(scaled, lgbm_window)
            lgbm_x_series = lgbm_view[
                targets - lgbm_window, ::-1
            ].astype(np.float32, copy=True)
        lgbm_prediction = lgbm.booster_.predict(lgbm_x_series) * scale + median

        metadata = (
            group.loc[
                prediction_mask,
                ["dataset_id", "series_id", "time_index", "timestamp", "split", "value"],
            ]
            .rename(columns={"value": "y_true"})
            .reset_index(drop=True)
        )
        metadata["ridge_prediction"] = ridge_prediction
        metadata["lightgbm_prediction"] = lgbm_prediction
        metadata["seasonal_naive_prediction"] = values[targets - SEASONAL_PERIOD]
        metadata["RMSSE_denominator"] = squared_denominator
        metadata["MASE_denominator"] = absolute_denominator
        prediction_parts.append(metadata)
    prediction_seconds = perf_counter() - prediction_start

    prediction = pd.concat(prediction_parts, ignore_index=True)
    prediction["ridge_residual"] = prediction["y_true"] - prediction["ridge_prediction"]
    prediction["lightgbm_residual"] = (
        prediction["y_true"] - prediction["lightgbm_prediction"]
    )
    prediction["ridge_squared_error"] = prediction["ridge_residual"] ** 2
    prediction["lightgbm_squared_error"] = prediction["lightgbm_residual"] ** 2
    prediction["black_box_better"] = (
        prediction["lightgbm_squared_error"] < prediction["ridge_squared_error"]
    )
    observed_prediction_splits = set(prediction["split"].astype(str).unique().tolist())
    expected_prediction_rows = int(
        sum(split_summary["aggregate_counts"][name] for name in PREDICTION_SPLITS)
    )
    if observed_prediction_splits != set(PREDICTION_SPLITS):
        raise AssertionError(f"Unexpected prediction splits: {observed_prediction_splits}")
    if len(prediction) != expected_prediction_rows:
        raise AssertionError(
            f"Prediction rows mismatch: {len(prediction)} vs {expected_prediction_rows}"
        )
    prediction.to_parquet(PREDICTION_PATH, index=False, compression="snappy")

    metric_records: list[dict[str, object]] = []
    for split_name in PREDICTION_SPLITS:
        split_data = prediction.loc[prediction["split"] == split_name]
        for series_id_raw, group in split_data.groupby(
            "series_id", sort=False, observed=True
        ):
            series_id = str(series_id_raw)
            y_true = group["y_true"].to_numpy(dtype=float)
            for model_name, prediction_column in (
                ("ridge", "ridge_prediction"),
                ("lightgbm", "lightgbm_prediction"),
            ):
                predicted = group[prediction_column].to_numpy(dtype=float)
                error = y_true - predicted
                metric_records.append(
                    {
                        "dataset_id": DATASET_ID,
                        "split": split_name,
                        "series_id": series_id,
                        "model": model_name,
                        "RMSSE": float(
                            np.sqrt(
                                np.mean(error**2)
                                / denominators[series_id]["squared"]
                            )
                        ),
                        "MASE": float(
                            np.mean(np.abs(error))
                            / denominators[series_id]["absolute"]
                        ),
                        "sMAPE": float(
                            100
                            * np.mean(
                                2
                                * np.abs(error)
                                / (np.abs(y_true) + np.abs(predicted) + 1e-8)
                            )
                        ),
                        "RMSE": float(np.sqrt(np.mean(error**2))),
                        "MAE": float(np.mean(np.abs(error))),
                    }
                )
    per_series = pd.DataFrame(metric_records)
    aggregate = (
        per_series.groupby(["split", "model"], as_index=False)
        .agg(
            mean_RMSSE=("RMSSE", "mean"),
            median_RMSSE=("RMSSE", "median"),
            mean_MASE=("MASE", "mean"),
            mean_sMAPE=("sMAPE", "mean"),
            mean_RMSE=("RMSE", "mean"),
            mean_MAE=("MAE", "mean"),
            series_count=("series_id", "nunique"),
        )
        .sort_values(["split", "mean_RMSSE"], kind="stable")
        .reset_index(drop=True)
    )

    all_predictions_finite = bool(
        np.isfinite(
            prediction[["ridge_prediction", "lightgbm_prediction"]].to_numpy(
                dtype=float
            )
        ).all()
    )
    all_metrics_finite = bool(
        np.isfinite(
            per_series[["RMSSE", "MASE", "sMAPE", "RMSE", "MAE"]].to_numpy(
                dtype=float
            )
        ).all()
    )
    check_items: list[tuple[str, bool, str]] = [
        ("ridge_tuning_audit_passed", bool(passed_column(ridge_checks["passed"]).all()), f"checks={len(ridge_checks)}"),
        ("lightgbm_tuning_audit_passed", bool(passed_column(lgbm_checks["passed"]).all()), f"checks={len(lgbm_checks)}"),
        ("input_excludes_test", "test" not in observed_splits, str(observed_splits)),
        ("input_row_count_matches_registered_splits", len(series_frames) == 66 and expected_allowed_rows > 0, f"series={len(series_frames)}; rows={expected_allowed_rows}"),
        ("time_indices_are_contiguous", contiguous_indices_ok, str(contiguous_indices_ok)),
        ("split_order_is_chronological", split_order_ok, str(split_order_ok)),
        ("scalers_use_base_train_only", bool(scalers["source_split"].eq("base_train").all()), f"scalers={len(scalers)}"),
        ("ridge_uses_all_eligible_base_targets", int(ridge_statistics["n"]) == expected_ridge_samples, f"actual={ridge_statistics['n']}; expected={expected_ridge_samples}"),
        ("lightgbm_uses_all_eligible_base_targets", cursor == expected_lgbm_samples, f"actual={cursor}; expected={expected_lgbm_samples}"),
        ("prediction_targets_are_causal", causal_windows_ok, str(causal_windows_ok)),
        ("prediction_scope_is_router_and_calibration", observed_prediction_splits == set(PREDICTION_SPLITS), str(observed_prediction_splits)),
        ("prediction_row_count_matches_manifest", len(prediction) == expected_prediction_rows, f"actual={len(prediction)}; expected={expected_prediction_rows}"),
        ("all_predictions_are_finite", all_predictions_finite, str(all_predictions_finite)),
        ("all_pretest_metrics_are_finite", all_metrics_finite, str(all_metrics_finite)),
        ("model_artifacts_exist", RIDGE_MODEL_PATH.is_file() and LGBM_MODEL_PATH.is_file(), f"ridge={RIDGE_MODEL_PATH.is_file()}; lgbm={LGBM_MODEL_PATH.is_file()}"),
    ]
    checks = pd.DataFrame(check_items, columns=["check", "passed", "detail"])
    failed = checks.loc[~checks["passed"]]
    if not failed.empty:
        message = "; ".join(
            f"{row.check}: {row.detail}" for row in failed.itertuples(index=False)
        )
        raise AssertionError(f"Pedestrian base-model fit audit failed: {message}")

    per_series.to_csv(PER_SERIES_PATH, index=False)
    aggregate.to_csv(AGGREGATE_PATH, index=False)
    checks.to_csv(CHECKS_PATH, index=False)
    model_hashes = {
        "ridge_joblib_sha256": sha256_file(RIDGE_MODEL_PATH),
        "lightgbm_joblib_sha256": sha256_file(LGBM_MODEL_PATH),
    }
    metadata = {
        "dataset_id": DATASET_ID,
        "parameter_files": {
            "ridge": str(RIDGE_PARAMS_PATH),
            "ridge_sha256": sha256_file(RIDGE_PARAMS_PATH),
            "lightgbm": str(LGBM_PARAMS_PATH),
            "lightgbm_sha256": sha256_file(LGBM_PARAMS_PATH),
        },
        "parameters_frozen_before_fit": True,
        "training_scope": "base_train_only",
        "prediction_splits": PREDICTION_SPLITS,
        "test_accessed": False,
        "ridge_window": ridge_window,
        "ridge_alpha": float(ridge_params["selected_alpha"]),
        "ridge_train_samples": int(ridge_statistics["n"]),
        "ridge_fit_method": "exact sufficient statistics",
        "ridge_fit_seconds": float(ridge_fit_seconds),
        "lightgbm_window": lgbm_window,
        "lightgbm_train_samples": cursor,
        "lightgbm_training_sampled": False,
        "lightgbm_fit_seconds": float(lgbm_fit_seconds),
        "lightgbm_training_matrix_gib": float(estimated_gib),
        "prediction_rows": int(len(prediction)),
        "prediction_seconds": float(prediction_seconds),
        "model_hashes": model_hashes,
    }
    METADATA_PATH.write_text(
        yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    example = prediction.loc[prediction["series_id"].astype(str) == "T66"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)
    for ax, split_name in zip(axes, PREDICTION_SPLITS):
        part = example.loc[example["split"] == split_name]
        ax.plot(part["time_index"], part["y_true"], color="#222222", linewidth=1.5, label="actual")
        ax.plot(part["time_index"], part["ridge_prediction"], color="#4C78A8", linewidth=1.1, label="ridge")
        ax.plot(part["time_index"], part["lightgbm_prediction"], color="#F28E2B", linewidth=1.1, label="lightgbm")
        ax.set_title(f"Pedestrian T66: {split_name}")
        ax.set_ylabel("Pedestrian count")
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=3, frameon=False)
    axes[-1].set_xlabel("Time index (hour)")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = {
        "status": "passed",
        "dataset_id": DATASET_ID,
        "check_count": int(len(checks)),
        "failed_check_count": 0,
        "training_scope": "base_train_only",
        "prediction_scope": PREDICTION_SPLITS,
        "test_accessed": False,
        "ridge_train_samples": int(ridge_statistics["n"]),
        "lightgbm_train_samples": cursor,
        "prediction_rows": int(len(prediction)),
        "model_hashes": model_hashes,
        "outputs": {
            "ridge_model": str(RIDGE_MODEL_PATH),
            "lightgbm_model": str(LGBM_MODEL_PATH),
            "predictions": str(PREDICTION_PATH),
            "per_series_metrics": str(PER_SERIES_PATH),
            "aggregate_metrics": str(AGGREGATE_PATH),
            "metadata": str(METADATA_PATH),
            "checks": str(CHECKS_PATH),
            "figure": str(FIGURE_PATH),
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print()
    print("Pedestrian 基础模型训练和预测全部通过")
    print("训练数据范围：仅 base_train")
    print("预测数据范围：router_train + calibration")
    print("test 是否访问：否")
    print("Ridge 训练样本：", int(ridge_statistics["n"]))
    print("LightGBM 训练样本：", cursor)
    print("LightGBM 是否使用调参抽样：否（这里使用全部合格窗口）")
    print("预测总行数：", len(prediction))
    print("因果窗口检查：通过")
    print("预测试指标：")
    print(
        aggregate[
            ["split", "model", "mean_RMSSE", "mean_MASE", "mean_sMAPE"]
        ].to_string(index=False)
    )
    print("Ridge 拟合秒数：", f"{ridge_fit_seconds:.2f}")
    print("LightGBM 拟合秒数：", f"{lgbm_fit_seconds:.2f}")
    print("预测文件：", PREDICTION_PATH)
    print("逐序列指标：", PER_SERIES_PATH)
    print("汇总指标：", AGGREGATE_PATH)
    print("拟合记录：", METADATA_PATH)
    print("预测图片：", FIGURE_PATH)


if __name__ == "__main__":
    main()
