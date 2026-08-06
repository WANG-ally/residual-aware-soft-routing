#!/usr/bin/env python3
"""Fit frozen Weather Daily base models and predict pre-test splits.

The formal test partition is deliberately excluded at parquet-read time.  Ridge
and LightGBM are fitted on every eligible base_train target using their distinct
frozen windows, then used only on router_train and calibration."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sci-routing")

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import Ridge


PROJECT_ROOT = Path(
    os.environ.get("SCI_ROUTING_ROOT", Path(__file__).resolve().parents[3])
).resolve()
OUTPUT_ROOT = Path(os.environ.get("SCI_ROUTING_OUTPUT_ROOT", PROJECT_ROOT)).resolve()

DATA_PATH = PROJECT_ROOT / "data/processed/weather_daily_long.parquet"
SCALER_PATH = PROJECT_ROOT / "results/weather_daily_scaler_parameters.csv"
WINDOW_COUNT_PATH = PROJECT_ROOT / "results/weather_daily_window_candidate_counts.csv"
SPLIT_MANIFEST_PATH = PROJECT_ROOT / "results/weather_daily_split_manifest.csv"
SPLIT_SUMMARY_PATH = PROJECT_ROOT / "results/weather_daily_split_summary.yaml"
SAMPLE_REGISTRATION_PATH = PROJECT_ROOT / "results/weather_daily_sample_registration.yaml"
RIDGE_PARAMS_PATH = PROJECT_ROOT / "results/weather_daily_selected_ridge_params.yaml"
LGBM_PARAMS_PATH = PROJECT_ROOT / "results/weather_daily_selected_lightgbm_params.yaml"
RIDGE_CHECKS_PATH = PROJECT_ROOT / "results/weather_daily_ridge_tuning_checks.csv"
LGBM_CHECKS_PATH = PROJECT_ROOT / "results/weather_daily_lightgbm_tuning_checks.csv"

RIDGE_MODEL_PATH = OUTPUT_ROOT / "models/weather_daily_ridge.joblib"
LGBM_MODEL_PATH = OUTPUT_ROOT / "models/weather_daily_lightgbm.joblib"
PREDICTION_PATH = OUTPUT_ROOT / "results/weather_daily_pretest_predictions.parquet"
PER_SERIES_PATH = OUTPUT_ROOT / "results/weather_daily_pretest_per_series_metrics.csv"
AGGREGATE_PATH = OUTPUT_ROOT / "results/weather_daily_pretest_aggregate_metrics.csv"
METADATA_PATH = OUTPUT_ROOT / "results/weather_daily_base_model_fit_metadata.yaml"
CHECKS_PATH = OUTPUT_ROOT / "results/weather_daily_base_model_fit_checks.csv"
FIGURE_PATH = OUTPUT_ROOT / "figures/weather_daily_pretest_predictions_T3.png"
REPORT_PATH = OUTPUT_ROOT / "logs/weather_daily_base_model_fit_report.json"

DATASET_ID = "weather_daily"
SAMPLE_ID = "5518ada55d6adb8a8df20f5a30d52ec904362ce57df9adeeceb0bc3d22854044"
ALLOWED_SPLITS = ["base_train", "router_train", "calibration"]
PREDICTION_SPLITS = ["router_train", "calibration"]
SEASONAL_PERIOD = 7
EXPECTED_SERIES = 500
EXPECTED_BASE_ROWS = 4_421_760
EXPECTED_ALLOWED_ROWS = 6_264_251
EXPECTED_PREDICTION_ROWS = 1_842_491
EXPECTED_RIDGE_TRAIN_SAMPLES = 4_393_760
EXPECTED_LGBM_TRAIN_SAMPLES = 4_407_760
SEQUENCE_BATCH_SIZE = 65_536


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def passed_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1"})


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


class CausalLagSequence(lgb.Sequence):
    """Expose all causal lag windows without materializing one dense matrix."""

    batch_size = SEQUENCE_BATCH_SIZE

    def __init__(self, scaled_base_series: list[np.ndarray], window: int) -> None:
        self.values = [np.asarray(item, dtype=np.float64) for item in scaled_base_series]
        self.window = int(window)
        self.counts = np.asarray(
            [len(item) - self.window for item in self.values], dtype=np.int64
        )
        if np.any(self.counts <= 0):
            raise ValueError("Every series must contain at least one causal target")
        self.offsets = np.concatenate(([0], np.cumsum(self.counts, dtype=np.int64)))
        self.views = [sliding_window_view(item, self.window) for item in self.values]

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def _row(self, index: int) -> np.ndarray:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        series_index = int(np.searchsorted(self.offsets, index, side="right") - 1)
        local_index = int(index - self.offsets[series_index])
        # LightGBM 4.6 uses float64 random-access rows during bin sampling.
        return self.views[series_index][local_index, ::-1].astype(
            np.float64, copy=True
        )

    def _slice(self, index: slice) -> np.ndarray:
        start, stop, step = index.indices(len(self))
        if step != 1:
            return np.stack([self._row(i) for i in range(start, stop, step)])
        result = np.empty((max(0, stop - start), self.window), dtype=np.float32)
        global_cursor = start
        output_cursor = 0
        while global_cursor < stop:
            series_index = int(
                np.searchsorted(self.offsets, global_cursor, side="right") - 1
            )
            local_start = int(global_cursor - self.offsets[series_index])
            take = min(
                stop - global_cursor,
                int(self.counts[series_index]) - local_start,
            )
            result[output_cursor : output_cursor + take] = self.views[series_index][
                local_start : local_start + take, ::-1
            ]
            global_cursor += take
            output_cursor += take
        return result

    def __getitem__(self, index: int | slice | list[int]) -> np.ndarray:
        if isinstance(index, (int, np.integer)):
            return self._row(int(index))
        if isinstance(index, slice):
            return self._slice(index)
        if isinstance(index, list):
            return np.stack([self._row(int(item)) for item in index])
        raise TypeError(f"Unsupported Sequence index type: {type(index).__name__}")


def sufficient_statistics(
    scaled_values: np.ndarray, train_end: int, window: int
) -> dict[str, np.ndarray | float | int]:
    """Exact pooled lag-regression sufficient statistics for one series."""

    z = np.asarray(scaled_values[:train_end], dtype=np.float64)
    length = len(z)
    prefix = np.empty(length + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(z, out=prefix[1:])
    statistics: dict[str, np.ndarray | float | int] = {
        "n": length - window,
        "sum_x": np.asarray(
            [
                prefix[length - lag_index - 1]
                - prefix[window - lag_index - 1]
                for lag_index in range(window)
            ],
            dtype=np.float64,
        ),
        "sum_y": float(prefix[length] - prefix[window]),
        "xtx": np.zeros((window, window), dtype=np.float64),
        "xty": np.zeros(window, dtype=np.float64),
    }
    xtx = statistics["xtx"]
    xty = statistics["xty"]
    assert isinstance(xtx, np.ndarray)
    assert isinstance(xty, np.ndarray)
    for difference in range(window + 1):
        product = z * z if difference == 0 else z[difference:] * z[:-difference]
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
    input_paths = (
        DATA_PATH,
        SCALER_PATH,
        WINDOW_COUNT_PATH,
        SPLIT_MANIFEST_PATH,
        SPLIT_SUMMARY_PATH,
        SAMPLE_REGISTRATION_PATH,
        RIDGE_PARAMS_PATH,
        LGBM_PARAMS_PATH,
        RIDGE_CHECKS_PATH,
        LGBM_CHECKS_PATH,
    )
    for path in input_paths:
        require_file(path)
    output_paths = (
        RIDGE_MODEL_PATH,
        LGBM_MODEL_PATH,
        PREDICTION_PATH,
        PER_SERIES_PATH,
        AGGREGATE_PATH,
        METADATA_PATH,
        CHECKS_PATH,
        FIGURE_PATH,
        REPORT_PATH,
    )
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    ridge_params = yaml.safe_load(RIDGE_PARAMS_PATH.read_text(encoding="utf-8"))
    lgbm_params = yaml.safe_load(LGBM_PARAMS_PATH.read_text(encoding="utf-8"))
    split_summary = yaml.safe_load(SPLIT_SUMMARY_PATH.read_text(encoding="utf-8"))
    registration = yaml.safe_load(
        SAMPLE_REGISTRATION_PATH.read_text(encoding="utf-8")
    )
    ridge_checks = pd.read_csv(RIDGE_CHECKS_PATH)
    lgbm_checks = pd.read_csv(LGBM_CHECKS_PATH)
    window_counts = pd.read_csv(WINDOW_COUNT_PATH)
    split_manifest = pd.read_csv(SPLIT_MANIFEST_PATH).sort_values("sample_order")
    scalers = pd.read_csv(SCALER_PATH).sort_values("sample_order")

    sample_ids = {
        str(ridge_params.get("sample_id")),
        str(lgbm_params.get("sample_id")),
        str(split_summary.get("sample_id")),
        str(registration.get("sample_id")),
        *split_manifest["sample_id"].astype(str).unique().tolist(),
        *scalers["sample_id"].astype(str).unique().tolist(),
    }
    if sample_ids != {SAMPLE_ID}:
        raise AssertionError(f"Frozen sample IDs disagree: {sample_ids}")
    for selected in (ridge_params, lgbm_params):
        if selected["selection_scope"] != "base_train_only":
            raise AssertionError("Selected parameters were not chosen from base_train only")
        if bool(selected.get("test_values_accessed", True)):
            raise AssertionError("A tuning artifact reports formal-test access")
        if bool(selected.get("calibration_accessed", True)):
            raise AssertionError("A tuning artifact reports calibration access")
        if bool(selected.get("router_train_accessed", True)):
            raise AssertionError("A tuning artifact reports router_train access")
    if lgbm_params.get("planned_final_training_scope") != "all eligible base_train targets":
        raise AssertionError("LightGBM final-training scope is not the frozen full set")
    if not passed_column(ridge_checks["passed"]).all():
        raise AssertionError("Ridge tuning checks did not all pass")
    if not passed_column(lgbm_checks["passed"]).all():
        raise AssertionError("LightGBM tuning checks did not all pass")
    if scalers["source_split"].ne("base_train").any():
        raise AssertionError("Scaler parameters are not base_train-only")
    if bool(registration.get("redraw_or_replacement_after_registration_allowed", True)):
        raise AssertionError("The registered Weather sample is not frozen")

    # Parquet predicate pushdown excludes formal test values before they reach Python.
    data = pd.read_parquet(
        DATA_PATH,
        columns=[
            "dataset_id",
            "series_id",
            "series_type",
            "time_index",
            "value",
            "split",
        ],
        filters=[[('split', '==', split_name)] for split_name in ALLOWED_SPLITS],
    )
    observed_splits = set(data["split"].astype(str).unique().tolist())
    if observed_splits != set(ALLOWED_SPLITS) or (data["split"] == "test").any():
        raise AssertionError(f"Unexpected pre-test data scope: {observed_splits}")
    expected_allowed_rows = int(
        sum(split_summary["aggregate_counts"][name] for name in ALLOWED_SPLITS)
    )
    if expected_allowed_rows != EXPECTED_ALLOWED_ROWS or len(data) != EXPECTED_ALLOWED_ROWS:
        raise AssertionError(
            f"Allowed row mismatch: data={len(data)}, registered={expected_allowed_rows}"
        )

    series_order = split_manifest["series_id"].astype(str).tolist()
    grouped = {
        str(series_id): group.sort_values("time_index", kind="stable").reset_index(drop=True)
        for series_id, group in data.groupby("series_id", sort=False, observed=True)
    }
    if set(grouped) != set(series_order) or len(grouped) != EXPECTED_SERIES:
        raise AssertionError("Processed series differ from the frozen 500-series manifest")
    series_frames = {series_id: grouped[series_id] for series_id in series_order}
    del data, grouped
    gc.collect()

    scaler_lookup = {
        str(row.series_id): (float(row.median), float(row.scale_used))
        for row in scalers.itertuples(index=False)
    }
    if set(series_frames) != set(scaler_lookup):
        raise AssertionError("Series IDs differ between data and scaler table")

    manifest_lookup = {
        str(row.series_id): row for row in split_manifest.itertuples(index=False)
    }
    contiguous_indices_ok = True
    split_order_ok = True
    exact_manifest_counts_ok = True
    series_types_match = True
    split_codes = {name: number for number, name in enumerate(ALLOWED_SPLITS)}
    for series_id, group in series_frames.items():
        row = manifest_lookup[series_id]
        indices = group["time_index"].to_numpy(dtype=np.int64)
        codes = group["split"].astype(str).map(split_codes).to_numpy(dtype=np.int8)
        contiguous_indices_ok &= np.array_equal(indices, np.arange(len(group)))
        split_order_ok &= bool(np.all(np.diff(codes) >= 0))
        observed_counts = group["split"].astype(str).value_counts().to_dict()
        expected_counts = {
            "base_train": int(row.base_train_count),
            "router_train": int(row.router_train_count),
            "calibration": int(row.calibration_count),
        }
        exact_manifest_counts_ok &= observed_counts == expected_counts
        exact_manifest_counts_ok &= len(group) == sum(expected_counts.values())
        series_types_match &= (
            group["series_type"].astype(str).nunique() == 1
            and str(group["series_type"].iloc[0]) == str(row.series_type)
        )

    ridge_window = int(ridge_params["selected_window"])
    lgbm_window = int(lgbm_params["selected_window"])
    if ridge_window != 56 or lgbm_window != 28:
        raise AssertionError(
            f"Expected frozen windows Ridge=56 and LightGBM=28, got "
            f"{ridge_window} and {lgbm_window}"
        )
    if float(ridge_params["selected_alpha"]) != 0.0:
        raise AssertionError("Expected the frozen natural OLS endpoint alpha=0")
    if bool(ridge_params.get("unresolved_alpha_grid_boundary", True)):
        raise AssertionError("Ridge artifact reports an unresolved grid boundary")
    expected_ridge_samples = int(
        window_counts.loc[window_counts["window"] == ridge_window, "base_train"].iloc[0]
    )
    expected_lgbm_samples = int(
        window_counts.loc[window_counts["window"] == lgbm_window, "base_train"].iloc[0]
    )
    if expected_ridge_samples != EXPECTED_RIDGE_TRAIN_SAMPLES:
        raise AssertionError("Registered Ridge target count changed")
    if expected_lgbm_samples != EXPECTED_LGBM_TRAIN_SAMPLES:
        raise AssertionError("Registered LightGBM target count changed")

    ridge_statistics: dict[str, np.ndarray | float | int] = {
        "n": 0,
        "sum_x": np.zeros(ridge_window, dtype=np.float64),
        "sum_y": 0.0,
        "xtx": np.zeros((ridge_window, ridge_window), dtype=np.float64),
        "xty": np.zeros(ridge_window, dtype=np.float64),
    }
    scaled_base_series: list[np.ndarray] = []
    ridge_fit_start = perf_counter()
    for series_id, group in series_frames.items():
        values = group["value"].to_numpy(dtype=np.float64)
        base_count = int(manifest_lookup[series_id].base_train_count)
        median, scale = scaler_lookup[series_id]
        scaled = (values - median) / scale
        scaled_base_series.append(scaled[:base_count].copy())
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

    estimated_gib = expected_lgbm_samples * lgbm_window * 4 / 1024**3
    print(
        f"[准备] LightGBM 全量因果窗口：{expected_lgbm_samples} × "
        f"{lgbm_window}；若密集展开约 {estimated_gib:.2f} GiB",
        flush=True,
    )
    lgbm_sequence = CausalLagSequence(scaled_base_series, lgbm_window)
    lgbm_y = np.concatenate(
        [values[lgbm_window:] for values in scaled_base_series]
    ).astype(np.float32, copy=False)
    lgbm_train_samples = len(lgbm_sequence)
    if lgbm_train_samples != expected_lgbm_samples or len(lgbm_y) != expected_lgbm_samples:
        raise AssertionError("LightGBM full training rows differ from the frozen count")

    sequence_probe_indices = [
        0,
        int(lgbm_sequence.counts[0]) - 1,
        int(lgbm_sequence.offsets[1]),
        len(lgbm_sequence) // 2,
        len(lgbm_sequence) - 1,
    ]
    sequence_probe_ok = True
    for global_index in sequence_probe_indices:
        series_index = int(
            np.searchsorted(lgbm_sequence.offsets, global_index, side="right") - 1
        )
        local_index = int(global_index - lgbm_sequence.offsets[series_index])
        expected_row = sliding_window_view(
            scaled_base_series[series_index], lgbm_window
        )[local_index, ::-1].astype(np.float64)
        sequence_probe_ok &= np.array_equal(lgbm_sequence[global_index], expected_row)
        sequence_probe_ok &= bool(
            lgbm_y[global_index]
            == np.float32(scaled_base_series[series_index][lgbm_window + local_index])
        )
    cross_boundary_start = int(lgbm_sequence.offsets[1]) - 2
    cross_boundary = lgbm_sequence[cross_boundary_start : cross_boundary_start + 4]
    sequence_probe_ok &= np.array_equal(
        cross_boundary,
        np.stack(
            [
                lgbm_sequence[index]
                for index in range(cross_boundary_start, cross_boundary_start + 4)
            ]
        ).astype(np.float32),
    )
    if not sequence_probe_ok:
        raise AssertionError("Out-of-core causal-window Sequence audit failed")
    print(
        f"[完成] 分批训练接口已构造；批大小={SEQUENCE_BATCH_SIZE}，未创建密集矩阵",
        flush=True,
    )

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
    train_params = {
        "objective": "regression_l2",
        "num_leaves": int(lgbm_params["num_leaves"]),
        "learning_rate": float(lgbm_params["learning_rate"]),
        "feature_fraction": float(lgbm_params["feature_fraction"]),
        "seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": -1,
        "verbosity": -1,
    }
    print("[开始] 使用全部 base_train 窗口拟合 LightGBM", flush=True)
    lgbm_fit_start = perf_counter()
    training_dataset = lgb.Dataset(
        data=lgbm_sequence,
        label=lgbm_y,
        params=train_params,
        free_raw_data=True,
    )
    booster = lgb.train(
        params=train_params,
        train_set=training_dataset,
        num_boost_round=int(lgbm_params["n_estimators"]),
    )
    lgbm._Booster = booster
    lgbm._n_features = booster.num_feature()
    lgbm._n_features_in = booster.num_feature()
    lgbm._evals_result = {}
    lgbm._best_iteration = booster.best_iteration
    lgbm._best_score = booster.best_score
    lgbm.fitted_ = True
    booster.free_dataset()
    lgbm_fit_seconds = perf_counter() - lgbm_fit_start
    artifact_probe_x = lgbm_sequence[0:8]
    artifact_probe_before = lgbm.booster_.predict(artifact_probe_x)
    joblib.dump(lgbm, LGBM_MODEL_PATH, compress=3)
    reloaded_lgbm = joblib.load(LGBM_MODEL_PATH)
    artifact_probe_after = reloaded_lgbm.booster_.predict(artifact_probe_x)
    model_artifact_reload_ok = bool(
        reloaded_lgbm.booster_.num_feature() == lgbm_window
        and np.array_equal(artifact_probe_before, artifact_probe_after)
    )
    if not model_artifact_reload_ok:
        raise AssertionError("Saved LightGBM artifact failed reload audit")
    lgbm = reloaded_lgbm
    del training_dataset, lgbm_sequence, lgbm_y, scaled_base_series
    gc.collect()
    print("[完成] LightGBM 全量拟合与模型保存", flush=True)

    prediction_parts: list[pd.DataFrame] = []
    causal_windows_ok = True
    denominators: dict[str, dict[str, float]] = {}
    prediction_start = perf_counter()
    for series_id, group in series_frames.items():
        values = group["value"].to_numpy(dtype=np.float64)
        split_text = group["split"].astype(str)
        base_count = int(manifest_lookup[series_id].base_train_count)
        base_values = values[:base_count]
        seasonal_differences = (
            base_values[SEASONAL_PERIOD:] - base_values[:-SEASONAL_PERIOD]
        )
        squared_denominator = float(np.mean(seasonal_differences**2))
        absolute_denominator = float(np.mean(np.abs(seasonal_differences)))
        if squared_denominator <= 1e-12 or absolute_denominator <= 1e-12:
            raise AssertionError(f"{series_id} has a zero metric denominator")
        denominators[series_id] = {
            "squared": squared_denominator,
            "absolute": absolute_denominator,
        }

        prediction_mask = split_text.isin(PREDICTION_SPLITS).to_numpy()
        targets = group.loc[prediction_mask, "time_index"].to_numpy(dtype=np.int64)
        causal_windows_ok &= bool(
            targets.min() >= max(ridge_window, lgbm_window, SEASONAL_PERIOD)
        )
        median, scale = scaler_lookup[series_id]
        scaled = (values - median) / scale
        ridge_view = sliding_window_view(scaled, ridge_window)
        ridge_x = ridge_view[targets - ridge_window, ::-1]
        ridge_prediction = (
            (ridge_x @ ridge_coefficient + ridge_intercept) * scale + median
        )
        lgbm_view = sliding_window_view(scaled, lgbm_window)
        lgbm_x = lgbm_view[targets - lgbm_window, ::-1].astype(np.float32, copy=True)
        lgbm_prediction = lgbm.booster_.predict(lgbm_x) * scale + median

        metadata = (
            group.loc[
                prediction_mask,
                [
                    "dataset_id",
                    "series_id",
                    "series_type",
                    "time_index",
                    "split",
                    "value",
                ],
            ]
            .rename(columns={"value": "y_true"})
            .reset_index(drop=True)
        )
        metadata.insert(1, "sample_id", SAMPLE_ID)
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
    observed_prediction_splits = set(prediction["split"].astype(str).unique())
    expected_prediction_rows = int(
        sum(split_summary["aggregate_counts"][name] for name in PREDICTION_SPLITS)
    )
    if expected_prediction_rows != EXPECTED_PREDICTION_ROWS:
        raise AssertionError("Registered pre-test prediction count changed")
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
            y_true = group["y_true"].to_numpy(dtype=np.float64)
            series_type = str(group["series_type"].iloc[0])
            for model_name, prediction_column in (
                ("ridge", "ridge_prediction"),
                ("lightgbm", "lightgbm_prediction"),
            ):
                predicted = group[prediction_column].to_numpy(dtype=np.float64)
                error = y_true - predicted
                metric_records.append(
                    {
                        "dataset_id": DATASET_ID,
                        "sample_id": SAMPLE_ID,
                        "split": split_name,
                        "series_id": series_id,
                        "series_type": series_type,
                        "model": model_name,
                        "RMSSE": float(
                            np.sqrt(
                                np.mean(error**2) / denominators[series_id]["squared"]
                            )
                        ),
                        "MASE": float(
                            np.mean(np.abs(error)) / denominators[series_id]["absolute"]
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
            prediction[["ridge_prediction", "lightgbm_prediction"]].to_numpy(float)
        ).all()
    )
    all_metrics_finite = bool(
        np.isfinite(
            per_series[["RMSSE", "MASE", "sMAPE", "RMSE", "MAE"]].to_numpy(float)
        ).all()
    )
    per_split_counts = prediction.groupby("split", observed=True).size().to_dict()
    observed_by_series_split = (
        prediction.groupby(["series_id", "split"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    variable_prediction_counts_match = True
    for series_id, row in manifest_lookup.items():
        variable_prediction_counts_match &= (
            int(observed_by_series_split.loc[series_id, "router_train"])
            == int(row.router_train_count)
        )
        variable_prediction_counts_match &= (
            int(observed_by_series_split.loc[series_id, "calibration"])
            == int(row.calibration_count)
        )

    check_items: list[tuple[str, bool, str]] = [
        (
            "frozen_sample_id_is_consistent",
            sample_ids == {SAMPLE_ID},
            SAMPLE_ID,
        ),
        (
            "registered_sample_cannot_be_redrawn",
            not bool(registration.get("redraw_or_replacement_after_registration_allowed", True)),
            str(registration.get("status")),
        ),
        (
            "ridge_tuning_audit_passed",
            bool(passed_column(ridge_checks["passed"]).all()),
            f"checks={len(ridge_checks)}",
        ),
        (
            "lightgbm_tuning_audit_passed",
            bool(passed_column(lgbm_checks["passed"]).all()),
            f"checks={len(lgbm_checks)}",
        ),
        ("input_excludes_test", "test" not in observed_splits, str(observed_splits)),
        (
            "input_row_count_matches_registered_pretest_splits",
            len(series_frames) == EXPECTED_SERIES
            and EXPECTED_BASE_ROWS == int(split_summary["aggregate_counts"]["base_train"])
            and EXPECTED_ALLOWED_ROWS == sum(len(group) for group in series_frames.values()),
            f"series={len(series_frames)}; rows={sum(len(g) for g in series_frames.values())}",
        ),
        (
            "every_series_split_count_matches_manifest",
            exact_manifest_counts_ok,
            f"series={len(series_frames)}; unequal_lengths_expected=True",
        ),
        ("series_types_match_manifest", series_types_match, str(series_types_match)),
        ("time_indices_are_contiguous", contiguous_indices_ok, str(contiguous_indices_ok)),
        ("split_order_is_chronological", split_order_ok, str(split_order_ok)),
        (
            "scalers_use_base_train_only",
            bool(scalers["source_split"].eq("base_train").all())
            and len(scalers) == EXPECTED_SERIES,
            f"scalers={len(scalers)}",
        ),
        (
            "distinct_frozen_model_windows_are_respected",
            ridge_window == 56 and lgbm_window == 28,
            f"ridge={ridge_window}; lightgbm={lgbm_window}",
        ),
        (
            "ridge_uses_all_eligible_base_targets",
            int(ridge_statistics["n"]) == expected_ridge_samples,
            f"actual={ridge_statistics['n']}; expected={expected_ridge_samples}",
        ),
        (
            "lightgbm_uses_all_eligible_base_targets",
            lgbm_train_samples == expected_lgbm_samples,
            f"actual={lgbm_train_samples}; expected={expected_lgbm_samples}",
        ),
        (
            "lightgbm_out_of_core_sequence_matches_explicit_windows",
            sequence_probe_ok,
            f"probe_indices={sequence_probe_indices}",
        ),
        (
            "lightgbm_dense_training_matrix_was_not_materialized",
            True,
            f"avoided_dense_gib={estimated_gib:.3f}; batch_size={SEQUENCE_BATCH_SIZE}",
        ),
        (
            "saved_lightgbm_artifact_reloads_with_identical_predictions",
            model_artifact_reload_ok,
            str(model_artifact_reload_ok),
        ),
        ("prediction_targets_are_causal", causal_windows_ok, str(causal_windows_ok)),
        (
            "prediction_scope_is_router_and_calibration",
            observed_prediction_splits == set(PREDICTION_SPLITS),
            str(observed_prediction_splits),
        ),
        (
            "prediction_row_counts_match_registered_aggregates",
            len(prediction) == expected_prediction_rows
            and int(per_split_counts.get("router_train", -1))
            == int(split_summary["aggregate_counts"]["router_train"])
            and int(per_split_counts.get("calibration", -1))
            == int(split_summary["aggregate_counts"]["calibration"]),
            f"actual={per_split_counts}; total={len(prediction)}",
        ),
        (
            "variable_per_series_prediction_counts_match_manifest",
            variable_prediction_counts_match,
            f"series={len(observed_by_series_split)}",
        ),
        (
            "each_prediction_split_contains_all_series",
            bool((aggregate["series_count"] == EXPECTED_SERIES).all()),
            str(aggregate[["split", "model", "series_count"]].to_dict("records")),
        ),
        ("all_predictions_are_finite", all_predictions_finite, str(all_predictions_finite)),
        ("all_pretest_metrics_are_finite", all_metrics_finite, str(all_metrics_finite)),
        (
            "model_artifacts_exist",
            RIDGE_MODEL_PATH.is_file() and LGBM_MODEL_PATH.is_file(),
            f"ridge={RIDGE_MODEL_PATH.is_file()}; lightgbm={LGBM_MODEL_PATH.is_file()}",
        ),
    ]
    checks = pd.DataFrame(check_items, columns=["check", "passed", "detail"])
    failed = checks.loc[~checks["passed"]]
    if not failed.empty:
        message = "; ".join(
            f"{row.check}: {row.detail}" for row in failed.itertuples(index=False)
        )
        raise AssertionError(f"Weather Daily base-model audit failed: {message}")

    per_series.to_csv(PER_SERIES_PATH, index=False)
    aggregate.to_csv(AGGREGATE_PATH, index=False)
    checks.to_csv(CHECKS_PATH, index=False)
    model_hashes = {
        "ridge_joblib_sha256": sha256_file(RIDGE_MODEL_PATH),
        "lightgbm_joblib_sha256": sha256_file(LGBM_MODEL_PATH),
    }
    fit_metadata = {
        "dataset_id": DATASET_ID,
        "sample_id": SAMPLE_ID,
        "parameters_frozen_before_fit": True,
        "parameter_files": {
            "ridge": str(RIDGE_PARAMS_PATH),
            "ridge_sha256": sha256_file(RIDGE_PARAMS_PATH),
            "lightgbm": str(LGBM_PARAMS_PATH),
            "lightgbm_sha256": sha256_file(LGBM_PARAMS_PATH),
        },
        "training_scope": "base_train_only",
        "prediction_splits": PREDICTION_SPLITS,
        "formal_test_accessed": False,
        "ridge_window": ridge_window,
        "ridge_alpha": float(ridge_params["selected_alpha"]),
        "ridge_train_samples": int(ridge_statistics["n"]),
        "ridge_fit_method": "exact sufficient statistics",
        "ridge_fit_seconds": float(ridge_fit_seconds),
        "lightgbm_window": lgbm_window,
        "lightgbm_parameters": {
            "num_leaves": int(lgbm_params["num_leaves"]),
            "learning_rate": float(lgbm_params["learning_rate"]),
            "n_estimators": int(lgbm_params["n_estimators"]),
            "feature_fraction": float(lgbm_params["feature_fraction"]),
        },
        "lightgbm_train_samples": lgbm_train_samples,
        "lightgbm_training_sampled": False,
        "lightgbm_fit_method": "LightGBM Sequence batches over every causal target",
        "lightgbm_dense_training_matrix_materialized": False,
        "lightgbm_sequence_batch_size": SEQUENCE_BATCH_SIZE,
        "lightgbm_sequence_probe_passed": sequence_probe_ok,
        "lightgbm_artifact_reload_passed": model_artifact_reload_ok,
        "lightgbm_fit_seconds": float(lgbm_fit_seconds),
        "avoided_dense_training_matrix_gib": float(estimated_gib),
        "prediction_rows": int(len(prediction)),
        "prediction_seconds": float(prediction_seconds),
        "model_hashes": model_hashes,
    }
    METADATA_PATH.write_text(
        yaml.safe_dump(fit_metadata, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    example = prediction.loc[prediction["series_id"].astype(str) == "T3"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)
    for ax, split_name in zip(axes, PREDICTION_SPLITS):
        part = example.loc[example["split"] == split_name].tail(365)
        ax.plot(part["time_index"], part["y_true"], color="#222222", linewidth=1.5, label="actual")
        ax.plot(part["time_index"], part["ridge_prediction"], color="#4C78A8", linewidth=1.1, label="ridge")
        ax.plot(part["time_index"], part["lightgbm_prediction"], color="#F28E2B", linewidth=1.1, label="lightgbm")
        ax.set_title(f"Weather Daily T3 (rain): final 365 days of {split_name}")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=3, frameon=False)
    axes[-1].set_xlabel("Time index (day)")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = {
        "status": "passed",
        "dataset_id": DATASET_ID,
        "sample_id": SAMPLE_ID,
        "check_count": int(len(checks)),
        "failed_check_count": 0,
        "training_scope": "base_train_only",
        "prediction_scope": PREDICTION_SPLITS,
        "formal_test_accessed": False,
        "ridge_train_samples": int(ridge_statistics["n"]),
        "lightgbm_train_samples": lgbm_train_samples,
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
    print("Weather Daily 基础模型训练和预测全部通过")
    print("固定样本编号：", SAMPLE_ID)
    print("训练数据范围：仅 base_train")
    print("预测数据范围：router_train + calibration")
    print("test 是否访问：否")
    print("Ridge 冻结窗口：", ridge_window)
    print("LightGBM 冻结窗口：", lgbm_window)
    print("Ridge 训练样本：", int(ridge_statistics["n"]))
    print("LightGBM 训练样本：", lgbm_train_samples)
    print("LightGBM 是否使用调参抽样：否（这里使用全部合格窗口）")
    print("预测总行数：", len(prediction))
    print("因果窗口检查：通过")
    print("预测试指标：")
    print(
        aggregate[["split", "model", "mean_RMSSE", "mean_MASE", "mean_sMAPE"]]
        .to_string(index=False)
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
