#!/usr/bin/env python3
"""Local weighting and final Weather Daily soft-router fit.

Local weighting is selected strictly within router_train.  Calibration is read
only after the choice and final fit are fixed, and formal test is never read."""

from __future__ import annotations

import gc
import json
import os
from itertools import product
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sci-routing")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import yaml


PROJECT_ROOT = Path(
    os.environ.get("SCI_ROUTING_ROOT", Path(__file__).resolve().parents[3])
).resolve()
OUTPUT_ROOT = Path(os.environ.get("SCI_ROUTING_OUTPUT_ROOT", PROJECT_ROOT)).resolve()

FEATURE_PATH = PROJECT_ROOT / "results/weather_daily_router_features.parquet"
FEATURE_MANIFEST_PATH = PROJECT_ROOT / "results/weather_daily_router_feature_manifest.csv"
FEATURE_CHECKS_PATH = PROJECT_ROOT / "results/weather_daily_router_feature_checks.csv"
SELECTED_ROUTER_PATH = PROJECT_ROOT / "results/weather_daily_selected_soft_router_params.yaml"
ROUTER_DETAIL_PATH = PROJECT_ROOT / "results/weather_daily_soft_router_rolling_validation.csv"
ROUTER_FOLD_PATH = PROJECT_ROOT / "results/weather_daily_soft_router_fold_manifest.csv"
ROUTER_SAMPLE_PATH = PROJECT_ROOT / "results/weather_daily_soft_router_sample_positions.parquet"
ROUTER_CHECKS_PATH = PROJECT_ROOT / "results/weather_daily_soft_router_tuning_checks.csv"
SPLIT_MANIFEST_PATH = PROJECT_ROOT / "results/weather_daily_split_manifest.csv"
PROTOCOL_AMENDMENT_PATH = PROJECT_ROOT / "logs/weather_daily_protocol_amendments.md"
CONFIG_PATH = PROJECT_ROOT / "experiment_config.yaml"

DETAIL_PATH = OUTPUT_ROOT / "results/weather_daily_local_weighting_rolling_validation.csv"
SUMMARY_PATH = OUTPUT_ROOT / "results/weather_daily_local_weighting_tuning_summary.csv"
SELECTED_LOCAL_PATH = OUTPUT_ROOT / "results/weather_daily_selected_local_weighting_params.yaml"
FINAL_SAMPLE_PATH = OUTPUT_ROOT / "results/weather_daily_final_router_sample_positions.parquet"
NEIGHBOR_MANIFEST_PATH = OUTPUT_ROOT / "results/weather_daily_local_weighting_neighbor_manifest.csv"
COEFFICIENT_PATH = OUTPUT_ROOT / "results/weather_daily_router_coefficients.csv"
CALIBRATION_SCORE_PATH = OUTPUT_ROOT / "results/weather_daily_calibration_router_scores.parquet"
CHECKS_PATH = OUTPUT_ROOT / "results/weather_daily_local_weighting_checks.csv"
MODEL_PATH = OUTPUT_ROOT / "models/weather_daily_soft_router.joblib"
FIGURE_PATH = OUTPUT_ROOT / "figures/weather_daily_router_coefficients_and_scores.png"
REPORT_PATH = OUTPUT_ROOT / "logs/weather_daily_local_weighting_report.json"

DATASET_ID = "weather_daily"
SAMPLE_ID = "5518ada55d6adb8a8df20f5a30d52ec904362ce57df9adeeceb0bc3d22854044"
N_FOLDS = 3
MAX_TRAINING_SAMPLE_ROWS = 100_000
NEIGHBOR_ALGORITHM = "kd_tree"
NEIGHBOR_LEAF_SIZE = 60
NEIGHBOR_SPACE = "standardized context features only"
FINAL_SAMPLE_METHOD = "global midpoint systematic sampling over complete per-series router_train"
SOLVER = "saga"
SOLVER_TOLERANCE = 1e-4
INTERCEPT_IMPLEMENTATION = "explicit constant feature, L1 penalized, then collapsed for inference"
EXPECTED_SERIES = 500
EXPECTED_SELECTED_FEATURES = 30
EXPECTED_ROUTER_ROWS = 1_097_616
EXPECTED_CALIBRATION_ROWS = 736_875
EXPECTED_VALIDATION_ROWS_PER_FOLD = 28_000
EXPECTED_AVAILABLE_TRAIN_ROWS = {1: 1_013_616, 2: 1_041_616, 3: 1_069_616}
EXPECTED_SAMPLE_ROWS = 100_000
REGISTERED_ALPHA_GRID = [0.0, 0.5, 1.0, 2.0]
EXPECTED_LOCAL_COMBINATIONS = 12
EXPECTED_ROLLING_FITS = 36


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def passed_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1"})


def stable_soft_target(values: np.ndarray, temperature: float) -> np.ndarray:
    normalized = np.clip(
        np.asarray(values, dtype=np.float64) / float(temperature), -40.0, 40.0
    )
    return 1.0 / (1.0 + np.exp(-normalized))


def equal_series_mean(values: np.ndarray, lengths: np.ndarray) -> float:
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    series_means = np.asarray(
        [
            np.mean(values[offsets[index] : offsets[index + 1]])
            for index in range(len(lengths))
        ],
        dtype=np.float64,
    )
    return float(series_means.mean())


def systematic_sample_indices(
    indices_by_series: list[np.ndarray], sample_cap: int
) -> tuple[np.ndarray, np.ndarray]:
    concatenated = np.concatenate(indices_by_series)
    sample_size = min(len(concatenated), int(sample_cap))
    positions = np.floor(
        (np.arange(sample_size, dtype=np.float64) + 0.5)
        * len(concatenated)
        / sample_size
    ).astype(np.int64)
    if len(np.unique(positions)) != sample_size:
        raise AssertionError("Systematic sample contains duplicate positions")
    return concatenated[positions], positions


def sample_indices_from_saved_positions(
    router: pd.DataFrame, saved_positions: pd.DataFrame, fold: int
) -> np.ndarray:
    fold_positions = saved_positions.loc[
        saved_positions["fold"] == fold,
        ["series_id", "time_index", "sample_order_within_fold"],
    ].copy()
    fold_positions["series_id"] = fold_positions["series_id"].astype(str)
    lookup = router[["series_id", "time_index"]].copy()
    lookup["series_id"] = lookup["series_id"].astype(str)
    lookup["dataframe_index"] = np.arange(len(lookup), dtype=np.int64)
    matched = fold_positions.merge(
        lookup,
        on=["series_id", "time_index"],
        how="left",
        validate="one_to_one",
    ).sort_values("sample_order_within_fold", kind="stable")
    if matched["dataframe_index"].isna().any():
        raise AssertionError(f"Fold {fold} has unmatched saved sample positions")
    expected_order = np.arange(len(matched), dtype=np.int64)
    if not np.array_equal(
        matched["sample_order_within_fold"].to_numpy(dtype=np.int64),
        expected_order,
    ):
        raise AssertionError(f"Fold {fold} saved sample order is not contiguous")
    return matched["dataframe_index"].to_numpy(dtype=np.int64)


def build_exact_neighbor_matrix(
    standardized_context: np.ndarray, maximum_k: int
) -> tuple[np.ndarray, float]:
    start = perf_counter()
    model = NearestNeighbors(
        n_neighbors=maximum_k + 1,
        algorithm=NEIGHBOR_ALGORITHM,
        leaf_size=NEIGHBOR_LEAF_SIZE,
        metric="euclidean",
        n_jobs=1,
    )
    model.fit(standardized_context)
    raw = model.kneighbors(standardized_context, return_distance=False)
    cleaned = np.empty((len(raw), maximum_k), dtype=np.int64)
    for row_number, neighbors in enumerate(raw):
        without_self = neighbors[neighbors != row_number]
        if len(without_self) < maximum_k:
            raise AssertionError("An exact-neighbor row contains too few neighbors")
        cleaned[row_number] = without_self[:maximum_k]
    return cleaned, float(perf_counter() - start)


def calculate_local_weights(
    hard_target: np.ndarray,
    neighbor_indices: np.ndarray,
    k_value: int,
    alpha_value: float,
    maximum_multiplier: float,
) -> np.ndarray:
    if alpha_value == 0.0:
        return np.ones(len(hard_target), dtype=np.float64)
    selected_neighbors = neighbor_indices[:, :k_value]
    neighbor_targets = hard_target[selected_neighbors]
    local_support = np.mean(
        neighbor_targets == hard_target[:, None], axis=1, dtype=np.float64
    )
    safe_support = np.maximum(local_support, 1.0 / (k_value + 1.0))
    weights = np.minimum(
        (1.0 / safe_support) ** alpha_value, maximum_multiplier
    )
    return weights / np.mean(weights)


def fit_soft_router_explicit_intercept(
    train_x: np.ndarray,
    soft_target: np.ndarray,
    local_weights: np.ndarray,
    c_value: float,
    seed: int,
) -> LogisticRegression:
    duplicated_x = np.vstack([train_x, train_x])
    duplicated_x = np.column_stack(
        [duplicated_x, np.ones(len(duplicated_x), dtype=np.float32)]
    )
    duplicated_y = np.concatenate(
        [
            np.ones(len(train_x), dtype=np.int8),
            np.zeros(len(train_x), dtype=np.int8),
        ]
    )
    duplicated_weights = np.concatenate(
        [local_weights * soft_target, local_weights * (1.0 - soft_target)]
    )
    model = LogisticRegression(
        solver=SOLVER,
        l1_ratio=1.0,
        C=c_value,
        fit_intercept=False,
        max_iter=3000,
        tol=SOLVER_TOLERANCE,
        random_state=seed,
    )
    model.fit(duplicated_x, duplicated_y, sample_weight=duplicated_weights)
    if int(model.n_iter_[0]) >= model.max_iter:
        raise AssertionError("Soft-router logistic regression did not converge")
    return model


def augmented_probability(model: LogisticRegression, x: np.ndarray) -> np.ndarray:
    augmented = np.column_stack([x, np.ones(len(x), dtype=np.float32)])
    return np.clip(
        model.predict_proba(augmented)[:, 1].astype(np.float64, copy=False),
        1e-8,
        1.0 - 1e-8,
    )


def collapse_for_inference(
    trained_model: LogisticRegression, feature_count: int, c_value: float, seed: int
) -> LogisticRegression:
    """Convert the explicit constant coefficient into a standard intercept."""

    if trained_model.coef_.shape != (1, feature_count + 1):
        raise AssertionError("Explicit-intercept training model shape changed")
    inference_model = LogisticRegression(
        solver=SOLVER,
        l1_ratio=1.0,
        C=c_value,
        fit_intercept=True,
        max_iter=3000,
        tol=SOLVER_TOLERANCE,
        random_state=seed,
    )
    inference_model.classes_ = trained_model.classes_.copy()
    inference_model.coef_ = trained_model.coef_[:, :feature_count].copy()
    inference_model.intercept_ = trained_model.coef_[:, feature_count].copy()
    inference_model.n_features_in_ = feature_count
    inference_model.n_iter_ = trained_model.n_iter_.copy()
    return inference_model


def main() -> None:
    input_paths = (
        FEATURE_PATH,
        FEATURE_MANIFEST_PATH,
        FEATURE_CHECKS_PATH,
        SELECTED_ROUTER_PATH,
        ROUTER_DETAIL_PATH,
        ROUTER_FOLD_PATH,
        ROUTER_SAMPLE_PATH,
        ROUTER_CHECKS_PATH,
        SPLIT_MANIFEST_PATH,
        PROTOCOL_AMENDMENT_PATH,
        CONFIG_PATH,
    )
    for path in input_paths:
        require_file(path)
    output_paths = (
        DETAIL_PATH,
        SUMMARY_PATH,
        SELECTED_LOCAL_PATH,
        FINAL_SAMPLE_PATH,
        NEIGHBOR_MANIFEST_PATH,
        COEFFICIENT_PATH,
        CALIBRATION_SCORE_PATH,
        CHECKS_PATH,
        MODEL_PATH,
        FIGURE_PATH,
        REPORT_PATH,
    )
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    overall_start = perf_counter()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    selected_router = yaml.safe_load(SELECTED_ROUTER_PATH.read_text(encoding="utf-8"))
    feature_manifest = pd.read_csv(FEATURE_MANIFEST_PATH)
    feature_checks = pd.read_csv(FEATURE_CHECKS_PATH)
    router_checks = pd.read_csv(ROUTER_CHECKS_PATH)
    split_manifest = pd.read_csv(SPLIT_MANIFEST_PATH).sort_values("sample_order")
    protocol_text = PROTOCOL_AMENDMENT_PATH.read_text(encoding="utf-8")
    upstream_checks_passed = bool(
        passed_column(feature_checks["passed"]).all()
        and passed_column(router_checks["passed"]).all()
    )
    if not upstream_checks_passed:
        raise AssertionError("An upstream Weather Daily audit did not pass")
    if selected_router["selection_scope"] != "router_train_only":
        raise AssertionError("Selected soft-router parameters are not router-only")
    if bool(selected_router.get("calibration_accessed", True)):
        raise AssertionError("Soft-router tuning accessed calibration")
    if bool(selected_router.get("test_accessed", True)):
        raise AssertionError("Soft-router tuning accessed test")
    if config["local_weighting"]["fit_scope"] != "router_train_only":
        raise AssertionError("Local weighting is not configured as router-only")

    sample_ids = {
        str(selected_router.get("sample_id")),
        *split_manifest["sample_id"].astype(str).unique().tolist(),
    }
    if sample_ids != {SAMPLE_ID}:
        raise AssertionError(f"Frozen sample IDs disagree: {sample_ids}")
    if (
        selected_router.get("solver") != SOLVER
        or float(selected_router.get("solver_tolerance", -1)) != SOLVER_TOLERANCE
        or "explicit constant" not in str(selected_router.get("intercept_implementation"))
        or "scalable L1 logistic solver" not in protocol_text
    ):
        raise AssertionError("The soft-router solver amendment is missing or inconsistent")

    seed = int(config["study"]["seed"])
    target_simple_coverage = float(config["study"]["primary_target_coverage"])
    temperature = float(selected_router["temperature"])
    residual_lag = int(selected_router["residual_lag"])
    c_value = float(selected_router["C"])
    feature_names = list(selected_router["feature_names"])
    k_grid = [int(value) for value in config["local_weighting"]["k_grid"]]
    alpha_grid = [float(value) for value in config["local_weighting"]["alpha_grid"]]
    maximum_multiplier = float(config["local_weighting"]["maximum_multiplier"])
    if k_grid != [5, 11, 21] or alpha_grid != REGISTERED_ALPHA_GRID:
        raise AssertionError(
            f"Unexpected local-weighting grid: k={k_grid}, alpha={alpha_grid}"
        )
    context_features = feature_manifest.loc[
        feature_manifest["group"] == "context", "feature"
    ].tolist()
    if len(context_features) != 14:
        raise AssertionError(f"Expected 14 context features, found {len(context_features)}")
    if not (
        temperature == 0.5
        and residual_lag == 8
        and c_value == 0.1
        and len(feature_names) == EXPECTED_SELECTED_FEATURES
    ):
        raise AssertionError("Frozen Weather soft-router parameters changed")
    if not set(context_features).issubset(feature_names):
        raise AssertionError("Selected router omitted a required context feature")
    if len(feature_names) != len(set(feature_names)):
        raise AssertionError("Selected router feature names are not unique")
    if not set(feature_names).issubset(set(feature_manifest["feature"])):
        raise AssertionError("Selected router contains an unregistered feature")
    context_column_indices = np.asarray(
        [feature_names.index(name) for name in context_features], dtype=np.int64
    )

    # Selection stage: router_train only, loaded with parquet predicate pushdown.
    router_columns = list(
        dict.fromkeys(
            [
                "dataset_id",
                "sample_id",
                "series_id",
                "series_type",
                "time_index",
                "split",
                "y_true",
                "ridge_prediction",
                "lightgbm_prediction",
                "hard_black_box_target",
                "loss_advantage_black_box",
                "seasonal_naive_mae_scale",
                *feature_names,
            ]
        )
    )
    router = pd.read_parquet(
        FEATURE_PATH, columns=router_columns, filters=[("split", "==", "router_train")]
    )
    if set(router["sample_id"].astype(str).unique()) != {SAMPLE_ID}:
        raise AssertionError("Router input does not carry the frozen sample ID")
    sample_order_map = split_manifest.set_index("series_id")["sample_order"].to_dict()
    router["_series_order"] = router["series_id"].astype(str).map(sample_order_map)
    if router["_series_order"].isna().any():
        raise AssertionError("Router input contains an unregistered series")
    router["series_id"] = router["series_id"].astype(str)
    router = (
        router.sort_values(["_series_order", "time_index"], kind="stable")
        .drop(columns="_series_order")
        .reset_index(drop=True)
    )
    router_splits = set(router["split"].astype(str).unique().tolist())
    if router.empty or router_splits != {"router_train"}:
        raise AssertionError(f"Invalid local-weighting input scope: {router_splits}")
    if len(router) != EXPECTED_ROUTER_ROWS:
        raise AssertionError(
            f"Expected {EXPECTED_ROUTER_ROWS} router rows, got {len(router)}"
        )
    expected_router_counts = {
        str(row.series_id): int(row.router_train_count) - 16
        for row in split_manifest.itertuples(index=False)
    }
    actual_router_counts = {
        str(series_id): int(len(group))
        for series_id, group in router.groupby("series_id", sort=False, observed=True)
    }
    variable_router_counts_match = actual_router_counts == expected_router_counts
    if router[feature_names].isna().any().any():
        raise AssertionError("router_train contains missing selected features")

    fold_manifest = pd.read_csv(ROUTER_FOLD_PATH)
    fold_manifest["series_id"] = fold_manifest["series_id"].astype(str)
    saved_sample_positions = pd.read_parquet(ROUTER_SAMPLE_PATH)
    saved_sample_positions["series_id"] = saved_sample_positions["series_id"].astype(str)
    router_details = pd.read_csv(ROUTER_DETAIL_PATH)
    if set(fold_manifest["sample_id"].astype(str).unique()) != {SAMPLE_ID}:
        raise AssertionError("Saved fold manifest sample ID changed")
    if set(saved_sample_positions["sample_id"].astype(str).unique()) != {SAMPLE_ID}:
        raise AssertionError("Saved sample-position sample ID changed")
    fold_lookup = fold_manifest.set_index(["series_id", "fold"])
    series_index_lookup = {
        str(series_id): group.index.to_numpy(dtype=np.int64)
        for series_id, group in router.groupby("series_id", sort=False, observed=True)
    }
    series_order = split_manifest["series_id"].astype(str).tolist()
    if set(series_index_lookup) != set(series_order) or len(series_order) != EXPECTED_SERIES:
        raise AssertionError("Router series differ from the frozen sample")
    observed_available_train_rows = {
        int(fold): int(value)
        for fold, value in fold_manifest.groupby("fold")["train_count"].sum().items()
    }
    exact_fold_schedule_ok = bool(
        observed_available_train_rows == EXPECTED_AVAILABLE_TRAIN_ROWS
        and set(fold_manifest["validation_count"].astype(int).unique()) == {56}
    )
    if not exact_fold_schedule_ok:
        raise AssertionError("Saved Weather router fold schedule changed")
    saved_sample_counts = saved_sample_positions.groupby("fold", observed=True).size().to_dict()
    if not (
        len(saved_sample_counts) == N_FOLDS
        and set(saved_sample_counts.values()) == {EXPECTED_SAMPLE_ROWS}
    ):
        raise AssertionError("Saved Weather router samples changed")

    records: list[dict[str, object]] = []
    neighbor_records: list[dict[str, object]] = []
    all_models_converged = True
    folds_chronological = True
    samples_match_soft_router_tuning = True

    for fold in range(1, N_FOLDS + 1):
        train_indices_by_series: list[np.ndarray] = []
        validation_indices_by_series: list[np.ndarray] = []
        for series_id in series_order:
            indices = series_index_lookup[series_id]
            schedule = fold_lookup.loc[(series_id, fold)]
            train_count = int(schedule["train_count"])
            validation_count = int(schedule["validation_count"])
            train_part = indices[:train_count]
            validation_part = indices[train_count : train_count + validation_count]
            folds_chronological &= bool(
                router.loc[train_part, "time_index"].max()
                < router.loc[validation_part, "time_index"].min()
            )
            train_indices_by_series.append(train_part)
            validation_indices_by_series.append(validation_part)

        full_train_indices = np.concatenate(train_indices_by_series)
        validation_indices = np.concatenate(validation_indices_by_series)
        validation_lengths = np.asarray(
            [len(values) for values in validation_indices_by_series], dtype=np.int64
        )
        sampled_train_indices = sample_indices_from_saved_positions(
            router, saved_sample_positions, fold
        )
        independently_sampled, _ = systematic_sample_indices(
            train_indices_by_series, MAX_TRAINING_SAMPLE_ROWS
        )
        samples_match_soft_router_tuning &= np.array_equal(
            sampled_train_indices, independently_sampled
        )
        if len(sampled_train_indices) != EXPECTED_SAMPLE_ROWS:
            raise AssertionError(f"Fold {fold} sample count changed")
        if len(validation_indices) != EXPECTED_VALIDATION_ROWS_PER_FOLD:
            raise AssertionError(f"Fold {fold} validation row count changed")

        scaler = StandardScaler()
        scaler.fit(
            router.iloc[full_train_indices][feature_names].to_numpy(dtype=np.float32)
        )
        train_x = scaler.transform(
            router.iloc[sampled_train_indices][feature_names].to_numpy(dtype=np.float32)
        ).astype(np.float32)
        validation_x = scaler.transform(
            router.iloc[validation_indices][feature_names].to_numpy(dtype=np.float32)
        ).astype(np.float32)
        neighbor_indices, neighbor_seconds = build_exact_neighbor_matrix(
            train_x[:, context_column_indices], max(k_grid)
        )
        train_hard_target = router.iloc[sampled_train_indices][
            "hard_black_box_target"
        ].to_numpy(dtype=np.int8)
        train_soft_target = stable_soft_target(
            router.iloc[sampled_train_indices]["loss_advantage_black_box"].to_numpy(
                dtype=np.float64
            ),
            temperature,
        )
        validation_hard_target = router.iloc[validation_indices][
            "hard_black_box_target"
        ].to_numpy(dtype=np.int8)
        validation_soft_target = stable_soft_target(
            router.iloc[validation_indices]["loss_advantage_black_box"].to_numpy(
                dtype=np.float64
            ),
            temperature,
        )
        validation_y = router.iloc[validation_indices]["y_true"].to_numpy(
            dtype=np.float64
        )
        validation_ridge = router.iloc[validation_indices]["ridge_prediction"].to_numpy(
            dtype=np.float64
        )
        validation_lgbm = router.iloc[validation_indices]["lightgbm_prediction"].to_numpy(
            dtype=np.float64
        )
        validation_scale = router.iloc[validation_indices][
            "seasonal_naive_mae_scale"
        ].to_numpy(dtype=np.float64)

        neighbor_records.append(
            {
                "dataset_id": DATASET_ID,
                "sample_id": SAMPLE_ID,
                "scope": "rolling_validation_fold",
                "fold": fold,
                "sample_rows": len(sampled_train_indices),
                "neighbor_feature_space": NEIGHBOR_SPACE,
                "neighbor_feature_count": len(context_features),
                "algorithm": NEIGHBOR_ALGORITHM,
                "leaf_size": NEIGHBOR_LEAF_SIZE,
                "maximum_k": max(k_grid),
                "exact_query": True,
                "seconds": neighbor_seconds,
            }
        )

        for k_value, alpha_value in product(k_grid, alpha_grid):
            local_weights = calculate_local_weights(
                train_hard_target,
                neighbor_indices,
                k_value,
                alpha_value,
                maximum_multiplier,
            )
            trained_model = fit_soft_router_explicit_intercept(
                train_x, train_soft_target, local_weights, c_value, seed
            )
            converged = bool(int(trained_model.n_iter_[0]) < trained_model.max_iter)
            all_models_converged &= converged
            probability = augmented_probability(trained_model, validation_x)
            threshold = float(np.quantile(probability, target_simple_coverage))
            use_black_box = probability >= threshold
            selected_prediction = np.where(
                use_black_box, validation_lgbm, validation_ridge
            )
            scaled_error = (
                (validation_y - selected_prediction) / validation_scale
            ) ** 2
            records.append(
                {
                    "dataset_id": DATASET_ID,
                    "sample_id": SAMPLE_ID,
                    "fold": fold,
                    "training_scope": "saved_deterministic_router_train_sample",
                    "validation_scope": "complete_later_router_train_blocks",
                    "neighbor_space": NEIGHBOR_SPACE,
                    "k": k_value,
                    "alpha": alpha_value,
                    "temperature": temperature,
                    "residual_lag": residual_lag,
                    "C": c_value,
                    "feature_count": len(feature_names),
                    "neighbor_feature_count": len(context_features),
                    "available_train_rows": len(full_train_indices),
                    "sampled_train_rows": len(sampled_train_indices),
                    "scaler_fit_rows": len(full_train_indices),
                    "validation_rows": len(validation_indices),
                    "validation_size_min": int(validation_lengths.min()),
                    "validation_size_max": int(validation_lengths.max()),
                    "constrained_scaled_loss": equal_series_mean(
                        scaled_error, validation_lengths
                    ),
                    "pooled_constrained_scaled_loss": float(np.mean(scaled_error)),
                    "simple_coverage": float(np.mean(~use_black_box)),
                    "soft_brier": float(
                        np.mean((probability - validation_soft_target) ** 2)
                    ),
                    "hard_brier": float(
                        brier_score_loss(validation_hard_target, probability)
                    ),
                    "black_box_AUPRC": float(
                        average_precision_score(validation_hard_target, probability)
                    ),
                    "mean_local_weight": float(np.mean(local_weights)),
                    "maximum_local_weight": float(np.max(local_weights)),
                    "nonzero_feature_coefficients": int(
                        np.count_nonzero(trained_model.coef_[:, :-1])
                    ),
                    "intercept_nonzero": bool(trained_model.coef_[0, -1] != 0.0),
                    "solver": SOLVER,
                    "converged": converged,
                    "seed": seed,
                }
            )

        print(
            f"[完成] fold={fold}/{N_FOLDS}，精确近邻={neighbor_seconds:.2f}秒，"
            f"局部加权组合={len(k_grid) * len(alpha_grid)}",
            flush=True,
        )
        del (
            train_x,
            validation_x,
            neighbor_indices,
            train_hard_target,
            train_soft_target,
            validation_hard_target,
            validation_soft_target,
            validation_y,
            validation_ridge,
            validation_lgbm,
            validation_scale,
        )
        gc.collect()

    details = pd.DataFrame(records)
    summary = (
        details.groupby(["k", "alpha"], as_index=False)
        .agg(
            mean_constrained_scaled_loss=("constrained_scaled_loss", "mean"),
            std_constrained_scaled_loss=("constrained_scaled_loss", "std"),
            mean_pooled_constrained_scaled_loss=("pooled_constrained_scaled_loss", "mean"),
            mean_simple_coverage=("simple_coverage", "mean"),
            mean_soft_brier=("soft_brier", "mean"),
            mean_hard_brier=("hard_brier", "mean"),
            mean_black_box_AUPRC=("black_box_AUPRC", "mean"),
            mean_nonzero_feature_coefficients=("nonzero_feature_coefficients", "mean"),
            mean_maximum_local_weight=("maximum_local_weight", "mean"),
            folds=("fold", "nunique"),
        )
        .sort_values(
            ["mean_constrained_scaled_loss", "mean_soft_brier", "alpha", "k"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    best = summary.iloc[0]
    best_k = int(best["k"])
    best_alpha = float(best["alpha"])
    local_weighting_selected = bool(best_alpha > 0.0)

    selected_router_rows = router_details.loc[
        np.isclose(router_details["temperature"], temperature)
        & (router_details["residual_lag"] == residual_lag)
        & np.isclose(router_details["C"], c_value)
    ].sort_values("fold")
    alpha_zero_rows = details.loc[
        (details["k"] == min(k_grid)) & np.isclose(details["alpha"], 0.0)
    ].sort_values("fold")
    if len(selected_router_rows) != N_FOLDS or len(alpha_zero_rows) != N_FOLDS:
        raise AssertionError("Cannot compare alpha=0 against the soft-router tuning stage")
    alpha_zero_max_difference = float(
        np.max(
            np.abs(
                selected_router_rows["constrained_scaled_loss"].to_numpy(np.float64)
                - alpha_zero_rows["constrained_scaled_loss"].to_numpy(np.float64)
            )
        )
    )

    # Selection is fixed. Fit the final router using router_train only.
    final_scaler = StandardScaler()
    final_scaler.fit(router[feature_names].to_numpy(dtype=np.float32))
    all_indices_by_series = [series_index_lookup[series_id] for series_id in series_order]
    final_sample_indices, _ = systematic_sample_indices(
        all_indices_by_series, MAX_TRAINING_SAMPLE_ROWS
    )
    final_train_x = final_scaler.transform(
        router.iloc[final_sample_indices][feature_names].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    final_hard_target = router.iloc[final_sample_indices][
        "hard_black_box_target"
    ].to_numpy(dtype=np.int8)
    final_soft_target = stable_soft_target(
        router.iloc[final_sample_indices]["loss_advantage_black_box"].to_numpy(
            dtype=np.float64
        ),
        temperature,
    )
    if local_weighting_selected:
        final_neighbors, final_neighbor_seconds = build_exact_neighbor_matrix(
            final_train_x[:, context_column_indices], best_k
        )
        final_local_weights = calculate_local_weights(
            final_hard_target,
            final_neighbors,
            best_k,
            best_alpha,
            maximum_multiplier,
        )
        del final_neighbors
    else:
        final_neighbor_seconds = 0.0
        final_local_weights = np.ones(len(final_sample_indices), dtype=np.float64)
    final_training_model = fit_soft_router_explicit_intercept(
        final_train_x, final_soft_target, final_local_weights, c_value, seed
    )
    final_converged = bool(
        int(final_training_model.n_iter_[0]) < final_training_model.max_iter
    )
    final_model = collapse_for_inference(
        final_training_model, len(feature_names), c_value, seed
    )
    collapse_probe = final_train_x[:32]
    collapse_max_difference = float(
        np.max(
            np.abs(
                augmented_probability(final_training_model, collapse_probe)
                - np.clip(
                    final_model.predict_proba(collapse_probe)[:, 1].astype(np.float64),
                    1e-8,
                    1.0 - 1e-8,
                )
            )
        )
    )
    if collapse_max_difference > 1e-6:
        raise AssertionError(
            f"Collapsed inference model differs by {collapse_max_difference:.3e}"
        )

    final_sample_positions = router.iloc[final_sample_indices][
        ["dataset_id", "sample_id", "series_id", "series_type", "time_index"]
    ].copy()
    final_sample_positions["sample_order"] = np.arange(
        len(final_sample_positions), dtype=np.int64
    )
    final_sample_positions["sampling_method"] = FINAL_SAMPLE_METHOD
    for column in ("dataset_id", "sample_id", "series_id", "series_type"):
        final_sample_positions[column] = final_sample_positions[column].astype("category")
    final_sample_positions.to_parquet(
        FINAL_SAMPLE_PATH, index=False, compression="snappy"
    )
    all_series_in_final_sample = (
        final_sample_positions["series_id"].nunique() == len(series_order)
    )

    neighbor_records.append(
        {
            "dataset_id": DATASET_ID,
            "sample_id": SAMPLE_ID,
            "scope": "final_router_fit",
            "fold": 0,
            "sample_rows": len(final_sample_indices),
            "neighbor_feature_space": NEIGHBOR_SPACE,
            "neighbor_feature_count": len(context_features),
            "algorithm": NEIGHBOR_ALGORITHM,
            "leaf_size": NEIGHBOR_LEAF_SIZE,
            "maximum_k": best_k,
            "exact_query": bool(local_weighting_selected),
            "seconds": final_neighbor_seconds,
        }
    )
    neighbor_manifest = pd.DataFrame(neighbor_records)

    model_bundle = {
        "dataset_id": DATASET_ID,
        "sample_id": SAMPLE_ID,
        "model": final_model,
        "scaler": final_scaler,
        "feature_names": feature_names,
        "temperature": temperature,
        "residual_lag": residual_lag,
        "C": c_value,
        "solver": SOLVER,
        "solver_tolerance": SOLVER_TOLERANCE,
        "intercept_implementation": INTERCEPT_IMPLEMENTATION,
        "trained_intercept_was_l1_penalized": True,
        "collapse_max_probability_difference": collapse_max_difference,
        "k": best_k,
        "alpha": best_alpha,
        "maximum_local_weight_multiplier": maximum_multiplier,
        "local_weighting_selected": local_weighting_selected,
        "neighbor_space": NEIGHBOR_SPACE,
        "neighbor_features": context_features,
        "neighbor_algorithm": NEIGHBOR_ALGORITHM,
        "target_simple_coverage": target_simple_coverage,
        "training_scope": "router_train_only",
        "scaler_fit_scope": "all_router_train_rows",
        "model_fit_scope": "saved_deterministic_sample_up_to_100000_rows",
        "final_training_sample_path": str(FINAL_SAMPLE_PATH),
        "calibration_used_for_model_fit": False,
        "test_accessed": False,
        "seed": seed,
    }
    artifact_probe_raw = router.iloc[final_sample_indices[:8]][feature_names].to_numpy(
        dtype=np.float32
    )
    artifact_probe_before = final_model.predict_proba(
        final_scaler.transform(artifact_probe_raw).astype(np.float32)
    )[:, 1]
    joblib.dump(model_bundle, MODEL_PATH)
    reloaded_bundle = joblib.load(MODEL_PATH)
    artifact_probe_after = reloaded_bundle["model"].predict_proba(
        reloaded_bundle["scaler"].transform(artifact_probe_raw).astype(np.float32)
    )[:, 1]
    model_artifact_reload_ok = bool(
        reloaded_bundle["dataset_id"] == DATASET_ID
        and reloaded_bundle["sample_id"] == SAMPLE_ID
        and reloaded_bundle["feature_names"] == feature_names
        and np.array_equal(artifact_probe_before, artifact_probe_after)
    )
    if not model_artifact_reload_ok:
        raise AssertionError("Saved final router bundle failed reload audit")

    coefficients = pd.DataFrame(
        {"feature": feature_names, "coefficient": final_model.coef_[0]}
    )
    coefficients["absolute_coefficient"] = coefficients["coefficient"].abs()
    coefficients = coefficients.sort_values(
        "absolute_coefficient", ascending=False, kind="stable"
    ).reset_index(drop=True)
    coefficients.to_csv(COEFFICIENT_PATH, index=False)

    # Only now, after selection and final fit, read calibration for scoring.
    calibration = pd.read_parquet(
        FEATURE_PATH,
        columns=router_columns,
        filters=[("split", "==", "calibration")],
    )
    if set(calibration["sample_id"].astype(str).unique()) != {SAMPLE_ID}:
        raise AssertionError("Calibration input sample ID changed")
    calibration["series_id"] = calibration["series_id"].astype(str)
    calibration["_series_order"] = calibration["series_id"].map(sample_order_map)
    calibration = (
        calibration.sort_values(["_series_order", "time_index"], kind="stable")
        .drop(columns="_series_order")
        .reset_index(drop=True)
    )
    calibration_splits = set(calibration["split"].astype(str).unique().tolist())
    if calibration.empty or calibration_splits != {"calibration"}:
        raise AssertionError(f"Invalid calibration scope: {calibration_splits}")
    expected_calibration_counts = {
        str(row.series_id): int(row.calibration_count)
        for row in split_manifest.itertuples(index=False)
    }
    actual_calibration_counts = {
        str(series_id): int(len(group))
        for series_id, group in calibration.groupby(
            "series_id", sort=False, observed=True
        )
    }
    variable_calibration_counts_match = (
        actual_calibration_counts == expected_calibration_counts
    )
    if len(calibration) != EXPECTED_CALIBRATION_ROWS:
        raise AssertionError("Weather calibration row count changed")
    if calibration[feature_names].isna().any().any():
        raise AssertionError("calibration contains missing selected features")
    calibration_x = final_scaler.transform(
        calibration[feature_names].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    calibration_probability = np.clip(
        final_model.predict_proba(calibration_x)[:, 1].astype(np.float64, copy=False),
        1e-8,
        1.0 - 1e-8,
    )
    calibration_hard_target = calibration["hard_black_box_target"].to_numpy(
        dtype=np.int8
    )
    calibration_soft_target = stable_soft_target(
        calibration["loss_advantage_black_box"].to_numpy(dtype=np.float64),
        temperature,
    )
    calibration_auprc = float(
        average_precision_score(calibration_hard_target, calibration_probability)
    )
    calibration_brier = float(
        brier_score_loss(calibration_hard_target, calibration_probability)
    )
    calibration_soft_brier = float(
        np.mean((calibration_probability - calibration_soft_target) ** 2)
    )
    calibration_scores = calibration[
        [
            "dataset_id",
            "sample_id",
            "series_id",
            "series_type",
            "time_index",
            "split",
            "y_true",
            "ridge_prediction",
            "lightgbm_prediction",
            "hard_black_box_target",
            "seasonal_naive_mae_scale",
        ]
    ].copy()
    calibration_scores["black_box_probability"] = calibration_probability
    calibration_scores.to_parquet(
        CALIBRATION_SCORE_PATH, index=False, compression="snappy"
    )

    expected_fits = N_FOLDS * len(k_grid) * len(alpha_grid)
    expected_combinations = len(k_grid) * len(alpha_grid)
    finite_columns = [
        "constrained_scaled_loss",
        "simple_coverage",
        "soft_brier",
        "hard_brier",
        "black_box_AUPRC",
    ]
    checks_items: list[tuple[str, bool, str]] = [
        ("frozen_sample_id_is_consistent", sample_ids == {SAMPLE_ID}, SAMPLE_ID),
        ("upstream_router_checks_passed", upstream_checks_passed, f"feature={len(feature_checks)}; router={len(router_checks)}"),
        ("soft_router_solver_amendment_is_reused", selected_router.get("solver") == SOLVER and "scalable L1 logistic solver" in protocol_text, f"solver={SOLVER}"),
        ("local_weighting_tuning_contains_only_router_train", router_splits == {"router_train"}, str(router_splits)),
        ("router_row_count_matches_registered_feature_output", len(router) == EXPECTED_ROUTER_ROWS, f"rows={len(router)}"),
        ("variable_router_counts_match_each_series_manifest", variable_router_counts_match, f"series={len(actual_router_counts)}"),
        ("selected_router_has_registered_30_features", len(feature_names) == EXPECTED_SELECTED_FEATURES, f"features={len(feature_names)}"),
        ("calibration_excluded_from_selection_and_model_fit", True, "loaded only after selection and final fit"),
        ("test_not_read", "test" not in router_splits | calibration_splits, str(router_splits | calibration_splits)),
        ("folds_are_chronological", folds_chronological, str(folds_chronological)),
        ("fold_schedule_matches_registered_weather_snapshot", exact_fold_schedule_ok, f"train_rows={observed_available_train_rows}; validation=56"),
        ("saved_samples_match_soft_router_tuning", samples_match_soft_router_tuning, str(samples_match_soft_router_tuning)),
        ("every_fold_reuses_exactly_100000_saved_rows", len(saved_sample_counts) == N_FOLDS and set(saved_sample_counts.values()) == {EXPECTED_SAMPLE_ROWS}, str(saved_sample_counts)),
        ("exact_neighbor_search_used_for_tuning", bool(neighbor_manifest.loc[neighbor_manifest["scope"] == "rolling_validation_fold", "exact_query"].all()), f"algorithm={NEIGHBOR_ALGORITHM}"),
        ("neighbor_space_is_context_only", len(context_features) == 14, f"features={len(context_features)}"),
        ("registered_alpha_grid_used_without_postdata_change", alpha_grid == REGISTERED_ALPHA_GRID, str(alpha_grid)),
        ("expected_model_fit_count", len(details) == expected_fits == EXPECTED_ROLLING_FITS, f"actual={len(details)}; expected={expected_fits}"),
        ("expected_parameter_combination_count", len(summary) == expected_combinations == EXPECTED_LOCAL_COMBINATIONS, f"actual={len(summary)}; expected={expected_combinations}"),
        ("every_configuration_has_three_folds", bool((summary["folds"] == N_FOLDS).all()), f"combinations={len(summary)}"),
        ("alpha_zero_reproduces_soft_router_tuning", alpha_zero_max_difference <= 1e-12, f"maximum_loss_difference={alpha_zero_max_difference:.3e}"),
        ("every_validation_fit_uses_complete_28000_rows", set(details["validation_rows"].astype(int).unique()) == {EXPECTED_VALIDATION_ROWS_PER_FOLD}, str(sorted(details["validation_rows"].astype(int).unique()))),
        ("all_models_converged", all_models_converged and final_converged, f"rolling={all_models_converged}; final={final_converged}"),
        ("all_tuning_metrics_are_finite", bool(np.isfinite(details[finite_columns].to_numpy(dtype=float)).all()), f"records={len(details)}"),
        ("collapsed_inference_model_matches_penalized_intercept_training_model", collapse_max_difference <= 1e-6, f"max_difference={collapse_max_difference:.3e}"),
        ("final_sample_count_is_minimum_of_available_and_cap", len(final_sample_positions) == min(len(router), MAX_TRAINING_SAMPLE_ROWS), f"rows={len(final_sample_positions)}"),
        ("all_series_are_in_final_sample", all_series_in_final_sample, f"series={final_sample_positions['series_id'].nunique()}"),
        ("saved_final_router_bundle_reloads_identically", model_artifact_reload_ok, str(model_artifact_reload_ok)),
        ("calibration_has_expected_rows", len(calibration_scores) == EXPECTED_CALIBRATION_ROWS, f"rows={len(calibration_scores)}"),
        ("variable_calibration_counts_match_each_series_manifest", variable_calibration_counts_match, f"series={len(actual_calibration_counts)}"),
        ("calibration_metrics_are_finite", bool(np.isfinite([calibration_auprc, calibration_brier, calibration_soft_brier]).all()), f"AUPRC={calibration_auprc:.6f}; Brier={calibration_brier:.6f}"),
    ]
    checks = pd.DataFrame(checks_items, columns=["check", "passed", "detail"])
    failed = checks.loc[~checks["passed"]]
    if not failed.empty:
        message = "; ".join(
            f"{row.check}: {row.detail}" for row in failed.itertuples(index=False)
        )
        raise AssertionError(f"Weather Daily local weighting failed: {message}")

    selected_local = {
        "dataset_id": DATASET_ID,
        "sample_id": SAMPLE_ID,
        "selection_scope": "router_train_only",
        "selection_metric": "equal-series coverage-constrained scaled loss",
        "target_simple_coverage": target_simple_coverage,
        "rolling_folds": N_FOLDS,
        "solver": SOLVER,
        "solver_tolerance": SOLVER_TOLERANCE,
        "intercept_implementation": INTERCEPT_IMPLEMENTATION,
        "training_sample": {
            "maximum_rows_per_fold": MAX_TRAINING_SAMPLE_ROWS,
            "positions_source": str(ROUTER_SAMPLE_PATH),
            "identical_to_soft_router_tuning": True,
        },
        "neighbor_definition": {
            "space": NEIGHBOR_SPACE,
            "features": context_features,
            "algorithm": NEIGHBOR_ALGORITHM,
            "metric": "euclidean",
            "leaf_size": NEIGHBOR_LEAF_SIZE,
            "exact_query": True,
            "self_neighbor_excluded": True,
        },
        "parameter_grid": {
            "k": k_grid,
            "alpha": alpha_grid,
            "maximum_multiplier": maximum_multiplier,
        },
        "k": best_k,
        "alpha": best_alpha,
        "maximum_multiplier": maximum_multiplier,
        "local_weighting_selected": local_weighting_selected,
        "temperature": temperature,
        "residual_lag": residual_lag,
        "C": c_value,
        "feature_count": len(feature_names),
        "mean_validation_constrained_scaled_loss": float(best["mean_constrained_scaled_loss"]),
        "mean_validation_simple_coverage": float(best["mean_simple_coverage"]),
        "mean_validation_black_box_AUPRC": float(best["mean_black_box_AUPRC"]),
        "alpha_zero_soft_router_tuning_maximum_loss_difference": alpha_zero_max_difference,
        "final_model_training": {
            "scaler_fit_rows": len(router),
            "model_fit_sample_rows": len(final_sample_positions),
            "sample_positions": str(FINAL_SAMPLE_PATH),
            "saved_bundle_reload_passed": model_artifact_reload_ok,
            "collapsed_inference_max_probability_difference": collapse_max_difference,
        },
        "calibration_used_for_selection": False,
        "calibration_used_for_model_fit": False,
        "calibration_used_for_postfit_scoring": True,
        "test_accessed": False,
        "seed": seed,
    }
    SELECTED_LOCAL_PATH.write_text(
        yaml.safe_dump(selected_local, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    details.to_csv(DETAIL_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    neighbor_manifest.to_csv(NEIGHBOR_MANIFEST_PATH, index=False)
    checks.to_csv(CHECKS_PATH, index=False)

    nonzero_coefficients = coefficients.loc[coefficients["coefficient"] != 0.0]
    plot_coefficients = (
        nonzero_coefficients.head(15).sort_values("coefficient", kind="stable").copy()
    )
    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    if plot_coefficients.empty:
        axes[0].text(0.5, 0.5, "No non-zero coefficients", ha="center", va="center")
    else:
        colors = np.where(
            plot_coefficients["coefficient"] >= 0, "#E45756", "#54A24B"
        )
        axes[0].barh(
            plot_coefficients["feature"], plot_coefficients["coefficient"], color=colors
        )
    axes[0].axvline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Final Weather Daily soft-router coefficients")
    axes[0].set_xlabel("Standardized logistic coefficient")
    negative_scores = calibration_probability[calibration_hard_target == 0]
    positive_scores = calibration_probability[calibration_hard_target == 1]
    axes[1].hist(
        negative_scores, bins=40, alpha=0.65, density=True, color="#54A24B", label="Ridge wins"
    )
    axes[1].hist(
        positive_scores, bins=40, alpha=0.65, density=True, color="#E45756", label="LightGBM wins"
    )
    axes[1].set_title("Calibration routing probabilities")
    axes[1].set_xlabel("Predicted probability that LightGBM wins")
    axes[1].set_ylabel("Density")
    axes[1].legend(frameon=False)
    figure.suptitle(
        "Weather Daily final soft router: coefficients and calibration scores",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)

    elapsed = perf_counter() - overall_start
    nonzero_feature_count = int(np.count_nonzero(final_model.coef_))
    report = {
        "status": "passed",
        "dataset_id": DATASET_ID,
        "sample_id": SAMPLE_ID,
        "check_count": int(len(checks)),
        "failed_check_count": 0,
        "local_weighting_selection_scope": "router_train_only",
        "calibration_used_for_selection": False,
        "calibration_used_for_model_fit": False,
        "calibration_used_for_postfit_scoring": True,
        "test_accessed": False,
        "parameter_combinations": int(len(summary)),
        "model_fits": int(len(details)),
        "solver": SOLVER,
        "selected_k": best_k,
        "selected_alpha": best_alpha,
        "local_weighting_selected": local_weighting_selected,
        "selected_mean_constrained_scaled_loss": float(best["mean_constrained_scaled_loss"]),
        "final_nonzero_feature_coefficients": nonzero_feature_count,
        "calibration_rows": int(len(calibration_scores)),
        "calibration_AUPRC": calibration_auprc,
        "calibration_Brier": calibration_brier,
        "saved_model_bundle_reload_passed": model_artifact_reload_ok,
        "runtime_seconds": float(elapsed),
        "outputs": {
            "details": str(DETAIL_PATH),
            "summary": str(SUMMARY_PATH),
            "selected": str(SELECTED_LOCAL_PATH),
            "final_sample_positions": str(FINAL_SAMPLE_PATH),
            "neighbor_manifest": str(NEIGHBOR_MANIFEST_PATH),
            "model": str(MODEL_PATH),
            "coefficients": str(COEFFICIENT_PATH),
            "calibration_scores": str(CALIBRATION_SCORE_PATH),
            "checks": str(CHECKS_PATH),
            "figure": str(FIGURE_PATH),
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print()
    print("Weather Daily 局部加权与最终路由器训练全部通过")
    print("固定样本编号：", SAMPLE_ID)
    print("局部加权调参范围：仅 router_train")
    print("calibration 是否用于调参：否")
    print("calibration 是否用于模型训练：否")
    print("calibration 用途：仅最终模型训练后的评分")
    print("test 是否访问：否")
    print("近邻空间：14 个标准化上下文特征")
    print("近邻算法：精确 KDTree")
    print("L1 求解器：", SOLVER)
    print("每折训练样本上限：", MAX_TRAINING_SAMPLE_ROWS)
    print("局部加权参数组合数量：", len(summary))
    print("实际模型拟合次数：", len(details))
    print("最佳 k：", best_k)
    print("最佳 alpha：", best_alpha)
    print("是否选择局部加权：", "是" if local_weighting_selected else "否")
    print("平均约束缩放损失：", f"{float(best['mean_constrained_scaled_loss']):.6f}")
    print("平均简单模型覆盖率：", f"{float(best['mean_simple_coverage']):.6f}")
    print("平均 LightGBM AUPRC：", f"{float(best['mean_black_box_AUPRC']):.6f}")
    print("与软路由器调参阶段 alpha=0 核对最大差异：", f"{alpha_zero_max_difference:.3e}")
    print("最终训练抽样数量：", len(final_sample_positions))
    print("最终非零特征系数数量：", nonzero_feature_count)
    print("calibration 样本数量：", len(calibration_scores))
    print("calibration AUPRC：", f"{calibration_auprc:.6f}")
    print("calibration Brier：", f"{calibration_brier:.6f}")
    print("运行秒数：", f"{elapsed:.2f}")
    print("逐折结果：", DETAIL_PATH)
    print("汇总结果：", SUMMARY_PATH)
    print("选定参数：", SELECTED_LOCAL_PATH)
    print("最终模型：", MODEL_PATH)
    print("模型系数：", COEFFICIENT_PATH)
    print("校准概率：", CALIBRATION_SCORE_PATH)
    print("结果图片：", FIGURE_PATH)


if __name__ == "__main__":
    main()
