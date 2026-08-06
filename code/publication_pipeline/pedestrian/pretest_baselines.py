#!/usr/bin/env python3
"""Train Pedestrian pretest routing baselines and ablations.

All hyperparameters are selected with router_train rolling validation.  Final
baseline models use the same saved deterministic 100,000-row training sample as
the full router, and their decision thresholds are obtained from calibration.
The test split is never read."""

from __future__ import annotations

import gc
import hashlib
from itertools import product
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
import yaml


PROJECT_ROOT = Path(
    os.environ.get("SCI_ROUTING_ROOT", Path(__file__).resolve().parents[3])
).resolve()
OUTPUT_ROOT = Path(os.environ.get("SCI_ROUTING_OUTPUT_ROOT", PROJECT_ROOT)).resolve()

FEATURE_PATH = PROJECT_ROOT / "results/pedestrian_router_features.parquet"
FEATURE_MANIFEST_PATH = (
    PROJECT_ROOT / "results/pedestrian_router_feature_manifest.csv"
)
ROUTER_PARAMETER_PATH = (
    PROJECT_ROOT / "results/pedestrian_selected_soft_router_params.yaml"
)
FOLD_MANIFEST_PATH = (
    PROJECT_ROOT / "results/pedestrian_soft_router_fold_manifest.csv"
)
FOLD_SAMPLE_PATH = (
    PROJECT_ROOT / "results/pedestrian_soft_router_sample_positions.parquet"
)
FINAL_SAMPLE_PATH = (
    PROJECT_ROOT / "results/pedestrian_final_router_sample_positions.parquet"
)
CONFIG_PATH = PROJECT_ROOT / "experiment_config.yaml"
FREEZE_PATH = PROJECT_ROOT / "results/pedestrian_pretest_freeze_manifest.json"

PROTOCOL_PATH = OUTPUT_ROOT / "results/pedestrian_baseline_protocol_supplement.yaml"
DETAIL_PATH = OUTPUT_ROOT / "results/pedestrian_baseline_rolling_validation.csv"
SUMMARY_PATH = OUTPUT_ROOT / "results/pedestrian_baseline_tuning_summary.csv"
SELECTED_PATH = OUTPUT_ROOT / "results/pedestrian_selected_baseline_params.yaml"
CALIBRATION_PATH = (
    OUTPUT_ROOT / "results/pedestrian_baseline_calibration_scores.parquet"
)
CHECKS_PATH = OUTPUT_ROOT / "results/pedestrian_baseline_training_checks.csv"
FIGURE_PATH = OUTPUT_ROOT / "figures/pedestrian_baseline_validation.png"
REPORT_PATH = OUTPUT_ROOT / "logs/pedestrian_baseline_training_report.json"
MODEL_DIR = OUTPUT_ROOT / "models"

DATASET_ID = "pedestrian_hourly"
N_FOLDS = 3
TRAINING_SAMPLE_ROWS = 100_000
EXPECTED_ROUTER_ROWS = 468_805
EXPECTED_CALIBRATION_ROWS = 313_219
EXPECTED_FREEZE_ID = "9ad596968ebfe3d1f71776d4d40c637c6421941a56fefa5dfb8554eb68f3ee67"


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_soft_target(values: np.ndarray, temperature: float) -> np.ndarray:
    normalized = np.clip(
        np.asarray(values, dtype=np.float64) / float(temperature),
        -40.0,
        40.0,
    )
    return 1.0 / (1.0 + np.exp(-normalized))


def fit_soft_logistic(
    feature_matrix: np.ndarray,
    loss_advantage: np.ndarray,
    temperature: float,
    c_value: float,
    seed: int,
) -> LogisticRegression:
    soft_target = stable_soft_target(loss_advantage, temperature)
    duplicated_x = np.vstack([feature_matrix, feature_matrix])
    duplicated_y = np.concatenate(
        [
            np.ones(len(feature_matrix), dtype=np.int8),
            np.zeros(len(feature_matrix), dtype=np.int8),
        ]
    )
    duplicated_weight = np.concatenate([soft_target, 1.0 - soft_target])
    model = LogisticRegression(
        solver="liblinear",
        l1_ratio=1.0,
        C=c_value,
        max_iter=3000,
        random_state=seed,
    )
    model.fit(
        duplicated_x,
        duplicated_y,
        sample_weight=duplicated_weight,
    )
    return model


def fit_hard_logistic(
    feature_matrix: np.ndarray,
    hard_target: np.ndarray,
    c_value: float,
    seed: int,
    class_weight: str | None = None,
) -> LogisticRegression:
    model = LogisticRegression(
        solver="liblinear",
        l1_ratio=1.0,
        C=c_value,
        class_weight=class_weight,
        max_iter=3000,
        random_state=seed,
    )
    model.fit(feature_matrix, hard_target)
    return model


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


def evaluate_score(
    validation: pd.DataFrame,
    score: np.ndarray,
    validation_lengths: np.ndarray,
    target_coverage: float,
    score_is_probability: bool,
) -> dict[str, float]:
    score = np.asarray(score, dtype=np.float64)
    if score_is_probability:
        score = np.clip(score, 1e-8, 1.0 - 1e-8)
    threshold = float(np.quantile(score, target_coverage))
    use_black_box = score >= threshold
    selected_prediction = np.where(
        use_black_box,
        validation["lightgbm_prediction"].to_numpy(dtype=np.float64),
        validation["ridge_prediction"].to_numpy(dtype=np.float64),
    )
    scaled_error = (
        (
            validation["y_true"].to_numpy(dtype=np.float64)
            - selected_prediction
        )
        / validation["seasonal_naive_mae_scale"].to_numpy(dtype=np.float64)
    ) ** 2
    hard_target = validation["hard_black_box_target"].to_numpy(dtype=np.int8)
    result = {
        "constrained_scaled_loss": equal_series_mean(
            scaled_error, validation_lengths
        ),
        "pooled_constrained_scaled_loss": float(np.mean(scaled_error)),
        "simple_coverage": float(np.mean(~use_black_box)),
        "absolute_coverage_violation": float(
            abs(np.mean(~use_black_box) - target_coverage)
        ),
        "black_box_AUPRC": float(
            average_precision_score(hard_target, score)
        ),
        "hard_brier": np.nan,
    }
    if score_is_probability:
        result["hard_brier"] = float(brier_score_loss(hard_target, score))
    return result


def aalf_score(frame: pd.DataFrame, lag: int) -> np.ndarray:
    ridge_columns = [f"ridge_residual_lag_{index}" for index in range(1, lag + 1)]
    lgbm_columns = [
        f"lightgbm_residual_lag_{index}" for index in range(1, lag + 1)
    ]
    ridge_past_loss = np.mean(
        frame[ridge_columns].to_numpy(dtype=np.float64) ** 2, axis=1
    )
    lgbm_past_loss = np.mean(
        frame[lgbm_columns].to_numpy(dtype=np.float64) ** 2, axis=1
    )
    return ridge_past_loss - lgbm_past_loss


def indices_from_fold_samples(
    router: pd.DataFrame, saved: pd.DataFrame, fold: int
) -> np.ndarray:
    positions = saved.loc[
        saved["fold"] == fold,
        ["series_id", "time_index", "sample_order_within_fold"],
    ].copy()
    positions["series_id"] = positions["series_id"].astype(str)
    lookup = router[["series_id", "time_index"]].copy()
    lookup["series_id"] = lookup["series_id"].astype(str)
    lookup["dataframe_index"] = np.arange(len(lookup), dtype=np.int64)
    matched = positions.merge(
        lookup,
        on=["series_id", "time_index"],
        how="left",
        validate="one_to_one",
    ).sort_values("sample_order_within_fold", kind="stable")
    if matched["dataframe_index"].isna().any():
        raise AssertionError(f"Fold {fold} sample positions did not match router data")
    if not np.array_equal(
        matched["sample_order_within_fold"].to_numpy(dtype=np.int64),
        np.arange(len(matched), dtype=np.int64),
    ):
        raise AssertionError(f"Fold {fold} sample order is not contiguous")
    return matched["dataframe_index"].to_numpy(dtype=np.int64)


def indices_from_final_samples(
    router: pd.DataFrame, saved: pd.DataFrame
) -> np.ndarray:
    positions = saved[
        ["series_id", "time_index", "sample_order"]
    ].copy()
    positions["series_id"] = positions["series_id"].astype(str)
    lookup = router[["series_id", "time_index"]].copy()
    lookup["series_id"] = lookup["series_id"].astype(str)
    lookup["dataframe_index"] = np.arange(len(lookup), dtype=np.int64)
    matched = positions.merge(
        lookup,
        on=["series_id", "time_index"],
        how="left",
        validate="one_to_one",
    ).sort_values("sample_order", kind="stable")
    if matched["dataframe_index"].isna().any():
        raise AssertionError("Final sample positions did not match router data")
    if not np.array_equal(
        matched["sample_order"].to_numpy(dtype=np.int64),
        np.arange(len(matched), dtype=np.int64),
    ):
        raise AssertionError("Final sample order is not contiguous")
    return matched["dataframe_index"].to_numpy(dtype=np.int64)


def verify_first_freeze() -> tuple[dict, str]:
    with FREEZE_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    changed: list[str] = []
    current_records: list[dict[str, object]] = []
    for item in manifest["files"]:
        path = PROJECT_ROOT / item["path"]
        if not path.is_file():
            changed.append(item["path"])
            continue
        observed = sha256_file(path)
        observed_size = int(path.stat().st_size)
        if observed != item["sha256"] or observed_size != int(item["size_bytes"]):
            changed.append(item["path"])
        current_records.append(
            {
                "path": item["path"],
                "size_bytes": observed_size,
                "sha256": observed,
            }
        )
    if changed:
        raise AssertionError(f"Frozen files changed after the main-method freeze: {changed}")
    canonical = json.dumps(
        current_records, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    observed_freeze_id = hashlib.sha256(canonical).hexdigest()
    if observed_freeze_id != manifest["freeze_id"]:
        raise AssertionError("The main-method freeze identifier cannot be reproduced")
    if manifest["formal_test_runs_completed"] != 0:
        raise AssertionError("Formal test run count is not zero")
    if manifest.get("formal_test_authorized", True):
        raise AssertionError("First freeze unexpectedly authorized formal test")
    return manifest, observed_freeze_id


def main() -> None:
    required = [
        FEATURE_PATH,
        FEATURE_MANIFEST_PATH,
        ROUTER_PARAMETER_PATH,
        FOLD_MANIFEST_PATH,
        FOLD_SAMPLE_PATH,
        FINAL_SAMPLE_PATH,
        CONFIG_PATH,
        FREEZE_PATH,
    ]
    for path in required:
        require_file(path)
    for path in (
        PROTOCOL_PATH,
        DETAIL_PATH,
        SUMMARY_PATH,
        SELECTED_PATH,
        CALIBRATION_PATH,
        CHECKS_PATH,
        FIGURE_PATH,
        REPORT_PATH,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    protected_outputs = [
        SELECTED_PATH,
        CALIBRATION_PATH,
        MODEL_DIR / "pedestrian_hard_logistic_same_features.joblib",
        MODEL_DIR / "pedestrian_hard_random_forest_same_features.joblib",
        MODEL_DIR / "pedestrian_soft_targets_only.joblib",
        MODEL_DIR / "pedestrian_residual_features_only.joblib",
        MODEL_DIR / "pedestrian_class_weight_only.joblib",
    ]
    existing_protected = [str(path) for path in protected_outputs if path.exists()]
    if existing_protected:
        raise FileExistsError(
            "Baseline outputs already exist; do not retune after the first run: "
            f"{existing_protected}"
        )

    freeze_manifest, observed_freeze_id = verify_first_freeze()
    if observed_freeze_id != EXPECTED_FREEZE_ID:
        raise AssertionError(
            f"Unexpected first freeze id: {observed_freeze_id}"
        )

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    full_router_parameters = yaml.safe_load(
        ROUTER_PARAMETER_PATH.read_text(encoding="utf-8")
    )
    feature_manifest = pd.read_csv(FEATURE_MANIFEST_PATH)
    seed = int(config["study"]["seed"])
    target_coverage = float(config["study"]["primary_target_coverage"])
    allowed_coverage_violation = float(
        config["coverage_controller"]["allowed_absolute_violation"]
    )
    sensitivity_coverages = [
        float(value)
        for value in config["study"]["sensitivity_target_coverages"]
    ]
    original_c_grid = [
        float(value) for value in config["soft_router"]["c_grid"]
    ]
    baseline_c_grid = sorted(
        set([0.0001, 0.001] + original_c_grid + [100.0, 1000.0])
    )
    temperature_grid = [
        float(value)
        for value in config["soft_router"]["temperature_grid"]
    ]
    full_features = list(full_router_parameters["feature_names"])
    context_features = feature_manifest.loc[
        feature_manifest["group"] == "context", "feature"
    ].tolist()
    residual_features = feature_manifest.loc[
        feature_manifest["group"] == "past_residual", "feature"
    ].tolist()
    if (
        len(context_features) != 14
        or len(residual_features) != 32
        or len(full_features) != 46
        or full_features != context_features + residual_features
    ):
        raise AssertionError("Pedestrian baseline feature sets are invalid")

    rf_grid = [
        {
            "n_estimators": 200,
            "max_depth": depth,
            "min_samples_leaf": leaf,
            "max_features": fraction,
        }
        for depth, leaf, fraction in product(
            [4, 8], [5, 20], ["sqrt", 0.8]
        )
    ]
    aalf_lag_grid = [1, 4, 8, 16]
    expected_record_count = N_FOLDS * (
        2 * len(baseline_c_grid)
        + 2 * len(temperature_grid) * len(baseline_c_grid)
        + len(rf_grid)
        + len(aalf_lag_grid)
    )
    if expected_record_count != 324:
        raise AssertionError(f"Unexpected baseline record count: {expected_record_count}")

    protocol = {
        "dataset_id": DATASET_ID,
        "created_after_primary_freeze_before_formal_test": True,
        "first_freeze_id": observed_freeze_id,
        "formal_test_runs_completed_at_creation": 0,
        "test_accessed": False,
        "selection_scope": "router_train_only",
        "threshold_scope": "calibration_only",
        "target_simple_coverage": target_coverage,
        "selection_metric": "equal-series coverage-constrained scaled loss",
        "selection_constraint": (
            "every rolling fold must have absolute simple-coverage violation "
            f"at most {allowed_coverage_violation}"
        ),
        "rolling_folds": N_FOLDS,
        "fold_boundaries": "identical to pedestrian_soft_router_fold_manifest.csv",
        "training_sample": {
            "rows_per_fold": TRAINING_SAMPLE_ROWS,
            "positions": "identical to Pedestrian full-router tuning",
            "final_positions": "identical to Pedestrian final full-router fit",
            "standardizer_fit_scope": "all chronologically allowed router rows",
        },
        "definitions": {
            "hard_logistic_same_features": (
                "hard labels, all 46 features, unweighted L1 logistic"
            ),
            "hard_random_forest_same_features": (
                "hard labels, all 46 features, random forest"
            ),
            "class_weight_only": (
                "hard labels, all 46 features, balanced-class L1 logistic"
            ),
            "soft_targets_only": (
                "soft labels and 14 context features, without residual features"
            ),
            "residual_features_only": (
                "soft labels and 32 residual features, without context features"
            ),
            "hard_aalf_like_router": (
                "past residual squared-loss advantage without a fitted classifier"
            ),
        },
        "logistic_C_grid": baseline_c_grid,
        "soft_temperature_grid": temperature_grid,
        "random_forest_grid": rf_grid,
        "aalf_residual_lag_grid": aalf_lag_grid,
        "seed": seed,
    }
    PROTOCOL_PATH.write_text(
        yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    router = pd.read_parquet(
        FEATURE_PATH, filters=[("split", "==", "router_train")]
    )
    if not isinstance(router["series_id"].dtype, pd.CategoricalDtype):
        raise AssertionError("router_train series order is no longer categorical")
    series_order = [str(value) for value in router["series_id"].cat.categories]
    router["series_id"] = router["series_id"].astype(str)
    router["_series_order"] = pd.Categorical(
        router["series_id"], categories=series_order, ordered=True
    )
    router = (
        router.sort_values(["_series_order", "time_index"], kind="stable")
        .drop(columns="_series_order")
        .reset_index(drop=True)
    )
    observed_router_splits = set(router["split"].astype(str).unique())
    if len(router) != EXPECTED_ROUTER_ROWS or observed_router_splits != {"router_train"}:
        raise AssertionError(
            f"Invalid router scope: rows={len(router)}, splits={observed_router_splits}"
        )
    if router[full_features].isna().any().any():
        raise AssertionError("Router baseline features contain missing values")

    fold_manifest = pd.read_csv(FOLD_MANIFEST_PATH)
    fold_manifest["series_id"] = fold_manifest["series_id"].astype(str)
    fold_lookup = fold_manifest.set_index(["series_id", "fold"])
    fold_samples = pd.read_parquet(FOLD_SAMPLE_PATH)
    fold_samples["series_id"] = fold_samples["series_id"].astype(str)
    series_index_lookup = {
        series_id: group.index.to_numpy(dtype=np.int64)
        for series_id, group in router.groupby(
            "series_id", sort=False, observed=True
        )
    }
    if list(series_index_lookup) != series_order:
        raise AssertionError("Router series order changed from the frozen order")

    records: list[dict[str, object]] = []
    all_logistic_converged = True
    all_series_sampled = True
    fold_chronology_ok = True
    tuning_start = perf_counter()

    for fold in range(1, N_FOLDS + 1):
        train_indices_by_series: list[np.ndarray] = []
        validation_indices_by_series: list[np.ndarray] = []
        for series_id in series_order:
            indices = series_index_lookup[series_id]
            schedule = fold_lookup.loc[(series_id, fold)]
            train_count = int(schedule["train_count"])
            validation_count = int(schedule["validation_count"])
            train_part = indices[:train_count]
            validation_part = indices[
                train_count : train_count + validation_count
            ]
            fold_chronology_ok = fold_chronology_ok and bool(
                router.loc[train_part, "time_index"].max()
                < router.loc[validation_part, "time_index"].min()
            )
            train_indices_by_series.append(train_part)
            validation_indices_by_series.append(validation_part)
        full_train_indices = np.concatenate(train_indices_by_series)
        validation_indices = np.concatenate(validation_indices_by_series)
        validation_lengths = np.asarray(
            [len(values) for values in validation_indices_by_series],
            dtype=np.int64,
        )
        sampled_train_indices = indices_from_fold_samples(
            router, fold_samples, fold
        )
        if len(sampled_train_indices) != TRAINING_SAMPLE_ROWS:
            raise AssertionError(f"Fold {fold} baseline sample is not 100,000 rows")
        sampled_series = router.loc[sampled_train_indices, "series_id"].nunique()
        all_series_sampled = all_series_sampled and sampled_series == len(series_order)
        validation = router.loc[validation_indices].copy()
        hard_train = router.loc[
            sampled_train_indices, "hard_black_box_target"
        ].to_numpy(dtype=np.int8)
        advantage_train = router.loc[
            sampled_train_indices, "loss_advantage_black_box"
        ].to_numpy(dtype=np.float64)

        scaled_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for set_name, feature_names in (
            ("full", full_features),
            ("context", context_features),
            ("residual", residual_features),
        ):
            scaler = StandardScaler()
            scaler.fit(
                router.loc[full_train_indices, feature_names].to_numpy(
                    dtype=np.float64
                )
            )
            train_x = scaler.transform(
                router.loc[sampled_train_indices, feature_names].to_numpy(
                    dtype=np.float64
                )
            )
            validation_x = scaler.transform(
                validation[feature_names].to_numpy(dtype=np.float64)
            )
            scaled_data[set_name] = (train_x, validation_x)

        full_train_x, full_validation_x = scaled_data["full"]
        for method, class_weight in (
            ("hard_logistic_same_features", None),
            ("class_weight_only", "balanced"),
        ):
            for c_value in baseline_c_grid:
                model = fit_hard_logistic(
                    full_train_x,
                    hard_train,
                    c_value,
                    seed,
                    class_weight,
                )
                converged = bool(int(model.n_iter_[0]) < model.max_iter)
                all_logistic_converged = all_logistic_converged and converged
                metrics = evaluate_score(
                    validation,
                    model.predict_proba(full_validation_x)[:, 1],
                    validation_lengths,
                    target_coverage,
                    True,
                )
                records.append(
                    {
                        "dataset_id": DATASET_ID,
                        "method": method,
                        "fold": fold,
                        "C": c_value,
                        "temperature": np.nan,
                        "rf_index": np.nan,
                        "aalf_lag": np.nan,
                        "available_train_rows": len(full_train_indices),
                        "sampled_train_rows": len(sampled_train_indices),
                        "validation_rows": len(validation_indices),
                        "validation_size_min": int(validation_lengths.min()),
                        "validation_size_max": int(validation_lengths.max()),
                        "converged": converged,
                        **metrics,
                    }
                )
        print(f"[完成] fold={fold}/3，硬标签 Logistic", flush=True)

        for method, set_name in (
            ("soft_targets_only", "context"),
            ("residual_features_only", "residual"),
        ):
            train_x, validation_x = scaled_data[set_name]
            for temperature, c_value in product(
                temperature_grid, baseline_c_grid
            ):
                model = fit_soft_logistic(
                    train_x,
                    advantage_train,
                    temperature,
                    c_value,
                    seed,
                )
                converged = bool(int(model.n_iter_[0]) < model.max_iter)
                all_logistic_converged = all_logistic_converged and converged
                metrics = evaluate_score(
                    validation,
                    model.predict_proba(validation_x)[:, 1],
                    validation_lengths,
                    target_coverage,
                    True,
                )
                records.append(
                    {
                        "dataset_id": DATASET_ID,
                        "method": method,
                        "fold": fold,
                        "C": c_value,
                        "temperature": temperature,
                        "rf_index": np.nan,
                        "aalf_lag": np.nan,
                        "available_train_rows": len(full_train_indices),
                        "sampled_train_rows": len(sampled_train_indices),
                        "validation_rows": len(validation_indices),
                        "validation_size_min": int(validation_lengths.min()),
                        "validation_size_max": int(validation_lengths.max()),
                        "converged": converged,
                        **metrics,
                    }
                )
            print(f"[完成] fold={fold}/3，{method}", flush=True)

        for rf_index, parameters in enumerate(rf_grid):
            model = RandomForestClassifier(
                **parameters,
                class_weight=None,
                n_jobs=1,
                random_state=seed,
            )
            model.fit(full_train_x, hard_train)
            metrics = evaluate_score(
                validation,
                model.predict_proba(full_validation_x)[:, 1],
                validation_lengths,
                target_coverage,
                True,
            )
            records.append(
                {
                    "dataset_id": DATASET_ID,
                    "method": "hard_random_forest_same_features",
                    "fold": fold,
                    "C": np.nan,
                    "temperature": np.nan,
                    "rf_index": rf_index,
                    "aalf_lag": np.nan,
                    "available_train_rows": len(full_train_indices),
                    "sampled_train_rows": len(sampled_train_indices),
                    "validation_rows": len(validation_indices),
                    "validation_size_min": int(validation_lengths.min()),
                    "validation_size_max": int(validation_lengths.max()),
                    "converged": True,
                    **metrics,
                }
            )
            print(
                f"[完成] fold={fold}/3，随机森林配置="
                f"{rf_index + 1}/{len(rf_grid)}",
                flush=True,
            )
        print(f"[完成] fold={fold}/3，随机森林全部配置", flush=True)

        for lag in aalf_lag_grid:
            metrics = evaluate_score(
                validation,
                aalf_score(validation, lag),
                validation_lengths,
                target_coverage,
                False,
            )
            records.append(
                {
                    "dataset_id": DATASET_ID,
                    "method": "hard_aalf_like_router",
                    "fold": fold,
                    "C": np.nan,
                    "temperature": np.nan,
                    "rf_index": np.nan,
                    "aalf_lag": lag,
                    "available_train_rows": len(full_train_indices),
                    "sampled_train_rows": 0,
                    "validation_rows": len(validation_indices),
                    "validation_size_min": int(validation_lengths.min()),
                    "validation_size_max": int(validation_lengths.max()),
                    "converged": True,
                    **metrics,
                }
            )
        print(f"[完成] fold={fold}/3，AALF-like", flush=True)
        partial_path = DETAIL_PATH.with_name(
            "pedestrian_baseline_rolling_validation.partial.csv"
        )
        pd.DataFrame(records).to_csv(partial_path, index=False)
        del scaled_data, validation, hard_train, advantage_train
        gc.collect()

    details = pd.DataFrame(records)
    if len(details) != expected_record_count:
        raise AssertionError(
            f"Baseline validation records: {len(details)} != {expected_record_count}"
        )
    parameter_columns = ["method", "C", "temperature", "rf_index", "aalf_lag"]
    summary = (
        details.groupby(parameter_columns, dropna=False, as_index=False)
        .agg(
            mean_constrained_scaled_loss=("constrained_scaled_loss", "mean"),
            std_constrained_scaled_loss=("constrained_scaled_loss", "std"),
            mean_pooled_constrained_scaled_loss=(
                "pooled_constrained_scaled_loss",
                "mean",
            ),
            mean_simple_coverage=("simple_coverage", "mean"),
            mean_absolute_coverage_violation=(
                "absolute_coverage_violation",
                "mean",
            ),
            maximum_absolute_coverage_violation=(
                "absolute_coverage_violation",
                "max",
            ),
            mean_black_box_AUPRC=("black_box_AUPRC", "mean"),
            mean_hard_brier=("hard_brier", "mean"),
            folds=("fold", "nunique"),
        )
    )
    summary["selected"] = False
    summary["coverage_feasible"] = (
        summary["maximum_absolute_coverage_violation"]
        <= allowed_coverage_violation
    )
    selected_rows: dict[str, dict[str, object]] = {}
    for method, group in summary.groupby("method", sort=True):
        feasible_group = group.loc[group["coverage_feasible"]].copy()
        if feasible_group.empty:
            raise AssertionError(
                f"No coverage-feasible configuration for baseline {method}"
            )
        best_index = feasible_group.sort_values(
            [
                "mean_constrained_scaled_loss",
                "mean_hard_brier",
                "C",
                "temperature",
                "rf_index",
                "aalf_lag",
            ],
            na_position="last",
            kind="stable",
        ).index[0]
        summary.loc[best_index, "selected"] = True
        selected_rows[method] = summary.loc[best_index].to_dict()
    summary = summary.sort_values(
        ["method", "mean_constrained_scaled_loss"], kind="stable"
    ).reset_index(drop=True)
    if len(selected_rows) != 6:
        raise AssertionError(f"Expected six selected baselines: {selected_rows.keys()}")

    # Parameters are now selected.  Calibration is read only for thresholds.
    calibration = pd.read_parquet(
        FEATURE_PATH, filters=[("split", "==", "calibration")]
    )
    calibration["series_id"] = calibration["series_id"].astype(str)
    calibration["_series_order"] = pd.Categorical(
        calibration["series_id"], categories=series_order, ordered=True
    )
    calibration = (
        calibration.sort_values(["_series_order", "time_index"], kind="stable")
        .drop(columns="_series_order")
        .reset_index(drop=True)
    )
    observed_calibration_splits = set(
        calibration["split"].astype(str).unique()
    )
    if (
        len(calibration) != EXPECTED_CALIBRATION_ROWS
        or observed_calibration_splits != {"calibration"}
    ):
        raise AssertionError(
            f"Invalid calibration scope: rows={len(calibration)}, "
            f"splits={observed_calibration_splits}"
        )

    final_sample_saved = pd.read_parquet(FINAL_SAMPLE_PATH)
    final_sample_indices = indices_from_final_samples(router, final_sample_saved)
    if len(final_sample_indices) != TRAINING_SAMPLE_ROWS:
        raise AssertionError("Final baseline sample is not 100,000 rows")
    all_series_in_final_sample = (
        router.loc[final_sample_indices, "series_id"].nunique() == len(series_order)
    )
    final_hard_target = router.loc[
        final_sample_indices, "hard_black_box_target"
    ].to_numpy(dtype=np.int8)
    final_advantage = router.loc[
        final_sample_indices, "loss_advantage_black_box"
    ].to_numpy(dtype=np.float64)

    model_definitions = {
        "hard_logistic_same_features": (full_features, "hard"),
        "class_weight_only": (full_features, "class_weight"),
        "soft_targets_only": (context_features, "soft"),
        "residual_features_only": (residual_features, "soft"),
        "hard_random_forest_same_features": (full_features, "rf"),
    }
    calibration_outputs = calibration[
        [
            "dataset_id",
            "series_id",
            "time_index",
            "timestamp",
            "split",
            "y_true",
            "ridge_prediction",
            "lightgbm_prediction",
            "hard_black_box_target",
            "seasonal_naive_mae_scale",
        ]
    ].copy()
    selected_parameters: dict[str, object] = {
        "dataset_id": DATASET_ID,
        "first_freeze_id": observed_freeze_id,
        "selection_scope": "router_train_only",
        "threshold_scope": "calibration_only",
        "model_training_scope": "saved deterministic 100000-row router_train sample",
        "standardizer_fit_scope": "all router_train rows",
        "calibration_used_for_model_fit": False,
        "test_accessed": False,
        "target_simple_coverage": target_coverage,
        "methods": {},
        "seed": seed,
    }
    model_hashes: dict[str, str] = {}
    threshold_coverages_ok = True
    final_models_converged = True

    for method, (feature_names, model_kind) in model_definitions.items():
        choice = selected_rows[method]
        scaler = StandardScaler()
        scaler.fit(router[feature_names].to_numpy(dtype=np.float64))
        train_x = scaler.transform(
            router.loc[final_sample_indices, feature_names].to_numpy(
                dtype=np.float64
            )
        )
        calibration_x = scaler.transform(
            calibration[feature_names].to_numpy(dtype=np.float64)
        )
        if model_kind == "hard":
            model = fit_hard_logistic(
                train_x,
                final_hard_target,
                float(choice["C"]),
                seed,
                None,
            )
        elif model_kind == "class_weight":
            model = fit_hard_logistic(
                train_x,
                final_hard_target,
                float(choice["C"]),
                seed,
                "balanced",
            )
        elif model_kind == "soft":
            model = fit_soft_logistic(
                train_x,
                final_advantage,
                float(choice["temperature"]),
                float(choice["C"]),
                seed,
            )
        else:
            rf_index = int(choice["rf_index"])
            model = RandomForestClassifier(
                **rf_grid[rf_index],
                class_weight=None,
                n_jobs=1,
                random_state=seed,
            )
            model.fit(train_x, final_hard_target)
        if hasattr(model, "n_iter_"):
            converged = bool(int(model.n_iter_[0]) < model.max_iter)
            final_models_converged = final_models_converged and converged
        calibration_probability = np.clip(
            model.predict_proba(calibration_x)[:, 1], 1e-8, 1.0 - 1e-8
        )
        thresholds: dict[str, float] = {}
        achieved_coverages: dict[str, float] = {}
        for coverage in sorted(set([target_coverage] + sensitivity_coverages)):
            threshold = float(np.quantile(calibration_probability, coverage))
            achieved = float(np.mean(calibration_probability < threshold))
            thresholds[str(coverage)] = threshold
            achieved_coverages[str(coverage)] = achieved
            threshold_coverages_ok = threshold_coverages_ok and bool(
                abs(achieved - coverage) <= 1.0 / len(calibration)
            )
        hyperparameters = {
            "C": None if pd.isna(choice["C"]) else float(choice["C"]),
            "temperature": (
                None
                if pd.isna(choice["temperature"])
                else float(choice["temperature"])
            ),
            "rf_index": (
                None
                if pd.isna(choice["rf_index"])
                else int(choice["rf_index"])
            ),
        }
        if model_kind == "rf":
            hyperparameters["random_forest_parameters"] = rf_grid[
                int(choice["rf_index"])
            ]
        model_path = MODEL_DIR / f"pedestrian_{method}.joblib"
        bundle = {
            "dataset_id": DATASET_ID,
            "method": method,
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "selection_scope": "router_train_only",
            "model_training_scope": "saved_deterministic_100000_row_sample",
            "standardizer_fit_scope": "all_router_train_rows",
            "threshold_scope": "calibration_only",
            "thresholds": thresholds,
            "hyperparameters": hyperparameters,
            "calibration_used_for_model_fit": False,
            "test_accessed": False,
            "seed": seed,
        }
        joblib.dump(bundle, model_path)
        model_hash = sha256_file(model_path)
        model_hashes[method] = model_hash
        calibration_outputs[f"{method}_probability"] = calibration_probability
        selected_parameters["methods"][method] = {
            "C": hyperparameters["C"],
            "temperature": hyperparameters["temperature"],
            "rf_index": hyperparameters["rf_index"],
            "feature_count": len(feature_names),
            "thresholds": thresholds,
            "achieved_calibration_coverages": achieved_coverages,
            "model_file": str(model_path.relative_to(OUTPUT_ROOT)),
            "model_sha256": model_hash,
            "mean_validation_scaled_loss": float(
                choice["mean_constrained_scaled_loss"]
            ),
            "mean_validation_AUPRC": float(choice["mean_black_box_AUPRC"]),
            "maximum_validation_coverage_violation": float(
                choice["maximum_absolute_coverage_violation"]
            ),
        }
        if model_kind == "rf":
            selected_parameters["methods"][method][
                "random_forest_parameters"
            ] = rf_grid[int(choice["rf_index"])]
        print(f"[完成] 最终模型与 calibration 分数：{method}", flush=True)
        del train_x, calibration_x
        gc.collect()

    aalf_choice = selected_rows["hard_aalf_like_router"]
    selected_aalf_lag = int(aalf_choice["aalf_lag"])
    aalf_calibration_score = aalf_score(calibration, selected_aalf_lag)
    aalf_thresholds: dict[str, float] = {}
    aalf_achieved: dict[str, float] = {}
    for coverage in sorted(set([target_coverage] + sensitivity_coverages)):
        threshold = float(np.quantile(aalf_calibration_score, coverage))
        achieved = float(np.mean(aalf_calibration_score < threshold))
        aalf_thresholds[str(coverage)] = threshold
        aalf_achieved[str(coverage)] = achieved
        threshold_coverages_ok = threshold_coverages_ok and bool(
            abs(achieved - coverage) <= 1.0 / len(calibration)
        )
    calibration_outputs["hard_aalf_like_router_score"] = aalf_calibration_score
    selected_parameters["methods"]["hard_aalf_like_router"] = {
        "residual_lag": selected_aalf_lag,
        "score_semantics": (
            "past_ridge_squared_loss_minus_past_lightgbm_squared_loss"
        ),
        "thresholds": aalf_thresholds,
        "achieved_calibration_coverages": aalf_achieved,
        "mean_validation_scaled_loss": float(
            aalf_choice["mean_constrained_scaled_loss"]
        ),
        "mean_validation_AUPRC": float(aalf_choice["mean_black_box_AUPRC"]),
        "maximum_validation_coverage_violation": float(
            aalf_choice["maximum_absolute_coverage_violation"]
        ),
    }

    finite_metric_columns = [
        "constrained_scaled_loss",
        "pooled_constrained_scaled_loss",
        "simple_coverage",
        "absolute_coverage_violation",
        "black_box_AUPRC",
    ]
    expected_methods = {
        "hard_logistic_same_features",
        "class_weight_only",
        "soft_targets_only",
        "residual_features_only",
        "hard_random_forest_same_features",
        "hard_aalf_like_router",
    }
    calibration_output_columns_valid = expected_methods - {
        "hard_aalf_like_router"
    } == {
        name.removesuffix("_probability")
        for name in calibration_outputs.columns
        if name.endswith("_probability")
    }
    formal_outputs = [
        PROJECT_ROOT / "results/pedestrian_test_predictions.parquet",
        PROJECT_ROOT / "results/pedestrian_test_aggregate_metrics.csv",
        PROJECT_ROOT / "logs/pedestrian_formal_test_access_receipt.json",
    ]
    formal_test_absent = not any(path.exists() for path in formal_outputs)
    check_items: list[tuple[str, bool, str]] = [
        (
            "first_freeze_integrity_passed",
            observed_freeze_id == freeze_manifest["freeze_id"],
            observed_freeze_id,
        ),
        (
            "formal_test_runs_remain_zero",
            freeze_manifest["formal_test_runs_completed"] == 0
            and formal_test_absent,
            "runs=0; outputs_absent=" + str(formal_test_absent),
        ),
        (
            "tuning_input_contains_only_router_train",
            observed_router_splits == {"router_train"},
            str(observed_router_splits),
        ),
        (
            "calibration_loaded_only_after_parameter_selection",
            observed_calibration_splits == {"calibration"},
            str(observed_calibration_splits),
        ),
        (
            "test_not_accessed",
            "test" not in observed_router_splits | observed_calibration_splits,
            str(observed_router_splits | observed_calibration_splits),
        ),
        (
            "rolling_folds_are_chronological",
            fold_chronology_ok,
            str(fold_chronology_ok),
        ),
        (
            "all_series_are_in_each_tuning_sample",
            all_series_sampled,
            str(all_series_sampled),
        ),
        (
            "all_series_are_in_final_sample",
            all_series_in_final_sample,
            str(all_series_in_final_sample),
        ),
        (
            "expected_rolling_validation_records",
            len(details) == expected_record_count,
            f"actual={len(details)}; expected={expected_record_count}",
        ),
        (
            "six_baselines_selected",
            set(selected_parameters["methods"]) == expected_methods,
            str(sorted(selected_parameters["methods"])),
        ),
        (
            "all_selected_configurations_are_coverage_feasible",
            all(
                bool(value["coverage_feasible"])
                for value in selected_rows.values()
            ),
            (
                "maximum allowed violation="
                f"{allowed_coverage_violation}"
            ),
        ),
        (
            "every_configuration_has_three_folds",
            bool((summary["folds"] == N_FOLDS).all()),
            f"configurations={len(summary)}",
        ),
        (
            "all_logistic_models_converged",
            all_logistic_converged and final_models_converged,
            f"rolling={all_logistic_converged}; final={final_models_converged}",
        ),
        (
            "all_validation_metrics_are_finite",
            bool(
                np.isfinite(
                    details[finite_metric_columns].to_numpy(dtype=float)
                ).all()
            ),
            f"records={len(details)}",
        ),
        (
            "calibration_threshold_coverages_match_targets",
            threshold_coverages_ok,
            str(threshold_coverages_ok),
        ),
        (
            "calibration_output_has_expected_rows",
            len(calibration_outputs) == EXPECTED_CALIBRATION_ROWS,
            f"rows={len(calibration_outputs)}",
        ),
        (
            "calibration_output_has_all_baseline_scores",
            calibration_output_columns_valid
            and "hard_aalf_like_router_score" in calibration_outputs,
            str(sorted(calibration_outputs.columns)),
        ),
        (
            "five_model_files_were_saved",
            len(model_hashes) == 5
            and all(
                (MODEL_DIR / f"pedestrian_{method}.joblib").is_file()
                for method in model_hashes
            ),
            str(sorted(model_hashes)),
        ),
    ]
    checks = pd.DataFrame(check_items, columns=["check", "passed", "detail"])
    failed = checks.loc[~checks["passed"]]
    if not failed.empty:
        message = "; ".join(
            f"{row.check}: {row.detail}" for row in failed.itertuples(index=False)
        )
        raise AssertionError(f"Pedestrian baseline training failed: {message}")

    details.to_csv(DETAIL_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)
    SELECTED_PATH.write_text(
        yaml.safe_dump(selected_parameters, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    calibration_outputs["dataset_id"] = calibration_outputs["dataset_id"].astype(
        "category"
    )
    calibration_outputs["series_id"] = calibration_outputs["series_id"].astype(
        "category"
    )
    calibration_outputs.to_parquet(
        CALIBRATION_PATH, index=False, compression="snappy"
    )
    checks.to_csv(CHECKS_PATH, index=False)
    partial_path = DETAIL_PATH.with_name(
        "pedestrian_baseline_rolling_validation.partial.csv"
    )
    if partial_path.exists():
        partial_path.unlink()

    selected_plot = summary.loc[summary["selected"]].sort_values(
        "mean_constrained_scaled_loss", kind="stable"
    )
    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].barh(
        selected_plot["method"],
        selected_plot["mean_constrained_scaled_loss"],
        color="#4C78A8",
    )
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Equal-series rolling-validation scaled loss")
    axes[0].set_title("Selected baseline validation loss")
    axes[1].barh(
        selected_plot["method"],
        selected_plot["mean_black_box_AUPRC"],
        color="#F28E2B",
    )
    axes[1].invert_yaxis()
    axes[1].set_xlabel("LightGBM-win AUPRC")
    axes[1].set_title("Selected baseline routing accuracy")
    figure.suptitle(
        "Pedestrian pretest baseline and ablation selection", fontsize=14
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)

    elapsed = perf_counter() - tuning_start
    report = {
        "status": "passed",
        "dataset_id": DATASET_ID,
        "first_freeze_id": observed_freeze_id,
        "first_freeze_integrity": True,
        "formal_test_runs_completed": 0,
        "selection_scope": "router_train_only",
        "threshold_scope": "calibration_only",
        "test_accessed": False,
        "rolling_validation_records": len(details),
        "selected_baselines": len(selected_parameters["methods"]),
        "check_count": len(checks),
        "failed_check_count": 0,
        "runtime_seconds": float(elapsed),
        "model_hashes": model_hashes,
        "outputs": {
            "protocol": str(PROTOCOL_PATH),
            "details": str(DETAIL_PATH),
            "summary": str(SUMMARY_PATH),
            "selected": str(SELECTED_PATH),
            "calibration": str(CALIBRATION_PATH),
            "checks": str(CHECKS_PATH),
            "figure": str(FIGURE_PATH),
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("Pedestrian 预测试基线与消融模型训练全部通过")
    print("第一次冻结完整性检查：通过")
    print("第一次冻结编号：", observed_freeze_id)
    print("训练与调参范围：仅 router_train")
    print("每折训练抽样数量：", TRAINING_SAMPLE_ROWS)
    print("阈值来源：仅 calibration")
    print("test 是否访问：否")
    print("滚动验证模型拟合/评价记录数：", len(details))
    print("完成基线数量：", len(selected_parameters["methods"]))
    for method in sorted(selected_parameters["methods"]):
        value = selected_parameters["methods"][method][
            "mean_validation_scaled_loss"
        ]
        print(f"  {method}: validation loss={value:.6f}")
    print("运行秒数：", f"{elapsed:.2f}")
    print("基线协议补充：", PROTOCOL_PATH)
    print("逐折结果：", DETAIL_PATH)
    print("调参汇总：", SUMMARY_PATH)
    print("选定参数：", SELECTED_PATH)
    print("校准分数：", CALIBRATION_PATH)
    print("检查表：", CHECKS_PATH)
    print("比较图片：", FIGURE_PATH)


if __name__ == "__main__":
    main()
