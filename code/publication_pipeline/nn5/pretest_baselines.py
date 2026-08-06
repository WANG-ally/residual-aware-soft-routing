"""在正式测试前训练路由基线与消融模型。"""

import os
from itertools import product
from pathlib import Path
from time import perf_counter
import hashlib
import json

os.environ.setdefault(
    "LOKY_MAX_CPU_COUNT",
    "1",
)

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
)
from sklearn.preprocessing import (
    StandardScaler,
)
import yaml


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]

FEATURE_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_router_features.parquet"
)

ROUTER_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_selected_soft_router_params.yaml"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "experiment_config.yaml"
)

FREEZE_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_pretest_freeze_manifest.json"
)

PROTOCOL_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_baseline_protocol_supplement.yaml"
)

DETAIL_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_baseline_rolling_validation.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_baseline_tuning_summary.csv"
)

SELECTED_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_selected_baseline_params.yaml"
)

CALIBRATION_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_baseline_calibration_scores.parquet"
)

FIGURE_PATH = (
    PROJECT_ROOT
    / "figures"
    / "nn5_baseline_validation.png"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

for path in [
    PROTOCOL_PATH,
    DETAIL_PATH,
    SUMMARY_PATH,
    SELECTED_PATH,
    CALIBRATION_PATH,
    FIGURE_PATH,
]:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Validate the frozen primary-method artifacts
# ============================================================

def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


with FREEZE_PATH.open(
    "r",
    encoding="utf-8",
) as handle:
    freeze_manifest = json.load(handle)

changed_frozen_files = []

for item in freeze_manifest["files"]:
    path = (
        PROJECT_ROOT
        / item["path"]
    )

    if (
        not path.is_file()
        or sha256_file(path)
        != item["sha256"]
    ):
        changed_frozen_files.append(
            item["path"]
        )

if changed_frozen_files:
    raise ValueError(
        "冻结文件发生变化，停止实验："
        f"{changed_frozen_files}"
    )


# ============================================================
# Load preregistered parameters
# ============================================================

with CONFIG_PATH.open(
    "r",
    encoding="utf-8",
) as handle:
    config = yaml.safe_load(handle)

with ROUTER_PATH.open(
    "r",
    encoding="utf-8",
) as handle:
    full_router_parameters = (
        yaml.safe_load(handle)
    )

seed = int(
    config["study"]["seed"]
)

target_coverage = float(
    config["study"][
        "primary_target_coverage"
    ]
)

sensitivity_coverages = [
    float(value)
    for value in config[
        "study"
    ]["sensitivity_target_coverages"]
]

original_c_grid = [
    float(value)
    for value in config[
        "soft_router"
    ]["c_grid"]
]

# Extend both ends of the baseline C grid under the registered supplement
# so the selected baseline is not forced to remain on a search boundary.
baseline_c_grid = sorted(
    set(
        [
            0.0001,
            0.001,
        ]
        + original_c_grid
        + [
            100.0,
            1000.0,
        ]
    )
)

temperature_grid = [
    float(value)
    for value in config[
        "soft_router"
    ]["temperature_grid"]
]

full_features = list(
    full_router_parameters[
        "feature_names"
    ]
)

context_features = [
    "last_value_scaled",
    "mean_7_scaled",
    "trend_7_scaled",
    "volatility_7_scaled",
    "trend_28_scaled",
    "volatility_28_scaled",
    "ridge_prediction_scaled",
    "lightgbm_prediction_scaled",
    "prediction_difference_scaled",
    "absolute_prediction_difference_scaled",
    "day_of_week_sin",
    "day_of_week_cos",
]

residual_features = [
    name
    for name in full_features
    if "residual_lag_" in name
]

if (
    len(context_features) != 12
    or len(residual_features) != 16
    or len(full_features) != 28
):
    raise ValueError(
        "上下文或残差特征数量异常"
    )

rf_grid = [
    {
        "n_estimators": 200,
        "max_depth": depth,
        "min_samples_leaf": leaf,
        "max_features": fraction,
    }
    for (
        depth,
        leaf,
        fraction,
    ) in product(
        [4, 8],
        [5, 20],
        ["sqrt", 0.8],
    )
]

aalf_lag_grid = [
    1,
    4,
    8,
    16,
]


# ============================================================
# Save the baseline protocol supplement
# ============================================================

protocol = {
    "dataset_id": "nn5_daily",
    "created_before_formal_test": True,
    "test_accessed": False,
    "selection_scope": (
        "router_train_only"
    ),
    "threshold_scope": (
        "calibration_only"
    ),
    "target_simple_coverage": (
        target_coverage
    ),
    "selection_metric": (
        "coverage_constrained_scaled_loss"
    ),
    "rolling_folds": 3,
    "validation_size_per_fold": 14,
    "definitions": {
        "hard_logistic_same_features": (
            "hard labels, full 28 features, "
            "unweighted L1 logistic"
        ),
        "hard_random_forest_same_features": (
            "hard labels, full 28 features, "
            "random forest"
        ),
        "class_weight_only": (
            "hard labels, full 28 features, "
            "balanced-class L1 logistic"
        ),
        "soft_targets_only": (
            "soft labels and 12 context features, "
            "without residual features"
        ),
        "residual_features_only": (
            "soft labels and 16 residual features, "
            "without context features"
        ),
        "hard_aalf_like_router": (
            "past residual squared-loss advantage "
            "with no fitted classifier"
        ),
    },
    "logistic_C_grid": (
        baseline_c_grid
    ),
    "soft_temperature_grid": (
        temperature_grid
    ),
    "random_forest_grid": rf_grid,
    "aalf_residual_lag_grid": (
        aalf_lag_grid
    ),
    "seed": seed,
}

with PROTOCOL_PATH.open(
    "w",
    encoding="utf-8",
) as handle:
    yaml.safe_dump(
        protocol,
        handle,
        sort_keys=False,
        allow_unicode=True,
    )


# ============================================================
# Load router_train and calibration only
# ============================================================

router = pd.read_parquet(
    FEATURE_PATH,
    filters=[
        (
            "split",
            "==",
            "router_train",
        )
    ],
)

calibration = pd.read_parquet(
    FEATURE_PATH,
    filters=[
        (
            "split",
            "==",
            "calibration",
        )
    ],
)

router = (
    router
    .sort_values(
        [
            "time_index",
            "series_id",
        ]
    )
    .reset_index(drop=True)
)

calibration = (
    calibration
    .sort_values(
        [
            "time_index",
            "series_id",
        ]
    )
    .reset_index(drop=True)
)

if (
    set(router["split"])
    != {"router_train"}
    or set(calibration["split"])
    != {"calibration"}
):
    raise ValueError(
        "数据段读取异常"
    )

if (
    len(router) != 11433
    or len(calibration) != 8769
):
    raise ValueError(
        "router_train或calibration行数异常"
    )

unique_times = np.sort(
    router["time_index"].unique()
)

if (
    len(unique_times) != 103
    or unique_times[0] != 490
    or unique_times[-1] != 592
):
    raise ValueError(
        "router_train时间范围异常"
    )


# ============================================================
# Define three expanding-window validation folds
# ============================================================

folds = []

first_validation_position = (
    len(unique_times)
    - 3 * 14
)

for fold in range(1, 4):
    start_position = (
        first_validation_position
        + (fold - 1) * 14
    )

    validation_times = unique_times[
        start_position:
        start_position + 14
    ]

    folds.append(
        (
            fold,
            int(validation_times[0]),
            int(validation_times[-1]),
        )
    )


# ============================================================
# Model helper functions
# ============================================================

def soft_target(
    loss_advantage,
    temperature,
):
    value = np.clip(
        loss_advantage
        / temperature,
        -40.0,
        40.0,
    )

    return 1.0 / (
        1.0 + np.exp(-value)
    )


def fit_soft_logistic(
    feature_matrix,
    loss_advantage,
    temperature,
    c_value,
):
    q_value = soft_target(
        loss_advantage,
        temperature,
    )

    duplicated_x = np.vstack(
        [
            feature_matrix,
            feature_matrix,
        ]
    )

    duplicated_y = np.concatenate(
        [
            np.ones(
                len(feature_matrix),
                dtype=np.int8,
            ),
            np.zeros(
                len(feature_matrix),
                dtype=np.int8,
            ),
        ]
    )

    duplicated_weight = np.concatenate(
        [
            q_value,
            1.0 - q_value,
        ]
    )

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
        sample_weight=(
            duplicated_weight
        ),
    )

    return model


def fit_hard_logistic(
    feature_matrix,
    hard_target,
    c_value,
    class_weight=None,
):
    model = LogisticRegression(
        solver="liblinear",
        l1_ratio=1.0,
        C=c_value,
        class_weight=class_weight,
        max_iter=3000,
        random_state=seed,
    )

    model.fit(
        feature_matrix,
        hard_target,
    )

    return model


def evaluate_probability(
    validation,
    probability,
):
    probability = np.clip(
        probability,
        1e-8,
        1.0 - 1e-8,
    )

    threshold = float(
        np.quantile(
            probability,
            target_coverage,
        )
    )

    use_black_box = (
        probability >= threshold
    )

    prediction = np.where(
        use_black_box,
        validation[
            "lightgbm_prediction"
        ].to_numpy(dtype=float),
        validation[
            "ridge_prediction"
        ].to_numpy(dtype=float),
    )

    scaled_loss = (
        (
            validation[
                "y_true"
            ].to_numpy(dtype=float)
            - prediction
        )
        / validation[
            "seasonal_naive_mae_scale"
        ].to_numpy(dtype=float)
    ) ** 2

    hard_target = validation[
        "hard_black_box_target"
    ].to_numpy(dtype=int)

    return {
        "constrained_scaled_loss": float(
            np.mean(scaled_loss)
        ),
        "simple_coverage": float(
            np.mean(
                ~use_black_box
            )
        ),
        "black_box_AUPRC": float(
            average_precision_score(
                hard_target,
                probability,
            )
        ),
        "hard_brier": float(
            brier_score_loss(
                hard_target,
                probability,
            )
        ),
    }


def aalf_score(
    frame,
    lag,
):
    ridge_columns = [
        f"ridge_residual_lag_{index}"
        for index in range(
            1,
            lag + 1,
        )
    ]

    lightgbm_columns = [
        f"lightgbm_residual_lag_{index}"
        for index in range(
            1,
            lag + 1,
        )
    ]

    ridge_past_loss = np.mean(
        frame[
            ridge_columns
        ].to_numpy(dtype=float) ** 2,
        axis=1,
    )

    lightgbm_past_loss = np.mean(
        frame[
            lightgbm_columns
        ].to_numpy(dtype=float) ** 2,
        axis=1,
    )

    # Positive values indicate a smaller historical LightGBM loss.
    return (
        ridge_past_loss
        - lightgbm_past_loss
    )


# ============================================================
# Select baseline parameters by expanding-window validation
# ============================================================

records = []
start_time = perf_counter()

for (
    fold,
    validation_start,
    validation_end,
) in folds:

    train = router[
        router["time_index"]
        < validation_start
    ].copy()

    validation = router[
        router["time_index"].between(
            validation_start,
            validation_end,
        )
    ].copy()

    if (
        train["time_index"].max()
        >= validation_start
    ):
        raise ValueError(
            "滚动切分存在时间泄漏"
        )

    hard_train = train[
        "hard_black_box_target"
    ].to_numpy(dtype=int)

    loss_advantage_train = train[
        "loss_advantage_black_box"
    ].to_numpy(dtype=float)

    feature_sets = {
        "hard_logistic_same_features": (
            full_features
        ),
        "class_weight_only": (
            full_features
        ),
        "soft_targets_only": (
            context_features
        ),
        "residual_features_only": (
            residual_features
        ),
        "hard_random_forest_same_features": (
            full_features
        ),
    }

    scaled_data = {}

    for (
        method,
        feature_names,
    ) in feature_sets.items():

        scaler = StandardScaler()

        train_x = scaler.fit_transform(
            train[
                feature_names
            ].to_numpy(dtype=float)
        )

        validation_x = scaler.transform(
            validation[
                feature_names
            ].to_numpy(dtype=float)
        )

        scaled_data[method] = (
            train_x,
            validation_x,
        )

    # Hard-label and class-weighted logistic routers
    for (
        method,
        class_weight,
    ) in [
        (
            "hard_logistic_same_features",
            None,
        ),
        (
            "class_weight_only",
            "balanced",
        ),
    ]:
        (
            train_x,
            validation_x,
        ) = scaled_data[method]

        for c_value in baseline_c_grid:
            model = fit_hard_logistic(
                train_x,
                hard_train,
                c_value,
                class_weight,
            )

            metrics = evaluate_probability(
                validation,
                model.predict_proba(
                    validation_x
                )[:, 1],
            )

            records.append(
                {
                    "method": method,
                    "fold": fold,
                    "C": c_value,
                    "temperature": np.nan,
                    "rf_index": np.nan,
                    "aalf_lag": np.nan,
                    **metrics,
                }
            )

    # Soft-target feature ablations
    for method in [
        "soft_targets_only",
        "residual_features_only",
    ]:
        (
            train_x,
            validation_x,
        ) = scaled_data[method]

        for (
            temperature,
            c_value,
        ) in product(
            temperature_grid,
            baseline_c_grid,
        ):
            model = fit_soft_logistic(
                train_x,
                loss_advantage_train,
                temperature,
                c_value,
            )

            metrics = evaluate_probability(
                validation,
                model.predict_proba(
                    validation_x
                )[:, 1],
            )

            records.append(
                {
                    "method": method,
                    "fold": fold,
                    "C": c_value,
                    "temperature": (
                        temperature
                    ),
                    "rf_index": np.nan,
                    "aalf_lag": np.nan,
                    **metrics,
                }
            )

    # Random-forest router
    (
        train_x,
        validation_x,
    ) = scaled_data[
        "hard_random_forest_same_features"
    ]

    for (
        rf_index,
        parameters,
    ) in enumerate(rf_grid):

        model = RandomForestClassifier(
            **parameters,
            class_weight=None,
            n_jobs=1,
            random_state=seed,
        )

        model.fit(
            train_x,
            hard_train,
        )

        metrics = evaluate_probability(
            validation,
            model.predict_proba(
                validation_x
            )[:, 1],
        )

        records.append(
            {
                "method": (
                    "hard_random_forest_same_features"
                ),
                "fold": fold,
                "C": np.nan,
                "temperature": np.nan,
                "rf_index": rf_index,
                "aalf_lag": np.nan,
                **metrics,
            }
        )

    # AALF-like historical-residual baseline
    for lag in aalf_lag_grid:
        score = aalf_score(
            validation,
            lag,
        )

        threshold = float(
            np.quantile(
                score,
                target_coverage,
            )
        )

        use_black_box = (
            score >= threshold
        )

        prediction = np.where(
            use_black_box,
            validation[
                "lightgbm_prediction"
            ].to_numpy(dtype=float),
            validation[
                "ridge_prediction"
            ].to_numpy(dtype=float),
        )

        scaled_loss = (
            (
                validation[
                    "y_true"
                ].to_numpy(dtype=float)
                - prediction
            )
            / validation[
                "seasonal_naive_mae_scale"
            ].to_numpy(dtype=float)
        ) ** 2

        hard_target = validation[
            "hard_black_box_target"
        ].to_numpy(dtype=int)

        metrics = {
            "constrained_scaled_loss": float(
                np.mean(scaled_loss)
            ),
            "simple_coverage": float(
                np.mean(
                    ~use_black_box
                )
            ),
            "black_box_AUPRC": float(
                average_precision_score(
                    hard_target,
                    score,
                )
            ),
            # The AALF-like score is not a calibrated probability,
            # so a Brier score is not reported.
            "hard_brier": np.nan,
        }

        records.append(
            {
                "method": (
                    "hard_aalf_like_router"
                ),
                "fold": fold,
                "C": np.nan,
                "temperature": np.nan,
                "rf_index": np.nan,
                "aalf_lag": lag,
                **metrics,
            }
        )

    print(
        f"[完成] 基线滚动验证 "
        f"fold={fold}/3"
    )


# ============================================================
# Aggregate results and select baseline parameters
# ============================================================

details = pd.DataFrame(records)

if len(details) != 324:
    raise ValueError(
        "滚动验证记录数量异常"
    )

details.to_csv(
    DETAIL_PATH,
    index=False,
)

parameter_columns = [
    "method",
    "C",
    "temperature",
    "rf_index",
    "aalf_lag",
]

summary = (
    details
    .groupby(
        parameter_columns,
        dropna=False,
        as_index=False,
    )
    .agg(
        mean_constrained_scaled_loss=(
            "constrained_scaled_loss",
            "mean",
        ),
        mean_simple_coverage=(
            "simple_coverage",
            "mean",
        ),
        mean_black_box_AUPRC=(
            "black_box_AUPRC",
            "mean",
        ),
        mean_hard_brier=(
            "hard_brier",
            "mean",
        ),
        folds=(
            "fold",
            "nunique",
        ),
    )
)

summary["selected"] = False
selected_rows = {}

for method, group in summary.groupby(
    "method"
):
    best_index = (
        group
        .sort_values(
            [
                "mean_constrained_scaled_loss",
                "mean_hard_brier",
                "C",
                "temperature",
                "rf_index",
                "aalf_lag",
            ],
            na_position="last",
        )
        .index[0]
    )

    summary.loc[
        best_index,
        "selected",
    ] = True

    selected_rows[method] = (
        summary
        .loc[best_index]
        .to_dict()
    )

summary = (
    summary
    .sort_values(
        [
            "method",
            "mean_constrained_scaled_loss",
        ]
    )
    .reset_index(drop=True)
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)


# ============================================================
# Fit the final baselines using all of router_train
# ============================================================

model_definitions = {
    "hard_logistic_same_features": (
        full_features,
        "hard",
    ),
    "class_weight_only": (
        full_features,
        "class_weight",
    ),
    "soft_targets_only": (
        context_features,
        "soft",
    ),
    "residual_features_only": (
        residual_features,
        "soft",
    ),
    "hard_random_forest_same_features": (
        full_features,
        "rf",
    ),
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

selected_parameters = {
    "dataset_id": "nn5_daily",
    "selection_scope": (
        "router_train_only"
    ),
    "threshold_scope": (
        "calibration_only"
    ),
    "test_accessed": False,
    "target_simple_coverage": (
        target_coverage
    ),
    "methods": {},
    "seed": seed,
}

for (
    method,
    (
        feature_names,
        model_kind,
    ),
) in model_definitions.items():

    choice = selected_rows[method]

    scaler = StandardScaler()

    train_x = scaler.fit_transform(
        router[
            feature_names
        ].to_numpy(dtype=float)
    )

    calibration_x = scaler.transform(
        calibration[
            feature_names
        ].to_numpy(dtype=float)
    )

    hard_train = router[
        "hard_black_box_target"
    ].to_numpy(dtype=int)

    if model_kind == "hard":
        model = fit_hard_logistic(
            train_x,
            hard_train,
            float(choice["C"]),
            None,
        )

    elif model_kind == "class_weight":
        model = fit_hard_logistic(
            train_x,
            hard_train,
            float(choice["C"]),
            "balanced",
        )

    elif model_kind == "soft":
        model = fit_soft_logistic(
            train_x,
            router[
                "loss_advantage_black_box"
            ].to_numpy(dtype=float),
            float(
                choice["temperature"]
            ),
            float(choice["C"]),
        )

    else:
        rf_index = int(
            choice["rf_index"]
        )

        model = RandomForestClassifier(
            **rf_grid[rf_index],
            class_weight=None,
            n_jobs=1,
            random_state=seed,
        )

        model.fit(
            train_x,
            hard_train,
        )

    calibration_probability = np.clip(
        model.predict_proba(
            calibration_x
        )[:, 1],
        1e-8,
        1.0 - 1e-8,
    )

    thresholds = {}

    for coverage in sorted(
        set(
            [target_coverage]
            + sensitivity_coverages
        )
    ):
        thresholds[
            str(coverage)
        ] = float(
            np.quantile(
                calibration_probability,
                coverage,
            )
        )

    hyperparameters = {
        "C": (
            None
            if pd.isna(choice["C"])
            else float(choice["C"])
        ),
        "temperature": (
            None
            if pd.isna(
                choice["temperature"]
            )
            else float(
                choice["temperature"]
            )
        ),
        "rf_index": (
            None
            if pd.isna(
                choice["rf_index"]
            )
            else int(
                choice["rf_index"]
            )
        ),
    }

    if model_kind == "rf":
        hyperparameters[
            "random_forest_parameters"
        ] = rf_grid[
            int(choice["rf_index"])
        ]

    model_path = (
        MODEL_DIR
        / f"nn5_{method}.joblib"
    )

    joblib.dump(
        {
            "dataset_id": "nn5_daily",
            "method": method,
            "model": model,
            "scaler": scaler,
            "feature_names": (
                feature_names
            ),
            "selection_scope": (
                "router_train_only"
            ),
            "threshold_scope": (
                "calibration_only"
            ),
            "thresholds": thresholds,
            "hyperparameters": (
                hyperparameters
            ),
            "seed": seed,
        },
        model_path,
    )

    calibration_outputs[
        f"{method}_probability"
    ] = calibration_probability

    selected_parameters[
        "methods"
    ][method] = {
        "C": hyperparameters["C"],
        "temperature": (
            hyperparameters[
                "temperature"
            ]
        ),
        "rf_index": (
            hyperparameters["rf_index"]
        ),
        "feature_count": len(
            feature_names
        ),
        "thresholds": thresholds,
        "model_file": str(
            model_path.relative_to(
                PROJECT_ROOT
            )
        ),
        "mean_validation_scaled_loss": float(
            choice[
                "mean_constrained_scaled_loss"
            ]
        ),
    }

    if model_kind == "rf":
        selected_parameters[
            "methods"
        ][method][
            "random_forest_parameters"
        ] = rf_grid[
            int(choice["rf_index"])
        ]


# ============================================================
# Save the frozen AALF-like parameters
# ============================================================

aalf_choice = selected_rows[
    "hard_aalf_like_router"
]

selected_aalf_lag = int(
    aalf_choice["aalf_lag"]
)

aalf_calibration_score = aalf_score(
    calibration,
    selected_aalf_lag,
)

aalf_thresholds = {
    str(coverage): float(
        np.quantile(
            aalf_calibration_score,
            coverage,
        )
    )
    for coverage in sorted(
        set(
            [target_coverage]
            + sensitivity_coverages
        )
    )
}

calibration_outputs[
    "hard_aalf_like_router_score"
] = aalf_calibration_score

selected_parameters[
    "methods"
][
    "hard_aalf_like_router"
] = {
    "residual_lag": (
        selected_aalf_lag
    ),
    "score_semantics": (
        "past_ridge_squared_loss_minus_"
        "past_lightgbm_squared_loss"
    ),
    "thresholds": aalf_thresholds,
    "mean_validation_scaled_loss": float(
        aalf_choice[
            "mean_constrained_scaled_loss"
        ]
    ),
}

calibration_outputs.to_parquet(
    CALIBRATION_PATH,
    index=False,
)

with SELECTED_PATH.open(
    "w",
    encoding="utf-8",
) as handle:
    yaml.safe_dump(
        selected_parameters,
        handle,
        sort_keys=False,
        allow_unicode=True,
    )


# ============================================================
# Plot the baseline comparison
# ============================================================

selected_plot = (
    summary[
        summary["selected"]
    ]
    .sort_values(
        "mean_constrained_scaled_loss"
    )
)

figure, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5),
)

axes[0].barh(
    selected_plot["method"],
    selected_plot[
        "mean_constrained_scaled_loss"
    ],
    color="#4c78a8",
)

axes[0].invert_yaxis()

axes[0].set_xlabel(
    "Rolling-validation scaled loss"
)

axes[0].set_title(
    "Selected baseline validation loss"
)

axes[1].barh(
    selected_plot["method"],
    selected_plot[
        "mean_black_box_AUPRC"
    ],
    color="#f28e2b",
)

axes[1].invert_yaxis()

axes[1].set_xlabel(
    "Black-box AUPRC"
)

axes[1].set_title(
    "Selected baseline routing accuracy"
)

figure.tight_layout()

figure.savefig(
    FIGURE_PATH,
    dpi=180,
    bbox_inches="tight",
)

plt.close(figure)


# ============================================================
# Report the baseline results
# ============================================================

elapsed_seconds = (
    perf_counter()
    - start_time
)

print()
print(
    "NN5 预测试基线与消融模型训练全部通过"
)
print("冻结文件完整性检查：通过")
print("训练与调参范围：仅 router_train")
print("阈值来源：仅 calibration")
print("测试集是否访问：否")
print(
    "滚动验证模型拟合/评价记录数：",
    len(details),
)
print(
    "完成基线数量：",
    len(
        selected_parameters[
            "methods"
        ]
    ),
)

for method in sorted(
    selected_parameters["methods"]
):
    value = selected_parameters[
        "methods"
    ][method][
        "mean_validation_scaled_loss"
    ]

    print(
        f"  {method}: "
        f"validation loss={value:.6f}"
    )

print(
    "运行秒数：",
    f"{elapsed_seconds:.2f}",
)
print("基线协议补充：", PROTOCOL_PATH)
print("逐折结果：", DETAIL_PATH)
print("调参汇总：", SUMMARY_PATH)
print("选定参数：", SELECTED_PATH)
print("校准概率：", CALIBRATION_PATH)
print("比较图片：", FIGURE_PATH)
