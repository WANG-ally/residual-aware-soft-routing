"""仅在 router_train 内滚动选择软路由参数。"""

from itertools import product
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
)
from sklearn.preprocessing import StandardScaler
import yaml


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURE_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_router_features.parquet"
)
MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_router_feature_manifest.csv"
)
CONFIG_PATH = (
    PROJECT_ROOT
    / "experiment_config.yaml"
)

DETAIL_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_soft_router_rolling_validation.csv"
)
SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_soft_router_tuning_summary.csv"
)
SELECTED_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_selected_soft_router_params.yaml"
)
FIGURE_PATH = (
    PROJECT_ROOT
    / "figures"
    / "nn5_soft_router_tuning.png"
)

for path in [
    DETAIL_PATH,
    SUMMARY_PATH,
    SELECTED_PATH,
    FIGURE_PATH,
]:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


# Load the registered hyperparameter grids
with CONFIG_PATH.open(
    "r",
    encoding="utf-8",
) as handle:
    config = yaml.safe_load(handle)

seed = int(
    config["study"]["seed"]
)

target_simple_coverage = float(
    config[
        "study"
    ]["primary_target_coverage"]
)

temperatures = [
    float(item)
    for item in config[
        "soft_router"
    ]["temperature_grid"]
]

residual_lags = [
    int(item)
    for item in config[
        "soft_router"
    ]["residual_lag_grid"]
]

c_grid = [
    float(item)
    for item in config[
        "soft_router"
    ]["c_grid"]
]


# Load router_train only
router = pd.read_parquet(
    FEATURE_PATH,
    filters=[
        ("split", "==", "router_train")
    ],
)

if (
    router.empty
    or set(router["split"])
    != {"router_train"}
):
    raise ValueError(
        "软路由调参只能读取 router_train"
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


# Load the registered feature set
manifest = pd.read_csv(
    MANIFEST_PATH
)

if not manifest[
    "available_before_target"
].all():
    raise ValueError(
        "存在目标时点后才可用的特征"
    )

context_features = manifest.loc[
    manifest["group"] == "context",
    "feature",
].tolist()

residual_features = manifest.loc[
    manifest["group"] == "past_residual",
    "feature",
].tolist()


# Define three expanding-window validation folds
unique_times = np.sort(
    router["time_index"].unique()
)

if (
    len(unique_times) != 103
    or unique_times[0] != 490
    or unique_times[-1] != 592
):
    raise ValueError(
        "router_train 时间范围异常"
    )

N_FOLDS = 3
VALIDATION_SIZE = 14

first_validation_position = (
    len(unique_times)
    - N_FOLDS * VALIDATION_SIZE
)

fold_boundaries = []

for fold in range(
    1,
    N_FOLDS + 1,
):
    start_position = (
        first_validation_position
        + (fold - 1) * VALIDATION_SIZE
    )

    validation_times = unique_times[
        start_position:
        start_position + VALIDATION_SIZE
    ]

    fold_boundaries.append(
        (
            fold,
            int(validation_times[0]),
            int(validation_times[-1]),
        )
    )


# Convert loss differences into soft targets
def stable_soft_target(
    loss_advantage,
    temperature,
):
    normalized = np.clip(
        loss_advantage / temperature,
        -40.0,
        40.0,
    )

    return 1.0 / (
        1.0 + np.exp(-normalized)
    )


# Select features for each maximum residual lag
def features_for_lag(max_lag):
    selected = list(
        context_features
    )

    for name in residual_features:
        lag = int(
            name.rsplit("_", 1)[1]
        )

        if lag <= max_lag:
            selected.append(name)

    expected_count = (
        len(context_features)
        + 2 * max_lag
    )

    if len(selected) != expected_count:
        raise ValueError(
            "残差特征数量与滞后不一致"
        )

    return selected


# Fit the 240 registered fold-configuration combinations
records = []
overall_start = perf_counter()

for (
    fold,
    validation_start,
    validation_end,
) in fold_boundaries:

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
        train.empty
        or validation.empty
        or train["time_index"].max()
        >= validation_start
    ):
        raise ValueError(
            "路由滚动切分存在时间交叉"
        )

    for max_lag in residual_lags:
        feature_names = features_for_lag(
            max_lag
        )

        # Fit the standardizer using only the training portion of each fold.
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

        for (
            temperature,
            c_value,
        ) in product(
            temperatures,
            c_grid,
        ):
            train_soft_target = (
                stable_soft_target(
                    train[
                        "loss_advantage_black_box"
                    ].to_numpy(dtype=float),
                    temperature,
                )
            )

            validation_soft_target = (
                stable_soft_target(
                    validation[
                        "loss_advantage_black_box"
                    ].to_numpy(dtype=float),
                    temperature,
                )
            )

            # Duplicate each observation into class-one and class-zero copies
            # to implement weighted soft-label logistic training.
            duplicated_x = np.vstack(
                [
                    train_x,
                    train_x,
                ]
            )

            duplicated_y = np.concatenate(
                [
                    np.ones(
                        len(train_x),
                        dtype=np.int8,
                    ),
                    np.zeros(
                        len(train_x),
                        dtype=np.int8,
                    ),
                ]
            )

            duplicated_weight = (
                np.concatenate(
                    [
                        train_soft_target,
                        1.0
                        - train_soft_target,
                    ]
                )
            )

            # l1_ratio=1 applies pure L1 regularization.
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
                sample_weight=(
                    duplicated_weight
                ),
            )

            probability = np.clip(
                model.predict_proba(
                    validation_x
                )[:, 1],
                1e-8,
                1.0 - 1e-8,
            )

            # Higher probabilities favor LightGBM.
            # The 70th percentile routes approximately 70% of cases to Ridge.
            threshold = float(
                np.quantile(
                    probability,
                    target_simple_coverage,
                )
            )

            use_black_box = (
                probability >= threshold
            )

            selected_prediction = np.where(
                use_black_box,
                validation[
                    "lightgbm_prediction"
                ].to_numpy(dtype=float),
                validation[
                    "ridge_prediction"
                ].to_numpy(dtype=float),
            )

            scaled_error = (
                (
                    validation[
                        "y_true"
                    ].to_numpy(dtype=float)
                    - selected_prediction
                )
                / validation[
                    "seasonal_naive_mae_scale"
                ].to_numpy(dtype=float)
            ) ** 2

            hard_target = validation[
                "hard_black_box_target"
            ].to_numpy(dtype=int)

            soft_log_loss = float(
                np.mean(
                    -validation_soft_target
                    * np.log(probability)
                    - (
                        1.0
                        - validation_soft_target
                    )
                    * np.log(
                        1.0 - probability
                    )
                )
            )

            records.append(
                {
                    "dataset_id": "nn5_daily",
                    "fold": fold,
                    "train_start": int(
                        train[
                            "time_index"
                        ].min()
                    ),
                    "train_end": int(
                        train[
                            "time_index"
                        ].max()
                    ),
                    "validation_start": (
                        validation_start
                    ),
                    "validation_end": (
                        validation_end
                    ),
                    "temperature": (
                        temperature
                    ),
                    "residual_lag": (
                        max_lag
                    ),
                    "C": c_value,
                    "feature_count": (
                        len(feature_names)
                    ),
                    "train_rows": (
                        len(train)
                    ),
                    "validation_rows": (
                        len(validation)
                    ),
                    "constrained_scaled_loss": float(
                        np.mean(
                            scaled_error
                        )
                    ),
                    "simple_coverage": float(
                        np.mean(
                            ~use_black_box
                        )
                    ),
                    "soft_log_loss": (
                        soft_log_loss
                    ),
                    "soft_brier": float(
                        np.mean(
                            (
                                probability
                                - validation_soft_target
                            ) ** 2
                        )
                    ),
                    "hard_brier": float(
                        brier_score_loss(
                            hard_target,
                            probability,
                        )
                    ),
                    "black_box_AUPRC": float(
                        average_precision_score(
                            hard_target,
                            probability,
                        )
                    ),
                    "nonzero_coefficients": int(
                        np.count_nonzero(
                            model.coef_
                        )
                    ),
                    "converged": bool(
                        int(
                            model.n_iter_[0]
                        )
                        < model.max_iter
                    ),
                    "seed": seed,
                }
            )

        print(
            f"[完成] fold={fold}/3, "
            f"residual_lag={max_lag}"
        )


# Save fold-level results
details = pd.DataFrame(records)

if not details["converged"].all():
    raise ValueError(
        "至少一个 Logistic 模型没有收敛"
    )

details.to_csv(
    DETAIL_PATH,
    index=False,
)


# Aggregate the 80 registered configurations
summary = (
    details
    .groupby(
        [
            "temperature",
            "residual_lag",
            "C",
        ],
        as_index=False,
    )
    .agg(
        mean_constrained_scaled_loss=(
            "constrained_scaled_loss",
            "mean",
        ),
        std_constrained_scaled_loss=(
            "constrained_scaled_loss",
            "std",
        ),
        mean_simple_coverage=(
            "simple_coverage",
            "mean",
        ),
        mean_soft_log_loss=(
            "soft_log_loss",
            "mean",
        ),
        mean_soft_brier=(
            "soft_brier",
            "mean",
        ),
        mean_hard_brier=(
            "hard_brier",
            "mean",
        ),
        mean_black_box_AUPRC=(
            "black_box_AUPRC",
            "mean",
        ),
        mean_nonzero_coefficients=(
            "nonzero_coefficients",
            "mean",
        ),
        folds=(
            "fold",
            "nunique",
        ),
    )
    .sort_values(
        [
            "mean_constrained_scaled_loss",
            "mean_soft_brier",
            "residual_lag",
            "C",
            "temperature",
        ]
    )
    .reset_index(drop=True)
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)


# Save the selected hyperparameters
best = summary.iloc[0]

selected_feature_names = (
    features_for_lag(
        int(
            best["residual_lag"]
        )
    )
)

selected = {
    "dataset_id": "nn5_daily",
    "model": (
        "l1_logistic_"
        "soft_label_duplication"
    ),
    "selection_scope": (
        "router_train_only"
    ),
    "selection_metric": (
        "coverage_constrained_"
        "scaled_loss"
    ),
    "target_simple_coverage": (
        target_simple_coverage
    ),
    "rolling_folds": N_FOLDS,
    "validation_size_per_fold": (
        VALIDATION_SIZE
    ),
    "temperature": float(
        best["temperature"]
    ),
    "residual_lag": int(
        best["residual_lag"]
    ),
    "C": float(
        best["C"]
    ),
    "feature_count": (
        len(selected_feature_names)
    ),
    "mean_validation_"
    "constrained_scaled_loss": float(
        best[
            "mean_constrained_scaled_loss"
        ]
    ),
    "mean_validation_"
    "simple_coverage": float(
        best[
            "mean_simple_coverage"
        ]
    ),
    "mean_validation_"
    "black_box_AUPRC": float(
        best[
            "mean_black_box_AUPRC"
        ]
    ),
    "seed": seed,
    "feature_names": (
        selected_feature_names
    ),
}

with SELECTED_PATH.open(
    "w",
    encoding="utf-8",
) as handle:
    yaml.safe_dump(
        selected,
        handle,
        sort_keys=False,
        allow_unicode=True,
    )


# Plot the 15 highest-ranked configurations
top = (
    summary
    .head(15)
    .copy()
    .iloc[::-1]
)

labels = [
    (
        f"T={row.temperature:g}, "
        f"lag={int(row.residual_lag)}, "
        f"C={row.C:g}"
    )
    for row in top.itertuples()
]

fig, ax = plt.subplots(
    figsize=(10, 6.5)
)

ax.barh(
    labels,
    top[
        "mean_constrained_scaled_loss"
    ],
    color="#2E74B5",
)

ax.set_title(
    "NN5 soft-router tuning "
    "within router_train"
)
ax.set_xlabel(
    "Coverage-constrained scaled loss "
    "(lower is better)"
)
ax.grid(
    axis="x",
    alpha=0.2,
)

fig.tight_layout()
fig.savefig(
    FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# Report the tuning results
elapsed = (
    perf_counter() - overall_start
)

print()
print(
    "NN5 软路由器滚动调参全部通过"
)
print(
    "调参数据范围：仅 router_train"
)
print(
    "calibration 是否用于调参：否"
)
print(
    "滚动验证折数：",
    N_FOLDS,
)
print(
    "参数组合数量：",
    len(summary),
)
print(
    "实际模型拟合次数：",
    len(details),
)
print(
    "最佳温度：",
    float(best["temperature"]),
)
print(
    "最佳残差滞后：",
    int(best["residual_lag"]),
)
print(
    "最佳 C：",
    float(best["C"]),
)
print(
    "最佳特征数量：",
    len(selected_feature_names),
)
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
print(
    "运行秒数：",
    f"{elapsed:.2f}",
)
print("逐折结果：", DETAIL_PATH)
print("汇总结果：", SUMMARY_PATH)
print("选定参数：", SELECTED_PATH)
print("调参图片：", FIGURE_PATH)
