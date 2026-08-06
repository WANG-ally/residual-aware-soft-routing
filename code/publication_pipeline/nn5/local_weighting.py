"""局部加权消融实验，并训练最终软路由器。"""

import os
from pathlib import Path
from time import perf_counter

# Suppress nonessential joblib CPU-detection warnings on macOS.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import yaml


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURE_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_router_features.parquet"
)

CONFIG_PATH = (
    PROJECT_ROOT
    / "experiment_config.yaml"
)

SELECTED_ROUTER_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_selected_soft_router_params.yaml"
)

DETAIL_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_local_weighting_rolling_validation.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_local_weighting_tuning_summary.csv"
)

SELECTED_LOCAL_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_selected_local_weighting_params.yaml"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "nn5_soft_router.joblib"
)

COEFFICIENT_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_router_coefficients.csv"
)

CALIBRATION_SCORE_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_calibration_router_scores.parquet"
)

FIGURE_PATH = (
    PROJECT_ROOT
    / "figures"
    / "nn5_router_coefficients_and_scores.png"
)

for path in [
    DETAIL_PATH,
    SUMMARY_PATH,
    SELECTED_LOCAL_PATH,
    MODEL_PATH,
    COEFFICIENT_PATH,
    CALIBRATION_SCORE_PATH,
    FIGURE_PATH,
]:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Load the experiment configuration
# ============================================================

with CONFIG_PATH.open(
    "r",
    encoding="utf-8",
) as handle:
    config = yaml.safe_load(handle)

with SELECTED_ROUTER_PATH.open(
    "r",
    encoding="utf-8",
) as handle:
    router_parameters = yaml.safe_load(handle)

if (
    router_parameters["selection_scope"]
    != "router_train_only"
):
    raise ValueError(
        "软路由参数不是仅使用 router_train 选择的"
    )

if (
    config["local_weighting"]["fit_scope"]
    != "router_train_only"
):
    raise ValueError(
        "局部加权只能在 router_train 内调参"
    )

seed = int(
    config["study"]["seed"]
)

target_simple_coverage = float(
    config["study"]["primary_target_coverage"]
)

temperature = float(
    router_parameters["temperature"]
)

c_value = float(
    router_parameters["C"]
)

residual_lag = int(
    router_parameters["residual_lag"]
)

feature_names = list(
    router_parameters["feature_names"]
)

k_grid = [
    int(value)
    for value in config[
        "local_weighting"
    ]["k_grid"]
]

alpha_grid = [
    float(value)
    for value in config[
        "local_weighting"
    ]["alpha_grid"]
]

maximum_multiplier = float(
    config[
        "local_weighting"
    ]["maximum_multiplier"]
)


# ============================================================
# Load the routing data
# ============================================================

router = pd.read_parquet(
    FEATURE_PATH,
    filters=[
        ("split", "==", "router_train")
    ],
)

calibration = pd.read_parquet(
    FEATURE_PATH,
    filters=[
        ("split", "==", "calibration")
    ],
)

if router.empty:
    raise ValueError(
        "没有读取到 router_train 数据"
    )

if calibration.empty:
    raise ValueError(
        "没有读取到 calibration 数据"
    )

if set(router["split"]) != {"router_train"}:
    raise ValueError(
        "router 数据中混入了其他数据段"
    )

if set(calibration["split"]) != {"calibration"}:
    raise ValueError(
        "calibration 数据中混入了其他数据段"
    )

if "test" in set(
    pd.concat(
        [
            router["split"],
            calibration["split"],
        ]
    )
):
    raise ValueError(
        "本步骤禁止访问 test"
    )

missing_features = [
    name
    for name in feature_names
    if name not in router.columns
]

if missing_features:
    raise ValueError(
        f"缺少路由特征：{missing_features}"
    )

if router[feature_names].isna().any().any():
    raise ValueError(
        "router_train 特征存在缺失值"
    )

if calibration[
    feature_names
].isna().any().any():
    raise ValueError(
        "calibration 特征存在缺失值"
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


# ============================================================
# Define three expanding-window validation folds
# ============================================================

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


# ============================================================
# Helper functions
# ============================================================

def stable_soft_target(
    loss_advantage,
    selected_temperature,
):
    """把模型损失差转换成0到1之间的软标签。"""

    normalized = np.clip(
        loss_advantage
        / selected_temperature,
        -40.0,
        40.0,
    )

    return 1.0 / (
        1.0 + np.exp(-normalized)
    )


def build_neighbor_matrix(
    feature_matrix,
    maximum_k,
):
    """只使用当前训练折建立最近邻。"""

    neighbor_model = NearestNeighbors(
        n_neighbors=maximum_k + 1,
        metric="euclidean",
        n_jobs=1,
    )

    neighbor_model.fit(
        feature_matrix
    )

    raw_indices = (
        neighbor_model.kneighbors(
            feature_matrix,
            return_distance=False,
        )
    )

    cleaned_indices = []

    for row_number, row in enumerate(
        raw_indices
    ):
        # Exclude each observation from its own neighbor set.
        without_self = row[
            row != row_number
        ]

        if len(without_self) < maximum_k:
            raise ValueError(
                "最近邻数量不足"
            )

        cleaned_indices.append(
            without_self[:maximum_k]
        )

    return np.asarray(
        cleaned_indices,
        dtype=int,
    )


def calculate_local_weights(
    hard_target,
    neighbor_indices,
    k_value,
    alpha_value,
):
    """根据邻域标签一致性计算局部样本权重。"""

    if alpha_value == 0.0:
        return np.ones(
            len(hard_target),
            dtype=float,
        )

    selected_neighbors = (
        neighbor_indices[:, :k_value]
    )

    neighbor_targets = hard_target[
        selected_neighbors
    ]

    local_support = np.mean(
        neighbor_targets
        == hard_target[:, None],
        axis=1,
    )

    minimum_support = 1.0 / (
        k_value + 1.0
    )

    safe_support = np.maximum(
        local_support,
        minimum_support,
    )

    raw_weights = (
        1.0 / safe_support
    ) ** alpha_value

    capped_weights = np.minimum(
        raw_weights,
        maximum_multiplier,
    )

    # Normalize the local weights to have mean one.
    normalized_weights = (
        capped_weights
        / np.mean(capped_weights)
    )

    return normalized_weights


def fit_soft_router(
    train_x,
    train_soft_target,
    local_weights,
):
    """使用样本复制法训练软标签 Logistic 回归。"""

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

    duplicated_weights = np.concatenate(
        [
            local_weights
            * train_soft_target,
            local_weights
            * (
                1.0
                - train_soft_target
            ),
        ]
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
        sample_weight=duplicated_weights,
    )

    if int(model.n_iter_[0]) >= model.max_iter:
        raise ValueError(
            "Logistic 回归没有收敛"
        )

    return model


# ============================================================
# Evaluate local weighting by expanding-window validation
# ============================================================

records = []
overall_start = perf_counter()
maximum_k = max(k_grid)

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
            "滚动时间切分存在交叉"
        )

    # Fit the standardizer using only the current training fold.
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

    train_soft_target = stable_soft_target(
        train[
            "loss_advantage_black_box"
        ].to_numpy(dtype=float),
        temperature,
    )

    train_hard_target = train[
        "hard_black_box_target"
    ].to_numpy(dtype=int)

    validation_hard_target = validation[
        "hard_black_box_target"
    ].to_numpy(dtype=int)

    neighbor_indices = build_neighbor_matrix(
        train_x,
        maximum_k,
    )

    for k_value in k_grid:
        for alpha_value in alpha_grid:

            local_weights = calculate_local_weights(
                train_hard_target,
                neighbor_indices,
                k_value,
                alpha_value,
            )

            model = fit_soft_router(
                train_x,
                train_soft_target,
                local_weights,
            )

            probability = np.clip(
                model.predict_proba(
                    validation_x
                )[:, 1],
                1e-8,
                1.0 - 1e-8,
            )

            # Higher probabilities favor LightGBM.
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

            scaled_squared_error = (
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

            records.append(
                {
                    "dataset_id": "nn5_daily",
                    "fold": fold,
                    "train_start": int(
                        train["time_index"].min()
                    ),
                    "train_end": int(
                        train["time_index"].max()
                    ),
                    "validation_start": validation_start,
                    "validation_end": validation_end,
                    "k": k_value,
                    "alpha": alpha_value,
                    "temperature": temperature,
                    "residual_lag": residual_lag,
                    "C": c_value,
                    "feature_count": len(
                        feature_names
                    ),
                    "train_rows": len(train),
                    "validation_rows": len(
                        validation
                    ),
                    "constrained_scaled_loss": float(
                        np.mean(
                            scaled_squared_error
                        )
                    ),
                    "simple_coverage": float(
                        np.mean(
                            ~use_black_box
                        )
                    ),
                    "black_box_AUPRC": float(
                        average_precision_score(
                            validation_hard_target,
                            probability,
                        )
                    ),
                    "hard_brier": float(
                        brier_score_loss(
                            validation_hard_target,
                            probability,
                        )
                    ),
                    "mean_local_weight": float(
                        np.mean(local_weights)
                    ),
                    "maximum_local_weight": float(
                        np.max(local_weights)
                    ),
                    "nonzero_coefficients": int(
                        np.count_nonzero(
                            model.coef_
                        )
                    ),
                    "converged": True,
                    "seed": seed,
                }
            )

    print(
        f"[完成] fold={fold}/3，"
        f"训练截止={validation_start - 1}，"
        f"验证={validation_start}-{validation_end}"
    )


# ============================================================
# Save fold-level results and select the weighting parameters
# ============================================================

details = pd.DataFrame(records)

expected_fits = (
    N_FOLDS
    * len(k_grid)
    * len(alpha_grid)
)

if len(details) != expected_fits:
    raise ValueError(
        "局部加权模型拟合次数异常"
    )

details.to_csv(
    DETAIL_PATH,
    index=False,
)

summary = (
    details
    .groupby(
        [
            "k",
            "alpha",
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
        mean_black_box_AUPRC=(
            "black_box_AUPRC",
            "mean",
        ),
        mean_hard_brier=(
            "hard_brier",
            "mean",
        ),
        mean_nonzero_coefficients=(
            "nonzero_coefficients",
            "mean",
        ),
        mean_maximum_local_weight=(
            "maximum_local_weight",
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
            "mean_hard_brier",
            "alpha",
            "k",
        ]
    )
    .reset_index(drop=True)
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)

best = summary.iloc[0]

best_k = int(
    best["k"]
)

best_alpha = float(
    best["alpha"]
)

local_weighting_selected = bool(
    best_alpha > 0.0
)

selected_local_parameters = {
    "dataset_id": "nn5_daily",
    "selection_scope": "router_train_only",
    "selection_metric": (
        "coverage_constrained_scaled_loss"
    ),
    "target_simple_coverage": (
        target_simple_coverage
    ),
    "rolling_folds": N_FOLDS,
    "validation_size_per_fold": (
        VALIDATION_SIZE
    ),
    "k": best_k,
    "alpha": best_alpha,
    "maximum_multiplier": (
        maximum_multiplier
    ),
    "local_weighting_selected": (
        local_weighting_selected
    ),
    "temperature": temperature,
    "residual_lag": residual_lag,
    "C": c_value,
    "feature_count": len(
        feature_names
    ),
    "mean_validation_constrained_scaled_loss": float(
        best[
            "mean_constrained_scaled_loss"
        ]
    ),
    "mean_validation_simple_coverage": float(
        best["mean_simple_coverage"]
    ),
    "mean_validation_black_box_AUPRC": float(
        best[
            "mean_black_box_AUPRC"
        ]
    ),
    "seed": seed,
}

with SELECTED_LOCAL_PATH.open(
    "w",
    encoding="utf-8",
) as handle:
    yaml.safe_dump(
        selected_local_parameters,
        handle,
        sort_keys=False,
        allow_unicode=True,
    )


# ============================================================
# Fit the final router using all of router_train
# ============================================================

final_scaler = StandardScaler()

final_train_x = (
    final_scaler.fit_transform(
        router[
            feature_names
        ].to_numpy(dtype=float)
    )
)

final_soft_target = stable_soft_target(
    router[
        "loss_advantage_black_box"
    ].to_numpy(dtype=float),
    temperature,
)

final_hard_target = router[
    "hard_black_box_target"
].to_numpy(dtype=int)

final_neighbor_indices = (
    build_neighbor_matrix(
        final_train_x,
        best_k,
    )
)

final_local_weights = (
    calculate_local_weights(
        final_hard_target,
        final_neighbor_indices,
        best_k,
        best_alpha,
    )
)

final_model = fit_soft_router(
    final_train_x,
    final_soft_target,
    final_local_weights,
)

model_bundle = {
    "dataset_id": "nn5_daily",
    "model": final_model,
    "scaler": final_scaler,
    "feature_names": feature_names,
    "temperature": temperature,
    "residual_lag": residual_lag,
    "C": c_value,
    "k": best_k,
    "alpha": best_alpha,
    "local_weighting_selected": (
        local_weighting_selected
    ),
    "target_simple_coverage": (
        target_simple_coverage
    ),
    "training_scope": "router_train_only",
    "seed": seed,
}

joblib.dump(
    model_bundle,
    MODEL_PATH,
)


# ============================================================
# Save the final router coefficients
# ============================================================

coefficients = pd.DataFrame(
    {
        "feature": feature_names,
        "coefficient": (
            final_model.coef_[0]
        ),
    }
)

coefficients[
    "absolute_coefficient"
] = coefficients[
    "coefficient"
].abs()

coefficients = (
    coefficients
    .sort_values(
        "absolute_coefficient",
        ascending=False,
    )
    .reset_index(drop=True)
)

coefficients.to_csv(
    COEFFICIENT_PATH,
    index=False,
)


# ============================================================
# Generate router probabilities for calibration only
# ============================================================

calibration_x = (
    final_scaler.transform(
        calibration[
            feature_names
        ].to_numpy(dtype=float)
    )
)

calibration_probability = np.clip(
    final_model.predict_proba(
        calibration_x
    )[:, 1],
    1e-8,
    1.0 - 1e-8,
)

calibration_hard_target = calibration[
    "hard_black_box_target"
].to_numpy(dtype=int)

calibration_auprc = float(
    average_precision_score(
        calibration_hard_target,
        calibration_probability,
    )
)

calibration_brier = float(
    brier_score_loss(
        calibration_hard_target,
        calibration_probability,
    )
)

calibration_scores = calibration[
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

calibration_scores[
    "black_box_probability"
] = calibration_probability

calibration_scores.to_parquet(
    CALIBRATION_SCORE_PATH,
    index=False,
)


# ============================================================
# Plot model coefficients and calibration probabilities
# ============================================================

nonzero_coefficients = coefficients[
    coefficients["coefficient"] != 0.0
].copy()

plot_coefficients = (
    nonzero_coefficients
    .head(15)
    .sort_values(
        "coefficient"
    )
)

figure, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5),
)

if plot_coefficients.empty:
    axes[0].text(
        0.5,
        0.5,
        "No non-zero coefficients",
        ha="center",
        va="center",
    )
else:
    colors = np.where(
        plot_coefficients[
            "coefficient"
        ] >= 0,
        "#d95f02",
        "#1b9e77",
    )

    axes[0].barh(
        plot_coefficients["feature"],
        plot_coefficients["coefficient"],
        color=colors,
    )

axes[0].axvline(
    0.0,
    color="black",
    linewidth=0.8,
)

axes[0].set_title(
    "Final soft-router coefficients"
)

axes[0].set_xlabel(
    "Standardized logistic coefficient"
)

negative_scores = calibration_probability[
    calibration_hard_target == 0
]

positive_scores = calibration_probability[
    calibration_hard_target == 1
]

axes[1].hist(
    negative_scores,
    bins=30,
    alpha=0.65,
    label="Ridge wins",
    color="#1b9e77",
    density=True,
)

axes[1].hist(
    positive_scores,
    bins=30,
    alpha=0.65,
    label="LightGBM wins",
    color="#d95f02",
    density=True,
)

axes[1].set_title(
    "Calibration routing probabilities"
)

axes[1].set_xlabel(
    "Predicted probability of using LightGBM"
)

axes[1].set_ylabel(
    "Density"
)

axes[1].legend()

figure.tight_layout()

figure.savefig(
    FIGURE_PATH,
    dpi=180,
    bbox_inches="tight",
)

plt.close(figure)


# ============================================================
# Report the local-weighting results
# ============================================================

elapsed_seconds = (
    perf_counter()
    - overall_start
)

print()
print("NN5 局部加权与最终路由器训练全部通过")
print("调参数据范围：仅 router_train")
print("测试集是否访问：否")
print("局部加权参数组合数量：", len(summary))
print("实际模型拟合次数：", len(details))
print("最佳 k：", best_k)
print("最佳 alpha：", best_alpha)
print(
    "是否选择局部加权：",
    "是"
    if local_weighting_selected
    else "否",
)
print(
    "平均约束缩放损失：",
    f"{best['mean_constrained_scaled_loss']:.6f}",
)
print(
    "平均简单模型覆盖率：",
    f"{best['mean_simple_coverage']:.6f}",
)
print(
    "平均 LightGBM AUPRC：",
    f"{best['mean_black_box_AUPRC']:.6f}",
)
print(
    "最终非零系数数量：",
    int(
        np.count_nonzero(
            final_model.coef_
        )
    ),
)
print(
    "calibration 样本数量：",
    len(calibration_scores),
)
print(
    "calibration AUPRC：",
    f"{calibration_auprc:.6f}",
)
print(
    "calibration Brier：",
    f"{calibration_brier:.6f}",
)
print(
    "运行秒数：",
    f"{elapsed_seconds:.2f}",
)
print("逐折结果：", DETAIL_PATH)
print("汇总结果：", SUMMARY_PATH)
print("选定参数：", SELECTED_LOCAL_PATH)
print("最终模型：", MODEL_PATH)
print("模型系数：", COEFFICIENT_PATH)
print(
    "校准概率：",
    CALIBRATION_SCORE_PATH,
)
print("结果图片：", FIGURE_PATH)
