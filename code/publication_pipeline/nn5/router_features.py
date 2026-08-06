"""生成只使用当前及过去信息的路由特征。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nn5_daily_long.parquet"
)
PREDICTION_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_pretest_predictions.parquet"
)
SCALER_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_scaler_parameters.csv"
)
CONFIG_PATH = (
    PROJECT_ROOT
    / "experiment_config.yaml"
)

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
COUNT_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_router_feature_counts.csv"
)
FIGURE_PATH = (
    PROJECT_ROOT
    / "figures"
    / "nn5_router_target_distribution.png"
)

for path in [
    FEATURE_PATH,
    MANIFEST_PATH,
    COUNT_PATH,
    FIGURE_PATH,
]:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


# Load the residual-lag configuration
with CONFIG_PATH.open(
    "r",
    encoding="utf-8",
) as handle:
    config = yaml.safe_load(handle)

residual_lag_candidates = [
    int(item)
    for item in config[
        "soft_router"
    ]["residual_lag_grid"]
]

if residual_lag_candidates != [
    1,
    4,
    8,
    16,
]:
    raise ValueError(
        "残差滞后配置与方案不一致"
    )

max_residual_lag = max(
    residual_lag_candidates
)


# Load raw pretest observations
allowed_splits = [
    "base_train",
    "router_train",
    "calibration",
]

raw = pd.read_parquet(
    DATA_PATH,
    columns=[
        "series_id",
        "time_index",
        "value",
        "split",
    ],
    filters=[
        [("split", "==", split_name)]
        for split_name in allowed_splits
    ],
)

if (raw["split"] == "test").any():
    raise ValueError(
        "特征脚本读取了 test"
    )


# Load base-model forecasts
prediction = (
    pd.read_parquet(PREDICTION_PATH)
    .sort_values(
        [
            "series_id",
            "time_index",
        ]
    )
    .reset_index(drop=True)
)

if set(prediction["split"]) != {
    "router_train",
    "calibration",
}:
    raise ValueError(
        "预测文件包含不允许的数据段"
    )


# Load scaling parameters estimated from base_train
scalers = pd.read_csv(SCALER_PATH)

if scalers["source_split"].ne(
    "base_train"
).any():
    raise ValueError(
        "缩放参数来源不是 base_train"
    )

median_lookup = (
    scalers
    .set_index("series_id")["median"]
    .to_dict()
)

scale_lookup = (
    scalers
    .set_index("series_id")[
        "scale_used"
    ]
    .to_dict()
)


# Organize each raw series
raw_series = {
    series_id: (
        group
        .sort_values("time_index")[
            "value"
        ]
        .to_numpy(dtype=float)
    )
    for series_id, group in raw.groupby(
        "series_id",
        sort=False,
    )
}


# Compute the seasonal-naive MAE scale for each series
seasonal_mae_scale = {}

for series_id, group in raw.groupby(
    "series_id",
    sort=False,
):
    base_values = (
        group[
            group["split"] == "base_train"
        ]
        .sort_values("time_index")[
            "value"
        ]
        .to_numpy(dtype=float)
    )

    seasonal_difference = (
        base_values[7:]
        - base_values[:-7]
    )

    scale = float(
        np.mean(
            np.abs(
                seasonal_difference
            )
        )
    )

    if scale <= 1e-12:
        raise ValueError(
            f"{series_id} 的季节尺度为零"
        )

    seasonal_mae_scale[
        series_id
    ] = scale


prediction["median_base"] = (
    prediction["series_id"]
    .map(median_lookup)
)

prediction["iqr_base"] = (
    prediction["series_id"]
    .map(scale_lookup)
)

prediction[
    "seasonal_naive_mae_scale"
] = (
    prediction["series_id"]
    .map(seasonal_mae_scale)
)


# Construct context features using observations strictly before the target
context_records = []

for row in prediction.itertuples():
    values = raw_series[row.series_id]
    target = int(row.time_index)

    median = float(row.median_base)
    scale = float(row.iqr_base)

    # Exclude the target day.
    past_7 = values[
        target - 7:target
    ]
    past_28 = values[
        target - 28:target
    ]

    if (
        len(past_7) != 7
        or len(past_28) != 28
    ):
        raise ValueError(
            "上下文窗口长度不足"
        )

    context_records.append(
        {
            "last_value_scaled": (
                values[target - 1]
                - median
            ) / scale,
            "mean_7_scaled": (
                float(np.mean(past_7))
                - median
            ) / scale,
            "trend_7_scaled": (
                values[target - 1]
                - values[target - 8]
            ) / scale,
            "volatility_7_scaled": (
                float(
                    np.std(
                        past_7,
                        ddof=0,
                    )
                )
                / scale
            ),
            "trend_28_scaled": (
                values[target - 1]
                - values[target - 29]
            ) / scale,
            "volatility_28_scaled": (
                float(
                    np.std(
                        past_28,
                        ddof=0,
                    )
                )
                / scale
            ),
        }
    )

context = pd.DataFrame(
    context_records
)

prediction = pd.concat(
    [
        prediction,
        context,
    ],
    axis=1,
)


# Construct current-forecast and disagreement features
prediction[
    "ridge_prediction_scaled"
] = (
    prediction["ridge_prediction"]
    - prediction["median_base"]
) / prediction["iqr_base"]

prediction[
    "lightgbm_prediction_scaled"
] = (
    prediction["lightgbm_prediction"]
    - prediction["median_base"]
) / prediction["iqr_base"]

prediction[
    "prediction_difference_scaled"
] = (
    prediction["lightgbm_prediction"]
    - prediction["ridge_prediction"]
) / prediction["iqr_base"]

prediction[
    "absolute_prediction_difference_scaled"
] = prediction[
    "prediction_difference_scaled"
].abs()


# Construct cyclic weekday features
timestamp = pd.to_datetime(
    prediction["timestamp"],
    utc=True,
)

day_of_week = (
    timestamp
    .dt
    .dayofweek
    .to_numpy(dtype=float)
)

prediction["day_of_week_sin"] = (
    np.sin(
        2
        * np.pi
        * day_of_week
        / 7.0
    )
)

prediction["day_of_week_cos"] = (
    np.cos(
        2
        * np.pi
        * day_of_week
        / 7.0
    )
)


# Construct residual features at lags 1 through 16
residual_feature_names = []

grouped = prediction.groupby(
    "series_id",
    sort=False,
)

for lag in range(
    1,
    max_residual_lag + 1,
):
    ridge_name = (
        f"ridge_residual_lag_{lag}"
    )
    lightgbm_name = (
        f"lightgbm_residual_lag_{lag}"
    )

    # shift(lag) prevents the current residual from entering its own features.
    prediction[ridge_name] = (
        grouped[
            "ridge_residual"
        ].shift(lag)
        / prediction["iqr_base"]
    )

    prediction[lightgbm_name] = (
        grouped[
            "lightgbm_residual"
        ].shift(lag)
        / prediction["iqr_base"]
    )

    residual_feature_names.extend(
        [
            ridge_name,
            lightgbm_name,
        ]
    )


# Construct routing targets
prediction[
    "ridge_scaled_loss"
] = (
    prediction["ridge_residual"]
    / prediction[
        "seasonal_naive_mae_scale"
    ]
) ** 2

prediction[
    "lightgbm_scaled_loss"
] = (
    prediction[
        "lightgbm_residual"
    ]
    / prediction[
        "seasonal_naive_mae_scale"
    ]
) ** 2

# Positive values indicate a smaller LightGBM error.
prediction[
    "loss_advantage_black_box"
] = (
    prediction["ridge_scaled_loss"]
    - prediction[
        "lightgbm_scaled_loss"
    ]
)

prediction[
    "hard_black_box_target"
] = (
    prediction[
        "loss_advantage_black_box"
    ]
    > 0
).astype(np.int8)


# Define the registered feature set
context_feature_names = [
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

feature_names = (
    context_feature_names
    + residual_feature_names
)


# Remove the first 16 warm-up observations from each series
missing_before_warmup = (
    prediction[feature_names]
    .isna()
    .any(axis=1)
)

features = (
    prediction.loc[
        ~missing_before_warmup
    ]
    .copy()
    .reset_index(drop=True)
)

if features[
    feature_names
].isna().any().any():
    raise ValueError(
        "保留的路由特征仍有缺失值"
    )

if not np.isfinite(
    features[
        feature_names
    ].to_numpy(dtype=float)
).all():
    raise ValueError(
        "路由特征出现非有限值"
    )


# Validate sample counts
counts = (
    features
    .groupby(
        "split",
        as_index=False,
    )
    .agg(
        rows=(
            "series_id",
            "size",
        ),
        series_count=(
            "series_id",
            "nunique",
        ),
    )
)

expected_counts = {
    "router_train": (
        111 * (119 - 16)
    ),
    "calibration": (
        111 * 79
    ),
}

observed_counts = (
    counts
    .set_index("split")["rows"]
    .to_dict()
)

if observed_counts != expected_counts:
    raise ValueError(
        "特征样本数量异常："
        f"{observed_counts}"
    )


# Save the routing features
output_columns = [
    "dataset_id",
    "series_id",
    "time_index",
    "timestamp",
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

features[
    output_columns
].to_parquet(
    FEATURE_PATH,
    index=False,
)

counts.to_csv(
    COUNT_PATH,
    index=False,
)


# Create the feature manifest
descriptions = {
    "last_value_scaled": (
        "Most recent observed value, "
        "median/IQR scaled"
    ),
    "mean_7_scaled": (
        "Mean of the seven observations "
        "before the target"
    ),
    "trend_7_scaled": (
        "Change between target-1 "
        "and target-8"
    ),
    "volatility_7_scaled": (
        "Standard deviation of the "
        "previous seven values"
    ),
    "trend_28_scaled": (
        "Change between target-1 "
        "and target-29"
    ),
    "volatility_28_scaled": (
        "Standard deviation of the "
        "previous 28 values"
    ),
    "ridge_prediction_scaled": (
        "Current causal Ridge prediction"
    ),
    "lightgbm_prediction_scaled": (
        "Current causal LightGBM prediction"
    ),
    "prediction_difference_scaled": (
        "LightGBM minus Ridge prediction"
    ),
    "absolute_prediction_difference_scaled": (
        "Absolute model disagreement"
    ),
    "day_of_week_sin": (
        "Sine encoding of day of week"
    ),
    "day_of_week_cos": (
        "Cosine encoding of day of week"
    ),
}

manifest_records = []

for name in context_feature_names:
    manifest_records.append(
        {
            "feature": name,
            "group": "context",
            "available_before_target": True,
            "description": descriptions[name],
        }
    )

for name in residual_feature_names:
    lag = int(
        name.rsplit("_", 1)[1]
    )

    model_name = (
        "Ridge"
        if name.startswith("ridge")
        else "LightGBM"
    )

    manifest_records.append(
        {
            "feature": name,
            "group": "past_residual",
            "available_before_target": True,
            "description": (
                f"{model_name} residual "
                f"from {lag} day(s) "
                "before target"
            ),
        }
    )

manifest = pd.DataFrame(
    manifest_records
)

manifest.to_csv(
    MANIFEST_PATH,
    index=False,
)


# Plot the routing-target distribution
fig, axes = plt.subplots(
    1,
    2,
    figsize=(11, 4.5),
)

win_rate = (
    features
    .groupby("split")[
        "hard_black_box_target"
    ]
    .mean()
    .reindex(
        [
            "router_train",
            "calibration",
        ]
    )
)

axes[0].bar(
    win_rate.index,
    win_rate.values,
    color=[
        "#2E74B5",
        "#E09F3E",
    ],
)
axes[0].axhline(
    0.5,
    color="#555555",
    linestyle="--",
    linewidth=1,
)
axes[0].set_ylim(0, 1)
axes[0].set_ylabel(
    "Black-box win rate"
)
axes[0].set_title(
    "Routing target balance"
)

router_advantage = features.loc[
    features["split"] == "router_train",
    "loss_advantage_black_box",
]

axes[1].hist(
    router_advantage,
    bins=60,
    color="#6BAED6",
    edgecolor="none",
)
axes[1].axvline(
    0,
    color="#9B1C1C",
    linestyle="--",
    linewidth=1,
)
axes[1].set_xlabel(
    "Scaled loss advantage of black box"
)
axes[1].set_ylabel("Count")
axes[1].set_title(
    "Router-train loss advantage"
)

fig.tight_layout()
fig.savefig(
    FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# Report the feature-generation results
print("NN5 路由特征生成全部通过")
print(
    "最大残差滞后：",
    max_residual_lag,
)
print(
    "路由特征数量：",
    len(feature_names),
)
print("未来信息检查：通过")
print("各数据段样本数量：")
print(counts.to_string(index=False))
print(
    "各数据段 LightGBM 单点胜率："
)
print(win_rate.to_string())
print("路由特征：", FEATURE_PATH)
print("特征清单：", MANIFEST_PATH)
print("样本计数：", COUNT_PATH)
print("目标分布图：", FIGURE_PATH)
