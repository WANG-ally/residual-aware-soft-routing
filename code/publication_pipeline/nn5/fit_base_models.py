"""训练最终基础模型并生成预测试残差。"""

from pathlib import Path
from time import perf_counter

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
import yaml


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nn5_daily_long.parquet"
)
SCALER_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_scaler_parameters.csv"
)
RIDGE_PARAMS_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_selected_ridge_params.yaml"
)
LGBM_PARAMS_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_selected_lightgbm_params.yaml"
)

RIDGE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "nn5_ridge.joblib"
)
LGBM_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "nn5_lightgbm.joblib"
)
PREDICTION_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_pretest_predictions.parquet"
)
PER_SERIES_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_pretest_per_series_metrics.csv"
)
AGGREGATE_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_pretest_aggregate_metrics.csv"
)
METADATA_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_base_model_fit_metadata.yaml"
)
FIGURE_PATH = (
    PROJECT_ROOT
    / "figures"
    / "nn5_pretest_predictions_T1.png"
)

for path in [
    RIDGE_MODEL_PATH,
    LGBM_MODEL_PATH,
    PREDICTION_PATH,
    PER_SERIES_PATH,
    AGGREGATE_PATH,
    METADATA_PATH,
    FIGURE_PATH,
]:
    path.parent.mkdir(parents=True, exist_ok=True)


# Load the selected hyperparameters
with RIDGE_PARAMS_PATH.open(
    "r",
    encoding="utf-8",
) as handle:
    ridge_params = yaml.safe_load(handle)

with LGBM_PARAMS_PATH.open(
    "r",
    encoding="utf-8",
) as handle:
    lgbm_params = yaml.safe_load(handle)

for selected in [
    ridge_params,
    lgbm_params,
]:
    if (
        selected["selection_scope"]
        != "base_train_only"
    ):
        raise ValueError(
            "参数不是仅从 base_train 选择的"
        )


# Load pretest observations only
allowed_splits = [
    "base_train",
    "router_train",
    "calibration",
]

data = pd.read_parquet(
    DATA_PATH,
    columns=[
        "dataset_id",
        "series_id",
        "time_index",
        "timestamp",
        "value",
        "split",
    ],
    filters=[
        [("split", "==", split_name)]
        for split_name in allowed_splits
    ],
)

if (
    set(data["split"])
    != set(allowed_splits)
    or (data["split"] == "test").any()
):
    raise ValueError(
        "预测试脚本读取了错误的数据段"
    )


# Load scaling parameters estimated from base_train
scalers = pd.read_csv(SCALER_PATH)

if scalers["source_split"].ne(
    "base_train"
).any():
    raise ValueError(
        "缩放参数不是仅由 base_train 计算"
    )

scaler_lookup = {
    row.series_id: (
        float(row.median),
        float(row.scale_used),
    )
    for row in scalers.itertuples()
}


# Organize each series
series_frames = {
    series_id: (
        group
        .sort_values("time_index")
        .reset_index(drop=True)
    )
    for series_id, group in data.groupby(
        "series_id",
        sort=False,
    )
}

for series_id, group in series_frames.items():
    indices = group[
        "time_index"
    ].to_numpy(dtype=np.int64)

    if not np.array_equal(
        indices,
        np.arange(len(group)),
    ):
        raise ValueError(
            f"{series_id} 的时间索引不连续"
        )

    # Only indices 0 through 671 are available at this stage.
    if len(group) != 672:
        raise ValueError(
            f"{series_id} 的预测试长度不是672"
        )


# Construct base-model training samples
def build_training_arrays(window):
    x_parts = []
    y_parts = []

    for series_id, group in series_frames.items():
        values = group[
            "value"
        ].to_numpy(dtype=float)

        median, scale = scaler_lookup[
            series_id
        ]

        scaled = (
            values - median
        ) / scale

        target_indices = group.loc[
            group["split"] == "base_train",
            "time_index",
        ].to_numpy(dtype=np.int64)

        target_indices = target_indices[
            target_indices >= window
        ]

        x_parts.append(
            np.asarray(
                [
                    scaled[
                        target - window:target
                    ][::-1]
                    for target in target_indices
                ],
                dtype=float,
            )
        )

        y_parts.append(
            scaled[target_indices]
        )

    return (
        np.vstack(x_parts),
        np.concatenate(y_parts),
    )


# Construct prediction windows for router_train and calibration
def build_prediction_arrays(window):
    x_parts = []
    metadata_parts = []
    median_parts = []
    scale_parts = []

    for series_id, group in series_frames.items():
        values = group[
            "value"
        ].to_numpy(dtype=float)

        median, scale = scaler_lookup[
            series_id
        ]

        scaled = (
            values - median
        ) / scale

        prediction_mask = group[
            "split"
        ].isin(
            [
                "router_train",
                "calibration",
            ]
        )

        targets = group.loc[
            prediction_mask,
            "time_index",
        ].to_numpy(dtype=np.int64)

        # Verify that every window excludes future observations.
        for target in targets:
            history = np.arange(
                target - window,
                target,
            )

            if (
                len(history) != window
                or history[-1] >= target
            ):
                raise ValueError(
                    "窗口包含目标当天或未来信息"
                )

        x_parts.append(
            np.asarray(
                [
                    scaled[
                        target - window:target
                    ][::-1]
                    for target in targets
                ],
                dtype=float,
            )
        )

        metadata_parts.append(
            group.loc[
                prediction_mask,
                [
                    "dataset_id",
                    "series_id",
                    "time_index",
                    "timestamp",
                    "split",
                    "value",
                ],
            ]
            .rename(
                columns={
                    "value": "y_true"
                }
            )
            .reset_index(drop=True)
        )

        median_parts.append(
            np.full(
                len(targets),
                median,
            )
        )
        scale_parts.append(
            np.full(
                len(targets),
                scale,
            )
        )

    return (
        np.vstack(x_parts),
        pd.concat(
            metadata_parts,
            ignore_index=True,
        ),
        np.concatenate(median_parts),
        np.concatenate(scale_parts),
    )


# Fit Ridge
ridge_window = int(
    ridge_params["selected_window"]
)

ridge_x, ridge_y = (
    build_training_arrays(ridge_window)
)

start_time = perf_counter()

ridge = Ridge(
    alpha=float(
        ridge_params["selected_alpha"]
    ),
    fit_intercept=True,
)

ridge.fit(ridge_x, ridge_y)

ridge_fit_seconds = (
    perf_counter() - start_time
)


# Fit LightGBM
lgbm_window = int(
    lgbm_params["selected_window"]
)

lgbm_x, lgbm_y = (
    build_training_arrays(lgbm_window)
)

seed = int(lgbm_params["seed"])

start_time = perf_counter()

lgbm = LGBMRegressor(
    objective="regression_l2",
    num_leaves=int(
        lgbm_params["num_leaves"]
    ),
    learning_rate=float(
        lgbm_params["learning_rate"]
    ),
    n_estimators=int(
        lgbm_params["n_estimators"]
    ),
    feature_fraction=float(
        lgbm_params["feature_fraction"]
    ),
    random_state=seed,
    feature_fraction_seed=seed,
    data_random_seed=seed,
    deterministic=True,
    force_col_wise=True,
    n_jobs=-1,
    verbosity=-1,
)

lgbm.fit(lgbm_x, lgbm_y)

lgbm_fit_seconds = (
    perf_counter() - start_time
)


# Save the fitted models
joblib.dump(
    ridge,
    RIDGE_MODEL_PATH,
)
joblib.dump(
    lgbm,
    LGBM_MODEL_PATH,
)


# Generate forecasts from both models
(
    ridge_prediction_x,
    ridge_metadata,
    ridge_medians,
    ridge_scales,
) = build_prediction_arrays(
    ridge_window
)

(
    lgbm_prediction_x,
    lgbm_metadata,
    lgbm_medians,
    lgbm_scales,
) = build_prediction_arrays(
    lgbm_window
)

key_columns = [
    "dataset_id",
    "series_id",
    "time_index",
    "split",
]

if not ridge_metadata[
    key_columns
].equals(
    lgbm_metadata[key_columns]
):
    raise ValueError(
        "两个模型的预测目标不一致"
    )

prediction = ridge_metadata.copy()

prediction["ridge_prediction"] = (
    ridge.predict(ridge_prediction_x)
    * ridge_scales
    + ridge_medians
)

prediction[
    "lightgbm_prediction"
] = (
    lgbm
    .booster_
    .predict(lgbm_prediction_x)
    * lgbm_scales
    + lgbm_medians
)


# Compute pointwise residuals
prediction["ridge_residual"] = (
    prediction["y_true"]
    - prediction["ridge_prediction"]
)

prediction["lightgbm_residual"] = (
    prediction["y_true"]
    - prediction["lightgbm_prediction"]
)

prediction["ridge_squared_error"] = (
    prediction["ridge_residual"] ** 2
)

prediction[
    "lightgbm_squared_error"
] = (
    prediction[
        "lightgbm_residual"
    ] ** 2
)

prediction["black_box_better"] = (
    prediction[
        "lightgbm_squared_error"
    ]
    < prediction[
        "ridge_squared_error"
    ]
)

if set(prediction["split"]) != {
    "router_train",
    "calibration",
}:
    raise ValueError(
        "预测结果包含不允许的数据段"
    )

prediction.to_parquet(
    PREDICTION_PATH,
    index=False,
)


# Compute per-series RMSSE and MASE denominators
denominators = {}

for series_id, group in series_frames.items():
    base_values = group.loc[
        group["split"] == "base_train",
        "value",
    ].to_numpy(dtype=float)

    seasonal_differences = (
        base_values[7:]
        - base_values[:-7]
    )

    denominators[series_id] = {
        "squared": float(
            np.mean(
                seasonal_differences ** 2
            )
        ),
        "absolute": float(
            np.mean(
                np.abs(
                    seasonal_differences
                )
            )
        ),
    }


# Compute pretest metrics
metric_records = []

for split_name in [
    "router_train",
    "calibration",
]:
    split_data = prediction[
        prediction["split"] == split_name
    ]

    for series_id, group in split_data.groupby(
        "series_id",
        sort=False,
    ):
        y_true = group[
            "y_true"
        ].to_numpy(dtype=float)

        for model_name, prediction_column in [
            (
                "ridge",
                "ridge_prediction",
            ),
            (
                "lightgbm",
                "lightgbm_prediction",
            ),
        ]:
            predicted = group[
                prediction_column
            ].to_numpy(dtype=float)

            error = y_true - predicted

            metric_records.append(
                {
                    "dataset_id": "nn5_daily",
                    "split": split_name,
                    "series_id": series_id,
                    "model": model_name,
                    "RMSSE": float(
                        np.sqrt(
                            np.mean(
                                error ** 2
                            )
                            / denominators[
                                series_id
                            ]["squared"]
                        )
                    ),
                    "MASE": float(
                        np.mean(
                            np.abs(error)
                        )
                        / denominators[
                            series_id
                        ]["absolute"]
                    ),
                    "sMAPE": float(
                        100
                        * np.mean(
                            2
                            * np.abs(error)
                            / (
                                np.abs(y_true)
                                + np.abs(predicted)
                                + 1e-8
                            )
                        )
                    ),
                    "RMSE": float(
                        np.sqrt(
                            np.mean(
                                error ** 2
                            )
                        )
                    ),
                    "MAE": float(
                        np.mean(
                            np.abs(error)
                        )
                    ),
                }
            )

per_series = pd.DataFrame(
    metric_records
)

per_series.to_csv(
    PER_SERIES_PATH,
    index=False,
)

aggregate = (
    per_series
    .groupby(
        ["split", "model"],
        as_index=False,
    )
    .agg(
        mean_RMSSE=(
            "RMSSE",
            "mean",
        ),
        median_RMSSE=(
            "RMSSE",
            "median",
        ),
        mean_MASE=(
            "MASE",
            "mean",
        ),
        mean_sMAPE=(
            "sMAPE",
            "mean",
        ),
        mean_RMSE=(
            "RMSE",
            "mean",
        ),
        mean_MAE=(
            "MAE",
            "mean",
        ),
        series_count=(
            "series_id",
            "nunique",
        ),
    )
)

aggregate.to_csv(
    AGGREGATE_PATH,
    index=False,
)


# Save model-fitting metadata
metadata = {
    "dataset_id": "nn5_daily",
    "training_scope": (
        "base_train_only"
    ),
    "prediction_splits": [
        "router_train",
        "calibration",
    ],
    "test_accessed": False,
    "ridge_train_samples": int(
        len(ridge_y)
    ),
    "lightgbm_train_samples": int(
        len(lgbm_y)
    ),
    "prediction_rows": int(
        len(prediction)
    ),
    "ridge_fit_seconds": float(
        ridge_fit_seconds
    ),
    "lightgbm_fit_seconds": float(
        lgbm_fit_seconds
    ),
}

with METADATA_PATH.open(
    "w",
    encoding="utf-8",
) as handle:
    yaml.safe_dump(
        metadata,
        handle,
        sort_keys=False,
        allow_unicode=True,
    )


# Plot forecasts for series T1
example = prediction[
    prediction["series_id"] == "T1"
]

fig, axes = plt.subplots(
    2,
    1,
    figsize=(12, 7),
)

for ax, split_name in zip(
    axes,
    [
        "router_train",
        "calibration",
    ],
):
    part = example[
        example["split"] == split_name
    ]

    ax.plot(
        part["time_index"],
        part["y_true"],
        color="#222222",
        linewidth=1.4,
        label="actual",
    )
    ax.plot(
        part["time_index"],
        part["ridge_prediction"],
        color="#2E74B5",
        linewidth=1.0,
        label="ridge",
    )
    ax.plot(
        part["time_index"],
        part["lightgbm_prediction"],
        color="#E09F3E",
        linewidth=1.0,
        label="lightgbm",
    )

    ax.set_title(
        f"NN5 T1: {split_name}"
    )
    ax.set_ylabel("Value")
    ax.grid(alpha=0.2)

axes[0].legend(
    ncol=3,
    frameon=False,
)
axes[-1].set_xlabel(
    "Time index (day)"
)

fig.tight_layout()
fig.savefig(
    FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# Report the fitting results
print(
    "NN5 基础模型训练和预测全部通过"
)
print(
    "训练数据范围：仅 base_train"
)
print(
    "预测数据范围："
    "router_train + calibration"
)
print("test 是否访问：否")
print(
    "Ridge 训练样本：",
    len(ridge_y),
)
print(
    "LightGBM 训练样本：",
    len(lgbm_y),
)
print(
    "预测总行数：",
    len(prediction),
)
print("因果窗口检查：通过")
print("预测试指标：")
print(
    aggregate[
        [
            "split",
            "model",
            "mean_RMSSE",
            "mean_MASE",
            "mean_sMAPE",
        ]
    ].to_string(index=False)
)
print("预测文件：", PREDICTION_PATH)
print("逐序列指标：", PER_SERIES_PATH)
print("汇总指标：", AGGREGATE_PATH)
print("拟合记录：", METADATA_PATH)
print("预测图片：", FIGURE_PATH)
