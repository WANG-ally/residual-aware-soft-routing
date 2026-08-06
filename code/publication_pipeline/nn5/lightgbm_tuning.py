"""仅在 base_train 内滚动选择 LightGBM 参数。"""

from itertools import product
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
import yaml


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nn5_daily_long.parquet"
)
CONFIG_PATH = PROJECT_ROOT / "experiment_config.yaml"

DETAIL_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_lightgbm_rolling_validation.csv"
)
SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_lightgbm_tuning_summary.csv"
)
SELECTED_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_selected_lightgbm_params.yaml"
)
FIGURE_PATH = (
    PROJECT_ROOT
    / "figures"
    / "nn5_lightgbm_tuning.png"
)

for path in [
    DETAIL_PATH,
    SUMMARY_PATH,
    SELECTED_PATH,
    FIGURE_PATH,
]:
    path.parent.mkdir(parents=True, exist_ok=True)


# Load the experiment configuration
with CONFIG_PATH.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

seed = int(config["study"]["seed"])

windows = [
    int(item)
    for item in config[
        "preprocessing"
    ]["window_by_frequency"]["daily"]
]

black_box = config[
    "base_models"
]["black_box"]

num_leaves_grid = [
    int(item)
    for item in black_box["num_leaves"]
]
learning_rate_grid = [
    float(item)
    for item in black_box["learning_rate"]
]
n_estimators_grid = [
    int(item)
    for item in black_box["n_estimators"]
]
feature_fraction_grid = [
    float(item)
    for item in black_box["feature_fraction"]
]

parameter_grid = list(
    product(
        num_leaves_grid,
        learning_rate_grid,
        n_estimators_grid,
        feature_fraction_grid,
    )
)


# Define the expanding-window validation protocol
N_FOLDS = 3
VALIDATION_SIZE = 28
SEASONAL_PERIOD = 7


# Load base_train only
base_data = pd.read_parquet(
    DATA_PATH,
    columns=[
        "series_id",
        "time_index",
        "value",
        "split",
    ],
    filters=[
        ("split", "==", "base_train")
    ],
)

if (
    base_data.empty
    or set(base_data["split"])
    != {"base_train"}
):
    raise ValueError(
        "调参数据必须只包含 base_train"
    )


# Organize the time series
series_values = {}

for series_id, group in base_data.groupby(
    "series_id",
    sort=False,
):
    group = group.sort_values("time_index")

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

    series_values[series_id] = group[
        "value"
    ].to_numpy(dtype=float)

lengths = {
    len(values)
    for values in series_values.values()
}

if lengths != {474}:
    raise ValueError(
        f"base_train 长度异常：{sorted(lengths)}"
    )

base_length = lengths.pop()

first_validation_start = (
    base_length
    - N_FOLDS * VALIDATION_SIZE
)


# Construct the data for one validation fold
def make_fold_arrays(
    window,
    train_end,
    val_start,
    val_end,
):
    train_x_parts = []
    train_y_parts = []

    val_x_parts = []
    val_y_raw_parts = []
    val_median_parts = []
    val_scale_parts = []
    val_denom_parts = []
    val_series_parts = []

    for series_id, values in series_values.items():
        training_values = values[:train_end]

        # Fit scaling parameters using only the training portion of each fold.
        median = float(
            np.median(training_values)
        )
        q1, q3 = np.percentile(
            training_values,
            [25, 75],
        )
        scale = max(
            float(q3 - q1),
            1e-12,
        )

        scaled = (
            values - median
        ) / scale

        train_x_parts.append(
            np.asarray(
                [
                    scaled[
                        t - window:t
                    ][::-1]
                    for t in range(
                        window,
                        train_end,
                    )
                ],
                dtype=float,
            )
        )

        train_y_parts.append(
            scaled[window:train_end]
        )

        val_x_parts.append(
            np.asarray(
                [
                    scaled[
                        t - window:t
                    ][::-1]
                    for t in range(
                        val_start,
                        val_end,
                    )
                ],
                dtype=float,
            )
        )

        seasonal_differences = (
            training_values[
                SEASONAL_PERIOD:
            ]
            - training_values[
                :-SEASONAL_PERIOD
            ]
        )

        denominator = float(
            np.mean(
                seasonal_differences ** 2
            )
        )

        if denominator <= 1e-12:
            raise ValueError(
                f"{series_id} 的 RMSSE 分母为零"
            )

        validation_count = (
            val_end - val_start
        )

        val_y_raw_parts.append(
            values[val_start:val_end]
        )
        val_median_parts.append(
            np.full(
                validation_count,
                median,
            )
        )
        val_scale_parts.append(
            np.full(
                validation_count,
                scale,
            )
        )
        val_denom_parts.append(
            np.full(
                validation_count,
                denominator,
            )
        )
        val_series_parts.append(
            np.full(
                validation_count,
                series_id,
                dtype=object,
            )
        )

    return (
        np.vstack(train_x_parts),
        np.concatenate(train_y_parts),
        np.vstack(val_x_parts),
        np.concatenate(val_y_raw_parts),
        np.concatenate(val_median_parts),
        np.concatenate(val_scale_parts),
        np.concatenate(val_denom_parts),
        np.concatenate(val_series_parts),
    )


# Fit the 192 registered model configurations
records = []
overall_start = perf_counter()

for fold in range(1, N_FOLDS + 1):
    val_start = (
        first_validation_start
        + (fold - 1) * VALIDATION_SIZE
    )
    val_end = (
        val_start + VALIDATION_SIZE
    )
    train_end = val_start

    for window in windows:
        (
            train_x,
            train_y,
            val_x,
            val_y_raw,
            val_medians,
            val_scales,
            val_denominators,
            val_series_ids,
        ) = make_fold_arrays(
            window,
            train_end,
            val_start,
            val_end,
        )

        for (
            num_leaves,
            learning_rate,
            n_estimators,
            feature_fraction,
        ) in parameter_grid:
            fit_start = perf_counter()

            model = LGBMRegressor(
                objective="regression_l2",
                num_leaves=num_leaves,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                feature_fraction=feature_fraction,
                random_state=seed,
                feature_fraction_seed=seed,
                data_random_seed=seed,
                deterministic=True,
                force_col_wise=True,
                n_jobs=-1,
                verbosity=-1,
            )

            model.fit(
                train_x,
                train_y,
            )

            fit_seconds = (
                perf_counter() - fit_start
            )

            # Use the low-level Booster API to avoid feature-name warnings.
            prediction_scaled = (
                model
                .booster_
                .predict(val_x)
            )

            prediction_raw = (
                prediction_scaled
                * val_scales
                + val_medians
            )

            evaluation = pd.DataFrame(
                {
                    "series_id": (
                        val_series_ids
                    ),
                    "squared_error": (
                        prediction_raw
                        - val_y_raw
                    ) ** 2,
                    "denominator": (
                        val_denominators
                    ),
                }
            )

            by_series = (
                evaluation
                .groupby(
                    "series_id",
                    sort=False,
                )
                .agg(
                    mean_squared_error=(
                        "squared_error",
                        "mean",
                    ),
                    denominator=(
                        "denominator",
                        "first",
                    ),
                )
            )

            by_series["RMSSE"] = np.sqrt(
                by_series[
                    "mean_squared_error"
                ]
                / by_series["denominator"]
            )

            records.append(
                {
                    "dataset_id": "nn5_daily",
                    "model": "lightgbm",
                    "fold": fold,
                    "train_start": 0,
                    "train_end": train_end - 1,
                    "validation_start": (
                        val_start
                    ),
                    "validation_end": (
                        val_end - 1
                    ),
                    "window": window,
                    "num_leaves": (
                        num_leaves
                    ),
                    "learning_rate": (
                        learning_rate
                    ),
                    "n_estimators": (
                        n_estimators
                    ),
                    "feature_fraction": (
                        feature_fraction
                    ),
                    "train_samples": (
                        len(train_y)
                    ),
                    "validation_samples": (
                        len(val_y_raw)
                    ),
                    "mean_series_RMSSE": float(
                        by_series[
                            "RMSSE"
                        ].mean()
                    ),
                    "median_series_RMSSE": float(
                        by_series[
                            "RMSSE"
                        ].median()
                    ),
                    "raw_RMSE": float(
                        np.sqrt(
                            np.mean(
                                (
                                    prediction_raw
                                    - val_y_raw
                                ) ** 2
                            )
                        )
                    ),
                    "fit_seconds": (
                        fit_seconds
                    ),
                    "seed": seed,
                }
            )

        print(
            f"[完成] fold={fold}/3, "
            f"window={window}, "
            f"configurations="
            f"{len(parameter_grid)}"
        )


# Save fold-level results
details = pd.DataFrame(records)

details.to_csv(
    DETAIL_PATH,
    index=False,
)


# Aggregate each configuration across the three folds
group_columns = [
    "window",
    "num_leaves",
    "learning_rate",
    "n_estimators",
    "feature_fraction",
]

summary = (
    details
    .groupby(
        group_columns,
        as_index=False,
    )
    .agg(
        mean_RMSSE=(
            "mean_series_RMSSE",
            "mean",
        ),
        std_RMSSE=(
            "mean_series_RMSSE",
            "std",
        ),
        mean_raw_RMSE=(
            "raw_RMSE",
            "mean",
        ),
        mean_fit_seconds=(
            "fit_seconds",
            "mean",
        ),
        folds=(
            "fold",
            "nunique",
        ),
    )
    .sort_values(
        [
            "mean_RMSSE",
            *group_columns,
        ]
    )
    .reset_index(drop=True)
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)


# Save the selected configuration
best = summary.iloc[0]

selected = {
    "dataset_id": "nn5_daily",
    "model": "lightgbm",
    "selection_scope": "base_train_only",
    "selection_metric": "mean_series_RMSSE",
    "seasonal_period": SEASONAL_PERIOD,
    "rolling_folds": N_FOLDS,
    "validation_size_per_fold": (
        VALIDATION_SIZE
    ),
    "selected_window": int(
        best["window"]
    ),
    "num_leaves": int(
        best["num_leaves"]
    ),
    "learning_rate": float(
        best["learning_rate"]
    ),
    "n_estimators": int(
        best["n_estimators"]
    ),
    "feature_fraction": float(
        best["feature_fraction"]
    ),
    "mean_validation_RMSSE": float(
        best["mean_RMSSE"]
    ),
    "seed": seed,
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
        f"w={int(row.window)}, "
        f"leaves={int(row.num_leaves)}, "
        f"lr={row.learning_rate:g}, "
        f"trees={int(row.n_estimators)}, "
        f"ff={row.feature_fraction:g}"
    )
    for row in top.itertuples()
]

fig, ax = plt.subplots(
    figsize=(11, 7)
)

ax.barh(
    labels,
    top["mean_RMSSE"],
    color="#2E74B5",
)

ax.set_title(
    "NN5 LightGBM: top 15 "
    "base_train configurations"
)
ax.set_xlabel(
    "Mean series RMSSE "
    "(lower is better)"
)
ax.grid(axis="x", alpha=0.2)

fig.tight_layout()
fig.savefig(
    FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# Report the selected configuration
elapsed = (
    perf_counter() - overall_start
)

print()
print("NN5 LightGBM 滚动调参全部通过")
print("调参数据范围：仅 base_train")
print("滚动验证折数：", N_FOLDS)
print("参数组合数量：", len(summary))
print("实际模型拟合次数：", len(details))
print(
    "最佳窗口：",
    int(best["window"]),
)
print(
    "最佳 num_leaves：",
    int(best["num_leaves"]),
)
print(
    "最佳 learning_rate：",
    float(best["learning_rate"]),
)
print(
    "最佳 n_estimators：",
    int(best["n_estimators"]),
)
print(
    "最佳 feature_fraction：",
    float(best["feature_fraction"]),
)
print(
    "最佳平均 RMSSE：",
    f"{float(best['mean_RMSSE']):.6f}",
)
print(
    "运行秒数：",
    f"{elapsed:.2f}",
)
print("逐折结果：", DETAIL_PATH)
print("汇总结果：", SUMMARY_PATH)
print("选定参数：", SELECTED_PATH)
print("调参图片：", FIGURE_PATH)
