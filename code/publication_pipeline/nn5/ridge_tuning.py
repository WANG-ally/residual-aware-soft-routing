"""仅在 base_train 内滚动选择 Ridge 参数。"""

from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
CONFIG_PATH = PROJECT_ROOT / "experiment_config.yaml"

DETAIL_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_ridge_rolling_validation.csv"
)
SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_ridge_tuning_summary.csv"
)
SELECTED_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_selected_ridge_params.yaml"
)
FIGURE_PATH = (
    PROJECT_ROOT
    / "figures"
    / "nn5_ridge_tuning.png"
)

for path in [
    DETAIL_PATH,
    SUMMARY_PATH,
    SELECTED_PATH,
    FIGURE_PATH,
]:
    path.parent.mkdir(parents=True, exist_ok=True)


# Load the preregistered hyperparameter grid
with CONFIG_PATH.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

seed = int(config["study"]["seed"])

windows = [
    int(item)
    for item in config[
        "preprocessing"
    ]["window_by_frequency"]["daily"]
]

alphas = [
    float(item)
    for item in config[
        "base_models"
    ]["simple"]["alpha_grid"]
]


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


# Organize the 111 series
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

if first_validation_start <= max(windows):
    raise ValueError(
        "滚动训练区无法容纳最大窗口"
    )


# Construct the training and validation matrices for one fold
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
        # Fit scaling parameters using only the training portion of each fold.
        training_values = values[:train_end]

        median = float(
            np.median(training_values)
        )
        q1, q3 = np.percentile(
            training_values,
            [25, 75],
        )
        scale = float(q3 - q1)

        if scale <= 1e-12:
            scale = 1.0

        scaled = (
            values - median
        ) / scale

        # Training targets must occur strictly before train_end.
        train_x = np.asarray(
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

        train_y = scaled[
            window:train_end
        ]

        # Validation windows predict targets only within the validation block.
        val_x = np.asarray(
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

        # Estimate the RMSSE denominator from the current fold's training data.
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

        train_x_parts.append(train_x)
        train_y_parts.append(train_y)
        val_x_parts.append(val_x)

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


# Evaluate 24 configurations over three expanding-window folds
records = []
start_time = perf_counter()

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

        if train_x.shape[1] != window:
            raise ValueError(
                "特征数量与窗口长度不一致"
            )

        for alpha in alphas:
            model = Ridge(
                alpha=alpha,
                fit_intercept=True,
            )

            model.fit(
                train_x,
                train_y,
            )

            prediction_scaled = model.predict(
                val_x
            )

            # Transform predictions back to the original scale.
            prediction_raw = (
                prediction_scaled
                * val_scales
                + val_medians
            )

            evaluation = pd.DataFrame(
                {
                    "series_id": val_series_ids,
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
                    "model": (
                        "ridge_autoregression"
                    ),
                    "fold": fold,
                    "train_start": 0,
                    "train_end": train_end - 1,
                    "validation_start": val_start,
                    "validation_end": val_end - 1,
                    "window": window,
                    "alpha": alpha,
                    "train_samples": len(train_y),
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
                    "seed": seed,
                }
            )

        print(
            f"[完成] fold={fold}/3, "
            f"window={window}"
        )


# Save fold-level results
details = pd.DataFrame(records)
details.to_csv(
    DETAIL_PATH,
    index=False,
)


# Average each configuration across the three folds
summary = (
    details
    .groupby(
        ["window", "alpha"],
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
        folds=(
            "fold",
            "nunique",
        ),
    )
    .sort_values(
        [
            "mean_RMSSE",
            "window",
            "alpha",
        ]
    )
    .reset_index(drop=True)
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)


# Select the configuration with the lowest RMSSE
best = summary.iloc[0]

selected = {
    "dataset_id": "nn5_daily",
    "model": "ridge_autoregression",
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
    "selected_alpha": float(
        best["alpha"]
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


# Plot the tuning results
fig, ax = plt.subplots(
    figsize=(9, 5.5)
)

for window in windows:
    part = (
        summary[
            summary["window"] == window
        ]
        .sort_values("alpha")
    )

    ax.plot(
        part["alpha"],
        part["mean_RMSSE"],
        marker="o",
        label=f"window={window}",
    )

ax.set_xscale("log")
ax.set_title(
    "NN5 Ridge tuning within base_train"
)
ax.set_xlabel(
    "Ridge alpha (log scale)"
)
ax.set_ylabel(
    "Mean series RMSSE (lower is better)"
)
ax.grid(alpha=0.2)
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig(
    FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# Report the selected configuration
elapsed = perf_counter() - start_time

print()
print("NN5 Ridge 滚动调参全部通过")
print("调参数据范围：仅 base_train")
print("滚动验证折数：", N_FOLDS)
print("参数组合数量：", len(summary))
print(
    "最佳窗口：",
    int(best["window"]),
)
print(
    "最佳 alpha：",
    float(best["alpha"]),
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
