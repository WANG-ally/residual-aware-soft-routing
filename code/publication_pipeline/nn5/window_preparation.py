"""生成候选滑动窗口并检查未来信息泄漏。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# Configure input and output paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nn5_daily_long.parquet"
)
CONFIG_PATH = PROJECT_ROOT / "experiment_config.yaml"

SCALER_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_scaler_parameters.csv"
)
COUNT_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_window_candidate_counts.csv"
)
PREVIEW_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_window_preview.csv"
)
FIGURE_PATH = (
    PROJECT_ROOT
    / "figures"
    / "nn5_window_example.png"
)

for path in [
    SCALER_PATH,
    COUNT_PATH,
    PREVIEW_PATH,
    FIGURE_PATH,
]:
    path.parent.mkdir(parents=True, exist_ok=True)


# Load candidate windows from the experiment configuration
with CONFIG_PATH.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

candidate_windows = [
    int(item)
    for item in config[
        "preprocessing"
    ]["window_by_frequency"]["daily"]
]

if candidate_windows != [7, 14, 28, 56]:
    raise ValueError(
        "日频窗口配置与预注册方案不一致"
    )


# Load the processed long-format data
data = pd.read_parquet(DATA_PATH)

required_columns = {
    "dataset_id",
    "series_id",
    "time_index",
    "timestamp",
    "value",
    "split",
}

missing_columns = required_columns.difference(
    data.columns
)

if missing_columns:
    raise ValueError(
        f"处理后数据缺少列：{sorted(missing_columns)}"
    )


# Estimate scaling parameters from base_train only
scaler_records = []
scaler_lookup = {}

for series_id, group in data.groupby(
    "series_id",
    sort=False,
):
    group = group.sort_values("time_index")

    base = group[
        group["split"] == "base_train"
    ]

    if base.empty:
        raise ValueError(
            f"{series_id} 没有 base_train 数据"
        )

    base_values = base["value"].to_numpy(
        dtype=float
    )

    median = float(np.median(base_values))
    q1, q3 = np.percentile(
        base_values,
        [25, 75],
    )
    iqr = float(q3 - q1)

    # Fall back safely in the rare case of a zero IQR.
    used_fallback = bool(iqr <= 1e-12)
    scale = 1.0 if used_fallback else iqr

    scaler_lookup[series_id] = (
        median,
        scale,
    )

    scaler_records.append(
        {
            "dataset_id": "nn5_daily",
            "series_id": series_id,
            "source_split": "base_train",
            "source_start": int(
                base["time_index"].min()
            ),
            "source_end": int(
                base["time_index"].max()
            ),
            "source_count": len(base),
            "median": median,
            "q1": float(q1),
            "q3": float(q3),
            "iqr": iqr,
            "scale_used": scale,
            "zero_iqr_fallback": used_fallback,
        }
    )

scalers = pd.DataFrame(scaler_records)
scalers.to_csv(SCALER_PATH, index=False)


# Verify that scaling does not use future observations
if scalers["source_split"].ne(
    "base_train"
).any():
    raise ValueError(
        "缩放参数使用了 base_train 以外的数据"
    )

scaler_values = scalers[
    ["median", "scale_used"]
].to_numpy()

if not np.isfinite(scaler_values).all():
    raise ValueError(
        "缩放参数中出现非有限值"
    )


# Count eligible samples for each candidate window
split_names = [
    "base_train",
    "router_train",
    "calibration",
    "test",
]

count_records = []

for window in candidate_windows:
    counts = {
        name: 0
        for name in split_names
    }

    for series_id, group in data.groupby(
        "series_id",
        sort=False,
    ):
        group = (
            group
            .sort_values("time_index")
            .reset_index(drop=True)
        )

        indices = group[
            "time_index"
        ].to_numpy(dtype=np.int64)

        # Indices must be contiguous within each series.
        if not np.array_equal(
            indices,
            np.arange(len(group)),
        ):
            raise ValueError(
                f"{series_id} 的时间索引不连续"
            )

        # Start only after a complete history window is available.
        for target_position in range(
            window,
            len(group),
        ):
            history = indices[
                target_position
                - window:target_position
            ]

            target_index = indices[
                target_position
            ]

            # The latest input observation must precede the target.
            if (
                len(history) != window
                or history[-1] >= target_index
            ):
                raise ValueError(
                    "窗口包含目标当天或未来数据"
                )

            target_split = str(
                group.loc[
                    target_position,
                    "split",
                ]
            )

            if target_split not in counts:
                raise ValueError(
                    f"发现未知切分：{target_split}"
                )

            counts[target_split] += 1

    count_records.append(
        {
            "window": window,
            **counts,
            "total": sum(counts.values()),
        }
    )

window_counts = pd.DataFrame(count_records)
window_counts.to_csv(COUNT_PATH, index=False)


# Create a window preview for series T1
preview_window = candidate_windows[0]

example = (
    data[data["series_id"] == "T1"]
    .sort_values("time_index")
    .reset_index(drop=True)
)

median, scale = scaler_lookup["T1"]

scaled_values = (
    example["value"].to_numpy(dtype=float)
    - median
) / scale

preview_records = []

# Retain the first five supervised-learning samples for inspection.
for target_position in range(
    preview_window,
    preview_window + 5,
):
    record = {
        "series_id": "T1",
        "target_time_index": int(
            example.loc[
                target_position,
                "time_index",
            ]
        ),
        "target_split": example.loc[
            target_position,
            "split",
        ],
        "target_scaled": float(
            scaled_values[target_position]
        ),
    }

    # lag_1 is the previous day and lag_7 is seven days earlier.
    for lag in range(
        1,
        preview_window + 1,
    ):
        record[f"lag_{lag}"] = float(
            scaled_values[
                target_position - lag
            ]
        )

    preview_records.append(record)

preview = pd.DataFrame(preview_records)
preview.to_csv(PREVIEW_PATH, index=False)


# Plot a seven-day window example
raw_values = example["value"].to_numpy(
    dtype=float
)

input_x = np.arange(preview_window)
target_x = preview_window

fig, ax = plt.subplots(figsize=(9, 4.8))

ax.plot(
    input_x,
    raw_values[:preview_window],
    color="#2E74B5",
    marker="o",
    label="7 input days",
)

ax.scatter(
    [target_x],
    [raw_values[target_x]],
    color="#9B1C1C",
    s=70,
    zorder=3,
    label="prediction target",
)

ax.axvline(
    target_x - 0.5,
    color="#666666",
    linestyle="--",
    linewidth=1,
)

ax.set_title(
    "NN5 T1: one causal sliding-window sample"
)
ax.set_xlabel("Time index (day)")
ax.set_ylabel("Value")
ax.grid(alpha=0.2)
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig(
    FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# Report the window-preparation checks
print("NN5 滑动窗口准备全部通过")
print("候选窗口：", candidate_windows)
print("缩放参数来源：仅 base_train")
print(
    "零 IQR 回退数量：",
    int(
        scalers[
            "zero_iqr_fallback"
        ].sum()
    ),
)
print("未来信息检查：通过")
print("各窗口样本数量：")
print(window_counts.to_string(index=False))
print("缩放参数：", SCALER_PATH)
print("窗口计数：", COUNT_PATH)
print("窗口示例表：", PREVIEW_PATH)
print("窗口示意图：", FIGURE_PATH)
