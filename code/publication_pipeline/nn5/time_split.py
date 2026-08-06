"""按照实验配置切分 NN5 数据并检查时间泄漏。"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# Locate the repository root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tsf_reader import read_tsf  # noqa: E402


# Configure input and output paths
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nn5_daily"
    / "nn5_daily_dataset_without_missing_values.tsf"
)
CONFIG_PATH = PROJECT_ROOT / "experiment_config.yaml"

PROCESSED_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nn5_daily_long.parquet"
)
MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_split_manifest.csv"
)
FIGURE_PATH = (
    PROJECT_ROOT
    / "figures"
    / "nn5_four_stage_split.png"
)

PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)


# Load the experiment configuration
with CONFIG_PATH.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

ratios = config["split"]["chronological_ratios"]

split_names = [
    "base_train",
    "router_train",
    "calibration",
    "test",
]

# Validate the registered split order and ratios.
if list(ratios) != split_names:
    raise ValueError("experiment_config.yaml 中的切分顺序不正确")

ratio_sum = sum(float(ratios[name]) for name in split_names)

if not np.isclose(ratio_sum, 1.0):
    raise ValueError("四个切分比例之和必须等于 1")


# Load the source data
data, metadata = read_tsf(DATA_PATH)

long_frames = []
manifest_records = []


# Split each series chronologically
for _, row in data.iterrows():
    values = np.asarray(row["series_value"], dtype=float)
    n = len(values)

    # Compute the three segment boundaries.
    cut_base = int(
        n * float(ratios["base_train"])
    )

    cut_router = int(
        n
        * float(
            ratios["base_train"]
            + ratios["router_train"]
        )
    )

    cut_calibration = int(
        n
        * float(
            ratios["base_train"]
            + ratios["router_train"]
            + ratios["calibration"]
        )
    )

    # Initialize all observations as test observations.
    split = np.full(n, "test", dtype=object)

    # Assign the three preceding segments in chronological order.
    split[:cut_base] = "base_train"
    split[cut_base:cut_router] = "router_train"
    split[cut_router:cut_calibration] = "calibration"

    # Generate daily timestamps from the registered start date.
    timestamp = (
        row["start_timestamp"]
        + pd.to_timedelta(np.arange(n), unit="D")
    )

    # Convert to long format with one row per series and day.
    frame = pd.DataFrame(
        {
            "dataset_id": "nn5_daily",
            "series_id": row["series_name"],
            "time_index": np.arange(n, dtype=np.int64),
            "timestamp": timestamp,
            "value": values,
            "split": split,
        }
    )

    long_frames.append(frame)

    # Record the boundaries for this series.
    manifest_records.append(
        {
            "dataset_id": "nn5_daily",
            "series_id": row["series_name"],
            "total_length": n,
            "base_train_start": 0,
            "base_train_end": cut_base - 1,
            "base_train_count": cut_base,
            "router_train_start": cut_base,
            "router_train_end": cut_router - 1,
            "router_train_count": cut_router - cut_base,
            "calibration_start": cut_router,
            "calibration_end": cut_calibration - 1,
            "calibration_count": cut_calibration - cut_router,
            "test_start": cut_calibration,
            "test_end": n - 1,
            "test_count": n - cut_calibration,
        }
    )


# Combine all 111 series
long_data = pd.concat(long_frames, ignore_index=True)
manifest = pd.DataFrame(manifest_records)


# Verify the absence of temporal leakage
expected_order = {
    name: number
    for number, name in enumerate(split_names)
}

observed_order = long_data["split"].map(expected_order)

if observed_order.isna().any():
    raise ValueError("发现未知的切分名称")

for series_id, group in long_data.groupby(
    "series_id",
    sort=False,
):
    # Time indices must be strictly increasing.
    if not group["time_index"].is_monotonic_increasing:
        raise ValueError(
            f"{series_id} 的时间索引没有递增"
        )

    # The four segments must not overlap.
    group_order = observed_order.loc[group.index]

    if not group_order.is_monotonic_increasing:
        raise ValueError(
            f"{series_id} 的切分顺序交叉，存在泄漏风险"
        )

    # Every series must contain all four segments.
    if group["split"].nunique() != 4:
        raise ValueError(
            f"{series_id} 没有完整的四个切分"
        )


# Save the processed data and split manifest
long_data.to_parquet(
    PROCESSED_PATH,
    index=False,
)

manifest.to_csv(
    MANIFEST_PATH,
    index=False,
)


# Plot the four-stage split for series T1
example = long_data[
    long_data["series_id"] == "T1"
]

colors = {
    "base_train": "#2E74B5",
    "router_train": "#6BAED6",
    "calibration": "#E09F3E",
    "test": "#9B1C1C",
}

fig, ax = plt.subplots(figsize=(12, 4.8))

for split_name in split_names:
    part = example[
        example["split"] == split_name
    ]

    ax.plot(
        part["time_index"],
        part["value"],
        color=colors[split_name],
        linewidth=1.1,
        label=split_name,
    )

ax.set_title(
    "NN5 T1: chronological 60/15/10/15 split"
)
ax.set_xlabel("Time index (day)")
ax.set_ylabel("Value")
ax.grid(alpha=0.2)
ax.legend(ncol=4, frameon=False)

fig.tight_layout()
fig.savefig(
    FIGURE_PATH,
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# Report the split results
first = manifest.iloc[0]

print("NN5 四阶段时间切分全部通过")
print("序列数量：", len(manifest))
print("长表总行数：", len(long_data))
print(
    "base_train：",
    int(first["base_train_count"]),
    "天",
)
print(
    "router_train：",
    int(first["router_train_count"]),
    "天",
)
print(
    "calibration：",
    int(first["calibration_count"]),
    "天",
)
print(
    "test：",
    int(first["test_count"]),
    "天",
)
print("防止时间泄漏检查：通过")
print("处理后数据：", PROCESSED_PATH)
print("切分清单：", MANIFEST_PATH)
print("切分图片：", FIGURE_PATH)
