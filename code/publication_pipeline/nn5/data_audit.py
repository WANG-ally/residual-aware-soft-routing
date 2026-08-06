"""读取并检查真实的 NN5 日频数据。"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Locate the repository root so the root-level TSF reader can be imported.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tsf_reader import read_tsf  # noqa: E402


# Input and output paths
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "nn5_daily"
    / "nn5_daily_dataset_without_missing_values.tsf"
)
AUDIT_PATH = PROJECT_ROOT / "results" / "nn5_series_audit.csv"
FIGURE_PATH = PROJECT_ROOT / "figures" / "nn5_first_three_series.png"

# Create output directories.
AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)


# Load the source data
data, metadata = read_tsf(DATA_PATH)

frequency = metadata["frequency"]
horizon = int(metadata["horizon"])


# Audit each series
audit_records = []

for _, row in data.iterrows():
    values = np.asarray(row["series_value"], dtype=float)
    valid_values = values[np.isfinite(values)]

    audit_records.append(
        {
            "dataset_id": "nn5_daily",
            "series_id": row["series_name"],
            "start_timestamp": row["start_timestamp"],
            "frequency": frequency,
            "horizon": horizon,
            "length": len(values),
            "missing_count": int(np.isnan(values).sum()),
            "missing_rate": float(np.isnan(values).mean()),
            "minimum": float(valid_values.min()),
            "maximum": float(valid_values.max()),
            "mean": float(valid_values.mean()),
            "standard_deviation": float(valid_values.std()),
            "zero_count": int((valid_values == 0).sum()),
            "negative_count": int((valid_values < 0).sum()),
            "is_constant": bool(np.ptp(valid_values) == 0),
        }
    )

audit = pd.DataFrame(audit_records)
audit.to_csv(AUDIT_PATH, index=False)


# Validate the audit invariants
checks = {
    "序列数量等于111": len(data) == 111,
    "频率为daily": frequency == "daily",
    "预测范围等于56": horizon == 56,
    "序列编号没有重复": data["series_name"].is_unique,
    "所有序列长度相等": audit["length"].nunique() == 1,
    "不存在缺失值": audit["missing_count"].sum() == 0,
    "不存在常数序列": audit["is_constant"].sum() == 0,
}

failed_checks = [name for name, passed in checks.items() if not passed]

if failed_checks:
    raise ValueError("数据质量检查失败：" + "、".join(failed_checks))


# Plot the first three series and mark the formal test region
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

for ax, (_, row) in zip(axes, data.head(3).iterrows()):
    values = np.asarray(row["series_value"], dtype=float)
    test_start = len(values) - horizon

    ax.plot(values, color="#1769aa", linewidth=0.8)
    ax.axvline(
        test_start,
        color="#d32f2f",
        linestyle="--",
        linewidth=1.2,
        label="Archive 56-day horizon starts",
    )
    ax.set_title(f"NN5 series {row['series_name']}")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.2)

axes[0].legend()
axes[-1].set_xlabel("Time index (day)")
fig.suptitle("NN5 data inspection: first three series", fontsize=14)
fig.tight_layout()
fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)


# Report the audit results
print("NN5 数据质量检查全部通过")
print("序列数量：", len(data))
print("每条序列长度：", int(audit["length"].iloc[0]))
print("总缺失值数量：", int(audit["missing_count"].sum()))
print("预测范围：", horizon, "天")
print("官方56天预测区起点：第", int(audit["length"].iloc[0] - horizon), "个时间点")
print("质量检查表：", AUDIT_PATH)
print("数据检查图片：", FIGURE_PATH)
