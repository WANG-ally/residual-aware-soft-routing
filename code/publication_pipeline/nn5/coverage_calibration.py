"""校准路由阈值并选择在线覆盖率控制器。"""

from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CONFIG_PATH = (
    PROJECT_ROOT
    / "experiment_config.yaml"
)

SCORE_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_calibration_router_scores.parquet"
)

LOCAL_PARAMETER_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_selected_local_weighting_params.yaml"
)

TUNING_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_coverage_controller_tuning.csv"
)

TRACE_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_calibration_controller_trace.csv"
)

SELECTED_PATH = (
    PROJECT_ROOT
    / "results"
    / "nn5_selected_coverage_controller.yaml"
)

FIGURE_PATH = (
    PROJECT_ROOT
    / "figures"
    / "nn5_coverage_calibration.png"
)

for path in [
    TUNING_PATH,
    TRACE_PATH,
    SELECTED_PATH,
    FIGURE_PATH,
]:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Load the preregistered controller configuration
# ============================================================

with CONFIG_PATH.open(
    "r",
    encoding="utf-8",
) as handle:
    config = yaml.safe_load(handle)

with LOCAL_PARAMETER_PATH.open(
    "r",
    encoding="utf-8",
) as handle:
    local_parameters = yaml.safe_load(handle)

if (
    local_parameters["selection_scope"]
    != "router_train_only"
):
    raise ValueError(
        "路由模型参数必须仅由 router_train 选择"
    )

seed = int(
    config["study"]["seed"]
)

primary_coverage = float(
    config["study"]["primary_target_coverage"]
)

sensitivity_coverages = [
    float(value)
    for value in config[
        "study"
    ]["sensitivity_target_coverages"]
]

eta_grid = [
    float(value)
    for value in config[
        "coverage_controller"
    ]["eta_grid"]
]

bias_limit = float(
    config[
        "coverage_controller"
    ]["bias_limit"]
)

allowed_violation = float(
    config[
        "coverage_controller"
    ]["allowed_absolute_violation"]
)

all_target_coverages = sorted(
    set(
        [primary_coverage]
        + sensitivity_coverages
    )
)


# ============================================================
# Load calibration data only
# ============================================================

calibration = pd.read_parquet(
    SCORE_PATH
)

if calibration.empty:
    raise ValueError(
        "没有读取到 calibration 路由概率"
    )

if set(calibration["split"]) != {
    "calibration"
}:
    raise ValueError(
        "校准文件混入了其他数据段"
    )

required_columns = [
    "dataset_id",
    "series_id",
    "time_index",
    "split",
    "y_true",
    "ridge_prediction",
    "lightgbm_prediction",
    "seasonal_naive_mae_scale",
    "black_box_probability",
]

missing_columns = [
    column
    for column in required_columns
    if column not in calibration.columns
]

if missing_columns:
    raise ValueError(
        f"校准文件缺少字段：{missing_columns}"
    )

if calibration[
    required_columns
].isna().any().any():
    raise ValueError(
        "校准数据中存在缺失值"
    )

probability = calibration[
    "black_box_probability"
].to_numpy(dtype=float)

if (
    np.any(probability < 0.0)
    or np.any(probability > 1.0)
):
    raise ValueError(
        "路由概率必须位于0到1之间"
    )

if np.any(
    calibration[
        "seasonal_naive_mae_scale"
    ].to_numpy(dtype=float)
    <= 0.0
):
    raise ValueError(
        "缩放分母必须大于0"
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

unique_times = np.sort(
    calibration["time_index"].unique()
)

series_count_per_time = (
    calibration
    .groupby("time_index")
    .size()
)

if (
    len(calibration) != 8769
    or len(unique_times) != 79
    or unique_times[0] != 593
    or unique_times[-1] != 671
):
    raise ValueError(
        "calibration 数据范围异常"
    )

if not (
    series_count_per_time == 111
).all():
    raise ValueError(
        "每个校准时点应包含111条序列"
    )


# ============================================================
# Compute the static usage threshold
# ============================================================

threshold_records = []

for target_coverage in all_target_coverages:

    threshold = float(
        np.quantile(
            probability,
            target_coverage,
        )
    )

    achieved_coverage = float(
        np.mean(
            probability < threshold
        )
    )

    threshold_records.append(
        {
            "target_simple_coverage": (
                target_coverage
            ),
            "base_threshold": threshold,
            "achieved_static_coverage": (
                achieved_coverage
            ),
        }
    )

primary_threshold = next(
    item["base_threshold"]
    for item in threshold_records
    if np.isclose(
        item[
            "target_simple_coverage"
        ],
        primary_coverage,
    )
)


# ============================================================
# Define the online model-usage controller
# ============================================================

def simulate_controller(
    eta,
    base_threshold,
    target_coverage,
):
    """
    按时间点模拟覆盖率控制器。

    每个时间点先做预测选择，再根据当期实际选择比例
    更新下一时间点的阈值偏移量。更新过程不使用真实y值。
    """

    bias = 0.0
    cumulative_simple_count = 0
    cumulative_row_count = 0
    total_scaled_loss = 0.0
    records = []

    for time_index, batch in calibration.groupby(
        "time_index",
        sort=True,
    ):
        batch_probability = batch[
            "black_box_probability"
        ].to_numpy(dtype=float)

        threshold_before_decision = float(
            np.clip(
                base_threshold + bias,
                0.0,
                1.0,
            )
        )

        # True indicates selection of the simpler Ridge model.
        use_simple_model = (
            batch_probability
            < threshold_before_decision
        )

        simple_count = int(
            np.sum(use_simple_model)
        )

        batch_rows = len(batch)

        batch_coverage = float(
            simple_count / batch_rows
        )

        selected_prediction = np.where(
            use_simple_model,
            batch[
                "ridge_prediction"
            ].to_numpy(dtype=float),
            batch[
                "lightgbm_prediction"
            ].to_numpy(dtype=float),
        )

        scaled_squared_error = (
            (
                batch[
                    "y_true"
                ].to_numpy(dtype=float)
                - selected_prediction
            )
            / batch[
                "seasonal_naive_mae_scale"
            ].to_numpy(dtype=float)
        ) ** 2

        batch_scaled_loss = float(
            np.mean(
                scaled_squared_error
            )
        )

        cumulative_simple_count += (
            simple_count
        )

        cumulative_row_count += (
            batch_rows
        )

        total_scaled_loss += float(
            np.sum(
                scaled_squared_error
            )
        )

        cumulative_coverage = float(
            cumulative_simple_count
            / cumulative_row_count
        )

        # Increase the bias when Ridge usage is below target, raising the next threshold.
        # Decrease it when Ridge usage is above target, lowering the next threshold.
        next_bias = float(
            np.clip(
                bias
                + eta
                * (
                    target_coverage
                    - batch_coverage
                ),
                -bias_limit,
                bias_limit,
            )
        )

        records.append(
            {
                "dataset_id": "nn5_daily",
                "time_index": int(
                    time_index
                ),
                "eta": float(eta),
                "base_threshold": float(
                    base_threshold
                ),
                "bias_before_decision": float(
                    bias
                ),
                "effective_threshold": (
                    threshold_before_decision
                ),
                "simple_count": simple_count,
                "batch_rows": batch_rows,
                "simple_coverage": (
                    batch_coverage
                ),
                "cumulative_simple_coverage": (
                    cumulative_coverage
                ),
                "batch_scaled_loss": (
                    batch_scaled_loss
                ),
                "bias_after_update": (
                    next_bias
                ),
            }
        )

        bias = next_bias

    trace = pd.DataFrame(records)

    rolling_coverage = (
        trace["simple_coverage"]
        .rolling(
            window=14,
            min_periods=14,
        )
        .mean()
    )

    valid_rolling_coverage = (
        rolling_coverage.dropna()
    )

    if valid_rolling_coverage.empty:
        worst_14_day_violation = float(
            abs(
                trace[
                    "simple_coverage"
                ].mean()
                - target_coverage
            )
        )
    else:
        worst_14_day_violation = float(
            np.max(
                np.abs(
                    valid_rolling_coverage
                    - target_coverage
                )
            )
        )

    overall_coverage = float(
        cumulative_simple_count
        / cumulative_row_count
    )

    mean_scaled_loss = float(
        total_scaled_loss
        / cumulative_row_count
    )

    summary = {
        "eta": float(eta),
        "base_threshold": float(
            base_threshold
        ),
        "target_simple_coverage": float(
            target_coverage
        ),
        "achieved_simple_coverage": (
            overall_coverage
        ),
        "absolute_coverage_violation": float(
            abs(
                overall_coverage
                - target_coverage
            )
        ),
        "mean_scaled_loss": (
            mean_scaled_loss
        ),
        "worst_14_day_violation": (
            worst_14_day_violation
        ),
        "mean_daily_absolute_violation": float(
            np.mean(
                np.abs(
                    trace[
                        "simple_coverage"
                    ]
                    - target_coverage
                )
            )
        ),
        "final_bias": float(bias),
        "maximum_absolute_bias": float(
            np.max(
                np.abs(
                    trace[
                        "bias_before_decision"
                    ]
                )
            )
        ),
        "calibration_rows": int(
            cumulative_row_count
        ),
        "calibration_time_points": int(
            len(trace)
        ),
    }

    return summary, trace


# ============================================================
# Compare the four registered controller step sizes
# ============================================================

start_time = perf_counter()

candidate_summaries = []

for eta in eta_grid:
    summary, _ = simulate_controller(
        eta=eta,
        base_threshold=primary_threshold,
        target_coverage=primary_coverage,
    )

    summary["feasible"] = bool(
        summary[
            "absolute_coverage_violation"
        ]
        <= allowed_violation
    )

    candidate_summaries.append(
        summary
    )

tuning = pd.DataFrame(
    candidate_summaries
)

tuning = (
    tuning
    .sort_values("eta")
    .reset_index(drop=True)
)

tuning.to_csv(
    TUNING_PATH,
    index=False,
)


# ============================================================
# Select eta under the registered usage constraint
# ============================================================

feasible_candidates = tuning[
    tuning["feasible"]
].copy()

if feasible_candidates.empty:
    selection_status = (
        "no_feasible_candidate_"
        "minimum_violation_fallback"
    )

    ranked_candidates = (
        tuning
        .sort_values(
            [
                "absolute_coverage_violation",
                "mean_scaled_loss",
                "worst_14_day_violation",
                "eta",
            ]
        )
    )
else:
    selection_status = (
        "feasible_minimum_forecast_loss"
    )

    ranked_candidates = (
        feasible_candidates
        .sort_values(
            [
                "mean_scaled_loss",
                "worst_14_day_violation",
                "eta",
            ]
        )
    )

best = ranked_candidates.iloc[0]

selected_eta = float(
    best["eta"]
)

selected_summary, selected_trace = (
    simulate_controller(
        eta=selected_eta,
        base_threshold=primary_threshold,
        target_coverage=primary_coverage,
    )
)

selected_trace.to_csv(
    TRACE_PATH,
    index=False,
)


# ============================================================
# Compute the static-threshold baseline
# ============================================================

static_summary, static_trace = (
    simulate_controller(
        eta=0.0,
        base_threshold=primary_threshold,
        target_coverage=primary_coverage,
    )
)


# ============================================================
# Save the selected controller parameters
# ============================================================

selected_parameters = {
    "dataset_id": "nn5_daily",
    "fit_scope": "calibration_only",
    "test_accessed": False,
    "probability_semantics": (
        "probability_that_lightgbm_is_preferred"
    ),
    "decision_rule": (
        "use_ridge_if_probability_is_below_"
        "effective_threshold"
    ),
    "threshold_source": (
        "calibration_quantile"
    ),
    "primary_target_simple_coverage": float(
        primary_coverage
    ),
    "primary_base_threshold": float(
        primary_threshold
    ),
    "selected_eta": float(
        selected_eta
    ),
    "bias_limit": float(
        bias_limit
    ),
    "allowed_absolute_violation": float(
        allowed_violation
    ),
    "update_rule": (
        "bias_next=clip("
        "bias+eta*(target-batch_coverage),"
        "-bias_limit,bias_limit)"
    ),
    "update_information": (
        "routing_decisions_only_no_true_target_needed"
    ),
    "selection_status": (
        selection_status
    ),
    "selection_metric": (
        "minimum_scaled_loss_among_"
        "coverage_feasible_candidates"
    ),
    "candidate_eta_values": [
        float(value)
        for value in eta_grid
    ],
    "calibration_rows": int(
        len(calibration)
    ),
    "calibration_time_points": int(
        len(unique_times)
    ),
    "achieved_calibration_simple_coverage": float(
        selected_summary[
            "achieved_simple_coverage"
        ]
    ),
    "calibration_absolute_violation": float(
        selected_summary[
            "absolute_coverage_violation"
        ]
    ),
    "calibration_mean_scaled_loss": float(
        selected_summary[
            "mean_scaled_loss"
        ]
    ),
    "calibration_worst_14_day_violation": float(
        selected_summary[
            "worst_14_day_violation"
        ]
    ),
    "static_calibration_simple_coverage": float(
        static_summary[
            "achieved_simple_coverage"
        ]
    ),
    "static_calibration_mean_scaled_loss": float(
        static_summary[
            "mean_scaled_loss"
        ]
    ),
    "thresholds": [
        {
            "target_simple_coverage": float(
                item[
                    "target_simple_coverage"
                ]
            ),
            "base_threshold": float(
                item["base_threshold"]
            ),
            "achieved_static_calibration_coverage": float(
                item[
                    "achieved_static_coverage"
                ]
            ),
        }
        for item in threshold_records
    ],
    "seed": int(seed),
}

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
# Plot calibration diagnostics
# ============================================================

figure, axes = plt.subplots(
    1,
    3,
    figsize=(16, 4.8),
)

# Calibration-score distribution and threshold
axes[0].hist(
    calibration[
        "black_box_probability"
    ],
    bins=35,
    color="#4c78a8",
    alpha=0.80,
)

threshold_colors = [
    "#59a14f",
    "#e15759",
    "#f28e2b",
]

for item, color in zip(
    threshold_records,
    threshold_colors,
):
    axes[0].axvline(
        item["base_threshold"],
        color=color,
        linestyle="--",
        linewidth=1.8,
        label=(
            f"coverage="
            f"{item['target_simple_coverage']:.1f}"
        ),
    )

axes[0].set_title(
    "Calibration score thresholds"
)

axes[0].set_xlabel(
    "Predicted probability of LightGBM"
)

axes[0].set_ylabel(
    "Count"
)

axes[0].legend()


# Daily Ridge-usage trajectory
rolling_14 = (
    selected_trace[
        "simple_coverage"
    ]
    .rolling(
        window=14,
        min_periods=1,
    )
    .mean()
)

axes[1].plot(
    selected_trace[
        "time_index"
    ],
    selected_trace[
        "simple_coverage"
    ],
    color="#bab0ac",
    alpha=0.60,
    linewidth=1.0,
    label="Daily coverage",
)

axes[1].plot(
    selected_trace[
        "time_index"
    ],
    rolling_14,
    color="#e15759",
    linewidth=2.0,
    label="14-day rolling coverage",
)

axes[1].axhline(
    primary_coverage,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="Target coverage",
)

axes[1].set_title(
    "Adaptive simple-model coverage"
)

axes[1].set_xlabel(
    "Time index"
)

axes[1].set_ylabel(
    "Simple-model coverage"
)

axes[1].set_ylim(
    0.0,
    1.0,
)

axes[1].legend()


# Adaptive effective-threshold trajectory
axes[2].plot(
    selected_trace[
        "time_index"
    ],
    selected_trace[
        "effective_threshold"
    ],
    color="#59a14f",
    linewidth=2.0,
    label="Effective threshold",
)

axes[2].axhline(
    primary_threshold,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="Base threshold",
)

axes[2].set_title(
    "Online threshold adjustment"
)

axes[2].set_xlabel(
    "Time index"
)

axes[2].set_ylabel(
    "Probability threshold"
)

axes[2].legend()

figure.tight_layout()

figure.savefig(
    FIGURE_PATH,
    dpi=180,
    bbox_inches="tight",
)

plt.close(figure)


# ============================================================
# Validate and report the calibration results
# ============================================================

if (
    selected_summary[
        "absolute_coverage_violation"
    ]
    > allowed_violation
    and selection_status.startswith(
        "feasible"
    )
):
    raise ValueError(
        "被选控制器没有满足覆盖率约束"
    )

elapsed_seconds = (
    perf_counter()
    - start_time
)

print()
print("NN5 覆盖率控制器校准全部通过")
print("参数选择数据范围：仅 calibration")
print("测试集是否访问：否")
print("calibration 样本数量：", len(calibration))
print("calibration 时间点数量：", len(unique_times))
print("目标 Ridge 覆盖率：", primary_coverage)
print(
    "基础概率阈值：",
    f"{primary_threshold:.9f}",
)
print("候选 eta 数量：", len(eta_grid))
print("选定 eta：", selected_eta)
print(
    "自适应实际覆盖率：",
    f"{selected_summary['achieved_simple_coverage']:.6f}",
)
print(
    "绝对覆盖率偏差：",
    f"{selected_summary['absolute_coverage_violation']:.6f}",
)
print(
    "自适应平均缩放损失：",
    f"{selected_summary['mean_scaled_loss']:.6f}",
)
print(
    "静态阈值平均缩放损失：",
    f"{static_summary['mean_scaled_loss']:.6f}",
)
print(
    "最差14日覆盖率偏差：",
    f"{selected_summary['worst_14_day_violation']:.6f}",
)
print(
    "最终 bias：",
    f"{selected_summary['final_bias']:.9f}",
)
print(
    "覆盖率约束：",
    "通过"
    if selected_summary[
        "absolute_coverage_violation"
    ] <= allowed_violation
    else "未通过",
)
print(
    "运行秒数：",
    f"{elapsed_seconds:.2f}",
)
print("候选参数结果：", TUNING_PATH)
print("逐时点控制记录：", TRACE_PATH)
print("选定控制器参数：", SELECTED_PATH)
print("校准诊断图片：", FIGURE_PATH)
