"""只使用已保存的正式测试指标进行统计分析。"""

from pathlib import Path
import hashlib
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
import yaml


PROJECT_ROOT = Path(
    os.environ.get("SCI_ROUTING_PROJECT_ROOT", Path(__file__).resolve().parents[3])
)
OUTPUT_ROOT = Path(os.environ.get("SCI_ROUTING_OUTPUT_ROOT", PROJECT_ROOT))
RECEIPT_PATH = PROJECT_ROOT / "logs" / "nn5_formal_test_access_receipt.json"
PER_SERIES_PATH = PROJECT_ROOT / "results" / "nn5_test_per_series_metrics.csv"
AGGREGATE_INPUT_PATH = PROJECT_ROOT / "results" / "nn5_test_aggregate_metrics.csv"
CONFIG_PATH = PROJECT_ROOT / "experiment_config.yaml"

CI_PATH = OUTPUT_ROOT / "results" / "nn5_method_bootstrap_ci.csv"
PAIRWISE_PATH = OUTPUT_ROOT / "results" / "nn5_primary_pairwise_tests.csv"
NONINFERIORITY_PATH = OUTPUT_ROOT / "results" / "nn5_noninferiority_tests.csv"
SUMMARY_PATH = OUTPUT_ROOT / "results" / "nn5_statistical_summary.yaml"
FIGURE_PATH = OUTPUT_ROOT / "figures" / "nn5_statistical_comparison.png"
for path in [CI_PATH, PAIRWISE_PATH, NONINFERIORITY_PATH, SUMMARY_PATH, FIGURE_PATH]:
    path.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted_sorted = np.empty(count, dtype=float)
    running_maximum = 0.0
    for position, original_index in enumerate(order):
        candidate = min(1.0, (count - position) * p_values[original_index])
        running_maximum = max(running_maximum, candidate)
        adjusted_sorted[position] = running_maximum
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def safe_wilcoxon(first, second, alternative="two-sided"):
    difference = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    if np.all(np.abs(difference) <= 1e-15):
        return 0.0, 1.0
    result = wilcoxon(first, second, alternative=alternative,
                      zero_method="wilcox", method="auto")
    return float(result.statistic), float(result.pvalue)


def rank_biserial_primary_better(primary, comparator):
    difference = np.asarray(primary, dtype=float) - np.asarray(comparator, dtype=float)
    difference = difference[np.abs(difference) > 1e-15]
    if len(difference) == 0:
        return 0.0
    ranks = rankdata(np.abs(difference), method="average")
    positive = float(np.sum(ranks[difference > 0]))
    negative = float(np.sum(ranks[difference < 0]))
    return (negative - positive) / (negative + positive)


# Validate the formal-test receipt and result hashes
with RECEIPT_PATH.open("r", encoding="utf-8") as handle:
    receipt = json.load(handle)
if receipt["status"] != "COMPLETED" or int(receipt["formal_test_run_number"]) != 1:
    raise ValueError("正式测试回执不是唯一一次已完成状态")

for path in [PER_SERIES_PATH, AGGREGATE_INPUT_PATH]:
    relative = str(path.relative_to(PROJECT_ROOT))
    expected_hash = receipt["result_sha256"].get(relative)
    if expected_hash is None or sha256_file(path) != expected_hash:
        raise ValueError(f"正式测试结果哈希不一致：{relative}")

with CONFIG_PATH.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
seed = int(config["study"]["seed"])
bootstrap_repetitions = int(config["statistics"]["bootstrap_repetitions"])
confidence_level = float(config["statistics"]["confidence_level"])
familywise_alpha = float(config["statistics"]["familywise_alpha"])
noninferiority_margin = float(config["statistics"]["noninferiority_margin_relative_RMSSE"])


# Load per-series metrics and validate completeness
per_series = pd.read_csv(PER_SERIES_PATH)
aggregate_input = pd.read_csv(AGGREGATE_INPUT_PATH)
if len(per_series) != 111 * 14 or per_series["series_id"].nunique() != 111:
    raise ValueError("逐序列指标行数或序列数异常")
if per_series.duplicated(["series_id", "method"]).any():
    raise ValueError("逐序列指标存在重复键")
if per_series[["RMSSE", "MASE", "sMAPE", "RMSE"]].isna().any().any():
    raise ValueError("逐序列预测指标存在缺失值")

wide = per_series.pivot(index="series_id", columns="method", values="RMSSE").sort_index()
methods = list(wide.columns)
if len(methods) != 14 or wide.isna().any().any():
    raise ValueError("方法数量或配对矩阵完整性异常")

aggregate_lookup = aggregate_input.set_index("method")["mean_RMSSE"]
recomputed_means = wide.mean(axis=0)
maximum_mean_difference = float(np.max(np.abs(
    recomputed_means.loc[aggregate_lookup.index].to_numpy()
    - aggregate_lookup.to_numpy()
)))
if maximum_mean_difference > 1e-12:
    raise ValueError("逐序列均值无法复现汇总RMSSE")


# Paired bootstrap confidence intervals
rng = np.random.default_rng(seed)
series_count = len(wide)
bootstrap_indices = rng.integers(
    0, series_count, size=(bootstrap_repetitions, series_count)
)
alpha_tail = (1.0 - confidence_level) / 2.0
method_ranks = wide.rank(axis=1, method="average", ascending=True).mean(axis=0)
ci_records = []
bootstrap_method_means = {}
for method in methods:
    values = wide[method].to_numpy(dtype=float)
    bootstrap_means = values[bootstrap_indices].mean(axis=1)
    bootstrap_method_means[method] = bootstrap_means
    ci_records.append({
        "dataset_id": "nn5_daily", "method": method,
        "mean_RMSSE": float(np.mean(values)), "median_RMSSE": float(np.median(values)),
        "bootstrap_standard_error": float(np.std(bootstrap_means, ddof=1)),
        "ci_level": confidence_level,
        "ci_lower": float(np.quantile(bootstrap_means, alpha_tail)),
        "ci_upper": float(np.quantile(bootstrap_means, 1.0 - alpha_tail)),
        "mean_within_series_rank": float(method_ranks[method]),
        "series_count": series_count, "bootstrap_repetitions": bootstrap_repetitions,
        "seed": seed,
    })
ci_table = pd.DataFrame(ci_records).sort_values("mean_RMSSE").reset_index(drop=True)
ci_table["mean_RMSSE_rank"] = np.arange(1, len(ci_table) + 1)
ci_table.to_csv(CI_PATH, index=False)


# Pairwise tests of the full method against the other 13 methods
primary_method = "adaptive_full_router"
primary_values = wide[primary_method].to_numpy(dtype=float)
pairwise_records = []
for comparator in methods:
    if comparator == primary_method:
        continue
    comparator_values = wide[comparator].to_numpy(dtype=float)
    statistic, p_value = safe_wilcoxon(primary_values, comparator_values, "two-sided")
    difference = primary_values - comparator_values
    bootstrap_difference = difference[bootstrap_indices].mean(axis=1)
    mean_difference = float(np.mean(difference))
    comparator_mean = float(np.mean(comparator_values))
    pairwise_records.append({
        "dataset_id": "nn5_daily", "primary_method": primary_method,
        "comparator": comparator, "mean_primary_RMSSE": float(np.mean(primary_values)),
        "mean_comparator_RMSSE": comparator_mean,
        "mean_difference_primary_minus_comparator": mean_difference,
        "relative_difference_percent": 100.0 * mean_difference / comparator_mean,
        "bootstrap_ci_lower_difference": float(np.quantile(bootstrap_difference, alpha_tail)),
        "bootstrap_ci_upper_difference": float(np.quantile(bootstrap_difference, 1.0 - alpha_tail)),
        "primary_series_win_rate": float(np.mean(primary_values < comparator_values)),
        "wilcoxon_statistic": statistic, "raw_p_value": p_value,
        "rank_biserial_primary_better": rank_biserial_primary_better(
            primary_values, comparator_values
        ),
    })
pairwise = pd.DataFrame(pairwise_records)
pairwise["holm_adjusted_p_value"] = holm_adjust(pairwise["raw_p_value"])
pairwise["significant_after_holm"] = pairwise["holm_adjusted_p_value"] < familywise_alpha
pairwise["direction"] = np.where(
    pairwise["mean_difference_primary_minus_comparator"] < 0,
    "primary_better", "comparator_better",
)
pairwise = pairwise.sort_values("holm_adjusted_p_value").reset_index(drop=True)
pairwise.to_csv(PAIRWISE_PATH, index=False)


# Preregistered noninferiority tests with a 1% margin
noninferiority_comparators = [
    "ridge_only", "lightgbm_only", "equal_weight_average",
    "hard_aalf_like_router", "hard_logistic_same_features",
    "hard_random_forest_same_features", "class_weight_only",
    "soft_targets_only", "residual_features_only", "static_full_router",
]
noninferiority_records = []
for comparator in noninferiority_comparators:
    comparator_values = wide[comparator].to_numpy(dtype=float)
    margin_adjusted_comparator = (1.0 + noninferiority_margin) * comparator_values
    statistic, p_value = safe_wilcoxon(
        primary_values, margin_adjusted_comparator, alternative="less"
    )
    adjusted_difference = primary_values - margin_adjusted_comparator
    bootstrap_adjusted = adjusted_difference[bootstrap_indices].mean(axis=1)
    relative_difference = (
        float(np.mean(primary_values)) / float(np.mean(comparator_values)) - 1.0
    )
    noninferiority_records.append({
        "dataset_id": "nn5_daily", "primary_method": primary_method,
        "comparator": comparator, "relative_margin": noninferiority_margin,
        "observed_relative_difference": relative_difference,
        "margin_adjusted_mean_difference": float(np.mean(adjusted_difference)),
        "one_sided_bootstrap_upper_bound": float(
            np.quantile(bootstrap_adjusted, confidence_level)
        ),
        "wilcoxon_statistic": statistic, "raw_one_sided_p_value": p_value,
    })
noninferiority = pd.DataFrame(noninferiority_records)
noninferiority["holm_adjusted_p_value"] = holm_adjust(
    noninferiority["raw_one_sided_p_value"]
)
noninferiority["noninferior_after_holm"] = (
    noninferiority["holm_adjusted_p_value"] < familywise_alpha
)
noninferiority["bootstrap_supports_noninferiority"] = (
    noninferiority["one_sided_bootstrap_upper_bound"] < 0.0
)
noninferiority = noninferiority.sort_values("holm_adjusted_p_value").reset_index(drop=True)
noninferiority.to_csv(NONINFERIORITY_PATH, index=False)


# Statistical summary
aggregate_by_method = aggregate_input.set_index("method")
primary_aggregate = aggregate_by_method.loc[primary_method]
significantly_better = pairwise.loc[
    pairwise["significant_after_holm"] & (pairwise["direction"] == "primary_better"),
    "comparator",
].tolist()
significantly_worse = pairwise.loc[
    pairwise["significant_after_holm"] & (pairwise["direction"] == "comparator_better"),
    "comparator",
].tolist()
noninferior_to = noninferiority.loc[
    noninferiority["noninferior_after_holm"], "comparator"
].tolist()

statistical_summary = {
    "dataset_id": "nn5_daily",
    "analysis_source": "saved_formal_test_per_series_metrics_only",
    "raw_test_data_reopened": False,
    "models_refit": False,
    "parameters_changed": False,
    "formal_test_run_number": 1,
    "series_count": series_count,
    "primary_method": primary_method,
    "primary_mean_RMSSE": float(primary_aggregate["mean_RMSSE"]),
    "primary_RMSSE_rank": int(primary_aggregate["RMSSE_rank"]),
    "target_simple_coverage": float(config["study"]["primary_target_coverage"]),
    "observed_simple_coverage": float(primary_aggregate["simple_coverage"]),
    "coverage_constraint_passed": bool(
        primary_aggregate["absolute_coverage_violation"]
        <= config["coverage_controller"]["allowed_absolute_violation"]
    ),
    "absolute_coverage_violation": float(
        primary_aggregate["absolute_coverage_violation"]
    ),
    "bootstrap_repetitions": bootstrap_repetitions,
    "confidence_level": confidence_level,
    "pairwise_family_size": len(pairwise),
    "holm_familywise_alpha": familywise_alpha,
    "significantly_better_than": significantly_better,
    "significantly_worse_than": significantly_worse,
    "noninferior_to_at_1_percent_margin": noninferior_to,
    "interpretation_rule": (
        "positive rank_biserial_primary_better favors the primary method; "
        "negative values favor the comparator"
    ),
    "scientific_warning": (
        "Do not retune NN5 models or exclude unfavorable formal-test comparisons"
    ),
}
with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(statistical_summary, handle, sort_keys=False, allow_unicode=True)


# Plot the statistical comparison
figure, axes = plt.subplots(1, 3, figsize=(18, 6))
plot_ci = ci_table.sort_values("mean_RMSSE", ascending=False)
axes[0].errorbar(
    plot_ci["mean_RMSSE"], plot_ci["method"],
    xerr=np.vstack([
        plot_ci["mean_RMSSE"] - plot_ci["ci_lower"],
        plot_ci["ci_upper"] - plot_ci["mean_RMSSE"],
    ]), fmt="o", color="#4c78a8", ecolor="#9ecae9", capsize=3,
)
axes[0].set_xlabel("Mean RMSSE with 95% bootstrap CI")
axes[0].set_title("Method uncertainty")

plot_pairwise = pairwise.sort_values("mean_difference_primary_minus_comparator")
axes[1].errorbar(
    plot_pairwise["mean_difference_primary_minus_comparator"],
    plot_pairwise["comparator"],
    xerr=np.vstack([
        plot_pairwise["mean_difference_primary_minus_comparator"]
        - plot_pairwise["bootstrap_ci_lower_difference"],
        plot_pairwise["bootstrap_ci_upper_difference"]
        - plot_pairwise["mean_difference_primary_minus_comparator"],
    ]), fmt="o", color="#e15759", ecolor="#ffb3b3", capsize=3,
)
axes[1].axvline(0.0, color="black", linestyle="--")
axes[1].set_xlabel("Adaptive full minus comparator RMSSE")
axes[1].set_title("Paired mean differences")

for comparator, color in [
    ("ridge_only", "#59a14f"),
    ("lightgbm_only", "#f28e2b"),
    ("equal_weight_average", "#4e79a7"),
]:
    difference = primary_values - wide[comparator].to_numpy(dtype=float)
    axes[2].hist(difference, bins=25, alpha=0.45, label=comparator, color=color)
axes[2].axvline(0.0, color="black", linestyle="--")
axes[2].set_xlabel("Per-series RMSSE difference")
axes[2].set_ylabel("Series count")
axes[2].set_title("Primary minus key comparator")
axes[2].legend()
figure.tight_layout()
figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
plt.close(figure)


print()
print("NN5 正式测试统计分析全部通过")
print("分析来源：仅已保存的逐序列正式指标")
print("原始测试数据是否重新读取：否")
print("模型是否重新拟合：否")
print("配对序列数量：", series_count)
print("Bootstrap次数：", bootstrap_repetitions)
print("完整方法mean RMSSE：", f"{primary_aggregate['mean_RMSSE']:.6f}")
print("完整方法正式排名：", int(primary_aggregate["RMSSE_rank"]))
print("覆盖率约束是否通过：", "是" if statistical_summary["coverage_constraint_passed"] else "否")
print("Holm校正后显著优于：", significantly_better)
print("Holm校正后显著劣于：", significantly_worse)
print("1%界值下非劣于：", noninferior_to)
print("Bootstrap置信区间：", CI_PATH)
print("配对检验：", PAIRWISE_PATH)
print("非劣效检验：", NONINFERIORITY_PATH)
print("统计摘要：", SUMMARY_PATH)
print("统计图片：", FIGURE_PATH)
