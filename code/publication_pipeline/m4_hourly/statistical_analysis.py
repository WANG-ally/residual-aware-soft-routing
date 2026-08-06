#!/usr/bin/env python3
"""Statistical analysis from saved M4 Hourly formal metrics only.

The raw/processed time series, per-timestamp test predictions, and all trained
models are intentionally absent from this program.  The analysis unit is one
M4 Hourly series, so variable-length series receive equal statistical weight."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sci-routing")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
import yaml


PROJECT_ROOT = Path(
    os.environ.get(
        "SCI_ROUTING_ROOT",
        os.environ.get("SCI_ROUTING_PROJECT_ROOT", Path(__file__).resolve().parents[3]),
    )
).resolve()
OUTPUT_ROOT = Path(os.environ.get("SCI_ROUTING_OUTPUT_ROOT", PROJECT_ROOT)).resolve()

DATASET_ID = "m4_hourly"
EXPECTED_SERIES = 414
EXPECTED_METHODS = 14
EXPECTED_EVALUATOR_HASH = (
    "8adc038fd32595de63db88865106efc7d0805cf0228569d2ea2c627ca1faae3c"
)
EXPECTED_AUTHORIZATION_ID = (
    "e69bce60cdf15e5007baf63ae1e2e84c887dbd3e72362d462ade6ce78f6bdb46"
)
EXPECTED_FINAL_FREEZE_ID = (
    "754e41679402bac7e7e987d29af5a9ac212ed869d71ef0d4c7c271a5bcbfebe6"
)

RECEIPT_PATH = PROJECT_ROOT / "logs/m4_hourly_formal_test_access_receipt.json"
AUTHORIZATION_PATH = PROJECT_ROOT / "logs/m4_hourly_evaluator_authorization.json"
EVALUATOR_PATH = PROJECT_ROOT / "code/publication_pipeline/m4_hourly/formal_test.py"
PER_SERIES_PATH = (
    PROJECT_ROOT / "results/m4_hourly_test_per_series_metrics.csv"
)
AGGREGATE_INPUT_PATH = (
    PROJECT_ROOT / "results/m4_hourly_test_aggregate_metrics.csv"
)
CONFIG_PATH = PROJECT_ROOT / "experiment_config.yaml"

CI_PATH = OUTPUT_ROOT / "results/m4_hourly_method_bootstrap_ci.csv"
PAIRWISE_PATH = OUTPUT_ROOT / "results/m4_hourly_primary_pairwise_tests.csv"
NONINFERIORITY_PATH = (
    OUTPUT_ROOT / "results/m4_hourly_noninferiority_tests.csv"
)
SUMMARY_PATH = OUTPUT_ROOT / "results/m4_hourly_statistical_summary.yaml"
CHECKS_PATH = OUTPUT_ROOT / "results/m4_hourly_statistical_analysis_checks.csv"
FIGURE_PATH = OUTPUT_ROOT / "figures/m4_hourly_statistical_comparison.png"
REPORT_PATH = OUTPUT_ROOT / "logs/m4_hourly_statistical_analysis_report.json"

for path in (
    CI_PATH,
    PAIRWISE_PATH,
    NONINFERIORITY_PATH,
    SUMMARY_PATH,
    CHECKS_PATH,
    FIGURE_PATH,
    REPORT_PATH,
):
    path.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def holm_adjust(p_values: np.ndarray | pd.Series) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=np.float64)
    count = len(p_values)
    order = np.argsort(p_values, kind="stable")
    adjusted_in_order = np.empty(count, dtype=np.float64)
    running_maximum = 0.0
    for position, original_index in enumerate(order):
        candidate = min(1.0, (count - position) * p_values[original_index])
        running_maximum = max(running_maximum, candidate)
        adjusted_in_order[position] = running_maximum
    adjusted = np.empty(count, dtype=np.float64)
    adjusted[order] = adjusted_in_order
    return adjusted


def safe_wilcoxon(
    first: np.ndarray,
    second: np.ndarray,
    alternative: str = "two-sided",
) -> tuple[float, float]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    difference = first - second
    if np.all(np.abs(difference) <= 1e-15):
        return 0.0, 1.0
    result = wilcoxon(
        first,
        second,
        alternative=alternative,
        zero_method="wilcox",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def rank_biserial_primary_better(
    primary: np.ndarray, comparator: np.ndarray
) -> float:
    difference = np.asarray(primary, dtype=np.float64) - np.asarray(
        comparator, dtype=np.float64
    )
    difference = difference[np.abs(difference) > 1e-15]
    if len(difference) == 0:
        return 0.0
    ranks = rankdata(np.abs(difference), method="average")
    positive_rank_sum = float(np.sum(ranks[difference > 0.0]))
    negative_rank_sum = float(np.sum(ranks[difference < 0.0]))
    return float(
        (negative_rank_sum - positive_rank_sum)
        / (negative_rank_sum + positive_rank_sum)
    )


def main() -> None:
    required = [
        RECEIPT_PATH,
        AUTHORIZATION_PATH,
        EVALUATOR_PATH,
        PER_SERIES_PATH,
        AGGREGATE_INPUT_PATH,
        CONFIG_PATH,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required formal-analysis files are missing: {missing}")

    checks: list[dict[str, object]] = []

    def record_check(name: str, passed: bool, detail: object) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    with RECEIPT_PATH.open("r", encoding="utf-8") as handle:
        receipt = json.load(handle)
    with AUTHORIZATION_PATH.open("r", encoding="utf-8") as handle:
        authorization = json.load(handle)
    formal_identity_valid = bool(
        receipt.get("dataset_id") == DATASET_ID
        and receipt.get("status") == "COMPLETED"
        and int(receipt.get("formal_test_run_number", -1)) == 1
        and int(receipt.get("formal_test_rows", -1)) == 56_337
        and int(receipt.get("series_count", -1)) == EXPECTED_SERIES
        and receipt.get("final_freeze_id") == EXPECTED_FINAL_FREEZE_ID
        and receipt.get("authorization_id") == EXPECTED_AUTHORIZATION_ID
        and receipt.get("evaluator_sha256") == EXPECTED_EVALUATOR_HASH
        and authorization.get("status") == "AUTHORIZED"
        and authorization.get("authorization_id") == EXPECTED_AUTHORIZATION_ID
        and authorization.get("dry_run", {}).get("test_accessed") is False
        and sha256_file(EVALUATOR_PATH) == EXPECTED_EVALUATOR_HASH
    )
    record_check(
        "unique_formal_test_receipt_and_authorization_valid",
        formal_identity_valid,
        (
            f"status={receipt.get('status')}; run="
            f"{receipt.get('formal_test_run_number')}; rows="
            f"{receipt.get('formal_test_rows')}"
        ),
    )

    result_hashes_valid = True
    for path in (PER_SERIES_PATH, AGGREGATE_INPUT_PATH):
        relative = str(path.relative_to(PROJECT_ROOT))
        expected_hash = receipt.get("result_sha256", {}).get(relative)
        result_hashes_valid = bool(
            result_hashes_valid
            and expected_hash is not None
            and sha256_file(path) == expected_hash
        )
    record_check(
        "saved_metric_file_hashes_match_receipt",
        result_hashes_valid,
        "per-series and aggregate metric hashes checked",
    )
    if not formal_identity_valid or not result_hashes_valid:
        raise ValueError("Formal-test identity or saved metric hashes are invalid")

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    seed = int(config["study"]["seed"])
    bootstrap_repetitions = int(config["statistics"]["bootstrap_repetitions"])
    confidence_level = float(config["statistics"]["confidence_level"])
    familywise_alpha = float(config["statistics"]["familywise_alpha"])
    noninferiority_margin = float(
        config["statistics"]["noninferiority_margin_relative_RMSSE"]
    )
    statistical_protocol_valid = bool(
        config["statistics"]["unit"] == "series"
        and config["statistics"]["within_dataset_test"] == "paired_wilcoxon"
        and bootstrap_repetitions == 10_000
        and confidence_level == 0.95
        and familywise_alpha == 0.05
        and noninferiority_margin == 0.01
    )
    record_check(
        "preregistered_statistical_protocol_valid",
        statistical_protocol_valid,
        (
            f"unit={config['statistics']['unit']}; bootstrap="
            f"{bootstrap_repetitions}; margin={noninferiority_margin}"
        ),
    )

    # These are the only formal-result tables parsed by this script.
    per_series = pd.read_csv(PER_SERIES_PATH)
    aggregate_input = pd.read_csv(AGGREGATE_INPUT_PATH)
    per_series_identity_valid = bool(
        len(per_series) == EXPECTED_SERIES * EXPECTED_METHODS
        and per_series["series_id"].nunique() == EXPECTED_SERIES
        and per_series["method"].nunique() == EXPECTED_METHODS
        and set(per_series["dataset_id"]) == {DATASET_ID}
        and not per_series.duplicated(["series_id", "method"]).any()
        and per_series.groupby("series_id", observed=True)["test_rows"]
        .nunique()
        .eq(1)
        .all()
        and int(per_series["test_rows"].min()) == 113
        and int(per_series["test_rows"].max()) == 152
    )
    record_check(
        "paired_per_series_table_complete",
        per_series_identity_valid,
        (
            f"rows={len(per_series)}; series="
            f"{per_series['series_id'].nunique()}; methods="
            f"{per_series['method'].nunique()}"
        ),
    )

    metric_columns = ["RMSSE", "MASE", "sMAPE", "RMSE", "MAE"]
    per_series_metrics_finite = bool(
        np.isfinite(per_series[metric_columns].to_numpy(dtype=np.float64)).all()
    )
    record_check(
        "all_per_series_metrics_are_finite",
        per_series_metrics_finite,
        f"metric_columns={metric_columns}",
    )

    wide = per_series.pivot(
        index="series_id", columns="method", values="RMSSE"
    ).sort_index()
    paired_matrix_valid = bool(
        wide.shape == (EXPECTED_SERIES, EXPECTED_METHODS)
        and not wide.isna().any().any()
    )
    record_check(
        "RMSSE_pairing_matrix_complete",
        paired_matrix_valid,
        f"shape={wide.shape}",
    )
    methods = list(wide.columns)

    aggregate_identity_valid = bool(
        len(aggregate_input) == EXPECTED_METHODS
        and set(aggregate_input["dataset_id"]) == {DATASET_ID}
        and set(aggregate_input["method"]) == set(methods)
        and aggregate_input["method"].is_unique
    )
    aggregate_lookup = aggregate_input.set_index("method")["mean_RMSSE"]
    recomputed_means = wide.mean(axis=0)
    maximum_mean_difference = float(
        np.max(
            np.abs(
                recomputed_means.loc[aggregate_lookup.index].to_numpy()
                - aggregate_lookup.to_numpy()
            )
        )
    )
    aggregate_reproduced = bool(
        aggregate_identity_valid and maximum_mean_difference <= 1e-12
    )
    record_check(
        "aggregate_mean_RMSSE_reproduced",
        aggregate_reproduced,
        f"maximum_difference={maximum_mean_difference:.3e}",
    )

    input_checks_pass = bool(
        statistical_protocol_valid
        and per_series_identity_valid
        and per_series_metrics_finite
        and paired_matrix_valid
        and aggregate_reproduced
    )
    if not input_checks_pass:
        failed = [item for item in checks if not item["passed"]]
        raise AssertionError(f"Statistical input audit failed: {failed}")

    # --------------------------------------------------------
    # Paired series bootstrap confidence intervals.
    # --------------------------------------------------------
    rng = np.random.default_rng(seed)
    series_count = len(wide)
    bootstrap_indices = rng.integers(
        0,
        series_count,
        size=(bootstrap_repetitions, series_count),
    )
    alpha_tail = (1.0 - confidence_level) / 2.0
    method_ranks = wide.rank(
        axis=1, method="average", ascending=True
    ).mean(axis=0)
    ci_records: list[dict[str, object]] = []
    for method in methods:
        values = wide[method].to_numpy(dtype=np.float64)
        bootstrap_means = values[bootstrap_indices].mean(axis=1)
        ci_records.append(
            {
                "dataset_id": DATASET_ID,
                "method": method,
                "mean_RMSSE": float(np.mean(values)),
                "median_RMSSE": float(np.median(values)),
                "bootstrap_standard_error": float(
                    np.std(bootstrap_means, ddof=1)
                ),
                "ci_level": confidence_level,
                "ci_lower": float(np.quantile(bootstrap_means, alpha_tail)),
                "ci_upper": float(
                    np.quantile(bootstrap_means, 1.0 - alpha_tail)
                ),
                "mean_within_series_rank": float(method_ranks[method]),
                "series_count": series_count,
                "bootstrap_repetitions": bootstrap_repetitions,
                "seed": seed,
            }
        )
    ci_table = pd.DataFrame(ci_records).sort_values(
        "mean_RMSSE", kind="stable"
    ).reset_index(drop=True)
    ci_table["mean_RMSSE_rank"] = np.arange(1, len(ci_table) + 1)
    ci_table.to_csv(CI_PATH, index=False)

    # --------------------------------------------------------
    # Primary-vs-all paired tests with one Holm family.
    # --------------------------------------------------------
    primary_method = "adaptive_full_router"
    primary_values = wide[primary_method].to_numpy(dtype=np.float64)
    pairwise_records: list[dict[str, object]] = []
    for comparator in methods:
        if comparator == primary_method:
            continue
        comparator_values = wide[comparator].to_numpy(dtype=np.float64)
        statistic, p_value = safe_wilcoxon(
            primary_values, comparator_values, alternative="two-sided"
        )
        difference = primary_values - comparator_values
        bootstrap_difference = difference[bootstrap_indices].mean(axis=1)
        mean_difference = float(np.mean(difference))
        comparator_mean = float(np.mean(comparator_values))
        pairwise_records.append(
            {
                "dataset_id": DATASET_ID,
                "primary_method": primary_method,
                "comparator": comparator,
                "mean_primary_RMSSE": float(np.mean(primary_values)),
                "mean_comparator_RMSSE": comparator_mean,
                "mean_difference_primary_minus_comparator": mean_difference,
                "relative_difference_percent": (
                    100.0 * mean_difference / comparator_mean
                ),
                "bootstrap_ci_lower_difference": float(
                    np.quantile(bootstrap_difference, alpha_tail)
                ),
                "bootstrap_ci_upper_difference": float(
                    np.quantile(bootstrap_difference, 1.0 - alpha_tail)
                ),
                "primary_series_win_rate": float(
                    np.mean(primary_values < comparator_values)
                ),
                "wilcoxon_statistic": statistic,
                "raw_p_value": p_value,
                "rank_biserial_primary_better": (
                    rank_biserial_primary_better(
                        primary_values, comparator_values
                    )
                ),
            }
        )
    pairwise = pd.DataFrame(pairwise_records)
    pairwise["holm_adjusted_p_value"] = holm_adjust(pairwise["raw_p_value"])
    pairwise["significant_after_holm"] = (
        pairwise["holm_adjusted_p_value"] < familywise_alpha
    )
    # Wilcoxon tests a paired rank/location contrast, whereas the paper's
    # primary estimand is the equal-weight mean RMSSE.  Keep their directions
    # explicit and require agreement before making a directional claim.
    pairwise["rank_direction"] = np.where(
        pairwise["rank_biserial_primary_better"] > 0.0,
        "primary_better",
        np.where(
            pairwise["rank_biserial_primary_better"] < 0.0,
            "comparator_better",
            "tie",
        ),
    )
    pairwise["bootstrap_mean_difference_excludes_zero"] = (
        (pairwise["bootstrap_ci_upper_difference"] < 0.0)
        | (pairwise["bootstrap_ci_lower_difference"] > 0.0)
    )
    pairwise["bootstrap_mean_direction"] = np.where(
        pairwise["bootstrap_ci_upper_difference"] < 0.0,
        "primary_better",
        np.where(
            pairwise["bootstrap_ci_lower_difference"] > 0.0,
            "comparator_better",
            "inconclusive",
        ),
    )
    pairwise["rank_and_bootstrap_direction_agree"] = (
        pairwise["rank_direction"] == pairwise["bootstrap_mean_direction"]
    )
    pairwise["holm_and_bootstrap_support_direction"] = (
        pairwise["significant_after_holm"]
        & pairwise["bootstrap_mean_difference_excludes_zero"]
        & pairwise["rank_and_bootstrap_direction_agree"]
    )
    pairwise = pairwise.sort_values(
        ["holm_adjusted_p_value", "comparator"], kind="stable"
    ).reset_index(drop=True)
    pairwise.to_csv(PAIRWISE_PATH, index=False)

    # --------------------------------------------------------
    # Preregistered 1% noninferiority margin for non-oracle comparators.
    # --------------------------------------------------------
    noninferiority_comparators = [
        "ridge_only",
        "lightgbm_only",
        "equal_weight_average",
        "hard_aalf_like_router",
        "hard_logistic_same_features",
        "hard_random_forest_same_features",
        "class_weight_only",
        "soft_targets_only",
        "residual_features_only",
        "static_full_router",
    ]
    noninferiority_records: list[dict[str, object]] = []
    for comparator in noninferiority_comparators:
        comparator_values = wide[comparator].to_numpy(dtype=np.float64)
        margin_adjusted_comparator = (
            1.0 + noninferiority_margin
        ) * comparator_values
        statistic, p_value = safe_wilcoxon(
            primary_values,
            margin_adjusted_comparator,
            alternative="less",
        )
        adjusted_difference = primary_values - margin_adjusted_comparator
        bootstrap_adjusted = adjusted_difference[bootstrap_indices].mean(axis=1)
        noninferiority_records.append(
            {
                "dataset_id": DATASET_ID,
                "primary_method": primary_method,
                "comparator": comparator,
                "relative_margin": noninferiority_margin,
                "observed_relative_difference": (
                    float(np.mean(primary_values))
                    / float(np.mean(comparator_values))
                    - 1.0
                ),
                "margin_adjusted_mean_difference": float(
                    np.mean(adjusted_difference)
                ),
                "one_sided_bootstrap_upper_bound": float(
                    np.quantile(bootstrap_adjusted, confidence_level)
                ),
                "wilcoxon_statistic": statistic,
                "raw_one_sided_p_value": p_value,
            }
        )
    noninferiority = pd.DataFrame(noninferiority_records)
    noninferiority["holm_adjusted_p_value"] = holm_adjust(
        noninferiority["raw_one_sided_p_value"]
    )
    noninferiority["wilcoxon_supports_noninferiority"] = (
        noninferiority["holm_adjusted_p_value"] < familywise_alpha
    )
    noninferiority["bootstrap_supports_noninferiority"] = (
        noninferiority["one_sided_bootstrap_upper_bound"] < 0.0
    )
    noninferiority["noninferior_by_both_criteria"] = (
        noninferiority["wilcoxon_supports_noninferiority"]
        & noninferiority["bootstrap_supports_noninferiority"]
    )
    noninferiority = noninferiority.sort_values(
        ["holm_adjusted_p_value", "comparator"], kind="stable"
    ).reset_index(drop=True)
    noninferiority.to_csv(NONINFERIORITY_PATH, index=False)

    aggregate_by_method = aggregate_input.set_index("method")
    primary_aggregate = aggregate_by_method.loc[primary_method]
    supported_better = pairwise.loc[
        pairwise["holm_and_bootstrap_support_direction"]
        & pairwise["bootstrap_mean_direction"].eq("primary_better"),
        "comparator",
    ].tolist()
    supported_worse = pairwise.loc[
        pairwise["holm_and_bootstrap_support_direction"]
        & pairwise["bootstrap_mean_direction"].eq("comparator_better"),
        "comparator",
    ].tolist()
    discordant_or_inconclusive = pairwise.loc[
        pairwise["significant_after_holm"]
        & ~pairwise["holm_and_bootstrap_support_direction"],
        "comparator",
    ].tolist()
    noninferior_to = noninferiority.loc[
        noninferiority["noninferior_by_both_criteria"], "comparator"
    ].tolist()
    coverage_constraint_passed = bool(
        float(primary_aggregate["absolute_coverage_violation"])
        <= float(config["coverage_controller"]["allowed_absolute_violation"])
    )
    record_check(
        "primary_coverage_constraint_passed",
        coverage_constraint_passed,
        (
            f"observed={primary_aggregate['simple_coverage']:.9f}; "
            f"violation={primary_aggregate['absolute_coverage_violation']:.9f}; "
            f"allowed={config['coverage_controller']['allowed_absolute_violation']}"
        ),
    )
    record_check(
        "statistical_output_dimensions_valid",
        bool(
            len(ci_table) == EXPECTED_METHODS
            and len(pairwise) == EXPECTED_METHODS - 1
            and len(noninferiority) == len(noninferiority_comparators)
        ),
        (
            f"CI={len(ci_table)}; pairwise={len(pairwise)}; "
            f"noninferiority={len(noninferiority)}"
        ),
    )
    record_check(
        "formal_analysis_did_not_reopen_raw_test_or_models",
        True,
        "only receipt, authorization, config, per-series metrics, and aggregate metrics parsed",
    )

    statistical_summary = {
        "dataset_id": DATASET_ID,
        "analysis_source": "saved_formal_test_per_series_metrics_only",
        "raw_test_data_reopened": False,
        "per_timestamp_predictions_reopened": False,
        "models_loaded_or_refit": False,
        "parameters_changed": False,
        "formal_test_run_number": 1,
        "series_count": series_count,
        "variable_length_series_weighting": "equal weight per series",
        "primary_method": primary_method,
        "primary_mean_RMSSE": float(primary_aggregate["mean_RMSSE"]),
        "primary_RMSSE_rank": int(primary_aggregate["RMSSE_rank"]),
        "target_simple_coverage": float(
            config["study"]["primary_target_coverage"]
        ),
        "observed_simple_coverage": float(primary_aggregate["simple_coverage"]),
        "coverage_constraint_passed": coverage_constraint_passed,
        "absolute_coverage_violation": float(
            primary_aggregate["absolute_coverage_violation"]
        ),
        "bootstrap_repetitions": bootstrap_repetitions,
        "confidence_level": confidence_level,
        "pairwise_family_size": len(pairwise),
        "holm_familywise_alpha": familywise_alpha,
        "holm_and_bootstrap_support_better_than": supported_better,
        "holm_and_bootstrap_support_worse_than": supported_worse,
        "holm_significant_but_mean_bootstrap_discordant_or_inconclusive": (
            discordant_or_inconclusive
        ),
        "noninferior_to_at_1_percent_margin_by_wilcoxon_and_bootstrap": (
            noninferior_to
        ),
        "effect_size_interpretation": (
            "positive rank_biserial_primary_better favors the primary method; "
            "negative values favor the comparator"
        ),
        "directional_claim_rule": (
            "claim a direction only when the Holm-adjusted paired Wilcoxon "
            "test is significant, the paired-bootstrap confidence interval "
            "for the equal-weight mean difference excludes zero, and both "
            "procedures favor the same method"
        ),
        "scientific_warning": (
            "Do not rerun, retune, exclude unfavorable comparisons, or claim "
            "superiority where adjusted inference does not support it"
        ),
    }
    SUMMARY_PATH.write_text(
        yaml.safe_dump(
            statistical_summary, sort_keys=False, allow_unicode=True
        ),
        encoding="utf-8",
    )

    checks_frame = pd.DataFrame(checks)
    failed_checks = checks_frame.loc[~checks_frame["passed"]]
    if not failed_checks.empty:
        raise AssertionError(
            f"M4 Hourly statistical checks failed: {failed_checks.to_dict('records')}"
        )
    checks_frame.to_csv(CHECKS_PATH, index=False)

    figure, axes = plt.subplots(1, 3, figsize=(20, 7))
    plot_ci = ci_table.sort_values("mean_RMSSE", ascending=False)
    axes[0].errorbar(
        plot_ci["mean_RMSSE"],
        plot_ci["method"],
        xerr=np.vstack(
            [
                plot_ci["mean_RMSSE"] - plot_ci["ci_lower"],
                plot_ci["ci_upper"] - plot_ci["mean_RMSSE"],
            ]
        ),
        fmt="o",
        color="#4c78a8",
        ecolor="#9ecae9",
        capsize=3,
    )
    axes[0].set_xlabel("Mean RMSSE with 95% paired-bootstrap CI")
    axes[0].set_title("Method uncertainty")

    plot_pairwise = pairwise.sort_values(
        "mean_difference_primary_minus_comparator"
    )
    axes[1].errorbar(
        plot_pairwise["mean_difference_primary_minus_comparator"],
        plot_pairwise["comparator"],
        xerr=np.vstack(
            [
                plot_pairwise["mean_difference_primary_minus_comparator"]
                - plot_pairwise["bootstrap_ci_lower_difference"],
                plot_pairwise["bootstrap_ci_upper_difference"]
                - plot_pairwise["mean_difference_primary_minus_comparator"],
            ]
        ),
        fmt="o",
        color="#e15759",
        ecolor="#ffb3b3",
        capsize=3,
    )
    axes[1].axvline(0.0, color="black", linestyle="--")
    axes[1].set_xlabel("Adaptive full minus comparator RMSSE")
    axes[1].set_title("Paired mean differences")

    for comparator, color in (
        ("ridge_only", "#59a14f"),
        ("lightgbm_only", "#f28e2b"),
        ("equal_weight_average", "#4e79a7"),
        ("hard_random_forest_same_features", "#b07aa1"),
    ):
        difference = primary_values - wide[comparator].to_numpy(dtype=np.float64)
        axes[2].hist(
            difference,
            bins=20,
            alpha=0.38,
            label=comparator,
            color=color,
        )
    axes[2].axvline(0.0, color="black", linestyle="--")
    axes[2].set_xlabel("Per-series RMSSE difference")
    axes[2].set_ylabel("Series count")
    axes[2].set_title("Primary minus key comparator")
    axes[2].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)

    report = {
        "status": "passed",
        "dataset_id": DATASET_ID,
        "formal_test_run_number": 1,
        "receipt_sha256": sha256_file(RECEIPT_PATH),
        "input_per_series_sha256": sha256_file(PER_SERIES_PATH),
        "input_aggregate_sha256": sha256_file(AGGREGATE_INPUT_PATH),
        "analysis_source": "saved_formal_test_metrics_only",
        "raw_test_data_reopened": False,
        "models_loaded_or_refit": False,
        "check_count": len(checks),
        "failed_check_count": 0,
        "primary_mean_RMSSE": float(primary_aggregate["mean_RMSSE"]),
        "primary_rank": int(primary_aggregate["RMSSE_rank"]),
        "coverage_constraint_passed": coverage_constraint_passed,
        "holm_and_bootstrap_support_better_than": supported_better,
        "holm_and_bootstrap_support_worse_than": supported_worse,
        "holm_significant_but_mean_bootstrap_discordant_or_inconclusive": (
            discordant_or_inconclusive
        ),
        "noninferior_to": noninferior_to,
        "outputs": {
            "bootstrap_ci": str(CI_PATH),
            "pairwise": str(PAIRWISE_PATH),
            "noninferiority": str(NONINFERIORITY_PATH),
            "summary": str(SUMMARY_PATH),
            "checks": str(CHECKS_PATH),
            "figure": str(FIGURE_PATH),
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("M4 Hourly 正式测试统计分析全部通过")
    print("分析来源：仅已保存的逐序列正式指标")
    print("原始测试数据是否重新读取：否")
    print("逐时点预测是否重新读取：否")
    print("模型是否重新加载或拟合：否")
    print("配对序列数量：", series_count)
    print("Bootstrap次数：", bootstrap_repetitions)
    print("完整方法mean RMSSE：", f"{primary_aggregate['mean_RMSSE']:.6f}")
    print("完整方法正式排名：", int(primary_aggregate["RMSSE_rank"]))
    print(
        "覆盖率约束是否通过：",
        "是" if coverage_constraint_passed else "否",
    )
    print("Holm与Bootstrap方向一致且支持优于：", supported_better)
    print("Holm与Bootstrap方向一致且支持劣于：", supported_worse)
    print(
        "Holm显著但均值Bootstrap方向不一致或不确定：",
        discordant_or_inconclusive,
    )
    print("1%界值下两种判据均支持非劣于：", noninferior_to)
    print("Bootstrap置信区间：", CI_PATH)
    print("配对检验：", PAIRWISE_PATH)
    print("非劣效检验：", NONINFERIORITY_PATH)
    print("统计摘要：", SUMMARY_PATH)
    print("统计检查表：", CHECKS_PATH)
    print("统计图片：", FIGURE_PATH)
    print("统计报告：", REPORT_PATH)


if __name__ == "__main__":
    main()
