#!/usr/bin/env python3
"""Locked five-dataset synthesis for manuscript preparation.

Only closeout manifests, evidence cards, and aggregate formal-test tables are
parsed. No per-series result, per-timestamp prediction, raw data, or model is
opened. The five closed datasets are the analysis blocks."""

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
from scipy.stats import friedmanchisquare, wilcoxon
import yaml


PROJECT_ROOT = Path(
    os.environ.get(
        "SCI_ROUTING_ROOT",
        os.environ.get("SCI_ROUTING_PROJECT_ROOT", Path(__file__).resolve().parents[3]),
    )
).resolve()
OUTPUT_ROOT = Path(os.environ.get("SCI_ROUTING_OUTPUT_ROOT", PROJECT_ROOT)).resolve()

DATASETS = [
    {
        "artifact_prefix": "nn5",
        "dataset_id": "nn5_daily",
        "display_name": "NN5 Daily",
        "frequency": "daily",
        "sample_policy": "all 111 series",
    },
    {
        "artifact_prefix": "pedestrian",
        "dataset_id": "pedestrian_hourly",
        "display_name": "Pedestrian Hourly",
        "frequency": "hourly",
        "sample_policy": "all 66 series",
    },
    {
        "artifact_prefix": "m4_hourly",
        "dataset_id": "m4_hourly",
        "display_name": "M4 Hourly",
        "frequency": "hourly",
        "sample_policy": "all 414 series",
    },
    {
        "artifact_prefix": "electricity_hourly",
        "dataset_id": "electricity_hourly",
        "display_name": "Electricity Hourly",
        "frequency": "hourly",
        "sample_policy": "all 321 series",
    },
    {
        "artifact_prefix": "weather_daily",
        "dataset_id": "weather_daily",
        "display_name": "Weather Daily",
        "frequency": "daily",
        "sample_policy": "fixed preregistered type-stratified 500-series sample",
    },
]
EXPECTED_DATASETS = 5
EXPECTED_METHODS = 14
TARGET_COVERAGE = 0.7
ALLOWED_COVERAGE_VIOLATION = 0.02
PRIMARY_METHOD = "adaptive_full_router"
STATIC_METHOD = "static_full_router"
ORACLE_METHODS = {"unconstrained_oracle", "coverage_constrained_oracle"}

METHODS = [
    "seasonal_naive",
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
    "adaptive_full_router",
    "unconstrained_oracle",
    "coverage_constrained_oracle",
]
DEPLOYABLE_METHODS = [method for method in METHODS if method not in ORACLE_METHODS]

DATASET_SUMMARY_PATH = OUTPUT_ROOT / "results/cross_dataset_dataset_summary.csv"
METHOD_SUMMARY_PATH = OUTPUT_ROOT / "results/cross_dataset_method_summary.csv"
PAIRWISE_PATH = OUTPUT_ROOT / "results/cross_dataset_primary_pairwise_tests.csv"
STATISTICAL_SUMMARY_PATH = (
    OUTPUT_ROOT / "results/cross_dataset_statistical_summary.yaml"
)
CLAIM_REGISTRY_PATH = OUTPUT_ROOT / "results/manuscript_claim_registry.yaml"
CHECKS_PATH = OUTPUT_ROOT / "results/cross_dataset_analysis_checks.csv"
FIGURE_PATH = OUTPUT_ROOT / "figures/cross_dataset_performance_and_coverage.png"
REPORT_PATH = OUTPUT_ROOT / "logs/cross_dataset_analysis_report.json"

for path in (
    DATASET_SUMMARY_PATH,
    METHOD_SUMMARY_PATH,
    PAIRWISE_PATH,
    STATISTICAL_SUMMARY_PATH,
    CLAIM_REGISTRY_PATH,
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


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def holm_adjust(p_values: np.ndarray | pd.Series) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    count = len(values)
    order = np.argsort(values, kind="stable")
    ordered_adjusted = np.empty(count, dtype=np.float64)
    running_maximum = 0.0
    for position, original_index in enumerate(order):
        candidate = min(1.0, (count - position) * values[original_index])
        running_maximum = max(running_maximum, candidate)
        ordered_adjusted[position] = running_maximum
    adjusted = np.empty(count, dtype=np.float64)
    adjusted[order] = ordered_adjusted
    return adjusted


def safe_wilcoxon(
    first: np.ndarray,
    second: np.ndarray,
    alternative: str = "two-sided",
) -> tuple[float, float]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if np.all(np.abs(first - second) <= 1e-15):
        return 0.0, 1.0
    result = wilcoxon(
        first,
        second,
        alternative=alternative,
        zero_method="wilcox",
        method="exact",
    )
    return float(result.statistic), float(result.pvalue)


def manifest_record(manifest: dict, relative_path: str) -> dict | None:
    for record in manifest.get("files", []):
        if record.get("path") == relative_path:
            return record
    return None


def np_isclose(first: float, second: float) -> bool:
    return bool(abs(first - second) <= 1e-12)


def main() -> None:
    checks: list[dict[str, object]] = []

    def record_check(name: str, passed: bool, detail: object) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    aggregate_parts: list[pd.DataFrame] = []
    evidence_cards: dict[str, dict] = {}
    closeout_ids: dict[str, str] = {}
    input_hashes: dict[str, str] = {}

    for metadata in DATASETS:
        dataset_id = metadata["dataset_id"]
        prefix = metadata["artifact_prefix"]
        manifest_path = PROJECT_ROOT / f"results/{prefix}_closeout_manifest.json"
        evidence_path = PROJECT_ROOT / f"results/{prefix}_evidence_card.yaml"
        aggregate_path = (
            PROJECT_ROOT / f"results/{prefix}_test_aggregate_metrics.csv"
        )
        required = [manifest_path, evidence_path, aggregate_path]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Closed dataset inputs are missing: {missing}")

        manifest = load_json(manifest_path)
        evidence = load_yaml(evidence_path)
        aggregate_record = manifest_record(
            manifest, f"results/{prefix}_test_aggregate_metrics.csv"
        )
        evidence_record = manifest_record(
            manifest, f"results/{prefix}_evidence_card.yaml"
        )
        closed_status_valid = bool(
            manifest.get("dataset_id") == dataset_id
            and str(manifest.get("status", "")).startswith("CLOSED_NO_FURTHER_")
            and int(manifest.get("formal_test_runs_completed", -1)) == 1
            and manifest.get("raw_test_data_reopened_for_closeout") is False
            # The legacy NN5 closeout predates this explicit manifest field;
            # its evidence card carries the same no-model-load assertion.
            and manifest.get("models_loaded_for_closeout", False) is False
            and evidence.get("dataset_id") == dataset_id
            and evidence.get("formal_test_status") == "completed_once_and_locked"
            and int(evidence.get("formal_test_run_number", -1)) == 1
            and evidence.get("raw_test_data_reopened_for_closeout") is False
            and evidence.get("models_loaded_for_closeout") is False
            and evidence.get("parameters_changed_after_test") is False
        )
        hashes_valid = bool(
            aggregate_record is not None
            and evidence_record is not None
            and sha256_file(aggregate_path) == aggregate_record["sha256"]
            and sha256_file(evidence_path) == evidence_record["sha256"]
        )
        record_check(
            f"{dataset_id}_closeout_chain_valid",
            closed_status_valid and hashes_valid,
            (
                f"status={manifest.get('status')}; files={manifest.get('file_count')}; "
                f"hashes={hashes_valid}"
            ),
        )
        if not closed_status_valid or not hashes_valid:
            raise ValueError(f"Closed evidence chain is invalid: {dataset_id}")

        aggregate = pd.read_csv(aggregate_path)
        aggregate["dataset_id"] = aggregate["dataset_id"].astype(str)
        if (
            len(aggregate) != EXPECTED_METHODS
            or set(aggregate["method"]) != set(METHODS)
            or set(aggregate["dataset_id"]) != {dataset_id}
            or not aggregate["method"].is_unique
            or not np.isfinite(aggregate["mean_RMSSE"].to_numpy(float)).all()
        ):
            raise ValueError(f"Aggregate table is invalid: {dataset_id}")
        primary = aggregate.loc[aggregate["method"] == PRIMARY_METHOD].iloc[0]
        evidence_matches = bool(
            abs(float(primary["mean_RMSSE"]) - float(evidence["primary_mean_RMSSE"]))
            <= 1e-12
            and int(primary["RMSSE_rank"])
            == int(evidence["primary_rank_among_14_methods"])
            and abs(
                float(primary["simple_coverage"])
                - float(evidence["observed_simple_coverage"])
            )
            <= 1e-12
            and abs(
                float(primary["absolute_coverage_violation"])
                - float(evidence["absolute_coverage_violation"])
            )
            <= 1e-12
            and bool(evidence["coverage_constraint_passed"])
            == bool(
                float(primary["absolute_coverage_violation"])
                <= ALLOWED_COVERAGE_VIOLATION
            )
        )
        record_check(
            f"{dataset_id}_aggregate_matches_evidence_card",
            evidence_matches,
            (
                f"RMSSE={primary['mean_RMSSE']:.6f}; rank="
                f"{int(primary['RMSSE_rank'])}; coverage="
                f"{primary['simple_coverage']:.6f}"
            ),
        )
        if not evidence_matches:
            raise ValueError(f"Aggregate and evidence card differ: {dataset_id}")

        aggregate["display_name"] = metadata["display_name"]
        aggregate["frequency"] = metadata["frequency"]
        aggregate["sample_policy"] = metadata["sample_policy"]
        aggregate_parts.append(aggregate)
        evidence_cards[dataset_id] = evidence
        closeout_ids[dataset_id] = str(manifest["closeout_id"])
        input_hashes[f"results/{prefix}_closeout_manifest.json"] = sha256_file(
            manifest_path
        )
        input_hashes[f"results/{prefix}_evidence_card.yaml"] = sha256_file(
            evidence_path
        )
        input_hashes[f"results/{prefix}_test_aggregate_metrics.csv"] = (
            sha256_file(aggregate_path)
        )

    combined = pd.concat(aggregate_parts, ignore_index=True)
    combined_valid = bool(
        len(combined) == EXPECTED_DATASETS * EXPECTED_METHODS
        and combined["dataset_id"].nunique() == EXPECTED_DATASETS
        and combined.groupby("dataset_id", observed=True)["method"].nunique()
        .eq(EXPECTED_METHODS).all()
        and not combined.duplicated(["dataset_id", "method"]).any()
    )
    record_check(
        "five_dataset_aggregate_matrix_complete",
        combined_valid,
        f"shape={combined.shape}; datasets={combined['dataset_id'].nunique()}",
    )
    if not combined_valid:
        raise AssertionError("The five-dataset aggregate matrix is incomplete")

    combined["official_rank_recomputed"] = combined.groupby(
        "dataset_id", observed=True
    )["mean_RMSSE"].rank(method="first", ascending=True)
    maximum_rank_difference = float(
        np.max(
            np.abs(
                combined["official_rank_recomputed"].to_numpy(float)
                - combined["RMSSE_rank"].to_numpy(float)
            )
        )
    )
    rank_reproduced = maximum_rank_difference <= 1e-12
    record_check(
        "all_official_method_ranks_reproduced",
        rank_reproduced,
        f"maximum_difference={maximum_rank_difference:.3e}",
    )

    deployable = combined.loc[~combined["method"].isin(ORACLE_METHODS)].copy()
    deployable["deployable_rank"] = deployable.groupby(
        "dataset_id", observed=True
    )["mean_RMSSE"].rank(method="average", ascending=True)

    dataset_records: list[dict[str, object]] = []
    for metadata in DATASETS:
        dataset_id = metadata["dataset_id"]
        table = combined.loc[combined["dataset_id"] == dataset_id].set_index("method")
        deployable_table = deployable.loc[
            deployable["dataset_id"] == dataset_id
        ].set_index("method")
        primary = table.loc[PRIMARY_METHOD]
        static = table.loc[STATIC_METHOD]
        rf = table.loc["hard_random_forest_same_features"]
        lgbm = table.loc["lightgbm_only"]
        dataset_records.append(
            {
                "dataset_id": dataset_id,
                "display_name": metadata["display_name"],
                "frequency": metadata["frequency"],
                "sample_policy": metadata["sample_policy"],
                "series_count": int(evidence_cards[dataset_id]["series_count"]),
                "formal_test_rows": int(
                    evidence_cards[dataset_id]["formal_test_rows"]
                ),
                "primary_mean_RMSSE": float(primary["mean_RMSSE"]),
                "primary_rank_all_14": int(primary["RMSSE_rank"]),
                "primary_rank_deployable_12": int(
                    deployable_table.loc[PRIMARY_METHOD, "deployable_rank"]
                ),
                "primary_simple_coverage": float(primary["simple_coverage"]),
                "primary_absolute_coverage_violation": float(
                    primary["absolute_coverage_violation"]
                ),
                "primary_coverage_constraint_passed": bool(
                    float(primary["absolute_coverage_violation"])
                    <= ALLOWED_COVERAGE_VIOLATION
                ),
                "static_mean_RMSSE": float(static["mean_RMSSE"]),
                "static_simple_coverage": float(static["simple_coverage"]),
                "static_absolute_coverage_violation": float(
                    static["absolute_coverage_violation"]
                ),
                "primary_minus_static_relative_RMSSE_percent": float(
                    100.0
                    * (float(primary["mean_RMSSE"]) / float(static["mean_RMSSE"]) - 1.0)
                ),
                "primary_coverage_deviation_reduction_vs_static": float(
                    static["absolute_coverage_violation"]
                    - primary["absolute_coverage_violation"]
                ),
                "random_forest_mean_RMSSE": float(rf["mean_RMSSE"]),
                "random_forest_simple_coverage": float(rf["simple_coverage"]),
                "lightgbm_mean_RMSSE": float(lgbm["mean_RMSSE"]),
                "closeout_id": closeout_ids[dataset_id],
            }
        )
    dataset_summary = pd.DataFrame(dataset_records)
    dataset_summary.to_csv(DATASET_SUMMARY_PATH, index=False)

    method_records: list[dict[str, object]] = []
    for method in METHODS:
        rows = combined.loc[combined["method"] == method]
        deployable_rows = deployable.loc[deployable["method"] == method]
        coverage = rows["simple_coverage"].to_numpy(dtype=np.float64)
        finite_coverage = np.isfinite(coverage)
        violations = rows.loc[finite_coverage, "absolute_coverage_violation"].to_numpy(
            dtype=np.float64
        )
        method_records.append(
            {
                "method": method,
                "deployable": method not in ORACLE_METHODS,
                "diagnostic_oracle": method in ORACLE_METHODS,
                "datasets": len(rows),
                "mean_official_rank_all_14": float(rows["RMSSE_rank"].mean()),
                "median_official_rank_all_14": float(rows["RMSSE_rank"].median()),
                "mean_deployable_rank_12": (
                    float(deployable_rows["deployable_rank"].mean())
                    if not deployable_rows.empty
                    else np.nan
                ),
                "mean_RMSSE_descriptive": float(rows["mean_RMSSE"].mean()),
                "geometric_mean_RMSSE_descriptive": float(
                    np.exp(np.mean(np.log(rows["mean_RMSSE"].to_numpy(float))))
                ),
                "coverage_evaluable_datasets": int(np.sum(finite_coverage)),
                "coverage_constraint_passes": int(
                    np.sum(violations <= ALLOWED_COVERAGE_VIOLATION)
                ),
                "mean_absolute_coverage_violation": (
                    float(np.mean(violations)) if len(violations) else np.nan
                ),
                "median_absolute_coverage_violation": (
                    float(np.median(violations)) if len(violations) else np.nan
                ),
            }
        )
    method_summary = pd.DataFrame(method_records).sort_values(
        ["mean_official_rank_all_14", "method"], kind="stable"
    ).reset_index(drop=True)
    method_summary["cross_dataset_average_rank_order"] = np.arange(
        1, len(method_summary) + 1
    )
    method_summary.to_csv(METHOD_SUMMARY_PATH, index=False)

    deployable_wide = deployable.pivot(
        index="dataset_id", columns="method", values="mean_RMSSE"
    ).loc[[item["dataset_id"] for item in DATASETS], DEPLOYABLE_METHODS]
    friedman = friedmanchisquare(
        *[
            deployable_wide[method].to_numpy(dtype=np.float64)
            for method in DEPLOYABLE_METHODS
        ]
    )
    friedman_statistic = float(friedman.statistic)
    friedman_p_value = float(friedman.pvalue)
    friedman_kendall_w = float(
        friedman_statistic / (EXPECTED_DATASETS * (len(DEPLOYABLE_METHODS) - 1))
    )
    omnibus_gate_passed = friedman_p_value < 0.05

    deployable_rank_wide = deployable.pivot(
        index="dataset_id", columns="method", values="deployable_rank"
    ).loc[[item["dataset_id"] for item in DATASETS], DEPLOYABLE_METHODS]
    primary_ranks = deployable_rank_wide[PRIMARY_METHOD].to_numpy(np.float64)
    primary_rmsse = deployable_wide[PRIMARY_METHOD].to_numpy(np.float64)
    pairwise_records: list[dict[str, object]] = []
    for comparator in DEPLOYABLE_METHODS:
        if comparator == PRIMARY_METHOD:
            continue
        comparator_ranks = deployable_rank_wide[comparator].to_numpy(np.float64)
        comparator_rmsse = deployable_wide[comparator].to_numpy(np.float64)
        statistic, p_value = safe_wilcoxon(primary_ranks, comparator_ranks)
        relative_difference = primary_rmsse / comparator_rmsse - 1.0
        pairwise_records.append(
            {
                "primary_method": PRIMARY_METHOD,
                "comparator": comparator,
                "datasets": EXPECTED_DATASETS,
                "primary_mean_deployable_rank": float(np.mean(primary_ranks)),
                "comparator_mean_deployable_rank": float(np.mean(comparator_ranks)),
                "mean_rank_difference_primary_minus_comparator": float(
                    np.mean(primary_ranks - comparator_ranks)
                ),
                "primary_accuracy_wins": int(np.sum(primary_rmsse < comparator_rmsse)),
                "primary_accuracy_ties": int(np.sum(primary_rmsse == comparator_rmsse)),
                "primary_accuracy_losses": int(np.sum(primary_rmsse > comparator_rmsse)),
                "mean_relative_RMSSE_difference": float(np.mean(relative_difference)),
                "median_relative_RMSSE_difference": float(
                    np.median(relative_difference)
                ),
                "wilcoxon_rank_statistic": statistic,
                "raw_two_sided_p_value": p_value,
            }
        )
    pairwise = pd.DataFrame(pairwise_records)
    pairwise["holm_adjusted_p_value"] = holm_adjust(
        pairwise["raw_two_sided_p_value"]
    )
    pairwise["significant_after_friedman_and_holm"] = (
        omnibus_gate_passed & (pairwise["holm_adjusted_p_value"] < 0.05)
    )
    pairwise["descriptive_direction"] = np.where(
        pairwise["mean_rank_difference_primary_minus_comparator"] < 0.0,
        "primary_better_average_rank",
        np.where(
            pairwise["mean_rank_difference_primary_minus_comparator"] > 0.0,
            "comparator_better_average_rank",
            "equal_average_rank",
        ),
    )
    pairwise = pairwise.sort_values(
        ["holm_adjusted_p_value", "comparator"], kind="stable"
    ).reset_index(drop=True)
    pairwise.to_csv(PAIRWISE_PATH, index=False)

    primary_deviation = dataset_summary[
        "primary_absolute_coverage_violation"
    ].to_numpy(np.float64)
    static_deviation = dataset_summary[
        "static_absolute_coverage_violation"
    ].to_numpy(np.float64)
    coverage_statistic, coverage_two_sided_p = safe_wilcoxon(
        primary_deviation, static_deviation
    )
    _, coverage_exploratory_one_sided_p = safe_wilcoxon(
        primary_deviation, static_deviation, alternative="less"
    )
    primary_coverage_better_count = int(np.sum(primary_deviation < static_deviation))
    primary_accuracy_better_static_count = int(
        np.sum(
            dataset_summary["primary_mean_RMSSE"].to_numpy(float)
            < dataset_summary["static_mean_RMSSE"].to_numpy(float)
        )
    )

    primary_summary = method_summary.loc[
        method_summary["method"] == PRIMARY_METHOD
    ].iloc[0]
    static_summary = method_summary.loc[
        method_summary["method"] == STATIC_METHOD
    ].iloc[0]
    coverage_pass_count = int(
        dataset_summary["primary_coverage_constraint_passed"].sum()
    )
    mean_accuracy_cost_vs_static = float(
        dataset_summary["primary_minus_static_relative_RMSSE_percent"].mean()
    )
    median_accuracy_cost_vs_static = float(
        dataset_summary["primary_minus_static_relative_RMSSE_percent"].median()
    )

    no_cross_dataset_pairwise_significance = bool(
        not pairwise["significant_after_friedman_and_holm"].any()
    )
    record_check(
        "preregistered_friedman_and_holm_analysis_complete",
        bool(
            np.isfinite(friedman_statistic)
            and np.isfinite(friedman_p_value)
            and len(pairwise) == len(DEPLOYABLE_METHODS) - 1
        ),
        (
            f"Friedman={friedman_statistic:.6f}; p={friedman_p_value:.6g}; "
            f"pairwise={len(pairwise)}"
        ),
    )
    record_check(
        "primary_coverage_summary_reproduced",
        bool(
            coverage_pass_count == 4
            and primary_coverage_better_count == EXPECTED_DATASETS
            and np_isclose(float(primary_summary["mean_official_rank_all_14"]), 9.2)
            and np_isclose(float(primary_summary["mean_deployable_rank_12"]), 7.2)
        ),
        (
            f"passes={coverage_pass_count}/5; better_than_static="
            f"{primary_coverage_better_count}/5; mean_rank="
            f"{primary_summary['mean_official_rank_all_14']:.3f}"
        ),
    )
    record_check(
        "synthesis_did_not_open_detailed_test_results_or_models",
        True,
        "only closeout manifests, evidence cards, and aggregate tables parsed",
    )

    statistical_summary = {
        "analysis_scope": "five_closed_datasets",
        "dataset_count": EXPECTED_DATASETS,
        "dataset_ids": [item["dataset_id"] for item in DATASETS],
        "analysis_unit": "dataset",
        "methods_in_accuracy_omnibus": DEPLOYABLE_METHODS,
        "diagnostic_oracles_excluded_from_omnibus": sorted(ORACLE_METHODS),
        "friedman_statistic": friedman_statistic,
        "friedman_degrees_of_freedom": len(DEPLOYABLE_METHODS) - 1,
        "friedman_p_value": friedman_p_value,
        "friedman_kendall_w": friedman_kendall_w,
        "friedman_omnibus_significant_at_0_05": omnibus_gate_passed,
        "holm_pairwise_family_size": len(pairwise),
        "holm_significant_primary_comparisons": pairwise.loc[
            pairwise["significant_after_friedman_and_holm"], "comparator"
        ].tolist(),
        "no_primary_pairwise_claim_due_to_small_dataset_count": (
            no_cross_dataset_pairwise_significance
        ),
        "primary_method": PRIMARY_METHOD,
        "primary_mean_official_rank_all_14": float(
            primary_summary["mean_official_rank_all_14"]
        ),
        "primary_mean_deployable_rank_12": float(
            primary_summary["mean_deployable_rank_12"]
        ),
        "primary_accuracy_rank_by_dataset_all_14": {
            str(row.dataset_id): int(row.primary_rank_all_14)
            for row in dataset_summary.itertuples(index=False)
        },
        "primary_coverage_constraint_passes": coverage_pass_count,
        "primary_coverage_constraint_failures": EXPECTED_DATASETS - coverage_pass_count,
        "coverage_failure_dataset": "nn5",
        "primary_coverage_deviation_better_than_static_datasets": (
            primary_coverage_better_count
        ),
        "coverage_deviation_paired_wilcoxon_two_sided": {
            "statistic": coverage_statistic,
            "p_value": coverage_two_sided_p,
            "confirmatory_significant_at_0_05": coverage_two_sided_p < 0.05,
        },
        "coverage_deviation_exploratory_one_sided_p_value": (
            coverage_exploratory_one_sided_p
        ),
        "primary_accuracy_better_than_static_datasets": (
            primary_accuracy_better_static_count
        ),
        "primary_mean_relative_RMSSE_cost_vs_static_percent": (
            mean_accuracy_cost_vs_static
        ),
        "primary_median_relative_RMSSE_cost_vs_static_percent": (
            median_accuracy_cost_vs_static
        ),
        "interpretation": (
            "The adaptive controller consistently improved target adherence relative "
            "to the static router, but did not establish cross-dataset accuracy "
            "superiority and usually paid a small accuracy cost."
        ),
        "raw_data_opened": False,
        "per_series_results_opened": False,
        "per_timestamp_predictions_opened": False,
        "models_loaded": False,
    }
    STATISTICAL_SUMMARY_PATH.write_text(
        yaml.safe_dump(statistical_summary, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    claim_registry = {
        "study_status": "all_five_datasets_closed_no_further_model_changes",
        "primary_method": PRIMARY_METHOD,
        "paper_positioning": (
            "coverage-controlled model routing with an explicit and audited "
            "accuracy-coverage trade-off; not a state-of-the-art accuracy claim"
        ),
        "supported_primary_claims": [
            (
                "The adaptive controller satisfied the registered two-percentage-"
                "point coverage tolerance on four of five datasets."
            ),
            (
                "Its absolute coverage deviation was smaller than the static full "
                "router on all five datasets."
            ),
            (
                "This five-of-five direction is descriptive: the preregistered "
                f"two-sided paired Wilcoxon p-value was {coverage_two_sided_p:.4f}, "
                "so it did not cross the 0.05 threshold with only five blocks."
            ),
            (
                "The primary method's mean rank was 9.2 among all 14 methods and "
                "7.2 among the 12 deployable methods."
            ),
            (
                f"Compared with static routing, adaptive routing improved RMSSE on "
                f"{primary_accuracy_better_static_count} of 5 datasets and had a "
                f"mean relative RMSSE change of {mean_accuracy_cost_vs_static:.3f} percent."
            ),
            (
                "The NN5 dataset is the registered coverage failure and must be "
                "reported as such; the other four datasets passed."
            ),
        ],
        "supported_secondary_claims": [
            (
                "Per-dataset paired inference, rather than the five-block omnibus, "
                "provides the main uncertainty evidence because each dataset contains "
                "66 to 500 paired series."
            ),
            (
                "Weather Daily used a fixed type-stratified 500-series sample chosen "
                "without performance information."
            ),
            (
                "Every formal test was executed once after parameter and evaluator "
                "freezing, and every dataset is cryptographically closed."
            ),
        ],
        "unsupported_claims": [
            "The proposed method achieved state-of-the-art forecasting accuracy.",
            "The proposed method was the most accurate method across datasets.",
            "The proposed method significantly outperformed all baselines across datasets.",
            "The coverage constraint was satisfied on every dataset.",
            (
                "Adaptive routing significantly improved coverage over static routing "
                "in a confirmatory two-sided five-dataset test."
            ),
            "Oracle methods are deployable competitors.",
            "All 3010 Weather series were evaluated in the formal experiment.",
            "Test results may be rerun, retuned, filtered, or replaced.",
        ],
        "mandatory_exceptions_and_limitations": [
            (
                "NN5 missed the target with Ridge coverage 0.668332 and absolute "
                "deviation 0.031668."
            ),
            (
                "M4 Hourly passed narrowly with coverage 0.718746 and deviation "
                "0.018746, close to the 0.02 tolerance."
            ),
            (
                "Adaptive routing ranked 12 of 14 on M4 Hourly and 10 of 14 on "
                "Electricity Hourly and Weather Daily."
            ),
            (
                "Only five dataset blocks were available, limiting power for "
                "cross-dataset pairwise tests."
            ),
            (
                "Weather conclusions apply to the preregistered 500-series sample, "
                "not automatically to the complete 3010-series archive."
            ),
        ],
        "recommended_title_direction": (
            "Residual-aware soft routing with adaptive coverage control for "
            "time-series forecasting"
        ),
        "recommended_abstract_conclusion": (
            "Adaptive coverage control improved adherence to the requested simple-"
            "model usage rate across heterogeneous datasets, while forecasting "
            "accuracy remained competitive in some settings but was not uniformly "
            "superior, revealing a measurable accuracy-control trade-off."
        ),
    }
    CLAIM_REGISTRY_PATH.write_text(
        yaml.safe_dump(claim_registry, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    checks_frame = pd.DataFrame(checks)
    failed_checks = checks_frame.loc[~checks_frame["passed"]]
    if not failed_checks.empty:
        raise AssertionError(
            f"Cross-dataset synthesis checks failed: {failed_checks.to_dict('records')}"
        )
    checks_frame.to_csv(CHECKS_PATH, index=False)

    display_order = [item["display_name"] for item in DATASETS]
    plot_dataset = dataset_summary.set_index("display_name").loc[display_order]
    deployable_plot = method_summary.loc[method_summary["deployable"]].sort_values(
        "mean_deployable_rank_12", ascending=False
    )
    figure, axes = plt.subplots(2, 2, figsize=(18, 13))

    axes[0, 0].barh(
        deployable_plot["method"],
        deployable_plot["mean_deployable_rank_12"],
        color=np.where(
            deployable_plot["method"].eq(PRIMARY_METHOD), "#e15759", "#4c78a8"
        ),
    )
    axes[0, 0].set_xlabel("Mean rank across five datasets (lower is better)")
    axes[0, 0].set_title("Deployable-method accuracy ranks")

    x = np.arange(EXPECTED_DATASETS)
    width = 0.36
    axes[0, 1].bar(
        x - width / 2,
        plot_dataset["primary_absolute_coverage_violation"],
        width,
        label="adaptive_full_router",
        color="#e15759",
    )
    axes[0, 1].bar(
        x + width / 2,
        plot_dataset["static_absolute_coverage_violation"],
        width,
        label="static_full_router",
        color="#f28e2b",
    )
    axes[0, 1].axhline(
        ALLOWED_COVERAGE_VIOLATION, color="black", linestyle="--", label="tolerance"
    )
    axes[0, 1].set_xticks(x, display_order, rotation=20, ha="right")
    axes[0, 1].set_ylabel("Absolute Ridge-coverage deviation")
    axes[0, 1].set_title("Target-adherence trade-off")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].bar(
        x,
        plot_dataset["primary_minus_static_relative_RMSSE_percent"],
        color=np.where(
            plot_dataset["primary_minus_static_relative_RMSSE_percent"] <= 0.0,
            "#59a14f",
            "#e15759",
        ),
    )
    axes[1, 0].axhline(0.0, color="black", linestyle="--")
    axes[1, 0].set_xticks(x, display_order, rotation=20, ha="right")
    axes[1, 0].set_ylabel("Adaptive minus static relative RMSSE (%)")
    axes[1, 0].set_title("Accuracy cost of adaptive control")

    heat = deployable.pivot(
        index="method", columns="display_name", values="deployable_rank"
    ).loc[DEPLOYABLE_METHODS, display_order]
    image = axes[1, 1].imshow(heat.to_numpy(float), cmap="viridis_r", aspect="auto")
    axes[1, 1].set_xticks(np.arange(EXPECTED_DATASETS), display_order, rotation=20, ha="right")
    axes[1, 1].set_yticks(np.arange(len(DEPLOYABLE_METHODS)), DEPLOYABLE_METHODS)
    axes[1, 1].set_title("Per-dataset deployable ranks")
    for row_index in range(len(DEPLOYABLE_METHODS)):
        for column_index in range(EXPECTED_DATASETS):
            axes[1, 1].text(
                column_index,
                row_index,
                f"{int(heat.iloc[row_index, column_index])}",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if heat.iloc[row_index, column_index] > 6 else "white",
            )
    figure.colorbar(image, ax=axes[1, 1], label="Rank")
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)

    output_hashes = {
        str(path.relative_to(OUTPUT_ROOT)): sha256_file(path)
        for path in (
            DATASET_SUMMARY_PATH,
            METHOD_SUMMARY_PATH,
            PAIRWISE_PATH,
            STATISTICAL_SUMMARY_PATH,
            CLAIM_REGISTRY_PATH,
            CHECKS_PATH,
            FIGURE_PATH,
        )
    }
    report = {
        "status": "passed",
        "analysis_scope": "five_closed_datasets",
        "dataset_count": EXPECTED_DATASETS,
        "input_closeout_ids": closeout_ids,
        "input_sha256": input_hashes,
        "check_count": len(checks),
        "failed_check_count": 0,
        "primary_method": PRIMARY_METHOD,
        "primary_mean_official_rank": float(
            primary_summary["mean_official_rank_all_14"]
        ),
        "primary_coverage_passes": coverage_pass_count,
        "primary_coverage_better_than_static_datasets": (
            primary_coverage_better_count
        ),
        "friedman_p_value": friedman_p_value,
        "holm_significant_primary_comparisons": pairwise.loc[
            pairwise["significant_after_friedman_and_holm"], "comparator"
        ].tolist(),
        "raw_data_opened": False,
        "per_series_results_opened": False,
        "per_timestamp_predictions_opened": False,
        "models_loaded": False,
        "output_sha256": output_hashes,
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("五数据集正式结果综合分析全部通过")
    print("已结项数据集数量：", EXPECTED_DATASETS)
    print("读取原始数据：否")
    print("读取逐序列或逐时点结果：否")
    print("加载模型：否")
    print("主方法五数据集正式排名：", dataset_summary["primary_rank_all_14"].tolist())
    print("主方法平均正式排名：", f"{primary_summary['mean_official_rank_all_14']:.3f}")
    print("主方法平均可部署方法排名：", f"{primary_summary['mean_deployable_rank_12']:.3f}")
    print("覆盖率约束通过：", f"{coverage_pass_count}/5")
    print("覆盖偏差优于静态路由：", f"{primary_coverage_better_count}/5")
    print("精度优于静态路由：", f"{primary_accuracy_better_static_count}/5")
    print("相对静态路由平均RMSSE变化：", f"{mean_accuracy_cost_vs_static:.3f}%")
    print("Friedman统计量：", f"{friedman_statistic:.6f}")
    print("Friedman p值：", f"{friedman_p_value:.6g}")
    print(
        "Friedman+Holm后主方法显著比较：",
        pairwise.loc[
            pairwise["significant_after_friedman_and_holm"], "comparator"
        ].tolist(),
    )
    print("论文定位：覆盖率控制与精度权衡，不宣称精度最优")
    print("数据集总表：", DATASET_SUMMARY_PATH)
    print("方法总表：", METHOD_SUMMARY_PATH)
    print("跨数据集配对检验：", PAIRWISE_PATH)
    print("统计摘要：", STATISTICAL_SUMMARY_PATH)
    print("论文主张登记表：", CLAIM_REGISTRY_PATH)
    print("检查表：", CHECKS_PATH)
    print("综合图片：", FIGURE_PATH)
    print("分析报告：", REPORT_PATH)


if __name__ == "__main__":
    main()
