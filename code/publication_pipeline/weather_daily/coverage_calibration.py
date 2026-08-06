#!/usr/bin/env python3
"""Calibration-only coverage controller for Weather Daily."""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sci-routing")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(
    os.environ.get("SCI_ROUTING_ROOT", Path(__file__).resolve().parents[3])
).resolve()
OUTPUT_ROOT = Path(os.environ.get("SCI_ROUTING_OUTPUT_ROOT", PROJECT_ROOT)).resolve()

CONFIG_PATH = PROJECT_ROOT / "experiment_config.yaml"
SCORE_PATH = PROJECT_ROOT / "results/weather_daily_calibration_router_scores.parquet"
LOCAL_PARAMETER_PATH = PROJECT_ROOT / "results/weather_daily_selected_local_weighting_params.yaml"
LOCAL_CHECKS_PATH = PROJECT_ROOT / "results/weather_daily_local_weighting_checks.csv"
SPLIT_MANIFEST_PATH = PROJECT_ROOT / "results/weather_daily_split_manifest.csv"

TUNING_PATH = OUTPUT_ROOT / "results/weather_daily_coverage_controller_tuning.csv"
TRACE_PATH = OUTPUT_ROOT / "results/weather_daily_calibration_controller_trace.csv"
DECISION_PATH = OUTPUT_ROOT / "results/weather_daily_calibration_controller_decisions.parquet"
THRESHOLD_PATH = OUTPUT_ROOT / "results/weather_daily_calibration_thresholds.csv"
SELECTED_PATH = OUTPUT_ROOT / "results/weather_daily_selected_coverage_controller.yaml"
CHECKS_PATH = OUTPUT_ROOT / "results/weather_daily_coverage_calibration_checks.csv"
FIGURE_PATH = OUTPUT_ROOT / "figures/weather_daily_coverage_calibration.png"
REPORT_PATH = OUTPUT_ROOT / "logs/weather_daily_coverage_calibration_report.json"

DATASET_ID = "weather_daily"
SAMPLE_ID = "5518ada55d6adb8a8df20f5a30d52ec904362ce57df9adeeceb0bc3d22854044"
EXPECTED_ROWS = 736_875
EXPECTED_SERIES = 500
EXPECTED_RELATIVE_STEPS = 5_867
EXPECTED_MINIMUM_SERIES_LENGTH = 134
EXPECTED_MAXIMUM_SERIES_LENGTH = 5_867
EXPECTED_MINIMUM_BATCH_ROWS = 1
EXPECTED_MAXIMUM_BATCH_ROWS = 500
ROLLING_WINDOW_DAYS = 56
ALIGNMENT_RULE = "days elapsed since each series entered calibration"


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def passed_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1"})


def simulate_controller(
    calibration: pd.DataFrame,
    eta: float,
    base_threshold: float,
    target_coverage: float,
    bias_limit: float,
    return_decisions: bool = False,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame | None]:
    """Replay the controller; its state update never uses observed targets."""

    bias = 0.0
    total_simple_count = 0
    total_rows = 0
    total_scaled_loss = 0.0
    trace_records: list[dict[str, object]] = []
    decision_frames: list[pd.DataFrame] = []
    per_row_loss: list[np.ndarray] = []
    per_row_simple: list[np.ndarray] = []
    per_row_series: list[np.ndarray] = []

    for relative_step, batch in calibration.groupby(
        "relative_calibration_step", sort=True, observed=True
    ):
        batch_probability = batch["black_box_probability"].to_numpy(np.float64)
        threshold_before_decision = float(np.clip(base_threshold + bias, 0.0, 1.0))
        use_simple = batch_probability < threshold_before_decision
        simple_count = int(np.sum(use_simple))
        batch_rows = int(len(batch))
        batch_coverage = float(simple_count / batch_rows)
        selected_prediction = np.where(
            use_simple,
            batch["ridge_prediction"].to_numpy(np.float64),
            batch["lightgbm_prediction"].to_numpy(np.float64),
        )
        scaled_squared_error = (
            (
                batch["y_true"].to_numpy(np.float64) - selected_prediction
            )
            / batch["seasonal_naive_mae_scale"].to_numpy(np.float64)
        ) ** 2
        total_simple_count += simple_count
        total_rows += batch_rows
        total_scaled_loss += float(np.sum(scaled_squared_error))
        cumulative_coverage = float(total_simple_count / total_rows)

        # Only the realized routing decisions enter this update.
        next_bias = float(
            np.clip(
                bias + eta * (target_coverage - batch_coverage),
                -bias_limit,
                bias_limit,
            )
        )
        trace_records.append(
            {
                "dataset_id": DATASET_ID,
                "sample_id": SAMPLE_ID,
                "relative_calibration_step": int(relative_step),
                "alignment_rule": ALIGNMENT_RULE,
                "eta": float(eta),
                "base_threshold": float(base_threshold),
                "bias_before_decision": float(bias),
                "effective_threshold": threshold_before_decision,
                "simple_count": simple_count,
                "batch_rows": batch_rows,
                "simple_coverage": batch_coverage,
                "cumulative_simple_count": total_simple_count,
                "cumulative_rows": total_rows,
                "cumulative_simple_coverage": cumulative_coverage,
                "batch_scaled_loss": float(np.mean(scaled_squared_error)),
                "bias_after_update": next_bias,
            }
        )
        per_row_loss.append(scaled_squared_error)
        per_row_simple.append(use_simple.astype(np.float64))
        per_row_series.append(batch["series_id"].to_numpy())

        if return_decisions:
            decision = batch[
                [
                    "dataset_id",
                    "sample_id",
                    "series_id",
                    "series_type",
                    "time_index",
                    "split",
                    "black_box_probability",
                    "y_true",
                    "ridge_prediction",
                    "lightgbm_prediction",
                    "seasonal_naive_mae_scale",
                ]
            ].copy()
            decision["relative_calibration_step"] = int(relative_step)
            decision["base_threshold"] = float(base_threshold)
            decision["bias_before_decision"] = float(bias)
            decision["effective_threshold"] = threshold_before_decision
            decision["use_ridge"] = use_simple
            decision["selected_prediction"] = selected_prediction
            decision["scaled_squared_error"] = scaled_squared_error
            decision["bias_after_update"] = next_bias
            decision_frames.append(decision)
        bias = next_bias

    trace = pd.DataFrame(trace_records)
    rolling_simple_count = trace["simple_count"].rolling(
        ROLLING_WINDOW_DAYS, min_periods=ROLLING_WINDOW_DAYS
    ).sum()
    rolling_rows = trace["batch_rows"].rolling(
        ROLLING_WINDOW_DAYS, min_periods=ROLLING_WINDOW_DAYS
    ).sum()
    rolling_name = f"rolling_{ROLLING_WINDOW_DAYS}d_simple_coverage"
    trace[rolling_name] = rolling_simple_count / rolling_rows
    valid_rolling = trace[rolling_name].dropna()
    if valid_rolling.empty:
        worst_rolling_violation = float(
            abs(total_simple_count / total_rows - target_coverage)
        )
    else:
        worst_rolling_violation = float(
            np.max(np.abs(valid_rolling.to_numpy() - target_coverage))
        )

    all_loss = np.concatenate(per_row_loss)
    all_simple = np.concatenate(per_row_simple)
    all_series = np.concatenate(per_row_series)
    per_series = pd.DataFrame(
        {"series_id": all_series, "scaled_loss": all_loss, "simple": all_simple}
    ).groupby("series_id", sort=False, observed=True).agg(
        mean_scaled_loss=("scaled_loss", "mean"),
        simple_coverage=("simple", "mean"),
    )
    overall_coverage = float(total_simple_count / total_rows)
    summary: dict[str, object] = {
        "eta": float(eta),
        "base_threshold": float(base_threshold),
        "target_simple_coverage": float(target_coverage),
        "achieved_simple_coverage": overall_coverage,
        "absolute_coverage_violation": float(abs(overall_coverage - target_coverage)),
        "equal_series_simple_coverage": float(per_series["simple_coverage"].mean()),
        "mean_per_series_absolute_coverage_violation": float(
            np.mean(np.abs(per_series["simple_coverage"].to_numpy() - target_coverage))
        ),
        "worst_per_series_absolute_coverage_violation": float(
            np.max(np.abs(per_series["simple_coverage"].to_numpy() - target_coverage))
        ),
        "equal_series_mean_scaled_loss": float(
            per_series["mean_scaled_loss"].mean()
        ),
        "pooled_mean_scaled_loss": float(total_scaled_loss / total_rows),
        f"worst_{ROLLING_WINDOW_DAYS}d_weighted_coverage_violation": worst_rolling_violation,
        "mean_batch_absolute_coverage_violation": float(
            np.mean(np.abs(trace["simple_coverage"].to_numpy() - target_coverage))
        ),
        "final_bias": float(bias),
        "maximum_absolute_bias": float(
            np.max(np.abs(trace["bias_before_decision"].to_numpy()))
        ),
        "calibration_rows": int(total_rows),
        "calibration_series": int(per_series.shape[0]),
        "calibration_relative_steps": int(len(trace)),
        "minimum_batch_rows": int(trace["batch_rows"].min()),
        "maximum_batch_rows": int(trace["batch_rows"].max()),
    }
    decisions = pd.concat(decision_frames, ignore_index=True) if return_decisions else None
    return summary, trace, decisions


def main() -> None:
    input_paths = (
        CONFIG_PATH,
        SCORE_PATH,
        LOCAL_PARAMETER_PATH,
        LOCAL_CHECKS_PATH,
        SPLIT_MANIFEST_PATH,
    )
    for path in input_paths:
        require_file(path)
    output_paths = (
        TUNING_PATH,
        TRACE_PATH,
        DECISION_PATH,
        THRESHOLD_PATH,
        SELECTED_PATH,
        CHECKS_PATH,
        FIGURE_PATH,
        REPORT_PATH,
    )
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    start = perf_counter()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    local_parameters = yaml.safe_load(LOCAL_PARAMETER_PATH.read_text(encoding="utf-8"))
    local_checks = pd.read_csv(LOCAL_CHECKS_PATH)
    split_manifest = pd.read_csv(SPLIT_MANIFEST_PATH).sort_values("sample_order")
    if not passed_column(local_checks["passed"]).all():
        raise AssertionError("Local-weighting checks did not all pass")
    if local_parameters["selection_scope"] != "router_train_only":
        raise AssertionError("Router parameters were not selected in router_train")
    if bool(local_parameters.get("calibration_used_for_selection", True)):
        raise AssertionError("Router selection used calibration")
    if bool(local_parameters.get("calibration_used_for_model_fit", True)):
        raise AssertionError("Router fitting used calibration")
    if bool(local_parameters.get("test_accessed", True)):
        raise AssertionError("Upstream router step accessed test")
    sample_ids = {
        str(local_parameters.get("sample_id")),
        *split_manifest["sample_id"].astype(str).unique().tolist(),
    }
    if sample_ids != {SAMPLE_ID}:
        raise AssertionError(f"Frozen sample IDs disagree: {sample_ids}")

    seed = int(config["study"]["seed"])
    primary_coverage = float(config["study"]["primary_target_coverage"])
    sensitivity_coverages = [
        float(value) for value in config["study"]["sensitivity_target_coverages"]
    ]
    eta_grid = [
        float(value) for value in config["coverage_controller"]["eta_grid"]
    ]
    bias_limit = float(config["coverage_controller"]["bias_limit"])
    allowed_violation = float(
        config["coverage_controller"]["allowed_absolute_violation"]
    )
    if primary_coverage != 0.7:
        raise AssertionError(f"Unexpected primary coverage: {primary_coverage}")
    if sensitivity_coverages != [0.5, 0.9]:
        raise AssertionError(f"Unexpected sensitivity coverages: {sensitivity_coverages}")
    if eta_grid != [0.001, 0.005, 0.01, 0.05]:
        raise AssertionError(f"Unexpected eta grid: {eta_grid}")
    if bias_limit != 0.5 or allowed_violation != 0.02:
        raise AssertionError(f"Unexpected controller limits: {bias_limit}, {allowed_violation}")

    calibration = pd.read_parquet(SCORE_PATH)
    required_columns = [
        "dataset_id",
        "sample_id",
        "series_id",
        "series_type",
        "time_index",
        "split",
        "y_true",
        "ridge_prediction",
        "lightgbm_prediction",
        "seasonal_naive_mae_scale",
        "black_box_probability",
    ]
    missing_columns = [name for name in required_columns if name not in calibration]
    if missing_columns:
        raise AssertionError(f"Calibration scores lack columns: {missing_columns}")
    if calibration[required_columns].isna().any().any():
        raise AssertionError("Calibration scores contain missing values")
    observed_splits = set(calibration["split"].astype(str).unique())
    if calibration.empty or observed_splits != {"calibration"}:
        raise AssertionError(f"Calibration score scope is invalid: {observed_splits}")
    if set(calibration["dataset_id"].astype(str).unique()) != {DATASET_ID}:
        raise AssertionError("Calibration dataset_id is invalid")
    if set(calibration["sample_id"].astype(str).unique()) != {SAMPLE_ID}:
        raise AssertionError("Calibration sample_id is invalid")

    sample_order_map = split_manifest.set_index("series_id")["sample_order"].to_dict()
    calibration["series_id"] = calibration["series_id"].astype(str)
    calibration["_series_order"] = calibration["series_id"].map(sample_order_map)
    if calibration["_series_order"].isna().any():
        raise AssertionError("Calibration contains an unregistered series")
    calibration = (
        calibration.sort_values(["_series_order", "time_index"], kind="stable")
        .drop(columns="_series_order")
        .reset_index(drop=True)
    )
    calibration["relative_calibration_step"] = (
        calibration.groupby("series_id", sort=False, observed=True)
        .cumcount()
        .astype(np.int32)
    )
    per_series_lengths = calibration.groupby(
        "series_id", sort=False, observed=True
    ).size()
    expected_lengths = {
        str(row.series_id): int(row.calibration_count)
        for row in split_manifest.itertuples(index=False)
    }
    observed_lengths = {str(key): int(value) for key, value in per_series_lengths.items()}
    variable_lengths_match_manifest = observed_lengths == expected_lengths
    per_series_contiguous = True
    for _, group in calibration.groupby("series_id", sort=False, observed=True):
        indices = group["time_index"].to_numpy(np.int64)
        per_series_contiguous &= bool(np.all(np.diff(indices) == 1))

    probability = calibration["black_box_probability"].to_numpy(np.float64)
    probability_valid = bool(
        np.isfinite(probability).all()
        and np.all(probability >= 0.0)
        and np.all(probability <= 1.0)
    )
    scales = calibration["seasonal_naive_mae_scale"].to_numpy(np.float64)
    scales_positive = bool(np.isfinite(scales).all() and np.all(scales > 0.0))

    expected_active_counts = np.asarray(
        [
            sum(length > relative_step for length in expected_lengths.values())
            for relative_step in range(max(expected_lengths.values()))
        ],
        dtype=np.int64,
    )
    observed_active_counts = (
        calibration.groupby("relative_calibration_step", observed=True)
        .size()
        .sort_index()
        .to_numpy(np.int64)
    )
    active_counts_match_manifest = np.array_equal(
        observed_active_counts, expected_active_counts
    )

    all_target_coverages = sorted(set([primary_coverage] + sensitivity_coverages))
    threshold_records: list[dict[str, object]] = []
    for target_coverage in all_target_coverages:
        threshold = float(np.quantile(probability, target_coverage))
        achieved_coverage = float(np.mean(probability < threshold))
        threshold_tie_rows = int(np.sum(probability == threshold))
        threshold_records.append(
            {
                "dataset_id": DATASET_ID,
                "sample_id": SAMPLE_ID,
                "target_simple_coverage": target_coverage,
                "base_threshold": threshold,
                "decision_rule": "use_ridge_if_probability_below_threshold",
                "achieved_static_calibration_coverage": achieved_coverage,
                "absolute_static_coverage_violation": abs(
                    achieved_coverage - target_coverage
                ),
                "threshold_tie_rows": threshold_tie_rows,
                "threshold_tie_fraction": threshold_tie_rows / len(calibration),
                "calibration_rows": len(calibration),
            }
        )
    thresholds = pd.DataFrame(threshold_records)
    primary_threshold = float(
        thresholds.loc[
            np.isclose(thresholds["target_simple_coverage"], primary_coverage),
            "base_threshold",
        ].iloc[0]
    )

    rolling_key = f"worst_{ROLLING_WINDOW_DAYS}d_weighted_coverage_violation"
    candidate_summaries: list[dict[str, object]] = []
    for eta in eta_grid:
        candidate_summary, _, _ = simulate_controller(
            calibration,
            eta,
            primary_threshold,
            primary_coverage,
            bias_limit,
            return_decisions=False,
        )
        candidate_summary["feasible"] = bool(
            float(candidate_summary["absolute_coverage_violation"])
            <= allowed_violation
        )
        candidate_summaries.append(candidate_summary)
    tuning = pd.DataFrame(candidate_summaries).sort_values("eta").reset_index(drop=True)
    feasible = tuning.loc[tuning["feasible"]].copy()
    if feasible.empty:
        selection_status = "no_feasible_candidate_minimum_violation_fallback"
        ranked = tuning.sort_values(
            [
                "absolute_coverage_violation",
                "equal_series_mean_scaled_loss",
                rolling_key,
                "eta",
            ],
            kind="stable",
        )
    else:
        selection_status = "feasible_minimum_equal_series_forecast_loss"
        ranked = feasible.sort_values(
            [
                "equal_series_mean_scaled_loss",
                rolling_key,
                "absolute_coverage_violation",
                "eta",
            ],
            kind="stable",
        )
    best = ranked.iloc[0]
    selected_eta = float(best["eta"])
    selected_summary, selected_trace, selected_decisions = simulate_controller(
        calibration,
        selected_eta,
        primary_threshold,
        primary_coverage,
        bias_limit,
        return_decisions=True,
    )
    static_summary, _, _ = simulate_controller(
        calibration,
        0.0,
        primary_threshold,
        primary_coverage,
        bias_limit,
        return_decisions=False,
    )
    if selected_decisions is None:
        raise AssertionError("Selected controller decisions were not produced")

    decision_coverage = float(selected_decisions["use_ridge"].mean())
    trace_coverage = float(
        selected_trace["simple_count"].sum() / selected_trace["batch_rows"].sum()
    )
    expected_primary_static_coverage = float(
        thresholds.loc[
            np.isclose(thresholds["target_simple_coverage"], primary_coverage),
            "achieved_static_calibration_coverage",
        ].iloc[0]
    )
    primary_threshold_tie_fraction = float(
        thresholds.loc[
            np.isclose(thresholds["target_simple_coverage"], primary_coverage),
            "threshold_tie_fraction",
        ].iloc[0]
    )
    # With a strict "probability < threshold" rule, all observations tied at
    # the empirical quantile stay on the LightGBM side.  The attainable static
    # coverage can therefore differ by the mass of that tie (plus one order-
    # statistic rounding unit), without indicating a controller error.
    primary_quantile_tolerance = primary_threshold_tie_fraction + 1.0 / len(
        calibration
    )
    finite_tuning_columns = [
        "achieved_simple_coverage",
        "absolute_coverage_violation",
        "equal_series_mean_scaled_loss",
        "pooled_mean_scaled_loss",
        rolling_key,
        "final_bias",
    ]
    checks_items: list[tuple[str, bool, str]] = [
        ("frozen_sample_id_is_consistent", sample_ids == {SAMPLE_ID}, SAMPLE_ID),
        ("upstream_local_weighting_checks_passed", bool(passed_column(local_checks["passed"]).all()), f"checks={len(local_checks)}"),
        ("input_contains_only_calibration", observed_splits == {"calibration"}, str(observed_splits)),
        ("model_was_not_loaded_or_refitted", True, "script reads saved scores only"),
        ("test_not_accessed", "test" not in observed_splits, str(observed_splits)),
        ("expected_calibration_rows", len(calibration) == EXPECTED_ROWS, f"rows={len(calibration)}"),
        ("expected_series_count", calibration["series_id"].nunique() == EXPECTED_SERIES, f"series={calibration['series_id'].nunique()}"),
        ("variable_series_lengths_match_frozen_manifest", variable_lengths_match_manifest, f"series={len(observed_lengths)}"),
        ("per_series_time_indices_are_contiguous", per_series_contiguous, str(per_series_contiguous)),
        ("expected_relative_time_points", calibration["relative_calibration_step"].nunique() == EXPECTED_RELATIVE_STEPS, f"steps={calibration['relative_calibration_step'].nunique()}"),
        ("expected_per_series_length_range", int(per_series_lengths.min()) == EXPECTED_MINIMUM_SERIES_LENGTH and int(per_series_lengths.max()) == EXPECTED_MAXIMUM_SERIES_LENGTH, f"minimum={per_series_lengths.min()}; maximum={per_series_lengths.max()}"),
        ("relative_step_active_counts_match_manifest", active_counts_match_manifest, f"minimum={observed_active_counts.min()}; maximum={observed_active_counts.max()}"),
        ("expected_active_series_range", int(selected_trace["batch_rows"].min()) == EXPECTED_MINIMUM_BATCH_ROWS and int(selected_trace["batch_rows"].max()) == EXPECTED_MAXIMUM_BATCH_ROWS, f"minimum={selected_trace['batch_rows'].min()}; maximum={selected_trace['batch_rows'].max()}"),
        ("routing_probabilities_are_valid", probability_valid, f"minimum={probability.min():.6f}; maximum={probability.max():.6f}"),
        ("loss_scales_are_positive", scales_positive, f"minimum={scales.min():.6g}"),
        ("three_coverage_thresholds_saved", len(thresholds) == 3, f"rows={len(thresholds)}"),
        ("primary_static_quantile_matches_target_with_ties", abs(expected_primary_static_coverage - primary_coverage) <= primary_quantile_tolerance, f"coverage={expected_primary_static_coverage:.9f}; tie_fraction={primary_threshold_tie_fraction:.9g}; tolerance={primary_quantile_tolerance:.9g}"),
        ("four_eta_candidates_evaluated", len(tuning) == len(eta_grid), f"rows={len(tuning)}"),
        ("all_tuning_metrics_are_finite", bool(np.isfinite(tuning[finite_tuning_columns].to_numpy(float)).all()), f"candidates={len(tuning)}"),
        ("selected_eta_is_preregistered", selected_eta in eta_grid, f"eta={selected_eta}"),
        ("selected_controller_obeys_constraint_when_feasible", bool(feasible.empty or float(selected_summary["absolute_coverage_violation"]) <= allowed_violation), f"violation={float(selected_summary['absolute_coverage_violation']):.6f}"),
        ("trace_has_one_row_per_relative_day", len(selected_trace) == EXPECTED_RELATIVE_STEPS, f"rows={len(selected_trace)}"),
        ("decision_rows_match_calibration", len(selected_decisions) == len(calibration) and not selected_decisions.duplicated(["series_id", "time_index"]).any(), f"rows={len(selected_decisions)}"),
        ("decision_and_trace_coverage_agree", abs(decision_coverage - trace_coverage) <= 1e-15, f"decision={decision_coverage:.12f}; trace={trace_coverage:.12f}"),
        ("controller_update_uses_decisions_only", True, "bias_next=clip(bias+eta*(target-batch_coverage))"),
    ]
    checks = pd.DataFrame(checks_items, columns=["check", "passed", "detail"])
    failed = checks.loc[~checks["passed"]]
    if not failed.empty:
        message = "; ".join(
            f"{row.check}: {row.detail}" for row in failed.itertuples(index=False)
        )
        raise AssertionError(f"Weather Daily coverage calibration failed: {message}")

    tuning.to_csv(TUNING_PATH, index=False)
    selected_trace.to_csv(TRACE_PATH, index=False)
    thresholds.to_csv(THRESHOLD_PATH, index=False)
    checks.to_csv(CHECKS_PATH, index=False)
    for column in ("dataset_id", "sample_id", "series_id", "series_type"):
        selected_decisions[column] = selected_decisions[column].astype("category")
    selected_decisions.to_parquet(DECISION_PATH, index=False, compression="snappy")

    selected_parameters = {
        "dataset_id": DATASET_ID,
        "sample_id": SAMPLE_ID,
        "fit_scope": "calibration_only",
        "model_refitted": False,
        "test_accessed": False,
        "probability_semantics": "probability_that_lightgbm_is_preferred",
        "decision_rule": "use_ridge_if_probability_is_below_effective_threshold",
        "threshold_source": "calibration_quantile",
        "time_alignment": {
            "rule": ALIGNMENT_RULE,
            "relative_step_starts_at": 0,
            "controller_state": "one global bias shared by active series",
            "variable_batch_size": True,
            "minimum_batch_rows": int(selected_summary["minimum_batch_rows"]),
            "maximum_batch_rows": int(selected_summary["maximum_batch_rows"]),
        },
        "new_segment_initialization": {
            "bias": 0.0,
            "calibration_final_bias_is_not_carried_to_test": True,
        },
        "primary_target_simple_coverage": primary_coverage,
        "primary_base_threshold": primary_threshold,
        "selected_eta": selected_eta,
        "bias_limit": bias_limit,
        "allowed_absolute_violation": allowed_violation,
        "rolling_diagnostic_window_days": ROLLING_WINDOW_DAYS,
        "rolling_coverage_weighting": "sum simple decisions divided by sum active rows",
        "update_rule": "bias_next=clip(bias+eta*(target-batch_coverage),-bias_limit,bias_limit)",
        "update_information": "routing_decisions_only_no_true_target_needed",
        "selection_status": selection_status,
        "selection_metric": "minimum equal-series scaled loss among pooled-coverage-feasible candidates",
        "candidate_eta_values": eta_grid,
        "calibration_rows": len(calibration),
        "calibration_series": int(calibration["series_id"].nunique()),
        "calibration_relative_steps": int(calibration["relative_calibration_step"].nunique()),
        "achieved_calibration_simple_coverage": float(selected_summary["achieved_simple_coverage"]),
        "calibration_absolute_violation": float(selected_summary["absolute_coverage_violation"]),
        "calibration_equal_series_mean_scaled_loss": float(selected_summary["equal_series_mean_scaled_loss"]),
        "calibration_pooled_mean_scaled_loss": float(selected_summary["pooled_mean_scaled_loss"]),
        f"calibration_worst_{ROLLING_WINDOW_DAYS}d_weighted_coverage_violation": float(selected_summary[rolling_key]),
        "calibration_final_bias": float(selected_summary["final_bias"]),
        "static_calibration_simple_coverage": float(static_summary["achieved_simple_coverage"]),
        "static_calibration_equal_series_mean_scaled_loss": float(static_summary["equal_series_mean_scaled_loss"]),
        "static_calibration_pooled_mean_scaled_loss": float(static_summary["pooled_mean_scaled_loss"]),
        "thresholds": [
            {
                "target_simple_coverage": float(row.target_simple_coverage),
                "base_threshold": float(row.base_threshold),
                "achieved_static_calibration_coverage": float(row.achieved_static_calibration_coverage),
            }
            for row in thresholds.itertuples(index=False)
        ],
        "seed": seed,
    }
    SELECTED_PATH.write_text(
        yaml.safe_dump(selected_parameters, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    rolling_name = f"rolling_{ROLLING_WINDOW_DAYS}d_simple_coverage"
    figure, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    axes[0].hist(calibration["black_box_probability"], bins=45, color="#4C78A8", alpha=0.82)
    threshold_colors = ["#59A14F", "#E15759", "#F28E2B"]
    for row, color in zip(thresholds.itertuples(index=False), threshold_colors):
        axes[0].axvline(
            row.base_threshold,
            color=color,
            linestyle="--",
            linewidth=1.8,
            label=f"Ridge coverage={row.target_simple_coverage:.1f}",
        )
    axes[0].set_title("Calibration probability thresholds")
    axes[0].set_xlabel("Predicted probability that LightGBM wins")
    axes[0].set_ylabel("Count")
    axes[0].legend(frameon=False)

    axes[1].plot(
        selected_trace["relative_calibration_step"],
        selected_trace["simple_coverage"],
        color="#BAB0AC",
        alpha=0.24,
        linewidth=0.7,
        label="Relative-day batch coverage",
    )
    axes[1].plot(
        selected_trace["relative_calibration_step"],
        selected_trace[rolling_name],
        color="#E15759",
        linewidth=1.5,
        label=f"Weighted {ROLLING_WINDOW_DAYS}-day coverage",
    )
    axes[1].axhline(primary_coverage, color="black", linestyle="--", linewidth=1.3, label="Target coverage")
    axes[1].set_title("Adaptive Ridge coverage")
    axes[1].set_xlabel("Days since entering calibration")
    axes[1].set_ylabel("Ridge coverage")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(frameon=False)

    axes[2].plot(
        selected_trace["relative_calibration_step"],
        selected_trace["effective_threshold"],
        color="#59A14F",
        linewidth=1.5,
        label="Effective threshold",
    )
    axes[2].axhline(primary_threshold, color="black", linestyle="--", linewidth=1.3, label="Base threshold")
    axes[2].set_title("Online threshold adjustment")
    axes[2].set_xlabel("Days since entering calibration")
    axes[2].set_ylabel("Probability threshold")
    axes[2].legend(frameon=False)

    figure.suptitle("Weather Daily calibration-only coverage-controller selection", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)

    elapsed = perf_counter() - start
    report = {
        "status": "passed",
        "dataset_id": DATASET_ID,
        "sample_id": SAMPLE_ID,
        "check_count": len(checks),
        "failed_check_count": 0,
        "input_scope": "calibration_only",
        "model_refitted": False,
        "test_accessed": False,
        "calibration_rows": len(calibration),
        "series_count": int(calibration["series_id"].nunique()),
        "relative_time_points": int(calibration["relative_calibration_step"].nunique()),
        "primary_target_simple_coverage": primary_coverage,
        "primary_base_threshold": primary_threshold,
        "eta_candidates": len(eta_grid),
        "selected_eta": selected_eta,
        "selected_coverage": float(selected_summary["achieved_simple_coverage"]),
        "selected_absolute_violation": float(selected_summary["absolute_coverage_violation"]),
        "selected_equal_series_scaled_loss": float(selected_summary["equal_series_mean_scaled_loss"]),
        "static_equal_series_scaled_loss": float(static_summary["equal_series_mean_scaled_loss"]),
        f"selected_worst_{ROLLING_WINDOW_DAYS}d_violation": float(selected_summary[rolling_key]),
        "coverage_constraint_passed": bool(float(selected_summary["absolute_coverage_violation"]) <= allowed_violation),
        "runtime_seconds": float(elapsed),
        "outputs": {
            "tuning": str(TUNING_PATH),
            "trace": str(TRACE_PATH),
            "decisions": str(DECISION_PATH),
            "thresholds": str(THRESHOLD_PATH),
            "selected": str(SELECTED_PATH),
            "checks": str(CHECKS_PATH),
            "figure": str(FIGURE_PATH),
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print()
    print("Weather Daily 覆盖率控制器校准全部通过")
    print("固定样本编号：", SAMPLE_ID)
    print("参数选择数据范围：仅 calibration")
    print("模型是否重新训练：否")
    print("test 是否访问：否")
    print("时间对齐：各序列进入 calibration 后的相对天数")
    print("calibration 样本数量：", len(calibration))
    print("序列数量：", calibration["series_id"].nunique())
    print("每条序列 calibration 长度范围：", f"{per_series_lengths.min()} 至 {per_series_lengths.max()} 天")
    print("相对时间点数量：", calibration["relative_calibration_step"].nunique())
    print("每个相对时点活动序列数量范围：", f"{selected_summary['minimum_batch_rows']} 至 {selected_summary['maximum_batch_rows']}")
    print("目标 Ridge 覆盖率：", primary_coverage)
    print("基础概率阈值：", f"{primary_threshold:.9f}")
    print("候选 eta 数量：", len(eta_grid))
    print("选定 eta：", selected_eta)
    print("自适应实际覆盖率：", f"{float(selected_summary['achieved_simple_coverage']):.6f}")
    print("绝对覆盖率偏差：", f"{float(selected_summary['absolute_coverage_violation']):.6f}")
    print("自适应等权逐序列缩放损失：", f"{float(selected_summary['equal_series_mean_scaled_loss']):.6f}")
    print("静态阈值等权逐序列缩放损失：", f"{float(static_summary['equal_series_mean_scaled_loss']):.6f}")
    print("自适应汇总缩放损失：", f"{float(selected_summary['pooled_mean_scaled_loss']):.6f}")
    print(f"最差{ROLLING_WINDOW_DAYS}天加权覆盖率偏差：", f"{float(selected_summary[rolling_key]):.6f}")
    print("最终 bias：", f"{float(selected_summary['final_bias']):.9f}")
    print("覆盖率约束：", "通过" if float(selected_summary["absolute_coverage_violation"]) <= allowed_violation else "未通过")
    print("运行秒数：", f"{elapsed:.2f}")
    print("候选参数结果：", TUNING_PATH)
    print("逐相对时点控制记录：", TRACE_PATH)
    print("逐样本校准决策：", DECISION_PATH)
    print("选定控制器参数：", SELECTED_PATH)
    print("校准诊断图片：", FIGURE_PATH)


if __name__ == "__main__":
    main()
