#!/usr/bin/env python3
"""Causal context and past-residual features for Pedestrian routing."""

from __future__ import annotations

import json
import os
from pathlib import Path

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

DATA_PATH = PROJECT_ROOT / "data/processed/pedestrian_hourly_long.parquet"
PREDICTION_PATH = PROJECT_ROOT / "results/pedestrian_pretest_predictions.parquet"
SCALER_PATH = PROJECT_ROOT / "results/pedestrian_scaler_parameters.csv"
CONFIG_PATH = PROJECT_ROOT / "experiment_config.yaml"
BASE_CHECKS_PATH = PROJECT_ROOT / "results/pedestrian_base_model_fit_checks.csv"
BASE_METADATA_PATH = PROJECT_ROOT / "results/pedestrian_base_model_fit_metadata.yaml"
SPLIT_SUMMARY_PATH = PROJECT_ROOT / "results/pedestrian_split_summary.yaml"

FEATURE_PATH = OUTPUT_ROOT / "results/pedestrian_router_features.parquet"
MANIFEST_PATH = OUTPUT_ROOT / "results/pedestrian_router_feature_manifest.csv"
COUNT_PATH = OUTPUT_ROOT / "results/pedestrian_router_feature_counts.csv"
CHECKS_PATH = OUTPUT_ROOT / "results/pedestrian_router_feature_checks.csv"
SUMMARY_PATH = OUTPUT_ROOT / "results/pedestrian_router_feature_summary.yaml"
FIGURE_PATH = OUTPUT_ROOT / "figures/pedestrian_router_target_distribution.png"
REPORT_PATH = OUTPUT_ROOT / "logs/pedestrian_router_feature_report.json"

DATASET_ID = "pedestrian_hourly"
ALLOWED_RAW_SPLITS = ["base_train", "router_train", "calibration"]
ROUTING_SPLITS = ["router_train", "calibration"]
SEASONAL_PERIOD = 24
LONG_CONTEXT = 168


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def passed_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.lower().eq("true")


def rolling_past_mean_std(
    values: np.ndarray, targets: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    prefix = np.empty(len(values) + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(values, out=prefix[1:])
    squared_prefix = np.empty(len(values) + 1, dtype=np.float64)
    squared_prefix[0] = 0.0
    np.cumsum(values * values, out=squared_prefix[1:])
    sums = prefix[targets] - prefix[targets - window]
    squared_sums = (
        squared_prefix[targets] - squared_prefix[targets - window]
    )
    means = sums / window
    variances = np.maximum(squared_sums / window - means * means, 0.0)
    return means, np.sqrt(variances)


def main() -> None:
    for path in (
        DATA_PATH,
        PREDICTION_PATH,
        SCALER_PATH,
        CONFIG_PATH,
        BASE_CHECKS_PATH,
        BASE_METADATA_PATH,
        SPLIT_SUMMARY_PATH,
    ):
        require_file(path)
    for path in (
        FEATURE_PATH,
        MANIFEST_PATH,
        COUNT_PATH,
        CHECKS_PATH,
        SUMMARY_PATH,
        FIGURE_PATH,
        REPORT_PATH,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    base_checks = pd.read_csv(BASE_CHECKS_PATH)
    base_metadata = yaml.safe_load(BASE_METADATA_PATH.read_text(encoding="utf-8"))
    split_summary = yaml.safe_load(SPLIT_SUMMARY_PATH.read_text(encoding="utf-8"))
    scalers = pd.read_csv(SCALER_PATH)

    residual_lag_candidates = [
        int(value) for value in config["soft_router"]["residual_lag_grid"]
    ]
    if residual_lag_candidates != [1, 4, 8, 16]:
        raise AssertionError(
            f"Unexpected residual lag grid: {residual_lag_candidates}"
        )
    max_residual_lag = max(residual_lag_candidates)
    if not passed_column(base_checks["passed"]).all():
        raise AssertionError("Base-model fit checks did not all pass")
    if bool(base_metadata.get("test_accessed", True)):
        raise AssertionError("Base-model metadata reports test access")
    if scalers["source_split"].ne("base_train").any():
        raise AssertionError("Scaler source is not base_train only")

    raw = pd.read_parquet(
        DATA_PATH,
        columns=["series_id", "time_index", "value", "split"],
        filters=[[("split", "==", name)] for name in ALLOWED_RAW_SPLITS],
    )
    observed_raw_splits = set(raw["split"].astype(str).unique().tolist())
    if observed_raw_splits != set(ALLOWED_RAW_SPLITS):
        raise AssertionError(f"Unexpected raw-data splits: {observed_raw_splits}")
    expected_raw_rows = int(
        sum(split_summary["aggregate_counts"][name] for name in ALLOWED_RAW_SPLITS)
    )
    if len(raw) != expected_raw_rows:
        raise AssertionError(f"Raw row count mismatch: {len(raw)} vs {expected_raw_rows}")

    prediction = (
        pd.read_parquet(PREDICTION_PATH)
        .sort_values(["series_id", "time_index"], kind="stable")
        .reset_index(drop=True)
    )
    observed_prediction_splits = set(
        prediction["split"].astype(str).unique().tolist()
    )
    if observed_prediction_splits != set(ROUTING_SPLITS):
        raise AssertionError(
            f"Unexpected prediction splits: {observed_prediction_splits}"
        )
    expected_prediction_rows = int(
        sum(split_summary["aggregate_counts"][name] for name in ROUTING_SPLITS)
    )
    if len(prediction) != expected_prediction_rows:
        raise AssertionError(
            f"Prediction row mismatch: {len(prediction)} vs {expected_prediction_rows}"
        )

    median_lookup = scalers.set_index("series_id")["median"].to_dict()
    scale_lookup = scalers.set_index("series_id")["scale_used"].to_dict()
    raw_series: dict[str, np.ndarray] = {}
    seasonal_mae_scale: dict[str, float] = {}
    raw_indices_contiguous = True
    for series_id_raw, group in raw.groupby("series_id", sort=False, observed=True):
        series_id = str(series_id_raw)
        group = group.sort_values("time_index", kind="stable")
        indices = group["time_index"].to_numpy(dtype=np.int64)
        raw_indices_contiguous = raw_indices_contiguous and bool(
            np.array_equal(indices, np.arange(len(group), dtype=np.int64))
        )
        values = group["value"].to_numpy(dtype=np.float64)
        raw_series[series_id] = values
        base_values = group.loc[group["split"] == "base_train", "value"].to_numpy(
            dtype=np.float64
        )
        seasonal_difference = (
            base_values[SEASONAL_PERIOD:] - base_values[:-SEASONAL_PERIOD]
        )
        mae_scale = float(np.mean(np.abs(seasonal_difference)))
        if not np.isfinite(mae_scale) or mae_scale <= 1e-12:
            raise AssertionError(f"{series_id} has a zero seasonal MAE scale")
        seasonal_mae_scale[series_id] = mae_scale

    prediction["median_base"] = prediction["series_id"].astype(str).map(median_lookup)
    prediction["iqr_base"] = prediction["series_id"].astype(str).map(scale_lookup)
    prediction["seasonal_naive_mae_scale"] = (
        prediction["series_id"].astype(str).map(seasonal_mae_scale)
    )

    context_names = [
        "last_value_scaled",
        "mean_24_scaled",
        "trend_24_scaled",
        "volatility_24_scaled",
        "trend_168_scaled",
        "volatility_168_scaled",
    ]
    context_arrays = {
        name: np.empty(len(prediction), dtype=np.float32) for name in context_names
    }
    prediction_targets_contiguous = True
    context_uses_only_past = True

    for series_id_raw, group in prediction.groupby(
        "series_id", sort=False, observed=True
    ):
        series_id = str(series_id_raw)
        row_indices = group.index.to_numpy(dtype=np.int64)
        targets = group["time_index"].to_numpy(dtype=np.int64)
        prediction_targets_contiguous = prediction_targets_contiguous and bool(
            np.all(np.diff(targets) == 1)
        )
        context_uses_only_past = context_uses_only_past and bool(
            targets.min() >= LONG_CONTEXT + 1
        )
        values = raw_series[series_id]
        median = float(median_lookup[series_id])
        scale = float(scale_lookup[series_id])
        mean_24, std_24 = rolling_past_mean_std(values, targets, SEASONAL_PERIOD)
        _, std_168 = rolling_past_mean_std(values, targets, LONG_CONTEXT)

        context_arrays["last_value_scaled"][row_indices] = (
            (values[targets - 1] - median) / scale
        ).astype(np.float32)
        context_arrays["mean_24_scaled"][row_indices] = (
            (mean_24 - median) / scale
        ).astype(np.float32)
        context_arrays["trend_24_scaled"][row_indices] = (
            (values[targets - 1] - values[targets - 1 - SEASONAL_PERIOD]) / scale
        ).astype(np.float32)
        context_arrays["volatility_24_scaled"][row_indices] = (
            std_24 / scale
        ).astype(np.float32)
        context_arrays["trend_168_scaled"][row_indices] = (
            (values[targets - 1] - values[targets - 1 - LONG_CONTEXT]) / scale
        ).astype(np.float32)
        context_arrays["volatility_168_scaled"][row_indices] = (
            std_168 / scale
        ).astype(np.float32)

    prediction = pd.concat(
        [prediction, pd.DataFrame(context_arrays, index=prediction.index)], axis=1
    )
    prediction["ridge_prediction_scaled"] = (
        (prediction["ridge_prediction"] - prediction["median_base"])
        / prediction["iqr_base"]
    ).astype(np.float32)
    prediction["lightgbm_prediction_scaled"] = (
        (prediction["lightgbm_prediction"] - prediction["median_base"])
        / prediction["iqr_base"]
    ).astype(np.float32)
    prediction["prediction_difference_scaled"] = (
        (prediction["lightgbm_prediction"] - prediction["ridge_prediction"])
        / prediction["iqr_base"]
    ).astype(np.float32)
    prediction["absolute_prediction_difference_scaled"] = prediction[
        "prediction_difference_scaled"
    ].abs().astype(np.float32)

    timestamp = pd.to_datetime(prediction["timestamp"])
    hour_of_day = timestamp.dt.hour.to_numpy(dtype=np.float64)
    day_of_week = timestamp.dt.dayofweek.to_numpy(dtype=np.float64)
    prediction["hour_of_day_sin"] = np.sin(
        2 * np.pi * hour_of_day / 24.0
    ).astype(np.float32)
    prediction["hour_of_day_cos"] = np.cos(
        2 * np.pi * hour_of_day / 24.0
    ).astype(np.float32)
    prediction["day_of_week_sin"] = np.sin(
        2 * np.pi * day_of_week / 7.0
    ).astype(np.float32)
    prediction["day_of_week_cos"] = np.cos(
        2 * np.pi * day_of_week / 7.0
    ).astype(np.float32)

    context_feature_names = [
        *context_names,
        "ridge_prediction_scaled",
        "lightgbm_prediction_scaled",
        "prediction_difference_scaled",
        "absolute_prediction_difference_scaled",
        "hour_of_day_sin",
        "hour_of_day_cos",
        "day_of_week_sin",
        "day_of_week_cos",
    ]

    grouped = prediction.groupby("series_id", sort=False, observed=True)
    residual_feature_data: dict[str, pd.Series] = {}
    residual_feature_names: list[str] = []
    for lag in range(1, max_residual_lag + 1):
        ridge_name = f"ridge_residual_lag_{lag}"
        lightgbm_name = f"lightgbm_residual_lag_{lag}"
        residual_feature_data[ridge_name] = (
            grouped["ridge_residual"].shift(lag) / prediction["iqr_base"]
        ).astype(np.float32)
        residual_feature_data[lightgbm_name] = (
            grouped["lightgbm_residual"].shift(lag) / prediction["iqr_base"]
        ).astype(np.float32)
        residual_feature_names.extend([ridge_name, lightgbm_name])
    prediction = pd.concat(
        [prediction, pd.DataFrame(residual_feature_data, index=prediction.index)],
        axis=1,
    )

    prediction["ridge_scaled_loss"] = (
        prediction["ridge_residual"] / prediction["seasonal_naive_mae_scale"]
    ) ** 2
    prediction["lightgbm_scaled_loss"] = (
        prediction["lightgbm_residual"]
        / prediction["seasonal_naive_mae_scale"]
    ) ** 2
    prediction["loss_advantage_black_box"] = (
        prediction["ridge_scaled_loss"] - prediction["lightgbm_scaled_loss"]
    )
    prediction["hard_black_box_target"] = (
        prediction["loss_advantage_black_box"] > 0
    ).astype(np.int8)

    feature_names = context_feature_names + residual_feature_names
    missing_before_warmup = prediction[residual_feature_names].isna().any(axis=1)
    retained_mask = ~missing_before_warmup
    features = prediction.loc[retained_mask].copy().reset_index(drop=True)

    all_features_finite = True
    for name in feature_names:
        values = features[name].to_numpy(dtype=np.float64)
        all_features_finite = all_features_finite and bool(np.isfinite(values).all())
    if not all_features_finite:
        raise AssertionError("Retained router features contain missing/nonfinite values")

    removed = prediction.loc[missing_before_warmup]
    removed_counts = removed.groupby("series_id", observed=True).size()
    warmup_exact = bool(
        len(removed_counts) == len(raw_series)
        and (removed_counts == max_residual_lag).all()
        and set(removed["split"].astype(str).unique().tolist()) == {"router_train"}
    )
    counts = (
        features.groupby("split", as_index=False, observed=True)
        .agg(rows=("series_id", "size"), series_count=("series_id", "nunique"))
        .sort_values("split", kind="stable")
        .reset_index(drop=True)
    )
    expected_counts = {
        "router_train": int(
            split_summary["aggregate_counts"]["router_train"]
            - len(raw_series) * max_residual_lag
        ),
        "calibration": int(split_summary["aggregate_counts"]["calibration"]),
    }
    observed_counts = {
        str(row.split): int(row.rows) for row in counts.itertuples(index=False)
    }
    if observed_counts != expected_counts:
        raise AssertionError(
            f"Router feature count mismatch: {observed_counts} vs {expected_counts}"
        )

    output_columns = [
        "dataset_id",
        "series_id",
        "time_index",
        "timestamp",
        "split",
        "y_true",
        "ridge_prediction",
        "lightgbm_prediction",
        "ridge_residual",
        "lightgbm_residual",
        "ridge_scaled_loss",
        "lightgbm_scaled_loss",
        "loss_advantage_black_box",
        "hard_black_box_target",
        "seasonal_naive_mae_scale",
        *feature_names,
    ]
    features[output_columns].to_parquet(
        FEATURE_PATH, index=False, compression="snappy"
    )
    counts.to_csv(COUNT_PATH, index=False)

    descriptions = {
        "last_value_scaled": "Most recent observed count, base-train median/IQR scaled",
        "mean_24_scaled": "Mean of the 24 observations immediately before the target",
        "trend_24_scaled": "Change from target-25 to target-1 (24-hour difference)",
        "volatility_24_scaled": "Population standard deviation over the previous 24 hours",
        "trend_168_scaled": "Change from target-169 to target-1 (168-hour difference)",
        "volatility_168_scaled": "Population standard deviation over the previous 168 hours",
        "ridge_prediction_scaled": "Current causal Ridge prediction, base-train scaled",
        "lightgbm_prediction_scaled": "Current causal LightGBM prediction, base-train scaled",
        "prediction_difference_scaled": "LightGBM minus Ridge prediction, base-train scaled",
        "absolute_prediction_difference_scaled": "Absolute current model disagreement",
        "hour_of_day_sin": "Sine encoding of hour of day",
        "hour_of_day_cos": "Cosine encoding of hour of day",
        "day_of_week_sin": "Sine encoding of day of week",
        "day_of_week_cos": "Cosine encoding of day of week",
    }
    manifest_records: list[dict[str, object]] = []
    for name in context_feature_names:
        manifest_records.append(
            {
                "feature": name,
                "group": "context",
                "available_before_target": True,
                "maximum_source_time_relative_to_target": -1,
                "description": descriptions[name],
            }
        )
    for name in residual_feature_names:
        lag = int(name.rsplit("_", 1)[1])
        model_name = "Ridge" if name.startswith("ridge") else "LightGBM"
        manifest_records.append(
            {
                "feature": name,
                "group": "past_residual",
                "available_before_target": True,
                "maximum_source_time_relative_to_target": -lag,
                "description": f"{model_name} residual from {lag} hour(s) before target",
            }
        )
    manifest = pd.DataFrame(manifest_records)
    manifest.to_csv(MANIFEST_PATH, index=False)

    win_rate = (
        features.groupby("split", observed=True)["hard_black_box_target"]
        .mean()
        .reindex(ROUTING_SPLITS)
    )
    check_items: list[tuple[str, bool, str]] = [
        ("base_model_fit_audit_passed", bool(passed_column(base_checks["passed"]).all()), f"checks={len(base_checks)}"),
        ("raw_input_excludes_test", "test" not in observed_raw_splits, str(observed_raw_splits)),
        ("prediction_input_excludes_test", "test" not in observed_prediction_splits, str(observed_prediction_splits)),
        ("raw_row_count_matches_manifest", len(raw) == expected_raw_rows, f"actual={len(raw)}; expected={expected_raw_rows}"),
        ("prediction_row_count_matches_manifest", len(prediction) == expected_prediction_rows, f"actual={len(prediction)}; expected={expected_prediction_rows}"),
        ("raw_time_indices_are_contiguous", raw_indices_contiguous, str(raw_indices_contiguous)),
        ("prediction_targets_are_contiguous_per_series", prediction_targets_contiguous, str(prediction_targets_contiguous)),
        ("context_features_use_only_past_observations", context_uses_only_past, str(context_uses_only_past)),
        ("residual_shift_warmup_is_exact", warmup_exact, f"removed={len(removed)}"),
        ("retained_feature_counts_match_expected", observed_counts == expected_counts, f"observed={observed_counts}; expected={expected_counts}"),
        ("feature_manifest_has_46_features", len(manifest) == 46 and len(feature_names) == 46, f"manifest={len(manifest)}; features={len(feature_names)}"),
        ("all_manifest_features_are_pre_target", bool(manifest["available_before_target"].all() and (manifest["maximum_source_time_relative_to_target"] < 0).all()), f"features={len(manifest)}"),
        ("all_retained_features_are_finite", all_features_finite, str(all_features_finite)),
    ]
    checks = pd.DataFrame(check_items, columns=["check", "passed", "detail"])
    failed = checks.loc[~checks["passed"]]
    if not failed.empty:
        message = "; ".join(
            f"{row.check}: {row.detail}" for row in failed.itertuples(index=False)
        )
        raise AssertionError(f"Pedestrian router-feature audit failed: {message}")
    checks.to_csv(CHECKS_PATH, index=False)

    summary = {
        "dataset_id": DATASET_ID,
        "router_feature_generation_passed": True,
        "context_feature_count": len(context_feature_names),
        "residual_feature_count": len(residual_feature_names),
        "total_feature_count": len(feature_names),
        "residual_lag_candidates_hours": residual_lag_candidates,
        "maximum_residual_lag_hours": max_residual_lag,
        "warmup_rows_removed_per_series": max_residual_lag,
        "retained_counts": observed_counts,
        "black_box_point_win_rate": {
            split_name: float(win_rate.loc[split_name]) for split_name in ROUTING_SPLITS
        },
        "seasonal_loss_scale": "base_train mean absolute 24-hour difference",
        "future_information_check_passed": True,
        "test_accessed": False,
    }
    SUMMARY_PATH.write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes[0, 0].bar(
        win_rate.index,
        win_rate.values,
        color=["#4C78A8", "#F28E2B"],
    )
    axes[0, 0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel("LightGBM pointwise win rate")
    axes[0, 0].set_title("Routing target balance")

    router = features.loc[features["split"] == "router_train"]
    lower, upper = router["loss_advantage_black_box"].quantile([0.01, 0.99])
    axes[0, 1].hist(
        router["loss_advantage_black_box"].clip(lower, upper),
        bins=70,
        color="#76B7B2",
        edgecolor="none",
    )
    axes[0, 1].axvline(0, color="#E15759", linestyle="--", linewidth=1)
    axes[0, 1].set_xlabel("Scaled loss advantage of LightGBM")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Router-train advantage (1st–99th percentile clipped)")

    sample_size = min(6000, len(router))
    sample_indices = np.floor(
        (np.arange(sample_size, dtype=float) + 0.5) * len(router) / sample_size
    ).astype(int)
    sample = router.iloc[sample_indices]
    axes[1, 0].scatter(
        sample["absolute_prediction_difference_scaled"],
        sample["loss_advantage_black_box"].clip(lower, upper),
        c=sample["hard_black_box_target"],
        cmap="coolwarm",
        s=8,
        alpha=0.35,
    )
    axes[1, 0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_xlabel("Absolute scaled model disagreement")
    axes[1, 0].set_ylabel("Scaled LightGBM loss advantage (clipped)")
    axes[1, 0].set_title("Model disagreement and routing advantage")

    hour_win_rate = (
        router.assign(hour=pd.to_datetime(router["timestamp"]).dt.hour)
        .groupby("hour", observed=True)["hard_black_box_target"]
        .mean()
        .reindex(range(24))
    )
    axes[1, 1].plot(
        hour_win_rate.index,
        hour_win_rate.values,
        color="#59A14F",
        marker="o",
        linewidth=1.5,
    )
    axes[1, 1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_xticks(range(0, 24, 3))
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xlabel("Hour of day")
    axes[1, 1].set_ylabel("LightGBM pointwise win rate")
    axes[1, 1].set_title("Router-train target balance by hour")
    axes[1, 1].grid(alpha=0.2)

    fig.suptitle("Pedestrian causal routing features and targets", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = {
        "status": "passed",
        "dataset_id": DATASET_ID,
        "check_count": int(len(checks)),
        "failed_check_count": 0,
        "context_features": len(context_feature_names),
        "residual_features": len(residual_feature_names),
        "total_features": len(feature_names),
        "retained_counts": observed_counts,
        "black_box_win_rate": {
            split_name: float(win_rate.loc[split_name]) for split_name in ROUTING_SPLITS
        },
        "future_information_check": "passed",
        "test_accessed": False,
        "outputs": {
            "features": str(FEATURE_PATH),
            "manifest": str(MANIFEST_PATH),
            "counts": str(COUNT_PATH),
            "checks": str(CHECKS_PATH),
            "summary": str(SUMMARY_PATH),
            "figure": str(FIGURE_PATH),
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("Pedestrian 路由特征生成全部通过")
    print("最大残差滞后：", max_residual_lag, "小时")
    print("上下文特征数量：", len(context_feature_names))
    print("残差特征数量：", len(residual_feature_names))
    print("路由特征总数：", len(feature_names))
    print("每条序列删除的残差预热行数：", max_residual_lag)
    print("未来信息检查：通过")
    print("各数据段样本数量：")
    print(counts.to_string(index=False))
    print("各数据段 LightGBM 单点胜率：")
    print(win_rate.to_string())
    print("路由特征：", FEATURE_PATH)
    print("特征清单：", MANIFEST_PATH)
    print("样本计数：", COUNT_PATH)
    print("目标分布图：", FIGURE_PATH)


if __name__ == "__main__":
    main()
