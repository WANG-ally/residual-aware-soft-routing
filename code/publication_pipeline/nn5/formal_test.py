"""冻结并执行一次NN5正式测试。"""

import argparse
import os
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT
EVALUATOR_PATH = Path(__file__).resolve()

FINAL_LOCK_PATH = PROJECT_ROOT / "results" / "nn5_final_test_lock_manifest.json"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "nn5_daily_long.parquet"
SCALER_PATH = PROJECT_ROOT / "results" / "nn5_scaler_parameters.csv"
PRETEST_PATH = PROJECT_ROOT / "results" / "nn5_pretest_predictions.parquet"
ROUTER_FEATURE_PATH = PROJECT_ROOT / "results" / "nn5_router_features.parquet"
CALIBRATION_ROUTER_PATH = PROJECT_ROOT / "results" / "nn5_calibration_router_scores.parquet"
BASELINE_CALIBRATION_PATH = PROJECT_ROOT / "results" / "nn5_baseline_calibration_scores.parquet"
RIDGE_PARAM_PATH = PROJECT_ROOT / "results" / "nn5_selected_ridge_params.yaml"
LGBM_PARAM_PATH = PROJECT_ROOT / "results" / "nn5_selected_lightgbm_params.yaml"
CONTROLLER_PATH = PROJECT_ROOT / "results" / "nn5_selected_coverage_controller.yaml"
BASELINE_PARAM_PATH = PROJECT_ROOT / "results" / "nn5_selected_baseline_params.yaml"
RIDGE_MODEL_PATH = PROJECT_ROOT / "models" / "nn5_ridge.joblib"
LGBM_MODEL_PATH = PROJECT_ROOT / "models" / "nn5_lightgbm.joblib"
FULL_ROUTER_MODEL_PATH = PROJECT_ROOT / "models" / "nn5_soft_router.joblib"

AUTHORIZATION_PATH = OUTPUT_ROOT / "logs" / "nn5_evaluator_authorization.json"
RECEIPT_PATH = OUTPUT_ROOT / "logs" / "nn5_formal_test_access_receipt.json"
PREDICTION_PATH = OUTPUT_ROOT / "results" / "nn5_test_predictions.parquet"
PER_SERIES_PATH = OUTPUT_ROOT / "results" / "nn5_test_per_series_metrics.csv"
AGGREGATE_PATH = OUTPUT_ROOT / "results" / "nn5_test_aggregate_metrics.csv"
DAILY_PATH = OUTPUT_ROOT / "results" / "nn5_test_daily_coverage.csv"
FIGURE_PATH = OUTPUT_ROOT / "figures" / "nn5_test_method_comparison.png"

for path in [AUTHORIZATION_PATH, RECEIPT_PATH, PREDICTION_PATH,
             PER_SERIES_PATH, AGGREGATE_PATH, DAILY_PATH, FIGURE_PATH]:
    path.parent.mkdir(parents=True, exist_ok=True)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json_atomic(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def verify_final_lock():
    with FINAL_LOCK_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    changed = []
    current_records = []
    for item in manifest["files"]:
        path = PROJECT_ROOT / item["path"]
        if not path.is_file():
            changed.append(item["path"])
            continue
        observed = sha256_file(path)
        if observed != item["sha256"] or int(path.stat().st_size) != int(item["size_bytes"]):
            changed.append(item["path"])
        current_records.append({"path": item["path"], "size_bytes": int(path.stat().st_size),
                                "sha256": observed})
    if changed:
        raise ValueError(f"最终冻结文件发生变化：{changed}")
    canonical = json.dumps(current_records, ensure_ascii=False, sort_keys=True).encode("utf-8")
    observed_id = hashlib.sha256(canonical).hexdigest()
    if observed_id != manifest["final_freeze_id"]:
        raise ValueError("最终冻结编号无法复现")
    if manifest["formal_test_runs_completed"] != 0:
        raise ValueError("冻结清单中的正式测试次数不是0")
    return manifest


def load_artifacts():
    ridge_params = load_yaml(RIDGE_PARAM_PATH)
    lgbm_params = load_yaml(LGBM_PARAM_PATH)
    controller = load_yaml(CONTROLLER_PATH)
    baseline_params = load_yaml(BASELINE_PARAM_PATH)
    full_router = joblib.load(FULL_ROUTER_MODEL_PATH)
    baseline_bundles = {}
    for method, metadata in baseline_params["methods"].items():
        if method == "hard_aalf_like_router":
            continue
        baseline_bundles[method] = joblib.load(PROJECT_ROOT / metadata["model_file"])
    artifacts = {
        "ridge_params": ridge_params,
        "lgbm_params": lgbm_params,
        "controller": controller,
        "baseline_params": baseline_params,
        "ridge_model": joblib.load(RIDGE_MODEL_PATH),
        "lgbm_model": joblib.load(LGBM_MODEL_PATH),
        "full_router": full_router,
        "baseline_bundles": baseline_bundles,
        "scalers": pd.read_csv(SCALER_PATH),
    }
    if controller["test_accessed"] or baseline_params["test_accessed"]:
        raise ValueError("冻结参数显示测试集曾被访问")
    return artifacts


def read_raw_splits(split_names):
    raw = pd.read_parquet(
        DATA_PATH,
        columns=["dataset_id", "series_id", "time_index", "timestamp", "value", "split"],
        filters=[[('split', '==', name)] for name in split_names],
    )
    if set(raw["split"]) != set(split_names):
        raise ValueError(f"读取的数据段异常：{sorted(set(raw['split']))}")
    return raw


def prepare_series(raw, required_end):
    frames = {}
    for series_id, group in raw.groupby("series_id", sort=False):
        group = group.sort_values("time_index").reset_index(drop=True)
        indices = group["time_index"].to_numpy(dtype=int)
        if not np.array_equal(indices, np.arange(required_end + 1)):
            raise ValueError(f"{series_id} 在0至{required_end}的索引不连续")
        frames[series_id] = group
    if len(frames) != 111:
        raise ValueError("序列数量不是111")
    return frames


def build_segment(raw, start_index, end_index, artifacts):
    frames = prepare_series(raw, end_index)
    scalers = artifacts["scalers"]
    if scalers["source_split"].ne("base_train").any():
        raise ValueError("缩放器来源不是base_train")
    scaler_lookup = {
        row.series_id: (float(row.median), float(row.scale_used))
        for row in scalers.itertuples()
    }
    series_ids = sorted(frames)
    ridge_window = int(artifacts["ridge_params"]["selected_window"])
    lgbm_window = int(artifacts["lgbm_params"]["selected_window"])
    ridge_model = artifacts["ridge_model"]
    lgbm_model = artifacts["lgbm_model"]

    pretest = pd.read_parquet(PRETEST_PATH)
    pretest = pretest[pretest["time_index"] < start_index]
    ridge_history = {series_id: np.full(end_index + 1, np.nan) for series_id in series_ids}
    lgbm_history = {series_id: np.full(end_index + 1, np.nan) for series_id in series_ids}
    for row in pretest.itertuples():
        ridge_history[row.series_id][int(row.time_index)] = float(row.ridge_residual)
        lgbm_history[row.series_id][int(row.time_index)] = float(row.lightgbm_residual)

    denominators = {}
    for series_id in series_ids:
        base_values = frames[series_id].loc[
            frames[series_id]["split"] == "base_train", "value"
        ].to_numpy(dtype=float)
        differences = base_values[7:] - base_values[:-7]
        denominators[series_id] = {
            "rmsse": float(np.mean(differences ** 2)),
            "mase": float(np.mean(np.abs(differences))),
        }
        if denominators[series_id]["rmsse"] <= 1e-12 or denominators[series_id]["mase"] <= 1e-12:
            raise ValueError(f"{series_id} 的季节尺度为零")

    records = []
    for time_index in range(start_index, end_index + 1):
        ridge_x, lgbm_x, medians, scales = [], [], [], []
        for series_id in series_ids:
            values = frames[series_id]["value"].to_numpy(dtype=float)
            median, scale = scaler_lookup[series_id]
            scaled = (values - median) / scale
            if time_index - ridge_window < 0 or time_index - lgbm_window < 0:
                raise ValueError("基础模型窗口不足")
            ridge_x.append(scaled[time_index - ridge_window:time_index][::-1])
            lgbm_x.append(scaled[time_index - lgbm_window:time_index][::-1])
            medians.append(median)
            scales.append(scale)
        ridge_x = np.asarray(ridge_x, dtype=float)
        lgbm_x = np.asarray(lgbm_x, dtype=float)
        medians = np.asarray(medians, dtype=float)
        scales = np.asarray(scales, dtype=float)
        ridge_prediction = ridge_model.predict(ridge_x) * scales + medians
        lgbm_prediction = lgbm_model.booster_.predict(lgbm_x) * scales + medians

        batch_records = []
        for position, series_id in enumerate(series_ids):
            frame = frames[series_id]
            values = frame["value"].to_numpy(dtype=float)
            timestamp = frame.loc[time_index, "timestamp"]
            median, scale = scaler_lookup[series_id]
            past_7 = values[time_index - 7:time_index]
            past_28 = values[time_index - 28:time_index]
            if len(past_7) != 7 or len(past_28) != 28:
                raise ValueError("上下文窗口长度异常")
            row = {
                "dataset_id": "nn5_daily", "series_id": series_id,
                "time_index": time_index, "timestamp": timestamp,
                "split": str(frame.loc[time_index, "split"]),
                "y_true": float(values[time_index]),
                "ridge_prediction": float(ridge_prediction[position]),
                "lightgbm_prediction": float(lgbm_prediction[position]),
                "seasonal_naive_prediction": float(values[time_index - 7]),
                "seasonal_naive_mae_scale": denominators[series_id]["mase"],
                "rmsse_denominator": denominators[series_id]["rmsse"],
                "mase_denominator": denominators[series_id]["mase"],
                "last_value_scaled": (values[time_index - 1] - median) / scale,
                "mean_7_scaled": (float(np.mean(past_7)) - median) / scale,
                "trend_7_scaled": (values[time_index - 1] - values[time_index - 8]) / scale,
                "volatility_7_scaled": float(np.std(past_7, ddof=0)) / scale,
                "trend_28_scaled": (values[time_index - 1] - values[time_index - 29]) / scale,
                "volatility_28_scaled": float(np.std(past_28, ddof=0)) / scale,
                "ridge_prediction_scaled": (ridge_prediction[position] - median) / scale,
                "lightgbm_prediction_scaled": (lgbm_prediction[position] - median) / scale,
                "prediction_difference_scaled": (lgbm_prediction[position] - ridge_prediction[position]) / scale,
                "absolute_prediction_difference_scaled": abs(lgbm_prediction[position] - ridge_prediction[position]) / scale,
            }
            day = pd.Timestamp(timestamp).dayofweek
            row["day_of_week_sin"] = float(np.sin(2.0 * np.pi * day / 7.0))
            row["day_of_week_cos"] = float(np.cos(2.0 * np.pi * day / 7.0))
            for lag in range(1, 17):
                ridge_residual = ridge_history[series_id][time_index - lag]
                lgbm_residual = lgbm_history[series_id][time_index - lag]
                if not np.isfinite(ridge_residual) or not np.isfinite(lgbm_residual):
                    raise ValueError(f"{series_id} 在{time_index}缺少残差滞后{lag}")
                row[f"ridge_residual_lag_{lag}"] = float(ridge_residual / scale)
                row[f"lightgbm_residual_lag_{lag}"] = float(lgbm_residual / scale)
            batch_records.append(row)

        # Reveal current targets and update residuals only after predicting the full batch.
        for row in batch_records:
            series_id = row["series_id"]
            ridge_history[series_id][time_index] = row["y_true"] - row["ridge_prediction"]
            lgbm_history[series_id][time_index] = row["y_true"] - row["lightgbm_prediction"]
        records.extend(batch_records)

    segment = pd.DataFrame(records).sort_values(["time_index", "series_id"]).reset_index(drop=True)
    expected_rows = (end_index - start_index + 1) * 111
    if len(segment) != expected_rows:
        raise ValueError(f"在线预测行数异常：{len(segment)} != {expected_rows}")
    return segment


def apply_methods(frame, artifacts):
    frame = frame.copy()
    controller = artifacts["controller"]
    target_coverage = float(controller["primary_target_simple_coverage"])
    base_threshold = float(controller["primary_base_threshold"])
    eta = float(controller["selected_eta"])
    bias_limit = float(controller["bias_limit"])

    full_bundle = artifacts["full_router"]
    full_x = full_bundle["scaler"].transform(
        frame[full_bundle["feature_names"]].to_numpy(dtype=float)
    )
    frame["full_router_probability"] = np.clip(
        full_bundle["model"].predict_proba(full_x)[:, 1], 1e-8, 1.0 - 1e-8
    )

    for method, bundle in artifacts["baseline_bundles"].items():
        x_value = bundle["scaler"].transform(frame[bundle["feature_names"]].to_numpy(dtype=float))
        frame[f"{method}_probability"] = np.clip(
            bundle["model"].predict_proba(x_value)[:, 1], 1e-8, 1.0 - 1e-8
        )

    aalf = artifacts["baseline_params"]["methods"]["hard_aalf_like_router"]
    aalf_lag = int(aalf["residual_lag"])
    ridge_columns = [f"ridge_residual_lag_{i}" for i in range(1, aalf_lag + 1)]
    lgbm_columns = [f"lightgbm_residual_lag_{i}" for i in range(1, aalf_lag + 1)]
    frame["hard_aalf_like_router_score"] = (
        np.mean(frame[ridge_columns].to_numpy(dtype=float) ** 2, axis=1)
        - np.mean(frame[lgbm_columns].to_numpy(dtype=float) ** 2, axis=1)
    )

    routing_methods = list(artifacts["baseline_bundles"])
    for method in routing_methods:
        threshold = float(artifacts["baseline_bundles"][method]["thresholds"][str(target_coverage)])
        frame[f"use_ridge_{method}"] = frame[f"{method}_probability"] < threshold
        frame[f"prediction_{method}"] = np.where(
            frame[f"use_ridge_{method}"], frame["ridge_prediction"], frame["lightgbm_prediction"]
        )

    aalf_threshold = float(aalf["thresholds"][str(target_coverage)])
    frame["use_ridge_hard_aalf_like_router"] = frame["hard_aalf_like_router_score"] < aalf_threshold
    frame["prediction_hard_aalf_like_router"] = np.where(
        frame["use_ridge_hard_aalf_like_router"], frame["ridge_prediction"], frame["lightgbm_prediction"]
    )

    frame["use_ridge_static_full_router"] = frame["full_router_probability"] < base_threshold
    frame["prediction_static_full_router"] = np.where(
        frame["use_ridge_static_full_router"], frame["ridge_prediction"], frame["lightgbm_prediction"]
    )

    frame["use_ridge_adaptive_full_router"] = False
    frame["adaptive_effective_threshold"] = np.nan
    frame["adaptive_bias_before"] = np.nan
    frame["adaptive_bias_after"] = np.nan
    bias = 0.0
    for _, indices in frame.groupby("time_index", sort=True).groups.items():
        index_array = np.asarray(list(indices), dtype=int)
        effective_threshold = float(np.clip(base_threshold + bias, 0.0, 1.0))
        use_simple = frame.loc[index_array, "full_router_probability"].to_numpy(dtype=float) < effective_threshold
        frame.loc[index_array, "use_ridge_adaptive_full_router"] = use_simple
        frame.loc[index_array, "adaptive_effective_threshold"] = effective_threshold
        frame.loc[index_array, "adaptive_bias_before"] = bias
        coverage = float(np.mean(use_simple))
        next_bias = float(np.clip(bias + eta * (target_coverage - coverage), -bias_limit, bias_limit))
        frame.loc[index_array, "adaptive_bias_after"] = next_bias
        bias = next_bias
    frame["prediction_adaptive_full_router"] = np.where(
        frame["use_ridge_adaptive_full_router"], frame["ridge_prediction"], frame["lightgbm_prediction"]
    )

    frame["prediction_ridge_only"] = frame["ridge_prediction"]
    frame["prediction_lightgbm_only"] = frame["lightgbm_prediction"]
    frame["prediction_seasonal_naive"] = frame["seasonal_naive_prediction"]
    frame["prediction_equal_weight_average"] = 0.5 * (
        frame["ridge_prediction"] + frame["lightgbm_prediction"]
    )
    frame["use_ridge_ridge_only"] = True
    frame["use_ridge_lightgbm_only"] = False

    ridge_se = (frame["y_true"] - frame["ridge_prediction"]) ** 2
    lgbm_se = (frame["y_true"] - frame["lightgbm_prediction"]) ** 2
    frame["hard_black_box_target"] = (lgbm_se < ridge_se).astype(np.int8)
    frame["use_ridge_unconstrained_oracle"] = ridge_se <= lgbm_se
    frame["prediction_unconstrained_oracle"] = np.where(
        frame["use_ridge_unconstrained_oracle"], frame["ridge_prediction"], frame["lightgbm_prediction"]
    )

    advantage = (
        ((frame["y_true"] - frame["ridge_prediction"]) / frame["seasonal_naive_mae_scale"]) ** 2
        - ((frame["y_true"] - frame["lightgbm_prediction"]) / frame["seasonal_naive_mae_scale"]) ** 2
    ).to_numpy(dtype=float)
    simple_count = int(round(target_coverage * len(frame)))
    black_box_count = len(frame) - simple_count
    order = np.argsort(-advantage, kind="mergesort")
    use_black_box = np.zeros(len(frame), dtype=bool)
    use_black_box[order[:black_box_count]] = True
    frame["use_ridge_coverage_constrained_oracle"] = ~use_black_box
    frame["prediction_coverage_constrained_oracle"] = np.where(
        frame["use_ridge_coverage_constrained_oracle"],
        frame["ridge_prediction"], frame["lightgbm_prediction"]
    )
    return frame


METHODS = [
    "seasonal_naive", "ridge_only", "lightgbm_only", "equal_weight_average",
    "hard_aalf_like_router", "hard_logistic_same_features",
    "hard_random_forest_same_features", "class_weight_only",
    "soft_targets_only", "residual_features_only", "static_full_router",
    "adaptive_full_router", "unconstrained_oracle", "coverage_constrained_oracle",
]


def expected_calibration_differences(computed):
    expected_prediction = pd.read_parquet(PRETEST_PATH)
    expected_prediction = expected_prediction[expected_prediction["split"] == "calibration"]
    expected_features = pd.read_parquet(ROUTER_FEATURE_PATH)
    expected_features = expected_features[expected_features["split"] == "calibration"]
    expected_router = pd.read_parquet(CALIBRATION_ROUTER_PATH)
    expected_baselines = pd.read_parquet(BASELINE_CALIBRATION_PATH)
    keys = ["series_id", "time_index"]
    calculated = computed.sort_values(keys).reset_index(drop=True)
    expected_prediction = expected_prediction.sort_values(keys).reset_index(drop=True)
    expected_features = expected_features.sort_values(keys).reset_index(drop=True)
    expected_router = expected_router.sort_values(keys).reset_index(drop=True)
    expected_baselines = expected_baselines.sort_values(keys).reset_index(drop=True)
    if not calculated[keys].equals(expected_features[keys]):
        raise ValueError("校准自检键值不一致")
    full_features = [
        "last_value_scaled", "mean_7_scaled", "trend_7_scaled", "volatility_7_scaled",
        "trend_28_scaled", "volatility_28_scaled", "ridge_prediction_scaled",
        "lightgbm_prediction_scaled", "prediction_difference_scaled",
        "absolute_prediction_difference_scaled", "day_of_week_sin", "day_of_week_cos",
    ] + [name for lag in range(1, 17) for name in
         (f"ridge_residual_lag_{lag}", f"lightgbm_residual_lag_{lag}")]
    differences = {
        "ridge_prediction": float(np.max(np.abs(calculated["ridge_prediction"] - expected_prediction["ridge_prediction"]))),
        "lightgbm_prediction": float(np.max(np.abs(calculated["lightgbm_prediction"] - expected_prediction["lightgbm_prediction"]))),
        "router_features": float(np.max(np.abs(
            calculated[full_features].to_numpy(dtype=float)
            - expected_features[full_features].to_numpy(dtype=float)
        ))),
        "full_router_probability": float(np.max(np.abs(
            calculated["full_router_probability"] - expected_router["black_box_probability"]
        ))),
    }
    baseline_columns = [
        "hard_logistic_same_features_probability",
        "hard_random_forest_same_features_probability", "class_weight_only_probability",
        "soft_targets_only_probability", "residual_features_only_probability",
        "hard_aalf_like_router_score",
    ]
    differences["baseline_scores"] = float(np.max(np.abs(
        calculated[baseline_columns].to_numpy(dtype=float)
        - expected_baselines[baseline_columns].to_numpy(dtype=float)
    )))
    return differences


def dry_run_calibration(artifacts):
    raw = read_raw_splits(["base_train", "router_train", "calibration"])
    computed = build_segment(raw, 593, 671, artifacts)
    computed = apply_methods(computed, artifacts)
    differences = expected_calibration_differences(computed)
    tolerance = 1e-7
    if any(value > tolerance for value in differences.values()):
        raise ValueError(f"校准重算与冻结结果不一致：{differences}")
    if len(computed) != 8769 or set(computed["split"]) != {"calibration"}:
        raise ValueError("校准干运行样本范围异常")
    return {"rows": len(computed), "tolerance": tolerance,
            "maximum_differences": differences, "test_accessed": False}


def expected_calibration_error(probability, hard_target, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            mask = (probability >= edges[index]) & (probability < edges[index + 1])
        if np.any(mask):
            result += np.mean(mask) * abs(np.mean(probability[mask]) - np.mean(hard_target[mask]))
    return float(result)


def compute_metrics(frame, target_coverage):
    per_series_records = []
    for series_id, group in frame.groupby("series_id", sort=False):
        y_true = group["y_true"].to_numpy(dtype=float)
        rmsse_denominator = float(group["rmsse_denominator"].iloc[0])
        mase_denominator = float(group["mase_denominator"].iloc[0])
        for method in METHODS:
            prediction = group[f"prediction_{method}"].to_numpy(dtype=float)
            error = y_true - prediction
            smape_denominator = np.abs(y_true) + np.abs(prediction)
            smape_terms = np.divide(
                200.0 * np.abs(error), smape_denominator,
                out=np.zeros_like(error), where=smape_denominator > 1e-12,
            )
            use_column = f"use_ridge_{method}"
            coverage = float(group[use_column].mean()) if use_column in group else np.nan
            per_series_records.append({
                "dataset_id": "nn5_daily", "series_id": series_id, "method": method,
                "RMSSE": float(np.sqrt(np.mean(error ** 2) / rmsse_denominator)),
                "MASE": float(np.mean(np.abs(error)) / mase_denominator),
                "sMAPE": float(np.mean(smape_terms)),
                "RMSE": float(np.sqrt(np.mean(error ** 2))),
                "simple_coverage": coverage,
            })
    per_series = pd.DataFrame(per_series_records)

    daily_records = []
    for method in METHODS:
        use_column = f"use_ridge_{method}"
        if use_column not in frame:
            continue
        for time_index, group in frame.groupby("time_index", sort=True):
            daily_records.append({"dataset_id": "nn5_daily", "method": method,
                                  "time_index": int(time_index),
                                  "simple_coverage": float(group[use_column].mean())})
    daily = pd.DataFrame(daily_records)

    score_map = {
        "hard_aalf_like_router": ("hard_aalf_like_router_score", False),
        "hard_logistic_same_features": ("hard_logistic_same_features_probability", True),
        "hard_random_forest_same_features": ("hard_random_forest_same_features_probability", True),
        "class_weight_only": ("class_weight_only_probability", True),
        "soft_targets_only": ("soft_targets_only_probability", True),
        "residual_features_only": ("residual_features_only_probability", True),
        "static_full_router": ("full_router_probability", True),
        "adaptive_full_router": ("full_router_probability", True),
    }
    hard_target = frame["hard_black_box_target"].to_numpy(dtype=int)
    aggregate_records = []
    for method in METHODS:
        method_metrics = per_series[per_series["method"] == method]
        prediction = frame[f"prediction_{method}"].to_numpy(dtype=float)
        y_true = frame["y_true"].to_numpy(dtype=float)
        use_column = f"use_ridge_{method}"
        coverage = float(frame[use_column].mean()) if use_column in frame else np.nan
        method_daily = daily[daily["method"] == method]
        if method_daily.empty:
            worst_14 = np.nan
        else:
            rolling = method_daily["simple_coverage"].rolling(14, min_periods=14).mean().dropna()
            worst_14 = float(np.max(np.abs(rolling - target_coverage))) if not rolling.empty else np.nan
        auprc = brier = ece = recall = np.nan
        if method in score_map:
            score_column, is_probability = score_map[method]
            score = frame[score_column].to_numpy(dtype=float)
            auprc = float(average_precision_score(hard_target, score))
            if is_probability:
                brier = float(brier_score_loss(hard_target, score))
                ece = expected_calibration_error(score, hard_target)
            use_black_box = ~frame[use_column].to_numpy(dtype=bool)
            recall = float(np.sum(use_black_box & (hard_target == 1)) / np.sum(hard_target == 1))
        aggregate_records.append({
            "dataset_id": "nn5_daily", "method": method,
            "mean_RMSSE": float(method_metrics["RMSSE"].mean()),
            "median_RMSSE": float(method_metrics["RMSSE"].median()),
            "std_RMSSE": float(method_metrics["RMSSE"].std(ddof=1)),
            "mean_MASE": float(method_metrics["MASE"].mean()),
            "mean_sMAPE": float(method_metrics["sMAPE"].mean()),
            "overall_RMSE": float(np.sqrt(np.mean((y_true - prediction) ** 2))),
            "simple_coverage": coverage,
            "absolute_coverage_violation": abs(coverage - target_coverage) if np.isfinite(coverage) else np.nan,
            "worst_14_day_coverage_violation": worst_14,
            "black_box_AUPRC": auprc, "black_box_Brier": brier,
            "black_box_ECE": ece, "black_box_recall": recall,
        })
    aggregate = pd.DataFrame(aggregate_records).sort_values("mean_RMSSE").reset_index(drop=True)
    aggregate["RMSSE_rank"] = np.arange(1, len(aggregate) + 1)
    return per_series, aggregate, daily


def save_figure(aggregate, daily, target_coverage):
    plot_data = aggregate.sort_values("mean_RMSSE", ascending=True)
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].barh(plot_data["method"], plot_data["mean_RMSSE"], color="#4c78a8")
    axes[0].set_xlabel("Mean per-series RMSSE")
    axes[0].set_title("Formal test forecasting accuracy")
    coverage_data = aggregate.dropna(subset=["simple_coverage"])
    axes[1].barh(coverage_data["method"], coverage_data["simple_coverage"], color="#59a14f")
    axes[1].axvline(target_coverage, color="black", linestyle="--")
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Simple-model coverage")
    axes[1].set_title("Formal test coverage")
    for method, color in [("static_full_router", "#f28e2b"),
                          ("adaptive_full_router", "#e15759")]:
        part = daily[daily["method"] == method]
        rolling = part["simple_coverage"].rolling(14, min_periods=1).mean()
        axes[2].plot(part["time_index"], rolling, label=method, color=color)
    axes[2].axhline(target_coverage, color="black", linestyle="--")
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Time index")
    axes[2].set_ylabel("14-day rolling coverage")
    axes[2].set_title("Coverage through time")
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)


def freeze_evaluator():
    if RECEIPT_PATH.exists():
        raise FileExistsError("正式测试访问回执已存在，不能重新授权")
    manifest = verify_final_lock()
    artifacts = load_artifacts()
    dry_run = dry_run_calibration(artifacts)
    evaluator_hash = sha256_file(EVALUATOR_PATH)
    authorization_material = f"{manifest['final_freeze_id']}:{evaluator_hash}".encode("utf-8")
    authorization_id = hashlib.sha256(authorization_material).hexdigest()
    try:
        evaluator_display_path = str(EVALUATOR_PATH.relative_to(PROJECT_ROOT))
    except ValueError:
        evaluator_display_path = str(EVALUATOR_PATH)
    payload = {
        "dataset_id": "nn5_daily", "status": "AUTHORIZED",
        "authorized_at_utc": utc_now(), "authorization_id": authorization_id,
        "final_freeze_id": manifest["final_freeze_id"],
        "evaluator_path": evaluator_display_path,
        "evaluator_sha256": evaluator_hash,
        "formal_test_runs_allowed": 1, "formal_test_runs_completed": 0,
        "dry_run": dry_run,
    }
    if AUTHORIZATION_PATH.exists():
        with AUTHORIZATION_PATH.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("authorization_id") != authorization_id:
            raise ValueError("已有授权与当前执行器哈希不一致")
    else:
        write_json_atomic(AUTHORIZATION_PATH, payload)
    print()
    print("NN5 正式测试执行器冻结与校准干运行全部通过")
    print("最终预测试冻结编号：", manifest["final_freeze_id"])
    print("执行器SHA-256：", evaluator_hash)
    print("授权编号：", authorization_id)
    print("校准干运行样本：", dry_run["rows"])
    print("校准重算最大差异：", max(dry_run["maximum_differences"].values()))
    print("测试集是否访问：否")
    print("正式测试运行次数：0")
    print("状态：已授权，但尚未执行正式测试")


def execute_formal_test():
    if not AUTHORIZATION_PATH.is_file():
        raise FileNotFoundError("缺少执行器授权文件，请先运行--freeze-evaluator")
    if RECEIPT_PATH.exists():
        raise FileExistsError("正式测试访问回执已存在，禁止重复运行")
    forbidden_existing = [path for path in [PREDICTION_PATH, PER_SERIES_PATH,
                          AGGREGATE_PATH, DAILY_PATH] if path.exists()]
    if forbidden_existing:
        raise FileExistsError(f"正式测试输出已存在：{forbidden_existing}")
    manifest = verify_final_lock()
    with AUTHORIZATION_PATH.open("r", encoding="utf-8") as handle:
        authorization = json.load(handle)
    evaluator_hash = sha256_file(EVALUATOR_PATH)
    if authorization["status"] != "AUTHORIZED":
        raise ValueError("执行器未获授权")
    if authorization["evaluator_sha256"] != evaluator_hash:
        raise ValueError("执行器在授权后发生变化")
    if authorization["final_freeze_id"] != manifest["final_freeze_id"]:
        raise ValueError("授权文件与最终冻结编号不一致")

    started = perf_counter()
    receipt = {
        "dataset_id": "nn5_daily", "status": "STARTED",
        "formal_test_run_number": 1, "started_at_utc": utc_now(),
        "authorization_id": authorization["authorization_id"],
        "final_freeze_id": manifest["final_freeze_id"],
        "evaluator_sha256": evaluator_hash,
    }
    # Write the one-time access receipt before reading test for the first time.
    write_json_atomic(RECEIPT_PATH, receipt)
    try:
        artifacts = load_artifacts()
        raw = read_raw_splits(["base_train", "router_train", "calibration", "test"])
        test_frame = build_segment(raw, 672, 790, artifacts)
        if len(test_frame) != 13209 or set(test_frame["split"]) != {"test"}:
            raise ValueError("正式测试样本范围异常")
        test_frame = apply_methods(test_frame, artifacts)
        target_coverage = float(artifacts["controller"]["primary_target_simple_coverage"])
        per_series, aggregate, daily = compute_metrics(test_frame, target_coverage)
        test_frame.to_parquet(PREDICTION_PATH, index=False)
        per_series.to_csv(PER_SERIES_PATH, index=False)
        aggregate.to_csv(AGGREGATE_PATH, index=False)
        daily.to_csv(DAILY_PATH, index=False)
        save_figure(aggregate, daily, target_coverage)
        result_hashes = {
            str(path.relative_to(OUTPUT_ROOT)): sha256_file(path)
            for path in [PREDICTION_PATH, PER_SERIES_PATH, AGGREGATE_PATH,
                         DAILY_PATH, FIGURE_PATH]
        }
        receipt.update({
            "status": "COMPLETED", "completed_at_utc": utc_now(),
            "formal_test_rows": len(test_frame), "series_count": 111,
            "time_points": 119, "elapsed_seconds": perf_counter() - started,
            "result_sha256": result_hashes,
        })
        write_json_atomic(RECEIPT_PATH, receipt)
    except Exception as error:
        receipt.update({"status": "FAILED_LOCKED", "failed_at_utc": utc_now(),
                        "error_type": type(error).__name__, "error_message": str(error)})
        write_json_atomic(RECEIPT_PATH, receipt)
        raise

    primary = aggregate[aggregate["method"] == "adaptive_full_router"].iloc[0]
    print()
    print("NN5 唯一一次正式测试全部完成")
    print("正式测试行数：", len(test_frame))
    print("序列数量：111")
    print("时间点数量：119")
    print("自适应完整方法 mean RMSSE：", f"{primary['mean_RMSSE']:.6f}")
    print("自适应完整方法 Ridge 覆盖率：", f"{primary['simple_coverage']:.6f}")
    print("自适应完整方法覆盖偏差：", f"{primary['absolute_coverage_violation']:.6f}")
    print("方法排名表：")
    print(aggregate[["RMSSE_rank", "method", "mean_RMSSE", "simple_coverage"]].to_string(index=False))
    print("正式测试运行次数：1（已锁定，禁止再次运行）")
    print("逐时点预测：", PREDICTION_PATH)
    print("逐序列指标：", PER_SERIES_PATH)
    print("汇总指标：", AGGREGATE_PATH)
    print("覆盖率轨迹：", DAILY_PATH)
    print("结果图片：", FIGURE_PATH)
    print("访问回执：", RECEIPT_PATH)


def main():
    parser = argparse.ArgumentParser(description="Freeze or execute the one-time NN5 formal test")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-evaluator", action="store_true",
                       help="verify all frozen artifacts and authorize this exact evaluator without test access")
    group.add_argument("--execute-final-test", action="store_true",
                       help="perform the single authorized formal test run")
    arguments = parser.parse_args()
    if arguments.freeze_evaluator:
        freeze_evaluator()
    else:
        execute_formal_test()


if __name__ == "__main__":
    main()
