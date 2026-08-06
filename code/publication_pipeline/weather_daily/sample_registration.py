#!/usr/bin/env python3
"""Preregister and freeze the Weather Daily 500-series sample.

Selection is proportional by weather-variable type and determined only by the
archived series ID, type, and the registered seed. Raw values, time splits,
models, forecast errors, calibration data, and test data are never read here."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sci-routing")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(
    os.environ.get("SCI_ROUTING_ROOT", Path(__file__).resolve().parents[3])
).resolve()
OUTPUT_ROOT = Path(os.environ.get("SCI_ROUTING_OUTPUT_ROOT", PROJECT_ROOT)).resolve()

DATASET_ID = "weather_daily"
REGISTERED_SEED = 20260714
REGISTERED_SAMPLE_SIZE = 500
MAXIMUM_CANDIDATE_WINDOW = 56
EXPECTED_ARCHIVE_SERIES = 3_010
EXPECTED_TYPE_COUNTS = {
    "maxtemp": 746,
    "mintemp": 748,
    "rain": 729,
    "solar": 787,
}
EXPECTED_SAMPLE_ALLOCATION = {
    "maxtemp": 124,
    "mintemp": 124,
    "rain": 121,
    "solar": 131,
}
SELECTION_ALGORITHM = "sha256_rank_within_series_type_v1"

CONFIG_PATH = PROJECT_ROOT / "experiment_config.yaml"
AUDIT_PATH = PROJECT_ROOT / "results/weather_daily_series_audit.csv"
AUDIT_CHECKS_PATH = PROJECT_ROOT / "results/weather_daily_audit_checks.csv"
AUDIT_SUMMARY_PATH = PROJECT_ROOT / "results/weather_daily_data_quality_summary.yaml"

SAMPLE_MANIFEST_PATH = OUTPUT_ROOT / "results/weather_daily_sample_manifest.csv"
SAMPLE_REGISTRATION_PATH = (
    OUTPUT_ROOT / "results/weather_daily_sample_registration.yaml"
)
SAMPLE_CHECKS_PATH = OUTPUT_ROOT / "results/weather_daily_sample_registration_checks.csv"
SAMPLE_FIGURE_PATH = OUTPUT_ROOT / "figures/weather_daily_registered_sample.png"
SAMPLE_REPORT_PATH = OUTPUT_ROOT / "logs/weather_daily_sample_registration_report.json"

for path in (
    SAMPLE_MANIFEST_PATH,
    SAMPLE_REGISTRATION_PATH,
    SAMPLE_CHECKS_PATH,
    SAMPLE_FIGURE_PATH,
    SAMPLE_REPORT_PATH,
):
    path.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().eq("true")


def proportional_allocation(counts: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder proportional allocation with lexical tie breaking."""
    population = sum(counts.values())
    exact = {key: counts[key] * total / population for key in sorted(counts)}
    allocation = {key: math.floor(exact[key]) for key in sorted(counts)}
    remaining = total - sum(allocation.values())
    priority = sorted(
        counts,
        key=lambda key: (-(exact[key] - allocation[key]), key),
    )
    for key in priority[:remaining]:
        allocation[key] += 1
    return allocation


def selection_digest(series_id: str, series_type: str) -> str:
    message = (
        f"{REGISTERED_SEED}|{DATASET_ID}|{series_type}|{series_id}"
    ).encode("utf-8")
    return hashlib.sha256(message).hexdigest()


def build_sample(audit: pd.DataFrame) -> pd.DataFrame:
    candidates = audit.copy()
    candidates["selection_digest"] = [
        selection_digest(str(series_id), str(series_type))
        for series_id, series_type in candidates[
            ["series_id", "series_type"]
        ].itertuples(index=False, name=None)
    ]
    allocation = proportional_allocation(EXPECTED_TYPE_COUNTS, REGISTERED_SAMPLE_SIZE)
    selected_frames = []
    for series_type in sorted(allocation):
        group = candidates.loc[candidates["series_type"] == series_type].copy()
        group = group.sort_values(
            ["selection_digest", "series_id"], kind="mergesort"
        ).reset_index(drop=True)
        group["selection_rank_within_type"] = np.arange(1, len(group) + 1)
        selected_frames.append(group.iloc[: allocation[series_type]].copy())
    selected = pd.concat(selected_frames, ignore_index=True)
    selected = selected.sort_values("source_order", kind="mergesort").reset_index(
        drop=True
    )
    selected.insert(0, "sample_order", np.arange(1, len(selected) + 1))
    return selected


def sample_id_for(selected: pd.DataFrame, allocation: dict[str, int]) -> str:
    records = [
        {
            "series_id": str(row.series_id),
            "series_type": str(row.series_type),
            "source_order": int(row.source_order),
            "selection_rank_within_type": int(row.selection_rank_within_type),
            "selection_digest": str(row.selection_digest),
        }
        for row in selected.itertuples(index=False)
    ]
    payload = {
        "dataset_id": DATASET_ID,
        "registered_seed": REGISTERED_SEED,
        "registered_sample_size": REGISTERED_SAMPLE_SIZE,
        "selection_algorithm": SELECTION_ALGORITHM,
        "sample_allocation": allocation,
        "selected_series": records,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def main() -> None:
    required = [CONFIG_PATH, AUDIT_PATH, AUDIT_CHECKS_PATH, AUDIT_SUMMARY_PATH]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required Weather Daily audit files are missing: {missing}")
    if SAMPLE_REGISTRATION_PATH.is_file():
        raise FileExistsError(
            "Weather Daily sample is already registered; do not redraw it: "
            f"{SAMPLE_REGISTRATION_PATH}"
        )

    checks: list[dict[str, object]] = []

    def record_check(name: str, passed: bool, detail: object) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config_valid = bool(
        int(config["study"]["seed"]) == REGISTERED_SEED
        and int(config["data"]["sample_seed"]) == REGISTERED_SEED
        and int(config["data"]["limited_compute_samples"][DATASET_ID])
        == REGISTERED_SAMPLE_SIZE
        and list(config["preprocessing"]["window_by_frequency"]["daily"])
        == [7, 14, 28, 56]
    )
    record_check(
        "registered_config_seed_sample_size_and_daily_windows_valid",
        config_valid,
        (
            f"seed={config['data']['sample_seed']}; sample="
            f"{config['data']['limited_compute_samples'][DATASET_ID]}; "
            f"windows={config['preprocessing']['window_by_frequency']['daily']}"
        ),
    )

    audit_checks = pd.read_csv(AUDIT_CHECKS_PATH)
    with AUDIT_SUMMARY_PATH.open("r", encoding="utf-8") as handle:
        audit_summary = yaml.safe_load(handle)
    audit_provenance_valid = bool(
        len(audit_checks) == 15
        and bool_series(audit_checks["passed"]).all()
        and audit_summary.get("dataset_id") == DATASET_ID
        and audit_summary.get("data_quality_audit_passed") is True
        and audit_summary.get("full_archive_audited") is True
        and audit_summary.get("registered_500_series_sample_created_now") is False
        and audit_summary.get("formal_test_performance_calculated") is False
    )
    record_check(
        "full_archive_audit_passed_before_sampling",
        audit_provenance_valid,
        f"audit_checks={int(bool_series(audit_checks['passed']).sum())}/15",
    )

    audit = pd.read_csv(AUDIT_PATH)
    observed_counts = {
        str(key): int(value)
        for key, value in audit["series_type"].value_counts().sort_index().items()
    }
    source_population_valid = bool(
        len(audit) == EXPECTED_ARCHIVE_SERIES
        and audit["series_id"].nunique() == EXPECTED_ARCHIVE_SERIES
        and audit["source_order"].tolist()
        == list(range(1, EXPECTED_ARCHIVE_SERIES + 1))
        and observed_counts == EXPECTED_TYPE_COUNTS
        and int(audit["missing_count"].sum()) == 0
        and int(audit["nonfinite_count"].sum()) == 0
    )
    record_check(
        "audited_sampling_frame_complete_unique_and_ordered",
        source_population_valid,
        f"rows={len(audit)}; types={observed_counts}",
    )

    allocation = proportional_allocation(EXPECTED_TYPE_COUNTS, REGISTERED_SAMPLE_SIZE)
    allocation_valid = bool(
        allocation == EXPECTED_SAMPLE_ALLOCATION
        and sum(allocation.values()) == REGISTERED_SAMPLE_SIZE
    )
    record_check(
        "largest_remainder_proportional_allocation_valid",
        allocation_valid,
        allocation,
    )

    selected = build_sample(audit)
    selected_counts = {
        str(key): int(value)
        for key, value in selected["series_type"].value_counts().sort_index().items()
    }
    sample_size_and_strata_valid = bool(
        len(selected) == REGISTERED_SAMPLE_SIZE
        and selected_counts == EXPECTED_SAMPLE_ALLOCATION
    )
    record_check(
        "registered_sample_size_and_type_allocation_valid",
        sample_size_and_strata_valid,
        f"rows={len(selected)}; allocation={selected_counts}",
    )

    identity_valid = bool(
        selected["series_id"].nunique() == REGISTERED_SAMPLE_SIZE
        and selected["selection_digest"].nunique() == REGISTERED_SAMPLE_SIZE
        and selected["sample_order"].tolist()
        == list(range(1, REGISTERED_SAMPLE_SIZE + 1))
        and selected["source_order"].is_monotonic_increasing
    )
    record_check(
        "selected_ids_are_unique_and_manifest_order_is_canonical",
        identity_valid,
        (
            f"unique_ids={selected['series_id'].nunique()}; "
            f"unique_digests={selected['selection_digest'].nunique()}"
        ),
    )

    regenerated = build_sample(audit)
    deterministic_valid = bool(
        selected[
            [
                "series_id",
                "series_type",
                "source_order",
                "selection_rank_within_type",
                "selection_digest",
            ]
        ].equals(
            regenerated[
                [
                    "series_id",
                    "series_type",
                    "source_order",
                    "selection_rank_within_type",
                    "selection_digest",
                ]
            ]
        )
    )
    record_check(
        "sha256_selection_regenerates_identical_sample",
        deterministic_valid,
        f"algorithm={SELECTION_ALGORITHM}; seed={REGISTERED_SEED}",
    )

    digest_recalculation_valid = bool(
        selected.apply(
            lambda row: row["selection_digest"]
            == selection_digest(str(row["series_id"]), str(row["series_type"])),
            axis=1,
        ).all()
    )
    record_check(
        "all_selection_digests_recalculate_from_id_type_and_seed",
        digest_recalculation_valid,
        "digest inputs are seed, dataset ID, series type, and series ID only",
    )

    history_capacity_valid = bool(
        selected["length"].ge(4 * MAXIMUM_CANDIDATE_WINDOW).all()
        and selected["mean_squared_first_difference"].gt(0.0).all()
    )
    record_check(
        "all_selected_histories_support_registered_window_and_rmsse_scale",
        history_capacity_valid,
        (
            f"minimum_length={int(selected['length'].min())}; "
            f"maximum_window={MAXIMUM_CANDIDATE_WINDOW}; "
            f"zero_difference_scale="
            f"{int(selected['mean_squared_first_difference'].le(0.0).sum())}"
        ),
    )

    representation_valid = bool(
        set(selected_counts) == set(EXPECTED_TYPE_COUNTS)
        and all(value > 0 for value in selected_counts.values())
    )
    record_check(
        "all_four_weather_variables_are_represented",
        representation_valid,
        selected_counts,
    )

    record_check(
        "selection_is_independent_of_length_values_and_forecast_performance",
        True,
        (
            "ranking uses only seed, dataset ID, series type, and series ID; "
            "length and audit statistics are documentation only"
        ),
    )
    record_check(
        "raw_values_time_splits_models_and_test_were_not_accessed",
        True,
        "input is saved audit metadata only; no TSF, processed data, model, or metric read",
    )

    failed = [item for item in checks if not item["passed"]]
    if failed:
        raise AssertionError(f"Weather Daily sample registration failed: {failed}")

    manifest_columns = [
        "sample_order",
        "dataset_id",
        "series_id",
        "series_type",
        "source_order",
        "selection_rank_within_type",
        "selection_digest",
        "length",
        "missing_count",
        "nonfinite_count",
        "zero_count",
        "negative_count",
        "iqr",
        "standard_deviation",
        "mean_squared_first_difference",
    ]
    selected[manifest_columns].to_csv(SAMPLE_MANIFEST_PATH, index=False)
    pd.DataFrame(checks).to_csv(SAMPLE_CHECKS_PATH, index=False)

    sample_id = sample_id_for(selected, allocation)
    registration = {
        "dataset_id": DATASET_ID,
        "status": "REGISTERED_AND_FROZEN_BEFORE_TIME_SPLIT_OR_MODELING",
        "sample_id": sample_id,
        "registered_seed": REGISTERED_SEED,
        "registered_sample_size": REGISTERED_SAMPLE_SIZE,
        "archive_series_count": EXPECTED_ARCHIVE_SERIES,
        "selection_algorithm": SELECTION_ALGORITHM,
        "selection_basis": [
            "registered_seed",
            "dataset_id",
            "series_type",
            "series_id",
        ],
        "allocation_rule": (
            "proportional largest-remainder allocation by series_type; "
            "lexical tie breaking"
        ),
        "archive_type_counts": EXPECTED_TYPE_COUNTS,
        "registered_sample_type_counts": allocation,
        "sample_length_minimum": int(selected["length"].min()),
        "sample_length_median": float(selected["length"].median()),
        "sample_length_maximum": int(selected["length"].max()),
        "sample_total_observations": int(selected["length"].sum()),
        "selected_zero_iqr_series_count": int(selected["iqr"].le(0.0).sum()),
        "sample_manifest": str(SAMPLE_MANIFEST_PATH.relative_to(OUTPUT_ROOT)),
        "sample_manifest_sha256": sha256_file(SAMPLE_MANIFEST_PATH),
        "source_audit": str(AUDIT_PATH.relative_to(PROJECT_ROOT)),
        "source_audit_sha256": sha256_file(AUDIT_PATH),
        "performance_values_used_for_selection": False,
        "raw_series_values_read_for_selection": False,
        "time_split_created": False,
        "models_fit_or_tuned": False,
        "formal_test_accessed": False,
        "redraw_or_replacement_after_registration_allowed": False,
    }
    SAMPLE_REGISTRATION_PATH.write_text(
        yaml.safe_dump(registration, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    type_order = ["rain", "mintemp", "maxtemp", "solar"]
    archive_proportions = np.array(
        [EXPECTED_TYPE_COUNTS[item] for item in type_order], dtype=float
    ) / EXPECTED_ARCHIVE_SERIES
    sample_proportions = np.array(
        [allocation[item] for item in type_order], dtype=float
    ) / REGISTERED_SAMPLE_SIZE
    positions = np.arange(len(type_order))
    width = 0.36
    axes[0].bar(
        positions - width / 2,
        archive_proportions,
        width,
        label="Full archive",
        color="#4c78a8",
    )
    axes[0].bar(
        positions + width / 2,
        sample_proportions,
        width,
        label="Registered sample",
        color="#f28e2b",
    )
    axes[0].set_xticks(positions, type_order, rotation=20)
    axes[0].set_ylabel("Proportion")
    axes[0].set_title("Variable-type composition")
    axes[0].legend()

    bins = np.geomspace(audit["length"].min(), audit["length"].max(), 35)
    axes[1].hist(
        audit["length"],
        bins=bins,
        density=True,
        alpha=0.45,
        label="Full archive",
        color="#4c78a8",
    )
    axes[1].hist(
        selected["length"],
        bins=bins,
        density=True,
        alpha=0.55,
        label="Registered sample",
        color="#f28e2b",
    )
    axes[1].set_xscale("log")
    axes[1].xaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].set_xlabel("Series length (days; log scale)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Length distribution (not used for selection)")
    axes[1].legend()

    colors = {
        "rain": "#4c78a8",
        "mintemp": "#59a14f",
        "maxtemp": "#e15759",
        "solar": "#f28e2b",
    }
    for series_type in type_order:
        rows = selected.loc[selected["series_type"] == series_type]
        axes[2].scatter(
            rows["source_order"],
            rows["length"],
            s=18,
            alpha=0.65,
            label=series_type,
            color=colors[series_type],
        )
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Original archive order")
    axes[2].set_ylabel("Series length (days; log scale)")
    axes[2].set_title("Frozen selected IDs across archive order")
    axes[2].legend(fontsize=8)

    figure.suptitle("Weather Daily preregistered 500-series sample", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(SAMPLE_FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(figure)

    report = {
        "status": "passed_and_frozen",
        "dataset_id": DATASET_ID,
        "sample_id": sample_id,
        "check_count": len(checks),
        "failed_check_count": 0,
        "archive_series_count": EXPECTED_ARCHIVE_SERIES,
        "registered_sample_size": REGISTERED_SAMPLE_SIZE,
        "registered_sample_type_counts": allocation,
        "raw_values_read": False,
        "time_split_created": False,
        "models_fit_or_tuned": False,
        "formal_test_accessed": False,
        "outputs": {
            "sample_manifest": str(SAMPLE_MANIFEST_PATH),
            "registration": str(SAMPLE_REGISTRATION_PATH),
            "checks": str(SAMPLE_CHECKS_PATH),
            "figure": str(SAMPLE_FIGURE_PATH),
        },
    }
    SAMPLE_REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("Weather Daily 500条固定样本登记全部通过")
    print("完整档案序列数量：", EXPECTED_ARCHIVE_SERIES)
    print("固定样本数量：", len(selected))
    print("固定种子：", REGISTERED_SEED)
    print("抽样算法：", SELECTION_ALGORITHM)
    print("按变量类型分配：", allocation)
    print(
        "样本序列长度（最短/中位数/最长）：",
        f"{int(selected['length'].min())} / {selected['length'].median():g} / "
        f"{int(selected['length'].max())}",
    )
    print("样本总观测数量：", int(selected["length"].sum()))
    print("样本中零IQR回退序列数量：", int(selected["iqr"].le(0.0).sum()))
    print("固定样本编号：", sample_id)
    print("原始序列值是否读取：否")
    print("预测性能是否用于抽样：否")
    print("时间切分是否创建：否")
    print("模型是否训练或调参：否")
    print("正式测试是否访问：否")
    print("抽样后是否允许重抽或替换：否")
    print("固定样本清单：", SAMPLE_MANIFEST_PATH)
    print("抽样登记文件：", SAMPLE_REGISTRATION_PATH)
    print("抽样检查表：", SAMPLE_CHECKS_PATH)
    print("抽样图片：", SAMPLE_FIGURE_PATH)
    print("抽样报告：", SAMPLE_REPORT_PATH)


if __name__ == "__main__":
    main()
