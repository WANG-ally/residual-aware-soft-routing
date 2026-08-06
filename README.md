# Residual-Aware Soft Routing with Adaptive Model-Usage Control for Time-Series Forecasting

This repository contains the code and compact evidence package for a study of
coverage-controlled model routing in causal one-step-ahead time-series forecasting.
The method routes each prediction between a linear Ridge autoregression and a
LightGBM forecaster. A sparse logistic router learns temperature-smoothed model
preferences from causal context, forecast disagreement, and lagged residuals. A
separate feedback controller adapts the routing threshold to target 70% Ridge usage.

The study evaluates an explicit accuracy--model-usage trade-off. It does **not**
claim state-of-the-art forecasting accuracy.

## Study design

Each series was divided chronologically into four non-overlapping segments:

1. `base_train`: base-forecaster fitting and hyperparameter selection;
2. `router_train`: router fitting and hyperparameter selection;
3. `calibration`: threshold and controller-step selection; and
4. `test`: one-time formal evaluation after the full protocol was frozen.

The base models, router, calibration rule, comparators, evaluation code, and
software environment were frozen before formal test access. All five dataset
experiments are closed to further model changes or test reruns.

## Main registered findings

- The adaptive router met the 70% Ridge-usage target within the registered
  two-percentage-point tolerance on 4 of 5 datasets.
- Its absolute Ridge-usage deviation was smaller than that of the static router
  on all five datasets.
- Relative to static routing, adaptive routing improved RMSSE on 1 of 5 datasets
  and changed RMSSE by `+0.639%` on average; positive values indicate worse RMSSE.
- A Friedman test across 12 deployable methods yielded `p = 0.00923`, but no
  comparison involving the proposed method remained significant after Holm
  correction.
- The preregistered two-sided dataset-level Wilcoxon test for the five
  adaptive-versus-static usage-deviation differences yielded `p = 0.0625`.

These results support improved descriptive target adherence with a
dataset-dependent accuracy cost, rather than universal accuracy superiority.

| Dataset | Adaptive RMSSE | Ridge usage | Absolute deviation | Constraint |
|---|---:|---:|---:|:---:|
| NN5 Daily | 0.772689 | 66.833% | 3.167 pp | Fail |
| Pedestrian Hourly | 0.398511 | 70.138% | 0.138 pp | Pass |
| M4 Hourly | 1.082009 | 71.875% | 1.875 pp | Pass |
| Electricity Hourly | 0.583202 | 70.004% | 0.004 pp | Pass |
| Weather Daily | 0.691972 | 70.004% | 0.004 pp | Pass |

The machine-readable source for this display is
[`supplement/tables/table_8_adaptive_static_display.csv`](supplement/tables/table_8_adaptive_static_display.csv).

## Public datasets

The data are public records from the Monash Time Series Forecasting Repository.
Raw archives are not redistributed by this repository.

| Dataset | Frequency | Series used | Public archive |
|---|---|---:|---|
| NN5 Daily | Daily | 111 | [10.5281/zenodo.4656117](https://doi.org/10.5281/zenodo.4656117) |
| Pedestrian Hourly | Hourly | 66 | [10.5281/zenodo.4656626](https://doi.org/10.5281/zenodo.4656626) |
| M4 Hourly | Hourly | 414 | [10.5281/zenodo.4656589](https://doi.org/10.5281/zenodo.4656589) |
| Electricity Hourly | Hourly | 321 | [10.5281/zenodo.4656140](https://doi.org/10.5281/zenodo.4656140) |
| Weather Daily | Daily | 500 | [10.5281/zenodo.4654822](https://doi.org/10.5281/zenodo.4654822) |

Weather Daily results apply only to the deterministic, preregistered
500-series sample identified by SHA-256
`5518ada55d6adb8a8df20f5a30d52ec904362ce57df9adeeceb0bc3d22854044`.

Archive URLs, filenames, MD5 values, licenses, and provenance notes are recorded
in [`data_manifest.csv`](data_manifest.csv).

## Repository structure

```text
.
├── code/publication_pipeline/       # Clean, path-neutral analysis modules
│   ├── nn5/
│   ├── pedestrian/
│   ├── m4_hourly/
│   ├── electricity_hourly/
│   ├── weather_daily/
│   └── cross_dataset/
├── figures/                         # Six selected statistical figures
├── supplement/
│   ├── evidence/                    # Dataset cards and cross-dataset summaries
│   └── tables/                      # Eight manuscript tables in CSV format
├── data_manifest.csv
├── download_public_data.py
├── environment.yml
├── experiment_config.yaml
├── tsf_reader.py
└── tsf_reader_compat.py
```

The publication pipeline contains 67 Python modules with repository-relative
paths and English technical comments. The exact evaluator sources corresponding
to the `evaluator_sha256` values are retained in the private audit archive. The
public modules are path-neutral publication copies; non-semantic cleanup of
paths, filenames, comments, and console messages changes their byte-level
hashes. The evaluator hashes in the evidence cards therefore identify the
original frozen evaluators and are not expected to match the publication
copies. The evidence package preserves the registered freeze identifiers,
one-time test status, aggregate results, and closeout identifiers. Use
`code/publication_pipeline/` for code review and reuse.

## Environment setup

The registered environment uses Python 3.11.15. `environment.yml` pins the
package versions observed in the frozen experimental environment. On macOS or
another Conda platform:

```bash
conda env create -f environment.yml
conda activate sci-routing
python --version
```

The main registered packages include NumPy, pandas, SciPy, scikit-learn,
LightGBM, PyArrow, Matplotlib, PyYAML, and joblib. The global random seed is
`20260714`.

## Data download

List the registered public snapshots without downloading them:

```bash
python download_public_data.py --list
```

Download and extract each formal-study collection into the path expected by
the corresponding audit module:

```bash
python download_public_data.py --names nn5_daily --output-dir data/raw --extract
python download_public_data.py --names pedestrian_hourly --output-dir data/raw/pedestrian_hourly_staging --extract
python download_public_data.py --names m4_hourly --output-dir data/raw/m4_hourly_staging --extract
python download_public_data.py --names electricity_hourly --output-dir data/raw/electricity_hourly_staging --extract
python download_public_data.py --names weather_daily --output-dir data/raw/weather_daily_staging --extract
```

The downloader verifies each archive against the registered MD5 value before
extraction. The `data/` directory is excluded from version control.

## Analysis modules

The conceptual order within each dataset directory is:

1. `data_audit.py`
2. `sample_registration.py` (Weather Daily only)
3. `time_split.py`
4. `window_preparation.py`
5. `ridge_tuning.py`
6. `lightgbm_tuning.py`
7. `fit_base_models.py`
8. `router_features.py`
9. `soft_router_tuning.py`
10. `local_weighting.py`
11. `coverage_calibration.py`
12. `pretest_baselines.py`
13. `formal_test.py`
14. `statistical_analysis.py`

The formal-test modules are integrity guarded and reflect a one-time closed
evaluation protocol. They must not be used for iterative tuning or repeated
test access. The repository does not include raw observations, fitted models,
formal test predictions, or regenerable working logs.

## Evidence package

- `supplement/evidence/*_evidence_card.yaml`: five dataset-level evidence cards;
- `supplement/evidence/cross_dataset_*`: cross-dataset summaries and tests;
- `supplement/evidence/manuscript_claim_registry.yaml`: supported and unsupported
  manuscript claims;
- `supplement/tables/table_1_*.csv` through `table_8_*.csv`: manuscript tables;
- `figures/cross_dataset_performance_and_coverage.png`: main cross-dataset figure;
- `figures/*_statistical_comparison.png`: dataset-level statistical figures.

These artifacts are compact read-only outputs. Large intermediate Parquet
files, fitted models, raw archives, and one-time test predictions are excluded.

## Citation

The associated manuscript is currently in preparation:

> Wang, Dongyang. *Residual-Aware Soft Routing with Adaptive Model-Usage Control
> for Time-Series Forecasting*. 2026.

The citation information will be updated when a persistent article identifier
becomes available.

## License and data terms

The source code in this repository is released under the [MIT License](LICENSE).
The public datasets remain subject to the licenses and attribution requirements
of their original archive records. Dataset ownership is not transferred by this
code license.
