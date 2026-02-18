# Metrics Repository

## What this module does

This module computes evaluation metrics from predictions and true labels.

- Main entrypoint: `flow_metrics.py`
- Local runner: `run_metrics.sh`
- Output: `{name}.flow_metrics.json.gz`

It supports per-sample tar inputs, label-key naming, macro and per-population
metrics, and filters noise labels (`<= 0`) before scoring.

## Run locally

```bash
bash metrics/run_metrics.sh
```

Or direct CLI:

```bash
python metrics/flow_metrics.py --name dgcytof --output_dir metrics/out/... --analysis.prediction <predictions.tar.gz> --data.true_labels <test.labels.tar.gz> --metric all
```

## Run as part of benchmark

Wired in `benchmark/Clustering_conda.yml` metrics stage; run via:

```bash
just benchmark
```

## What `run_metrics.sh` needs

- Prediction tarball from a model under `metrics/out/data/analysis/...`
- True-label tarball under `metrics/out/data/data_preprocessing/...`
- Optional `data_import.order.json.gz` for wrapped-fold dedup metadata
- Python environment for `flow_metrics.py`
