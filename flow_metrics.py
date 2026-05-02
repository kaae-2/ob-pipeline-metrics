#!/usr/bin/env python

"""
Lightweight CLI for evaluating prediction-style outputs (e.g. cell type
classification) using the same inputs as `run_metrics.py`, but focused on
per-population prediction metrics instead of clustering comparison.

Inputs mirror run_metrics:
- --analysis.prediction: csv/txt (optionally gzipped) with columns of predictions
  or a gzipped tar archive containing multiple prediction files (one per run)
- --data.true_labels: text file (optionally gzipped) of ground-truth labels (1D)
- --metric: comma-separated list from VALID_METRICS (or "all")
- --output_dir/--name: where to write results; printed to stdout if omitted

Metrics implemented (selected via --metric):
- accuracy, precision, recall/sensitivity, f1 (per-population with macro averages)
- mcc (multi-class extension), pop_freq_corr (frequency correlation),
  scaling_rate (per-pop accuracy divided by pop size)
- runtime: time spent computing the metrics for that run
- overlap: Jaccard overlap between predicted and true label sets (ignores 0)
- scalability: runtime normalized by number of evaluated samples

Note: this is a scaffolding for future refinement; runtime here measures the
metric computation itself, not the upstream model execution.
"""

import argparse
import csv
import gzip
import io
import json
import os
import re
import sys
import tarfile
import time
from glob import glob

import numpy as np
import pandas as pd
from pandas import Series

VALID_METRICS = {
    "accuracy",
    "precision",
    "recall",
    "sensitivity",
    "f1",
    "f1_score",
    "mcc",
    "pop_freq_corr",
    "scaling_rate",
    "runtime",
    "overlap",
    "scalability",
    "all",
}


CLASSIFICATION_METRICS = {
    "accuracy",
    "precision",
    "recall",
    "sensitivity",
    "f1",
    "mcc",
    "pop_freq_corr",
    "scaling_rate",
}


def _read_json_maybe_gzip(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def _load_metadata_payload(metadata_path):
    if not metadata_path:
        return None
    try:
        payload = _read_json_maybe_gzip(metadata_path)
    except Exception as exc:
        print(f"Warning: failed to read data.metadata payload: {exc}", file=sys.stderr)
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _load_dataset_metadata(metadata_payload):
    if not isinstance(metadata_payload, dict):
        return None
    dataset = metadata_payload.get("dataset")
    if isinstance(dataset, dict):
        return dataset
    metadata = metadata_payload.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    return None


def _read_first_line(path):
    """Read the first line of a (possibly gzipped) file."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        return handle.readline()


def _parse_label_content(content):
    lines = content.splitlines()
    first_line = lines[0] if lines else ""
    has_header = _has_header(first_line)
    if not has_header:
        first_token = first_line.strip().lower()
        if first_token in {
            "label",
            "labels",
            "population",
            "cell_type",
            "celltype",
            "cluster",
            "cluster_id",
        }:
            has_header = True

    def _read_with_sep(sep, text):
        return pd.read_csv(
            io.StringIO(text),
            sep=sep,
            engine="python",
            header=0 if has_header else None,
            comment="#",
            na_values=["", '""', "nan", "NaN"],
            skip_blank_lines=False,
        )

    try:
        df = _read_with_sep(",", content)
    except pd.errors.ParserError:
        df = _read_with_sep(r"\s+", content)

    if df.empty:
        return pd.Series([], dtype=float)
    return df.iloc[:, 0]


def _has_header(first_line):
    """Heuristically decide whether the first line is a header row.

    Treat a single-token line as data (not a header). A header is more likely
    when the first line contains multiple non-numeric tokens (column names).
    """
    tokens = [tok for tok in first_line.replace(",", " ").split() if tok]
    if not tokens:
        return False
    # Single token (e.g. a single string label) should not be treated as header.
    if len(tokens) == 1:
        return False
    for tok in tokens:
        try:
            float(tok)
        except ValueError:
            return True
    return False


def load_true_labels(data_file):
    """
    Load labels as 1D array; keeps missing labels as NaN (needed for
    semi-supervised handling in preprocessing).
    """
    labels, _, _ = load_true_labels_with_samples(data_file)
    return labels


def load_true_labels_with_samples(data_file):
    """
    Load labels as a concatenated 1D array plus per-sample mapping.
    """
    if tarfile.is_tarfile(data_file):
        return _load_true_labels_from_tar(data_file)

    opener = gzip.open if data_file.endswith(".gz") else open
    with opener(data_file, "rt") as handle:
        series = _parse_label_content(handle.read())

    try:
        labels = series.to_numpy()
    except Exception as exc:
        raise ValueError("Invalid data structure, cannot parse labels.") from exc

    if labels.ndim != 1:
        raise ValueError("Invalid data structure, not a 1D matrix?")
    sample_id = "sample0"
    return labels, {sample_id: labels}, [sample_id]


def _load_true_labels_from_tar(data_file):
    labels_list = []
    labels_by_sample = {}
    sample_order = []
    with tarfile.open(data_file, "r:*") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        for member in members:
            file_obj = tar.extractfile(member)
            if file_obj is None:
                continue
            content = file_obj.read()
            if member.name.endswith(".gz"):
                content = gzip.decompress(content)
            if not content or content.strip() == b"":
                series = pd.Series([], dtype=float)
            else:
                try:
                    series = _parse_label_content(content.decode("utf-8", errors="replace"))
                except (pd.errors.EmptyDataError, ValueError):
                    series = pd.Series([], dtype=float)
            labels_list.append(series)
            sample_id = member.name
            labels_by_sample[sample_id] = series.to_numpy()
            sample_order.append(sample_id)

    if not labels_list:
        return np.array([]), {}, []
    return (
        pd.concat(labels_list, ignore_index=True).to_numpy(),
        labels_by_sample,
        sample_order,
    )


def load_predicted_labels(data_file):
    """
    Load predicted labels allowing for optional header rows and gzip input.
    Returns a tuple of (column_headers, predictions_matrix).
    """
    first_line = _read_first_line(data_file)
    has_header = _has_header(first_line)

    opener = gzip.open if data_file.endswith(".gz") else open

    def _read_with_sep(sep):
        with opener(data_file, "rt") as handle:
            return pd.read_csv(
                handle,
                sep=sep,
                engine="python",
                header=0 if has_header else None,
                comment="#",
                na_values=["", '""', "nan", "NaN"],
                skip_blank_lines=False,
            )

    try:
        df = _read_with_sep(",")
    except pd.errors.ParserError:
        df = _read_with_sep(r"\s+")

    if df.empty:
        raise ValueError("Prediction file is empty.")

    try:
        values = df.to_numpy()
    except Exception as exc:
        raise ValueError("Invalid data structure, cannot parse predictions.") from exc

    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError("Invalid data structure, not a 2D matrix?")

    header = (
        [str(col) for col in df.columns]
        if has_header
        else [f"run{i}" for i in range(values.shape[1])]
    )
    return [np.array(header, dtype=str), values]


def _parse_prediction_content(content):
    """Parse a text blob of predictions into (headers, matrix)."""
    first_line = content.splitlines()[0] if content else ""
    has_header = _has_header(first_line)
    def _read_with_sep(sep, text):
        return pd.read_csv(
            io.StringIO(text),
            sep=sep,
            engine="python",
            header=0 if has_header else None,
            comment="#",
            na_values=["", '""', "nan", "NaN"],
            skip_blank_lines=False,
        )

    try:
        df = _read_with_sep(",", content)
    except pd.errors.ParserError:
        df = _read_with_sep(r"\s+", content)

    if df.empty:
        raise ValueError("Prediction file is empty.")

    values = df.to_numpy()
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError("Invalid data structure, not a 2D matrix?")

    headers = (
        [str(col) for col in df.columns]
        if has_header
        else [f"run{i}" for i in range(values.shape[1])]
    )
    return headers, values


def _load_predictions_from_tar(path):
    """Load predictions from a tar.gz of per-sample CSVs, preserving archive order."""
    predictions = []
    predictions_by_sample = {}
    sample_order = []
    with tarfile.open(path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        if not members:
            raise ValueError("Prediction archive is empty.")
        for member in members:
            file_obj = tar.extractfile(member)
            if file_obj is None:
                continue
            content = file_obj.read()
            if member.name.endswith(".gz"):
                content = gzip.decompress(content)
            if not content or content.strip() == b"":
                series = pd.Series([], dtype=float)
            else:
                try:
                    _, values = _parse_prediction_content(
                        content.decode("utf-8", errors="replace")
                    )
                    series = pd.Series(values[:, 0])
                except (pd.errors.EmptyDataError, ValueError):
                    series = pd.Series([], dtype=float)
            predictions.append(series)
            sample_id = member.name
            predictions_by_sample[sample_id] = series.to_numpy()
            sample_order.append(sample_id)

    if not predictions:
        raise ValueError("Prediction archive is empty.")

    concatenated = pd.concat(predictions, ignore_index=True).to_numpy()
    return {"run0": concatenated}, {"run0": predictions_by_sample}, sample_order


def _load_predictions_from_file(path):
    headers, values = load_predicted_labels(path)
    return {str(header): values[:, idx] for idx, header in enumerate(headers)}


def load_predicted_runs(path):
    """Load predictions from a plain file or a gzipped tar archive."""
    if tarfile.is_tarfile(path):
        return _load_predictions_from_tar(path)
    return _load_predictions_from_file(path), None, None


def _split_predictions_by_truth(predictions, truth_by_sample, sample_order):
    if not truth_by_sample or not sample_order:
        return None
    lengths = [len(truth_by_sample[sample_id]) for sample_id in sample_order]
    if sum(lengths) != len(predictions):
        return None
    split = {}
    offset = 0
    for sample_id, length in zip(sample_order, lengths):
        split[sample_id] = predictions[offset : offset + length]
        offset += length
    return split


def _sample_key_from_name(name):
    base = os.path.basename(name)
    for suffix in (".csv.gz", ".labels.gz", ".label.gz", ".txt.gz", ".csv", ".txt", ".gz"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    matches = re.findall(r"\d+", base)
    if matches:
        return matches[-1]
    return base


def _build_sample_key_map(sample_ids, label):
    mapping = {}
    for sample_id in sample_ids:
        key = _sample_key_from_name(sample_id)
        if key in mapping:
            raise ValueError(
                f"Duplicate {label} sample key '{key}' from '{sample_id}' and '{mapping[key]}'"
            )
        mapping[key] = sample_id
    return mapping


def _align_predictions_to_truth(
    truth_by_sample,
    truth_order,
    pred_by_sample,
    run_name,
    ungated_id=None,
    warn_state=None,
):
    truth_map = _build_sample_key_map(truth_order, "truth")
    pred_map = _build_sample_key_map(pred_by_sample.keys(), "prediction")

    missing = sorted(set(truth_map) - set(pred_map))
    extra = sorted(set(pred_map) - set(truth_map))
    if missing or extra:
        message = [
            f"Sample mismatch for run '{run_name}'.",
            f"Missing prediction keys: {missing}" if missing else None,
            f"Extra prediction keys: {extra}" if extra else None,
        ]
        raise ValueError(" ".join([part for part in message if part]))

    aligned_predictions = {}
    aligned_concat = []
    effective_ungated_id = ungated_id
    for truth_id in truth_order:
        key = _sample_key_from_name(truth_id)
        pred_id = pred_map[key]
        truth_values = truth_by_sample[truth_id]
        pred_values = pred_by_sample[pred_id]
        if len(pred_values) == 0:
            if effective_ungated_id is None:
                effective_ungated_id = _synthesize_ungated_id(truth_values)
                if effective_ungated_id is None:
                    raise ValueError(
                        "Empty prediction file and no 'Ungated' label id available."
                    )
                if warn_state is not None and not warn_state.get("synthetic_ungated"):
                    print(
                        "Warning: empty prediction file; using synthetic 'Ungated' label for metrics only.",
                        file=sys.stderr,
                    )
                    warn_state["synthetic_ungated"] = True
            pred_values = np.full(len(truth_values), effective_ungated_id)
        if len(truth_values) != len(pred_values):
            raise ValueError(
                "Sample length mismatch for run "
                f"'{run_name}' (truth '{truth_id}' vs prediction '{pred_id}')."
            )
        aligned_predictions[truth_id] = pred_values
        aligned_concat.append(pred_values)

    if not aligned_concat:
        return aligned_predictions, np.array([]), effective_ungated_id
    return aligned_predictions, np.concatenate(aligned_concat), effective_ungated_id


def _read_id_to_label(metadata_payload):
    if not isinstance(metadata_payload, dict):
        return {}
    labels = metadata_payload.get("labels")
    if isinstance(labels, dict):
        id_to_label = labels.get("id_to_label")
    else:
        id_to_label = metadata_payload.get("id_to_label")
    if not isinstance(id_to_label, dict):
        return {}
    return {str(key): str(value) for key, value in id_to_label.items()}


def _lookup_ungated_id(id_to_label):
    for key, label in id_to_label.items():
        if str(label).strip().lower() == "ungated":
            try:
                return int(key)
            except ValueError:
                return str(key)
    return None


def _synthesize_ungated_id(truth_values):
    truth_numeric = pd.to_numeric(np.ravel(truth_values), errors="coerce")
    truth_numeric = np.asarray(truth_numeric, dtype=float)
    truth_numeric = truth_numeric[~np.isnan(truth_numeric)]
    if truth_numeric.size == 0:
        return None
    try:
        max_value = int(np.max(truth_numeric))
    except (ValueError, TypeError):
        return None
    return max_value + 1


def _fill_missing_predictions(values, replacement):
    if replacement is None:
        return values
    if values.size == 0:
        return values
    mask = pd.isna(values)
    if not np.any(mask):
        return values
    filled = values.copy()
    filled[mask] = replacement
    return filled


def _infer_metadata_path(true_labels_path, name):
    directory = os.path.dirname(true_labels_path)
    candidate = os.path.join(directory, f"{name}.metadata.json.gz")
    if os.path.exists(candidate):
        return candidate

    filename = os.path.basename(true_labels_path)
    suffixes = [
        ".test.labels.tar.gz",
        ".train.labels.tar.gz",
        ".labels.tar.gz",
        ".labels.gz",
        ".labels",
    ]
    for suffix in suffixes:
        if filename.endswith(suffix):
            base = filename[: -len(suffix)]
            candidate = os.path.join(directory, f"{base}.metadata.json.gz")
            if os.path.exists(candidate):
                return candidate
            break

    candidates = glob(os.path.join(directory, "*.metadata.json.gz"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def parse_metric_argument(metric_arg):
    metrics = [m.strip().lower() for m in metric_arg.split(",") if m.strip()]
    if not metrics:
        raise ValueError("No metrics provided.")
    if "all" in metrics:
        metrics = sorted([m for m in VALID_METRICS if m != "all"])
    # Normalize aliases
    metrics = ["f1" if m == "f1_score" else m for m in metrics]
    metrics = list(dict.fromkeys(metrics))  # drop duplicates while preserving order
    invalid = [m for m in metrics if m not in VALID_METRICS]
    if invalid:
        raise ValueError(f"Invalid metric(s): {', '.join(invalid)}")
    return metrics


def _nan_safe_mean(values):
    vals = [v for v in values if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _count_populations(values):
    series: Series = pd.Series(np.ravel(values))
    series = series.dropna()
    if series.empty:
        return 0
    return int(series.nunique())


def _write_sample_stats_tsv(path, truth_by_sample, sample_order):
    order = sample_order or sorted(truth_by_sample)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sample_name", "n_cells", "n_populations"])
        for sample_id in order:
            values = truth_by_sample.get(sample_id)
            if values is None:
                continue
            writer.writerow(
                [
                    sample_id,
                    int(len(values)),
                    _count_populations(values),
                ]
            )


def strip_noise_labels(y_true, y_pred):
    y_true = np.array(y_true, ndmin=1)
    y_pred = np.array(y_pred, ndmin=1)

    true_numeric = pd.to_numeric(y_true, errors="coerce")
    pred_numeric = pd.to_numeric(y_pred, errors="coerce")
    numeric_mask = np.all(pd.isna(y_true) | ~pd.isna(true_numeric))

    if numeric_mask:
        y_true = np.asarray(true_numeric)
        y_pred = np.asarray(pred_numeric)
        mask = y_true > 0
        return y_true[mask], y_pred[mask]

    return y_true, y_pred


def compute_per_population_stats(y_true, y_pred, id_to_label=None):
    per_population = {}
    labels = np.unique(y_true)
    total = y_true.size
    label_lookup = id_to_label or {}
    for label in labels:
        pop_mask = y_true == label
        pop_size = pop_mask.sum()
        correct = (y_pred[pop_mask] == label).sum()
        tp = correct
        fp = ((y_true != label) & (y_pred == label)).sum()
        fn = ((y_true == label) & (y_pred != label)).sum()
        tn = total - tp - fp - fn

        pop_accuracy = float(correct / pop_size) if pop_size else float("nan")
        pop_precision = float(tp / (tp + fp)) if (tp + fp) else float("nan")
        pop_recall = float(tp / (tp + fn)) if (tp + fn) else float("nan")
        if (
            np.isnan(pop_precision)
            or np.isnan(pop_recall)
            or (pop_precision + pop_recall) == 0
        ):
            pop_f1 = float("nan")
        else:
            pop_f1 = float(
                2 * pop_precision * pop_recall / (pop_precision + pop_recall)
            )
        pop_scaling_rate = (
            float(pop_accuracy / pop_size) if pop_size else float("nan")
        )

        per_population[str(label)] = {
            "accuracy": pop_accuracy,
            "precision": pop_precision,
            "recall": pop_recall,
            "f1": pop_f1,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "scaling_rate": pop_scaling_rate,
            "support": int(pop_size),
            "n_cells": int(pop_size),
            "population_name": label_lookup.get(str(label)),
        }
    return per_population


def compute_macro_scores(per_population):
    macro_precision = _nan_safe_mean([v["precision"] for v in per_population.values()])
    macro_recall = _nan_safe_mean([v["recall"] for v in per_population.values()])
    macro_f1 = _nan_safe_mean([v["f1"] for v in per_population.values()])
    macro_accuracy = _nan_safe_mean([v["accuracy"] for v in per_population.values()])
    macro_scaling_rate = _nan_safe_mean(
        [v["scaling_rate"] for v in per_population.values()]
    )
    return macro_precision, macro_recall, macro_f1, macro_accuracy, macro_scaling_rate


def compute_confusion_matrix(y_true, y_pred):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((labels.size, labels.size), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[label_to_idx[t], label_to_idx[p]] += 1
    return labels, matrix


def compute_mcc(y_true, y_pred):
    if y_true.size == 0:
        return float("nan")
    labels, cm = compute_confusion_matrix(y_true, y_pred)
    n_samples = cm.sum()
    if n_samples == 0:
        return float("nan")

    t_sum = cm.sum(axis=1)
    p_sum = cm.sum(axis=0)
    c = np.trace(cm)
    s = (p_sum * t_sum).sum()

    numerator = c * n_samples - s
    denom_left = float(n_samples ** 2 - (p_sum ** 2).sum())
    denom_right = float(n_samples ** 2 - (t_sum ** 2).sum())
    denom_left = max(denom_left, 0.0)
    denom_right = max(denom_right, 0.0)
    denominator = np.sqrt(denom_left) * np.sqrt(denom_right)
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def compute_pop_freq_corr(y_true, y_pred):
    true_labels, true_counts = np.unique(y_true, return_counts=True)
    pred_labels, pred_counts = np.unique(y_pred, return_counts=True)

    true_map = {label: count for label, count in zip(true_labels, true_counts)}
    pred_map = {label: count for label, count in zip(pred_labels, pred_counts)}

    labels = sorted(set(true_map) | set(pred_map))
    if len(labels) < 2:
        return float("nan")

    true_freq = np.array([true_map.get(label, 0) for label in labels], dtype=float)
    pred_freq = np.array([pred_map.get(label, 0) for label in labels], dtype=float)

    true_std = np.std(true_freq)
    pred_std = np.std(pred_freq)
    if true_std == 0 or pred_std == 0:
        return float("nan")

    corr_matrix = np.corrcoef(pred_freq, true_freq)
    return float(corr_matrix[0, 1])


def metric_accuracy(base_stats):
    return {"accuracy": base_stats["overall_accuracy"]}


def metric_precision(base_stats):
    return {"precision_macro": base_stats["macro_precision"]}


def metric_recall(base_stats):
    return {"recall_macro": base_stats["macro_recall"]}


def metric_sensitivity(base_stats):
    return {"sensitivity_macro": base_stats["macro_recall"]}


def metric_f1(base_stats):
    return {"f1_macro": base_stats["macro_f1"]}


def metric_mcc(mcc_value):
    return {"mcc": mcc_value}


def metric_pop_freq_corr(correlation):
    return {"pop_freq_corr": correlation}


def metric_scaling_rate(base_stats):
    return {"scaling_rate_macro": base_stats["macro_scaling_rate"]}


def warn_missing_metrics(run_name, metrics_to_compute, results):
    metric_keys = {
        "accuracy": ["accuracy"],
        "precision": ["precision_macro"],
        "recall": ["recall_macro"],
        "sensitivity": ["sensitivity_macro"],
        "f1": ["f1_macro"],
        "mcc": ["mcc"],
        "pop_freq_corr": ["pop_freq_corr"],
        "scaling_rate": ["scaling_rate_macro"],
        "runtime": ["runtime_seconds"],
        "overlap": ["overlap"],
        "scalability": ["scalability_seconds_per_item"],
    }

    missing = []
    for metric in metrics_to_compute:
        keys = metric_keys.get(metric, [])
        for key in keys:
            if key not in results or results[key] is None:
                missing.append(metric)
                break

    if missing:
        print(
            "Warning: metrics missing for run "
            f"'{run_name}': {', '.join(sorted(set(missing)))}"
        )


def metric_overlap(y_true, y_pred):
    true_labels = set(np.unique(y_true))
    pred_labels = set(np.unique(y_pred))
    true_labels.discard(0)
    pred_labels.discard(0)
    union = true_labels | pred_labels
    intersection = true_labels & pred_labels
    overlap = float(len(intersection) / len(union)) if union else float("nan")
    return {"overlap": overlap}


def metric_runtime(runtime_seconds):
    return {"runtime_seconds": runtime_seconds}


def metric_scalability(runtime_seconds, n_items):
    return {
        "scalability_seconds_per_item": (
            float(runtime_seconds / n_items) if n_items else float("nan")
        )
    }


def compute_prediction_metrics(y_true, y_pred, metrics_to_compute, id_to_label=None):
    """
    Computes per-population metrics and optional runtime/overlap/scalability
    for a single set of predictions.
    """
    start = time.perf_counter()

    y_true = np.atleast_1d(y_true)
    y_pred = np.atleast_1d(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("Predicted labels and true labels must align in length.")

    # Keep the raw input size for auditability; filtered/evaluated counts can
    # differ after noise-label/NaN filtering below.
    n_cells_total = int(y_true.size)

    y_true, y_pred = strip_noise_labels(y_true, y_pred)

    valid_mask = (~pd.isna(y_true)) & (~pd.isna(y_pred))
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    results = {}
    results["n_cells_total"] = int(n_cells_total)
    results["n_cells"] = int(y_true.size)
    results["n"] = results["n_cells"]

    # Base stats computed once for classification-style metrics
    if any(metric in CLASSIFICATION_METRICS for metric in metrics_to_compute):
        per_population = compute_per_population_stats(
            y_true, y_pred, id_to_label=id_to_label
        )
        (
            macro_precision,
            macro_recall,
            macro_f1,
            macro_accuracy,
            macro_scaling_rate,
        ) = compute_macro_scores(per_population)
        base_stats = {
            "per_population": per_population,
            "overall_accuracy": (
                float((y_true == y_pred).mean()) if y_true.size else float("nan")
            ),
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "macro_accuracy": macro_accuracy,
            "macro_scaling_rate": macro_scaling_rate,
        }

        mcc_value = compute_mcc(y_true, y_pred) if "mcc" in metrics_to_compute else None
        pop_freq_corr = (
            compute_pop_freq_corr(y_true, y_pred)
            if "pop_freq_corr" in metrics_to_compute
            else None
        )

        metric_dispatch = {
            "accuracy": lambda: metric_accuracy(base_stats),
            "precision": lambda: metric_precision(base_stats),
            "recall": lambda: metric_recall(base_stats),
            "sensitivity": lambda: metric_sensitivity(base_stats),
            "f1": lambda: metric_f1(base_stats),
            "mcc": lambda: metric_mcc(mcc_value),
            "pop_freq_corr": lambda: metric_pop_freq_corr(pop_freq_corr),
            "scaling_rate": lambda: metric_scaling_rate(base_stats),
        }

        for metric_name, fn in metric_dispatch.items():
            if metric_name in metrics_to_compute:
                results.update(fn())

        results["per_population"] = per_population

    if "overlap" in metrics_to_compute:
        results.update(metric_overlap(y_true, y_pred))

    runtime_seconds = time.perf_counter() - start
    if "runtime" in metrics_to_compute:
        results.update(metric_runtime(runtime_seconds))
    if "scalability" in metrics_to_compute:
        results.update(metric_scalability(runtime_seconds, y_true.size))

    return results


def main():
    parser = argparse.ArgumentParser(description="Flow prediction metrics runner")

    parser.add_argument(
        "--analysis.prediction",
        type=str,
        required=True,
        help="csv/txt predictions (optionally gzipped) or a gzipped tar of multiple prediction files",
    )
    parser.add_argument(
        "--data.true_labels",
        type=str,
        required=True,
        help="text file containing the true labels (1D)",
    )
    parser.add_argument(
        "--data.metadata",
        type=str,
        help="metadata JSON.gz file containing label mappings and dataset context",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="output directory to store results (prints to stdout if omitted)",
    )
    parser.add_argument("--name", type=str, help="name of the dataset", default="flow")
    parser.add_argument(
        "--metric",
        type=str,
        default="all",
        help="comma-separated metrics to compute (or 'all')",
    )

    try:
        args = parser.parse_args()
    except SystemExit:
        parser.print_help()
        sys.exit(0)

    truth, truth_by_sample, truth_sample_order = load_true_labels_with_samples(
        getattr(args, "data.true_labels")
    )
    predicted_runs, predicted_samples, _ = load_predicted_runs(
        getattr(args, "analysis.prediction")
    )
    metadata_path = getattr(args, "data.metadata", None) or _infer_metadata_path(
        getattr(args, "data.true_labels"), args.name
    )
    metadata_payload = _load_metadata_payload(metadata_path)
    id_to_label = _read_id_to_label(metadata_payload)
    ungated_id = _lookup_ungated_id(id_to_label)
    metrics_to_compute = parse_metric_argument(args.metric)

    warn_state = {"synthetic_ungated": False}
    results = {}
    for run_name, predictions in predicted_runs.items():
        per_sample_predictions = None
        if predicted_samples and run_name in predicted_samples:
            per_sample_predictions, predictions, ungated_id = _align_predictions_to_truth(
                truth_by_sample,
                truth_sample_order,
                predicted_samples[run_name],
                run_name,
                ungated_id=ungated_id,
                warn_state=warn_state,
            )
        else:
            per_sample_predictions = _split_predictions_by_truth(
                predictions, truth_by_sample, truth_sample_order
            )

        if ungated_id is None and per_sample_predictions:
            missing = any(len(values) == 0 for values in per_sample_predictions.values())
            if missing:
                ungated_id = _synthesize_ungated_id(truth)
                if ungated_id is None:
                    raise ValueError(
                        "Empty prediction file and no 'Ungated' label id available."
                    )
                if not warn_state.get("synthetic_ungated"):
                    print(
                        "Warning: empty prediction file; using synthetic 'Ungated' label for metrics only.",
                        file=sys.stderr,
                    )
                    warn_state["synthetic_ungated"] = True

        if per_sample_predictions:
            per_sample_predictions = {
                key: _fill_missing_predictions(values, ungated_id)
                for key, values in per_sample_predictions.items()
            }
        predictions = _fill_missing_predictions(predictions, ungated_id)

        if predictions.shape[0] != truth.shape[0]:
            raise ValueError(
                f"Predicted labels rows ({predictions.shape[0]}) do not match true labels ({truth.shape[0]}) for run '{run_name}'."
            )

        metrics_for_run = compute_prediction_metrics(
            truth, predictions, metrics_to_compute, id_to_label=id_to_label
        )

        per_sample_metrics = {}
        if per_sample_predictions:
            for sample_id, sample_pred in per_sample_predictions.items():
                sample_truth = truth_by_sample.get(sample_id)
                if sample_truth is None and len(truth_by_sample) == 1:
                    only_truth = next(iter(truth_by_sample.values()))
                    if len(only_truth) == len(sample_pred):
                        sample_truth = only_truth
                if sample_truth is None:
                    continue
                per_sample_metrics[sample_id] = compute_prediction_metrics(
                    sample_truth,
                    sample_pred,
                    metrics_to_compute,
                    id_to_label=id_to_label,
                )

        if not per_sample_metrics:
            per_sample_metrics = {
                "sample0": compute_prediction_metrics(
                    truth,
                    predictions,
                    metrics_to_compute,
                    id_to_label=id_to_label,
                )
            }

        metrics_for_run["per_sample"] = per_sample_metrics
        warn_missing_metrics(run_name, metrics_to_compute, metrics_for_run)
        results[str(run_name)] = metrics_for_run

    payload = {
        "name": args.name,
        "metrics_requested": metrics_to_compute,
        "results": results,
    }
    dataset_metadata = _load_dataset_metadata(metadata_payload)
    if dataset_metadata is not None:
        payload["dataset_metadata"] = dataset_metadata
        n_variables = dataset_metadata.get("n_variables")
        if n_variables is not None:
            payload["n_variables"] = n_variables
    if metadata_payload is not None:
        payload["data_metadata"] = metadata_payload

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"{args.name}.flow_metrics.json.gz")
        with gzip.open(out_path, "wt") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Saved metrics to {out_path}")
        tsv_path = os.path.join(args.output_dir, f"{args.name}.sample_stats.tsv")
        _write_sample_stats_tsv(tsv_path, truth_by_sample, truth_sample_order)
        print(f"Saved sample stats to {tsv_path}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
