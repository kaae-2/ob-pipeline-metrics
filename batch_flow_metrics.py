#!/usr/bin/env python

"""Regenerate metric artifacts in fold-sized batches from collector run status."""

import argparse
import csv
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import flow_metrics


SCORED_POPULATION_METRICS = (
    "f1",
    "precision",
    "recall",
    "accuracy",
    "scaling_rate",
)


def artifact_needs_rerun(path):
    """Return whether a completed artifact has a non-finite truth-present score."""
    payload = flow_metrics._read_json_maybe_gzip(path)
    for run in payload.get("results", {}).values():
        if run.get("status", "completed") != "completed":
            continue
        for population in run.get("per_population", {}).values():
            if population.get("support", 0) <= 0:
                continue
            for metric in SCORED_POPULATION_METRICS:
                value = population.get(metric)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    return True
    return False


def read_jobs(run_status_path, stale_only):
    """Read unique artifact jobs and group them by shared fold inputs."""
    groups = defaultdict(list)
    seen_paths = set()
    with open(run_status_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            metric_path = row["metric_path"]
            if metric_path in seen_paths:
                raise ValueError(f"Duplicate metric path in run status: {metric_path}")
            seen_paths.add(metric_path)
            if stale_only and not artifact_needs_rerun(metric_path):
                continue

            path_parts = metric_path.split("/analysis/", 1)
            if len(path_parts) != 2:
                raise ValueError(f"Cannot derive truth path from metric path: {metric_path}")
            stratification_root = path_parts[0]
            truth_path = os.path.join(
                stratification_root, "data_import.test.labels.tar.gz"
            )
            groups[(truth_path, row["metadata_path"])].append(
                (metric_path, row["prediction_path"])
            )
    return groups


def process_group(group, metric_argument):
    """Load shared fold inputs once and regenerate every artifact in the group."""
    (truth_path, metadata_path), artifacts = group
    truth, truth_by_sample, truth_sample_order = (
        flow_metrics.load_true_labels_with_samples(truth_path)
    )
    metadata_payload = flow_metrics._load_metadata_payload(metadata_path)
    metrics_to_compute = flow_metrics.parse_metric_argument(metric_argument)

    for metric_path, prediction_path in artifacts:
        suffix = ".flow_metrics.json.gz"
        name = os.path.basename(metric_path)
        if not name.endswith(suffix):
            raise ValueError(f"Unexpected metric artifact name: {metric_path}")
        name = name[: -len(suffix)]
        payload = flow_metrics.compute_metrics_payload(
            prediction_path,
            truth,
            truth_by_sample,
            truth_sample_order,
            metadata_payload,
            metrics_to_compute,
            name,
        )
        output_path, _ = flow_metrics.write_metrics_outputs(
            payload,
            os.path.dirname(metric_path),
            name,
            truth_by_sample,
            truth_sample_order,
        )
        if os.path.abspath(output_path) != os.path.abspath(metric_path):
            raise ValueError(
                f"Batch output path differs from requested path: {metric_path}"
            )
    return len(artifacts)


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate flow metrics while loading each shared fold only once"
    )
    parser.add_argument("--run-status", required=True)
    parser.add_argument(
        "--metric",
        default="accuracy,precision,recall,balanced_accuracy,f1",
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--stale-only", action="store_true")
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("--jobs must be positive")

    groups = read_jobs(args.run_status, args.stale_only)
    artifact_count = sum(len(artifacts) for artifacts in groups.values())
    print(f"Processing {artifact_count} artifacts in {len(groups)} shared-input groups")
    if not groups:
        return

    completed = 0
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(groups))) as executor:
        futures = {
            executor.submit(process_group, group, args.metric): group[0]
            for group in groups.items()
        }
        for future in as_completed(futures):
            completed += future.result()
            print(f"Completed {completed}/{artifact_count} artifacts", flush=True)


if __name__ == "__main__":
    main()
