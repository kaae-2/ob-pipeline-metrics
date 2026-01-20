#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"

echo "--- run_metrics.sh output ---"
"${script_dir}/run_metrics.sh" 2>&1

out_path="${script_dir}/out/data/metrics/all/flow_metrics/dgcytof.flow_metrics.json.gz"

if [[ -f "$out_path" ]];
then
  echo "--- Metrics output (${out_path}) ---"
  gzip -cd "$out_path"
else
  echo "Metrics output not found at $out_path" >&2
  exit 1
fi
