#!/usr/bin/env bash
set -euo pipefail

# Run data_preprocessing.py with the requested parameters.
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
python_bin="${script_dir}/.venv/bin/python"
[ -x "$python_bin" ] || python_bin="python"

data_metadata_path="${script_dir}/out/data/data_import/default/data_import.metadata.json.gz"
if [[ ! -f "$data_metadata_path" ]]; then
  for candidate in \
    "${script_dir}/../data/out/data/data_import/default/data_import.metadata.json.gz" \
    "${script_dir}/../preprocessing/out/data/data_import/preprocessing/data_preprocessing/default/data_import.metadata.json.gz" \
    "${script_dir}/../stratify/out/data/data_import/preprocessing/data_preprocessing/stratify/data_stratify/default/data_import.metadata.json.gz"; do
    if [[ -f "$candidate" ]]; then
      data_metadata_path="$candidate"
      break
    fi
  done
fi

args=(
  --name "dgcytof"
  --output_dir "${script_dir}/out/data/metrics/all/flow_metrics"
  --analysis.prediction "${script_dir}/out/data/analysis/default/dgcytof/dgcytof_predicted_labels.tar.gz"
  --data.true_labels "${script_dir}/out/data/data_preprocessing/default/data_import.test.labels.tar.gz"
  --metric "all"
)

if [[ -f "$data_metadata_path" ]]; then
  args+=(--data.metadata "$data_metadata_path")
fi

"${python_bin}" "${script_dir}/flow_metrics.py" "${args[@]}"
