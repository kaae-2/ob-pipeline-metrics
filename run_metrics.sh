#!/usr/bin/env bash
set -euo pipefail

# Run data_preprocessing.py with the requested parameters.
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
python_bin="${script_dir}/.venv/bin/python"
[ -x "$python_bin" ] || python_bin="python"

labels_dir="${script_dir}/out/data/data_preprocessing/default"
label_key_dest="${labels_dir}/data_import.label_key.json.gz"
if [[ ! -f "$label_key_dest" ]]; then
  for candidate in \
    "${script_dir}/../preprocessing/out/data/data_import/preprocessing/data_preprocessing/default/data_import.label_key.json.gz" \
    "${script_dir}/../models/dgcytof/out/data/data_preprocessing/default/data_import.label_key.json.gz"; do
    if [[ -f "$candidate" ]]; then
      mkdir -p "$labels_dir"
      ln -sfn "$candidate" "$label_key_dest"
      break
    fi
  done
fi

"${python_bin}" "${script_dir}/flow_metrics.py" \
  --name "dgcytof" \
  --output_dir "${script_dir}/out/data/metrics/all/flow_metrics" \
  --analysis.prediction "${script_dir}/out/data/analysis/default/dgcytof/dgcytof_predicted_labels.tar.gz" \
  --data.true_labels "${script_dir}/out/data/data_preprocessing/default/data_import.test.labels.tar.gz" \
  --metric "all"
