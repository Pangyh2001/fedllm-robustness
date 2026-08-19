#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
MODEL_3B=${MODEL_3B:-Qwen/Qwen2.5-3B-Instruct}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_v2/agnews_qwen3b}
LOG_DIR=${LOG_DIR:-run_logs/v2_pilot}
SEED=${SEED:-42}
METHODS=(fedpgd sfat fedrda)

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

for method in "${METHODS[@]}"; do
  run_name=agnews_alpha01_v2pilot
  run_dir="${OUTPUT_ROOT}/${run_name}__${method}__seed${SEED}"
  log="${LOG_DIR}/${run_name}__${method}__seed${SEED}.log"
  if [[ -f "${run_dir}/summary.json" ]]; then
    echo "[$(date '+%F %T')] SKIP ${method}: already complete"
    continue
  fi
  args=(
    run_experiment.py
    --config configs/agnews_qwen3b_v2_pilot.yaml
    --model-name-or-path "${MODEL_3B}"
    --output-dir "${OUTPUT_ROOT}"
    --run-name "${run_name}"
    --algorithm "${method}"
    --seed "${SEED}"
  )
  if [[ -f "${run_dir}/latest_checkpoint.pt" ]]; then
    args+=(--resume)
  fi
  echo "[$(date '+%F %T')] START ${method} seed=${SEED}"
  "${PYTHON_BIN}" -u "${args[@]}" 2>&1 | tee "${log}"
done
