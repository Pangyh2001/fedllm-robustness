#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/agnews_qwen3b.yaml}
GPU=${2:-0}
PYTHON_BIN=${PYTHON_BIN:-python}
SEEDS=${SEEDS:-"42 43 44"}
METHODS=${METHODS:-"fedavg fedpgd calfat sfat qfedavg_eat fedrda"}
SUITE_NAME=${SUITE_NAME:-agnews_alpha01}
OUTPUT_DIR=${OUTPUT_DIR:-}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-}
LOG_DIR=${LOG_DIR:-run_logs}

mkdir -p "${LOG_DIR}"

echo "config=${CONFIG}"
echo "gpu=${GPU}"
echo "python=${PYTHON_BIN}"
echo "seeds=${SEEDS}"
echo "methods=${METHODS}"
echo "output_dir=${OUTPUT_DIR:-<from-config>}"
echo "model=${MODEL_NAME_OR_PATH:-<from-config>}"

for seed in ${SEEDS}; do
  for method in ${METHODS}; do
    args=(
      run_experiment.py
      --config "${CONFIG}"
      --algorithm "${method}"
      --seed "${seed}"
      --run-name "${SUITE_NAME}"
    )
    if [[ -n "${OUTPUT_DIR}" ]]; then
      args+=(--output-dir "${OUTPUT_DIR}")
    fi
    if [[ -n "${MODEL_NAME_OR_PATH}" ]]; then
      args+=(--model-name-or-path "${MODEL_NAME_OR_PATH}")
    fi

    run_root="${OUTPUT_DIR:-outputs}"
    run_dir="${run_root}/${SUITE_NAME}__${method}__seed${seed}"
    if [[ -f "${run_dir}/summary.json" ]]; then
      echo "[$(date '+%F %T')] skip completed ${run_dir}"
      continue
    fi
    if [[ -f "${run_dir}/latest_checkpoint.pt" ]]; then
      args+=(--resume)
      echo "[$(date '+%F %T')] resume ${run_dir}"
    fi

    log="${LOG_DIR}/${SUITE_NAME}__${method}__seed${seed}.log"
    echo "[$(date '+%F %T')] start method=${method} seed=${seed} log=${log}"
    CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON_BIN}" -u "${args[@]}" 2>&1 | tee "${log}"
    echo "[$(date '+%F %T')] completed method=${method} seed=${seed}"
  done
done
