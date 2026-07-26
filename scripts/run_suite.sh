#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/agnews_qwen3b.yaml}
GPU=${2:-0}
PYTHON_BIN=${PYTHON_BIN:-/data/yhpang/miniconda3/envs/fedllm/bin/python3.12}
SEEDS=${SEEDS:-"42 43 44"}
METHODS=${METHODS:-"fedavg fedpgd calfat sfat qfedavg_eat fedrda"}
SUITE_NAME=${SUITE_NAME:-agnews_alpha01}

mkdir -p run_logs

echo "config=${CONFIG}"
echo "gpu=${GPU}"
echo "python=${PYTHON_BIN}"
echo "seeds=${SEEDS}"
echo "methods=${METHODS}"

for seed in ${SEEDS}; do
  for method in ${METHODS}; do
    log="run_logs/${SUITE_NAME}__${method}__seed${seed}.log"
    echo "[$(date '+%F %T')] start method=${method} seed=${seed} log=${log}"
    CUDA_VISIBLE_DEVICES=${GPU} "${PYTHON_BIN}" -u run_experiment.py \
      --config "${CONFIG}" \
      --algorithm "${method}" \
      --seed "${seed}" \
      --run-name "${SUITE_NAME}" \
      2>&1 | tee "${log}"
    echo "[$(date '+%F %T')] completed method=${method} seed=${seed}"
  done
done
