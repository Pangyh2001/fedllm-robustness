#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/data/yhpang/miniconda3/envs/fedllm/bin/python3.12}
MODEL_PATH=${MODEL_PATH:-/data/yhpang/model_weights/qwen2_5_3b_instruct}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/yhpang/fedrda_residual_screen_20260819/outputs}
SEED=${SEED:-42}
METHODS=(fedavg fedpgd sfat fedrda)

mkdir -p "${OUTPUT_ROOT}"
for method in "${METHODS[@]}"; do
  "${PYTHON_BIN}" -u run_experiment.py \
    --config configs/agnews_qwen3b_residual_screen.yaml \
    --model-name-or-path "${MODEL_PATH}" \
    --output-dir "${OUTPUT_ROOT}" \
    --algorithm "${method}" \
    --seed "${SEED}"
done

