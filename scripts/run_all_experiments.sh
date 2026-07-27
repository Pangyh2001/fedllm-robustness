#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/formal}
LOG_DIR=${LOG_DIR:-run_logs/formal}
MODEL_SMOKE=${MODEL_SMOKE:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_3B=${MODEL_3B:-Qwen/Qwen2.5-3B-Instruct}
MODEL_7B=${MODEL_7B:-Qwen/Qwen2.5-7B-Instruct}
RUN_TEXT_ATTACKS=${RUN_TEXT_ATTACKS:-1}
SEEDS=(42 43 44)
METHODS=(fedavg fedpgd calfat sfat qfedavg_eat fedrda)

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

run_one() {
  local config=$1
  local model=$2
  local output_dir=$3
  local run_name=$4
  local method=$5
  local seed=$6
  shift 6

  local run_dir="${output_dir}/${run_name}__${method}__seed${seed}"
  local log="${LOG_DIR}/${run_name}__${method}__seed${seed}.log"
  if [[ -f "${run_dir}/summary.json" ]]; then
    echo "[$(date '+%F %T')] SKIP completed ${run_dir}"
    return
  fi

  local args=(
    run_experiment.py
    --config "${config}"
    --model-name-or-path "${model}"
    --output-dir "${output_dir}"
    --run-name "${run_name}"
    --algorithm "${method}"
    --seed "${seed}"
    "$@"
  )
  if [[ -f "${run_dir}/latest_checkpoint.pt" ]]; then
    args+=(--resume)
    echo "[$(date '+%F %T')] RESUME ${run_dir}"
  fi

  echo "[$(date '+%F %T')] START ${run_name} method=${method} seed=${seed}"
  "${PYTHON_BIN}" -u "${args[@]}" 2>&1 | tee "${log}"
  echo "[$(date '+%F %T')] DONE ${run_name} method=${method} seed=${seed}"
}

echo "[$(date '+%F %T')] stage=unit_tests"
"${PYTHON_BIN}" -m unittest discover -s tests -v

echo "[$(date '+%F %T')] stage=smoke"
for method in "${METHODS[@]}"; do
  run_one configs/smoke.yaml "${MODEL_SMOKE}" \
    "${OUTPUT_ROOT}/smoke" smoke "${method}" 42
done

echo "[$(date '+%F %T')] stage=agnews_qwen3b_main"
for seed in "${SEEDS[@]}"; do
  for method in "${METHODS[@]}"; do
    run_one configs/agnews_qwen3b.yaml "${MODEL_3B}" \
      "${OUTPUT_ROOT}/agnews_qwen3b" agnews_alpha01 "${method}" "${seed}"
  done
done

echo "[$(date '+%F %T')] stage=agnews_non_iid"
for alpha_spec in "05:0.5" "10:1.0"; do
  run_suffix=${alpha_spec%%:*}
  alpha=${alpha_spec##*:}
  for seed in "${SEEDS[@]}"; do
    for method in fedpgd sfat fedrda; do
      run_one configs/agnews_qwen3b.yaml "${MODEL_3B}" \
        "${OUTPUT_ROOT}/agnews_qwen3b" "agnews_alpha${run_suffix}" \
        "${method}" "${seed}" --dirichlet-alpha "${alpha}"
    done
  done
done

echo "[$(date '+%F %T')] stage=dbpedia_qwen3b"
for seed in "${SEEDS[@]}"; do
  for method in "${METHODS[@]}"; do
    run_one configs/dbpedia_qwen3b.yaml "${MODEL_3B}" \
      "${OUTPUT_ROOT}/dbpedia_qwen3b" dbpedia_alpha01 "${method}" "${seed}"
  done
done

echo "[$(date '+%F %T')] stage=agnews_qwen7b"
for seed in "${SEEDS[@]}"; do
  for method in fedavg fedpgd sfat qfedavg_eat fedrda; do
    run_one configs/agnews_qwen7b.yaml "${MODEL_7B}" \
      "${OUTPUT_ROOT}/agnews_qwen7b" agnews_alpha01 "${method}" "${seed}"
  done
done

echo "[$(date '+%F %T')] stage=ablations"
for seed in "${SEEDS[@]}"; do
  run_one configs/agnews_qwen3b.yaml "${MODEL_3B}" \
    "${OUTPUT_ROOT}/ablations" ablation_no_consistency fedrda "${seed}" \
    --clean-consistency-weight 0
  run_one configs/agnews_qwen3b.yaml "${MODEL_3B}" \
    "${OUTPUT_ROOT}/ablations" ablation_no_tail fedrda "${seed}" \
    --warmup-rounds 50
  for ratio in 0.1 0.3; do
    suffix=${ratio/./}
    run_one configs/agnews_qwen3b.yaml "${MODEL_3B}" \
      "${OUTPUT_ROOT}/ablations" "ablation_tail${suffix}" fedrda "${seed}" \
      --tail-ratio "${ratio}"
  done
  for rank in 4 16; do
    run_one configs/agnews_qwen3b.yaml "${MODEL_3B}" \
      "${OUTPUT_ROOT}/ablations" "ablation_rank${rank}" fedrda "${seed}" \
      --lora-rank "${rank}"
  done
  for weight in 0.5 1.5; do
    suffix=${weight/./}
    run_one configs/agnews_qwen3b.yaml "${MODEL_3B}" \
      "${OUTPUT_ROOT}/ablations" "ablation_residual${suffix}" fedrda "${seed}" \
      --residual-weight "${weight}"
  done
done

if [[ "${RUN_TEXT_ATTACKS}" == "1" ]]; then
  echo "[$(date '+%F %T')] stage=text_attacks"
  "${PYTHON_BIN}" -c "import textattack" || {
    echo "TextAttack is missing; training is complete but discrete attacks cannot start." >&2
    exit 3
  }
  for seed in "${SEEDS[@]}"; do
    for method in "${METHODS[@]}"; do
      run_dir="${OUTPUT_ROOT}/agnews_qwen3b/agnews_alpha01__${method}__seed${seed}"
      if [[ -f "${run_dir}/text_attack_metrics.json" ]]; then
        echo "[$(date '+%F %T')] SKIP completed text attacks ${run_dir}"
        continue
      fi
      log="${LOG_DIR}/text_attack__${method}__seed${seed}.log"
      "${PYTHON_BIN}" -u evaluate_text_attacks.py \
        --config configs/agnews_qwen3b.yaml \
        --model-name-or-path "${MODEL_3B}" \
        --run-dir "${run_dir}" \
        --attacks bert_attack deepwordbug \
        --num-examples-per-client -1 \
        --query-budget 1000 \
        --seed "${seed}" 2>&1 | tee "${log}"
    done
  done
fi

"${PYTHON_BIN}" scripts/update_experiment_report.py \
  --outputs "${OUTPUT_ROOT}" \
  --report EXPERIMENTS.md
echo "[$(date '+%F %T')] ALL EXPERIMENTS COMPLETED"
