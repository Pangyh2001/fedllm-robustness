#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 GPU_INDEX MAX_USED_MIB MAX_UTIL_PERCENT POLL_SECONDS COMMAND..." >&2
  exit 2
fi

gpu_index=$1
max_used_mib=$2
max_util=$3
poll_seconds=$4
shift 4
ready_checks=${READY_CHECKS:-3}
ready_count=0

while (( ready_count < ready_checks )); do
  read -r used_mib util_percent < <(
    nvidia-smi \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits \
      -i "${gpu_index}" |
      tr -d ','
  )
  if (( used_mib <= max_used_mib && util_percent <= max_util )); then
    ready_count=$((ready_count + 1))
    echo "[$(date '+%F %T')] GPU ${gpu_index} ready check ${ready_count}/${ready_checks}: used=${used_mib}MiB util=${util_percent}%"
  else
    ready_count=0
    echo "[$(date '+%F %T')] waiting GPU ${gpu_index}: used=${used_mib}MiB util=${util_percent}%"
  fi
  if (( ready_count < ready_checks )); then
    sleep "${poll_seconds}"
  fi
done

export CUDA_VISIBLE_DEVICES=${gpu_index}
exec "$@"
