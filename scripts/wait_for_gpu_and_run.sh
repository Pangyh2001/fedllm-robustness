#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 {GPU_INDEX|any} MAX_USED_MIB MAX_UTIL_PERCENT POLL_SECONDS COMMAND..." >&2
  exit 2
fi

gpu_selector=$1
max_used_mib=$2
max_util=$3
poll_seconds=$4
shift 4
ready_checks=${READY_CHECKS:-3}
ready_count=0
candidate=""

select_available_gpu() {
  while IFS=',' read -r index used_mib util_percent; do
    index=${index//[[:space:]]/}
    used_mib=${used_mib//[[:space:]]/}
    util_percent=${util_percent//[[:space:]]/}
    if [[ "${gpu_selector}" != "any" && "${index}" != "${gpu_selector}" ]]; then
      continue
    fi
    if (( used_mib <= max_used_mib && util_percent <= max_util )); then
      echo "${index} ${used_mib} ${util_percent}"
      return 0
    fi
  done < <(
    nvidia-smi \
      --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )
  return 1
}

while (( ready_count < ready_checks )); do
  if selection=$(select_available_gpu); then
    read -r selected_gpu used_mib util_percent <<<"${selection}"
    if [[ "${selected_gpu}" == "${candidate}" ]]; then
      ready_count=$((ready_count + 1))
    else
      candidate=${selected_gpu}
      ready_count=1
    fi
    echo "[$(date '+%F %T')] GPU ${candidate} ready check ${ready_count}/${ready_checks}: used=${used_mib}MiB util=${util_percent}%"
  else
    candidate=""
    ready_count=0
    snapshot=$(
      nvidia-smi \
        --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits |
        tr '\n' ';'
    )
    echo "[$(date '+%F %T')] waiting for any eligible GPU: ${snapshot}"
  fi
  if (( ready_count < ready_checks )); then
    sleep "${poll_seconds}"
  fi
done

echo "[$(date '+%F %T')] selected physical GPU ${candidate}"
export CUDA_VISIBLE_DEVICES=${candidate}
exec "$@"
