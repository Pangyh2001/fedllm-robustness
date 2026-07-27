#!/usr/bin/env bash
set -u

GPU_SELECTOR=${GPU_SELECTOR:-any}
MAX_USED_MIB=${MAX_USED_MIB:-30000}
MAX_UTIL_PERCENT=${MAX_UTIL_PERCENT:-100}
POLL_SECONDS=${POLL_SECONDS:-60}
RETRY_SECONDS=${RETRY_SECONDS:-60}
attempt=0

while true; do
  attempt=$((attempt + 1))
  echo "[$(date '+%F %T')] supervisor attempt=${attempt}"
  READY_CHECKS=1 bash scripts/wait_for_gpu_and_run.sh \
    "${GPU_SELECTOR}" \
    "${MAX_USED_MIB}" \
    "${MAX_UTIL_PERCENT}" \
    "${POLL_SECONDS}" \
    bash scripts/run_all_experiments.sh
  status=$?
  if (( status == 0 )); then
    echo "[$(date '+%F %T')] full experiment pipeline completed"
    exit 0
  fi
  echo "[$(date '+%F %T')] pipeline exited status=${status}; waiting ${RETRY_SECONDS}s before automatic resume"
  sleep "${RETRY_SECONDS}"
done
