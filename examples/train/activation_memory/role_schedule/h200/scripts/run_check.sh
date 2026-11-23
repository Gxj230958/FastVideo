#!/bin/bash
# 数值等价性：L1 同输入，单进程比较两种调度的一次确定性更新
source "$(dirname "$0")/common.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
free -g
out="$RESULTS/raw/check/one_update.json"
mkdir -p "$(dirname "$out")"
set +e
python "$EXAMPLE/profile_dmd2.py" \
  --model "$WAN_MODEL" --training-sample "$SAMPLE_PT" --data "$PARQUET_DIR" \
  --layout-file "$L1" --check-role-schedules --output "$out" \
  > "$out.stdout" 2>&1
rc=$?
set -e
echo "check exit code: $rc"
exit $rc