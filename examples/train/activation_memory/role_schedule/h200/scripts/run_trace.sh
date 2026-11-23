#!/bin/bash
# 停顿定位 trace：仅 L1，A/B 各一组无屏障 trace 和一组 sync-event trace
# 每组预热 5 步、记录 2 步
source "$(dirname "$0")/common.sh"

for sched in A B; do
    extra=()
    if [ "$sched" = B ]; then extra+=(--sequential-backward); fi

    out="$RESULTS/raw/trace/L1_${sched}_free.json"
    mkdir -p "$(dirname "$out")"
    echo "=== trace free L1 $sched -> $out"
    python "$EXAMPLE/profile_dmd2.py" \
      --model "$WAN_MODEL" --training-sample "$SAMPLE_PT" --data "$PARQUET_DIR" \
      --layout-file "$L1" --warmup 5 --steps 2 --trace-events "${extra[@]}" \
      --output "$out" > "$out.stdout" 2>&1
    tail -c 200 "$out.stdout" | head -c 200; echo

    out="$RESULTS/raw/trace/L1_${sched}_sync20.json"
    echo "=== trace sync20 L1 $sched -> $out"
    python "$EXAMPLE/profile_dmd2.py" \
      --model "$WAN_MODEL" --training-sample "$SAMPLE_PT" --data "$PARQUET_DIR" \
      --layout-file "$L1" --warmup 5 --steps 2 --trace-events \
      --sync-event critic.backward.20 "${extra[@]}" \
      --output "$out" > "$out.stdout" 2>&1
    tail -c 200 "$out.stdout" | head -c 200; echo
done
echo ALL_TRACE_DONE