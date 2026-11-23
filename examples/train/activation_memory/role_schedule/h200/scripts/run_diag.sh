#!/bin/bash
# 独立诊断：--allocator-events，L1/L2 × A/B，预热 5 步、记录 5 步
source "$(dirname "$0")/common.sh"

for spec in "L1 $L1" "L2 $L2"; do
    set -- $spec; tag=$1; layout=$2
    for sched in A B; do
        out="$RESULTS/raw/diag/${tag}_${sched}_alloc.json"
        mkdir -p "$(dirname "$out")"
        extra=()
        if [ "$sched" = B ]; then extra+=(--sequential-backward); fi
        echo "=== diag $tag $sched -> $out"
        python "$EXAMPLE/profile_dmd2.py" \
          --model "$WAN_MODEL" --training-sample "$SAMPLE_PT" --data "$PARQUET_DIR" \
          --layout-file "$layout" --warmup 5 --steps 5 --allocator-events "${extra[@]}" \
          --output "$out" > "$out.stdout" 2>&1
        tail -c 200 "$out.stdout" | head -c 200; echo
    done
done
echo ALL_DIAG_DONE