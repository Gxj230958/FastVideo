#!/bin/bash
# 受控显存压力实验（L1）：每档先跑 A/B 计数，再按规则决定是否配对测速
# 用法: run_pressure.sh <counter|timing|trace> <cap_gib>
source "$(dirname "$0")/common.sh"
mode=$1; cap_gib=$2
cap_bytes=$((cap_gib * 1024 * 1024 * 1024))
LAUNCHER="$RESULTS/scripts/capped_launcher.py"

case "$mode" in
counter)  # A/B 各一次 --allocator-events，预热5记录5
    for sched in A B; do
        out="$RESULTS/raw/pressure/cap${cap_gib}_L1_${sched}_alloc.json"
        mkdir -p "$(dirname "$out")"
        extra=(); [ "$sched" = B ] && extra+=(--sequential-backward)
        echo "=== pressure counter cap=${cap_gib}GiB L1 $sched"
        set +e
        python "$LAUNCHER" "$cap_bytes" \
          --model "$WAN_MODEL" --training-sample "$SAMPLE_PT" --data "$PARQUET_DIR" \
          --layout-file "$L1" --warmup 5 --steps 5 --allocator-events "${extra[@]}" \
          --output "$out" > "$out.stdout" 2>&1
        rc=$?
        set -e
        echo "exit=$rc"; [ $rc -ne 0 ] && tail -20 "$out.stdout"
    done
    ;;
timing)  # 三组同卡配对测速（BA、AB、BA），预热5测10
    for rnd in 1 2 3; do
        case $rnd in 1|3) order="B A";; 2) order="A B";; esac
        for sched in $order; do
            out="$RESULTS/raw/pressure/cap${cap_gib}_L1_r${rnd}_${sched}.json"
            extra=(); [ "$sched" = B ] && extra+=(--sequential-backward)
            echo "=== pressure timing cap=${cap_gib}GiB round$rnd $sched"
            python "$LAUNCHER" "$cap_bytes" \
              --model "$WAN_MODEL" --training-sample "$SAMPLE_PT" --data "$PARQUET_DIR" \
              --layout-file "$L1" --warmup 5 --steps 10 "${extra[@]}" \
              --output "$out" > "$out.stdout" 2>&1
        done
    done
    ;;
trace)  # 热点 trace：sync-event critic.backward.20，预热5记录2
    for sched in A B; do
        out="$RESULTS/raw/pressure/cap${cap_gib}_L1_${sched}_sync20.json"
        extra=(); [ "$sched" = B ] && extra+=(--sequential-backward)
        echo "=== pressure trace cap=${cap_gib}GiB L1 $sched"
        python "$LAUNCHER" "$cap_bytes" \
          --model "$WAN_MODEL" --training-sample "$SAMPLE_PT" --data "$PARQUET_DIR" \
          --layout-file "$L1" --warmup 5 --steps 2 --trace-events \
          --sync-event critic.backward.20 "${extra[@]}" \
          --output "$out" > "$out.stdout" 2>&1
    done
    ;;
*) echo "unknown mode $mode" >&2; exit 2;;
esac
echo "PRESSURE_${mode}_DONE cap=${cap_gib}GiB"