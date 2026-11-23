#!/bin/bash
# 正常条件正式测速：L1/L2 各四轮配对，轮内顺序 AB、BA、BA、AB
# 每进程：预热 5 步、测 10 步，不开 profiler/事件钩子
source "$(dirname "$0")/common.sh"

run_one() {  # $1=layout_tag $2=layout_file $3=round $4=sched(A|B)
    local tag="$1" layout="$2" round="$3" sched="$4"
    local out="$RESULTS/raw/normal/${tag}_r${round}_${sched}.json"
    mkdir -p "$(dirname "$out")"
    local extra=()
    if [ "$sched" = B ]; then extra+=(--sequential-backward); fi
    echo "=== normal $tag round$round $sched -> $out"
    python "$EXAMPLE/profile_dmd2.py" \
      --model "$WAN_MODEL" --training-sample "$SAMPLE_PT" --data "$PARQUET_DIR" \
      --layout-file "$layout" --warmup 5 --steps 10 "${extra[@]}" --output "$out" \
      > "$out.stdout" 2>&1
    tail -c 300 "$out.stdout" | head -c 300; echo
}

for spec in "L1 $L1" "L2 $L2"; do
    set -- $spec; tag=$1; layout=$2
    run_one "$tag" "$layout" 1 A; run_one "$tag" "$layout" 1 B
    run_one "$tag" "$layout" 2 B; run_one "$tag" "$layout" 2 A
    run_one "$tag" "$layout" 3 B; run_one "$tag" "$layout" 3 A
    run_one "$tag" "$layout" 4 A; run_one "$tag" "$layout" 4 B
done
echo ALL_NORMAL_DONE