# Minimal edits to the Wan DMD2 critic layout

Can an activation-lifetime estimate identify a block to recompute that prevents an allocator retry? The comparison keeps the solver's 24 saved student blocks, all training settings and a 54 GiB allocated-memory budget fixed. The early (A) and late (B) layouts within each pair save the same number of critic blocks. Block numbers are zero-based.

| Layout | Critic blocks switched from save to recompute | Predicted at forward 18 (GiB) | Predicted peak (GiB) |
|---|---|---:|---:|
| Solver | None | 50.475 | 53.891 |
| A1 | 12 | 50.181 | 53.597 |
| B1 | 24 | 50.475 | 53.597 |
| A2 | 11, 12 | 49.886 | 53.302 |
| B2 | 24, 25 | 50.475 | 53.302 |

The initial prediction was fewer forward-18 retries for A1 than B1. An activation retained at block 12 is still live at block 18. An activation at block 24 has yet to be produced. The profile estimates allocated bytes. Allocation requests can also depend on reserve left over from a previous update. [Initial prediction](discovery_prediction.json) · [Calibration and original layout](../stall_study/)

## Reproduce

Use the [environment and video preparation commands](../stall_study/README.md#inputs). The study used Ascend 910B2C, CANN 8.5.1, PyTorch 2.9.0, torch-npu 2.9.0.post1, Wan2.1-1.3B, 17 frames at 256 × 448, FP32 master parameters, BF16 compute, Torch SDPA, one rank, batch one and seed 2026. A new process loads the exact map for each condition, performs five warmup updates and measures iterations 5–9. Each process uses eight PyTorch threads. Use a fresh output directory when changing inputs or runtime; completed process files can be resumed within a batch. Cards 0, 1 and 2 used 12 disjoint logical CPUs apiece and rotated layout order.

From the checkout root, set `WAN_MODEL` to the local checkpoint and `data` to the directory containing the two encoded samples and their parquet directories. Run each mode as a separate batch. The following Bash function reproduces the comparisons sequentially on one available NPU. The published runs launched the three matched blocks concurrently on separate devices with distinct `MASTER_PORT` values. The local launcher sourced CANN and checked device health and availability before every process.

```bash
export PYTHONPATH="$PWD"
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
study=examples/train/activation_memory/minimal_layout_edits
data=inputs
out=measurements/minimal_layout_edits
batch() {
    for block in 0 1 2; do
        python "$study/run.py" --model "$WAN_MODEL" --data "$data" --output "$out" \
            --stage "$1" --sample "$2" --mode "$3" --block "$block" || return 1
    done
}
for stage in single double; do
    batch "$stage" hiker counters && batch "$stage" hiker timing || exit 1
done
batch migration hiker counters || exit 1
```

The baseline and shortlist winner were measured contemporaneously in the single-edit group. The two-edit group uses that Hiker baseline for its within-card ratios. Validation starts from a fresh solver baseline on each clip. The runner computes the 127-event budget from the exact 60-block map before launch and records the map echoed by the training program.

## Read the measurements

Treat the mean of five uninstrumented updates in a process as one timing observation. The paired difference is A minus B in milliseconds on the same physical card. The ratio to the solver is edited time divided by solver time. A ratio below one is faster. Peak allocated and reserved values are maxima over the timed updates.

Counter-only processes use the same warmups, samples and iterations. The whole-update counter includes the final device synchronization. At each of forward 18, backward 20 and backward 21, a pre-hook opens the window and the next transition closes it. The windows are disjoint. Subtracting their retry counts from the whole-update count gives retries elsewhere. Cumulative bytes released measure reserve returned over an interval. The change in current reserve also includes new reservations. The optional migration run reads counters at every role transition to locate retries outside the three windows.

## Prospective test

Both discovery doses reduced forward-18 retries by one per update and left whole-update retries unchanged. B1 and B2 each beat their corresponding early edit on all three cards. The prediction for two previously untimed layouts was recorded in the fork before launch. Ap2 recomputes critic blocks 9 and 10, Bp2 recomputes 26 and 27. The predicted forward-18 counts were 0 and 1, respectively, with 2 whole-step retries for both and for the solver. The strict timing order was solver, Bp2, Ap2. Test it independently on Hiker and Cosmos. [Frozen prediction](https://github.com/Gxj230958/FastVideo/blob/762d9447b329d827288417cafb8160613bd28932/examples/train/activation_memory/minimal_layout_edits/prediction.json)

```bash
cp "$study/prediction.json" "$out/prediction.json"
for sample in hiker cosmos; do
    batch validation "$sample" timing && batch validation "$sample" counters || exit 1
done
python "$study/analyze.py" "$out" figures
```

## Evidence

[Exact layouts](layouts/) · [Raw per-step records](evidence/records/) · [Process means](evidence/processes.csv) · [Same-card differences and ratios](evidence/paired.csv) · [Timed steps and losses](evidence/steps.csv) · [Window and whole-step counters](evidence/windows.csv) · [All-event retry locations](evidence/migration.csv) · [Main figure (PDF)](evidence/paired_layouts.pdf)

The archive contains 81 completed single-device processes and 405 measured updates, including 180 uninstrumented timing updates and 225 counter updates. The raw gzip files store all five measured updates, the applied maps, card, sample, start time and runtime versions. They omit the redundant second copy of the single rank's measurements. Run `python "$study/analyze.py" "$study/evidence/records" figures` to regenerate the tables and figure on CPU.
