# Removing a 700 ms allocation stall in Wan DMD2

On an Ascend 910B2C, running the student's backward pass before the critic builds its training graph reduced complete updates from **2.78 s to 1.25 s**. The original solver layout still saved 24 student and 30 critic blocks. Both schedules peaked at **53.142 GiB allocated**, within the 54 GiB budget. The change removed both allocator retries on every observed update across two video inputs.

## Same-card measurements

Each time is the mean of three single-rank processes on separate cards. Each process ran five warmups followed by five timed updates. Forward, backward, gradient clipping, AdamW and device synchronization are included. The third column remeasures the earlier shortlist winner using its own 24+16 layout and the original schedule.

| Input | Original 24+30 (ms) | Shortlist 24+16 (ms) | Early student backward 24+30 (ms) | Speedup vs. original 24+30 |
|---|---:|---:|---:|---:|
| Hiker | 2775.93 | 1296.23 | 1250.40 | 2.22× |
| Cosmos | 2787.81 | 1298.33 | 1242.41 | 2.24× |

Against the remeasured shortlist, early backward shortened mean step time by 3.5% on Hiker and 4.3% on Cosmos. The same-card reductions for the 24+30 layout were 1.52–1.57 s per update. The maximum reserved memory fell from 60.369 to 56.910 GiB in the uninstrumented runs. Allocated memory counts storage occupied by live tensors. Reserved memory also includes cached space held by the allocator. [Process means](processes.csv) · [Per-step measurements](records/)

## What changed inside the step

The student first predicts a denoised latent. Frozen teacher and detached critic scores define its loss. The critic subsequently trains on a fresh student sample obtained with gradients disabled. DMD2 normally builds both loss graphs, then backpropagates the student loss and the critic loss. The new method finishes student backward immediately after its loss is available. The critic uses the same pre-update student weights. Both optimizers still step after both gradients are ready.

In the diagnostic runs the critic entered forward block 18 with **47.836 GiB** allocated, down from **49.448 GiB**. The original schedule released about **10.59 GiB** of allocator reserve there on every update. The new schedule released **zero** there. At backward block 20 the two schedules both entered with **50.333 GiB allocated**, yet only the original schedule retried. Live tensor bytes at this later instant alone cannot explain the pause. The order in which memory becomes reusable also matters.

| Input and schedule | Critic forward 18 host (ms) | Critic backward 20 host (ms) | Retries at forward 18 / backward 20 | Whole-update retries |
|---|---:|---:|---:|---:|
| Hiker, original | 759.82 | 666.93 | 15 / 14 | 30 |
| Hiker, early backward | 3.33 | 5.03 | 0 / 0 | 0 |
| Cosmos, original | 751.56 | 705.98 | 15 / 15 | 30 |
| Cosmos, early backward | 3.04 | 4.97 | 0 / 0 | 0 |

Each row covers 15 updates across three fresh counter processes. Whole-step counters extend through the final synchronization. One original Hiker retry occurred outside the three monitored critic windows (forward 18, backward 20 and backward 21). Eliminating the two retries per update distinguishes this change from a small layout edit that moves the cost elsewhere. [Earlier layout intervention](../minimal_layout_edits/)

In a separate Hiker trace, a device synchronization immediately before critic backward block 20 took **0.42 ms** on the original schedule. The subsequent host interval still took **718.05 ms**, including **711.80 ms** of CPU self time attributed to `empty_tensor`. With early student backward the barrier took **3.51 ms**, the interval **6.37 ms**, and `empty_tensor` **0.44 ms**. Each result averages six traced updates in three processes. Draining previously queued work at that boundary did little to relieve the original pause. The reserve releases and retry counts in the independent counter runs localize the delay to allocation under memory pressure.

## Does it apply the same update?

A separate deterministic Hiker replay used the same batch, initial weights, and Python, NumPy, PyTorch and device generator states. All three loss values and both pre-clipping gradient norms matched. Each of the 1,650 trainable tensors matched element for element for gradients, updated parameters and both AdamW moments, covering 2,837,993,600 scalar values per group. All 1,650 optimizer step counters matched. A small regression test also exercises two accumulated microbatches and a critic-only update. [Full-update comparison](one_update.json) · [Regression test](../../../../fastvideo/tests/train/methods/test_dmd2_role_schedule.py)

## Reproduction

Use the Wan2.1-T2V-1.3B-Diffusers checkpoint and the Hiker and Cosmos inputs prepared in the earlier study. Runs used 17 frames at 256 × 448, batch one, FP32 master weights, BF16 autocast, Torch SDPA, CANN 8.5.1, PyTorch 2.9.0 and torch-npu 2.9.0.post1. On this host each single-card worker had eight PyTorch threads and twelve dedicated logical CPUs. [Input preparation](../stall_study/README.md#inputs)

Run each invocation in a fresh process with one visible device. The published 24+30 pairs ran on physical cards 0–2. Cards 0 and 2 started with the original schedule on Hiker, card 1 started with early backward, and the orders were reversed on Cosmos. Timings, allocator counters and synchronization traces used separate processes. The shortlist reference ran in a subsequent timing batch.

```bash
export PYTHONPATH="$PWD"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES=0
example=examples/train/activation_memory
sample=hiker
input=(--model "$WAN_MODEL" --training-sample "inputs/${sample}_17_256_448.pt"
       --data "inputs/${sample}_parquet" --warmup 5)
layout=(--layout-file "$example/stall_study/layouts/solver.json")
for schedule in original sequential; do
    flag=()
    if [ "$schedule" = sequential ]; then flag=(--sequential-backward); fi
    python "$example/profile_dmd2.py" "${input[@]}" "${layout[@]}" "${flag[@]}" \
        --steps 5 --output "timing_$schedule.json"
    python "$example/profile_dmd2.py" "${input[@]}" "${layout[@]}" "${flag[@]}" \
        --steps 5 --allocator-events --output "counters_$schedule.json"
    python "$example/profile_dmd2.py" "${input[@]}" "${layout[@]}" "${flag[@]}" \
        --steps 2 --trace-events --sync-event critic.backward.20 --output "trace_$schedule.json"
done
python "$example/profile_dmd2.py" "${input[@]}" \
    --layout-file "$example/minimal_layout_edits/layouts/selected.json" --steps 5 --output shortlist.json
HCCL_DETERMINISTIC=true python "$example/profile_dmd2.py" "${input[@]}" "${layout[@]}" \
    --check-role-schedules --output one_update.json
python "$example/role_schedule/summarize.py" "$example/role_schedule/records" > processes.csv
```

For the modular trainer, choose `fastvideo.train.methods.distribution_matching.dmd2.SequentialDMD2Method` as the method target and supply the block maps under `models.student.activation_memory_plan` and `models.critic.activation_memory_plan`. The existing automatic budget profile was calibrated with the original event order. The new schedule supports explicit layouts. The deterministic replay stores full reference tensors in host memory and compares them outside the timed experiment.

## Limits

The speedup is relative to the original solver layout on one Ascend device at this shape. The two clips share the same dimensions. The one-update equality check covers this Wan recipe. Long-run convergence, other shapes, multi-rank execution and H200 speed under the new schedule remain unmeasured. The window measurements include adjacent FSDP transitions. Allocator counters identify reserve release and retry but do not time lower-level driver operations individually.
