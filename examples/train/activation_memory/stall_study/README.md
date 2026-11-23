# Reproducing the Wan DMD2 allocation-stall experiment

[Project report](../PROJECT_REPORT.md) · [Process means](evidence/processes.csv) · [Traced intervals](evidence/events.csv) · [Allocator counter deltas](evidence/allocator.csv) · [Per-step measurements](evidence/records/)

## Inputs

Use Wan2.1-T2V-1.3B-Diffusers, batch one, 17 frames at 256 × 448. The two inputs use real video latents, a positive text embedding and an empty-prompt embedding. The training parquet repeats each example eight times. Hiker uses the first five causal latent frames of the 49-frame encoding. Cosmos uses 17 evenly resampled frames from a nine-frame clip. [Wan checkpoint](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) · [Hiker clip](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hiker.mp4) · [Cosmos clip](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cosmos/cosmos-video2world-input-vid.mp4)

The measurements used CANN 8.5.1, PyTorch 2.9.0, torch-npu 2.9.0.post1, FP32 master parameters, BF16 autocast and Torch SDPA. Activate the FastVideo environment and set `WAN_MODEL` to the checkpoint directory. From the checkout root, put the two source clips in `inputs/` as `hiker.mp4` and `cosmos.mp4`. Run on an available device. The sample encoder uses [the Wan VAE and UMT5 encoder](../README.md#prepare-a-video-sample).

```bash
export PYTHONPATH="$PWD"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES=0
example=examples/train/activation_memory
data=inputs
for sample in hiker cosmos; do
    frames=17
    prompt='A car drives along a winding road through snowy mountains.'
    if [ "$sample" = hiker ]; then
        frames=49
        prompt='A hiker stands on a rocky mountain summit at sunset.'
    fi
    output="$data/${sample}_17_256_448.pt"
    python "$example/prepare_sample.py" --model "$WAN_MODEL" --video "$data/$sample.mp4" \
        --frames "$frames" --prompt "$prompt" --output "$output"
    python "$example/prepare_sample.py" --model "$WAN_MODEL" --video "$data/$sample.mp4" \
        --text --prompt "$prompt" --output "$output"
    if [ "$sample" = hiker ]; then
        python - "$output" <<'PY'
import sys
import torch
p = sys.argv[1]
s = torch.load(p, map_location='cpu', weights_only=True)
s['latents'] = s['latents'][:, :, :5].clone()
s['frame_indices'] = s['frame_indices'][:17]
torch.save(s, p)
PY
    fi
    python "$example/prepare_parquet.py" --model "$WAN_MODEL" --sample "$output" \
        --output "$data/${sample}_parquet/train.parquet"
done
```

## Runs and aggregation

The files in [layouts](layouts/) fix all 60 role/block choices. The original solver's 24+30 map is nonuniform. The solver layout predicts 53.891 GiB allocated and each of the other four predicts 53.142 GiB. The driver recomputes these predictions from [the Ascend calibration](ascend_profile.json.gz) before launch. Each layout and intervention gets a fresh Python process. A completed process can be reused by rerunning the same command with the same output directory and inputs.

For a complete single-card reproduction, use the following commands. A block is a matched repeat of all conditions on the same physical card. Run uninstrumented cases separately from diagnostic processes. The published experiment used up to three cards in parallel per measurement protocol, with one device per process. Each worker had a disjoint set of 12 logical CPUs and eight PyTorch threads. Primary blocks 0–2 ran on cards 0–2. Counter-only blocks 0–2 ran on cards 3–5, partly concurrently with tracing on cards 0–2. Baselines were collected in a separate phase. The final Hiker serial check used card 0 with no other study worker running. Parallel copies require a distinct `MASTER_PORT` and separate visible device per worker. An optional `--launcher` accepts a shell script followed by the training program and arguments.

```bash
study=examples/train/activation_memory/stall_study
common=(--model "$WAN_MODEL" --data inputs --output runs)
for block in 0 1 2; do
    python "$study/run.py" "${common[@]}" --block "$block" --sample hiker --phase baseline
    python "$study/run.py" "${common[@]}" --block "$block" --sample hiker --phase trace
    python "$study/run.py" "${common[@]}" --block "$block" --sample hiker --phase allocator
    python "$study/run.py" "${common[@]}" --block "$block" --sample cosmos --phase confirm-baseline
    python "$study/run.py" "${common[@]}" --block "$block" --sample cosmos --phase allocator
    python "$study/run.py" "${common[@]}" --block "$block" --sample cosmos --phase confirm-trace \
        --confirm-event critic.backward.20
done
python "$study/run.py" "${common[@]}" --block 0 --sample hiker --phase serial
python "$study/analyze.py" runs figures --archive packed_runs
```

To regenerate both figures and the tables on CPU directly from the published measurements, run `python "$study/analyze.py" "$study/evidence/records" figures` with Matplotlib installed. The gzip files retain all recorded steps, block policies, software versions, event timings, CPU operator self times and allocator snapshots. They omit the redundant second copy of the single rank's steps. In the filenames, `off` denotes a complete uninstrumented update and `none` denotes tracing without a barrier. In the trace table, `barrier_ms` is the host wait immediately before the named target. For a synchronization at another event, look up that event's row. Full-step wall time, host operator self time and timing between recorded stream markers measure different quantities.

The [earlier evaluation](prior/) includes the 56-layout budget screen, the 18 timed candidates, the held-out validation, the original paired trace summary and two held-out time-model comparisons. [H200 profiles and results](../../../../meas_h200/)
