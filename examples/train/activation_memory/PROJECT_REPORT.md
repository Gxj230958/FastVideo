# Activation-memory budgeting for FastVideo Wan training on Ascend 910B2C and NVIDIA H200

A training layout can meet its memory budget and still be twice as slow as another layout with the same peak allocation. In 17-frame Wan2.1-1.3B DMD2 training on Ascend 910B2C, timing a shortlist of memory-feasible layouts reduced the mean update from **2757.26 to 1300.42 ms**. Both layouts peaked at **53.142 GiB** under a 54 GiB budget. On NVIDIA H200, the same event-based planner, calibrated on CUDA, met all three tested allocated-memory budgets. The H200 measurements also show why a layout selected on one device merits a fresh timing comparison on the other.

## What the planner measures

FastVideo chooses a policy for each transformer block before FSDP2 wraps the model. A block can retain activations, recompute them during backward, or offload them to host memory. The profiler measures fixed device storage and the additional storage each choice requires over the training step. At each event, the planner constrains their sum to the byte budget. A mixed-integer solver then minimizes the sum of the selected block costs. Modeling the sequence matters in DMD2, where trainable student and critic graphs overlap with scoring by a frozen teacher. The role profile covers 127 events through clipping and both optimizer updates. [Planner and profiling guide](https://github.com/Gxj230958/FastVideo/tree/125aae85a57a083979ac357e49fad3dd21705054/examples/train/activation_memory)

The experiments use batch size one per device, FP32 master parameters, BF16 computation and 256 × 448 video inputs. A measured update includes forward, backward, optimizer updates and device synchronization. The memory metric is the largest allocated-byte count during a timed update, expressed in GiB (2³⁰ bytes). With this definition, saving 25 of 30 blocks on Ascend allowed 17-frame single-role training to run in 471.82 ms at 23.498 GiB. Full recomputation took 554.91 ms and saving every block used 24.885 GiB. Earlier integration tests also exercised gradients, AdamW state restoration, two-rank execution and sampling from an updated checkpoint. [Native training results](https://github.com/Gxj230958/FastVideo/blob/125aae85a57a083979ac357e49fad3dd21705054/docs/training/activation_memory.md)

## Why the Ascend DMD2 time model missed

The fitted additive model predicted 1192.46 ms for a solver layout retaining 24 student and 30 critic blocks. Its measured update took 2757.26 ms. A paired trace against the tuned manual reference localized 97.7% of the device-stream time difference to two intervals. The interval starting at critic forward block 18 averaged 764.49 ms versus 4.56 ms under the manual layout. At critic backward block 20 the corresponding means were 706.54 and 8.90 ms. In both traced backward steps, the host spent about 700 ms in `empty_tensor`. These intervals include adjacent FSDP transitions. They locate a stall on the allocation path, while its allocator-internal cause remains unresolved.

Two low-cost corrections were tested on the same six held-out Cosmos layouts. A bilinear student–critic interaction had a maximum absolute relative error of 73.5%. A threshold on the total number of saved blocks reduced the median absolute relative error to 1.51%, but its maximum remained 56.1%. For 15 student and 30 critic saves, the threshold model predicted 1297.22 ms against 2953.00 ms measured. A good typical prediction still left enough error to select an expensive configuration.

The selector enumerated 56 uniformly spaced student/critic layouts, filtered all 56 against the event-level budget, and benchmarked 18 of the 54 feasible layouts on Cosmos. Each candidate received two warmup updates and five timed updates. The winning 24+16 layout averaged 1350.12 ms on Cosmos. After freezing the choice, three fresh processes evaluated it on hiker with five warmups and five timed updates each.

| Ascend DMD2 layout, 17 frames | Mean update (ms) | Peak allocated (GiB) |
|---|---:|---:|
| Full recomputation | 1497.69 | 53.142 |
| Tuned manual periodic layout | 1318.19 | 53.142 |
| Additive solver, 24+30 | 2757.26 | 53.142 |
| Measured shortlist, uniform 24+16 | **1300.42** | **53.142** |

The three shortlist evaluation means were 1296.25, 1311.88 and 1293.14 ms, all within the 1450 ms target. The selected layout recovered the performance of the manual reference. Its 2.12× speedup is relative to the earlier additive solver.

## Budget feasibility on H200

H200 profiles were collected for each workload and input shape using PyTorch 2.9.1+cu128. The table gives the predicted peak of each solver plan and the observed maxima across five timed updates. All three measured allocations fell within budget. The narrowest observed margin was 0.334 GiB for 33-frame single-role training. [Profiles, plans and measurements](https://github.com/Gxj230958/FastVideo/tree/125aae85a57a083979ac357e49fad3dd21705054/meas_h200)

| H200 workload | Budget (GiB) | Blocks saved | Predicted allocated (GiB) | Measured allocated (GiB) | Measured reserved (GiB) | Mean update (ms) |
|---|---:|---|---:|---:|---:|---:|
| Single role, 17 frames | 24 | 25 | 23.933 | 23.544 | 29.855 | 555.14 |
| Single role, 33 frames | 28 | 23 | 27.901 | 27.666 | 34.943 | 570.51 |
| DMD2, 17 frames | 54 | 24 student + 30 critic | 53.902 | 53.514 | 66.160 | 1222.07 |

At 33 frames the earlier Ascend single-role plan saved 22 blocks and averaged 579.67 ms at 27.479 GiB. Both devices therefore met the 28 GiB target with a shape-specific plan, using different save counts. The 17-frame single-role plans selected the same count of 25. Counts alone do not identify which blocks the solver retained.

## Transferring a DMD2 layout

The fixed 24+16 layout provides a direct check on whether the Ascend choice can execute under the same H200 budget. It reached 53.544 GiB and averaged 1244.42 ms on H200, compared with 53.142 GiB and 1300.42 ms in the Ascend validation. Comparing layouts within H200 gives a different picture. The solver had a 22.35 ms lower observed mean than the transferred layout and the largest reserved peak. The uniform 15+30 layout had the lowest observed mean of the three. On Ascend, the same 15+30 uniform layout had been slow enough to exclude during tuning.

| H200 DMD2 layout, 17 frames | Warmups | Timed updates | Mean update (ms) | Peak allocated (GiB) | Peak reserved (GiB) |
|---|---:|---:|---:|---:|---:|
| Transferred uniform 24+16 | 2 | 5 | 1244.42 | 53.544 | 62.057 |
| Device-calibrated solver 24+30 | 5 | 5 | 1222.07 | 53.514 | 66.160 |
| Manual uniform 15+30 | 2 | 5 | 1188.19 | 53.536 | 63.426 |

## Interpretation and limits

The memory constraints transferred through device-local calibration. Layout timing was more sensitive to the runtime and to which blocks were live together. For this workload, event-level constraints provided a practical way to reduce the search space, and timing complete updates exposed costs that the block model missed.

- **Memory scope.** All budgets apply after warmup. H200 reserved peaks exceeded the nominal budgets in all five runs. Initialization, warmup and driver memory require additional capacity when sizing a physical device.
- **Timing scope.** Each H200 configuration contributed one process with five sequential updates. Different warmups place the solver and fixed layouts at different training iterations. Ascend used torch-npu 2.9 with expandable segments, whereas H200 used the default CUDA allocator. The 17-frame H200 sample was robot and the Ascend validation sample was hiker. The means describe these runs and cannot isolate a hardware-only speedup or establish a reliably fastest H200 layout.
- **Data and calibration.** The 33-frame H200 sample repeats nine Cosmos source frames. DMD2 parquet rows reuse one source video. These runs exercise training and memory behavior, with no measure of downstream video quality. The DMD2 fixed-memory residual comes from its own 127-event all-recompute observation. Two 62-event references and the incremental calibration metadata in the H200 DMD2 profile were inherited from its single-role template.
