# H200 上的 FastVideo DMD2 分配重试与训练调度实验报告

**摘要（三句话回答三个研究问题）**

1. **正常 H200 上没有分配重试，也没有主机停顿**：默认 native 分配器、整卡可用（139.81 GiB）时，L1/L2 × 原调度(A)/提前反向(B) 各 5 个独立诊断步的整步 `num_alloc_retries` 与 `num_ooms` 增量全部为 0，`critic.forward.18`、`critic.backward.20`、`critic.backward.21` 三个昇腾热点窗口的主机耗时仅 3–6 ms（昇腾参照为 718.05 ms）。
2. **正常条件下提前反向没有缩短完整步，反而在 L1 上一致变慢**：L1 四轮配对 B 全部更慢（A/B = 0.938–0.985，均值 1095.05 vs 1140.40 ms）；L2 方向不一致（A/B = 0.955–1.038，均值 1088.09 vs 1085.50 ms），属波动；提前反向把峰值 reserved 一致降低约 7.0 GiB（66.14→59.14 GiB），峰值 allocated 仅降 0.10 GiB（53.51→53.41 GiB）。
3. **人工分配器上限 65 GiB（=⌈R⌉−2 GiB，R = 正常 L1-A reserved 峰值 66.14 GiB）复现了重试并反转了快慢方向**：A 每个记录步恰好 2 次重试（5 步共 10 次，0 OOM），B 为 0；三组配对 A/B = 1.19–1.22（1298.82 vs 1077.49 ms 均值）；重试定位于 `critic.forward.27` 与 `critic.backward.6`（各 1 次/步，分别归还 11.4/14.7 GiB 缓存 reserved），窗口 self CPU 以 `cudaFree` 为主（最高 63 ms/窗口），而非昇腾上的 `critic.backward.20` 窗口。

## 1. 环境、代码、输入与可复现命令

| 项 | 值 |
|---|---|
| GPU | NVIDIA H200（UUID GPU-61b98fcb-d249-8bd0-6716-b85c8607c29a），可见总量 143771 MiB = 139.81 GiB，启动时空闲（占用 1 MiB），MIG Disabled |
| 驱动 / CUDA | 570.124.06 / 12.8 |
| Python / PyTorch | 3.10.12 / 2.9.1+cu128，分配器后端 native（默认缓存配置，`PYTORCH_CUDA_ALLOC_CONF` 未设置，正常条件无人工上限） |
| 共用 venv | `/inspire/hdd/global_user/s26045/fastvideo/.venv-h200` |
| 克隆 | `https://github.com/Gxj230958/FastVideo.git`，分支 `research/allocator-stall-optimization`，分支头 `87fcc78bf7b3b752340b83ffb1c0e9d75a02fef5`（离线经 `git rev-parse` 核对；fetch/merge 按交接说明跳过） |
| 被测实现 | `code_rev = d723950099a34a58dcf8ba3baa96a93edf11a9bb`（"[research] Release DMD2 student activations before critic training"，已核实为分支头祖先），实验工作树 `h200_stall_results/code_d7239500`（detached HEAD，已核对） |
| 模型 | 本机 `Wan2.1-T2V-1.3B-Diffusers`，17 帧、256×448、batch 1、单 rank、FP32 主参数、BF16 计算、Torch SDPA、种子 2026 |
| 样本 | Hiker（`hiker.mp4` 49 帧编码取前 5 个 causal latent 帧）：`SAMPLE_PT=inputs/hiker_17_256_448.pt`，`PARQUET_DIR=inputs/hiker_parquet`，按 `stall_study/README.md` 用 CUDA 现场制备（非兜底 robot 样本） |
| 布局 L1（主比较） | `code_d7239500/examples/train/activation_memory/stall_study/layouts/solver.json`（昇腾停顿布局：学生 24 save + 6 recompute，评价者 30 save） |
| 布局 L2（本地对照） | `layouts/L2_h200_local.json`，取自 `meas_h200/dmd_budget54.plan.json` 顶层 `policies`（同样 24+30），已剔除元数据键 |
| 布局核对 | 每个输出的 `manual_policies` 均确认 student/critic 逐块映射，save 计数 24/30 |

实际重复次数：正式测速 16 个进程（L1/L2 × 4 轮配对，轮内顺序 AB、BA、BA、AB）；独立诊断 8 个进程（预热 5、记录 5）；trace 4 个进程（无屏障 / `--sync-event critic.backward.20` 各一组，预热 5、记录 2）；压力实验 2 计数 + 6 测速（B A、A B、B A）+ 2 trace + 1 map-retries；数值检查 1 个进程。每次调用均为全新 Python 进程。

一次可复现的命令（正式测速 B 进程）：

```bash
source /inspire/hdd/global_user/s26045/fastvideo/.venv-h200/bin/activate
cd /inspire/hdd/global_user/s26045/fastvideo/h200_stall_results/code_d7239500
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0
python examples/train/activation_memory/profile_dmd2.py \
  --model /inspire/hdd/global_user/s26045/fastvideo/weights/Wan2.1-T2V-1.3B-Diffusers \
  --training-sample .../h200_stall_results/inputs/hiker_17_256_448.pt \
  --data .../h200_stall_results/inputs/hiker_parquet \
  --layout-file examples/train/activation_memory/stall_study/layouts/solver.json \
  --warmup 5 --steps 10 --sequential-backward --output OUT/result.json
```

## 2. 正常条件：完整步测速（每行为一个进程的 10 步均值）

**L1（昇腾停顿布局）**

| 配对 | A 原调度 (ms) | B 提前反向 (ms) | A−B (ms) | A/B |
|---|---|---|---|---|
| r1 (AB) | 1066.06 | 1082.77 | −16.71 | 0.9846 |
| r2 (BA) | 1082.60 | 1153.86 | −71.26 | 0.9382 |
| r3 (BA) | 1098.96 | 1162.42 | −63.46 | 0.9454 |
| r4 (AB) | 1132.56 | 1162.54 | −29.98 | 0.9742 |
| 均值 | 1095.05 | 1140.40 | −45.35 | 0.9607 |

**L2（H200 本地对照布局）**

| 配对 | A (ms) | B (ms) | A−B (ms) | A/B |
|---|---|---|---|---|
| r1 (AB) | 1068.87 | 1061.99 | +6.88 | 1.0065 |
| r2 (BA) | 1083.14 | 1068.37 | +14.77 | 1.0138 |
| r3 (BA) | 1095.30 | 1147.14 | −51.84 | 0.9548 |
| r4 (AB) | 1105.06 | 1064.49 | +40.57 | 1.0381 |
| 均值 | 1088.09 | 1085.50 | +2.60 | 1.0024 |

方向一致性：L1 四轮全部 A 快于 B（方向一致，B 慢 1.5%–6.2%）；L2 四轮方向不一致（差值 ≤5%，有正有负），如实为波动。逐步用时见 `csv/normal.csv` 与原始 JSON。

峰值（四轮取最大，逐进程值彼此一致）：

| 布局/调度 | 峰值 allocated | 峰值 reserved |
|---|---|---|
| L1-A | 53.514 GiB | 66.141 GiB |
| L1-B | 53.413 GiB | 59.143 GiB |
| L2-A | 53.514 GiB | 66.160 GiB |
| L2-B | 53.414 GiB | 59.160 GiB |

峰值 allocated 53.5 GiB 低于项目 54 GiB 目标。提前反向在两种布局上都将 reserved 峰值降低约 7.0 GiB，allocated 仅降 0.10 GiB——它释放的是缓存池需求而非张量总量。

## 3. 正常条件：分配器诊断与停顿定位（数据来自独立诊断进程）

`--allocator-events`（预热 5、record 5，每进程独立）：

| 运行 | 记录步数 | 有重试的步数 | 整步重试合计 | 整步 OOM 合计 | f.18 / b.20 / b.21 重试 | 三窗口主机耗时均值 (ms) | 缓存归还 |
|---|---|---|---|---|---|---|---|
| L1-A | 5 | 0 | 0 | 0 | 0 / 0 / 0 | 3.17 / 5.11 / 5.07 | 0 |
| L1-B | 5 | 0 | 0 | 0 | 0 / 0 / 0 | 2.99 / 4.96 / 5.04 | 0 |
| L2-A | 5 | 0 | 0 | 0 | 0 / 0 / 0 | 3.14 / 5.58 / 5.63 | 0 |
| L2-B | 5 | 0 | 0 | 0 | 0 / 0 / 0 | 3.07 / 4.96 / 5.07 | 0 |

整步重试等于三窗口之和（都为零），无需 `--map-retries`。诊断进程时间不用于正式加速比。

停顿定位（L1，trace 进程，预热 5、record 2）：对 `critic.backward.20` 的同步本身 0.042–0.057 ms（A/B 各两步）；同步后窗口主机耗时 A 8.11/30.63 ms、B 7.98/8.01 ms，无屏障对照组 A 8.22/8.21、B 8.11/8.00 ms。同步本身及其后窗口都与昇腾参照（0.42 ms 同步 + 718.05 ms 窗口）无对应的长停顿；A 有一次 30.63 ms 的单步窗口（窗口内最耗时 CPU 操作为 `aten::_chunk_cat` 4.04 ms，非同步等待），其余窗口最耗时 CPU 操作为 `FSDP::post_backward_reduce (blocks.20)`（≤0.8 ms）。trace 步的完整步时长含 profiler 开销（约 18–22 s），不代表正式步时。

**问题 1 结论**：正常 H200 无分配重试，无可重复的主机停顿。

## 4. 人工分配器上限（受控压力，L1、同一输入）

方法：薄启动器 `scripts/capped_launcher.py` 在加载模型前对逻辑设备 0 调 `torch.cuda.set_per_process_memory_fraction(cap/150121545728, 0)`，同一进程内执行原 profiler；启动器打印请求的 cap 字节与实际 fraction。参照 R = 正常 L1-A reserved 峰值 71017955328 B（66.141 GiB），⌈R⌉ = 67 GiB，预设档位 65 GiB 与 61 GiB，从宽到紧扫描。首档 65 GiB 即满足"A 出现重试且 A/B 均可完成"，按预设规则停止扫描，61 GiB 未执行（记为未执行项，原因：扫描规则已命中）。无 OOM 发生于任何阶段。

计数（`--allocator-events`，预热 5、record 5）：A 每个记录步恰好 2 次重试、5 步共 10 次、0 OOM；B 全程 0 重试（B reserved 峰值 59.14 GiB 低于上限，A 峰值 66.14 GiB 高于上限）。重试位置（`--map-retries`，A，2 步一致）：`critic.forward.27` 各 1 次（各归还缓存 11.4 GiB）、`critic.backward.6` 各 1 次（各归还 14.7 GiB）——不在昇腾的三个热点窗口内。

配对测速（预热 5、测 10，顺序 B A / A B / B A）：

| 配对 | A (ms) | B (ms) | A−B (ms) | A/B |
|---|---|---|---|---|
| r1 | 1281.28 | 1076.67 | +204.61 | 1.1900 |
| r2 | 1317.94 | 1076.06 | +241.87 | 1.2248 |
| r3 | 1297.26 | 1079.66 | +217.60 | 1.2015 |
| 均值 | 1298.82 | 1077.49 | +221.33 | 1.2054 |

压力下 A 的 reserved 峰值被钉在 64.99 GiB（上限处），B 仍为 59.14 GiB；allocated 峰值不变（53.52/53.41 GiB）。热点 trace（`--sync-event critic.backward.20`，record 2）：同步本身 0.042–0.047 ms，同步后窗口 A 8.30/8.40 ms、B 8.04/8.14 ms，`critic.backward.20` 窗口即使在上限下也未变长；重试窗口的 self CPU——A 的 `critic.forward.27` 30.6–30.8 ms（`cudaFree` 16.1 ms + `cudaMalloc` 8.9 ms）、`critic.backward.6` 71.7–72.9 ms（`cudaFree` 62.8–63.5 ms），B 同窗口仅 4.6–4.7 / 7.9–8.2 ms 且无 cudaFree——合计约 100 ms/步的分配器主机开销，与 A 在此档位约 220 ms/步的加重方向一致（其余部分散布于缓存池周转）。

**问题 3 结论**：正常条件未复现；仅在人为压缩分配器可用容量后，A 出现可重复重试与约 1.20× 的完整步减速，B 在相同绝对上限下无重试。机制是超限触发缓存归还（cudaFree/cudaMalloc 同步开销），而非昇腾式单窗口长停顿。该结论限定于 65 GiB 人工上限条件。

## 5. 训练更新数值等价性（L1、同一输入、单进程两调度对比）

命令：同第 1 节，换 `--check-role-schedules --output raw/check/one_update.json`，Python 启动前设 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；主机内存 2 TB，充足。结果 `exact = true`，进程退出码 0：

- 两种调度的损失逐位相同（total 1.052255392074585、generator 0.9904878735542297、fake_score 0.061767578125），梯度范数相同（student 1.441845417022705、critic 5.516586780548096）。
- 梯度、更新后参数、AdamW 的 exp_avg 与 exp_avg_sq 各 1650 个张量 / 28.4 亿（2,837,993,600）个元素，加上优化器 step 计数，全部逐元素相等（unequal = 0，max_abs = 0.0）。

数值等价性在 CUDA 上验证通过，无需 A→A 重放区分。

## 6. 与昇腾 910B2C 对照的解释

同一 L1 布局、同一模型与输入规格下，两台设备的图景相反：昇腾原调度虽只花 0.42 ms 同步，却在 `critic.backward.20` 后的主机窗口消耗 718.05 ms，30 个诊断更新累计 60 次整步重试，提前反向消除重试并把完整步从 2.78 s 降到 1.25 s；H200 正常条件下两种调度都零重试、热点窗口 5–8 ms，完整步约 1.1 s，提前反向在 L1 上稳定慢 1.5%–6.2%。H200 的 HBM 容量（139.81 GiB）对 66.1 GiB 的 reserved 峰值留有两倍余量，分配器从不需要归还缓存，昇腾停顿的触发条件（缓存池顶到设备上限）在正常 H200 上根本不存在。因此"提前反向缩短完整步"在 H200 正常条件下不成立，其收益也没有以别的形式出现；L1 上 B 的小幅变慢方向一致但无重试证据，原因未确定（不属于分配停顿）。当把可用容量人工压到 65 GiB、让 A 的 reserved 峰值超过上限时，H200 复现了每步 2 次重试与约 1.20× 的 A 侧减速，而 B 因峰值低于上限 6 GiB 而完全不受影响——提前反向降低 reserved 峰值约 7 GiB 的特性在容量受限时构成实质优势。该机率为容量受限部署提供的结论，不外推到整卡正常使用。

## 附录：文件清单（均以 `h200_stall_results/` 为根的相对路径）

| 内容 | 路径 |
|---|---|
| 本报告 | `H200_REPORT.md` |
| 实验工作树（code_rev，不进压缩包） | `code_d7239500/` |
| Hiker 样本 / parquet | `inputs/hiker_17_256_448.pt`、`inputs/hiker_parquet/train.parquet` |
| L2 布局 | `layouts/L2_h200_local.json` |
| 环境元数据 | `raw/env.json` |
| 正式测速原始 JSON（16 + stdout） | `raw/normal/` |
| 分配器诊断（8） | `raw/diag/` |
| 正常 trace（4） | `raw/trace/` |
| 压力实验（2 计数 + 6 测速 + 2 trace + 1 map-retries） | `raw/pressure/` |
| 数值等价性 | `raw/check/one_update.json` |
| 汇总 CSV / JSON | `csv/normal.csv`、`csv/diag.json`、`csv/trace.json` |
| 启动与汇总脚本 | `scripts/common.sh`、`scripts/run_normal.sh`、`scripts/run_diag.sh`、`scripts/run_trace.sh`、`scripts/run_pressure.sh`、`scripts/capped_launcher.py`、`scripts/run_check.sh`、`scripts/analyze.py` |
| 打包 | `h200_dmd2_results.tar.gz`（含 raw/、csv/、scripts/、layouts/ 与本报告；排除权重、数据集、inputs 与实验工作树） |

未执行/未命中项：正常条件 `--map-retries`（整步重试=窗口重试=0，无需定位）；61 GiB 档位（扫描规则已在 65 GiB 命中）；A→A 确定性重放（`exact=true`，无需区分）。
