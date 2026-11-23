# CPU 侧独立复算记录（2026-11-23）

联网 CPU 机对 `H200_REPORT.md` 的数字做了独立复算，数据来源为本目录下的原始 JSON。
报告的全部结论数字与原始记录一致，可以采信。两处标注瑕疵不影响结论，见文末。

## 复算通过项

- **正常测速**：16 个进程的 10 步均值逐行复算，与报告 L1/L2 两张表的四轮数值逐一相符（含 A−B、A/B 列，差异 ≤0.01 ms，来自中间值舍入）。L1 四轮全部 A 快于 B，L2 方向不一致，与报告一致。
- **峰值**：L1-A 53.514/66.141 GiB，L1-B 53.413/59.143 GiB，L2-A 53.514/66.160 GiB，L2-B 53.414/59.160 GiB（allocated/reserved，四轮取最大），与报告一致。
- **正常诊断**：4 个 `--allocator-events` 进程各 5 步，整步 `num_alloc_retries` 与 `num_ooms` 增量全为 0。
- **压力测速**：cap65 三组配对均值逐行复算相符（A 1298.82 vs B 1077.49 ms）。A 每步恰好 2 次重试、5 步共 10 次、0 OOM；B 全为 0。`--map-retries` 两步均定位于 `critic.forward.27`（1 次）与 `critic.backward.6`（1 次）。
- **压力 trace**：同步本身 0.0416–0.0469 ms；`critic.backward.20` 窗口 A 8.30/8.40 ms、B 8.04/8.14 ms；A 的 `critic.forward.27` 窗口 30.57/30.81 ms（self CPU 前二 cudaFree 15.56/16.07 + cudaMalloc 8.90/9.47 ms），`critic.backward.6` 窗口 71.67/72.89 ms（cudaFree 62.82/63.48 ms）；B 同窗口 4.59–4.65 / 7.94–8.16 ms 且无 cudaFree，与报告一致。
- **正常 trace**：同步本身 0.042–0.057 ms；同步后窗口 A 8.11/30.63、B 7.98/8.01，无屏障组 A 8.22/8.21、B 8.11/8.00；30.63 ms 窗口的最耗时 CPU 操作确为 `aten::_chunk_cat`（4.04 ms），其余窗口为 `FSDP::post_backward_reduce (blocks.20)`（0.76–0.79 ms）；trace 步时长 18.1–22.1 s。
- **数值等价性**：`raw/check/one_update.json` 中 `exact: true`，两种调度的三项损失与两项梯度范数逐值相等；梯度、参数、exp_avg、exp_avg_sq 各 1650 张量 / 2,837,993,600 元素 unequal=0、max_abs=0.0，step 计数 1650 全等。
- **对照要素**：全部 22 个含布局的进程 `manual_policies` 均为学生 24 save + 评价者 30 save；L1 各进程布局与 `stall_study/layouts/solver.json` 完全一致，L2 与 `layouts/L2_h200_local.json` 完全一致；A 进程 `role_schedule=DMD2Method`、B 进程 `SequentialDMD2Method`；seed=2026、warmup=5；测速 10 步、诊断 5 步、trace 2 步，均与报告口径一致。`raw/env.json` 的 GPU/驱动/PyTorch/分配器与报告环境表一致。
- **打包**：`h200_dmd2_results.tar.gz`（共享盘 `h200_stall_results/`）含 86 个文件，与该处磁盘证据集（排除 `inputs/`、`code_d7239500/` 与压缩包自身）一一对应；本目录 86 个文件与共享盘原件逐字节一致。

## 瑕疵（不改结论，原报告保持原样）

1. 报告 L1「均值」行的 A/B 写作 0.9607；按四轮比值取平均为 0.9606，按四轮均值相除为 0.9602。第 4 位小数的舍入差异。
2. 报告第 4 节「各归还 11.4/14.7 GiB 缓存」的单位应为 GB（10⁹ 字节）：原始字节 11,408,506,880 B = 10.62 GiB = 11.41 GB；14,747,172,864 / 14,774,435,840 B = 13.73–13.76 GiB = 14.75–14.77 GB。报告其余 GiB 换算（峰值、R、上限）核对无误。
3. 报告第 2 节「逐步用时见 csv/normal.csv」一句中，`csv/normal.csv` 实际存的是每进程的 10 步均值（16 行，2 位小数，与 JSON 复算一致）；逐步数据在 `raw/normal/*.json`。
