# H200 上的 FastVideo DMD2 分配重试与训练调度实验

请在这台 NVIDIA H200 机器上实际执行以下 FastVideo 实验，完成取数、分析和报告。最终交付一份可独立阅读的中文 Markdown 报告 `H200_REPORT.md`。读者只有这份文件也应能判断结论是否成立。

## 开始前：在 H200 上取得研究分支并准备环境

H200 和昇腾主机没有共享目录。代码、布局、历史测量和本提示词从 GitHub 的 `Gxj230958/FastVideo` fork 获取。权重、视频、Python 环境和输出目录均使用 **H200 机器自己的路径**。归档 JSON 里的绝对路径只是当时的运行记录。

**1. 核对 H200 上的 checkout。** 在 H200 的可写工作目录运行下列 Bash 命令。如果已有专用克隆，可先把 `H200_CHECKOUT` 设为它的绝对路径。已有目录如果指向其他 fork、处在其他分支或含有未提交的工作，换一个新的 `H200_CHECKOUT`，再从头运行。下面的分支头用于取得最新研究文件，`code_rev` 固定被测实现。

```bash
set -eu
repo=https://github.com/Gxj230958/FastVideo.git
branch=research/allocator-stall-optimization
code_rev=d723950099a34a58dcf8ba3baa96a93edf11a9bb
checkout=${H200_CHECKOUT:-"$PWD/FastVideo-h200-study"}
if [ ! -e "$checkout" ]; then
    git clone --single-branch --branch "$branch" "$repo" "$checkout"
fi
cd "$checkout"
git remote get-url origin
git status --short --branch
case "$(git remote get-url origin)" in
    "$repo"|https://github.com/Gxj230958/FastVideo|git@github.com:Gxj230958/FastVideo.git) ;;
    *) echo "请选择研究 fork 的克隆目录"; exit 1 ;;
esac
test "$(git branch --show-current)" = "$branch"
test -z "$(git status --porcelain --untracked-files=no)"
git fetch origin "refs/heads/$branch:refs/remotes/origin/$branch"
git merge --ff-only "origin/$branch"
test "$(git rev-parse HEAD)" = "$(git rev-parse "origin/$branch")"
git merge-base --is-ancestor "$code_rev" HEAD
git rev-parse HEAD
test -s examples/train/activation_memory/role_schedule/H200_AGENT_PROMPT.md
```

把打印出的分支名、远端地址、分支头 SHA 写入报告。使用分支上的最新版提示词；为确保跑到指定代码，在同一份克隆旁创建固定版本的实验工作树。新工作树显示 detached HEAD 是预期结果，前面的分支核对针对下载研究文件的克隆。

```bash
run_dir=$(mktemp -d "${TMPDIR:-/tmp}/fastvideo-h200-code.XXXXXX")
git worktree add --detach "$run_dir" "$code_rev"
cd "$run_dir"
test "$(git rev-parse HEAD)" = "$code_rev"
export EXAMPLE="$PWD/examples/train/activation_memory"
test -s "$EXAMPLE/profile_dmd2.py"
test -s "$EXAMPLE/stall_study/layouts/solver.json"
test -s meas_h200/dmd_budget54.plan.json
```

**2. 准备 CUDA 环境。** 用 `nvidia-smi` 找到已授权且空闲的 H200，将 `CUDA_VISIBLE_DEVICES` 设为该卡的编号或 UUID。优先激活这台机器上此前成功运行 Wan 的环境（历史版本为 PyTorch 2.9.1+cu128），在实验工作树根目录执行 `export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"`。若需要新建环境，按该 checkout 的 `docs/getting_started/installation/gpu.md` 在独立虚拟环境安装，并在报告里记录实际 PyTorch/CUDA 版本。所有配对进程复用同一环境。在运行任何测量前执行：

```bash
python - <<'PY'
from pathlib import Path
import fastvideo
import torch
assert Path(fastvideo.__file__).resolve().parent == Path("fastvideo").resolve()
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
gpu = torch.cuda.get_device_properties(0)
print("FastVideo", fastvideo.__file__, "PyTorch", torch.__version__, "CUDA", torch.version.cuda)
print("GPU", gpu.name, "GiB", gpu.total_memory / 2**30, "allocator", torch.cuda.get_allocator_backend())
assert "H200" in gpu.name and torch.cuda.get_allocator_backend() == "native"
PY
python "$EXAMPLE/profile_dmd2.py" --help
```

**3. 准备 H200 本地输入。** 将 `WAN_MODEL` 指向本机完整的 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` checkpoint，`SAMPLE_PT` 和 `PARQUET_DIR` 指向同一视频对应的训练样本和 parquet 目录。先查 H200 本机已有文件。缺少权重时从公开模型仓库下载；缺少 Hiker 时，按 `$EXAMPLE/stall_study/README.md` 的 Inputs 小节获取视频、准备样本和 parquet，其中设备选用 CUDA。用 H200 的持久存储创建结果目录，保存 `H200_REPORT.md`、原始 JSON、CSV 和压缩包，并为每个进程设置单独的 `OUT`。把这些本机路径记入报告附录。

## 一、研究问题与已知参照

在单卡昇腾 910B2C 上，Wan DMD2 原调度会同时保留学生和评价者的训练图。把学生反向提前到评价者建立训练图之前，保持逐块布局、损失和优化器更新顺序相同，曾得到以下结果：

- 原调度在评价者反向块 20 前同步设备只花 0.42 ms，随后的主机窗口仍耗时 718.05 ms。提前反向后，该窗口降到 6.37 ms。
- 两个视频合计，原调度 30 个诊断更新有 60 次整步分配重试，提前反向的 30 个更新为零。
- 在同一 24+30 布局上，完整步从约 2.78 s 降到 1.25 s。

请用 H200 的新数据独立回答三个问题：

1. 默认 CUDA 原生分配器、整卡正常可用时，有没有发生分配重试？有的话，是否伴随可重复的主机停顿，发生在哪里？
2. 固定精确布局和输入，提前做学生反向对 H200 完整步时、峰值 allocated 和峰值 reserved 有什么影响？
3. 如果正常条件没有复现，在人为收紧分配器可用容量后，是否出现重试或停顿？两种条件分别下结论。720 ms 是参照值，不是判定门槛。

## 二、固定代码、输入和对照

使用已授权的空闲 H200，单卡单进程。记录 GPU 型号、可见总容量和启动时空闲容量、MIG 状态、驱动、PyTorch、CUDA、分配器后端及配置。正常条件使用 native 后端、默认缓存配置、无人工容量上限。

模型用 Wan2.1-T2V-1.3B-Diffusers，17 帧、256×448、batch 1、单 rank、FP32 主参数、BF16 计算、Torch SDPA、种子 2026。优先复用 Hiker 的 .pt 和与其匹配的 parquet；缺少时按 `examples/train/activation_memory/stall_study/README.md` 的 Inputs 小节准备。若只能取得历史 H200 的 robot 样本，可以用它完成 H200 内部的配对比较，在报告中注明样本差异。

测试两份精确布局，两者都是学生保存 24 块、评价者保存 30 块，但学生有 12 个位置不同。

- **L1，主要比较。** 直接使用 `examples/train/activation_memory/stall_study/layouts/solver.json`，这是发生过停顿的昇腾布局。
- **L2，设备本地对照。** 从 `meas_h200/dmd_budget54.plan.json` 取出顶层 `policies` 对象，另存为布局 JSON。这是历史 H200 求解器选出的布局。

对每份布局分别运行 A 原调度和 B 提前学生反向。A 使用脚本默认值，B 只增加 `--sequential-backward`。两者都用 `--layout-file`，通过输出中的 `manual_policies` 核对实际加载的逐块映射。历史 H200 的 1222 ms 只用作背景，新实验的 A 才是 B 的对照。

## 三、先测正常使用的 H200

主入口为 `examples/train/activation_memory/profile_dmd2.py`。基本命令如下，OUT 的父目录须先建好。每次调用启动全新 Python 进程。

```bash
python "$EXAMPLE/profile_dmd2.py" \
  --model "$WAN_MODEL" --training-sample "$SAMPLE_PT" --data "$PARQUET_DIR" \
  --layout-file "$LAYOUT" --warmup 5 --steps 10 --output "$OUT"
```

1. **正式测速。** 对 L1、L2 各做四轮同卡配对，各轮的先后顺序为 AB、BA、BA、AB。在同一物理卡上完成一对，单卡串行即可。每个进程预热 5 步，测 10 个完整更新，不开 profiler、事件钩子或额外同步。保存逐步时间、loss、allocated 和 reserved 峰值。以每个进程的 10 步均值为一个观测，列出四个配对的 A、B、A−B 和 A/B。若差异只有几个百分点且方向不一致，如实报告波动。将实际 allocated 峰值与项目的 54 GiB 目标比较。
2. **独立诊断。** 对 L1、L2 的 A/B 各开两个新进程，用 `--allocator-events`，分别预热 5 步、记录 5 步。统计每步从开始到最终同步的 `num_alloc_retries` 和 `num_ooms` 增量，以及 `critic.forward.18`、`critic.backward.20`、`critic.backward.21` 各窗口的重试数、主机时间、累计归还的 reserved 字节。记录重试出现于多少步。若整步重试多于三个窗口之和，用 `--map-retries` 定位其余重试。诊断进程的时间不用于计算正式加速比。
3. **定位停顿。** L1 即使零重试，也对 A/B 各做一组无屏障 trace 和一组 `--trace-events --sync-event critic.backward.20` 的 trace，每组预热 5 步、记录 2 步。无屏障组只加 `--trace-events`。若前一步发现热点在别处，再对该热点做同样对照，优先选每步只访问一次的事件。报告同步本身耗时、同步后的窗口耗时、窗口内最耗时的实际 CPU 操作名。CUDA 的操作名可能和昇腾不同。若能观察到重试，但窗口耗时与正常窗口接近，请直接写“观察到重试，未观察到相应长停顿”。若出现明显长停顿却无重试，同样分别报告。

## 四、受控显存压力实验

若正常 H200 未出现可重复的重试停顿，或要确认显存余量的作用，在 L1、同一输入下追加压力实验。优先完成上面的默认条件。

用很薄的本地启动器，在加载模型之前、同一 Python 进程内对逻辑设备 0 调用 `torch.cuda.set_per_process_memory_fraction(cap_bytes / total_visible_bytes, 0)`，再执行原 profiler。保存启动器和实际生效的容量。54 GiB 是训练张量的 allocated 目标，分配器上限还要容纳缓存，不能直接把上限设为 54 GiB 来代表原实验。

以正常 L1-A 测得的 reserved 峰值 R 为参照，预先固定两个上限：R 向上取整到 GiB 后分别减 2 GiB、6 GiB。两种调度在同一档使用相同的绝对上限。每档先各跑一次 A/B counter。按从宽到紧的顺序，取首个 A 出现重试且 A/B 均可完成的档位，用新进程重复三组同卡配对测速，并复测计数和热点 trace。若两档都没有重试，或只有 OOM，报告该结果后结束扫描。记录 OOM 发生在加载、预热还是正式测量。报告统一称为“人工分配器上限”，保留正常整卡的结果作为主结论。

## 五、验证训练更新

用 L1 和同一输入，在独立进程运行 `--check-role-schedules --output one_update.json`。CUDA 确定性重放需要时，在启动 Python 前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。该程序会把参考张量存入主机内存，启动前确认可用 RAM。记录两种调度的损失、梯度范数，以及梯度、更新后参数、AdamW 两组动量和步数的逐元素比较结果。若 `exact=false`，保留原始差异，并在相同初态下做一次 A 对 A 重放以区分调度差异和后端自身波动。若受确定性算子或内存限制无法完成，明确标记数值等价性未验证。不要用两个独立测速进程的 loss 代替这项检查。

## 六、我要带回的报告

在结果目录生成 `H200_REPORT.md`，以三句摘要开头，依次回答“正常 H200 有无重试停顿”“提前反向能否缩短完整步”“人工压力改变了什么”。每句附上数字和比较条件。报告至少包含：

- 环境、commit、样本、精确布局来源、实际重复次数和一次可复现的命令。
- 正常条件的完整步表。列出每轮配对值、平均步时、加速比、峰值 allocated/reserved，并说明几轮方向一致。
- 分配器诊断表。列出观测步数、有重试的步数、整步与热点的重试总数、热点耗时、同步耗时、缓存归还量。注明哪些数来自独立诊断进程。
- 人工压力结果和数值一致性结果。未执行或失败的项目写明原因；零重试是有效结果。
- 与昇腾对照的一段解释。若 H200 加速却没有重试证据，将加速原因列为未确定；若仅在限制容量后复现，把结论限定于该压力条件。
- 附录列出原始 JSON、汇总 CSV 和启动脚本的相对路径。将这些文件与报告打包为 `h200_dmd2_results.tar.gz`，排除权重和数据集。最后在终端打印报告和压缩包的绝对路径。

执行到报告生成。若硬件、输入或环境阻断部分实验，仍以实际完成的部分生成报告，写明缺少哪项观测、它影响哪个结论。
