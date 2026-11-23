# 公共环境：所有实验脚本 source 本文件
set -eu
source /inspire/hdd/global_user/s26045/fastvideo/.venv-h200/bin/activate
export RUN_DIR=/inspire/hdd/global_user/s26045/fastvideo/h200_stall_results/code_d7239500
cd "$RUN_DIR"
export PYTHONPATH="$RUN_DIR${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export EXAMPLE="$RUN_DIR/examples/train/activation_memory"
export WAN_MODEL=/inspire/hdd/global_user/s26045/fastvideo/weights/Wan2.1-T2V-1.3B-Diffusers
export RESULTS=/inspire/hdd/global_user/s26045/fastvideo/h200_stall_results
export SAMPLE_PT="$RESULTS/inputs/hiker_17_256_448.pt"
export PARQUET_DIR="$RESULTS/inputs/hiker_parquet"
export L1="$EXAMPLE/stall_study/layouts/solver.json"
export L2="$RESULTS/layouts/L2_h200_local.json"
