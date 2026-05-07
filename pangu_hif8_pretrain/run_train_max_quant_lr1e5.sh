#!/bin/bash
# HiF8 QAT 训练脚本（lr=1e-5）
# 改动：学习率 1e-4→1e-5，其余与 max_quant 方案相同
# 输出 checkpoint: checkpoints/max_quant_lr1e5/

PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"

source /home/chengyingying/miniconda3/etc/profile.d/conda.sh
conda activate pangu1b_hif8

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HIF8_PATH=/home/chengyingying/pangu_pretrain/HiFloat8/hif8_cuda

PROJ=/home/chengyingying/pangu_pretrain
MODEL_PATH="$PROJ/pangu_hif8_pretrain/models/openPangu-Embedded-1B"
OUTPUT_DIR="$PROJ/checkpoints/max_quant_lr1e5"
CACHE_DIR="/data0/dataset/train_pangu1b"

current_time=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="$PROJ/logs"
mkdir -p "$OUTPUT_DIR"
log_file="$LOG_DIR/train_max_quant_lr1e5_${current_time}.log"

cmd=(
torchrun --nproc_per_node=8 "$PATCH_DIR/train.py" \
  --model_path          "$MODEL_PATH" \
  --micro_batch_size    16    \
  --global_batch_size   1024  \
  --max_steps           10000 \
  --lr                  1e-5  \
  --warmup_steps        300   \
  --high_precision_layers 5   \
  --seq_len             1024  \
  --fineweb_subset      sample-10BT \
  --log_interval        1     \
  --save_interval       50000 \
  --output_dir          "$OUTPUT_DIR" \
  --cache_dir           "$CACHE_DIR"
  --use_hif8 true
  --hif8-enable
  --hif8-max-fwd=15
  --hif8-max-bwd=15
  --hif8-amax-history-len=64
  --hif8-warmup-steps=500
  --hif8-interval-schedule="0-10000:1"
  --hif8-amax-compute-algo-schedule="0-10000:max"
)

echo "Log: $log_file"
"${cmd[@]}" 2>&1 | tee "$log_file"

# 后台运行示例:
# tmux new -s train_max_quant_lr1e5
# bash /home/chengyingying/pangu_pretrain/pangu_hif8_pretrain/run_train_max_quant_lr1e5.sh
# Ctrl+B, D 脱离会话
