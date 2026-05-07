#!/bin/bash
# 评测 bf16 / cts / dts_exp 三种模型在比赛指定 benchmark 上的精度
#
# 用法:
#   bash run_eval.sh              # 评测全部三个模型
#   bash run_eval.sh bf16         # 只评测某一个模型
#
# 结果保存到 results/<model_name>/

source /home/chengyingying/miniconda3/etc/profile.d/conda.sh
conda activate pangu1b_eval
export PATH="/home/chengyingying/miniconda3/envs/pangu1b_eval/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="/home/chengyingying/pangu_pretrain/evaluate_benchmarks/results"
CKPT_DIR="/home/chengyingying/pangu_pretrain/checkpoints"

# ── 评测任务配置 ─────────────────────────────────────────────────────────────
# 任务名称（lm-eval 标准名），shot 数参考各 benchmark 论文惯例
TASKS="mmlu,gsm8k,minerva_math,hellaswag,arc_easy,arc_challenge"

declare -A TASK_SHOTS=(
    ["mmlu"]="5"
    ["gsm8k"]="5"
    ["minerva_math"]="4"   # MATH500 使用 minerva_math（4-shot CoT）
    ["hellaswag"]="10"
    ["arc_easy"]="25"
    ["arc_challenge"]="25"
    ["piqa"]="0"
)

# ── 待评测模型 ────────────────────────────────────────────────────────────────
# declare -A MODELS=(
    # ["bf16"]="$CKPT_DIR/bf16/final"
    # ["cts"]="$CKPT_DIR/cts/final"
    # ["dts_exp"]="$CKPT_DIR/dts_exp/final"
# )

declare -A MODELS=(
    ["bf16"]="$CKPT_DIR/bf16/final"
    ["cts"]="$CKPT_DIR/cts/final"
    ["max_quant"]="$CKPT_DIR/max_quant/final"
    ["bf16_lr1e5"]="$CKPT_DIR/bf16_lr1e5/final"
    ["max_quant_lr1e5"]="$CKPT_DIR/max_quant_lr1e5/final"
    # ["dts_exp"]="$CKPT_DIR/dts_exp_2/final"
)

# 若指定了参数，只评测该模型
if [[ -n "$1" ]]; then
    if [[ -z "${MODELS[$1]}" ]]; then
        echo "Unknown model: $1. Available: ${!MODELS[*]}"
        exit 1
    fi
    TARGETS=("$1")
else
    TARGETS=("${!MODELS[@]}")
fi

# ── 逐个模型评测 ──────────────────────────────────────────────────────────────
for MODEL_NAME in "${TARGETS[@]}"; do
    MODEL_PATH="${MODELS[$MODEL_NAME]}"
    OUT_DIR="$RESULTS_DIR/$MODEL_NAME"
    mkdir -p "$OUT_DIR"

    echo "======================================================"
    echo "Evaluating: $MODEL_NAME  ($MODEL_PATH)"
    echo "======================================================"

    for TASK in mmlu gsm8k minerva_math hellaswag arc_easy arc_challenge; do
        SHOTS="${TASK_SHOTS[$TASK]}"
        lm_eval \
            --model vllm \
            --model_args "pretrained=$MODEL_PATH,dtype=bfloat16,trust_remote_code=True,tensor_parallel_size=1" \
            --tasks "$TASK" \
            --num_fewshot "$SHOTS" \
            --apply_chat_template False \
            --output_path "$OUT_DIR" \
            --log_samples \
            2>&1 | tee -a "$OUT_DIR/eval.log"
    done

    echo "Done: $MODEL_NAME → $OUT_DIR"
done

echo ""
echo "All evaluations complete. Run compare_results.py to see summary."