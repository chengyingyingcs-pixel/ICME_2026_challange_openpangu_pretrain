# Reproduction Guide — HiF8 QAT for OpenPangu-Embedded-1B

IEEE ICME 2026 Low Bit-width Large Model Quantization Challenge submission.

## Environment Setup

### Training environment (pangu1b_hif8)
```bash
conda env create -f pangu1b_hif8.yml
conda activate pangu1b_hif8
```

### Evaluation environment (pangu1b_eval)
```bash
conda env create -f pangu1b_eval.yml
conda activate pangu1b_eval
pip install lm-eval==0.4.11 ray==2.55.1 "antlr4-python3-runtime==4.11" sympy math_verify
```

## Quantized Model

The submitted quantized checkpoint corresponds to **max_quant_lr1e5** (HiF8 W8A8 QAT, lr=1e-5).

Place the checkpoint at:
```
checkpoints/max_quant_lr1e5/final/
```

## Reproduce Training

### Step 1 — BF16 baseline (lr=1e-5)
```bash
bash pangu_hif8_pretrain/run_train_bf16_lr1e5.sh
```
Output: `checkpoints/bf16_lr1e5/final/`

### Step 2 — HiF8 QAT (max_quant_lr1e5, lr=1e-5)
```bash
bash pangu_hif8_pretrain/run_train_max_quant_lr1e5.sh
```
Output: `checkpoints/max_quant_lr1e5/final/`

Key quantization settings:
| Parameter | Value |
|-----------|-------|
| Quantization | W8A8 HiF8 |
| amax algorithm | `max` (over history window) |
| amax history length | 64 steps |
| BF16 warmup steps | 500 |
| HiF8 max value (fwd/bwd) | 15 |
| Learning rate | 1e-5 |
| Global batch size | 1024 |
| Max steps | 10000 |
| High-precision layers | 5 |

Training takes approximately **21 hours** on 8× NVIDIA H100 80GB.

## Reproduce Evaluation

```bash
cd evaluate_benchmarks

# Evaluate a single model
bash run_eval.sh max_quant_lr1e5

# Generate comparison table (baseline: bf16_lr1e5)
python compare_results.py
```

Benchmarks: MMLU (5-shot), GSM8K (5-shot), MATH500/minerva_math (4-shot),
HellaSwag (10-shot), ARC-Easy (25-shot), ARC-Challenge (25-shot).

## Key Results (max_quant_lr1e5 vs bf16_lr1e5 baseline)

| Task | BF16 (lr=1e-5) | HiF8 QAT (lr=1e-5) | Drop |
|------|---------------|---------------------|------|
| MMLU (5-shot) | 43.36% | 43.17% | 0.43% |
| GSM8K (5-shot) | 1.59% | 1.29% | — (noise) |
| MATH500 (4-shot) | 0.50% | 0.46% | — (noise) |
| HellaSwag (10-shot) | 41.10% | 40.86% | 0.58% |
| ARC-Easy (25-shot) | 51.56% | 51.01% | 1.06% |
| ARC-Challenge (25-shot) | 36.77% | 36.69% | 0.22% |
