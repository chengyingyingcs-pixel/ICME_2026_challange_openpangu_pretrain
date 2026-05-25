# OpenPangu-Embedded-1B — HiF8 W8A8 QAT Submission

IEEE ICME 2026 Low Bit-width Large Model Quantization Challenge submission, featuring HiF8 W8A8 Quantization-Aware Training (QAT) applied to OpenPangu-Embedded-1B.

Quantized model weights: [yycheng0122/pangu_pretrain_submit](https://huggingface.co/yycheng0122/pangu_pretrain_submit)

## Repository Structure

```
.
├── pangu_hif8_pretrain/
│   ├── train.py                          # Main training script
│   ├── hif8.py                           # HiF8 amax tracking & DTS algorithms
│   ├── run_train_bf16_lr1e5.sh           # BF16 baseline training
│   └── run_train_max_quant_lr1e5.sh      # HiF8 QAT training (final submission)
├── HiFloat8/                             # CUDA & NPU quantization library
├── evaluate_benchmarks/
│   ├── run_eval.sh                       # Benchmark evaluation script
│   └── compare_results.py               # Summary table generator
├── data_visualize/                       # Training loss visualization
├── pangu1b_hif8.yml                      # Training conda environment
├── pangu1b_eval.yml                      # Evaluation conda environment
├── Dockerfile                            # Container environment
└── REPRODUCE.md                          # Detailed reproduction guide
```

## Method

**HiF8 W8A8 Quantization-Aware Training** with key parameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| amax algorithm | `max` (over history window) | Prevents quantization saturation |
| amax history length | 64 steps | Stable scale estimation |
| BF16 warmup steps | 500 | Stabilize pretrained weights before quantization |
| HiF8 max value (fwd/bwd) | 15 | Maps peak representable value to highest-precision tier |
| Learning rate | 1e-5 | Reduces catastrophic forgetting of pretrained knowledge |
| Global batch size | 1024 | — |
| Training steps | 10,000 | — |
| High-precision layers | 5 | First and last 5 layers kept in BF16 |

## Training Loss

The HiF8 QAT training loss tracks the BF16 baseline extremely closely:
- **Average Percentage Error (APE)**: 0.11% over all 10,000 steps; 0.12% over the final 1,000 steps
- Demonstrates effective elimination of quantization saturation

## Benchmark Results

| Task | BF16 (lr=1e-5) | HiF8 QAT (lr=1e-5) | Drop |
|------|:-:|:-:|:-:|
| MMLU (5-shot) | 43.36% | 43.17% | **0.43%** ✓ |
| GSM8K (5-shot) | 1.59% | 1.29% | — (noise) |
| MATH500 (4-shot) | 0.50% | 0.46% | — (noise) |
| HellaSwag (10-shot) | 41.10% | 40.86% | **0.58%** ✓ |
| ARC-Easy (25-shot) | 51.56% | 51.01% | **1.06%** |
| ARC-Challenge (25-shot) | 36.77% | 36.69% | **0.22%** ✓ |

GSM8K and MATH500 drops are statistically insignificant at 1B scale (<0.3%).

## Reproduction Instructions

### 1. Environment Setup

```bash
# Training environment
conda env create -f pangu1b_hif8.yml
conda activate pangu1b_hif8

# Evaluation environment
conda env create -f pangu1b_eval.yml
conda activate pangu1b_eval
pip install lm-eval==0.4.11 ray==2.55.1 "antlr4-python3-runtime==4.11" sympy math_verify
```

### 2. Prepare Base Model

Download the base model and place it at:
```
pangu_hif8_pretrain/models/openPangu-Embedded-1B/
```

The FineWeb training dataset (sample-10BT, ~48 GB) is downloaded automatically from HuggingFace on first run, or set `--cache_dir` in the training script to a local path.

### 3. Build HiFloat8 CUDA Extension

```bash
cd HiFloat8/hif8_cuda
pip install -e .
```

### 4. Train BF16 Baseline

```bash
bash pangu_hif8_pretrain/run_train_bf16_lr1e5.sh
# Output: checkpoints/bf16_lr1e5/final/
# Runtime: ~21 hours on 8× NVIDIA H100 80GB
```

### 5. Train HiF8 QAT (this submission)

```bash
bash pangu_hif8_pretrain/run_train_max_quant_lr1e5.sh
# Output: checkpoints/max_quant_lr1e5/final/
# Runtime: ~21 hours on 8× NVIDIA H100 80GB
```

### 6. Evaluate

```bash
cd evaluate_benchmarks

# Evaluate a single model
bash run_eval.sh max_quant_lr1e5

# Generate full comparison table
python compare_results.py
```

Benchmarks: MMLU (5-shot), GSM8K (5-shot), MATH500/minerva_math (4-shot), HellaSwag (10-shot), ARC-Easy (25-shot), ARC-Challenge (25-shot).

## Hardware

- **GPUs**: 8× GPU 80GB
- **Training time**: ~21 hours per run
- **Framework**: PyTorch 2.6 + torchrun (DDP)

## Docker (Alternative)

```bash
docker build -t pangu1b-hif8:latest .

docker run --gpus all \
    -v /data0/dataset:/data0/dataset \
    -v $(pwd)/checkpoints:/workspace/checkpoints \
    -v $(pwd)/logs:/workspace/logs \
    --shm-size=64g \
    pangu1b-hif8:latest \
    bash pangu_hif8_pretrain/run_train_max_quant_lr1e5.sh
```
