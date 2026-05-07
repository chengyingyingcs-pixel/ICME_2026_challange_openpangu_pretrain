# HiF8 QAT Training Environment
# Base: CUDA 12.4 devel（与 torch 2.6.0+cu124 匹配，需要 nvcc 编译 quant_cy 扩展）
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# ── 系统依赖 ──────────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    git wget curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip

# ── PyTorch（CUDA 12.4）──────────────────────────────────────────────────────
RUN pip install \
    torch==2.6.0+cu124 \
    torchaudio==2.6.0+cu124 \
    torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# ── Python 依赖 ───────────────────────────────────────────────────────────────
RUN pip install \
    transformers==4.53.3 \
    accelerate==1.13.0 \
    datasets==4.8.4 \
    huggingface_hub==0.36.2 \
    tokenizers==0.21.4 \
    safetensors==0.7.0 \
    sentencepiece==0.2.1 \
    numpy==2.2.6 \
    pyarrow==23.0.1 \
    fsspec==2026.2.0 \
    requests==2.33.1 \
    packaging==26.0 \
    triton==3.2.0 \
    sympy==1.13.1 \
    antlr4-python3-runtime==4.11.0 \
    lm-eval==0.4.4

# ── 复制项目代码 ──────────────────────────────────────────────────────────────
WORKDIR /workspace
COPY . /workspace/tracy

# ── 编译 HiFloat8 quant_cy CUDA 扩展 ─────────────────────────────────────────
WORKDIR /workspace/tracy/HiFloat8/hif8_cuda
RUN bash build.sh

# ── 设置 PYTHONPATH，使 hif8.py / quant_cy 可直接 import ─────────────────────
ENV PYTHONPATH="/workspace/tracy/HiFloat8/hif8_cuda:/workspace/tracy/pangu_hif8_pretrain:${PYTHONPATH}"

WORKDIR /workspace/tracy