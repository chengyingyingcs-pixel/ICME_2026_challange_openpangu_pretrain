#!/bin/bash
# 安装 lm-evaluation-harness 及评测依赖

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pangu1b_hif8

pip install lm-eval==0.4.4
pip install antlr4-python3-runtime==4.11  # MATH500 解析依赖
pip install sympy                          # MATH500 符号计算