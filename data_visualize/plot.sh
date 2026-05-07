#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pangu1b_hif8

python log_visualizer.py train_bf16_lr1e5_20260505_174847_success.log train_max_quant_lr1e5_20260506_015312_success.log