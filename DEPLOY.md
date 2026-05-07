# 部署文档

## 环境信息

| 组件 | 版本 |
|---|---|
| CUDA Driver | 570.133.20 |
| CUDA Runtime | 12.4 |
| Python | 3.10 |
| PyTorch | 2.6.0+cu124 |
| Transformers | 4.53.3 |
| lm-eval | 0.4.4 |

## 构建 Docker 镜像

在 `tracy/` 目录下执行：

```bash
cd /root/tracy
docker build -t pangu1b-hif8:latest .
```

首次构建需下载基础镜像并编译 quant_cy CUDA 扩展，约需 10–20 分钟。

## 启动容器

### 训练

```bash
docker run --gpus all \
    -v /data0/dataset:/data0/dataset \
    -v /root/tracy/checkpoints:/workspace/tracy/checkpoints \
    -v /root/logs:/workspace/tracy/logs \
    --shm-size=64g \
    pangu1b-hif8:latest \
    bash pangu_hif8_pretrain/run_train.sh
```

参数说明：
- `--gpus all`：挂载所有 GPU
- `-v /data0/dataset`：挂载本地数据集缓存，避免重复下载
- `-v checkpoints / logs`：持久化训练结果和日志到宿主机
- `--shm-size=64g`：DataLoader 多进程需要足够的共享内存

### 评测

```bash
docker run --gpus all \
    -v /root/tracy/checkpoints:/workspace/tracy/checkpoints \
    -v /root/tracy/evaluate_benchmarks/results:/workspace/tracy/evaluate_benchmarks/results \
    pangu1b-hif8:latest \
    bash -c "cd evaluate_benchmarks && bash run_eval.sh"
```

### 交互式调试

```bash
docker run --gpus all -it \
    -v /root/tracy:/workspace/tracy \
    -v /data0/dataset:/data0/dataset \
    --shm-size=64g \
    pangu1b-hif8:latest \
    bash
```

## 目录结构

```
tracy/
├── Dockerfile
├── DEPLOY.md                        # 本文件
├── HiFloat8/                        # HiF8 量化库（含 quant_cy CUDA 扩展）
├── pangu_hif8_pretrain/             # 训练代码
│   ├── train.py
│   ├── hif8.py
│   └── run_train.sh
├── evaluate_benchmarks/             # 评测代码
│   ├── run_eval.sh
│   ├── compare_results.py
│   └── results/
├── checkpoints/                     # 训练产出模型权重
│   ├── bf16/final/
│   ├── cts/final/
│   └── dts_exp/final/
└── logs/                            # 训练日志
```

## 常见问题

**Q: `quant_cy` 编译失败**

确认宿主机 CUDA Driver ≥ 12.4（`nvidia-smi` 查看），且 Docker 使用了 `--gpus all`。

**Q: 训练时 OOM**

当前配置 `micro_batch_size=16, seq_len=1024`，需要约 60 GiB 显存。
如果 GPU 显存不足，在 `run_train.sh` 中减小 `micro_batch_size`。

**Q: 数据集下载慢**

首次运行会从 HuggingFace 下载 FineWeb sample-10BT（约 48 GB）到 `/data0/dataset/train_pangu1b`。
挂载 `-v /data0/dataset:/data0/dataset` 后后续运行直接读本地缓存。