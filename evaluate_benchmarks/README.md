# Benchmark Evaluation

评测 bf16 / cts / dts_exp 三种模型在比赛指定 benchmark 上的精度，并与 bf16 基线对比精度损失。

## 目录结构

```
evaluate_benchmarks/
├── setup.sh            # 安装依赖（只需运行一次）
├── run_eval.sh         # 运行评测，结果保存到 results/
├── compare_results.py  # 汇总对比各模型精度及精度损失
└── results/            # 评测结果自动生成目录
    ├── bf16/
    ├── cts/
    └── dts_exp/
```

## 评测任务

| Benchmark     | Shot | 说明 |
|---|---|---|
| MMLU          | 5-shot（默认）  | 多学科选择题 |
| GSM8K         | 5-shot（默认）  | 小学数学应用题 |
| MATH500       | 4-shot（默认）  | 竞赛数学（minerva_math） |
| HellaSwag     | 10-shot（默认） | 常识推理补全 |
| ARC-Easy      | 25-shot（默认） | 科学选择题（简单） |
| ARC-Challenge | 25-shot（默认） | 科学选择题（困难） |
| ~~PIQA~~      | -              | 数据集加载脚本已不受支持，已移除 |

## 比赛目标指标

- 训练损失 APE < **0.5%**（通过 `data_visualize/log_visualizer.py` 验证）
- 精度损失 < **1.0%**（相对于 bf16 基线的平均精度下降）

## 使用流程

### 第一步：安装依赖（只需一次）

```bash
cd /root/tracy/evaluate_benchmarks
bash setup.sh
```

### 第二步：运行评测

评测全部三个模型（推荐在 tmux 中运行，耗时较长）：

```bash
bash run_eval.sh
bash run_eval.sh 2>&1 | tee /root/chengyingying/evaluate_benchmarks/results/eval_$(date +%Y%m%d_%H%M%S).log

```

也可单独评测某一个模型：

```bash
bash run_eval.sh bf16
bash run_eval.sh cts
bash run_eval.sh dts_exp
```

评测使用全部可用 GPU 并行推理（`parallelize=True`），结果保存到 `results/<model_name>/`。

### 第三步：查看对比结果

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate tracy
python3 compare_results.py
```

输出示例：

```
Task                          bf16         cts     dts_exp  vs bf16 drop  vs bf16 drop
──────────────────────────────────────────────────────────────────────────────────────
MMLU          (5-shot)       45.23%      44.98%    45.10%        0.5535%       0.2878%
GSM8K         (5-shot)       12.10%      12.05%    12.08%        0.4132%       0.1653%
...

── Average accuracy drop vs bf16 baseline ──
  cts        : avg drop = 0.3821%  [✓ PASS < 1.0%]
  dts_exp    : avg drop = 0.2156%  [✓ PASS < 1.0%]
```

超出 1.0% 阈值的任务会标注 `✗` 并在末尾汇总。

## 模型路径

| 模型 | 路径 |
|---|---|
| bf16 基线 | `/root/tracy/checkpoints/bf16/final` |
| cts        | `/root/tracy/checkpoints/cts/final` |
| dts_exp    | `/root/tracy/checkpoints/dts_exp/final` |

## 常见问题

**Q: 出现 `` `trust_remote_code` is not supported anymore `` 警告**

A: 这是 datasets 库的无害警告，不影响评测结果，忽略即可。

**Q: MATH500 任务找不到**

A: lm-eval 中 MATH500 对应任务名为 `minerva_math`，已在脚本中配置好。

**Q: 想重新评测某个模型**

A: 删除对应结果目录后重新运行：
```bash
rm -rf results/cts
bash run_eval.sh cts
```