"""
比较 bf16 / cts / dts_exp 三种模型在各 benchmark 上的精度，
打印精度对比表及相对于 bf16 基线的精度损失，并判断是否满足比赛要求（< 1.0%）。

用法:
    python compare_results.py
"""

import json
import os
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

# MODELS = ["bf16", "cts", "dts_exp"]
MODELS = ["bf16_lr1e5", "max_quant_lr1e5", "bf16", "cts", "max_quant", "dts_exp"]
BASELINE = "bf16_lr1e5"

# lm-eval 中各任务的主要 metric
TASK_METRICS = {
    "mmlu":            "acc,none",
    "gsm8k":           "exact_match,flexible-extract",
    "minerva_math":    "math_verify,none",
    "hellaswag":       "acc_norm,none",
    "arc_easy":        "acc_norm,none",
    "arc_challenge":   "acc_norm,none",
}

TASK_DISPLAY = {
    "mmlu":            "MMLU          (5-shot)",
    "gsm8k":           "GSM8K         (5-shot)",
    "minerva_math":    "MATH500       (4-shot)",
    "hellaswag":       "HellaSwag    (10-shot)",
    "arc_easy":        "ARC-Easy     (25-shot)",
    "arc_challenge":   "ARC-Challenge(25-shot)",
}

ACCURACY_LOSS_THRESHOLD = 1.0  # 比赛要求精度损失 < 1.0%


def load_results(model_name: str) -> dict:
    """从 results/<model>/ 下所有最新的 lm-eval 结果文件中合并结果。"""
    result_dir = RESULTS_DIR / model_name
    candidates = sorted(result_dir.glob("**/results*.json"), key=os.path.getmtime)
    if not candidates:
        return {}
    # 找到最新一批文件（同一次完整评测可能分多个任务写入多个文件）
    # 取最新文件的时间戳往前推，收集同一轮评测的所有文件
    latest_mtime = os.path.getmtime(candidates[-1])
    cutoff = latest_mtime - 24 * 3600  # 最近 24 小时内的文件视为同一轮
    recent = [f for f in candidates if os.path.getmtime(f) >= cutoff]
    merged = {"results": {}, "configs": {}}
    for f in recent:
        with open(f) as fp:
            data = json.load(fp)
        merged["results"].update(data.get("results", {}))
        merged["configs"].update(data.get("configs", {}))
    return merged


def extract_score(results: dict, task: str, metric: str) -> float | None:
    """从 lm-eval 结果 JSON 中提取指定任务和 metric 的分数。"""
    task_results = results.get("results", {}).get(task, {})
    if not task_results:
        return None
    return task_results.get(metric)


def main():
    import datetime
    # ── 加载所有模型结果 ──────────────────────────────────────────────────────
    all_results = {}
    for model in MODELS:
        data = load_results(model)
        if not data:
            print(f"[WARN] No results found for model: {model}")
        all_results[model] = data

    lines = []

    # ── 打印表头 ──────────────────────────────────────────────────────────────
    col_w = 26
    score_w = 12
    drop_w = 14

    header = f"{'Task':<{col_w}}"
    for m in MODELS:
        header += f"{m:>{score_w}}"
    for m in MODELS:
        if m == BASELINE:
            continue
        header += f"{'vs '+BASELINE+' drop':>{drop_w}}"
    sep = "=" * (col_w + score_w * len(MODELS) + drop_w * (len(MODELS) - 1))
    lines.append("")
    lines.append(sep)
    lines.append(header)
    lines.append("-" * (col_w + score_w * len(MODELS) + drop_w * (len(MODELS) - 1)))

    # ── 逐任务打印结果 ────────────────────────────────────────────────────────
    fail_cases = []
    na_cases   = []

    for task, metric in TASK_METRICS.items():
        scores = {}
        for model in MODELS:
            scores[model] = extract_score(all_results[model], task, metric)

        row = f"{TASK_DISPLAY[task]:<{col_w}}"
        for m in MODELS:
            s = scores[m]
            row += f"{s*100:>{score_w}.2f}%" if s is not None else f"{'N/A':>{score_w}}"

        for m in MODELS:
            if m == BASELINE:
                continue
            base = scores[BASELINE]
            s    = scores[m]
            if base is None or s is None:
                row += f"{'N/A':>{drop_w}}"
                na_cases.append((m, task))
            else:
                drop = (base - s) / base * 100.0 if base != 0 else 0.0
                flag = " x" if drop > ACCURACY_LOSS_THRESHOLD else "  "
                row += f"{drop:>{drop_w-2}.4f}%{flag}"
                if drop > ACCURACY_LOSS_THRESHOLD:
                    fail_cases.append((m, task, drop))

        lines.append(row)

    lines.append(sep)

    # ── 平均精度损失汇总 ──────────────────────────────────────────────────────
    lines.append("")
    lines.append("-- Average accuracy drop vs bf16 baseline --")
    for m in MODELS:
        if m == BASELINE:
            continue
        drops = []
        for task, metric in TASK_METRICS.items():
            base = extract_score(all_results[BASELINE], task, metric)
            s    = extract_score(all_results[m], task, metric)
            if base is not None and s is not None:
                drops.append((base - s) / base * 100.0 if base != 0 else 0.0)
        if drops:
            avg = sum(drops) / len(drops)
            status = "PASS" if avg < ACCURACY_LOSS_THRESHOLD else "FAIL"
            lines.append(f"  {m:<12}: avg drop = {avg:.4f}%  [{status} < {ACCURACY_LOSS_THRESHOLD}%]")
        else:
            lines.append(f"  {m:<12}: N/A")

    # ── 不达标提示 ────────────────────────────────────────────────────────────
    if fail_cases:
        lines.append(f"\n[!] Tasks exceeding {ACCURACY_LOSS_THRESHOLD}% accuracy loss threshold:")
        for m, task, drop in fail_cases:
            lines.append(f"    {m} / {task}: {drop:.4f}%")

    if na_cases:
        lines.append("\n[!] Missing results (run run_eval.sh first):")
        for m, task in na_cases:
            lines.append(f"    {m} / {task}")

    # ── 输出到终端和 txt 文件 ─────────────────────────────────────────────────
    output = "\n".join(lines)
    print(output)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"summary_{timestamp}.txt"
    with open(out_path, "w") as f:
        f.write(f"Evaluation Summary  ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write(output + "\n")
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()