"""
Log Visualizer for HiF8 QAT Training

Usage:
    python log_visualizer.py baseline.log exp1.log exp2.log ...

Produces 3 plots:
  1. Loss curves for all runs
  2. Learning rate curves for all runs
  3. Absolute Percentage Error of each experiment vs baseline (per step)
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── Log parsing ──────────────────────────────────────────────────────────────

STEP_RE = re.compile(
    r"step=(\d+)/\d+"
    r".*?loss=([\d.]+)"
    r".*?lr=([\d.e+\-]+)"
)


def parse_log(path: str) -> dict:
    """Return {step: int, loss: float, lr: float} records from a training log."""
    records = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                records.append({
                    "step": int(m.group(1)),
                    "loss": float(m.group(2)),
                    "lr":   float(m.group(3)),
                })
    if not records:
        raise ValueError(f"No training step lines found in: {path}")
    return records


def to_arrays(records):
    steps = [r["step"] for r in records]
    losses = [r["loss"] for r in records]
    lrs    = [r["lr"]   for r in records]
    return steps, losses, lrs


# ── Plotting helpers ──────────────────────────────────────────────────────────

COLORS = [
    "#1f77b4",  # blue   – baseline
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
]

BASELINE_STYLE = dict(linewidth=2.0, linestyle="-")
EXP_STYLE      = dict(linewidth=1.5, linestyle="--")


def short_name(path: str) -> str:
    return Path(path).stem


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    log_paths = sys.argv[1:]
    names     = [short_name(p) for p in log_paths]

    print(f"Parsing {len(log_paths)} log file(s)...")
    all_records = []
    for p in log_paths:
        recs = parse_log(p)
        all_records.append(recs)
        print(f"  {Path(p).name}: {len(recs)} steps")

    baseline_records = all_records[0]
    baseline_steps, baseline_losses, baseline_lrs = to_arrays(baseline_records)
    baseline_step_set = {r["step"]: r for r in baseline_records}

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.suptitle("HiF8 QAT Training Comparison", fontsize=14, fontweight="bold")

    ax_loss, ax_lr, ax_ape = axes

    # ── Plot 1: Loss curves ───────────────────────────────────────────────────
    ax_loss.set_title("Training Loss")
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")

    for i, (recs, name) in enumerate(zip(all_records, names)):
        steps, losses, _ = to_arrays(recs)
        style = BASELINE_STYLE if i == 0 else EXP_STYLE
        label = f"{name} (baseline)" if i == 0 else name
        ax_loss.plot(steps, losses, color=COLORS[i % len(COLORS)],
                     label=label, **style)

    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.3)

    # ── Plot 2: Learning rate curves ──────────────────────────────────────────
    ax_lr.set_title("Learning Rate")
    ax_lr.set_xlabel("Step")
    ax_lr.set_ylabel("LR")

    for i, (recs, name) in enumerate(zip(all_records, names)):
        steps, _, lrs = to_arrays(recs)
        style = BASELINE_STYLE if i == 0 else EXP_STYLE
        label = f"{name} (baseline)" if i == 0 else name
        ax_lr.plot(steps, lrs, color=COLORS[i % len(COLORS)],
                   label=label, **style)

    ax_lr.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax_lr.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax_lr.legend(fontsize=8)
    ax_lr.grid(True, alpha=0.3)

    # ── Plot 3: Absolute Percentage Error vs baseline ─────────────────────────
    ax_ape.set_title("Loss Absolute Percentage Error vs Baseline")
    ax_ape.set_xlabel("Step")
    ax_ape.set_ylabel("APE (%)")

    has_exp = False
    for i, (recs, name) in enumerate(zip(all_records, names)):
        if i == 0:
            continue  # skip baseline itself

        ape_steps, ape_vals = [], []
        for r in recs:
            s = r["step"]
            if s in baseline_step_set:
                base_loss = baseline_step_set[s]["loss"]
                if base_loss != 0:
                    ape = abs(r["loss"] - base_loss) / abs(base_loss) * 100.0
                    ape_steps.append(s)
                    ape_vals.append(ape)

        if ape_steps:
            ax_ape.plot(ape_steps, ape_vals,
                        color=COLORS[i % len(COLORS)],
                        label=name, **EXP_STYLE)
            has_exp = True

    if not has_exp:
        ax_ape.text(0.5, 0.5, "No experiment logs provided\n(need at least 2 log files)",
                    ha="center", va="center", transform=ax_ape.transAxes, fontsize=11)
    else:
        ax_ape.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax_ape.legend(fontsize=8)

    ax_ape.grid(True, alpha=0.3)

    # ── Print average APE over all steps and last 1000 steps vs baseline ────────
    def _compute_ape(recs, step_threshold=None):
        """返回与 baseline 重叠步的 APE 列表，step_threshold 为 None 时取全部步。"""
        vals = []
        for r in recs:
            if step_threshold is not None and r["step"] <= step_threshold:
                continue
            s = r["step"]
            if s in baseline_step_set:
                base_loss = baseline_step_set[s]["loss"]
                if base_loss != 0:
                    vals.append(abs(r["loss"] - base_loss) / abs(base_loss) * 100.0)
        return vals

    for label, threshold_fn in [
        ("all steps",    lambda recs: None),
        ("last 1000 steps", lambda recs: max(r["step"] for r in recs) - 1000),
    ]:
        print(f"\nAverage APE vs baseline ({label}):")
        for i, (recs, name) in enumerate(zip(all_records, names)):
            if i == 0:
                continue
            ape_vals = _compute_ape(recs, threshold_fn(recs))
            if ape_vals:
                print(f"  {name}: {sum(ape_vals)/len(ape_vals):.4f}%  (n={len(ape_vals)})")
            else:
                print(f"  {name}: N/A (no overlapping steps with baseline)")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir  = Path(__file__).parent
    out_path = out_dir / "training_comparison.png"
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
