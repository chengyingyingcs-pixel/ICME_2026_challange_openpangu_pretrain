"""
W8A8 HiF8 Quantization-Aware Training for OpenPangu1B
Mini-Challenge: ICME 2026 Low Bit-width Large Model Quantization Challenge

Strategy:
  - Quantize MLP Linear layers only (mlp.gate_proj, mlp.up_proj, mlp.down_proj); self_attn kept in BF16
  - Before each MLP GEMM: quantize activation, weight, gradient with HiF8
    - Forward:  y  = quant(x) @ quant(W)^T
    - Backward: dx = quant(grad_out) @ quant(W)
                dW = quant(grad_out)^T @ quant(x)
  - Keep first and last N blocks in full precision (up to 5 blocks total)
  - Quantization scheduling is handled by --hif8-amax-compute-algo-schedule
  - Training dataset: HuggingFaceFW/fineweb (streaming)
"""

import os
import sys
import math
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

# Add HiFloat8 to path
TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
HIF8_PATH = os.environ.get(
    "HIF8_PATH",
    "/home/chengyingying/pangu_pretrain/HiFloat8/hif8_cuda",
)
# Training dir first so local hif8.py takes priority over HiFloat8/hif8_cuda/hif8.py
sys.path.insert(0, HIF8_PATH)
sys.path.insert(0, TRAIN_DIR)

from quant_cy import QType, quant_dequant_float
from hif8 import hif8_qdq_with_amax, HiF8GlobalStateManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()

def cleanup_ddp():
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# HiF8 schedule parsing
# ---------------------------------------------------------------------------

def parse_schedule(schedule_str: str) -> list[tuple[int, int, str]]:
    """
    Parse a schedule string like "0-1000:v1,1000-20000:v2" into a list of
    (start, end, value) tuples. Works for both int and str values.
    """
    segments = []
    for part in schedule_str.split(","):
        part = part.strip()
        range_part, val = part.split(":")
        start, end = range_part.split("-")
        segments.append((int(start), int(end), val.strip()))
    return segments


def get_schedule_value(segments: list[tuple[int, int, str]], step: int) -> str:
    """Return the schedule value for the given step. Last segment wins if past end."""
    for start, end, val in segments:
        if start <= step < end:
            return val
    return segments[-1][2]


# ---------------------------------------------------------------------------
# Custom autograd Function: HiF8 GEMM with hif8_qdq_with_amax
# ---------------------------------------------------------------------------

class HiF8GemmFunction(torch.autograd.Function):
    """
    Implements  y = x @ W^T + b  with HiF8 quantization applied to
    activation (A), weight (W), and gradient (G) before every GEMM,
    using hif8_qdq_with_amax for amax tracking and dynamic scaling:

      Forward:   x_q = hif8_qdq_with_amax(x, key=.A, max_val=max_fwd, ...)
                 w_q = hif8_qdq_with_amax(w, key=.W, max_val=max_fwd, ...)
                 y   = x_q @ w_q^T + b

      Backward:  g_q = hif8_qdq_with_amax(g, key=.G, max_val=max_bwd, ...)
                 dx  = g_q @ w_q
                 dW  = g_q^T @ x_q
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor,
                bias: torch.Tensor | None,
                key: str, max_fwd: float, max_bwd: float,
                update_period: int, amax_history_len: int,
                amax_compute_algo: str) -> torch.Tensor:

        x_q = hif8_qdq_with_amax(
            x, key=f"{key}.A", max_val=max_fwd,
            update_period=update_period, amax_history_len=amax_history_len,
            amax_compute_algo=amax_compute_algo, tensor_type="A",
        )
        w_q = hif8_qdq_with_amax(
            weight, key=f"{key}.W", max_val=max_fwd,
            update_period=update_period, amax_history_len=amax_history_len,
            amax_compute_algo=amax_compute_algo, tensor_type="W",
        )
        ctx.save_for_backward(x_q, w_q)
        ctx.has_bias = bias is not None
        ctx.key = key
        ctx.max_bwd = max_bwd
        ctx.update_period = update_period
        ctx.amax_history_len = amax_history_len
        ctx.amax_compute_algo = amax_compute_algo
        return torch.nn.functional.linear(x_q, w_q, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_q, w_q = ctx.saved_tensors

        g_q = hif8_qdq_with_amax(
            grad_output, key=f"{ctx.key}.G", max_val=ctx.max_bwd,
            update_period=ctx.update_period, amax_history_len=ctx.amax_history_len,
            amax_compute_algo=ctx.amax_compute_algo, tensor_type="D",
        )

        grad_x = g_q.matmul(w_q)

        g_2d = g_q.reshape(-1, g_q.shape[-1])
        x_2d = x_q.reshape(-1, x_q.shape[-1])
        grad_w = g_2d.t().matmul(x_2d)

        grad_b = g_q.sum(list(range(g_q.dim() - 1))) if ctx.has_bias else None

        # 9 inputs → 9 grads (x, weight, bias, key, max_fwd, max_bwd,
        #                      update_period, amax_history_len, amax_compute_algo)
        return grad_x, grad_w, grad_b, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# HiF8 Quantized Linear Layer
# ---------------------------------------------------------------------------

class HiF8Linear(nn.Module):
    """
    Drop-in replacement for nn.Linear.
    When quantized=True: activation, weight, and gradient are quantized
    via hif8_qdq_with_amax (amax tracking + dynamic scaling).
    When quantized=False: behaves identically to standard Linear.
    """

    def __init__(self, linear: nn.Linear, key: str):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.key = key
        self.quantized = False      # toggled by delayed scaling scheduler
        # HiF8 config — set by patch_model_with_hif8 after arg parsing
        self.max_fwd: float = 15.0
        self.max_bwd: float = 15.0
        self.update_period: int = 1
        self.amax_history_len: int = 30
        self.amax_compute_algo: str = "most_recent"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantized:
            return HiF8GemmFunction.apply(
                x, self.weight, self.bias,
                self.key, self.max_fwd, self.max_bwd,
                self.update_period, self.amax_history_len,
                self.amax_compute_algo,
            )
        return torch.nn.functional.linear(x, self.weight, self.bias)


# ---------------------------------------------------------------------------
# Model patching: replace Linear layers with HiF8Linear
# ---------------------------------------------------------------------------

def patch_model_with_hif8(
    model: nn.Module,
    high_precision_layers: int = 5,
) -> nn.Module:
    """
    Replace Linear layers in MLP (gate_proj / up_proj / down_proj) with HiF8Linear.
    self_attn (q/k/v/o_proj) is kept in full BF16.

    The first ceil(high_precision_layers/2) and last floor(high_precision_layers/2)
    Transformer blocks are fully skipped (all BF16).
    lm_head is always kept in full precision.

    Returns the patched model with model._hif8_linears list attached.
    """
    transformer_blocks = list(model.model.layers)
    num_blocks = len(transformer_blocks)

    n_first = math.ceil(high_precision_layers / 2)
    n_last = high_precision_layers - n_first
    high_prec_indices = set(range(n_first)) | set(range(num_blocks - n_last, num_blocks))

    logger.info(
        f"Total Transformer blocks: {num_blocks}. "
        f"High-precision (skipped) block indices: {sorted(high_prec_indices)}"
    )

    hif8_linears = []

    # Sub-modules to patch: (attribute_name_on_block, sub_module_tag)
    # Attention projections are kept in BF16 — they are more sensitive to
    # quantization noise (outlier activations, softmax distribution shifts).
    # MLP dominates FLOPs and recovers well under QAT.
    submodule_targets = [
        ("mlp", "mlp"),
    ]

    for idx, block in enumerate(transformer_blocks):
        if idx in high_prec_indices:
            logger.info(f"  Block {idx}: fully skipped (BF16)")
            continue
        for attr, tag in submodule_targets:
            submod = getattr(block, attr)
            for name, child in list(submod.named_children()):
                if isinstance(child, nn.Linear):
                    key = f"layers.{idx}.{tag}.{name}"
                    hif8_mod = HiF8Linear(child, key=key)
                    setattr(submod, name, hif8_mod)
                    hif8_linears.append(hif8_mod)
                    logger.debug(f"  Block {idx} {tag}.{name}: replaced with HiF8Linear (key={key})")

    model._hif8_linears = hif8_linears
    logger.info(
        f"HiF8 patched: {len(hif8_linears)} Linear layers "
        f"(mlp gate/up/down_proj only; attn kept in BF16, "
        f"activation + weight + gradient quantized)."
    )
    return model


def set_quantization_enabled(model: nn.Module, enabled: bool):
    """Enable or disable quantization for all HiF8Linear modules."""
    for mod in model._hif8_linears:
        mod.quantized = enabled


def update_hif8_config(model: nn.Module, step: int,
                       max_fwd: float, max_bwd: float,
                       amax_history_len: int,
                       interval_segments: list,
                       algo_segments: list) -> None:
    """Update per-step HiF8 hyperparameters (update_period, algo) on all HiF8Linear layers."""
    update_period = int(get_schedule_value(interval_segments, step))
    amax_compute_algo = get_schedule_value(algo_segments, step)
    for mod in model._hif8_linears:
        mod.max_fwd = max_fwd
        mod.max_bwd = max_bwd
        mod.amax_history_len = amax_history_len
        mod.update_period = update_period
        mod.amax_compute_algo = amax_compute_algo


# ---------------------------------------------------------------------------
# FineWeb Streaming Dataset
# ---------------------------------------------------------------------------

class FineWebDataset(IterableDataset):
    """
    Dataset wrapper for HuggingFaceFW/fineweb.
    Tokenizes on the fly and chunks into fixed-length sequences.

    当 cache_dir 不为 None 时，以非流式模式加载数据集并缓存到本地磁盘，
    避免训练过程中依赖 HuggingFace 网络；同时按 rank 预分片，每个 rank
    只读取自己负责的分片，无需在迭代时跳过其他 rank 的文档。

    当 cache_dir 为 None 时，退回到原有在线流式模式。
    """

    def __init__(
        self,
        tokenizer,
        seq_len: int = 2048,
        max_samples: Optional[int] = None,
        subset: str = "sample-10BT",
        rank: int = 0,
        world_size: int = 1,
        cache_dir: Optional[str] = None,
    ):
        from datasets import load_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_samples = max_samples
        self.rank = rank
        self.world_size = world_size
        self.use_cache = cache_dir is not None

        if self.use_cache:
            # 非流式：首次运行自动下载并缓存到 cache_dir，后续直接从本地读取
            hf_dataset = load_dataset(
                "HuggingFaceFW/fineweb",
                name=subset,
                split="train",
                streaming=False,
                cache_dir=cache_dir,
            )
            # 按 rank 预分片，每个 rank 只持有自己的数据，避免重复跳过
            self.hf_dataset = hf_dataset.shard(
                num_shards=world_size, index=rank
            )
        else:
            # 在线流式模式（兜底）
            self.hf_dataset = load_dataset(
                "HuggingFaceFW/fineweb",
                name=subset,
                split="train",
                streaming=True,
            )

    def __iter__(self):
        buffer = []
        count = 0
        for doc_idx, sample in enumerate(self.hf_dataset):
            # 流式模式需要按 rank 跳过；缓存模式已预分片，无需跳过
            if not self.use_cache and doc_idx % self.world_size != self.rank:
                continue
            if self.max_samples and count >= self.max_samples:
                break
            tokens = self.tokenizer.encode(
                sample["text"], add_special_tokens=True, truncation=False
            )
            buffer.extend(tokens)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len + 1 :]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}
                count += 1
                if self.max_samples and count >= self.max_samples:
                    return


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _save_model_safe(model: nn.Module, tokenizer, save_dir: str):
    """Save model, fixing invalid GenerationConfig from the pretrained checkpoint."""
    from transformers import GenerationConfig
    # Fix top_p/top_k conflict with do_sample=False in the original generation_config
    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = True
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="W8A8 HiF8 QAT for OpenPangu1B")
    parser.add_argument("--model_path", type=str,
                        default="/root/pangu1b_hif8_training/models/openPangu-Embedded-1B")
    parser.add_argument("--output_dir", type=str,
                        default="/root/pangu1b_hif8_training/output")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--micro_batch_size", type=int, default=1,
                        help="Per-GPU batch size (micro batch size)")
    parser.add_argument("--global_batch_size", type=int, default=64,
                        help="Global batch size across all GPUs; grad_accum_steps is derived automatically")
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="LR warmup steps")
    parser.add_argument("--high_precision_layers", type=int, default=5,
                        help="Number of Transformer blocks to keep in high precision (max 5 per challenge rules)")
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--fineweb_subset", type=str, default="sample-10BT",
                        help="FineWeb subset: 'sample-10BT' or 'sample-100BT'")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap dataset samples (for debugging)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="本地数据集缓存目录；设置后以非流式模式下载并缓存数据集，"
                             "避免训练中依赖 HuggingFace 网络（推荐: /data0/dataset）")
    parser.add_argument("--use_hif8", type=lambda x: x.lower() == "true",
                        default=True,
                        help="Enable HiF8 quantization (true) or train in full BF16 (false)")
    # HiF8 amax-based quantization arguments
    parser.add_argument("--hif8-enable", action="store_true", default=False,
                        help="Use hif8_qdq_with_amax (amax tracking). "
                             "If not set, falls back to simple quant-dequant.")
    parser.add_argument("--hif8-max-fwd", type=float, default=15.0,
                        help="max_val for forward GEMM quantization (activation & weight)")
    parser.add_argument("--hif8-max-bwd", type=float, default=15.0,
                        help="max_val for backward GEMM quantization (gradient)")
    parser.add_argument("--hif8-amax-history-len", type=int, default=30,
                        help="Length of amax history window in HiF8GlobalStateManager")
    parser.add_argument("--hif8-interval-schedule", type=str,
                        default="0-10000:1",
                        help="update_period schedule, e.g. '0-1000:1,1000-20000:10'")
    parser.add_argument("--hif8-amax-compute-algo-schedule", type=str,
                        default="0-10000:most_recent",
                        help="amax_compute_algo schedule, "
                             "e.g. '0-1000:most_recent,1000-20000:exp_smooth'")
    parser.add_argument("--hif8-warmup-steps", type=int, default=0,
                        help="Train in BF16 for this many steps before enabling HiF8 "
                             "quantization. 0 = enable from step 0.")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── DDP 初始化 ─────────────────────────────────────────────────────────
    local_rank, rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    is_main = (rank == 0)   # 只有 rank 0 负责日志和保存

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"DDP: world_size={world_size}, rank={rank}, local_rank={local_rank}")
        logger.info(f"GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}")

    # ── Derive grad_accum_steps ────────────────────────────────────────────
    assert args.global_batch_size % (args.micro_batch_size * world_size) == 0, (
        f"global_batch_size ({args.global_batch_size}) must be divisible by "
        f"micro_batch_size ({args.micro_batch_size}) * world_size ({world_size})"
    )
    grad_accum_steps = args.global_batch_size // (args.micro_batch_size * world_size)
    if is_main:
        logger.info(
            f"Batch size: micro_batch_size={args.micro_batch_size}, "
            f"world_size={world_size}, grad_accum_steps={grad_accum_steps}, "
            f"global_batch_size={args.global_batch_size}"
        )

    # ── Load tokenizer ─────────────────────────────────────────────────────
    if is_main:
        logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load model ─────────────────────────────────────────────────────────
    if is_main:
        logger.info("Loading OpenPangu1B model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # ── Parse HiF8 amax schedules ──────────────────────────────────────────
    hif8_enable = args.use_hif8 and getattr(args, "hif8_enable", False)
    interval_segments = parse_schedule(args.hif8_interval_schedule)
    algo_segments = parse_schedule(args.hif8_amax_compute_algo_schedule)
    hif8_warmup_steps = getattr(args, "hif8_warmup_steps", 0)

    # ── Patch with HiF8 quantization (or skip for full BF16) ──────────────
    if args.use_hif8:
        if is_main:
            logger.info(f"HiF8 mode: patching mlp (gate/up/down_proj) only; attn kept in BF16 "
                        f"(keeping {args.high_precision_layers} blocks in high precision)...")
            if hif8_enable:
                logger.info(
                    f"  amax tracking ON | max_fwd={args.hif8_max_fwd} "
                    f"max_bwd={args.hif8_max_bwd} "
                    f"history_len={args.hif8_amax_history_len} "
                    f"interval_schedule='{args.hif8_interval_schedule}' "
                    f"algo_schedule='{args.hif8_amax_compute_algo_schedule}'"
                )
            if hif8_warmup_steps > 0:
                logger.info(f"  BF16 warmup: first {hif8_warmup_steps} steps run without HiF8 quantization")
        model = patch_model_with_hif8(model, high_precision_layers=args.high_precision_layers)
        # Delay quantization until after BF16 warmup (if configured)
        set_quantization_enabled(model, hif8_warmup_steps == 0)
    else:
        model._hif8_linears = []
        if is_main:
            logger.info("BF16 mode: HiF8 quantization disabled, training in full BF16")

    # ── DDP 模型包装 ───────────────────────────────────────────────────────
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ── Dataset & DataLoader（按 rank 分片）────────────────────────────────
    if is_main:
        if args.cache_dir:
            logger.info(f"Setting up FineWeb dataset (local cache: {args.cache_dir})...")
        else:
            logger.info("Setting up FineWeb streaming dataset (online)...")
    dataset = FineWebDataset(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        max_samples=args.max_samples,
        subset=args.fineweb_subset,
        rank=rank,
        world_size=world_size,
        cache_dir=args.cache_dir,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        num_workers=2,
        pin_memory=True,
        timeout=300,  # 300s 内取不到数据则抛异常，避免 NCCL 集合通信无限等待
    )

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    # ── Resume from checkpoint ─────────────────────────────────────────────
    global_step = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        model.module.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        global_step = ckpt["global_step"]
        if is_main:
            logger.info(f"Resumed from step {global_step}")
        # Restore correct quantization state for the resumed step
        if args.use_hif8:
            already_past_warmup = (global_step >= hif8_warmup_steps)
            set_quantization_enabled(model.module, already_past_warmup)
            if is_main and already_past_warmup:
                logger.info(f"  Resumed past warmup ({global_step} >= {hif8_warmup_steps}): HiF8 ON")

    # ── Training loop ──────────────────────────────────────────────────────
    if is_main:
        logger.info(f"Starting training: max_steps={args.max_steps}, "
                    f"micro_batch_size={args.micro_batch_size}, "
                    f"grad_accum_steps={grad_accum_steps}, "
                    f"global_batch_size={args.global_batch_size}")

    global_bs = args.global_batch_size
    tokens_per_step = global_bs * args.seq_len
    consumed_samples = global_step * global_bs
    consumed_tokens = global_step * tokens_per_step

    model.train()
    optimizer.zero_grad()
    running_loss = 0.0
    accum_count = 0
    step_start_time = torch.cuda.Event(enable_timing=True)
    step_end_time = torch.cuda.Event(enable_timing=True)
    if is_main:
        step_start_time.record()

    for batch in dataloader:
        if global_step >= args.max_steps:
            break

        raw_model = model.module

        # ── BF16 → HiF8 warmup transition ─────────────────────────────
        if (args.use_hif8
                and hif8_warmup_steps > 0
                and raw_model._hif8_linears
                and not raw_model._hif8_linears[0].quantized
                and global_step >= hif8_warmup_steps):
            HiF8GlobalStateManager.reset()   # fresh amax history after warmup
            set_quantization_enabled(raw_model, True)
            if is_main:
                logger.info(
                    f"Step {global_step}: BF16 warmup complete — "
                    f"HiF8 quantization enabled"
                )

        # ── Update per-step HiF8 amax config ──────────────────────────
        if hif8_enable and raw_model._hif8_linears and raw_model._hif8_linears[0].quantized:
            update_hif8_config(
                raw_model, step=global_step,
                max_fwd=args.hif8_max_fwd,
                max_bwd=args.hif8_max_bwd,
                amax_history_len=args.hif8_amax_history_len,
                interval_segments=interval_segments,
                algo_segments=algo_segments,
            )

        # ── Forward pass ───────────────────────────────────────────────
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / grad_accum_steps

        # ── Backward ───────────────────────────────────────────────────
        loss.backward()
        running_loss += loss.item() * grad_accum_steps
        accum_count += 1

        # ── Optimizer step ─────────────────────────────────────────────
        if accum_count == grad_accum_steps:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            accum_count = 0

            # ── All-reduce loss for accurate logging ────────────────────
            loss_tensor = torch.tensor(running_loss / grad_accum_steps,
                                       device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
            running_loss = 0.0

            if is_main:
                # ── Timing ─────────────────────────────────────────────
                step_end_time.record()
                torch.cuda.synchronize()
                elapsed_ms = step_start_time.elapsed_time(step_end_time)
                step_start_time.record()

                # ── Counters ───────────────────────────────────────────
                consumed_samples += global_bs
                consumed_tokens += tokens_per_step

                if global_step % args.log_interval == 0:
                    quant_status = "ON" if (raw_model._hif8_linears and
                                            raw_model._hif8_linears[0].quantized) else "OFF"
                    hif8_alg = get_schedule_value(algo_segments, global_step) if hif8_enable else "N/A"
                    logger.info(
                        f"step={global_step}/{args.max_steps} | "
                        f"loss={avg_loss:.4f} | "
                        f"lr={scheduler.get_last_lr()[0]:.2e} | "
                        f"quant={quant_status} | "
                        f"hif8_alg={hif8_alg} | "
                        f"samples={consumed_samples:,} | "
                        f"tokens={consumed_tokens/1e6:.2f}M | "
                        f"step_time={elapsed_ms/1000:.2f}s | "
                        f"toks/s={tokens_per_step/(elapsed_ms/1000):,.0f} | "
                        f"global_bs={global_bs}"
                    )

                # ── Checkpoint ─────────────────────────────────────────
                if global_step % args.save_interval == 0:
                    ckpt_path = Path(args.output_dir) / f"checkpoint-step{global_step}"
                    ckpt_path.mkdir(exist_ok=True)
                    torch.save(
                        {
                            "global_step": global_step,
                            "model_state_dict": raw_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                        },
                        ckpt_path / "training_state.pt",
                    )
                    _save_model_safe(raw_model, tokenizer, str(ckpt_path))
                    logger.info(f"Checkpoint saved to {ckpt_path}")

    # ── Final save (rank 0 only) ───────────────────────────────────────────
    if is_main:
        final_path = Path(args.output_dir) / "final"
        final_path.mkdir(exist_ok=True)
        _save_model_safe(model.module, tokenizer, str(final_path))
        logger.info(f"Training complete. Final model saved to {final_path}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
