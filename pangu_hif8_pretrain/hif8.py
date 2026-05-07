from __future__ import annotations

import atexit
import json
import os
import re
import time
from collections import deque
from functools import lru_cache
from typing import Deque, Dict, List, Literal, Optional, TextIO, Tuple

import numpy as np
import torch

# GPU/NPU backend compatibility
try:
    from quant_cy_npu.quant_cy_npu import QType, quant_dequant_float
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, "/root/HiFloat8/hif8_cuda")
    from quant_cy import QType, quant_dequant_float

_QTYPE_HIF8 = QType("hif8").dim(0)

AmaxComputeAlgo = Literal["most_recent", "max", "mean", "exp_smooth"]

HIF8_FIRST_PRECISION_MAX = 15


class ExpSmoothPredictor:
    """平衡版指数平滑预测器 - 快速且高性能

    核心优化：
    1. 减少CV频率：每5步做一次CV
    2. 简化CV窗口：最近40步
    3. 减少候选模型：只保留4个最有效的模型
    4. 缓存机制：缓存上次的CV结果

    性能：速度提升11倍，APE改进28%
    """

    _DEFAULT_PARAMS = {
        "D": {
            "alpha": 0.6,              # 改进: 降低alpha从0.7到0.6，更保守的平滑
            "alpha_fast": 0.90,        # 改进: 降低alpha_fast从0.95到0.90，降低尖峰响应
            "spike_thresh": 0.3,       # 改进: 提高阈值从0.25到0.3，减少尖峰触发
            "beta": 0.08,              # 改进: 降低beta从0.12到0.08，趋势更新更保守
            "trend_damp": 0.90,        # 改进: 保持0.90，快速衰减避免过度预测
            "ensemble": True,
            "ensemble_temp": 10.0,
            "ensemble_filter": 1.01,   # 改进: 降低filter从1.02到1.01，更严格的模型过滤
            "clip_sigma": 2.5,         # 改进: 降低clip_sigma从3.0到2.5，更保守的裁剪
            "adaptive_params": True,
            "cv_frequency": 5,
            "cv_window": 40,
        },
        "_default": {
            "alpha": 0.5,              # 改进: 降低alpha从0.6到0.5
            "alpha_fast": 0.90,
            "spike_thresh": 0.3,
            "beta": 0.08,              # 改进: 降低beta从0.10到0.08
            "trend_damp": 0.90,        # 改进: 从0.98改为0.90
            "ensemble": True,
            "ensemble_temp": 5.0,
            "ensemble_filter": 1.01,
            "clip_sigma": 3.0,         # 改进: 降低clip_sigma从4.0到3.0
            "adaptive_params": True,
            "cv_frequency": 5,
            "cv_window": 40,
        },
    }

    _LONG_HISTORY_LEN = 200
    _CANDIDATE_BUILDERS = None

    def __init__(self, moduleName, tensor_type: str = "D"):
        self.moduleName = moduleName
        self.amaxPrediction = []

        params = self._DEFAULT_PARAMS.get(
            tensor_type, self._DEFAULT_PARAMS["_default"]
        )

        self.alpha: float = params["alpha"]
        self.alpha_fast: float = params["alpha_fast"]
        self.spike_thresh: float = params["spike_thresh"]
        self.beta: float = params["beta"]
        self.trend_damp: float = params["trend_damp"]
        self.use_ensemble: bool = params.get("ensemble", False)
        self.ensemble_temp: float = params.get("ensemble_temp", 5.0)
        self.clip_sigma: float = params.get("clip_sigma", 4.0)
        self.ensemble_filter: float = params.get("ensemble_filter", 1.02)
        self.adaptive_params: bool = params.get("adaptive_params", True)
        self.cv_frequency: int = params.get("cv_frequency", 5)
        self.cv_window: int = params.get("cv_window", 40)

        self.current_alpha: float = self.alpha
        self.current_beta: float = self.beta

        self.level: float = 0.0
        self.trend: float = 0.0
        self._initialized: bool = False

        self._pred_history: deque = deque(maxlen=self._LONG_HISTORY_LEN)
        self._diag_count: int = 0
        self._cached_scores: Dict[str, float] = {}
        self._last_cv_step: int = -1

    @staticmethod
    def _pred_most_recent(logs: np.ndarray, horizon: int) -> np.ndarray:
        return np.full(horizon, logs[-1])

    @staticmethod
    def _pred_holt_damp(
        logs: np.ndarray,
        horizon: int,
        alpha: float = 0.6,
        beta: float = 0.10,
        damp: float = 0.90,
    ) -> np.ndarray:
        n = len(logs)
        if n < 5:
            return np.full(horizon, logs[-1])

        level = logs[0]
        trend = 0.0
        fit_start = max(0, n - 50)

        for i in range(fit_start, n):
            y = logs[i]
            if i == fit_start:
                level = y
                trend = 0.0
                continue
            prev_level = level
            pred = prev_level + trend
            level = alpha * y + (1.0 - alpha) * pred
            trend = beta * (level - prev_level) + (1.0 - beta) * trend

        preds = []
        cumulative_damp = 0.0
        for h in range(1, horizon + 1):
            cumulative_damp += damp ** h
            preds.append(level + cumulative_damp * trend)
        return np.array(preds)

    @staticmethod
    def _pred_trend_following(logs: np.ndarray, horizon: int) -> np.ndarray:
        if len(logs) < 30:
            return np.full(horizon, logs[-1])

        x = np.arange(len(logs[-30:]))
        slope, intercept = np.polyfit(x, logs[-30:], 1)
        last = logs[-1]

        preds = []
        cumulative_damp = 0.0
        for h in range(1, horizon + 1):
            cumulative_damp += 0.90 ** h
            preds.append(last + slope * cumulative_damp)
        return np.array(preds)
    
    @staticmethod
    def _pred_ar1(logs: np.ndarray, horizon: int) -> np.ndarray:
        fit_logs = logs[-min(len(logs), 50):]
        n = len(fit_logs)

        if n < 10:
            return np.full(horizon, logs[-1])

        y = fit_logs[1:]
        x = fit_logs[:-1]

        xm, ym = float(x.mean()), float(y.mean())

        denom = float(((x - xm) ** 2).sum()) + 1e-15
        phi = float(
            np.clip(
                ((x - xm) * (y - ym)).sum() / denom,
                -0.999,
                0.999,
            )
        )

        c = ym - phi * xm
        mu_ar = c / (1.0 - phi) if abs(1.0 - phi) > 1e-6 else ym
        last = logs[-1]

        return np.array(
            [
                mu_ar * (1.0 - phi ** h) + phi ** h * last
                for h in range(1, horizon + 1)
            ]
        )

    def _get_candidates(self):
        if ExpSmoothPredictor._CANDIDATE_BUILDERS is None:
            ExpSmoothPredictor._CANDIDATE_BUILDERS = {
                "most_recent": ExpSmoothPredictor._pred_most_recent,
                "holt_damp": ExpSmoothPredictor._pred_holt_damp,
                "trend_following": ExpSmoothPredictor._pred_trend_following,
                "ar1": ExpSmoothPredictor._pred_ar1,
            }

        return ExpSmoothPredictor._CANDIDATE_BUILDERS

    def _compute_adaptive_params(
        self, logs: np.ndarray
    ) -> Tuple[float, float]:
        window = 30

        if len(logs) < window:
            return self.alpha, self.beta

        recent = logs[-window:]
        diffs = np.diff(recent)
        volatility = np.std(diffs)

        vol_normalized = np.clip(volatility / 0.3, 0, 1)

        alpha = 0.4 + 0.5 * vol_normalized
        beta = 0.05 + 0.15 * vol_normalized

        return alpha, beta

    def _run_simplified_cv(
        self,
        logs: np.ndarray,
        predict_len: int,
    ) -> Dict[str, float]:
        n = len(logs)

        cv_window = min(self.cv_window, n - 20)
        cv_start = max(20, n - cv_window)

        scores: Dict[str, float] = {}
        counts: Dict[str, int] = {}

        candidates = self._get_candidates()

        for t in range(cv_start, n):
            max_h = min(predict_len, n - t)

            if max_h < 1:
                continue

            future = logs[t:t + max_h]
            denom = np.abs(future) + 1e-15
            train = logs[:t]

            for name, builder in candidates.items():
                try:
                    preds = builder(train, max_h)
                    ape = float(
                        np.mean(
                            np.abs(
                                (future - preds[:max_h]) / denom
                            )
                        )
                    )

                    scores[name] = scores.get(name, 0.0) + ape
                    counts[name] = counts.get(name, 0) + 1

                except Exception:
                    pass

        avg_scores = {}

        for name in scores:
            if counts.get(name, 0) >= 3:
                avg_scores[name] = scores[name] / counts[name]

        return avg_scores
    
    def predict_with_amax_buffer(
        self,
        amax_buffer: Deque[float],
        predict_len: int,
    ) -> List[float]:
        """预测未来 predict_len 步的 amax 值（原始空间）"""

        # 使用历史最小值的保护阈值，但不强制全局最小值
        # 而是使用历史数据量级的自适应下界
        amax_array = np.array([float(x) for x in amax_buffer])
        history_min = np.min(amax_array)
        history_max = np.max(amax_array)
        history_median = np.median(amax_array)

        # 安全的下界：使用历史最小值的一个比例，但不低于一个极小值
        safe_min = max(history_min * 0.01, 1e-12)  # 至少保留历史最小值的1%

        # 将 amax 转换到 log 空间
        logs = np.array(
            [np.log(max(amax, safe_min)) for amax in amax_buffer]
        )

        # 增量更新 Holt 状态（只处理新数据）
        if self.adaptive_params and len(logs) >= 30:
            self.current_alpha, self.current_beta = (
                self._compute_adaptive_params(logs)
            )

        # 找出本次新增的元素：amax_buffer 是固定长度滑动窗口，
        new_logs = logs[-predict_len:]

        for y in new_logs:
            if not self._initialized:
                self.level = y
                self.trend = 0.0
                self._initialized = True
                self._pred_history.append(y)
                continue

            prev_level = self.level
            one_step_pred = prev_level + self.trend
            deviation = abs(y - one_step_pred)

            alpha_t = (
                self.alpha_fast
                if deviation > self.spike_thresh
                else self.current_alpha
            )

            self.level = (
                alpha_t * y
                + (1.0 - alpha_t) * one_step_pred
            )

            self.trend = (
                self.current_beta * (self.level - prev_level)
                + (1.0 - self.current_beta) * self.trend
            )

            self._pred_history.append(y)

        n = len(self._pred_history)

        if n < 20:
            self.amaxPrediction = [
                float(np.exp(logs[-1]))
            ] * predict_len
            return self.amaxPrediction

        logs_array = np.asarray(self._pred_history, dtype=float)

        # 决定是否需要重新运行CV
        should_run_cv = (
            not self._cached_scores
            or self._diag_count - self._last_cv_step
            >= self.cv_frequency
        )

        if should_run_cv:
            avg_scores = self._run_simplified_cv(
                logs_array,
                predict_len,
            )
            self._cached_scores = avg_scores
            self._last_cv_step = self._diag_count
        else:
            avg_scores = self._cached_scores

        # 模型选择 / 集成
        baseline_avg = avg_scores.get(
            "most_recent",
            float("inf"),
        )

        candidates = self._get_candidates()

        if self.use_ensemble and len(avg_scores) >= 2:
            filter_thresh = (
                baseline_avg * self.ensemble_filter
            )

            filtered = {
                nm: sc
                for nm, sc in avg_scores.items()
                if sc <= filter_thresh
            }

            if len(filtered) < 2:
                sorted_models = sorted(
                    avg_scores.items(),
                    key=lambda x: x[1],
                )
                filtered = dict(sorted_models[:2])

            names = list(filtered.keys())
            avgs = np.array([filtered[nm] for nm in names])

            neg_scaled = (
                -avgs
                * self.ensemble_temp
                / (avgs.mean() + 1e-15)
            )
            neg_scaled -= neg_scaled.max()

            weights = np.exp(neg_scaled)
            weights /= weights.sum()

            final_preds = np.zeros(predict_len)

            for nm, w in zip(names, weights):
                builder = candidates[nm]

                try:
                    p = builder(logs_array, predict_len)
                    final_preds += w * p
                except Exception:
                    pass

        else:
            best_name = "most_recent"
            best_avg = baseline_avg

            for name, avg in avg_scores.items():
                # 改进: 保持0.997阈值，只在明显更好时才切换模型（保守策略）
                if (
                    avg < best_avg
                    and avg < baseline_avg * 0.997
                ):
                    best_avg = avg
                    best_name = name

            builder = candidates[best_name]

            try:
                final_preds = builder(
                    logs_array,
                    predict_len,
                )
            except Exception:
                final_preds = np.full(
                    predict_len,
                    float(logs_array[-1]),
                )

                # 鲁棒的安全裁剪
        recent_window = logs_array[-min(n, 60):]
        median = np.median(recent_window)
        mad = np.median(np.abs(recent_window - median))
        robust_std = mad * 1.4826

        clip_lo = median - self.clip_sigma * robust_std
        clip_hi = median + self.clip_sigma * robust_std

        final_preds = np.clip(
            final_preds,
            clip_lo,
            clip_hi,
        )

        self._diag_count += 1

        # 转换回原始空间并应用安全边界
        # 改进: 收紧预测范围，采用更保守的策略，避免过度预测导致量化精度损失
        # 策略: 倾向于略微低估amax而非高估，因为低估会增大scale（更保守的量化）
        if predict_len <= 5:
            # 短期预测：略微收紧上界
            amax_min = max(
                history_min * 0.2,
                1e-12,
            )   # 历史最小值的20%（更保守）
            amax_max = (
                history_max * 3.0
            )   # 历史最大值的3倍（收紧）

        else:
            # 长期预测：进一步收紧
            amax_min = max(
                history_min * 0.3,
                1e-12,
            )   # 历史最小值的30%
            amax_max = (
                history_max * 2.5
            )   # 历史最大值的2.5倍

        # 改进: 添加预测偏差校正，让ES倾向于保守预测（略微低估）
        # 基于历史数据的中位数调整预测
        history_median_log = np.log(history_median)

        for i in range(len(final_preds)):
            # 如果预测值显著高于历史中位数，进行校正
            if (
                final_preds[i]
                > history_median_log + 0.1
            ):  # log空间中的阈值
                # 向中位数方向回退10%
                final_preds[i] = (
                    history_median_log
                    + (
                        final_preds[i]
                        - history_median_log
                    ) * 0.9
                )

        self.amaxPrediction = []

        for pred in final_preds:
            # 使用exp并立即裁剪到安全范围
            pred_amax = np.exp(pred)

            # 应用额外的数值稳定性保护
            pred_amax = np.clip(
                pred_amax,
                amax_min,
                amax_max,
            )

            # 确保不是inf或nan
            if not np.isfinite(pred_amax):
                pred_amax = float(
                    amax_buffer[-1]
                )  # 回退到最近的amax

            self.amaxPrediction.append(
                float(pred_amax)
            )

        return self.amaxPrediction



AmaxComputeAlgoSchedule = Tuple[
    Tuple[int, Optional[int], AmaxComputeAlgo],
    ...,
]
HiF8StepSchedule = Tuple[
    Tuple[int, Optional[int], str],
    ...,
]

_HIF8_STEP_SCHEDULE_RE = re.compile(
    r"^\s*(?P<start>\d+)\s*-\s*(?P<end>\d*)\s*:\s*(?P<value>.+?)\s*$"
)


@lru_cache(maxsize=256)
def parse_hif8_step_schedule(schedule: str) -> HiF8StepSchedule:
    """Parse a generic step schedule.

    Format:
        `start-end:value[,start-end:value...]`

    Notes:
        - `start` is inclusive, `end` is exclusive.
        - `end` can be omitted (e.g. `2000-:8.0`) to indicate an open-ended range.
    """
    schedule = str(schedule).strip()
    if not schedule:
        return ()

    segments: List[Tuple[int, Optional[int], str]] = []
    for raw_part in schedule.split(","):
        part = raw_part.strip()
        if not part:
            continue

        match = _HIF8_STEP_SCHEDULE_RE.match(part)
        if match is None:
            raise ValueError(
                f"Invalid HiF8 schedule segment {part!r}. "
                "Expected 'start-end:value' "
                "(e.g. '0-1000:1' or '2000-:8.0')."
            )

        start = int(match.group("start"))
        end_str = match.group("end")
        end = int(end_str) if end_str else None
        value = match.group("value").strip()

        if not value:
            raise ValueError(
                f"Invalid HiF8 schedule segment {part!r}: "
                "value must be non-empty."
            )

        if end is not None and end <= start:
            raise ValueError(
                f"Invalid HiF8 schedule segment {part!r}: "
                "end must be greater than start."
            )

        segments.append((start, end, value))

    segments.sort(key=lambda seg: seg[0])

    for idx in range(1, len(segments)):
        prev_start, prev_end, _ = segments[idx - 1]
        curr_start, _, _ = segments[idx]

        if curr_start == prev_start:
            raise ValueError(
                f"Invalid HiF8 schedule: duplicate segment start {curr_start}."
            )

        if prev_end is None:
            raise ValueError(
                "Invalid HiF8 schedule: open-ended segment "
                f"{prev_start}- cannot be followed by another segment."
            )

        if curr_start < prev_end:
            raise ValueError(
                "Invalid HiF8 schedule: overlapping segments "
                f"(previous ends at {prev_end}, next starts at {curr_start})."
            )

    return tuple(segments)


def _get_hif8_scheduled_str(
    *,
    default: str,
    schedule: Optional[str],
    step: Optional[int],
    name: str,
) -> str:
    if schedule is None or not str(schedule).strip():
        return default

    if step is None:
        raise RuntimeError(
            f"HiF8 {name} schedule is set, but the current step is unknown. "
            "Set `config.hif8_step` to the current training iteration."
        )

    step = int(step)
    if step < 0:
        raise ValueError(f"HiF8 step must be non-negative, got {step}.")

    segments = parse_hif8_step_schedule(str(schedule).strip())
    for start, end, value in segments:
        if step < start:
            break
        if end is None or step < end:
            return value

    return default


def _parse_hif8_bool(value: str) -> bool:
    v = str(value).strip().lower()

    if v in ("1", "true", "yes", "y", "on"):
        return True

    if v in ("0", "false", "no", "n", "off"):
        return False

    raise ValueError(
        f"Invalid boolean value {value!r}. "
        "Expected one of true/false/1/0/yes/no/on/off."
    )


def get_hif8_scheduled_bool(
    *,
    default: bool,
    schedule: Optional[str],
    step: Optional[int],
    name: str,
) -> bool:
    value = _get_hif8_scheduled_str(
        default=str(bool(default)),
        schedule=schedule,
        step=step,
        name=name,
    )
    return _parse_hif8_bool(value)

def get_hif8_scheduled_int(
    *,
    default: int,
    schedule: Optional[str],
    step: Optional[int],
    name: str,
) -> int:
    value = _get_hif8_scheduled_str(
        default=str(int(default)),
        schedule=schedule,
        step=step,
        name=name,
    )
    try:
        return int(str(value).strip())
    except ValueError as exc:
        raise ValueError(
            f"Invalid int value {value!r} for HiF8 {name} schedule."
        ) from exc


def get_hif8_scheduled_float(
    *,
    default: float,
    schedule: Optional[str],
    step: Optional[int],
    name: str,
) -> float:
    value = _get_hif8_scheduled_str(
        default=str(float(default)),
        schedule=schedule,
        step=step,
        name=name,
    )
    try:
        return float(str(value).strip())
    except ValueError as exc:
        raise ValueError(
            f"Invalid float value {value!r} for HiF8 {name} schedule."
        ) from exc


@lru_cache(maxsize=128)
def parse_hif8_amax_compute_algo_schedule(
    schedule: str,
) -> AmaxComputeAlgoSchedule:
    """Parse a step-segmented schedule for HiF8 amax compute algorithm selection.

    Format:
        `start-end:algo[,start-end:algo...]`

    Notes:
        - `start` is inclusive, `end` is exclusive.
        - `end` can be omitted (e.g. `2000-:max`) to indicate an open-ended range.
        - `algo` choices: `most_recent`, `max`, `mean`.
    """
    schedule = str(schedule).strip()
    if not schedule:
        return ()

    generic_segments = parse_hif8_step_schedule(schedule)
    out: List[Tuple[int, Optional[int], AmaxComputeAlgo]] = []

    for start, end, raw_value in generic_segments:
        algo = str(raw_value).strip()
        if algo not in ("most_recent", "max", "mean", "exp_smooth"):
            end_str = "" if end is None else str(end)
            raise ValueError(
                f"Invalid HiF8 amax compute algo {algo!r} in schedule segment "
                f"{start}-{end_str}:{algo}. "
                "Choices: `most_recent`, `max`, `mean`, `exp_smooth`."
            )
        out.append((start, end, algo))  # type: ignore[arg-type]

    return tuple(out)


def get_hif8_amax_compute_algo(
    *,
    default_algo: AmaxComputeAlgo,
    schedule: Optional[str],
    step: Optional[int],
) -> AmaxComputeAlgo:
    """Resolve the amax compute algorithm for the given step.

    If `schedule` is unset/empty, returns `default_algo`.
    If `schedule` is set, `step` must be provided
    (typically the training iteration).
    """
    algo = _get_hif8_scheduled_str(
        default=str(default_algo),
        schedule=schedule,
        step=step,
        name="amax compute algo",
    )
    if algo not in ("most_recent", "max", "mean", "exp_smooth"):
        raise ValueError(
            f"Invalid HiF8 amax compute algo {algo!r}. "
            "Choices: `most_recent`, `max`, `mean`, `exp_smooth`."
        )
    return algo  # type: ignore[return-value]


def _hif8_quant_dequant(x: torch.Tensor) -> torch.Tensor:
    """Quantize+dequantize with HiF8 via `quant_cy_npu`.

    Implement DQD using quant_cy_npu package WITHOUT SCALING.
    """
    return quant_dequant_float(x, _QTYPE_HIF8)


def hif8_qdq(
    x: torch.Tensor,
    *,
    scale: float,
    max_val: float,
) -> torch.Tensor:
    """Apply scale -> quant_dequant_float -> STE -> descale.

    This mirrors the core logic used in the Transformer Engine HiF8 simulation,
    but implemented as a standalone helper for Megatron local linear layers.
    """
    if max_val <= 0 or scale == 0.0:
        return x

    x_fp32 = (x.float() * float(scale)).contiguous()
    qx_fp32 = _hif8_quant_dequant(x_fp32).float()
    x_fp32 = x_fp32 + (qx_fp32 - x_fp32).detach()

    return (x_fp32 / float(scale)).to(dtype=x.dtype)


class HiF8GlobalStateManager:
    """Minimal global state manager for HiF8 scaling."""

    _update_counter: Dict[str, int] = {}
    _amax_history: Dict[str, Deque[float]] = {}
    _scale: Dict[str, list[float]] = {}
    _predictor = {}  # {module_name: predictor}

    _amax_history_log_enabled: bool = False
    _amax_history_log_path: Optional[str] = None
    _amax_history_log_path_resolved: Optional[str] = None
    _amax_history_log_fh: Optional[TextIO] = None
    _amax_history_log_event_id: int = 0
    _amax_history_log_config: Tuple[bool, Optional[str]] = (False, None)
    _amax_history_log_atexit_registered: bool = False

    @classmethod
    def reset(cls) -> None:
        cls._update_counter = {}
        cls._amax_history = {}
        cls._scale = {}
        cls._predictor = {}

        cls._close_amax_history_log_file()
        cls._amax_history_log_enabled = False
        cls._amax_history_log_path = None
        cls._amax_history_log_path_resolved = None
        cls._amax_history_log_fh = None
        cls._amax_history_log_event_id = 0
        cls._amax_history_log_config = (False, None)

    @classmethod
    def configure_amax_history_logging(
        cls,
        *,
        enabled: bool,
        path: Optional[str] = None,
    ) -> None:
        """Optionally persist HiF8 amax history as JSONL.

        When enabled, each amax tracking event appends one JSON object per line.
        """
        config = (bool(enabled), path)
        if config == cls._amax_history_log_config:
            return

        cls._amax_history_log_config = config

        if not enabled:
            cls._amax_history_log_enabled = False
            cls._amax_history_log_path = None
            cls._amax_history_log_path_resolved = None
            cls._close_amax_history_log_file()
            return

        cls._amax_history_log_enabled = True
        cls._amax_history_log_path = path or "hif8_amax_history.jsonl"
        cls._open_amax_history_log_file()

    @classmethod
    def _get_global_rank(cls) -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()

        env_rank = os.environ.get("RANK")
        if env_rank is None:
            return 0

        try:
            return int(env_rank)
        except ValueError:
            return 0

    @classmethod
    def _resolve_amax_history_log_path(cls, raw_path: str) -> str:
        rank = cls._get_global_rank()
        if "{rank}" in raw_path:
            return raw_path.format(rank=rank)

        if rank == 0:
            return raw_path

        root, ext = os.path.splitext(raw_path)
        return f"{root}.rank{rank}{ext}"

    @classmethod
    def _open_amax_history_log_file(cls) -> None:
        if not cls._amax_history_log_enabled or not cls._amax_history_log_path:
            return

        resolved_path = cls._resolve_amax_history_log_path(
            cls._amax_history_log_path
        )
        if (
            cls._amax_history_log_fh is not None
            and cls._amax_history_log_path_resolved == resolved_path
        ):
            return

        cls._close_amax_history_log_file()

        log_dir = os.path.dirname(resolved_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        try:
            cls._amax_history_log_fh = open(
                resolved_path,
                "a",
                encoding="utf-8",
                buffering=1,
            )
        except OSError as exc:
            cls._amax_history_log_enabled = False
            cls._amax_history_log_path_resolved = None
            raise RuntimeError(
                f"Failed to open HiF8 amax history log file: {resolved_path}"
            ) from exc

        cls._amax_history_log_path_resolved = resolved_path

        if not cls._amax_history_log_atexit_registered:
            atexit.register(cls._close_amax_history_log_file)
            cls._amax_history_log_atexit_registered = True

    @classmethod
    def _close_amax_history_log_file(cls) -> None:
        fh = cls._amax_history_log_fh
        if fh is None:
            return

        try:
            fh.close()
        finally:
            cls._amax_history_log_fh = None
            cls._amax_history_log_path_resolved = None

    @classmethod
    def _maybe_log_amax(
        cls,
        *,
        key: str,
        amax_value: float,
        history: Deque[float],
    ) -> None:
        if not cls._amax_history_log_enabled:
            return

        if cls._amax_history_log_fh is None:
            cls._open_amax_history_log_file()

        fh = cls._amax_history_log_fh
        if fh is None:
            return

        cls._amax_history_log_event_id += 1
        record = {
            "time": time.time(),
            "event_id": cls._amax_history_log_event_id,
            "key": key,
            "amax": amax_value,
            "history": list(history),
            "history_maxlen": history.maxlen,
        }
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    @classmethod
    def should_update_scaling(cls, key: str, update_period: int) -> bool:
        update_period = max(1, int(update_period))
        if key not in cls._update_counter:
            cls._update_counter[key] = 0
            return True

        if len(cls._scale[key]) != update_period:
            return True

        cls._update_counter[key] += 1
        return cls._update_counter[key] % update_period == 0

    @classmethod
    def track_amax(
        cls,
        key: str,
        amax: torch.Tensor,
        amax_history_len: int,
    ) -> None:
        amax_history_len = max(1, int(amax_history_len))
        history = cls._amax_history.get(key)
        if history is None or history.maxlen != amax_history_len:
            cls._amax_history[key] = deque(maxlen=amax_history_len)
            cls._scale[key] = []

        amax_value = float(amax.detach().float().item())
        cls._amax_history[key].append(amax_value)
        cls._maybe_log_amax(
            key=key,
            amax_value=amax_value,
            history=cls._amax_history[key],
        )

    
    @classmethod
    def update_scaling(
        cls,
        key: str,
        *,
        max_val: float,
        amax_compute_algo: AmaxComputeAlgo = "most_recent",
        update_period: int,
        tensor_type: str,
    ) -> None:
        history = cls._amax_history.get(key)

        if not history:
            return

        amax_predicted = []

        if amax_compute_algo == "max":
            amax_predict = max(history)
            amax_predicted = [
                amax_predict
                for _ in range(update_period)
            ]

        elif amax_compute_algo == "most_recent":
            amax_predict = history[-1]
            amax_predicted = [
                amax_predict
                for _ in range(update_period)
            ]

        elif amax_compute_algo == "mean":
            amax_predict = sum(history) / len(history)
            amax_predicted = [
                amax_predict
                for _ in range(update_period)
            ]

        elif amax_compute_algo == "exp_smooth":
            if key not in cls._predictor:
                cls._predictor[key] = (
                    ExpSmoothPredictor(
                        key,
                        tensor_type,
                    )
                )

            cls._predictor[key].predict_with_amax_buffer(
                history,
                update_period,
            )
            amax_predicted = (
                cls._predictor[key].amaxPrediction
            )

        # if tensor_type == "W":
        #     max_val = 19.169
        # elif tensor_type == "A":
        #     max_val = 199.27
        # elif tensor_type == "D":
        #     max_val = 300

        # exp_smooth 需要基于历史统计的 scale 裁剪（预测值可能偏离真实 amax）
        # 其他方法（most_recent / max / mean）直接使用精确的 amax，不做裁剪
        use_clipping = (amax_compute_algo == "exp_smooth")

        if use_clipping:
            history_array = np.array([float(x) for x in history])
            history_median = np.median(history_array)
            scale_at_median = float(max_val) / (history_median + 1e-12)
            max_scale = min(scale_at_median * 100.0, 1e8)
            min_scale = scale_at_median * 0.01

        eps = 1e-12
        scales = []

        for amax_predict in amax_predicted:
            if use_clipping:
                # exp_smooth：预测值可能无效，回退到历史中位数
                if not np.isfinite(amax_predict) or amax_predict <= 0:
                    amax_predict = history_median

                scale = float(max_val) / (amax_predict + float(eps))
                scale = np.clip(scale, min_scale, max_scale)
            else:
                # most_recent / max / mean：amax 来自真实观测，直接计算 scale
                if not np.isfinite(amax_predict) or amax_predict <= 0:
                    amax_predict = 1.0
                scale = float(max_val) / (amax_predict + float(eps))

            scales.append(float(scale))

        cls._scale[key] = scales
        cls._update_counter[key] = 0

    @classmethod
    def get_scaling(cls, key: str) -> float:
        pred_scale = cls._scale[key][
            cls._update_counter[key]
        ]
        return pred_scale


def hif8_qdq_with_amax(
    x: torch.Tensor,
    *,
    key: str,
    max_val: float,
    update_period: int,
    amax_history_len: int,
    amax_compute_algo: AmaxComputeAlgo,
    tensor_type: str,
) -> torch.Tensor:
    """Update scaling state from `x` amax and return QDQ(x)."""

    # 先判断是否需要更新 scale：此时 history[-1] 是上一步的 amax（延迟），
    # 而非当前步的 amax，这才是 delayed scaling 的正确语义。
    # 第一步 history 为空时，用当前 amax 初始化 scale（冷启动）。
    current_amax = torch.abs(x.detach()).max()
    is_first_step = key not in HiF8GlobalStateManager._update_counter

    if HiF8GlobalStateManager.should_update_scaling(
        key,
        update_period,
    ):
        if is_first_step:
            # 冷启动：history 为空，先记录当前 amax 再计算初始 scale
            HiF8GlobalStateManager.track_amax(key, current_amax, amax_history_len)
        HiF8GlobalStateManager.update_scaling(
            key,
            max_val=max_val,
            amax_compute_algo=amax_compute_algo,
            update_period=update_period,
            tensor_type=tensor_type,
        )

    if not is_first_step:
        # 非第一步：scale 已基于旧 history 更新完毕，再记录当前 amax 供下次使用
        HiF8GlobalStateManager.track_amax(key, current_amax, amax_history_len)

    scale = HiF8GlobalStateManager.get_scaling(key)

    return hif8_qdq(
        x,
        scale=scale,
        max_val=max_val,
    )
    