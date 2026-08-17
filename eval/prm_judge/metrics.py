from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isnan
from statistics import mean, median
from typing import Any, Callable

DEFAULT_DRAWDOWN_TOLERANCE = 1e-12

BINARY_RATE_METRICS = ("M25", "M50", "M75", "SR")
CONTINUOUS_MODEL_METRICS = (
    "MP",
    "PPL",
    "CRA",
    "STR",
    "DRR",
    "FNS",
    "SQS",
)
MODEL_SUMMARY_METRICS = (*BINARY_RATE_METRICS, *CONTINUOUS_MODEL_METRICS)
LEGACY_METRIC_ALIASES = {"MP": ("MaxP",)}


@dataclass
class MetricConfig:
    stagnation_delta: float = 0.005
    success_threshold: float = 0.99
    success_source: str = "progress"
    fns_weights: tuple[float, float, float] = (0.5, 0.3, 0.2)
    sqs_weights: tuple[float, float, float] = (0.5, 0.3, 0.2)
    drr_epsilon: float = 1e-6
    drawdown_tolerance: float = DEFAULT_DRAWDOWN_TOLERANCE

    def as_dict(self) -> dict[str, object]:
        return {
            "stagnation_delta": self.stagnation_delta,
            "success_threshold": self.success_threshold,
            "success_source": self.success_source,
            "fns_weights": list(self.fns_weights),
            "sqs_weights": list(self.sqs_weights),
            "drr_epsilon": self.drr_epsilon,
            "drawdown_tolerance": self.drawdown_tolerance,
        }


def _safe_mean(values: list[float]) -> float:
    valid = [value for value in values if not isnan(value)]
    return mean(valid) if valid else 0.0


def _safe_median(values: list[float]) -> float:
    """忽略 NaN 后计算中位数，空适用子集定义为零。"""
    valid = [value for value in values if not isnan(value)]
    return median(valid) if valid else 0.0


def metric_value(metrics: dict[str, Any], name: str, default: Any = None) -> Any:
    """读取规范指标值，并兼容历史结果中的旧字段名。"""
    if name in metrics:
        return metrics[name]
    for alias in LEGACY_METRIC_ALIASES.get(name, ()):
        if alias in metrics:
            return metrics[alias]
    return default


def first_index_of_max(values: list[float]) -> int:
    if not values:
        return -1
    max_value = max(values)
    return values.index(max_value)


def trace_metrics(progress: list[float], config: MetricConfig | None = None) -> dict[str, float]:
    """计算归一化进度曲线的逐轨迹指标。"""
    cfg = config or MetricConfig()
    if not progress:
        return {
            "M25": 0.0,
            "M50": 0.0,
            "M75": 0.0,
            "SR": 0.0,
            "MP": 0.0,
            "PPL": 0.0,
            "CRA": 0.0,
            "STR": 0.0,
            "DRR": 1.0,
        }

    mp = max(progress)
    series = [0.0] + progress
    path_length = sum(abs(series[idx] - series[idx - 1]) for idx in range(1, len(series)))
    ppl = (mp * mp / path_length) if path_length > 1e-9 else 0.0

    regret_sum = 0.0
    current_max = 0.0
    stagnation_count = 0
    for idx in range(1, len(series)):
        current_max = max(current_max, series[idx])
        regret_sum += max(0.0, current_max - series[idx])
        if abs(series[idx] - series[idx - 1]) < cfg.stagnation_delta:
            stagnation_count += 1
    drr = drawdown_recovery_ratio(
        progress,
        epsilon=cfg.drr_epsilon,
        tolerance=cfg.drawdown_tolerance,
    )

    raw_cra = regret_sum / len(progress)
    cra = raw_cra / mp if mp > 0.0 else 0.0
    return {
        "M25": 1.0 if mp >= 0.25 else 0.0,
        "M50": 1.0 if mp >= 0.50 else 0.0,
        "M75": 1.0 if mp >= 0.75 else 0.0,
        "SR": 1.0 if mp >= cfg.success_threshold else 0.0,
        "MP": mp,
        "PPL": max(0.0, min(1.0, ppl)),
        "CRA": cra,
        "STR": stagnation_count / len(progress),
        "DRR": drr,
    }


def recompute_record_metrics(
    records: list[dict[str, Any]],
    config: MetricConfig | None = None,
) -> list[dict[str, Any]]:
    """用保存的处理后进度曲线原地刷新 episode 指标字段。"""
    cfg = config or MetricConfig()
    for record in records:
        if record.get("status") != "ok":
            continue
        progress = record.get("progress_processed")
        if isinstance(progress, list):
            record["metrics"] = trace_metrics([float(value) for value in progress], cfg)
    return records


def has_drawdown(
    progress: list[float],
    tolerance: float = DEFAULT_DRAWDOWN_TOLERANCE,
) -> bool:
    """判断处理后进度是否相对历史峰值发生超过数值容差的回退。"""
    if not progress:
        return False

    peak_value = progress[0]
    for value in progress[1:]:
        if peak_value - value > tolerance:
            return True
        peak_value = max(peak_value, value)
    return False


def drawdown_recovery_ratio(
    progress: list[float],
    epsilon: float = 1e-6,
    tolerance: float = DEFAULT_DRAWDOWN_TOLERANCE,
) -> float:
    """计算最大回退后的恢复比率，容差只用于判定真实回退。"""
    if not progress:
        return 1.0

    peak_value = progress[0]
    max_drawdown = 0.0
    trough_idx = 0

    for idx, value in enumerate(progress):
        if value > peak_value:
            peak_value = value
        drop = peak_value - value
        if drop > max_drawdown:
            max_drawdown = drop
            trough_idx = idx

    if max_drawdown <= tolerance:
        return 1.0
    recovered = max(progress[trough_idx:]) - progress[trough_idx]
    drr = min(1.0, (recovered + epsilon) / (max_drawdown + epsilon))
    return max(0.0, drr)


def _model_metric_values(
    items: list[dict[str, Any]],
    cfg: MetricConfig,
) -> dict[str, list[float]]:
    """按指标收集模型内 episode 值，条件指标先筛选其适用轨迹。"""
    metrics = [item["metrics"] for item in items]
    traces = [trace_summary(item, cfg) for item in items]
    successful_indices = [index for index, trace in enumerate(traces) if trace["SR_trace"] == 1.0]
    drawdown_indices = [
        index
        for index, item in enumerate(items)
        if record_has_drawdown(item, cfg)
    ]
    values = {
        name: [float(metric_value(metric, name, 0.0)) for metric in metrics]
        for name in ["M25", "M50", "M75", "MP", "PPL", "CRA", "STR"]
    }
    values["SR"] = [trace["SR_trace"] for trace in traces]
    values["DRR"] = [traces[index]["DRR_trace"] for index in drawdown_indices]
    values["FNS"] = [trace["FNS_trace"] for trace in traces if trace["SR_trace"] == 0.0]
    values["SQS"] = [traces[index]["SQS_trace"] for index in successful_indices]
    return values


def record_has_drawdown(item: dict[str, Any], cfg: MetricConfig) -> bool:
    """优先用保存轨迹判定真实回退，并兼容只有旧指标的历史记录。"""
    for progress_key in ("progress_processed", "progress_raw"):
        progress = item.get(progress_key)
        if isinstance(progress, list):
            return has_drawdown([float(value) for value in progress], cfg.drawdown_tolerance)

    metrics = item.get("metrics") or {}
    try:
        return float(metrics.get("DRR", 1.0)) < 1.0 - cfg.drawdown_tolerance
    except (TypeError, ValueError):
        return False


def _summarize_items(items: list[dict[str, Any]], cfg: MetricConfig) -> dict[str, float]:
    """按 episode 等权汇总，并在适用结果子集内计算条件指标。"""
    values = _model_metric_values(items, cfg)
    return {
        name: _safe_mean(values[name])
        for name in [*BINARY_RATE_METRICS, *CONTINUOUS_MODEL_METRICS]
    }


def _summarize_items_median(
    items: list[dict[str, Any]],
    cfg: MetricConfig,
) -> dict[str, float]:
    """模型级中位数汇总；二值率保留原比例，条件指标先筛适用轨迹。"""
    values = _model_metric_values(items, cfg)
    summary = {name: _safe_mean(values[name]) for name in BINARY_RATE_METRICS}
    summary.update({name: _safe_median(values[name]) for name in CONTINUOUS_MODEL_METRICS})
    return summary


def aggregate_metrics(records: list[dict[str, Any]], config: MetricConfig | None = None) -> dict[str, Any]:
    """Aggregate task summaries and episode-weighted model leaderboards."""
    cfg = config or MetricConfig()
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_model: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("status") != "ok":
            continue
        benchmark = str(record.get("benchmark", "manifest"))
        model = str(record.get("model", "default"))
        task = str(record.get("task_name") or record.get("case_task") or "all")
        groups[(benchmark, model, task)].append(record)
        by_model[(benchmark, model)].append(record)

    rows: list[dict[str, Any]] = []
    for (benchmark, model, task), items in sorted(groups.items()):
        row: dict[str, Any] = {
            "benchmark": benchmark,
            "model": model,
            "task": task,
            "num_cases": len(items),
        }
        row.update(_summarize_items(items, cfg))
        rows.append(row)

    leaderboard: list[dict[str, Any]] = []
    for (benchmark, model), items in sorted(by_model.items()):
        leaderboard_row: dict[str, Any] = {
            "benchmark": benchmark,
            "model": model,
            "num_tasks": len(
                {
                    str(item.get("task_name") or item.get("case_task") or "all")
                    for item in items
                }
            ),
            "num_cases": len(items),
        }
        leaderboard_row.update(_summarize_items(items, cfg))
        leaderboard.append(leaderboard_row)

    return {"groups": rows, "leaderboard": leaderboard, "config": cfg.as_dict()}


def _aggregate_model_metrics_with(
    records: list[dict[str, Any]],
    cfg: MetricConfig,
    summarizer: Callable[[list[dict[str, Any]], MetricConfig], dict[str, float]],
) -> dict[str, Any]:
    """使用给定统计函数按模型聚合有效 episode。"""
    by_model: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("status") != "ok":
            continue
        benchmark = str(record.get("benchmark", "manifest"))
        model = str(record.get("model", "default"))
        by_model[(benchmark, model)].append(record)

    leaderboard: list[dict[str, Any]] = []
    for (benchmark, model), items in sorted(by_model.items()):
        row: dict[str, Any] = {
            "benchmark": benchmark,
            "model": model,
            "num_tasks": len(
                {
                    str(item.get("task_name") or item.get("case_task") or "all")
                    for item in items
                }
            ),
            "num_cases": len(items),
        }
        row.update(summarizer(items, cfg))
        leaderboard.append(row)
    return {"leaderboard": leaderboard, "config": cfg.as_dict()}


def aggregate_median_model_metrics(
    records: list[dict[str, Any]],
    config: MetricConfig | None = None,
) -> dict[str, Any]:
    """按模型汇总连续指标中位数，二值率保持 episode 比例。"""
    cfg = config or MetricConfig()
    return _aggregate_model_metrics_with(records, cfg, _summarize_items_median)


def _is_success_item(item: dict[str, Any], cfg: MetricConfig) -> bool:
    if cfg.success_source == "label":
        label = str(item.get("label") or "").lower()
        if label in {"success", "succeeded", "true", "1"}:
            return True
        if label in {"failed", "failure", "false", "0"}:
            return False
    return float(metric_value(item["metrics"], "MP", 0.0)) >= cfg.success_threshold


def trace_summary(item: dict[str, Any], cfg: MetricConfig) -> dict[str, float]:
    """计算逐轨迹结果门控和 v1.5 诊断值。"""
    metrics = item["metrics"]
    sr_trace = 1.0 if _is_success_item(item, cfg) else 0.0
    sqs = success_quality_score(metrics, cfg) if sr_trace else 0.0
    fns = failure_near_success_score(metrics, cfg) if not sr_trace else 0.0
    return {
        "SR_trace": sr_trace,
        "SQS_trace": sqs,
        "FNS_trace": fns,
        "DRR_trace": float(metrics.get("DRR", 1.0)),
    }


def failure_near_success_score(metrics: dict[str, float], cfg: MetricConfig) -> float:
    w1, w2, w3 = cfg.fns_weights
    return (
        w1 * float(metric_value(metrics, "MP", 0.0))
        + w2 * float(metrics.get("M75", 0.0))
        + w3 * float(metrics.get("M50", 0.0))
    )


def success_quality_score(metrics: dict[str, float], cfg: MetricConfig) -> float:
    ppl = float(metrics.get("PPL", 0.0))
    cra_plus = 1.0 - float(metrics.get("CRA", 0.0))
    str_plus = 1.0 - float(metrics.get("STR", 0.0))
    w1, w2, w3 = cfg.sqs_weights
    return max(0.0, min(1.0, w1 * ppl + w2 * cra_plus + w3 * str_plus))
