from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable


@dataclass
class PostprocessConfig:
    normalize: str = "auto"
    source_type: str = "absolute"
    clip_min: float = 0.0
    clip_max: float = 1.0
    outlier_method: str = "local_median"
    outlier_threshold: float = 0.35
    smoothing: str = "weighted_window"
    smoothing_weights: tuple[float, ...] = (0.1, 0.2, 0.4, 0.2, 0.1)

    def as_dict(self) -> dict[str, object]:
        return {
            "normalize": self.normalize,
            "source_type": self.source_type,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "outlier_method": self.outlier_method,
            "outlier_threshold": self.outlier_threshold,
            "smoothing": self.smoothing,
            "smoothing_weights": list(self.smoothing_weights),
        }


def parse_smoothing_weights(value: str | Iterable[float]) -> tuple[float, ...]:
    """Parse and normalize an odd-length weighted smoothing window."""
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            raise ValueError("smoothing weights must contain at least one number")
        try:
            weights = tuple(float(part) for part in parts)
        except ValueError as exc:
            raise ValueError(f"invalid smoothing weights: {value}") from exc
    else:
        weights = tuple(float(item) for item in value)

    if not weights:
        raise ValueError("smoothing weights must contain at least one number")
    if len(weights) % 2 == 0:
        raise ValueError("smoothing weights must have odd length so the window is centered")
    if any(weight < 0 for weight in weights):
        raise ValueError("smoothing weights must be non-negative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("smoothing weights must have a positive sum")
    return tuple(weight / total for weight in weights)


def normalize_progress(values: list[float], mode: str = "auto", source_type: str = "absolute") -> list[float]:
    """Convert a raw PRM sequence to absolute progress in [0, 1] scale."""
    if not values:
        return []

    if source_type == "delta":
        total = 0.0
        absolute = []
        for value in values:
            total += float(value)
            absolute.append(total)
        values = absolute
    else:
        values = [float(value) for value in values]

    if mode == "0_100" or (mode == "auto" and max(abs(value) for value in values) > 1.5):
        values = [value / 100.0 for value in values]
    elif mode not in {"auto", "0_1", "0_100"}:
        raise ValueError(f"Unsupported normalize mode: {mode}")
    return values


def clip_progress(values: list[float], low: float = 0.0, high: float = 1.0) -> list[float]:
    return [min(high, max(low, float(value))) for value in values]


def remove_local_outliers(values: list[float], threshold: float = 0.35) -> list[float]:
    """Replace isolated spikes with the local median."""
    if len(values) < 5:
        return list(values)
    cleaned = list(values)
    for idx in range(2, len(values) - 2):
        neighbors = [values[idx - 2], values[idx - 1], values[idx + 1], values[idx + 2]]
        med = statistics.median(neighbors)
        if abs(values[idx] - med) > threshold:
            cleaned[idx] = med
    return cleaned


def weighted_smooth(values: list[float], weights: tuple[float, ...] = (0.1, 0.2, 0.4, 0.2, 0.1)) -> list[float]:
    """Apply edge-aware weighted-window smoothing."""
    if not values:
        return []
    weights = parse_smoothing_weights(weights)
    radius = len(weights) // 2
    smoothed: list[float] = []
    for idx in range(len(values)):
        numerator = 0.0
        denominator = 0.0
        for offset, weight in enumerate(weights):
            src_idx = idx + offset - radius
            if 0 <= src_idx < len(values):
                numerator += values[src_idx] * weight
                denominator += weight
        smoothed.append(numerator / denominator if denominator else values[idx])
    return smoothed


def postprocess_progress(values: list[float], config: PostprocessConfig | None = None) -> list[float]:
    cfg = config or PostprocessConfig()
    processed = normalize_progress(values, cfg.normalize, cfg.source_type)
    processed = clip_progress(processed, cfg.clip_min, cfg.clip_max)
    if cfg.outlier_method == "local_median":
        processed = remove_local_outliers(processed, cfg.outlier_threshold)
    elif cfg.outlier_method not in {"none", ""}:
        raise ValueError(f"Unsupported outlier method: {cfg.outlier_method}")
    if cfg.smoothing == "weighted_window":
        processed = weighted_smooth(processed, cfg.smoothing_weights)
    elif cfg.smoothing not in {"none", ""}:
        raise ValueError(f"Unsupported smoothing method: {cfg.smoothing}")
    return clip_progress(processed, cfg.clip_min, cfg.clip_max)
