from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def safe_name(text: str) -> str:
    """Return a stable filesystem-safe name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_") or "sample"


@dataclass
class EvalCase:
    """One user-provided rollout case."""

    case_id: str
    task: str
    videos: dict[str, Path]
    task_name: str | None = None
    goal_image: Path | None = None
    label: str | None = None
    benchmark: str = "manifest"
    model: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)
    progress: list[float] | None = None
    progress_path: Path | None = None

    def output_dir(self, run_root: Path) -> Path:
        return (
            run_root
            / safe_name(self.benchmark)
            / safe_name(self.model)
            / safe_name(self.task_name or self.task)
            / safe_name(self.case_id)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "task": self.task,
            "task_name": self.task_name or self.task,
            "videos": {key: str(value) for key, value in self.videos.items()},
            "goal_image": str(self.goal_image) if self.goal_image else None,
            "label": self.label,
            "benchmark": self.benchmark,
            "model": self.model,
            "metadata": self.metadata,
            "progress_path": str(self.progress_path) if self.progress_path else None,
        }


@dataclass
class ProgressTrace:
    """Standard PRM output consumed by post-processing and metrics."""

    case_id: str
    progress: list[float]
    timestamps: list[float] | None = None
    source_scale: str = "0_1"
    source_type: str = "absolute"
    raw_output_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "progress": self.progress,
            "timestamps": self.timestamps,
            "source_scale": self.source_scale,
            "source_type": self.source_type,
            "raw_output_path": self.raw_output_path,
            "metadata": self.metadata,
        }
