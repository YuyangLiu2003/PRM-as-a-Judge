from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io import read_jsonl
from .schema import EvalCase

VIDEO_FIELDS = ("video", "wrist_video", "left_wrist_video", "right_wrist_video")

# 旧版 ``videos`` 对象仍可读取，但加载后统一转换为新的字段名称。
VIEW_ALIASES = {
    "video": "video",
    "high": "video",
    "front": "video",
    "cam_high": "video",
    "wrist": "wrist_video",
    "wrist_video": "wrist_video",
    "left": "left_wrist_video",
    "left_wrist": "left_wrist_video",
    "left_wrist_video": "left_wrist_video",
    "cam_left_wrist": "left_wrist_video",
    "right": "right_wrist_video",
    "right_wrist": "right_wrist_video",
    "right_wrist_video": "right_wrist_video",
    "cam_right_wrist": "right_wrist_video",
}


def _resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def _required_video_path(base_dir: Path, value: Any, line_no: int, field: str) -> Path:
    """解析非空视频路径，并为 manifest 错误提供明确的字段定位。"""
    if value is None or not str(value).strip():
        raise ValueError(f"Line {line_no}: empty video path for field {field!r}.")
    path = _resolve_path(base_dir, str(value))
    if path is None:  # pragma: no cover - guarded above for type narrowing
        raise ValueError(f"Line {line_no}: empty video path for field {field!r}.")
    return path


def _validate_standard_views(videos: dict[str, Path], line_no: int) -> None:
    """校验四字段视频规范，避免单臂和双臂腕部字段混用。"""
    if "video" not in videos:
        raise ValueError(f"Line {line_no}: missing required 'video' field.")

    has_single_arm_wrist = "wrist_video" in videos
    has_left_wrist = "left_wrist_video" in videos
    has_right_wrist = "right_wrist_video" in videos
    if has_single_arm_wrist and (has_left_wrist or has_right_wrist):
        raise ValueError(
            f"Line {line_no}: 'wrist_video' cannot be combined with "
            "'left_wrist_video' or 'right_wrist_video'."
        )
    if has_left_wrist != has_right_wrist:
        raise ValueError(
            f"Line {line_no}: 'left_wrist_video' and 'right_wrist_video' must be provided together."
        )


def _normalize_videos(base_dir: Path, row: dict[str, Any], line_no: int) -> dict[str, Path]:
    """加载标准四字段格式，并兼容旧版 ``videos`` 视角对象。"""
    supplied_fields = [field for field in VIDEO_FIELDS if field in row]
    if "videos" in row and supplied_fields:
        joined = ", ".join(supplied_fields)
        raise ValueError(f"Line {line_no}: legacy 'videos' cannot be combined with {joined}.")

    if supplied_fields:
        videos = {
            field: _required_video_path(base_dir, row[field], line_no, field)
            for field in supplied_fields
        }
        _validate_standard_views(videos, line_no)
        return videos

    raw_videos = row.get("videos")
    if not isinstance(raw_videos, dict) or not raw_videos:
        raise ValueError(
            f"Line {line_no}: provide required 'video' and optional wrist fields; "
            "legacy manifests may provide a non-empty 'videos' object."
        )

    videos: dict[str, Path] = {}
    for key, value in raw_videos.items():
        canonical = VIEW_ALIASES.get(str(key), str(key))
        if canonical in videos:
            raise ValueError(f"Line {line_no}: duplicate video view after normalization: {canonical!r}.")
        videos[canonical] = _required_video_path(base_dir, value, line_no, str(key))
    return videos


def _load_progress(base_dir: Path, row: dict[str, Any], line_no: int) -> tuple[list[float] | None, Path | None]:
    if "progress" in row and "progress_path" in row:
        raise ValueError(f"Line {line_no}: use either 'progress' or 'progress_path', not both.")
    if "progress" in row:
        progress = row["progress"]
        if not isinstance(progress, list):
            raise ValueError(f"Line {line_no}: 'progress' must be a list of numbers.")
        return [float(item) for item in progress], None
    if "progress_path" in row:
        progress_path = _resolve_path(base_dir, str(row["progress_path"]))
        if progress_path is None:
            raise ValueError(f"Line {line_no}: empty progress_path.")
        data = json.loads(progress_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "progress" in data:
            data = data["progress"]
        if not isinstance(data, list):
            raise ValueError(f"Line {line_no}: progress_path must contain a list or an object with a progress list.")
        return [float(item.get("progress", item)) if isinstance(item, dict) else float(item) for item in data], progress_path
    return None, None


def load_manifest(path: Path) -> list[EvalCase]:
    """Load one JSONL manifest where each line describes one rollout case."""
    rows = read_jsonl(path)
    base_dir = path.parent
    cases: list[EvalCase] = []

    for line_no, row in enumerate(rows, start=1):
        case_id = str(row.get("case_id") or row.get("id") or "").strip()
        metadata = dict(row.get("metadata") or {})
        task_name = str(row.get("task_name") or row.get("task_id") or metadata.get("task_name") or "").strip()
        task_field = str(row.get("task") or "").strip()
        instruction = str(row.get("instruction") or row.get("task_prompt") or "").strip()
        task = instruction or task_field
        if not task_name:
            task_name = task_field or task
        if not case_id:
            raise ValueError(f"Line {line_no}: missing required 'case_id'.")
        if not task:
            raise ValueError(f"Line {line_no}: missing required 'task' or 'instruction'.")

        progress, progress_path = _load_progress(base_dir, row, line_no)
        cases.append(
            EvalCase(
                case_id=case_id,
                task=task,
                task_name=task_name,
                videos=_normalize_videos(base_dir, row, line_no),
                goal_image=_resolve_path(base_dir, row.get("goal_image")),
                label=str(row["label"]).lower() if row.get("label") is not None else None,
                benchmark=str(row.get("benchmark") or "manifest"),
                model=str(row.get("model") or row.get("policy") or "default"),
                metadata=metadata,
                progress=progress,
                progress_path=progress_path,
            )
        )
    return cases


def filter_cases(
    cases: list[EvalCase],
    benchmark_filters: list[str] | None = None,
    task_filters: list[str] | None = None,
    model_filters: list[str] | None = None,
    case_filters: list[str] | None = None,
    limit: int | None = None,
) -> list[EvalCase]:
    selected = cases
    if benchmark_filters:
        allowed = {item.lower() for item in benchmark_filters}
        selected = [case for case in selected if case.benchmark.lower() in allowed]
    if task_filters:
        tokens = [item.lower() for item in task_filters]
        selected = [
            case
            for case in selected
            if any(token in case.task.lower() or token in (case.task_name or "").lower() for token in tokens)
        ]
    if model_filters:
        tokens = [item.lower() for item in model_filters]
        selected = [case for case in selected if any(token in case.model.lower() for token in tokens)]
    if case_filters:
        tokens = [item.lower() for item in case_filters]
        selected = [case for case in selected if any(token in case.case_id.lower() for token in tokens)]
    if limit is not None:
        selected = selected[:limit]
    return selected
