from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from ..io import read_json
from ..schema import EvalCase, ProgressTrace
from .base import BasePRMAdapter


class DopamineAdapter(BasePRMAdapter):
    """Robo-Dopamine GRM adapter."""

    def __init__(
        self,
        model_path: Path,
        frame_interval: int = 72,
        batch_size: int = 10,
        eval_mode: str = "incremental",
        tensor_parallel_size: int = 1,
        visualize: bool = False,
        keep_cache: bool = False,
        fallback_goal_image: Path | None = None,
    ) -> None:
        self.model_path = model_path
        self.frame_interval = frame_interval
        self.batch_size = batch_size
        self.eval_mode = eval_mode
        self.tensor_parallel_size = tensor_parallel_size
        self.visualize = visualize
        self.keep_cache = keep_cache
        self.fallback_goal_image = fallback_goal_image
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        saved_local_rank = os.environ.get("LOCAL_RANK")
        saved_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        from .dopamine_inference import GRMInference  # noqa: PLC0415

        if saved_local_rank is None:
            os.environ.pop("LOCAL_RANK", None)
        else:
            os.environ["LOCAL_RANK"] = saved_local_rank
        if saved_cuda_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved_cuda_visible_devices

        self._model = GRMInference(
            model_path=str(self.model_path),
            tensor_parallel_size=self.tensor_parallel_size,
        )
        return self._model

    def predict(self, case: EvalCase, output_dir: Path) -> ProgressTrace:
        output_dir.mkdir(parents=True, exist_ok=True)
        model = self._load_model()
        videos = _dopamine_video_args(case)
        goal_image = case.goal_image or self.fallback_goal_image

        pipeline_output_dir = Path(
            model.run_pipeline(
                cam_high_path=str(videos["cam_high"]),
                cam_left_path=str(videos["cam_left_wrist"]),
                cam_right_path=str(videos["cam_right_wrist"]),
                out_root=str(output_dir),
                task=case.task,
                frame_interval=self.frame_interval,
                batch_size=self.batch_size,
                goal_image=str(goal_image) if goal_image else None,
                eval_mode=self.eval_mode,
                visualize=self.visualize,
            )
        )
        _flatten_pipeline_output(pipeline_output_dir, output_dir)

        pred_path = output_dir / "pred_vllm.json"
        progress = _read_progress_predictions(pred_path)

        if not self.keep_cache:
            for path in [output_dir / ".cache", output_dir / "sample.json"]:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()

        return ProgressTrace(
            case_id=case.case_id,
            progress=progress,
            raw_output_path=str(pred_path),
            metadata={
                "adapter": "dopamine",
                "eval_mode": self.eval_mode,
                "frame_interval": self.frame_interval,
                "batch_size": self.batch_size,
            },
        )


def _read_progress_predictions(pred_path: Path) -> list[float]:
    """Read and validate the progress curve emitted by the Dopamine pipeline."""
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Dopamine did not produce {pred_path}. "
            "Check the checkpoint path, input videos, dependencies, and backend logs."
        )
    predictions = read_json(pred_path)
    if not isinstance(predictions, list):
        raise ValueError(f"Dopamine output must be a list in {pred_path}, got {type(predictions).__name__}.")
    progress = [
        float(item.get("progress", 0.0) if isinstance(item, dict) else item)
        for item in predictions
    ]
    if not progress:
        raise ValueError(f"Dopamine produced an empty progress curve in {pred_path}.")
    return progress


def _dopamine_video_args(case: EvalCase) -> dict[str, Path]:
    """将 manifest 的标准视角字段展开为 Dopamine 所需的三个相机槽位。"""
    main_video = case.videos.get("video")
    if main_video is None:
        raise ValueError(
            f"Dopamine requires the main 'video' field for case {case.case_id}. "
            f"Available views: {sorted(case.videos)}"
        )

    single_arm_wrist = case.videos.get("wrist_video")
    if single_arm_wrist is not None:
        return {
            "cam_high": main_video,
            "cam_left_wrist": single_arm_wrist,
            "cam_right_wrist": single_arm_wrist,
        }

    if {"left_wrist_video", "right_wrist_video"} <= set(case.videos):
        return {
            "cam_high": main_video,
            "cam_left_wrist": case.videos["left_wrist_video"],
            "cam_right_wrist": case.videos["right_wrist_video"],
        }

    return {
        "cam_high": main_video,
        "cam_left_wrist": main_video,
        "cam_right_wrist": main_video,
    }


def _flatten_pipeline_output(pipeline_output_dir: Path, sample_root: Path) -> None:
    if not pipeline_output_dir.exists() or pipeline_output_dir == sample_root:
        return
    for child in pipeline_output_dir.iterdir():
        target = sample_root / child.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(child), str(target))
    if pipeline_output_dir.exists():
        pipeline_output_dir.rmdir()
