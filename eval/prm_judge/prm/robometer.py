from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
import urllib.request
import atexit
from pathlib import Path
from typing import Any

import numpy as np

from ..io import write_json
from ..manifest import VIEW_ALIASES
from ..schema import EvalCase, ProgressTrace
from .base import BasePRMAdapter


class RobometerHttpAdapter(BasePRMAdapter):
    """HTTP client adapter for a running Robometer eval server."""

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        model_path: Path | None = None,
        auto_start: bool = False,
        server_host: str = "127.0.0.1",
        server_port: int = 8000,
        robometer_root: Path | None = None,
        python: Path | str | None = None,
        num_gpus: int = 1,
        max_workers: int = 1,
        frame_steps_micro_batch_size: int = 16,
        startup_timeout_s: float = 720.0,
        fps: float = 1.0,
        use_frame_steps: bool = True,
        timeout_s: float = 1600.0,
        view: str = "video",
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.model_path = model_path
        self.auto_start = bool(auto_start)
        self.server_host = server_host
        self.server_port = int(server_port)
        self.robometer_root = robometer_root
        self.python = str(python or sys.executable)
        self.num_gpus = int(num_gpus)
        self.max_workers = int(max_workers)
        self.frame_steps_micro_batch_size = int(frame_steps_micro_batch_size)
        self.startup_timeout_s = float(startup_timeout_s)
        self.fps = float(fps)
        self.use_frame_steps = bool(use_frame_steps)
        self.timeout_s = float(timeout_s)
        self.view = view
        self._server_process: subprocess.Popen[str] | None = None
        self._server_log_path: Path | None = None
        self._server_model_info: dict[str, Any] | None = None
        self._run_root: Path | None = None
        atexit.register(self.close)

    def set_run_root(self, run_root: Path) -> None:
        self._run_root = run_root

    def predict(self, case: EvalCase, output_dir: Path) -> ProgressTrace:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.ensure_server(output_dir)
        video_path = select_video(case, self.view)
        frames = load_frames_input(video_path, fps=self.fps)
        sample = make_progress_sample(frames, task=case.task, sample_id=case.case_id)
        response = post_evaluate_batch_npy(
            self.server_url,
            [sample],
            timeout_s=self.timeout_s,
            use_frame_steps=self.use_frame_steps,
        )
        progress, success_probs = extract_progress_from_server_output(response)
        progress = validate_unit_interval_curve(progress, "progress_pred", case.case_id)
        success_probs = validate_unit_interval_curve(
            success_probs,
            "success_probs",
            case.case_id,
            allow_empty=True,
        )

        response_path = output_dir / "robometer_response.json"
        progress_path = output_dir / "robometer_progress.json"
        rewards_path = output_dir / "robometer_rewards.npy"
        success_path = output_dir / "robometer_success_probs.npy"
        write_json(response_path, response)
        np.save(rewards_path, progress)
        np.save(success_path, success_probs)
        write_json(
            progress_path,
            {
                "case_id": case.case_id,
                "task": case.task,
                "task_name": case.task_name or case.task,
                "video": str(video_path),
                "view": self.view,
                "fps": self.fps,
                "use_frame_steps": self.use_frame_steps,
                "num_input_frames": int(frames.shape[0]),
                "num_progress_values": int(progress.shape[0]),
                "progress": [float(value) for value in progress.tolist()],
                "success_probs": [float(value) for value in success_probs.tolist()],
            },
        )

        return ProgressTrace(
            case_id=case.case_id,
            progress=[float(value) for value in progress.tolist()],
            source_scale="0_1",
            source_type="absolute",
            raw_output_path=str(progress_path),
            metadata={
                "adapter": "robometer",
                "server_url": self.server_url,
                "auto_started_server": self._server_process is not None,
                "server_log_path": str(self._server_log_path) if self._server_log_path else None,
                "server_model_validated": self._server_model_info is not None,
                "server_model_path": (
                    self._server_model_info.get("model_path") if self._server_model_info else None
                ),
                "view": self.view,
                "fps": self.fps,
                "use_frame_steps": self.use_frame_steps,
                "timeout_s": self.timeout_s,
                "input_frame_count": int(frames.shape[0]),
                "success_probs_path": str(success_path),
                "progress_semantics": "robometer_progress_head_treated_as_absolute_progress",
            },
        )

    def ensure_server(self, output_dir: Path) -> None:
        if is_server_healthy(self.server_url):
            self.validate_server_identity()
            return
        if not self.auto_start:
            raise ConnectionError(
                f"Robometer server is not reachable at {self.server_url}. "
                "Start robometer/evals/eval_server.py first or omit --robometer-server-url to auto-start it."
            )
        self.start_server(output_dir)
        wait_for_server(self.server_url, self.startup_timeout_s, self._server_process, self._server_log_path)
        self.validate_server_identity()

    def validate_server_identity(self) -> None:
        if self.model_path is None:
            return
        model_info = get_server_model_info(self.server_url)
        actual_model_path = model_info.get("model_path")
        if not actual_model_path:
            raise RuntimeError(
                f"Robometer server at {self.server_url} is healthy but did not return `model_path` from /model_info. "
                "Use a compatible Robometer server or choose a free port for auto-start."
            )
        if not model_paths_match(self.model_path, actual_model_path):
            raise RuntimeError(
                "Robometer server model mismatch: "
                f"expected {self.model_path}, but server at {self.server_url} reports {actual_model_path}. "
                "Use --robometer-server-port to choose a free port, or set --prm-path to the model loaded by the server."
            )
        self._server_model_info = model_info

    def start_server(self, output_dir: Path) -> None:
        if self.model_path is None:
            raise ValueError("Robometer auto-start requires a model_path.")
        if self.robometer_root is None:
            raise ValueError("Robometer auto-start requires robometer_root.")
        server_script = self.robometer_root / "robometer" / "evals" / "eval_server.py"
        if not server_script.exists():
            raise FileNotFoundError(f"Robometer server script not found: {server_script}")
        if not Path(self.python).exists():
            raise FileNotFoundError(f"Robometer python executable not found: {self.python}")

        log_path = robometer_server_log_path(output_dir, self.server_port, self._run_root)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.robometer_root) + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("HYDRA_FULL_ERROR", "1")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        command = [
            self.python,
            str(server_script),
            f"server_url={self.server_host}",
            f"server_port={self.server_port}",
            f"model_path={self.model_path}",
            f"num_gpus={self.num_gpus}",
            f"max_workers={self.max_workers}",
            f"frame_steps_micro_batch_size={self.frame_steps_micro_batch_size}",
        ]
        self._server_log_path = log_path
        try:
            self._server_process = subprocess.Popen(
                command,
                cwd=str(self.robometer_root),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        finally:
            log_file.close()

    def close(self) -> None:
        process = self._server_process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def is_server_healthy(server_url: str, timeout_s: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(server_url.rstrip("/") + "/health", timeout=timeout_s) as response:
            return 200 <= int(response.status) < 300
    except Exception:  # noqa: BLE001
        return False


def get_server_model_info(server_url: str, timeout_s: float = 10.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(server_url.rstrip("/") + "/model_info", timeout=timeout_s) as response:
            if not (200 <= int(response.status) < 300):
                raise RuntimeError(f"GET /model_info returned HTTP {response.status}.")
            payload = response.read().decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not read Robometer /model_info from {server_url}: {exc}") from exc
    try:
        model_info = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Robometer /model_info returned invalid JSON from {server_url}.") from exc
    if not isinstance(model_info, dict):
        raise RuntimeError(f"Robometer /model_info returned {type(model_info).__name__}, expected object.")
    return model_info


def model_paths_match(expected: Path | str, actual: Path | str) -> bool:
    expected_text = str(expected)
    actual_text = str(actual)
    if expected_text == actual_text:
        return True
    return normalize_model_path(expected_text) == normalize_model_path(actual_text)


def normalize_model_path(value: str) -> str:
    try:
        return str(Path(value).expanduser().resolve(strict=False))
    except Exception:  # noqa: BLE001
        return value


def robometer_server_log_path(output_dir: Path, server_port: int, run_root: Path | None = None) -> Path:
    if run_root is not None:
        return run_root / f"robometer_server_{server_port}.log"
    return output_dir / f"robometer_server_{server_port}.log"


def wait_for_server(
    server_url: str,
    timeout_s: float,
    process: subprocess.Popen[str] | None,
    log_path: Path | None,
) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        if process is not None and process.poll() is not None:
            tail = tail_text(log_path)
            raise RuntimeError(f"Robometer server exited with code {process.returncode}.\n{tail}")
        if is_server_healthy(server_url):
            return
        time.sleep(5)
    tail = tail_text(log_path)
    raise TimeoutError(f"Timed out waiting for Robometer server at {server_url}.\n{tail}")


def tail_text(path: Path | None, max_chars: int = 4000) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")[-max_chars:]


def select_video(case: EvalCase, view: str) -> Path:
    """按标准字段选择 RoboMeter 单视角，并解析旧名称兼容别名。"""
    canonical = VIEW_ALIASES.get(view, view)
    if canonical in case.videos:
        return case.videos[canonical]
    if canonical in {"left_wrist_video", "right_wrist_video"} and "wrist_video" in case.videos:
        return case.videos["wrist_video"]
    if canonical != "video":
        available = ", ".join(sorted(case.videos)) or "none"
        raise ValueError(
            f"Case {case.case_id} does not contain requested Robometer view "
            f"{view!r} (canonical {canonical!r}); available views: {available}."
        )
    if "video" in case.videos:
        return case.videos["video"]
    if case.videos:
        return next(iter(case.videos.values()))
    raise ValueError(f"Case {case.case_id} has no video inputs.")


def load_frames_input(video_or_array_path: Path, fps: float = 1.0) -> np.ndarray:
    path = Path(video_or_array_path)
    if path.suffix == ".npy":
        frames = np.load(path)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as npz:
            frames = npz["arr_0"].copy() if "arr_0" in npz else next(iter(npz.values())).copy()
    else:
        frames = extract_video_frames(path, fps=fps)

    if frames is None or frames.size == 0:
        raise RuntimeError(f"Could not extract frames from {path}.")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    if frames.ndim == 4 and frames.shape[1] in (1, 3) and frames.shape[-1] not in (1, 3):
        frames = frames.transpose(0, 2, 3, 1)
    if frames.ndim != 4 or frames.shape[-1] not in (1, 3):
        raise ValueError(f"Expected frames shaped (T,H,W,C), got {frames.shape}.")
    return frames


def extract_video_frames(video_path: Path, fps: float = 1.0) -> np.ndarray:
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    try:
        import decord  # type: ignore
    except Exception:  # noqa: BLE001
        return extract_video_frames_cv2(video_path, fps=fps)

    try:
        vr = decord.VideoReader(str(video_path), num_threads=1)
        total_frames = len(vr)
        if total_frames <= 0:
            raise RuntimeError(f"Video has no frames: {video_path}")
        try:
            native_fps = float(vr.get_avg_fps())
        except Exception:  # noqa: BLE001
            native_fps = 1.0
        frame_indices = sample_frame_indices(total_frames, native_fps, fps)
        frames = vr.get_batch(frame_indices).asnumpy()
        del vr
        return frames
    except Exception:  # noqa: BLE001
        return extract_video_frames_cv2(video_path, fps=fps)


def sample_frame_indices(total_frames: int, native_fps: float, fps: float) -> list[int]:
    if fps <= 0:
        fps = native_fps
    desired_frames = int(round(total_frames * (fps / native_fps))) if native_fps > 0 else total_frames
    desired_frames = max(1, min(desired_frames, total_frames))
    if desired_frames == total_frames:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, desired_frames, dtype=int).tolist()


def extract_video_frames_cv2(video_path: Path, fps: float = 1.0) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Robometer video input requires `decord` or `opencv-python`.") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 1.0)
    frame_indices = sample_frame_indices(total_frames, native_fps, fps)
    frames = []
    for index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"Could not extract frames from {video_path}")
    return np.stack(frames, axis=0)


def make_progress_sample(frames: np.ndarray, task: str, sample_id: str) -> dict[str, Any]:
    return {
        "sample_type": "progress",
        "trajectory": {
            "frames": frames,
            "frames_shape": tuple(frames.shape),
            "task": task,
            "id": sample_id,
            "metadata": {"subsequence_length": int(frames.shape[0])},
            "video_embeddings": None,
        },
    }


def post_evaluate_batch_npy(
    server_url: str,
    samples: list[dict[str, Any]],
    timeout_s: float = 1600.0,
    use_frame_steps: bool = True,
) -> dict[str, Any]:
    try:
        import requests
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Robometer HTTP mode requires `requests` in the active environment.") from exc

    files, data = build_multipart_payload(samples)
    try:
        data["use_frame_steps"] = "true" if use_frame_steps else "false"
        response = requests.post(
            server_url.rstrip("/") + "/evaluate_batch_npy",
            files=files,
            data=data,
            timeout=timeout_s,
        )
        response.raise_for_status()
        return response.json()
    finally:
        close_multipart_files(files)


def build_multipart_payload(samples: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, str]]:
    files: dict[str, Any] = {}
    data: dict[str, str] = {}
    numpy_fields = ["frames", "lang_vector", "video_embeddings"]

    for idx, sample in enumerate(samples):
        sample_copy = {
            key: to_jsonable_metadata(value, f"sample[{idx}].{key}")
            for key, value in sample.items()
            if key != "trajectory"
        }
        trajectory = sample.get("trajectory", {})
        trajectory_copy = {
            key: to_jsonable_metadata(value, f"sample[{idx}].trajectory.{key}")
            for key, value in trajectory.items()
            if key not in numpy_fields
        }
        for field in numpy_fields:
            value = trajectory.get(field)
            if value is None:
                continue
            if hasattr(value, "detach") and hasattr(value, "cpu"):
                value = value.detach().cpu().numpy()
            if isinstance(value, np.ndarray):
                file_key = f"sample_{idx}_trajectory_{field}"
                files[file_key] = numpy_to_npy_file_tuple(value, f"{file_key}.npy")
                trajectory_copy[field] = {"__numpy_file__": file_key}
            else:
                trajectory_copy[field] = to_jsonable_metadata(value, f"sample[{idx}].trajectory.{field}")
        if "frames_shape" in trajectory_copy and isinstance(trajectory_copy["frames_shape"], (tuple, list)):
            trajectory_copy["frames_shape"] = [int(value) for value in trajectory_copy["frames_shape"]]
        sample_copy["trajectory"] = trajectory_copy
        data[f"sample_{idx}"] = json.dumps(sample_copy)
    return files, data


def to_jsonable_metadata(value: Any, field_path: str) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        raise TypeError(f"Unexpected numpy array in JSON metadata field {field_path}; upload it as an npy field.")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable_metadata(item, f"{field_path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable_metadata(item, field_path) for item in value]
    return value


def numpy_to_npy_file_tuple(array: np.ndarray, filename: str) -> tuple[str, io.BytesIO, str]:
    buffer = io.BytesIO()
    np.save(buffer, array)
    buffer.seek(0)
    return (filename, buffer, "application/octet-stream")


def close_multipart_files(files: dict[str, Any]) -> None:
    for value in files.values():
        if isinstance(value, tuple) and len(value) >= 2 and hasattr(value[1], "close"):
            value[1].close()


def extract_progress_from_server_output(outputs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    outputs_progress = outputs.get("outputs_progress")
    if outputs_progress is None:
        raise ValueError("No `outputs_progress` in Robometer server response.")
    progress_pred = outputs_progress.get("progress_pred", [])
    progress = np.array(progress_pred[0], dtype=np.float32) if progress_pred else np.array([], dtype=np.float32)

    outputs_success = outputs.get("outputs_success") or {}
    success_probs = outputs_success.get("success_probs", []) if isinstance(outputs_success, dict) else []
    success = np.array(success_probs[0], dtype=np.float32) if success_probs else np.array([], dtype=np.float32)
    return progress, success


def validate_unit_interval_curve(
    values: np.ndarray,
    name: str,
    case_id: str,
    allow_empty: bool = False,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        if allow_empty:
            return array.reshape(-1)
        raise ValueError(f"Robometer produced an empty {name} curve for case {case_id}.")
    if array.ndim != 1:
        raise ValueError(f"Robometer produced {name} with shape {array.shape} for case {case_id}; expected 1D.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Robometer produced non-finite values in {name} for case {case_id}.")
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    tolerance = 1e-4
    if min_value < -tolerance or max_value > 1.0 + tolerance:
        raise ValueError(
            f"Robometer produced {name} outside [0, 1] for case {case_id}: "
            f"min={min_value:.6g}, max={max_value:.6g}."
        )
    return np.clip(array, 0.0, 1.0)
