from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .curves import PostprocessConfig, parse_smoothing_weights, postprocess_progress
from .excel_report import write_metrics_workbook
from .io import append_jsonl, write_json
from .manifest import filter_cases, load_manifest
from .metrics import MetricConfig, aggregate_metrics, recompute_record_metrics, trace_metrics
from .reports import write_markdown_report
from .schema import EvalCase
from .visualize import visualize_run

EVAL_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EVAL_DIR.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "PRM" / "Robo-Dopamine-GRM-2.0-8B-Preview"
DEFAULT_ROBOMETER_MODEL_PATH = PROJECT_ROOT.parent / "PRM" / "Robometer-4B"
DEFAULT_ROBOMETER_ROOT = PROJECT_ROOT.parent / "robometer"
ROBOMETER_PYTHON_CANDIDATES = [
    PROJECT_ROOT.parent / "env" / "robometer" / "bin" / "python",
    PROJECT_ROOT.parent.parent / "env" / "robometer" / "bin" / "python",
]
DEFAULT_ROBOMETER_PYTHON = next(
    (path for path in ROBOMETER_PYTHON_CANDIDATES if path.exists()),
    Path(sys.executable),
)
DEFAULT_OUTPUT_ROOT = EVAL_DIR / "results"
DEFAULT_GOAL_IMAGE = EVAL_DIR / "examples" / "blank.png"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PRM-as-a-Judge v1.5 evaluation toolkit.")
    subparsers = parser.add_subparsers(dest="command")
    eval_parser = subparsers.add_parser("eval", help="Evaluate rollout videos from a JSONL manifest.")
    add_eval_args(eval_parser, manifest_required=True)
    visualize_parser = subparsers.add_parser("visualize", help="Generate progress-curve visualizations for an existing run.")
    add_visualize_args(visualize_parser)
    serve_parser = subparsers.add_parser("serve", help="Serve an interactive report with HTTP Range video support.")
    add_serve_args(serve_parser)
    add_eval_args(parser)
    return parser


def add_eval_args(parser: argparse.ArgumentParser, *, manifest_required: bool = False) -> None:
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        required=manifest_required,
        help="Required JSONL case manifest.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--goal-fallback", type=Path, default=DEFAULT_GOAL_IMAGE)
    parser.add_argument("--prm", choices=["dopamine", "recorded", "robometer"], default="dopamine")
    parser.add_argument("--prm-path", "--grm-path", dest="prm_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--frame-interval", type=int, default=72)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--eval-mode", choices=["incremental", "forward", "backward"], default="incremental")
    parser.add_argument("--robometer-server-url", default=None)
    parser.add_argument("--robometer-auto-start", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--robometer-server-host", default="127.0.0.1")
    parser.add_argument("--robometer-server-port", type=int, default=8000)
    parser.add_argument("--robometer-root", type=Path, default=Path(os.environ.get("ROBOMETER_ROOT", DEFAULT_ROBOMETER_ROOT)))
    parser.add_argument(
        "--robometer-python",
        default=os.environ.get(
            "ROBOMETER_PYTHON",
            str(DEFAULT_ROBOMETER_PYTHON),
        ),
    )
    parser.add_argument("--robometer-num-gpus", type=int, default=1)
    parser.add_argument("--robometer-max-workers", type=int, default=1)
    parser.add_argument("--robometer-frame-steps-micro-batch-size", type=int, default=16)
    parser.add_argument("--robometer-startup-timeout-s", type=float, default=720.0)
    parser.add_argument("--robometer-fps", type=float, default=1.0)
    parser.add_argument(
        "--robometer-view",
        default="video",
        help="Manifest view field for RoboMeter: video, wrist_video, left_wrist_video, or right_wrist_video.",
    )
    parser.add_argument("--robometer-timeout-s", type=float, default=1600.0)
    parser.add_argument("--robometer-use-frame-steps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpus", default=os.environ.get("GPUS", "0"), help="'0', '0,1,2,3', or 'auto'.")
    parser.add_argument("--benchmark", nargs="*", default=None)
    parser.add_argument("--task-filter", nargs="*", default=None)
    parser.add_argument("--model-filter", nargs="*", default=None)
    parser.add_argument("--case-filter", "--sample-filter", dest="case_filter", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--normalize", choices=["auto", "0_1", "0_100"], default="auto")
    parser.add_argument("--source-type", choices=["absolute", "delta"], default="absolute")
    parser.add_argument("--outlier-method", choices=["local_median", "none"], default="local_median")
    parser.add_argument("--smoothing", choices=["weighted_window", "none"], default="weighted_window")
    parser.add_argument(
        "--smoothing-weights",
        default="0.1,0.2,0.4,0.2,0.1",
        help="Comma-separated odd-length smoothing weights, normalized automatically.",
    )
    parser.add_argument("--success-threshold", type=float, default=0.99)
    parser.add_argument(
        "--success-source",
        choices=["progress", "label"],
        default="progress",
        help="Use progress threshold or manifest labels for success-conditioned metrics.",
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exit successfully even when one or more cases have status=error.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--shard-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num-shards", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--run-root", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--visualize-output-dir", type=Path, default=None)
    parser.add_argument("--visualize-max-cases", type=int, default=0, help="Maximum per-case plots to write; 0 means all.")


def add_visualize_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", type=Path, required=True, help="Existing run directory containing per_case.jsonl.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Visualization output directory.")
    parser.add_argument("--max-cases", type=int, default=0, help="Maximum per-case plots to write; 0 means all.")
    parser.add_argument("--success-threshold", type=float, default=None, help="Override threshold line in curve plots.")


def add_serve_args(parser: argparse.ArgumentParser) -> None:
    """注册支持视频 Range 请求的报告服务参数。"""
    parser.add_argument("--run-root", type=Path, required=True, help="Existing run directory containing per_case.jsonl.")
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind. Use 0.0.0.0 only on a trusted network.")
    parser.add_argument("--port", type=int, default=8000, help="TCP port for the report server.")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        args.command = "eval"
    if args.command == "eval":
        run_eval(args)
        return
    if args.command == "visualize":
        run_visualize(args)
        return
    if args.command == "serve":
        run_serve(args)
        return
    parser.error(f"Unsupported command: {args.command}")


def run_visualize(args: argparse.Namespace) -> None:
    artifacts = visualize_run(
        run_root=args.run_root,
        output_dir=args.output_dir,
        max_cases=args.max_cases,
        success_threshold=args.success_threshold,
    )
    for name, path in artifacts.items():
        print(f"[VIS] {name}: {path}")


def run_serve(args: argparse.Namespace) -> None:
    """启动只暴露当前 run 视频的本地报告服务。"""
    from .serve import serve_run

    serve_run(run_root=args.run_root, host=args.host, port=args.port)


def run_eval(args: argparse.Namespace) -> None:
    try:
        parse_smoothing_weights(args.smoothing_weights)
    except ValueError as exc:
        raise SystemExit(f"--smoothing-weights: {exc}") from exc
    cases = load_cases(args)
    selected = filter_cases(
        cases,
        benchmark_filters=args.benchmark,
        task_filters=args.task_filter,
        model_filters=args.model_filter,
        case_filters=args.case_filter,
        limit=args.limit,
    )
    if args.num_shards and args.shard_index is not None:
        selected = [
            case
            for idx, case in enumerate(selected)
            if idx % args.num_shards == args.shard_index
        ]

    run_root = args.run_root or args.output_root / f"run_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=True)
    if args.num_shards is not None and args.shard_index is not None:
        worker_root = _worker_shard_root(args, run_root)
        worker_root.mkdir(parents=True, exist_ok=True)
        write_json(
            worker_root / "run_params.json",
            _jsonable(vars(args) | {"run_root": str(run_root), "working_directory": str(Path.cwd().resolve())}),
        )
        write_json(worker_root / "discovery_manifest.json", [case.as_dict() for case in selected])
    else:
        write_json(
            run_root / "run_params.json",
            _jsonable(vars(args) | {"run_root": str(run_root), "working_directory": str(Path.cwd().resolve())}),
        )
        write_json(run_root / "discovery_manifest.json", [case.as_dict() for case in selected])

    if args.dry_run:
        print(f"[DRY-RUN] Discovered {len(selected)} cases. Manifest saved to {run_root}")
        return

    gpus = resolve_gpus(args.gpus)
    if len(gpus) > 1 and args.num_shards is None:
        summary = run_multi_gpu(args, run_root, len(selected), gpus)
        finish_run(summary, run_root, args)
        return
    if len(gpus) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])

    records = process_cases(args, selected, run_root)
    if args.num_shards is not None and args.shard_index is not None:
        print(f"[SHARD {args.shard_index}/{args.num_shards}] Wrote {len(records)} records.")
        return
    summary = finalize_run(records, run_root, args)
    finish_run(summary, run_root, args)


def load_cases(args: argparse.Namespace) -> list[EvalCase]:
    """加载必需的 JSONL manifest，不再回退到旧目录发现流程。"""
    if args.manifest is None:
        raise SystemExit("--manifest is required for evaluation.")
    return load_manifest(args.manifest)


def build_adapter(args: argparse.Namespace):
    if args.prm == "recorded":
        from .prm.recorded import RecordedProgressAdapter

        return RecordedProgressAdapter()
    if args.prm == "robometer":
        from .prm.robometer import RobometerHttpAdapter

        shard_index = args.shard_index or 0
        server_port = args.robometer_server_port + (shard_index if args.num_shards else 0)
        explicit_server_url = args.robometer_server_url is not None
        server_url = args.robometer_server_url or f"http://{args.robometer_server_host}:{server_port}"
        auto_start = args.robometer_auto_start
        if auto_start is None:
            auto_start = not explicit_server_url
        server_host = args.robometer_server_host
        if explicit_server_url and auto_start:
            parsed = urlparse(server_url)
            server_host = parsed.hostname or server_host
            server_port = parsed.port or server_port
        model_path = args.prm_path
        if model_path == DEFAULT_MODEL_PATH:
            model_path = DEFAULT_ROBOMETER_MODEL_PATH
        return RobometerHttpAdapter(
            server_url=server_url,
            model_path=model_path,
            auto_start=auto_start,
            server_host=server_host,
            server_port=server_port,
            robometer_root=args.robometer_root,
            python=args.robometer_python,
            num_gpus=args.robometer_num_gpus,
            max_workers=args.robometer_max_workers,
            frame_steps_micro_batch_size=args.robometer_frame_steps_micro_batch_size,
            startup_timeout_s=args.robometer_startup_timeout_s,
            fps=args.robometer_fps,
            use_frame_steps=args.robometer_use_frame_steps,
            timeout_s=args.robometer_timeout_s,
            view=args.robometer_view,
        )
    from .prm.dopamine import DopamineAdapter

    return DopamineAdapter(
        model_path=args.prm_path,
        frame_interval=args.frame_interval,
        batch_size=args.batch_size,
        eval_mode=args.eval_mode,
        tensor_parallel_size=args.tensor_parallel_size,
        visualize=args.visualize,
        keep_cache=args.keep_cache,
        fallback_goal_image=args.goal_fallback if args.goal_fallback.exists() else None,
    )


def process_cases(args: argparse.Namespace, cases: list[EvalCase], run_root: Path) -> list[dict[str, Any]]:
    adapter = build_adapter(args)
    if hasattr(adapter, "set_run_root"):
        adapter.set_run_root(run_root)
    post_cfg = PostprocessConfig(
        normalize=args.normalize,
        source_type=args.source_type,
        outlier_method=args.outlier_method,
        smoothing=args.smoothing,
        smoothing_weights=parse_smoothing_weights(args.smoothing_weights),
    )
    metric_cfg = MetricConfig(
        success_threshold=args.success_threshold,
        success_source=args.success_source,
    )
    worker_root = _worker_shard_root(args, run_root)
    per_case_path = (worker_root / "per_case.jsonl") if worker_root is not None else (run_root / "per_case.jsonl")
    records: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        case_root = case.output_dir(run_root)
        result_path = case_root / "result_summary.json"
        if args.skip_existing and result_path.exists():
            records.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue
        try:
            trace = adapter.predict(case, case_root)
            raw = list(trace.progress)
            processed = postprocess_progress(raw, post_cfg)
            metrics = trace_metrics(processed, metric_cfg)
            record: dict[str, Any] = {
                **case.as_dict(),
                "case_task": case.task,
                "task_name": case.task_name or case.task,
                "status": "ok",
                "output_dir": str(case_root),
                "progress_raw": raw,
                "progress_processed": processed,
                "trace": trace.as_dict(),
                "postprocess": post_cfg.as_dict(),
                "metrics": metrics,
            }
        except Exception as exc:  # noqa: BLE001
            record = {
                **case.as_dict(),
                "case_task": case.task,
                "task_name": case.task_name or case.task,
                "status": "error",
                "error": repr(exc),
                "output_dir": str(case_root),
            }
        write_json(result_path, record)
        append_jsonl(per_case_path, record)
        records.append(record)
        print(f"[{index}/{len(cases)}] {case.case_id}: {record['status']}")
    return records


def finalize_run(records: list[dict[str, Any]], run_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    metric_cfg = MetricConfig(
        success_threshold=args.success_threshold,
        success_source=args.success_source,
    )
    recompute_record_metrics(records, metric_cfg)
    summary = {
        "total_cases": len(records),
        "processed_ok": sum(1 for item in records if item.get("status") == "ok"),
        "processed_error": sum(1 for item in records if item.get("status") != "ok"),
    }
    aggregate = aggregate_metrics(records, metric_cfg)
    per_case_path = run_root / "per_case.jsonl"
    if per_case_path.exists():
        per_case_path.unlink()
    per_case_path.touch()
    for record in records:
        append_jsonl(per_case_path, record)
    write_json(run_root / "run_summary.json", summary | {"metrics": aggregate})
    write_metrics_workbook(run_root / "metrics.xlsx", records, aggregate, metric_cfg)
    write_markdown_report(run_root / "report.md", summary)
    if args.visualize:
        artifacts = visualize_run(
            run_root=run_root,
            output_dir=args.visualize_output_dir,
            max_cases=args.visualize_max_cases,
            success_threshold=args.success_threshold,
        )
        for name, path in artifacts.items():
            print(f"[VIS] {name}: {path}")
    return summary


def finish_run(summary: dict[str, Any], run_root: Path, args: argparse.Namespace) -> None:
    """在产物落盘后根据案例错误数设置进程退出语义。"""
    error_count = int(summary.get("processed_error", 0))
    total_cases = int(summary.get("total_cases", 0))
    if error_count and not args.allow_partial:
        print(
            f"[FAILED] {error_count}/{total_cases} cases failed. Results saved to {run_root}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if error_count:
        print(f"[DONE WITH ERRORS] {error_count}/{total_cases} cases failed. Results saved to {run_root}")
        return
    print(f"[DONE] Results saved to {run_root}")


def run_multi_gpu(
    args: argparse.Namespace,
    run_root: Path,
    total_cases: int,
    gpus: list[str],
) -> dict[str, Any]:
    print(f"[INFO] Launching {len(gpus)} GPU workers for {total_cases} cases: {','.join(gpus)}")
    commands: list[tuple[str, list[str]]] = []
    python_executable = worker_python(args)
    for shard_index, gpu in enumerate(gpus):
        command = [
            python_executable,
            "-m",
            "prm_judge.cli",
            "eval",
            "--run-root",
            str(run_root),
            "--shard-index",
            str(shard_index),
            "--num-shards",
            str(len(gpus)),
            "--gpus",
            str(gpu),
        ] + _forward_args(args)
        commands.append((str(gpu), command))

    processes = []
    for gpu, command in commands:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONPATH"] = str(EVAL_DIR) + os.pathsep + env.get("PYTHONPATH", "")
        processes.append(subprocess.Popen(command, env=env))
    exit_codes = [proc.wait() for proc in processes]
    failed = [code for code in exit_codes if code != 0]
    if failed:
        raise SystemExit(f"{len(failed)} GPU workers failed.")

    records: list[dict[str, Any]] = []
    for shard_path in sorted((run_root / "shards").glob("shard_*/per_case.jsonl")):
        with shard_path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                if line.strip():
                    records.append(json.loads(line))
    return finalize_run(records, run_root, args)


def worker_python(args: argparse.Namespace) -> str:
    if args.prm == "robometer":
        return str(args.robometer_python)
    return sys.executable


def _forward_args(args: argparse.Namespace) -> list[str]:
    forwarded: list[str] = []
    for name in [
        "manifest",
        "output_root",
        "goal_fallback",
        "prm",
        "prm_path",
        "frame_interval",
        "batch_size",
        "tensor_parallel_size",
        "eval_mode",
        "robometer_server_url",
        "robometer_server_host",
        "robometer_server_port",
        "robometer_root",
        "robometer_python",
        "robometer_num_gpus",
        "robometer_max_workers",
        "robometer_frame_steps_micro_batch_size",
        "robometer_startup_timeout_s",
        "robometer_fps",
        "robometer_view",
        "robometer_timeout_s",
        "normalize",
        "source_type",
        "outlier_method",
        "smoothing",
        "smoothing_weights",
        "success_threshold",
        "success_source",
    ]:
        value = getattr(args, name)
        option = "--" + name.replace("_", "-")
        if value is not None:
            forwarded.extend([option, str(value)])
    for flag in ["keep_cache", "skip_existing", "allow_partial"]:
        if getattr(args, flag):
            forwarded.append("--" + flag.replace("_", "-"))
    if getattr(args, "robometer_use_frame_steps", True):
        forwarded.append("--robometer-use-frame-steps")
    else:
        forwarded.append("--no-robometer-use-frame-steps")
    if getattr(args, "robometer_auto_start", None) is True:
        forwarded.append("--robometer-auto-start")
    elif getattr(args, "robometer_auto_start", None) is False:
        forwarded.append("--no-robometer-auto-start")
    for name in ["benchmark", "task_filter", "model_filter", "case_filter"]:
        values = getattr(args, name)
        if values:
            forwarded.append("--" + name.replace("_", "-"))
            forwarded.extend(str(value) for value in values)
    if args.limit is not None:
        forwarded.extend(["--limit", str(args.limit)])
    return forwarded


def _worker_shard_root(args: argparse.Namespace, run_root: Path) -> Path | None:
    """返回当前多 GPU worker 的过程文件目录；主进程和单卡运行返回 ``None``。"""
    if args.num_shards is not None and args.shard_index is not None:
        return run_root / "shards" / f"shard_{args.shard_index:03d}"
    return None


def resolve_gpus(value: str) -> list[str]:
    if value == "auto":
        detected = detect_cuda_gpus()
        return detected or ["0"]
    return [item.strip() for item in value.split(",") if item.strip()] or ["0"]


def detect_cuda_gpus() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:  # noqa: BLE001
        return []
    gpus: list[str] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not parts:
            continue
        if len(parts) == 1 or int(float(parts[1])) < 1024:
            gpus.append(parts[0])
    return gpus


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
