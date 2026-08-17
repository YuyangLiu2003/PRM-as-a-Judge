from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from .io import read_jsonl
from .metrics import (
    MODEL_SUMMARY_METRICS,
    MetricConfig,
    aggregate_metrics,
    metric_value,
    record_has_drawdown,
    recompute_record_metrics,
    trace_summary,
)

METRIC_COLUMNS = list(MODEL_SUMMARY_METRICS)
ECHARTS_ASSET_PATH = Path(__file__).with_name("assets") / "echarts-5.5.1.min.js"


def visualize_run(
    run_root: Path,
    output_dir: Path | None = None,
    max_cases: int = 0,
    success_threshold: float | None = None,
) -> dict[str, str]:
    """Create reusable visualization artifacts for a completed run."""
    run_root = run_root.resolve()
    records = load_visualization_records(run_root)
    output_dir = (output_dir or run_root / "visualizations").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold = resolve_success_threshold(run_root, success_threshold)
    metric_config = resolve_metric_config(run_root, threshold)
    recompute_record_metrics(records, metric_config)
    prm_backend = resolve_prm_backend(run_root)

    artifacts: dict[str, str] = {}
    artifacts["curve_metrics_csv"] = str(
        write_curve_metrics_csv(records, output_dir, prm_backend, metric_config)
    )
    artifacts["report_html"] = str(
        write_interactive_html_report(
            records,
            run_root,
            output_dir,
            threshold,
            prm_backend,
        )
    )
    try:
        case_plots = render_case_curve_plots(records, output_dir, max_cases, threshold, prm_backend)
        artifacts["case_plots_dir"] = str(output_dir / "cases")
        artifacts["case_plots_csv"] = str(write_case_plots_csv(case_plots, output_dir))
        plot_status = f"{len(case_plots)} per-case curve PNG files generated."
    except ImportError as exc:
        plot_status = f"Per-case curve PNG files skipped because matplotlib is unavailable: {exc}"
    except ValueError as exc:
        plot_status = f"Per-case curve PNG files skipped: {exc}"
    artifacts["report_md"] = str(write_visualization_report(records, output_dir, artifacts, plot_status, threshold))
    return artifacts


def load_visualization_records(run_root: Path) -> list[dict[str, Any]]:
    per_case_path = run_root / "per_case.jsonl"
    if per_case_path.exists():
        return read_jsonl(per_case_path)
    records: list[dict[str, Any]] = []
    for shard_path in sorted((run_root / "shards").glob("shard_*/per_case.jsonl")):
        records.extend(read_jsonl(shard_path))
    if records:
        return records
    raise FileNotFoundError(f"No per_case.jsonl or shards/shard_*/per_case.jsonl found under {run_root}")


def resolve_success_threshold(run_root: Path, override: float | None = None) -> float:
    if override is not None:
        return float(override)
    params_path = run_root / "run_params.json"
    if not params_path.exists():
        return 0.99
    try:
        params = json.loads(params_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0.99
    try:
        return float(params.get("success_threshold", 0.99))
    except (TypeError, ValueError):
        return 0.99


def resolve_prm_backend(run_root: Path) -> str:
    params_path = run_root / "run_params.json"
    if not params_path.exists():
        return ""
    try:
        params = json.loads(params_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    return str(params.get("prm") or "")


def resolve_metric_config(run_root: Path, success_threshold: float) -> MetricConfig:
    params_path = run_root / "run_params.json"
    if not params_path.exists():
        return MetricConfig(success_threshold=success_threshold)
    try:
        params = json.loads(params_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return MetricConfig(success_threshold=success_threshold)
    try:
        threshold = float(params.get("success_threshold", success_threshold))
    except (TypeError, ValueError):
        threshold = success_threshold
    success_source = str(params.get("success_source") or "progress")
    return MetricConfig(success_threshold=threshold, success_source=success_source)


def write_curve_metrics_csv(
    records: list[dict[str, Any]],
    output_dir: Path,
    prm_backend: str = "",
    metric_config: MetricConfig | None = None,
) -> Path:
    """写出逐轨迹曲线指标，并补齐逐轨迹条件指标。"""
    path = output_dir / "curve_metrics.csv"
    cfg = metric_config or MetricConfig()
    columns = [
        "case_id",
        "task_name",
        "task",
        "benchmark",
        "prm_backend",
        "policy_model",
        "label",
        "status",
        "num_points",
    ] + METRIC_COLUMNS
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=columns)
        writer.writeheader()
        for record in records:
            metrics = dict(record.get("metrics") or {})
            if record.get("status") == "ok" and metrics:
                diagnostics = trace_summary(record, cfg)
                metrics["SR"] = diagnostics["SR_trace"]
                metrics["FNS"] = diagnostics["FNS_trace"]
                metrics["SQS"] = diagnostics["SQS_trace"]
                metrics["DRR"] = diagnostics["DRR_trace"]
            processed = record.get("progress_processed") or []
            row = {
                "case_id": record.get("case_id", ""),
                "task_name": record.get("task_name", ""),
                "task": record.get("task") or record.get("case_task", ""),
                "benchmark": record.get("benchmark", ""),
                "prm_backend": prm_backend,
                "policy_model": record.get("model", ""),
                "label": record.get("label", ""),
                "status": record.get("status", ""),
                "num_points": len(processed),
            }
            for key in METRIC_COLUMNS:
                row[key] = metric_value(metrics, key, "")
            writer.writerow(row)
    return path


def render_case_curve_plots(
    records: list[dict[str, Any]],
    output_dir: Path,
    max_cases: int = 0,
    success_threshold: float = 0.99,
    prm_backend: str = "",
) -> list[dict[str, str]]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:
        raise ImportError("install matplotlib to render per-case curve PNG files") from exc

    ok_records = [
        record
        for record in records
        if record.get("status") == "ok" and (record.get("progress_processed") or record.get("progress_raw"))
    ]
    selected = ok_records[:max_cases] if max_cases and max_cases > 0 else ok_records
    if not selected:
        raise ValueError("No successful records with progress curves are available for plotting")

    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, str]] = []
    for index, record in enumerate(selected, start=1):
        raw = _float_list(record.get("progress_raw") or [])
        processed = _float_list(record.get("progress_processed") or [])
        metrics = record.get("metrics") or {}

        fig, curve_axis = plt.subplots(figsize=(10.5, 6.2))
        if raw:
            curve_axis.plot(raw, color="#9ca3af", linewidth=1.1, linestyle="--", label="raw")
        if processed:
            curve_axis.plot(processed, color="#2563eb", linewidth=2.1, label="processed")
            _annotate_curve_points(curve_axis, processed, metrics)
        curve_axis.axhline(
            success_threshold,
            color="#16a34a",
            linewidth=1.0,
            linestyle=":",
            label=f"success threshold={success_threshold:g}",
        )
        curve_axis.set_ylim(-0.02, 1.02)
        curve_axis.set_xlabel("step")
        curve_axis.set_ylabel("progress")
        curve_axis.grid(True, color="#e5e7eb", linewidth=0.7)
        curve_axis.legend(loc=_legend_location(processed), fontsize=9)

        title = _case_title(record, prm_backend)
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

        filename = f"{index:04d}_{_safe_filename(str(record.get('case_id') or 'case'))}.png"
        path = cases_dir / filename
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(
            {
                "case_id": str(record.get("case_id", "")),
                "task_name": str(record.get("task_name", "")),
                "prm_backend": prm_backend,
                "policy_model": str(record.get("model", "")),
                "path": str(path),
            }
        )
    return outputs


def write_case_plots_csv(case_plots: list[dict[str, str]], output_dir: Path) -> Path:
    path = output_dir / "case_plots.csv"
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["case_id", "task_name", "prm_backend", "policy_model", "path"])
        writer.writeheader()
        writer.writerows(case_plots)
    return path


def write_interactive_html_report(
    records: list[dict[str, Any]],
    run_root: Path,
    output_dir: Path,
    success_threshold: float,
    prm_backend: str = "",
) -> Path:
    """将交互报告写入磁盘，视频路径保持为可直接打开的本地地址。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    html = build_interactive_html_report(
        records=records,
        run_root=run_root,
        success_threshold=success_threshold,
        prm_backend=prm_backend,
        output_dir=output_dir,
    )
    path = output_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


def build_interactive_html_report(
    records: list[dict[str, Any]],
    run_root: Path,
    success_threshold: float,
    prm_backend: str = "",
    output_dir: Path | None = None,
    video_src_resolver: Callable[[dict[str, Any]], str] | None = None,
) -> str:
    """构建自包含图表资源的 HTML，供文件写入和 HTTP 服务共同复用。"""
    metric_cfg = resolve_metric_config(run_root, success_threshold)
    aggregate = aggregate_metrics(records, metric_cfg)
    data = {
        "runRoot": str(run_root),
        "prmBackend": prm_backend,
        "successThreshold": metric_cfg.success_threshold,
        "successSource": metric_cfg.success_source,
        "metricColumns": list(MODEL_SUMMARY_METRICS),
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "cases": [
            _html_case_record(
                record,
                prm_backend,
                metric_cfg,
                output_dir,
                video_src_resolver,
            )
            for record in records
        ],
        "leaderboard": aggregate["leaderboard"],
        "groups": aggregate["groups"],
    }
    encoded = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return _interactive_html_template(encoded)


def _html_case_record(
    record: dict[str, Any],
    prm_backend: str,
    metric_cfg: MetricConfig,
    output_dir: Path | None = None,
    video_src_resolver: Callable[[dict[str, Any]], str] | None = None,
) -> dict[str, Any]:
    source_metrics = dict(record.get("metrics") or {})
    missing = object()
    metrics: dict[str, Any] = {}
    for key in MODEL_SUMMARY_METRICS:
        value = metric_value(source_metrics, key, missing)
        if value is not missing:
            metrics[key] = value
    if record.get("status") == "ok" and source_metrics:
        diagnostics = trace_summary(
            {"metrics": source_metrics, "label": record.get("label")},
            metric_cfg,
        )
        metrics["SR"] = diagnostics["SR_trace"]
        metrics["FNS"] = diagnostics["FNS_trace"]
        metrics["SQS"] = diagnostics["SQS_trace"]
    return {
        "case_id": record.get("case_id", ""),
        "task_name": record.get("task_name") or record.get("case_task") or record.get("task") or "",
        "task": record.get("task") or record.get("case_task") or "",
        "benchmark": record.get("benchmark", ""),
        "model": record.get("model", ""),
        "label": record.get("label", ""),
        "status": record.get("status", ""),
        "error": record.get("error", ""),
        "output_dir": record.get("output_dir", ""),
        "prm_backend": (record.get("metadata") or {}).get("backend") or record.get("prm_backend") or prm_backend,
        "video": (
            video_src_resolver(record)
            if video_src_resolver is not None
            else _select_video_src(record, output_dir)
        ),
        "progress_raw": _float_list(record.get("progress_raw") or []),
        "progress_processed": _float_list(record.get("progress_processed") or []),
        "has_drawdown": (
            record_has_drawdown(record, metric_cfg)
            if record.get("status") == "ok"
            else False
        ),
        "metrics": {key: _as_float(value) for key, value in metrics.items()},
    }


def _select_video_src(record: dict[str, Any], output_dir: Path | None = None) -> str:
    return _path_to_video_src(select_video_value(record), output_dir)


def select_video_value(record: dict[str, Any]) -> str:
    """按公共 manifest 视角优先级返回案例的主视频原始路径。"""
    videos = record.get("videos") or {}
    if isinstance(videos, dict):
        for key in ["video", "high", "cam_high", "front"]:
            if videos.get(key):
                return str(videos[key])
        for value in videos.values():
            if value:
                return str(value)
    if record.get("video"):
        return str(record["video"])
    return ""


def _path_to_video_src(value: Any, output_dir: Path | None = None) -> str:
    """将视频路径转换为适合嵌入 HTML 报告的地址。

    已存在的相对路径以及仓库内绝对路径会基于报告输出目录重新计算，
    便于通过 HTTP 服务查看远端报告；显式 URI 和仓库外绝对路径继续
    保持原有行为。
    """
    text = str(value or "")
    if not text:
        return ""
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", text):
        return text
    try:
        path = Path(text).expanduser()
        if path.exists():
            resolved = path.resolve()
            if output_dir is not None:
                report_dir = output_dir.resolve()
                repository_root = next(
                    (
                        candidate
                        for candidate in (report_dir, *report_dir.parents)
                        if (candidate / ".git").exists()
                    ),
                    None,
                )
                is_repository_asset = (
                    repository_root is not None
                    and resolved.is_relative_to(repository_root)
                )
                if not path.is_absolute() or is_repository_asset:
                    try:
                        return Path(os.path.relpath(resolved, report_dir)).as_posix()
                    except ValueError:
                        pass
            return resolved.as_uri()
    except (OSError, ValueError):
        pass
    return text


@lru_cache(maxsize=1)
def _load_echarts_js() -> str:
    """读取随包分发的 ECharts，并避免内容意外闭合外层 script 标签。"""
    try:
        source = ECHARTS_ASSET_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Bundled ECharts asset is unavailable: {ECHARTS_ASSET_PATH}") from exc
    return re.sub(r"</script", r"<\\/script", source, flags=re.IGNORECASE)


def _interactive_html_template(report_json: str) -> str:
    echarts_js = _load_echarts_js()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PRM-as-a-Judge Visualization Report</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <script>{echarts_js}</script>
  <style>
    :root {{
      --ink: #172033;
      --muted: #667085;
      --line: #e4e8ef;
      --soft: #f6f8fb;
      --green: #2f8f46;
      --green-soft: rgba(129, 199, 132, 0.28);
      --red: #bf3a35;
      --red-soft: rgba(229, 92, 87, 0.16);
      --blue: #6aa9d8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #fafafa;
      color: #1a1a1a;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      line-height: 1.7;
    }}
    button, input, select {{ font: inherit; }}
    .page {{ max-width: 1180px; margin: 0 auto; padding: 1.6rem 1.5rem 3rem; }}
    .hero {{ max-width: none; margin: 0 0 1.15rem; display: block; }}
    .hero-copy {{ max-width: none; }}
    .hero h1 {{
      margin: 0 0 0.45rem;
      color: #1a1a1a;
      font-family: Georgia, "Times New Roman", serif;
      font-size: 1.8rem;
      font-weight: 600;
    }}
    .hero p {{ margin: 0; color: #5f5f5f; font-size: 0.98rem; line-height: 1.55; max-width: 62rem; }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 0.72rem;
      margin: 0 0 0.9rem;
    }}
    .block-title {{ margin: 0 0 0.5rem; color: #243044; font-family: Georgia, "Times New Roman", serif; font-size: 1.26rem; font-weight: 600; }}
    .summary-card {{
      padding: 0.68rem 0.76rem;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      box-shadow: 0 1px 3px rgba(16, 24, 40, 0.04);
    }}
    .summary-card.wide {{ grid-column: span 2; }}
    .summary-label {{
      display: block;
      color: #667085;
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.035em;
    }}
    .summary-value {{
      display: block;
      margin-top: 0.12rem;
      color: #1f2937;
      font-size: 1.05rem;
      font-weight: 750;
      font-variant-numeric: tabular-nums;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .summary-value.small {{ font-size: 0.86rem; font-weight: 650; }}
    .issue-card {{ padding: 0.82rem 0.9rem; margin: 0 0 1rem; }}
    .issue-head {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 1rem;
      margin-bottom: 0.42rem;
    }}
    .issue-head h2 {{ margin: 0; color: #243044; font-size: 1.02rem; }}
    .issue-badge {{
      border: 1px solid #d8dee8;
      border-radius: 999px;
      padding: 0.1rem 0.48rem;
      color: #5b6472;
      background: #f7f8fa;
      font-size: 0.76rem;
      font-weight: 700;
      white-space: nowrap;
    }}
    .issue-badge.warn {{ border-color: #f0c4c3; background: #fff5f5; color: #b32724; }}
    .issue-text {{ margin: 0; color: #667085; font-size: 0.88rem; line-height: 1.45; }}
    .issue-list {{ display: grid; gap: 0.34rem; margin-top: 0.52rem; }}
    .issue-actions {{ display: flex; justify-content: flex-end; margin-top: 0.5rem; }}
    .issue-toggle, .page-button {{ border: 1px solid #d8dee8; border-radius: 6px; background: #fff; color: #344054; padding: 0.34rem 0.62rem; font-size: 0.78rem; cursor: pointer; }}
    .issue-toggle:hover, .page-button:hover:not(:disabled) {{ background: #f5f8fc; }}
    .issue-row {{
      display: grid;
      grid-template-columns: minmax(9rem, 1fr) minmax(10rem, 1fr) minmax(12rem, 2fr);
      gap: 0.6rem;
      align-items: baseline;
      padding: 0.36rem 0.46rem;
      border: 1px solid #edf0f5;
      border-radius: 6px;
      background: #fcfdff;
      color: #344054;
      font-size: 0.82rem;
    }}
    .issue-row strong {{ color: #1f2937; }}
    .section-heading {{ max-width: 1020px; margin: 1.45rem auto 0.6rem; }}
    .section-heading h2 {{
      margin: 0 0 0.16rem;
      color: #243044;
      font-family: Georgia, "Times New Roman", serif;
      font-size: 1.26rem;
      font-weight: 600;
    }}
    .section-heading p {{ margin: 0; color: #667085; font-size: 0.9rem; line-height: 1.45; max-width: 42rem; }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 560px) minmax(390px, 440px);
      gap: 1.15rem;
      align-items: start;
      justify-content: center;
    }}
    .card {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
    }}
    .left-stack {{ min-width: 0; }}
    .viewer {{ overflow: hidden; }}
    .viewer video {{
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: contain;
      background: #111;
    }}
    .video-footer {{
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto auto;
      gap: 0.7rem;
      align-items: center;
      padding: 0.55rem 0.75rem;
      border-top: 1px solid var(--line);
      background: #fbfcfe;
    }}
    .play-button {{
      min-width: 4.6rem;
      padding: 0.36rem 0.62rem;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: #fff;
      color: #263244;
      font-weight: 600;
      cursor: pointer;
    }}
    .play-button:hover {{ background: #f3f6fb; }}
    .timeline {{ width: 100%; accent-color: #2f5fa7; }}
    .time-label {{ color: #5f6b7a; font-size: 0.86rem; font-variant-numeric: tabular-nums; white-space: nowrap; }}
    .speed-select {{
      padding: 0.3rem 0.45rem;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      background: #fff;
      color: #263244;
      font-size: 0.86rem;
      font-weight: 600;
    }}
    .chart-card {{ margin-top: 0.85rem; padding: 0.7rem 0.8rem 0.65rem; }}
    .chart-head {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 1rem;
      margin-bottom: 0.25rem;
    }}
    .chart-title {{ font-weight: 650; color: #243044; font-size: 0.92rem; }}
    .state-chip {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 4.8rem;
      padding: 0.14rem 0.4rem;
      border-radius: 999px;
      border: 1px solid #d9e1ec;
      background: #f8fafc;
      color: #4b5563;
      font-size: 0.72rem;
      font-weight: 650;
    }}
    .state-chip.state-progress {{ border-color: #bfe3c3; background: #eef9f0; color: #287a37; }}
    .state-chip.state-regress {{ border-color: #f0c4c3; background: #fff1f1; color: #b32724; }}
    .state-chip.state-stall {{ border-color: #d8dee8; background: #f7f8fa; color: #5b6472; }}
    .case-chart {{ width: 100%; height: 225px; cursor: ew-resize; }}
    .metrics {{ align-self: start; padding: 0.74rem; }}
    .controls {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.46rem;
      margin-bottom: 0.52rem;
    }}
    label {{ display: block; margin-bottom: 0.28rem; color: #555; font-size: 0.86rem; font-weight: 600; }}
    select, input[type="search"] {{
      width: 100%;
      border: 1px solid #d8dee8;
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 0.46rem 0.6rem;
      font-size: 0.94rem;
    }}
    .section-title {{
      margin: 0.46rem 0 0.28rem;
      color: #3e4a5e;
      font-size: 0.86rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .metric-subsection-title {{
      margin: 0.34rem 0 0.25rem;
      color: #687386;
      font-size: 0.74rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.37rem; margin-bottom: 0.56rem; }}
    .metric-grid.live {{ grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.27rem; }}
    .metric-grid.triple {{ grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.29rem; }}
    .tile {{ border: 1px solid #e3e8f0; border-radius: 7px; padding: 0.36rem 0.44rem; background: #fbfcfe; }}
    .live .tile {{ border-color: #d8e7fb; background: #f4f8ff; padding: 0.29rem 0.32rem; }}
    .good .tile {{ border-color: #d8eadf; background: #f5fbf7; }}
    .bad .tile {{ border-color: #efd8d7; background: #fff5f5; }}
    .key {{ display: block; color: #697586; font-size: 0.78rem; font-weight: 650; }}
    .value {{ display: block; margin-top: 0.04rem; color: #1f2937; font-size: 0.98rem; font-weight: 700; font-variant-numeric: tabular-nums; }}
    .live .key {{ font-size: 0.66rem; white-space: nowrap; }}
    .live .value {{ font-size: 0.82rem; }}
    .triple .tile {{ padding: 0.31rem 0.36rem; }}
    .triple .key {{ font-size: 0.68rem; }}
    .triple .value {{ font-size: 0.84rem; }}
    .milestones {{ display: flex; flex-wrap: wrap; gap: 0.3rem; margin-bottom: 0.2rem; }}
    .milestone {{
      border: 1px solid #e1e6ee;
      border-radius: 5px;
      background: #f4f6f8;
      color: #7a8493;
      font-size: 0.76rem;
      font-weight: 700;
      line-height: 1;
      min-width: 3.1rem;
      padding: 0.24rem 0.58rem;
      text-align: center;
    }}
    .milestone.reached {{ border-color: #bfe3c3; background: var(--green-soft); color: #2f7d3a; }}
    .tables {{ margin: 0 0 1.55rem; display: grid; gap: 18px; }}
    .metric-guidance {{
      margin: 0 0 -0.2rem;
      padding-left: 1.05rem;
      color: #667085;
      font-size: 0.9rem;
      line-height: 1.45;
    }}
    .metric-guidance li {{ margin: 0.12rem 0; }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.9rem;
    }}
    .metric-chart-card {{ padding: 0.82rem 0.92rem 0.78rem; }}
    .metric-chart-card.wide {{ grid-column: 1 / -1; }}
    .metric-chart-head {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 1rem;
      margin-bottom: 0.34rem;
    }}
    .metric-chart-head h3 {{ margin: 0; color: #243044; font-size: 1rem; }}
    .metric-chart-note {{ color: #667085; font-size: 0.78rem; white-space: nowrap; }}
    .metric-chart {{ width: 100%; height: 245px; }}
    .metric-chart.wide-chart {{ height: 285px; }}
    .conditional-profile-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.9rem; }}
    .conditional-panel {{ min-width: 0; }}
    .conditional-panel-title {{ margin: 0 0 0.1rem; color: #475467; font-size: 0.8rem; font-weight: 600; }}
    .conditional-chart {{ height: 250px; }}
    .failure-toolbar {{ display: flex; align-items: end; justify-content: space-between; gap: 1.2rem; margin-bottom: 0.34rem; }}
    .failure-note {{ flex: 1 1 520px; margin: 0.24rem 0 0; color: #667085; font-size: 0.76rem; line-height: 1.4; max-width: 650px; }}
    .failure-filters {{ display: flex; flex: 0 0 auto; align-items: end; justify-content: flex-end; gap: 0.45rem; }}
    .failure-filter {{ display: grid; gap: 0.12rem; margin: 0; color: #667085; font-size: 0.7rem; }}
    .failure-filter select {{ width: 142px; max-width: 100%; padding: 0.28rem 1.6rem 0.28rem 0.42rem; border: 1px solid #d7dee8; border-radius: 6px; background: #fff; color: #344054; font-size: 0.78rem; }}
    .failure-chart-layout {{ display: grid; grid-template-columns: minmax(0, 800px) 180px; justify-content: center; gap: 1.05rem; align-items: end; width: 100%; }}
    .failure-chart-layout .wide-chart {{ height: 285px; align-self: end; }}
    .failure-stats {{ display: grid; grid-template-rows: repeat(3, 1fr); gap: 0.55rem; height: 285px; padding: 0.35rem 0; }}
    .failure-stat {{ display: flex; flex-direction: column; justify-content: center; padding: 0.65rem 0.75rem; border: 1px solid #e4e9f1; border-radius: 7px; background: #f7faff; }}
    .failure-stat-label {{ color: #667085; font-size: 0.75rem; }}
    .failure-stat-value {{ margin-top: 0.15rem; color: #243044; font-size: 1.1rem; font-weight: 650; }}
    .table-card {{ padding: 14px; }}
    .table-head {{ display: flex; align-items: end; justify-content: space-between; gap: 14px; margin-bottom: 10px; }}
    .table-head h2 {{ margin: 0; font-size: 18px; }}
    .table-note {{ margin: 3px 0 0; color: var(--muted); font-size: 13px; }}
    .table-wrap {{ overflow: hidden; border: 1px solid var(--line); border-radius: 7px; }}
    .table-pagination {{ display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; margin-top: 0.6rem; color: #667085; font-size: 0.78rem; }}
    .pagination-actions {{ display: flex; align-items: center; gap: 0.4rem; }}
    .page-button:disabled {{ cursor: default; opacity: 0.45; }}
    table {{ width: 100%; border-collapse: collapse; table-layout: fixed; font-size: 12px; }}
    th, td {{
      padding: 7px 5px;
      border-bottom: 1px solid #edf0f5;
      text-align: center;
      white-space: nowrap;
      max-width: 8rem;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    th {{ position: sticky; top: 0; background: #f7f9fc; color: #3e4a5e; cursor: pointer; user-select: none; }}
    th:hover {{ background: #eef3f8; }}
    th.num, td.num {{ max-width: 4.8rem; text-align: center; font-variant-numeric: tabular-nums; }}
    th.col-benchmark, td.col-benchmark {{ max-width: 7rem; }}
    th.col-model, td.col-model {{ max-width: 7rem; }}
    th.col-task, td.col-task {{
      max-width: 13rem;
    }}
    th.col-evaluator, td.col-evaluator {{ max-width: 7rem; }}
    th.col-case-id, td.col-case-id {{ max-width: 9rem; }}
    th.col-status, td.col-status,
    th.col-label, td.col-label {{ max-width: 6rem; }}
    th.metric-boundary, td.metric-boundary {{ border-left: 0.5px solid #dbe2ec; }}
    tr.case-clickable {{ cursor: pointer; }}
    tr.case-clickable:hover td {{ background: #f8fbff; }}
    .empty {{ color: var(--muted); padding: 18px; }}
    @media (max-width: 940px) {{
      .summary-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .summary-card.wide {{ grid-column: span 2; }}
      .issue-row {{ grid-template-columns: 1fr; gap: 0.12rem; }}
      .chart-grid {{ grid-template-columns: 1fr; }}
      .metric-chart-card.wide {{ grid-column: auto; }}
      .conditional-profile-grid {{ grid-template-columns: 1fr; }}
      .layout {{ grid-template-columns: 1fr; }}
      .controls {{ grid-template-columns: 1fr; }}
      .failure-chart-layout {{ grid-template-columns: 1fr; }}
      .failure-toolbar {{ align-items: flex-start; flex-direction: column; }}
      .failure-note {{ flex: none; max-width: none; }}
      .failure-filters {{ justify-content: flex-start; }}
      .failure-stats {{ grid-template-columns: repeat(3, minmax(0, 1fr)); grid-template-rows: auto; height: auto; padding: 0; }}
    }}
    @media (max-width: 520px) {{
      .metric-chart-head {{ align-items: flex-start; flex-direction: column; gap: 0.15rem; }}
      .metric-chart-note {{ white-space: normal; }}
      .metric-grid.live, .metric-grid.triple {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .failure-filters {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); width: 100%; }}
      .failure-filter select {{ width: 100%; }}
      .failure-stats {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header class="hero">
      <div class="hero-copy">
        <h1>PRM-as-a-Judge Evaluation Report</h1>
        <p>This local report summarizes the completed PRM-as-a-Judge evaluation run. The tables and charts below provide model-level and model-by-task metrics for comparing policies and locating weak areas, while the trajectory explorer lets you inspect the corresponding rollout video, progress curve, live frame metrics, and milestone states. Videos are loaded from the paths recorded in the run outputs, so they remain playable as long as those local files are still available.</p>
      </div>
    </header>

    <h2 class="block-title">Run Summary</h2>
    <section class="summary-grid" id="run-summary" aria-label="Run summary"></section>

    <section class="card issue-card" id="issue-summary" aria-label="Case coverage summary"></section>

    <h2 class="block-title">Metric Results</h2>
    <section class="tables">
      <ul class="metric-guidance">
        <li>Charts aggregate OK cases with complete required metrics; exceptions and missing data appear in Case Coverage.</li>
        <li>Metrics follow the same order as metrics.xlsx: M25, M50, M75, SR, MP, PPL, CRA, STR, DRR, FNS, SQS.</li>
        <li>Scores are shown as percentages. Higher is better except for CRA and STR, where lower is better.</li>
        <li>Aggregate DRR averages only trajectories with a real regression; episode DRR is shown as -- when no regression exceeds the 1e-12 floating-point tolerance.</li>
        <li>Failure Progress &amp; Recovery groups failed trajectories by MP. Its dashed marker is their mean MP, while the DRR line uses only failed trajectories with a real regression.</li>
        <li>Charts and tables default to SR descending; charts show only the leading models when needed, while tables retain all searchable and sortable rows.</li>
        <li>Conditional profiles show the applicable sample count n. Small subsets are diagnostic only and should not be used for direct comparison.</li>
      </ul>
      <div class="chart-grid" aria-label="Metric visualizations">
        <div class="card metric-chart-card wide">
          <div class="metric-chart-head">
            <h3>Milestone Reachability</h3>
            <span class="metric-chart-note" id="milestone-chart-note">M25, M50, M75, SR</span>
          </div>
          <div id="milestone-chart" class="metric-chart" role="img" aria-label="Model-level milestone reachability profile"></div>
        </div>
        <div class="card metric-chart-card wide">
          <div class="metric-chart-head">
            <h3>Success / Failure Conditional Profile</h3>
            <span class="metric-chart-note" id="conditional-chart-note">Applicable outcome subsets</span>
          </div>
          <div class="conditional-profile-grid">
            <div class="conditional-panel">
              <div class="conditional-panel-title" id="failure-profile-title">Failed trajectories</div>
              <div id="failure-profile-chart" class="metric-chart conditional-chart" role="img" aria-label="Failure-conditioned FNS and MP profile"></div>
            </div>
            <div class="conditional-panel">
              <div class="conditional-panel-title" id="success-profile-title">Successful trajectories</div>
              <div id="success-profile-chart" class="metric-chart conditional-chart" role="img" aria-label="Success-conditioned SQS and PPL profile"></div>
            </div>
          </div>
        </div>
        <div class="card metric-chart-card wide">
          <div class="metric-chart-head">
            <h3>Failure Progress &amp; Recovery</h3>
            <span class="metric-chart-note" id="failure-recovery-note">Failed-case MP distribution and regression-conditioned DRR</span>
          </div>
          <div class="failure-toolbar">
            <p class="failure-note">Bars show failed trajectories grouped by MP. The dashed MP (failed cases) marker shows their mean progress; the DRR line averages recovery only among failed trajectories with a real regression.</p>
            <div class="failure-filters" aria-label="Failure chart filters">
              <label class="failure-filter">Model<select id="failure-model-filter"></select></label>
              <label class="failure-filter">Task<select id="failure-task-filter"></select></label>
            </div>
          </div>
          <div class="failure-chart-layout">
            <div id="failure-recovery-chart" class="metric-chart wide-chart" role="img" aria-label="Histogram of failed cases by MP with a DRR recovery line"></div>
            <div class="failure-stats" aria-label="Failure diagnostics">
              <div class="failure-stat"><span class="failure-stat-label">Failed Cases</span><span class="failure-stat-value" id="failure-count">--</span></div>
              <div class="failure-stat"><span class="failure-stat-label">MP (failed cases)</span><span class="failure-stat-value" id="failure-mp">--</span></div>
              <div class="failure-stat"><span class="failure-stat-label">DRR Cases</span><span class="failure-stat-value" id="failure-drr-count">--</span></div>
            </div>
          </div>
        </div>
      </div>
      <div class="card table-card">
        <div class="table-head">
          <div><h2>Model Leaderboard</h2><p class="table-note">Aggregated by benchmark and model.</p></div>
          <input type="search" id="leaderboard-filter" placeholder="Filter rows">
        </div>
        <div class="table-wrap"><table id="leaderboard-table"></table></div>
        <div class="table-pagination" id="leaderboard-pagination"></div>
      </div>
      <div class="card table-card">
        <div class="table-head">
          <div><h2>Model x Task Metrics</h2><p class="table-note">Aggregated by benchmark, model, and task.</p></div>
          <input type="search" id="group-filter" placeholder="Filter rows">
        </div>
        <div class="table-wrap"><table id="group-table"></table></div>
        <div class="table-pagination" id="group-pagination"></div>
      </div>
    </section>

    <div class="section-heading">
      <h2>Interactive Trajectory Explorer</h2>
      <p>Inspect rollout videos, progress curves, live frame metrics, and milestone states for individual cases.</p>
    </div>

    <section class="layout">
      <div class="left-stack">
        <div class="card viewer">
          <video id="case-video" preload="metadata"></video>
          <div class="video-footer">
            <button class="play-button" id="play-button" type="button">Play</button>
            <input class="timeline" id="timeline" type="range" min="0" max="1000" value="0" aria-label="Video timeline">
            <span class="time-label" id="time-label">0:00 / 0:00</span>
            <select class="speed-select" id="speed-select" aria-label="Playback speed">
              <option value="0.5">0.5x</option>
              <option value="1" selected>1x</option>
              <option value="1.5">1.5x</option>
              <option value="2">2x</option>
            </select>
          </div>
        </div>
        <div class="card chart-card">
          <div class="chart-head">
            <div class="chart-title">Progress Curve</div>
            <span class="state-chip" id="state-chip">Stall</span>
          </div>
          <div id="case-chart" class="case-chart" role="img" aria-label="Progress curve"></div>
        </div>
      </div>

      <aside class="card metrics">
        <div class="controls">
          <div><label for="model-select">Model</label><select id="model-select"></select></div>
          <div><label for="task-select">Task</label><select id="task-select"></select></div>
          <div><label for="case-select">Case</label><select id="case-select"></select></div>
          <div><label for="backend-select">Evaluator</label><select id="backend-select"></select></div>
        </div>

        <div class="section-title">Live Frame</div>
        <div class="metric-grid live">
          <div class="tile"><span class="key">Progress</span><span class="value" id="live-progress">--</span></div>
          <div class="tile"><span class="key">Best So Far</span><span class="value" id="live-best">--</span></div>
          <div class="tile"><span class="key">Slope</span><span class="value" id="live-slope">--</span></div>
          <div class="tile"><span class="key">Frame</span><span class="value" id="live-frame">--</span></div>
        </div>

        <div class="section-title">Episode Metrics</div>
        <div class="metric-grid triple good">
          <div class="tile"><span class="key">M25 &#8593;</span><span class="value" id="metric-m25">--</span></div>
          <div class="tile"><span class="key">M50 &#8593;</span><span class="value" id="metric-m50">--</span></div>
          <div class="tile"><span class="key">M75 &#8593;</span><span class="value" id="metric-m75">--</span></div>
        </div>
        <div class="metric-grid triple good" style="margin-top:7px">
          <div class="tile"><span class="key">SR &#8593;</span><span class="value" id="metric-sr">--</span></div>
          <div class="tile"><span class="key">MP &#8593;</span><span class="value" id="metric-mp">--</span></div>
          <div class="tile"><span class="key">PPL &#8593;</span><span class="value" id="metric-ppl">--</span></div>
        </div>
        <div class="metric-grid bad" style="margin-top:7px">
          <div class="tile"><span class="key">CRA &#8595;</span><span class="value" id="metric-cra">--</span></div>
          <div class="tile"><span class="key">STR &#8595;</span><span class="value" id="metric-str">--</span></div>
        </div>
        <div class="metric-grid triple good" style="margin-top:7px">
          <div class="tile" title="Defined only for trajectories with a real regression"><span class="key">DRR &#8593;</span><span class="value" id="metric-drr">--</span></div>
          <div class="tile"><span class="key">FNS &#8593;</span><span class="value" id="metric-fns">--</span></div>
          <div class="tile"><span class="key">SQS &#8593;</span><span class="value" id="metric-sqs">--</span></div>
        </div>

        <div class="metric-subsection-title">Live Milestone State</div>
        <div class="milestones">
          <span class="milestone" data-ms="25">25%</span>
          <span class="milestone" data-ms="50">50%</span>
          <span class="milestone" data-ms="75">75%</span>
          <span class="milestone" data-ms="100">100%</span>
        </div>
      </aside>
    </section>

  </main>

  <script>
    const REPORT_DATA = {report_json};
    const metricColumns = REPORT_DATA.metricColumns || [];
    const pctColumns = new Set(metricColumns);
    const state = {{
      case: null,
      dragging: false,
      issueExpanded: false,
      page: {{ 'leaderboard-table': 1, 'group-table': 1 }},
      sort: {{
        'leaderboard-table': {{ col: 'SR', dir: -1 }},
        'group-table': {{ col: 'SR', dir: -1 }},
        'case-table': {{ col: 'model', dir: 1 }}
      }}
    }};
    const video = document.getElementById('case-video');
    const playButton = document.getElementById('play-button');
    const timeline = document.getElementById('timeline');
    const speedSelect = document.getElementById('speed-select');
    const stateChip = document.getElementById('state-chip');
    const chartDom = document.getElementById('case-chart');
    let chart = null;
    let failureRecoveryChart = null;
    const metricCharts = [];

    function allCases() {{ return REPORT_DATA.cases || []; }}
    function okCases() {{ return allCases().filter(c => c.status === 'ok' && (c.progress_processed || []).length); }}
    function hasCurve(item) {{ return (item.progress_processed || []).length > 0; }}
    function hasVideo(item) {{ return Boolean(item.video); }}
    function uniq(rows, key) {{ return [...new Set(rows.map(r => r[key] || '').filter(Boolean))].sort(); }}
    function uniqueCount(rows, key) {{ return uniq(rows, key).length; }}
    function renderRunSummary() {{
      const cases = allCases();
      const ok = cases.filter(c => c.status === 'ok').length;
      const cards = [
        {{ label: 'Total Cases', value: `${{ok}} OK / ${{cases.length}} Total` }},
        {{ label: 'Models', value: uniqueCount(cases, 'model') }},
        {{ label: 'Tasks', value: uniqueCount(cases, 'task_name') }},
        {{ label: 'Evaluators', value: uniqueCount(cases, 'prm_backend') }},
        {{ label: 'Success Source / Threshold', value: `${{REPORT_DATA.successSource || 'progress'}} / ${{fmtNum(REPORT_DATA.successThreshold)}}` }},
        {{ label: 'Generated', value: fmtDate(REPORT_DATA.generatedAt), small: true }},
        {{ label: 'Run Root', value: REPORT_DATA.runRoot || '-', small: true, wide: true }}
      ];
      document.getElementById('run-summary').innerHTML = cards.map(card =>
        `<div class="summary-card${{card.wide ? ' wide' : ''}}"><span class="summary-label">${{escapeHtml(card.label)}}</span><span class="summary-value${{card.small ? ' small' : ''}}" title="${{escapeAttr(card.value)}}">${{escapeHtml(card.value)}}</span></div>`
      ).join('');
    }}
    function caseIssues(item) {{
      const issues = [];
      if ((item.status || '') !== 'ok') issues.push(`status=${{item.status || 'unknown'}}`);
      if (!hasVideo(item)) issues.push('missing video');
      if (!hasCurve(item)) issues.push('missing progress curve');
      const requiredMetrics = metricColumns;
      const missingMetrics = requiredMetrics.filter(key => !hasFiniteMetrics(item.metrics, [key]));
      if (missingMetrics.length) issues.push(`missing metrics: ${{missingMetrics.join(', ')}}`);
      return issues;
    }}
    function renderIssueSummary() {{
      const rows = allCases().map(item => ({{ item, issues: caseIssues(item) }})).filter(row => row.issues.length);
      const target = document.getElementById('issue-summary');
      if (!rows.length) {{
        target.innerHTML = '<div class="issue-head"><h2>Case Coverage</h2><span class="issue-badge">All clear</span></div><p class="issue-text">All cases in this report have an OK status, complete required metrics, a progress curve, and an available video path.</p>';
        return;
      }}
      const shown = state.issueExpanded ? rows : rows.slice(0, 8);
      const toggle = rows.length > 8
        ? `<div class="issue-actions"><button type="button" class="issue-toggle" id="issue-toggle">${{state.issueExpanded ? 'Show first 8' : `Show all ${{rows.length}}`}}</button></div>`
        : '';
      target.innerHTML = `<div class="issue-head"><h2>Case Coverage</h2><span class="issue-badge warn">${{rows.length}} case${{rows.length === 1 ? '' : 's'}} need attention</span></div>` +
        '<p class="issue-text">Cases listed here failed during evaluation or are missing video/progress data needed for trajectory review.</p>' +
        `<div class="issue-list">${{shown.map(row => {{
          const detail = row.issues.concat(row.item.error ? [`error: ${{row.item.error}}`] : []).join('; ');
          return `<div class="issue-row"><span><strong>${{escapeHtml(row.item.case_id || 'case')}}</strong></span><span>${{escapeHtml(row.item.model || '-')}} / ${{escapeHtml(row.item.task_name || '-')}}</span><span>${{escapeHtml(detail)}}</span></div>`;
        }}).join('')}}</div>` + toggle;
      document.getElementById('issue-toggle')?.addEventListener('click', () => {{
        state.issueExpanded = !state.issueExpanded;
        renderIssueSummary();
      }});
    }}
    function initMetricCharts() {{
      if (!window.echarts) return;
      renderMilestoneReachability();
      renderConditionalProfiles();
      initFailureProgressRecovery();
    }}
    function registerMetricChart(domId) {{
      const dom = document.getElementById(domId);
      if (!dom) return null;
      const instance = echarts.init(dom);
      metricCharts.push(instance);
      return instance;
    }}
    function metricLeaderboardRows() {{
      return (REPORT_DATA.leaderboard || []).slice().sort((a, b) =>
        Number(b.SR || 0) - Number(a.SR || 0) || policyLabel(a).localeCompare(policyLabel(b))
      );
    }}
    function hasFiniteMetrics(row, keys) {{
      return keys.every(key => row?.[key] !== null && row?.[key] !== undefined && row?.[key] !== '' && Number.isFinite(Number(row[key])));
    }}
    function hasFiniteCaseMetrics(row, keys) {{ return hasFiniteMetrics(row?.metrics, keys); }}
    function scopeNote(base, shown, total) {{ return total > shown ? `${{base}} · Top ${{shown}} of ${{total}} by SR` : base; }}
    function policyLabel(row) {{
      const benchmark = row.benchmark || '';
      return benchmark ? `${{benchmark}} / ${{row.model || 'model'}}` : (row.model || 'model');
    }}
    function meanMetric(rows, key) {{
      const values = rows.map(row => Number(row.metrics?.[key])).filter(Number.isFinite);
      return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
    }}
    function caseIsSuccess(row) {{ return Number(row.metrics?.SR_trace ?? row.metrics?.SR ?? 0) >= 1; }}
    function caseHasRegression(row) {{
      if (typeof row?.has_drawdown === 'boolean') return row.has_drawdown;
      const processed = row?.progress_processed || [];
      const raw = row?.progress_raw || [];
      const progress = processed.length ? processed : raw;
      if (!progress.length) return false;
      let peak = Number(progress[0]);
      if (!Number.isFinite(peak)) return false;
      for (let index = 1; index < progress.length; index += 1) {{
        const value = Number(progress[index]);
        if (!Number.isFinite(value)) continue;
        if (peak - value > 1e-12) return true;
        if (value > peak) peak = value;
      }}
      return false;
    }}
    function policyCaseGroups() {{
      const grouped = new Map();
      allCases().filter(row => row.status === 'ok' && row.metrics).forEach(row => {{
        const key = `${{row.benchmark || ''}}\\u0000${{row.model || 'model'}}`;
        if (!grouped.has(key)) grouped.set(key, {{ benchmark: row.benchmark || '', model: row.model || 'model', items: [] }});
        grouped.get(key).items.push(row);
      }});
      return [...grouped.values()].map(group => {{
        const success = group.items.filter(caseIsSuccess);
        const failure = group.items.filter(row => !caseIsSuccess(row));
        return Object.assign(group, {{
          label: policyLabel(group), success, failure,
          SR: meanMetric(group.items, 'SR') || 0
        }});
      }}).sort((a, b) => b.SR - a.SR || a.label.localeCompare(b.label));
    }}
    function renderMilestoneReachability() {{
      const allRows = metricLeaderboardRows().filter(row => hasFiniteMetrics(row, ['M25', 'M50', 'M75', 'SR']));
      const rows = allRows.slice(0, 8);
      const chartInstance = registerMetricChart('milestone-chart');
      if (!chartInstance) return;
      if (!rows.length) {{ chartInstance.setOption(emptyChartOption('No complete leaderboard rows')); return; }}
      setText('milestone-chart-note', scopeNote('M25, M50, M75, SR', rows.length, allRows.length));
      const palette = ['#335c9f', '#2f8f46', '#b46d28', '#7a5aa6', '#2f8790', '#a84d61', '#65758b', '#9b7b45'];
      chartInstance.setOption({{
        animation: false,
        color: palette,
        tooltip: {{ trigger: 'axis', valueFormatter: value => fmtPct(value) }},
        legend: {{ type: 'scroll', top: 0, left: 8, right: 8, textStyle: {{ color: '#667085', fontSize: 10 }} }},
        grid: {{ left: 58, right: 40, top: 40, bottom: 34 }},
        xAxis: {{ type: 'category', boundaryGap: false, data: ['25%', '50%', '75%', 'Success'], axisLabel: {{ color: '#667085', fontSize: 11 }} }},
        yAxis: {{ type: 'value', name: 'Case coverage', nameLocation: 'middle', nameGap: 42, nameRotate: 90, min: 0, max: 1, nameTextStyle: {{ color: '#667085', fontSize: 10 }}, axisLabel: {{ color: '#667085', fontSize: 10, formatter: value => `${{Math.round(value * 100)}}%` }}, splitLine: {{ lineStyle: {{ color: '#e8edf4' }} }} }},
        series: rows.map((row, index) => ({{
          name: policyLabel(row), type: 'line', smooth: false, symbol: 'circle', symbolSize: 6,
          data: [Number(row.M25 || 0), Number(row.M50 || 0), Number(row.M75 || 0), Number(row.SR || 0)],
          lineStyle: {{ width: 2 }}, itemStyle: {{ color: palette[index % palette.length] }}
        }}))
      }});
    }}
    function renderConditionalProfiles() {{
      const allGroups = policyCaseGroups();
      const groups = allGroups.slice(0, 10);
      setText('conditional-chart-note', scopeNote('Applicable outcome subsets', groups.length, allGroups.length));
      renderConditionalProfileChart('failure-profile-chart', groups, 'failure');
      renderConditionalProfileChart('success-profile-chart', groups, 'success');
    }}
    function conditionalSubset(group, outcome) {{
      const required = outcome === 'failure' ? ['FNS', 'MP'] : ['SQS', 'PPL', 'CRA'];
      return group[outcome].filter(row => hasFiniteCaseMetrics(row, required));
    }}
    function renderConditionalProfileChart(domId, groups, outcome) {{
      const chartInstance = registerMetricChart(domId);
      if (!chartInstance) return;
      const subsets = groups.map(group => conditionalSubset(group, outcome));
      const labels = groups.map(group => group.model || 'model');
      const totalCases = subsets.reduce((sum, subset) => sum + subset.length, 0);
      setText(outcome === 'failure' ? 'failure-profile-title' : 'success-profile-title', `${{outcome === 'failure' ? 'Failed' : 'Successful'}} trajectories · n=${{totalCases}}`);
      const hasRows = subsets.some(subset => subset.length);
      if (!hasRows) {{ chartInstance.setOption(emptyChartOption(outcome === 'failure' ? 'No failed trajectories' : 'No successful trajectories')); return; }}
      const isFailure = outcome === 'failure';
      const chartHeight = Math.max(250, groups.length * 38 + 58);
      chartInstance.getDom().style.height = `${{chartHeight}}px`;
      chartInstance.resize();
      const metrics = isFailure
        ? [{{ name: 'FNS (failed cases)', key: 'FNS', color: '#b5d1e5' }}, {{ name: 'MP (failed cases)', key: 'MP', color: '#86abc9' }}]
        : [
            {{ name: 'SQS', key: 'SQS', color: '#b8d3c0' }},
            {{ name: 'PPL', key: 'PPL', color: '#8fb8a0' }},
            {{ name: 'CRA (lower is better)', key: 'CRA', color: '#d0dfd2' }}
          ];
      const profileValue = (subset, metric) => {{
        const value = meanMetric(subset, metric.key);
        return value === null ? null : value;
      }};
      chartInstance.setOption({{
        animation: false,
        tooltip: {{
          trigger: 'axis', axisPointer: {{ type: 'shadow' }},
          formatter: params => {{
            const index = params[0]?.dataIndex ?? 0;
            const group = groups[index];
            const subset = subsets[index];
            const support = subset.length < 3 ? '<br>Support: Low (n &lt; 3)' : '';
            return `${{escapeHtml(group.label)}}<br>${{isFailure ? 'Failed' : 'Successful'}} cases: ${{subset.length}}${{support}}` + metrics.map(metric => `<br>${{metric.name}}: ${{fmtPct(profileValue(subset, metric))}}`).join('');
          }}
        }},
        legend: {{ top: 0, right: 0, textStyle: {{ color: '#667085', fontSize: 10 }} }},
        grid: {{ left: 12, right: 16, top: 36, bottom: 24, containLabel: true }},
        xAxis: {{ type: 'value', min: 0, max: 1, axisLabel: {{ color: '#667085', fontSize: 10, formatter: value => `${{Math.round(value * 100)}}%` }}, splitLine: {{ lineStyle: {{ color: '#edf0f5' }} }} }},
        yAxis: {{ type: 'category', inverse: true, data: labels, axisLabel: {{ color: '#667085', fontSize: 10 }} }},
        series: metrics.map(metric => ({{
          name: metric.name, type: 'bar', barMaxWidth: 18,
          data: subsets.map(subset => subset.length ? profileValue(subset, metric) : null),
          itemStyle: {{ color: metric.color }}
        }}))
      }});
    }}
    function failureCases() {{
      return allCases().filter(row =>
        row.status === 'ok' && row.metrics && !caseIsSuccess(row) && hasFiniteCaseMetrics(row, ['MP'])
      );
    }}
    function fillFailureSelect(id, values, allLabel) {{
      const target = document.getElementById(id);
      const current = target.value;
      target.innerHTML = `<option value="">${{escapeHtml(allLabel)}}</option>` + values.map(value => `<option value="${{escapeAttr(value)}}">${{escapeHtml(value)}}</option>`).join('');
      if (values.includes(current)) target.value = current;
    }}
    function syncFailureTaskFilter() {{
      const model = document.getElementById('failure-model-filter').value;
      const rows = failureCases().filter(row => !model || row.model === model);
      fillFailureSelect('failure-task-filter', uniq(rows, 'task_name'), 'All tasks');
    }}
    function initFailureProgressRecovery() {{
      const modelFilter = document.getElementById('failure-model-filter');
      const taskFilter = document.getElementById('failure-task-filter');
      fillFailureSelect('failure-model-filter', uniq(failureCases(), 'model'), 'All models');
      syncFailureTaskFilter();
      failureRecoveryChart = registerMetricChart('failure-recovery-chart');
      modelFilter.addEventListener('change', () => {{
        syncFailureTaskFilter();
        renderFailureProgressRecovery();
      }});
      taskFilter.addEventListener('change', renderFailureProgressRecovery);
      renderFailureProgressRecovery();
    }}
    function failureBinIndex(mp, binCount) {{
      const value = Math.max(0, Math.min(1, Number(mp || 0)));
      return Math.min(binCount - 1, Math.floor(value * binCount));
    }}
    function renderFailureProgressRecovery() {{
      const model = document.getElementById('failure-model-filter').value;
      const task = document.getElementById('failure-task-filter').value;
      const rows = failureCases().filter(row => (!model || row.model === model) && (!task || row.task_name === task));
      const chartInstance = failureRecoveryChart;
      if (!chartInstance) return;
      chartInstance.clear();
      if (!rows.length) {{
        setText('failure-count', 0);
        setText('failure-mp', '--');
        setText('failure-drr-count', '0 / 0');
        chartInstance.setOption(emptyChartOption('No failed trajectories for this selection'));
        return;
      }}
      const binCount = rows.length < 10 ? 5 : (rows.length >= 50 ? 20 : 10);
      const binStep = 100 / binCount;
      const labels = Array.from({{ length: binCount }}, (_, index) => `${{index * binStep}}-${{(index + 1) * binStep}}%`);
      const bins = labels.map(() => []);
      rows.forEach(row => bins[failureBinIndex(row.metrics?.MP, binCount)].push(row));
      const counts = bins.map(bin => bin.length);
      const regressionBins = bins.map(bin => bin.filter(row => caseHasRegression(row) && hasFiniteCaseMetrics(row, ['DRR'])));
      const averageDrr = regressionBins.map(bin => meanMetric(bin, 'DRR'));
      const meanFailureMp = meanMetric(rows, 'MP');
      const regressionCount = regressionBins.reduce((sum, bin) => sum + bin.length, 0);
      const centers = labels.map((_, index) => (index + 0.5) / binCount);
      setText('failure-count', rows.length);
      setText('failure-mp', fmtPct(meanFailureMp));
      setText('failure-drr-count', `${{regressionCount}} / ${{rows.length}}`);
      chartInstance.setOption({{
        animation: false,
        color: ['#9bcbed', '#335c9f'],
        tooltip: {{
          trigger: 'item',
          formatter: params => {{
            const index = params.data?.binIndex ?? params.dataIndex ?? 0;
            const share = rows.length ? counts[index] / rows.length : 0;
            const drr = averageDrr[index];
            return `MP interval: ${{labels[index]}}<br>Failed cases: ${{counts[index]}} (${{fmtPct(share)}})<br>DRR cases: ${{regressionBins[index].length}}<br>Average DRR: ${{drr === null ? '--' : fmtPct(drr)}}`;
          }}
        }},
        legend: {{ top: 0, right: 0, data: ['Failed cases', 'DRR (failed regressions)'], textStyle: {{ color: '#667085', fontSize: 11 }} }},
        grid: {{ left: 58, right: 64, top: 38, bottom: 48 }},
        xAxis: {{
          type: 'value', min: 0, max: 1, interval: 0.1, name: 'MP of failed cases',
          nameLocation: 'middle', nameGap: 36,
          nameTextStyle: {{ color: '#667085', fontSize: 11 }},
          axisLabel: {{ color: '#667085', fontSize: 10, formatter: value => `${{Math.round(value * 100)}}%` }},
          splitLine: {{ show: false }}
        }},
        yAxis: [
          {{ type: 'value', name: 'Failed cases', nameLocation: 'middle', nameGap: 40, nameRotate: 90, minInterval: 1, axisLabel: {{ color: '#667085', fontSize: 11 }}, nameTextStyle: {{ color: '#667085', fontSize: 11 }}, splitLine: {{ show: false }} }},
          {{ type: 'value', name: 'DRR', nameLocation: 'middle', nameGap: 42, nameRotate: 90, min: 0, max: 1, axisLabel: {{ color: '#667085', fontSize: 11, formatter: value => `${{Math.round(value * 100)}}%` }}, nameTextStyle: {{ color: '#667085', fontSize: 11 }}, splitLine: {{ show: false }} }}
        ],
        series: [
          {{
            name: 'Failed cases', type: 'bar', data: counts.map((count, index) => ({{ value: [centers[index], count], binIndex: index }})), barWidth: Math.max(16, Math.min(42, 420 / binCount)),
            itemStyle: {{ color: '#9bcbed', borderRadius: [3, 3, 0, 0] }},
            label: {{ show: true, position: 'top', color: '#344054', fontSize: 11, formatter: params => params.value[1] }},
            markLine: {{
              symbol: ['none', 'none'], silent: true,
              label: {{ formatter: `MP (failed cases): ${{fmtPct(meanFailureMp)}}`, color: '#667085', fontSize: 11, position: 'insideEndTop', rotate: 0 }},
              lineStyle: {{ color: '#8b95a7', type: 'dashed', width: 1 }},
              data: [{{ xAxis: meanFailureMp }}]
            }}
          }},
          {{
            name: 'DRR (failed regressions)', type: 'line', yAxisIndex: 1,
            data: averageDrr.map((value, index) => value === null ? null : ({{ value: [centers[index], value], binIndex: index }})),
            connectNulls: false, symbol: 'circle', symbolSize: 7,
            lineStyle: {{ color: '#335c9f', width: 2 }}, itemStyle: {{ color: '#335c9f' }}
          }}
        ]
      }});
    }}
    function emptyChartOption(message) {{
      return {{
        title: {{ text: message, left: 'center', top: 'middle', textStyle: {{ color: '#667085', fontSize: 13, fontWeight: 500 }} }},
        xAxis: {{ show: false }},
        yAxis: {{ show: false }},
        series: []
      }};
    }}
    function fillSelect(id, values) {{
      const el = document.getElementById(id);
      const current = el.value;
      el.innerHTML = values.map(v => `<option value="${{escapeAttr(v)}}">${{escapeHtml(v)}}</option>`).join('');
      if (values.includes(current)) el.value = current;
    }}
    function selectTrajectoryCase(item) {{
      const rows = okCases();
      fillSelect('model-select', uniq(rows, 'model'));
      document.getElementById('model-select').value = item.model || '';
      fillSelect('task-select', uniq(rows.filter(r => r.model === item.model), 'task_name'));
      document.getElementById('task-select').value = item.task_name || '';
      fillSelect('backend-select', uniq(rows.filter(r => r.model === item.model && r.task_name === item.task_name), 'prm_backend'));
      document.getElementById('backend-select').value = item.prm_backend || '';
      const cases = rows.filter(r => r.model === item.model && r.task_name === item.task_name && r.prm_backend === item.prm_backend);
      fillSelect('case-select', cases.map(r => r.case_id));
      document.getElementById('case-select').value = item.case_id || '';
      loadCase(item);
      document.querySelector('.section-heading')?.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    }}
    function rowCanOpenCase(row) {{
      return okCases().some(item => item.case_id === row.case_id && item.model === row.model && item.task_name === row.task && item.prm_backend === row.evaluator);
    }}
    function selectCaseFromRow(row) {{
      const item = okCases().find(candidate =>
        candidate.case_id === row.case_id &&
        candidate.model === row.model &&
        candidate.task_name === row.task &&
        candidate.prm_backend === row.evaluator
      );
      if (item) selectTrajectoryCase(item);
    }}
    function syncSelectors() {{
      const rows = okCases();
      fillSelect('model-select', uniq(rows, 'model'));
      const model = document.getElementById('model-select').value;
      fillSelect('task-select', uniq(rows.filter(r => r.model === model), 'task_name'));
      const task = document.getElementById('task-select').value;
      fillSelect('backend-select', uniq(rows.filter(r => r.model === model && r.task_name === task), 'prm_backend'));
      const backend = document.getElementById('backend-select').value;
      const cases = rows.filter(r => r.model === model && r.task_name === task && r.prm_backend === backend);
      fillSelect('case-select', cases.map(r => r.case_id));
      const caseId = document.getElementById('case-select').value;
      state.case = cases.find(r => r.case_id === caseId) || cases[0] || rows[0] || null;
      if (state.case) loadCase(state.case);
    }}
    function loadCase(item) {{
      state.case = item;
      video.src = item.video || '';
      video.pause();
      playButton.textContent = 'Play';
      timeline.value = 0;
      updateUI(0);
    }}
    function duration() {{
      if (Number.isFinite(video.duration) && video.duration > 0) return video.duration;
      const n = (state.case?.progress_processed || []).length;
      return Math.max(1, n - 1);
    }}
    function timeToIndex(t) {{
      const n = (state.case?.progress_processed || []).length;
      if (n <= 1) return 0;
      return Math.max(0, Math.min(n - 1, Math.round((t / duration()) * (n - 1))));
    }}
    function pointAt(t) {{
      const p = state.case?.progress_processed || [];
      const idx = timeToIndex(t);
      return {{ idx, progress: p[idx] || 0 }};
    }}
    function slopeAt(idx) {{
      const p = state.case?.progress_processed || [];
      if (idx <= 0 || !p.length) return 0;
      return (p[idx] || 0) - (p[idx - 1] || 0);
    }}
    function bestAt(idx) {{
      const p = state.case?.progress_processed || [];
      return Math.max(0, ...p.slice(0, idx + 1));
    }}
    function framesWithTime() {{
      const p = state.case?.progress_processed || [];
      const raw = state.case?.progress_raw || [];
      const d = duration() || 1;
      return p.map((progress, idx) => ({{
        t: p.length <= 1 ? 0 : (idx / (p.length - 1)) * d,
        progress,
        raw: raw[idx] ?? progress,
        idx
      }}));
    }}
    function interpolateAt(time) {{
      const frames = framesWithTime();
      if (!frames.length) return null;
      if (time <= frames[0].t) return frames[0];
      if (time >= frames[frames.length - 1].t) return frames[frames.length - 1];
      for (let i = 0; i < frames.length - 1; i++) {{
        const a = frames[i], b = frames[i + 1];
        if (a.t <= time && time <= b.t) {{
          const ratio = (time - a.t) / Math.max(0.00001, b.t - a.t);
          return {{
            t: time,
            progress: a.progress + ratio * (b.progress - a.progress),
            raw: a.raw + ratio * (b.raw - a.raw),
            idx: Math.round(a.idx + ratio * (b.idx - a.idx))
          }};
        }}
      }}
      return frames[frames.length - 1];
    }}
    function bestSoFar(time) {{
      const frames = framesWithTime();
      let best = 0;
      frames.forEach(frame => {{ if (frame.t <= time && frame.progress > best) best = frame.progress; }});
      const here = interpolateAt(time);
      if (here && here.progress > best) best = here.progress;
      return best;
    }}
    function slopeAtTime(time) {{
      const d = duration();
      const prev = interpolateAt(Math.max(0, time - Math.max(0.15, d * 0.01)));
      const here = interpolateAt(time);
      if (!prev || !here) return 0;
      return here.progress - prev.progress;
    }}
    function updateUI(t) {{
      const here = interpolateAt(t) || pointAt(t);
      const delta = slopeAtTime(t);
      document.getElementById('time-label').textContent = `${{fmtTime(t)}} / ${{fmtTime(duration())}}`;
      timeline.value = duration() > 0 ? Math.round((t / duration()) * 1000) : 0;
      setText('live-progress', fmtPct(here.progress));
      setText('live-best', fmtPct(bestSoFar(t)));
      setText('live-slope', (delta >= 0 ? '+' : '') + fmtPct(delta));
      setText('live-frame', String(here.idx + 1));
      const m = state.case?.metrics || {{}};
      setText('metric-m25', fmtPct(m.M25));
      setText('metric-m50', fmtPct(m.M50));
      setText('metric-m75', fmtPct(m.M75));
      setText('metric-sr', fmtPct(m.SR));
      setText('metric-mp', fmtPct(m.MP));
      setText('metric-ppl', fmtPct(m.PPL));
      setText('metric-cra', fmtPct(m.CRA));
      setText('metric-str', fmtPct(m.STR));
      setText('metric-drr', caseHasRegression(state.case) ? fmtPct(m.DRR) : '--');
      setText('metric-fns', fmtPct(m.FNS));
      setText('metric-sqs', fmtPct(m.SQS));
      document.querySelectorAll('.milestone').forEach(el => {{
        el.classList.toggle('reached', here.progress + 1e-6 >= Number(el.dataset.ms) / 100);
      }});
      setStateChip(delta);
      updateChart(t);
    }}
    function setStateChip(delta) {{
      stateChip.classList.remove('state-progress', 'state-regress', 'state-stall');
      if (delta > 0.006) {{
        stateChip.classList.add('state-progress');
        stateChip.textContent = 'Progress';
      }} else if (delta < -0.006) {{
        stateChip.classList.add('state-regress');
        stateChip.textContent = 'Regress';
      }} else {{
        stateChip.classList.add('state-stall');
        stateChip.textContent = 'Stall';
      }}
    }}
    function splitProgressSegments(frames) {{
      const progressData = [];
      const stallData = [];
      const threshold = 0.006;
      function appendSegment(target, a, b) {{
        if (target.length) target.push([null, null]);
        target.push([a.t, a.progress], [b.t, b.progress]);
      }}
      for (let i = 0; i < frames.length - 1; i++) {{
        const a = frames[i], b = frames[i + 1];
        if ((b.progress - a.progress) > threshold) appendSegment(progressData, a, b);
        else appendSegment(stallData, a, b);
      }}
      return {{ progress: progressData, stall: stallData }};
    }}
    function framesUpToTime(frames, time, here) {{
      const elapsed = frames.filter(frame => frame.t <= time);
      if (here && (!elapsed.length || elapsed[elapsed.length - 1].t < time)) elapsed.push(here);
      return elapsed;
    }}
    function framesFromTime(frames, time, here) {{
      const remaining = [];
      if (here) remaining.push(here);
      frames.forEach(frame => {{ if (frame.t > time) remaining.push(frame); }});
      return remaining;
    }}
    function updateChart(time) {{
      if (!chart || !(state.case?.progress_processed || []).length) return;
      const frames = framesWithTime();
      const here = interpolateAt(time);
      const rawPath = frames.map(frame => [frame.t, frame.raw]);
      const elapsedFrames = framesUpToTime(frames, time, here);
      const remainingFrames = framesFromTime(frames, time, here);
      const segments = splitProgressSegments(elapsedFrames);
      const remainingPath = remainingFrames.map(frame => [frame.t, frame.progress]);
      chart.setOption({{
        xAxis: {{ max: Math.max(duration(), 1) }},
        series: [
          {{ id: 'raw-path', data: rawPath }},
          {{ id: 'remaining-path', data: remainingPath }},
          {{ id: 'progress-path', data: segments.progress }},
          {{ id: 'stall-path', data: segments.stall }},
          {{ id: 'cursor-line', data: [[time, 0], [time, 1]] }},
          {{ id: 'cursor-point', data: here ? [[time, here.progress]] : [] }}
        ]
      }});
    }}
    function initChart() {{
      if (!window.echarts || !chartDom) return;
      if (chart) chart.dispose();
      chart = echarts.init(chartDom);
      chart.setOption({{
        animation: false,
        backgroundColor: 'transparent',
        tooltip: {{
          trigger: 'axis',
          valueFormatter: value => typeof value === 'number' ? value.toFixed(3) : value
        }},
        grid: {{ left: 36, right: 10, top: 32, bottom: 28 }},
        xAxis: {{
          type: 'value',
          name: 'time (s)',
          min: 0,
          nameTextStyle: {{ color: '#667084', fontSize: 10 }},
          axisLabel: {{ color: '#667084', fontSize: 10 }},
          splitLine: {{ show: false }}
        }},
        yAxis: {{
          type: 'value',
          name: 'progress',
          min: 0,
          max: 1,
          nameTextStyle: {{ color: '#667084', fontSize: 10 }},
          axisLabel: {{ color: '#667084', fontSize: 10 }},
          splitLine: {{ lineStyle: {{ color: '#e8edf4', type: 'dashed' }} }}
        }},
        series: [
          {{ id: 'raw-path', name: 'Raw', type: 'line', data: [], smooth: true, symbol: 'none', lineStyle: {{ color: 'rgba(77, 139, 202, 0.58)', width: 1.25, type: 'dotted' }}, z: 0, tooltip: {{ show: false }} }},
          {{ id: 'remaining-path', name: 'Remaining', type: 'line', data: [], smooth: true, symbol: 'none', lineStyle: {{ color: 'rgba(105, 116, 135, 0.48)', width: 1.6, type: 'dashed' }}, z: 1 }},
          {{ id: 'progress-path', name: 'Progress', type: 'line', data: [], smooth: true, symbol: 'none', connectNulls: false, lineStyle: {{ color: '#2e9d49', width: 2.7 }}, areaStyle: {{ color: 'rgba(46, 157, 73, 0.18)' }} }},
          {{ id: 'stall-path', name: 'Stagnating', type: 'line', data: [], smooth: true, symbol: 'none', connectNulls: false, lineStyle: {{ color: '#d0443e', width: 2.7 }}, areaStyle: {{ color: 'rgba(208, 68, 62, 0.16)' }} }},
          {{ id: 'cursor-line', type: 'line', data: [], symbol: 'none', lineStyle: {{ color: 'rgba(232, 100, 27, 0.88)', width: 2.1 }}, tooltip: {{ show: false }}, z: 8 }},
          {{ id: 'cursor-point', type: 'scatter', data: [], symbolSize: 10, itemStyle: {{ color: '#E8641B', borderColor: '#fff', borderWidth: 2 }} }}
        ]
      }});
    }}
    function seekTo(t) {{
      const next = Math.max(0, Math.min(duration(), t));
      video.currentTime = next;
      updateUI(next);
    }}
    function seekFromClientX(clientX) {{
      const rect = chartDom.getBoundingClientRect();
      const p = state.case?.progress_processed || [];
      if (!p.length) return;
      const ratio = Math.max(0, Math.min(1, (clientX - rect.left - 36) / Math.max(1, rect.width - 46)));
      seekTo(ratio * duration());
    }}
    let resumeAfterScrub = false;
    chartDom.addEventListener('pointerdown', e => {{
      resumeAfterScrub = !video.paused;
      state.dragging = true;
      chartDom.setPointerCapture(e.pointerId);
      seekFromClientX(e.clientX);
    }});
    chartDom.addEventListener('pointermove', e => {{ if (state.dragging) seekFromClientX(e.clientX); }});
    chartDom.addEventListener('pointerup', () => {{
      state.dragging = false;
      if (resumeAfterScrub) video.play().catch(() => {{}});
      resumeAfterScrub = false;
    }});
    playButton.addEventListener('click', () => {{
      if (video.paused) video.play().catch(() => {{}});
      else video.pause();
    }});
    timeline.addEventListener('input', () => seekTo((Number(timeline.value) / 1000) * duration()));
    speedSelect.addEventListener('change', () => {{ video.playbackRate = Number(speedSelect.value) || 1; }});
    video.addEventListener('play', () => {{ playButton.textContent = 'Pause'; }});
    video.addEventListener('pause', () => {{ playButton.textContent = 'Play'; }});
    video.addEventListener('timeupdate', () => {{ if (!state.dragging) updateUI(video.currentTime || 0); }});
    video.addEventListener('loadedmetadata', () => updateUI(video.currentTime || 0));
    window.addEventListener('resize', () => {{
      if (chart) chart.resize();
      metricCharts.forEach(instance => instance.resize());
      updateUI(video.currentTime || 0);
    }});
    ['model-select','task-select','case-select','backend-select'].forEach(id => {{
      document.getElementById(id).addEventListener('change', syncSelectors);
    }});

    function makeRows(type) {{
      if (type === 'leaderboard') return REPORT_DATA.leaderboard || [];
      if (type === 'groups') return REPORT_DATA.groups || [];
      return (REPORT_DATA.cases || []).map(c => Object.assign({{
        benchmark: c.benchmark, model: c.model, task: c.task_name, case_id: c.case_id, evaluator: c.prm_backend,
        label: c.label, status: c.status
      }}, c.metrics || {{}}));
    }}
    function renderTable(tableId, filterId, columns, rows, options = {{}}) {{
      const table = document.getElementById(tableId), filter = document.getElementById(filterId);
      const pagination = document.getElementById(tableId.replace('-table', '-pagination'));
      const key = tableId;
      const pageSize = 25;
      function draw() {{
        const q = (filter.value || '').toLowerCase();
        let out = rows.filter(r => !q || JSON.stringify(r).toLowerCase().includes(q));
        const s = state.sort[key];
        if (s) out = out.slice().sort((a,b) => compare(a[s.col], b[s.col]) * s.dir);
        if (!out.length) {{
          table.innerHTML = `<tbody><tr><td class="empty">No rows</td></tr></tbody>`;
          if (pagination) pagination.innerHTML = '<span>0 rows</span>';
          return;
        }}
        const pageCount = Math.ceil(out.length / pageSize);
        state.page[key] = Math.max(1, Math.min(state.page[key] || 1, pageCount));
        const page = state.page[key];
        const start = (page - 1) * pageSize;
        const pageRows = out.slice(start, start + pageSize);
        table.innerHTML = `<colgroup>${{columns.map(c => `<col style="width:${{columnWidth(c)}}">`).join('')}}</colgroup>` +
          `<thead><tr>${{columns.map(c => `<th class="${{cellClass(c, null)}}" data-col="${{c}}" title="${{escapeAttr(c)}}">${{escapeHtml(c)}}${{s?.col===c ? (s.dir>0?' ^':' v') : ''}}</th>`).join('')}}</tr></thead>` +
          `<tbody>${{pageRows.map((r, idx) => {{
            const canOpen = options.onRowClick && (!options.canOpen || options.canOpen(r));
            return `<tr class="${{canOpen ? 'case-clickable' : ''}}" data-row-index="${{idx}}">${{columns.map(c => `<td class="${{cellClass(c, r[c])}}" title="${{cellTitle(c, r[c])}}">${{fmtCell(c, r[c])}}</td>`).join('')}}</tr>`;
          }}).join('')}}</tbody>`;
        if (pagination) {{
          const end = Math.min(start + pageRows.length, out.length);
          pagination.innerHTML = `<span>Rows ${{start + 1}}-${{end}} of ${{out.length}}</span>` +
            (pageCount > 1 ? `<div class="pagination-actions"><button type="button" class="page-button" data-page="prev" ${{page === 1 ? 'disabled' : ''}}>Previous</button><span>Page ${{page}} / ${{pageCount}}</span><button type="button" class="page-button" data-page="next" ${{page === pageCount ? 'disabled' : ''}}>Next</button></div>` : '');
          pagination.querySelector('[data-page="prev"]')?.addEventListener('click', () => {{ state.page[key] = page - 1; draw(); }});
          pagination.querySelector('[data-page="next"]')?.addEventListener('click', () => {{ state.page[key] = page + 1; draw(); }});
        }}
        table.querySelectorAll('th').forEach(th => th.addEventListener('click', () => {{
          const col = th.dataset.col;
          const prev = state.sort[key];
          state.sort[key] = {{ col, dir: prev?.col === col ? -prev.dir : 1 }};
          state.page[key] = 1;
          draw();
        }}));
        if (options.onRowClick) {{
          table.querySelectorAll('tbody tr.case-clickable').forEach(tr => tr.addEventListener('click', () => {{
            const row = pageRows[Number(tr.dataset.rowIndex)];
            if (row) options.onRowClick(row);
          }}));
        }}
      }}
      filter.addEventListener('input', () => {{ state.page[key] = 1; draw(); }});
      draw();
    }}
    function initTables() {{
      renderTable('leaderboard-table', 'leaderboard-filter', ['benchmark','model', ...metricColumns], makeRows('leaderboard'));
      renderTable('group-table', 'group-filter', ['benchmark','model','task', ...metricColumns], makeRows('groups'));
    }}
    function fmtTime(s) {{ s = Number.isFinite(s) ? s : 0; const m = Math.floor(s / 60); return `${{m}}:${{String(Math.floor(s % 60)).padStart(2,'0')}}`; }}
    function fmtDate(value) {{
      if (!value) return '-';
      const date = new Date(value);
      return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
    }}
    function fmtPct(v) {{ return Number.isFinite(Number(v)) ? `${{(Number(v) * 100).toFixed(1)}}%` : '--'; }}
    function fmtNum(v) {{ return Number.isFinite(Number(v)) ? Number(v).toFixed(3) : '--'; }}
    function fmtCell(k, v) {{
      if (v === undefined || v === null || v === '') return k === 'label' ? '-' : '';
      return pctColumns.has(k) ? fmtPct(v) : escapeHtml(String(v));
    }}
    function cellClass(k, v) {{
      const classes = [];
      if (typeof v === 'number' || isNumericColumn(k)) classes.push('num');
      if (k === 'benchmark') classes.push('col-benchmark');
      if (k === 'model') classes.push('col-model');
      if (k === 'task') classes.push('col-task');
      if (k === 'evaluator') classes.push('col-evaluator');
      if (k === 'case_id') classes.push('col-case-id');
      if (k === 'status') classes.push('col-status');
      if (k === 'label') classes.push('col-label');
      if (k === 'SR' || k === 'FNS') classes.push('metric-boundary');
      return classes.join(' ');
    }}
    function cellTitle(k, v) {{
      if (v === undefined || v === null || v === '') return '';
      return escapeAttr(pctColumns.has(k) ? fmtPct(v) : String(v));
    }}
    function isNumericColumn(k) {{ return pctColumns.has(k) || ['num_tasks','num_cases'].includes(k); }}
    function columnWidth(k) {{
      if (k === 'benchmark') return '11%';
      if (k === 'model') return '11%';
      if (k === 'task') return '16%';
      if (k === 'evaluator') return '9%';
      if (k === 'case_id') return '10%';
      return '6%';
    }}
    function compare(a,b) {{ const na = Number(a), nb = Number(b); if (Number.isFinite(na) && Number.isFinite(nb)) return na - nb; return String(a ?? '').localeCompare(String(b ?? '')); }}
    function setText(id, value) {{ document.getElementById(id).textContent = value; }}
    function escapeHtml(s) {{ return String(s).replace(/[&<>"']/g, m => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[m])); }}
    function escapeAttr(s) {{ return escapeHtml(s); }}
    renderRunSummary();
    renderIssueSummary();
    initMetricCharts();
    initChart();
    syncSelectors();
    initTables();
  </script>
</body>
</html>
"""


def write_visualization_report(
    records: list[dict[str, Any]],
    output_dir: Path,
    artifacts: dict[str, str],
    plot_status: str,
    success_threshold: float,
) -> Path:
    ok_count = sum(1 for record in records if record.get("status") == "ok")
    lines = [
        "# Visualization Report",
        "",
        f"- Total cases: {len(records)}",
        f"- Successful cases: {ok_count}",
        f"- Success threshold line: {success_threshold:g}",
        f"- Plot status: {plot_status}",
        "",
        "## Artifacts",
        "",
    ]
    for name, path in artifacts.items():
        lines.append(f"- `{name}`: `{path}`")
    path = output_dir / "visualization_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _annotate_curve_points(axis: Any, processed: list[float], metrics: dict[str, Any]) -> None:
    if not processed:
        return
    max_idx = max(range(len(processed)), key=processed.__getitem__)
    max_value = processed[max_idx]
    final_idx = len(processed) - 1
    final_value = processed[-1]
    axis.scatter([max_idx], [max_value], color="#dc2626", s=36, zorder=5, label="MP")
    axis.annotate(
        f"MP={max_value:.3f}",
        xy=(max_idx, max_value),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=8.5,
        color="#991b1b",
    )
    axis.scatter([final_idx], [final_value], color="#0f172a", s=28, zorder=5, label="final")
    axis.annotate(
        f"final={final_value:.3f}",
        xy=(final_idx, final_value),
        xytext=(8, -16),
        textcoords="offset points",
        fontsize=8.5,
        color="#0f172a",
    )
    start, end, drawdown = _max_drawdown_span(processed)
    if drawdown > 1e-9:
        axis.axvspan(start, end, color="#f97316", alpha=0.13, label="max drawdown")
        axis.plot([start, end], [processed[start], processed[end]], color="#f97316", linewidth=1.2)


def _case_title(record: dict[str, Any], prm_backend: str = "") -> str:
    case_id = record.get("case_id") or "case"
    task_name = record.get("task_name") or record.get("task") or "task"
    backend = prm_backend or "unknown_backend"
    return f"{case_id} | backend={backend} | task={task_name}"


def _legend_location(values: list[float]) -> str:
    if not values:
        return "upper right"
    tail = values[max(0, len(values) - max(3, len(values) // 4)) :]
    tail_mean = sum(tail) / len(tail)
    return "lower right" if tail_mean >= 0.5 else "upper right"


def _max_drawdown_span(values: list[float]) -> tuple[int, int, float]:
    if not values:
        return 0, 0, 0.0
    peak_idx = 0
    best_start = 0
    best_end = 0
    best_drawdown = 0.0
    for idx, value in enumerate(values):
        if value > values[peak_idx]:
            peak_idx = idx
        drawdown = values[peak_idx] - value
        if drawdown > best_drawdown:
            best_drawdown = drawdown
            best_start = peak_idx
            best_end = idx
    return best_start, best_end, best_drawdown


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned[:96] or "case"


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _float_list(values: list[Any]) -> list[float]:
    output: list[float] = []
    for value in values:
        try:
            output.append(float(value))
        except (TypeError, ValueError):
            continue
    return output
