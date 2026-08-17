#!/usr/bin/env python3
"""使用已有 ``per_case.jsonl`` 离线重算 PRM-as-a-Judge v1.5 指标报表。

脚本只读取保存的轨迹进度，不调用 PRM 或 GPU。默认将 ``metrics.xlsx``
写回运行目录；指定 ``--output-dir`` 时只向独立输出目录写入该工作簿，
不修改输入目录。

用法示例：
    python eval/calculate_advanced_metrics.py /path/to/run
    python eval/calculate_advanced_metrics.py /path/to/run --output-dir /path/to/reports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prm_judge.excel_report import write_metrics_workbook
from prm_judge.metrics import MetricConfig, aggregate_metrics, recompute_record_metrics


def load_records(run_root: Path) -> list[dict]:
    """从运行目录的主文件或分片文件中加载全部逐轨迹记录。"""
    records: list[dict] = []
    canonical_path = run_root / "per_case.jsonl"
    jsonl_paths = (
        [canonical_path]
        if canonical_path.exists()
        else sorted((run_root / "shards").glob("shard_*/per_case.jsonl"))
    )
    if not jsonl_paths:
        raise FileNotFoundError(f"No per_case.jsonl or shards/shard_*/per_case.jsonl found under {run_root}")
    for path in jsonl_paths:
        with path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                if line.strip():
                    records.append(json.loads(line))
    return records


def main() -> None:
    """解析命令行参数并生成单一指标工作簿。"""
    parser = argparse.ArgumentParser(description="Compute v1.5 advanced metrics from per_case.jsonl.")
    parser.add_argument("run_root", type=Path, help="Run directory produced by eval/run_judge.py")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="报表输出目录；默认写入 run_root，指定后不会写入输入目录",
    )
    parser.add_argument("--success-threshold", type=float, default=0.99)
    parser.add_argument("--success-source", choices=["progress", "label"], default="progress")
    args = parser.parse_args()

    records = load_records(args.run_root)
    metric_config = MetricConfig(
        success_threshold=args.success_threshold,
        success_source=args.success_source,
    )
    recompute_record_metrics(records, metric_config)
    aggregate = aggregate_metrics(records, metric_config)
    output_dir = args.output_dir or args.run_root
    write_metrics_workbook(output_dir / "metrics.xlsx", records, aggregate, metric_config)
    print(f"Saved {output_dir / 'metrics.xlsx'}")


if __name__ == "__main__":
    main()
