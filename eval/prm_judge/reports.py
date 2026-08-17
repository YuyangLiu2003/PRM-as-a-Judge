from __future__ import annotations

from pathlib import Path
from typing import Any

def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PRM-as-a-Judge Evaluation Report",
        "",
        f"- Total cases: {summary.get('total_cases', 0)}",
        f"- Processed OK: {summary.get('processed_ok', 0)}",
        f"- Errors: {summary.get('processed_error', 0)}",
        "",
        "See `metrics.xlsx` and `per_case.jsonl` for machine-readable outputs.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
