from __future__ import annotations

from pathlib import Path

from ..schema import EvalCase, ProgressTrace
from .base import BasePRMAdapter


class RecordedProgressAdapter(BasePRMAdapter):
    """Adapter for manifests that already contain progress traces."""

    def predict(self, case: EvalCase, output_dir: Path) -> ProgressTrace:
        if case.progress is None:
            raise ValueError(
                f"Case {case.case_id} has no progress list. Use --prm dopamine or add progress/progress_path."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        return ProgressTrace(
            case_id=case.case_id,
            progress=list(case.progress),
            raw_output_path=str(case.progress_path) if case.progress_path else None,
            metadata={"adapter": "recorded"},
        )
