from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..schema import EvalCase, ProgressTrace


class BasePRMAdapter(ABC):
    """Interface for PRM backends."""

    @abstractmethod
    def predict(self, case: EvalCase, output_dir: Path) -> ProgressTrace:
        """Run the backend and return a standardized progress trace."""
