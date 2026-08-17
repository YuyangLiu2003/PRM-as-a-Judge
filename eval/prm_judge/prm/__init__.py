from typing import Any

from .base import BasePRMAdapter

__all__ = ["BasePRMAdapter", "DopamineAdapter", "RecordedProgressAdapter", "RobometerHttpAdapter"]


def __getattr__(name: str) -> Any:
    """按需导入后端，避免基础 CLI 启动时强制加载可选依赖。"""
    if name == "DopamineAdapter":
        from .dopamine import DopamineAdapter

        return DopamineAdapter
    if name == "RecordedProgressAdapter":
        from .recorded import RecordedProgressAdapter

        return RecordedProgressAdapter
    if name == "RobometerHttpAdapter":
        from .robometer import RobometerHttpAdapter

        return RobometerHttpAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
