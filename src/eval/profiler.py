"""
System Performance and Resource Profiler.
"""

from src.utils.system_monitor import measure_stage, get_memory_usage
from typing import Dict, Any


class SystemMonitor:
    """
    Lightweight resource profiler for pipeline stages.
    """

    def __init__(self):
        self.stages: Dict[str, Any] = {}

    def profile_stage(self, stage_name: str):
        return measure_stage(stage_name)

    def current_memory(self) -> Dict[str, float]:
        return get_memory_usage()


__all__ = [
    "SystemMonitor",
    "measure_stage",
    "get_memory_usage"
]
