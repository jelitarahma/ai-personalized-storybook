"""
System monitor utility to measure latency and memory consumption
(RAM and CUDA VRAM) across the storybook generation pipeline.
"""

import time
import os
import psutil
import torch
from contextlib import contextmanager
from typing import Dict, Any


def get_memory_usage() -> Dict[str, float]:
    """
    Get current RAM and VRAM usage in Megabytes (MB).
    """
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    vram_mb = 0.0
    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    return {"ram_mb": round(ram_mb, 2), "vram_mb": round(vram_mb, 2)}


@contextmanager
def measure_stage(stage_name: str):
    """
    Context manager to profile execution time and memory change of a pipeline stage.

    Yields a dictionary that is populated with timing and memory metrics upon exit.
    """
    metrics: Dict[str, Any] = {"stage": stage_name}
    start_mem = get_memory_usage()
    start_time = time.perf_counter()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        yield metrics
    finally:
        end_time = time.perf_counter()
        end_mem = get_memory_usage()
        elapsed = end_time - start_time
        metrics["duration_seconds"] = round(elapsed, 4)
        metrics["start_ram_mb"] = start_mem["ram_mb"]
        metrics["end_ram_mb"] = end_mem["ram_mb"]
        metrics["ram_delta_mb"] = round(end_mem["ram_mb"] - start_mem["ram_mb"], 2)
        if torch.cuda.is_available():
            metrics["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        else:
            metrics["peak_vram_mb"] = 0.0

