"""Apollo record shard discovery for GRIL ingress adapters."""

from __future__ import annotations

from gril.adapters.base import AdapterConfig
from lidar2lidar.record_utils import discover_record_files


def record_paths(config: AdapterConfig) -> tuple[str, ...]:
    paths = tuple(
        record_path
        for input_path in config.inputs
        for record_path in discover_record_files(str(input_path))
    )
    if not paths:
        raise ValueError("No Apollo record files found in the configured inputs")
    return paths
