"""Shared adapter contracts and packing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from gril.models import CanonicalDataset, ImuBatch, LidarBatch


@dataclass(frozen=True)
class AdapterConfig:
    inputs: tuple[Path, ...]
    lidar_topic: str
    imu_topic: str
    scan_lines: int = 16
    start_ns: int | None = None
    end_ns: int | None = None

    def includes(self, timestamp_ns: int) -> bool:
        if self.start_ns is not None and timestamp_ns < self.start_ns:
            return False
        return self.end_ns is None or timestamp_ns < self.end_ns

    def validate(self) -> None:
        if not self.inputs:
            raise ValueError("At least one input path is required")
        missing = [str(path) for path in self.inputs if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Input files not found: {', '.join(missing)}")
        if self.scan_lines <= 0:
            raise ValueError("scan_lines must be positive")
        if (
            self.start_ns is not None
            and self.end_ns is not None
            and self.start_ns >= self.end_ns
        ):
            raise ValueError("start time must be earlier than end time")


@dataclass(frozen=True)
class Scan:
    timestamp_ns: int
    xyz: np.ndarray
    intensity: np.ndarray
    ring: np.ndarray
    point_time_s: np.ndarray


class InputAdapter(Protocol):
    def read(self) -> CanonicalDataset: ...


def pack_lidar(scans: list[Scan], frame_id: str) -> LidarBatch:
    offsets = np.zeros(len(scans) + 1, dtype=np.int64)
    for index, scan in enumerate(scans):
        offsets[index + 1] = offsets[index] + len(scan.xyz)
    if scans:
        xyz = np.concatenate([scan.xyz for scan in scans])
        intensity = np.concatenate([scan.intensity for scan in scans])
        ring = np.concatenate([scan.ring for scan in scans])
        point_time_s = np.concatenate([scan.point_time_s for scan in scans])
    else:
        xyz = np.empty((0, 3), dtype=np.float32)
        intensity = np.empty(0, dtype=np.float32)
        ring = np.empty(0, dtype=np.uint16)
        point_time_s = np.empty(0, dtype=np.float32)
    return LidarBatch(
        frame_id=frame_id,
        scan_timestamps_ns=np.asarray(
            [scan.timestamp_ns for scan in scans], dtype=np.int64
        ),
        scan_offsets=offsets,
        xyz=xyz,
        intensity=intensity,
        ring=ring,
        point_time_s=point_time_s,
    )


def pack_imu(
    samples: list[tuple[int, np.ndarray, np.ndarray]], frame_id: str
) -> ImuBatch:
    return ImuBatch(
        frame_id=frame_id,
        timestamps_ns=np.asarray([sample[0] for sample in samples], dtype=np.int64),
        angular_velocity=np.asarray(
            [sample[1] for sample in samples], dtype=np.float64
        ).reshape((-1, 3)),
        linear_acceleration=np.asarray(
            [sample[2] for sample in samples], dtype=np.float64
        ).reshape((-1, 3)),
    )
