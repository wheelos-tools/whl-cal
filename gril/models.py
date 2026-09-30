"""Versioned, ROS-independent GRIL input models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


def _array(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value, dtype=dtype))


@dataclass(frozen=True)
class LidarBatch:
    """Packed ragged LiDAR scans.

    ``scan_offsets`` indexes the flattened per-point arrays. Point time is
    relative to the scan timestamp and expressed in seconds.
    """

    frame_id: str
    scan_timestamps_ns: np.ndarray
    scan_offsets: np.ndarray
    xyz: np.ndarray
    intensity: np.ndarray
    ring: np.ndarray
    point_time_s: np.ndarray

    def normalized(self) -> "LidarBatch":
        return LidarBatch(
            frame_id=str(self.frame_id),
            scan_timestamps_ns=_array(self.scan_timestamps_ns, np.int64),
            scan_offsets=_array(self.scan_offsets, np.int64),
            xyz=_array(self.xyz, np.float32),
            intensity=_array(self.intensity, np.float32),
            ring=_array(self.ring, np.uint16),
            point_time_s=_array(self.point_time_s, np.float32),
        )

    def validate(self) -> None:
        lidar = self.normalized()
        scan_count = len(lidar.scan_timestamps_ns)
        point_count = len(lidar.xyz)
        if not lidar.frame_id:
            raise ValueError("LiDAR frame_id is required")
        if lidar.xyz.ndim != 2 or lidar.xyz.shape[1] != 3:
            raise ValueError("LiDAR xyz must have shape (N, 3)")
        if lidar.scan_offsets.shape != (scan_count + 1,):
            raise ValueError("scan_offsets must have one more item than scans")
        if lidar.scan_offsets[0] != 0 or lidar.scan_offsets[-1] != point_count:
            raise ValueError("scan_offsets do not span the flattened points")
        if np.any(np.diff(lidar.scan_offsets) < 0):
            raise ValueError("scan_offsets must be monotonic")
        if np.any(np.diff(lidar.scan_timestamps_ns) < 0):
            raise ValueError("LiDAR scan timestamps must be monotonic")
        for name, values in (
            ("intensity", lidar.intensity),
            ("ring", lidar.ring),
            ("point_time_s", lidar.point_time_s),
        ):
            if values.shape != (point_count,):
                raise ValueError(f"LiDAR {name} must have shape ({point_count},)")
        if not np.isfinite(lidar.xyz).all():
            raise ValueError("LiDAR xyz contains non-finite values")
        if not np.isfinite(lidar.intensity).all():
            raise ValueError("LiDAR intensity contains non-finite values")
        if not np.isfinite(lidar.point_time_s).all():
            raise ValueError("LiDAR point time contains non-finite values")
        if np.any(lidar.point_time_s < 0.0):
            raise ValueError("LiDAR point time must be relative and non-negative")
        for start, end in zip(lidar.scan_offsets[:-1], lidar.scan_offsets[1:]):
            if np.any(np.diff(lidar.point_time_s[start:end]) < 0.0):
                raise ValueError("LiDAR point time must be monotonic within each scan")


@dataclass(frozen=True)
class ImuBatch:
    frame_id: str
    timestamps_ns: np.ndarray
    angular_velocity: np.ndarray
    linear_acceleration: np.ndarray

    def normalized(self) -> "ImuBatch":
        return ImuBatch(
            frame_id=str(self.frame_id),
            timestamps_ns=_array(self.timestamps_ns, np.int64),
            angular_velocity=_array(self.angular_velocity, np.float64),
            linear_acceleration=_array(self.linear_acceleration, np.float64),
        )

    def validate(self) -> None:
        imu = self.normalized()
        sample_count = len(imu.timestamps_ns)
        if not imu.frame_id:
            raise ValueError("IMU frame_id is required")
        if np.any(np.diff(imu.timestamps_ns) < 0):
            raise ValueError("IMU timestamps must be monotonic")
        for name, values in (
            ("angular_velocity", imu.angular_velocity),
            ("linear_acceleration", imu.linear_acceleration),
        ):
            if values.shape != (sample_count, 3):
                raise ValueError(f"IMU {name} must have shape ({sample_count}, 3)")
            if not np.isfinite(values).all():
                raise ValueError(f"IMU {name} contains non-finite values")


@dataclass(frozen=True)
class StaticTransform:
    parent_frame: str
    child_frame: str
    matrix: np.ndarray
    source_topic: str = "/tf_static"

    def normalized(self) -> "StaticTransform":
        return StaticTransform(
            parent_frame=str(self.parent_frame),
            child_frame=str(self.child_frame),
            matrix=_array(self.matrix, np.float64),
            source_topic=str(self.source_topic),
        )

    def validate(self) -> None:
        transform = self.normalized()
        if not transform.parent_frame or not transform.child_frame:
            raise ValueError("Transform parent and child frames are required")
        if transform.matrix.shape != (4, 4):
            raise ValueError("Transform matrix must have shape (4, 4)")
        if not np.isfinite(transform.matrix).all():
            raise ValueError("Transform matrix contains non-finite values")
        if not np.allclose(transform.matrix[3], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError("Transform matrix has an invalid homogeneous row")


@dataclass(frozen=True)
class CanonicalDataset:
    source_type: str
    source_files: tuple[str, ...]
    lidar_topic: str
    imu_topic: str
    lidar: LidarBatch
    imu: ImuBatch
    transforms: tuple[StaticTransform, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported GRIL dataset schema version: {self.schema_version}"
            )
        if self.source_type not in {"apollo_record", "ros1_bag", "synthetic"}:
            raise ValueError(f"Unsupported source type: {self.source_type}")
        if not self.source_files:
            raise ValueError("At least one source file is required")
        if not self.lidar_topic or not self.imu_topic:
            raise ValueError("LiDAR and IMU topics are required")
        self.lidar.validate()
        self.imu.validate()
        if len(self.lidar.scan_timestamps_ns) == 0:
            raise ValueError("Canonical dataset contains no LiDAR scans")
        if len(self.imu.timestamps_ns) == 0:
            raise ValueError("Canonical dataset contains no IMU samples")
        for transform in self.transforms:
            transform.validate()
