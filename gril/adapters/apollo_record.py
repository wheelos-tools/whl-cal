"""Apollo record adapter for ROS-independent GRIL input."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from gril.adapters.base import AdapterConfig, Scan, pack_imu, pack_lidar
from gril.models import CanonicalDataset, StaticTransform
from lidar2lidar.record_adapter import Record
from lidar2lidar.record_utils import (
    discover_record_files,
    extract_tf_edges,
    imu_payload,
    message_timestamp_ns,
)


def _frame_id(message, fallback: str) -> str:
    header = getattr(message, "header", None)
    return str(
        getattr(message, "frame_id", "") or getattr(header, "frame_id", "") or fallback
    )


def _record_paths(config: AdapterConfig) -> tuple[str, ...]:
    paths = tuple(
        record_path
        for input_path in config.inputs
        for record_path in discover_record_files(str(input_path))
    )
    if not paths:
        raise ValueError("No Apollo record files found in the configured inputs")
    return paths


class ApolloRecordAdapter:
    def __init__(self, config: AdapterConfig):
        self.config = config

    def read(self) -> CanonicalDataset:
        self.config.validate()
        record_paths = _record_paths(self.config)
        scans: list[Scan] = []
        imu_samples: list[tuple[int, np.ndarray, np.ndarray]] = []
        lidar_frame = ""
        imu_frame = ""
        rejected: Counter[str] = Counter()

        for record_path in record_paths:
            with Record(str(record_path)) as record:
                for topic, message, record_timestamp_ns in record.read_messages(
                    topics=(self.config.lidar_topic, self.config.imu_topic)
                ):
                    if message is None:
                        raise RuntimeError(
                            f"Could not decode {topic} from {record_path}"
                        )
                    if topic == self.config.imu_topic:
                        imu = imu_payload(message)
                        timestamp_ns = message_timestamp_ns(
                            topic, message, int(record_timestamp_ns)
                        )
                        if not self.config.includes(timestamp_ns):
                            continue
                        imu_frame = imu_frame or _frame_id(message, "imu")
                        imu_samples.append(
                            (
                                timestamp_ns,
                                np.array(
                                    [
                                        imu.angular_velocity.x,
                                        imu.angular_velocity.y,
                                        imu.angular_velocity.z,
                                    ],
                                    dtype=np.float64,
                                ),
                                np.array(
                                    [
                                        imu.linear_acceleration.x,
                                        imu.linear_acceleration.y,
                                        imu.linear_acceleration.z,
                                    ],
                                    dtype=np.float64,
                                ),
                            )
                        )
                        continue

                    points = message.point
                    if not points:
                        rejected["empty_lidar_scan"] += 1
                        continue
                    point_timestamps = np.fromiter(
                        (point.timestamp for point in points),
                        dtype=np.uint64,
                        count=len(points),
                    )
                    timestamp_ns = int(point_timestamps.min())
                    if not self.config.includes(timestamp_ns):
                        continue
                    xyz = np.asarray(
                        [(point.x, point.y, point.z) for point in points],
                        dtype=np.float32,
                    )
                    intensity = np.fromiter(
                        (point.intensity for point in points),
                        dtype=np.float32,
                        count=len(points),
                    )
                    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(intensity)
                    if not finite.any():
                        rejected["no_finite_lidar_points"] += 1
                        continue
                    lidar_frame = lidar_frame or _frame_id(message, "lidar")
                    scans.append(
                        Scan(
                            timestamp_ns=timestamp_ns,
                            xyz=xyz[finite],
                            intensity=intensity[finite],
                            ring=(
                                np.arange(len(points), dtype=np.uint16)
                                % self.config.scan_lines
                            )[finite],
                            point_time_s=(
                                point_timestamps[finite] - np.uint64(timestamp_ns)
                            ).astype(np.float64)
                            * 1e-9,
                        )
                    )

        record_files = tuple(str(Path(path).resolve()) for path in record_paths)
        transforms = tuple(
            StaticTransform(
                parent_frame=edge.parent_frame,
                child_frame=edge.child_frame,
                matrix=edge.transform,
                source_topic=edge.source_topic,
            )
            for edge in extract_tf_edges(record_files)
            if edge.is_static
        )
        dataset = CanonicalDataset(
            source_type="apollo_record",
            source_files=record_files,
            lidar_topic=self.config.lidar_topic,
            imu_topic=self.config.imu_topic,
            lidar=pack_lidar(scans, lidar_frame or "lidar"),
            imu=pack_imu(imu_samples, imu_frame or "imu"),
            transforms=transforms,
            metadata={
                "adapter": "ApolloRecordAdapter",
                "scan_lines": self.config.scan_lines,
                "rejected": dict(rejected),
                "point_time": {
                    "unit": "seconds",
                    "reference": "minimum point timestamp",
                },
                "ring": {
                    "source": "point_index_modulo_scan_lines",
                    "review_required_for_new_sensor": True,
                },
            },
        )
        dataset.validate()
        return dataset
