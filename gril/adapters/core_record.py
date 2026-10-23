"""C++ wheelos_core-backed Apollo record adapter."""

from __future__ import annotations

import os
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from gril.adapters.base import AdapterConfig, InputAdapter
from gril.adapters.record_files import record_paths
from gril.models import CanonicalDataset, ImuBatch, LidarBatch, StaticTransform

_MAGIC = b"GRIL_CANONICAL_NATIVE 1\n"
_POINT_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("intensity", "<f4"),
        ("ring", "<u2"),
        ("point_time_s", "<f4"),
    ]
)
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


def _read_exact(stream, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ValueError("Truncated C++ canonical interchange file")
    return value


def _read_u32(stream) -> int:
    return struct.unpack("<I", _read_exact(stream, 4))[0]


def _read_u64(stream) -> int:
    return struct.unpack("<Q", _read_exact(stream, 8))[0]


def _read_string(stream) -> str:
    return _read_exact(stream, _read_u32(stream)).decode("utf-8")


def _resolve_executable(config: AdapterConfig) -> Path:
    executable = config.core_record_ingress
    if executable is None:
        configured = os.environ.get("GRIL_CORE_RECORD_INGRESS")
        if configured:
            executable = Path(configured)
    if executable is None:
        raise FileNotFoundError(
            "C++ record ingress executable is required; build "
            "tools/gril_record_core_reader:record_ingress and pass "
            "--core-record-ingress or set GRIL_CORE_RECORD_INGRESS"
        )
    executable = Path(executable).expanduser().resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FileNotFoundError(
            f"C++ record ingress executable not found: {executable}"
        )
    return executable


def _read_interchange(path: Path, config: AdapterConfig, source_files: tuple[str, ...]):
    with path.open("rb") as stream:
        if _read_exact(stream, len(_MAGIC)) != _MAGIC:
            raise ValueError("Unsupported C++ canonical interchange version")
        lidar_frame = _read_string(stream)
        imu_frame = _read_string(stream)
        scan_count = _read_u64(stream)
        point_count = _read_u64(stream)
        imu_count = _read_u64(stream)
        transform_count = _read_u64(stream)
        if scan_count > point_count + 1 or point_count > 1_000_000_000:
            raise ValueError("Invalid counts in C++ canonical interchange file")
        if imu_count > 100_000_000 or transform_count > 1_000_000:
            raise ValueError("Invalid counts in C++ canonical interchange file")

        scan_timestamps = np.fromfile(stream, dtype="<i8", count=scan_count)
        offsets = np.fromfile(stream, dtype="<u8", count=scan_count + 1)
        if len(scan_timestamps) != scan_count or len(offsets) != scan_count + 1:
            raise ValueError("Truncated scan index in C++ canonical interchange file")
        if int(offsets[-1]) != point_count:
            raise ValueError("Point count does not match scan offsets")

        points = np.fromfile(stream, dtype=_POINT_DTYPE, count=point_count)
        if len(points) != point_count:
            raise ValueError("Truncated point data in C++ canonical interchange file")
        imu_timestamps = np.fromfile(stream, dtype="<i8", count=imu_count)
        imu_values = np.fromfile(
            stream,
            dtype=np.dtype(
                [
                    ("angular_velocity", "<f8", (3,)),
                    ("linear_acceleration", "<f8", (3,)),
                ]
            ),
            count=imu_count,
        )
        if len(imu_timestamps) != imu_count or len(imu_values) != imu_count:
            raise ValueError("Truncated IMU data in C++ canonical interchange file")

        transforms = []
        for _ in range(transform_count):
            parent = _read_string(stream)
            child = _read_string(stream)
            source_topic = _read_string(stream)
            matrix = np.fromfile(stream, dtype="<f8", count=16)
            if len(matrix) != 16:
                raise ValueError("Truncated transform in C++ interchange file")
            transforms.append(
                StaticTransform(
                    parent_frame=parent,
                    child_frame=child,
                    matrix=matrix.reshape((4, 4)),
                    source_topic=source_topic,
                )
            )
        if stream.read(1):
            raise ValueError("Unexpected trailing bytes in C++ interchange file")

    lidar = LidarBatch(
        frame_id=lidar_frame,
        scan_timestamps_ns=scan_timestamps,
        scan_offsets=offsets.astype(np.int64, copy=False),
        xyz=np.column_stack((points["x"], points["y"], points["z"])),
        intensity=points["intensity"],
        ring=points["ring"],
        point_time_s=points["point_time_s"],
    )
    imu = ImuBatch(
        frame_id=imu_frame,
        timestamps_ns=imu_timestamps,
        angular_velocity=imu_values["angular_velocity"],
        linear_acceleration=imu_values["linear_acceleration"],
    )
    dataset = CanonicalDataset(
        source_type="apollo_record",
        source_files=source_files,
        lidar_topic=config.lidar_topic,
        imu_topic=config.imu_topic,
        lidar=lidar,
        imu=imu,
        transforms=tuple(transforms),
        metadata={
            "adapter": "CoreApolloRecordAdapter",
            "scan_lines": config.scan_lines,
            "rejected": {},
            "point_time": {
                "unit": "seconds",
                "reference": "minimum point timestamp",
            },
            "ring": {
                "source": "point_index_modulo_scan_lines",
                "review_required_for_new_sensor": True,
            },
            "record_reader": "wheelos_core::RecordReader",
        },
    )
    dataset.validate()
    return dataset


class CoreApolloRecordAdapter(InputAdapter):
    """Decode Apollo records and construct canonical arrays in C++."""

    def __init__(self, config: AdapterConfig):
        self.config = config

    def read(self) -> CanonicalDataset:
        self.config.validate()
        executable = _resolve_executable(self.config)
        record_files = record_paths(self.config)
        source_files = tuple(str(Path(path).resolve()) for path in record_files)
        start_ns = _INT64_MIN if self.config.start_ns is None else self.config.start_ns
        end_ns = _INT64_MAX if self.config.end_ns is None else self.config.end_ns

        with tempfile.TemporaryDirectory(prefix="gril-core-record-") as temporary:
            interchange_path = Path(temporary) / "canonical.bin"
            command = [
                str(executable),
                str(interchange_path),
                self.config.lidar_topic,
                self.config.imu_topic,
                str(self.config.scan_lines),
                str(start_ns),
                str(end_ns),
                *record_files,
            ]
            subprocess.run(command, check=True)
            return _read_interchange(interchange_path, self.config, source_files)
