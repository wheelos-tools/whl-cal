"""Stable serialization for canonical GRIL datasets."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from gril.models import CanonicalDataset, ImuBatch, LidarBatch, StaticTransform

MANIFEST_NAME = "dataset.yaml"
LIDAR_NAME = "lidar.npz"
IMU_NAME = "imu.npz"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_metadata(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_dataset(dataset: CanonicalDataset, output_dir: Path) -> Path:
    dataset.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lidar_path = output_dir / LIDAR_NAME
    imu_path = output_dir / IMU_NAME
    manifest_path = output_dir / MANIFEST_NAME

    lidar = dataset.lidar.normalized()
    imu = dataset.imu.normalized()
    np.savez(
        lidar_path,
        scan_timestamps_ns=lidar.scan_timestamps_ns,
        scan_offsets=lidar.scan_offsets,
        xyz=lidar.xyz,
        intensity=lidar.intensity,
        ring=lidar.ring,
        point_time_s=lidar.point_time_s,
    )
    np.savez(
        imu_path,
        timestamps_ns=imu.timestamps_ns,
        angular_velocity=imu.angular_velocity,
        linear_acceleration=imu.linear_acceleration,
    )

    manifest = {
        "schema_version": dataset.schema_version,
        "source": {
            "type": dataset.source_type,
            "files": list(dataset.source_files),
        },
        "topics": {
            "lidar": dataset.lidar_topic,
            "imu": dataset.imu_topic,
        },
        "frames": {
            "lidar": lidar.frame_id,
            "imu": imu.frame_id,
        },
        "counts": {
            "lidar_scans": int(len(lidar.scan_timestamps_ns)),
            "lidar_points": int(len(lidar.xyz)),
            "imu_samples": int(len(imu.timestamps_ns)),
        },
        "arrays": {
            "lidar": {
                "path": LIDAR_NAME,
                "sha256": file_sha256(lidar_path),
            },
            "imu": {
                "path": IMU_NAME,
                "sha256": file_sha256(imu_path),
            },
        },
        "transforms": [
            {
                "parent_frame": transform.parent_frame,
                "child_frame": transform.child_frame,
                "matrix": transform.normalized().matrix.tolist(),
                "source_topic": transform.source_topic,
            }
            for transform in dataset.transforms
        ],
        "metadata": _safe_metadata(dataset.metadata),
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest_path


def load_dataset(path: Path, verify_hashes: bool = True) -> CanonicalDataset:
    path = Path(path)
    manifest_path = path if path.is_file() else path / MANIFEST_NAME
    root = manifest_path.parent
    manifest = yaml.safe_load(manifest_path.read_text())
    arrays = manifest["arrays"]
    lidar_path = root / arrays["lidar"]["path"]
    imu_path = root / arrays["imu"]["path"]
    if verify_hashes:
        for name, array_path in (("lidar", lidar_path), ("imu", imu_path)):
            actual = file_sha256(array_path)
            expected = arrays[name]["sha256"]
            if actual != expected:
                raise ValueError(
                    f"{name} array hash mismatch: expected {expected}, got {actual}"
                )

    with np.load(lidar_path, allow_pickle=False) as values:
        lidar = LidarBatch(
            frame_id=manifest["frames"]["lidar"],
            scan_timestamps_ns=values["scan_timestamps_ns"],
            scan_offsets=values["scan_offsets"],
            xyz=values["xyz"],
            intensity=values["intensity"],
            ring=values["ring"],
            point_time_s=values["point_time_s"],
        )
    with np.load(imu_path, allow_pickle=False) as values:
        imu = ImuBatch(
            frame_id=manifest["frames"]["imu"],
            timestamps_ns=values["timestamps_ns"],
            angular_velocity=values["angular_velocity"],
            linear_acceleration=values["linear_acceleration"],
        )

    dataset = CanonicalDataset(
        source_type=manifest["source"]["type"],
        source_files=tuple(manifest["source"]["files"]),
        lidar_topic=manifest["topics"]["lidar"],
        imu_topic=manifest["topics"]["imu"],
        lidar=lidar,
        imu=imu,
        transforms=tuple(
            StaticTransform(
                parent_frame=item["parent_frame"],
                child_frame=item["child_frame"],
                matrix=np.asarray(item["matrix"], dtype=np.float64),
                source_topic=item["source_topic"],
            )
            for item in manifest.get("transforms", [])
        ),
        metadata=manifest.get("metadata", {}),
        schema_version=int(manifest["schema_version"]),
    )
    dataset.validate()
    return dataset
