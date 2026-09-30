"""Input-contract diagnostics kept outside adapters and algorithms."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from gril.models import CanonicalDataset


def _percentiles(values: np.ndarray) -> dict[str, float] | None:
    if len(values) == 0:
        return None
    quantiles = (0, 50, 95, 100)
    return {
        f"p{quantile}": float(value)
        for quantile, value in zip(quantiles, np.percentile(values, quantiles))
    }


def build_input_contract(dataset: CanonicalDataset) -> dict[str, Any]:
    dataset.validate()
    lidar = dataset.lidar.normalized()
    imu = dataset.imu.normalized()
    scan_durations = np.array(
        [
            float(lidar.point_time_s[end - 1]) if end > start else 0.0
            for start, end in zip(lidar.scan_offsets[:-1], lidar.scan_offsets[1:])
        ],
        dtype=np.float64,
    )
    lidar_gaps = np.diff(lidar.scan_timestamps_ns).astype(np.float64) * 1e-9
    imu_gaps = np.diff(imu.timestamps_ns).astype(np.float64) * 1e-9
    acceleration_norm = np.linalg.norm(imu.linear_acceleration, axis=1)
    unique_rings = np.unique(lidar.ring)
    return {
        "verdict": "accepted",
        "schema_version": dataset.schema_version,
        "source_type": dataset.source_type,
        "counts": {
            "lidar_scans": int(len(lidar.scan_timestamps_ns)),
            "lidar_points": int(len(lidar.xyz)),
            "imu_samples": int(len(imu.timestamps_ns)),
            "static_transforms": int(len(dataset.transforms)),
        },
        "frames": {
            "lidar": lidar.frame_id,
            "imu": imu.frame_id,
        },
        "point_contract": {
            "all_xyz_finite": bool(np.isfinite(lidar.xyz).all()),
            "all_intensity_finite": bool(np.isfinite(lidar.intensity).all()),
            "all_point_time_finite": bool(np.isfinite(lidar.point_time_s).all()),
            "point_time_unit": "seconds",
            "scan_duration_s": _percentiles(scan_durations),
            "ring_min": int(unique_rings.min()),
            "ring_max": int(unique_rings.max()),
            "ring_count": int(len(unique_rings)),
        },
        "timestamp_contract": {
            "lidar_monotonic": bool(np.all(lidar_gaps >= 0.0)),
            "imu_monotonic": bool(np.all(imu_gaps >= 0.0)),
            "lidar_gap_s": _percentiles(lidar_gaps),
            "imu_gap_s": _percentiles(imu_gaps),
        },
        "imu_contract": {
            "all_values_finite": bool(
                np.isfinite(imu.angular_velocity).all()
                and np.isfinite(imu.linear_acceleration).all()
            ),
            "acceleration_norm_m_s2": _percentiles(acceleration_norm),
        },
        "adapter_metadata": dataset.metadata,
    }


def write_input_contract(dataset: CanonicalDataset, output_dir: Path) -> Path:
    output_path = Path(output_dir) / "input_contract.yaml"
    output_path.write_text(
        yaml.safe_dump(build_input_contract(dataset), sort_keys=False)
    )
    return output_path
