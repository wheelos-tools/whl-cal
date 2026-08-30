"""Dataset and result equivalence checks for GRIL migration."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from common.geometry import euler_xyz_degrees_to_matrix
from gril.models import CanonicalDataset


def _max_error(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.shape != right.shape or left.size == 0:
        return None
    return float(np.max(np.abs(left.astype(np.float64) - right)))


def compare_datasets(
    left: CanonicalDataset,
    right: CanonicalDataset,
    point_tolerance: float = 1e-6,
    point_time_tolerance_s: float = 1e-8,
    imu_tolerance: float = 1e-12,
) -> dict[str, Any]:
    left.validate()
    right.validate()
    checks = {
        "lidar_frame": left.lidar.frame_id == right.lidar.frame_id,
        "imu_frame": left.imu.frame_id == right.imu.frame_id,
        "scan_count": len(left.lidar.scan_timestamps_ns)
        == len(right.lidar.scan_timestamps_ns),
        "point_count": len(left.lidar.xyz) == len(right.lidar.xyz),
        "imu_count": len(left.imu.timestamps_ns) == len(right.imu.timestamps_ns),
        "scan_timestamps": np.array_equal(
            left.lidar.scan_timestamps_ns, right.lidar.scan_timestamps_ns
        ),
        "scan_offsets": np.array_equal(
            left.lidar.scan_offsets, right.lidar.scan_offsets
        ),
        "rings": np.array_equal(left.lidar.ring, right.lidar.ring),
        "imu_timestamps": np.array_equal(
            left.imu.timestamps_ns, right.imu.timestamps_ns
        ),
    }
    errors = {
        "xyz_max_absolute_error": _max_error(left.lidar.xyz, right.lidar.xyz),
        "intensity_max_absolute_error": _max_error(
            left.lidar.intensity, right.lidar.intensity
        ),
        "point_time_max_absolute_error_s": _max_error(
            left.lidar.point_time_s, right.lidar.point_time_s
        ),
        "angular_velocity_max_absolute_error": _max_error(
            left.imu.angular_velocity, right.imu.angular_velocity
        ),
        "linear_acceleration_max_absolute_error": _max_error(
            left.imu.linear_acceleration, right.imu.linear_acceleration
        ),
    }
    numeric_checks = {
        "xyz": errors["xyz_max_absolute_error"] is not None
        and errors["xyz_max_absolute_error"] <= point_tolerance,
        "intensity": errors["intensity_max_absolute_error"] is not None
        and errors["intensity_max_absolute_error"] <= point_tolerance,
        "point_time": errors["point_time_max_absolute_error_s"] is not None
        and errors["point_time_max_absolute_error_s"] <= point_time_tolerance_s,
        "angular_velocity": errors["angular_velocity_max_absolute_error"] is not None
        and errors["angular_velocity_max_absolute_error"] <= imu_tolerance,
        "linear_acceleration": (
            errors["linear_acceleration_max_absolute_error"] is not None
            and errors["linear_acceleration_max_absolute_error"] <= imu_tolerance
        ),
    }
    checks.update(numeric_checks)
    return {
        "verdict": "equivalent" if all(checks.values()) else "different",
        "checks": checks,
        "errors": errors,
        "tolerances": {
            "point": point_tolerance,
            "point_time_s": point_time_tolerance_s,
            "imu": imu_tolerance,
        },
    }


def parse_gril_result(path: Path) -> dict[str, Any]:
    text = Path(path).read_text()

    def values(label: str) -> list[float]:
        match = re.search(rf"{label}[^=]*=\s*([^\n]+)", text)
        if match is None:
            raise ValueError(f"Missing {label!r} in {path}")
        return [float(value) for value in match.group(1).split()]

    rotation = values("Rotation LiDAR to IMU")
    translation = values("Translation LiDAR to IMU")
    time_offset = values("Time Lag IMU to LiDAR")
    if len(rotation) != 3 or len(translation) != 3 or len(time_offset) != 1:
        raise ValueError(f"Unexpected GRIL result shape in {path}")
    return {
        "rotation_xyz_deg": rotation,
        "translation_m": translation,
        "time_offset_s": time_offset[0],
    }


def compare_results(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    rotation_tolerance_deg: float = 0.2,
    translation_tolerance_m: float = 0.03,
    time_tolerance_s: float = 0.001,
) -> dict[str, Any]:
    reference_rotation = euler_xyz_degrees_to_matrix(reference["rotation_xyz_deg"])
    candidate_rotation = euler_xyz_degrees_to_matrix(candidate["rotation_xyz_deg"])
    rotation_delta = reference_rotation.T @ candidate_rotation
    rotation_error = math.degrees(
        math.acos(float(np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0)))
    )
    translation_error = float(
        np.linalg.norm(
            np.asarray(reference["translation_m"])
            - np.asarray(candidate["translation_m"])
        )
    )
    time_error = abs(reference["time_offset_s"] - candidate["time_offset_s"])
    checks = {
        "rotation": rotation_error <= rotation_tolerance_deg,
        "translation": translation_error <= translation_tolerance_m,
        "time_offset": time_error <= time_tolerance_s,
    }
    return {
        "verdict": "equivalent" if all(checks.values()) else "different",
        "checks": checks,
        "errors": {
            "rotation_deg": rotation_error,
            "translation_m": translation_error,
            "time_offset_s": time_error,
        },
        "tolerances": {
            "rotation_deg": rotation_tolerance_deg,
            "translation_m": translation_tolerance_m,
            "time_offset_s": time_tolerance_s,
        },
        "reference": reference,
        "candidate": candidate,
    }
