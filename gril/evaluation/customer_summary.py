"""Concise customer-facing summary for a completed native GRIL run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from gril.comparison import parse_gril_result
from gril.models import CanonicalDataset


def build_customer_summary(
    result_path: Path,
    dataset: CanonicalDataset,
    *,
    developer_diagnostics: dict[str, str],
) -> dict[str, Any]:
    result = parse_gril_result(result_path)
    return {
        "schema_version": 1,
        "module": "lidar2imu_gril",
        "verdict": "review_required",
        "release_ready": False,
        "message": (
            "Calibration completed. Independent trajectory, point-cloud, "
            "repeatability, and holdout review is still required."
        ),
        "result": result,
        "key_metrics": {
            "lidar_scans": int(len(dataset.lidar.scan_timestamps_ns)),
            "imu_samples": int(len(dataset.imu.timestamps_ns)),
        },
        "next_action": "review_lidar2imu_diagnostics",
        "developer_diagnostics": developer_diagnostics,
    }


def write_customer_summary(
    output_path: Path,
    result_path: Path,
    dataset: CanonicalDataset,
    *,
    developer_diagnostics: dict[str, str],
) -> Path:
    summary = build_customer_summary(
        result_path,
        dataset,
        developer_diagnostics=developer_diagnostics,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(summary, sort_keys=False))
    return output_path
