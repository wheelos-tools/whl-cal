"""Frozen ROS-GRIL reference orchestration.

This module delegates to the executable validation workflow instead of
duplicating conversion, container, or diagnostic behavior.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PIPELINE = (
    Path(__file__).resolve().parents[1]
    / ".agents/skills/gril-calib-validation/scripts/gril_pipeline.py"
)


@dataclass(frozen=True)
class ReferenceRunConfig:
    record_files: tuple[Path, ...]
    output_dir: Path
    lidar_topic: str
    imu_topic: str
    pose_topic: str
    tf_parent: str
    tf_child: str
    scan_lines: int
    runs: int = 2
    workspace: Path = Path(".cache/gril-validation")
    image: str = "gril-calib-validation:2026-08"
    skip_submap: bool = False

    def validate(self) -> None:
        if not self.record_files:
            raise ValueError("The ROS-GRIL reference requires Apollo record files")
        missing = [str(path) for path in self.record_files if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Record files not found: {', '.join(missing)}")
        if self.scan_lines <= 0 or self.runs <= 0:
            raise ValueError("scan_lines and runs must be positive")


def build_reference_command(config: ReferenceRunConfig) -> list[str]:
    config.validate()
    command = [
        sys.executable,
        str(PIPELINE),
        "all",
        "--output-dir",
        str(config.output_dir),
        "--workspace",
        str(config.workspace),
        "--lidar-topic",
        config.lidar_topic,
        "--imu-topic",
        config.imu_topic,
        "--pose-topic",
        config.pose_topic,
        "--tf-parent",
        config.tf_parent,
        "--tf-child",
        config.tf_child,
        "--scan-lines",
        str(config.scan_lines),
        "--runs",
        str(config.runs),
        "--image",
        config.image,
    ]
    for path in config.record_files:
        command.extend(["--record-file", str(path)])
    if config.skip_submap:
        command.append("--skip-submap")
    return command


def run_reference(config: ReferenceRunConfig) -> None:
    subprocess.run(build_reference_command(config), check=True)
