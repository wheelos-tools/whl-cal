"""Execution boundary for migrated ROS-free GRIL frontend stages."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from gril.config import config_digest, load_algorithm_config
from gril.dataset_io import file_sha256, load_dataset
from gril.frontend_event import write_frontend_event_input
from gril.ground_trace import write_ground_input


@dataclass(frozen=True)
class NativeFrontendRunConfig:
    dataset: Path
    gril_config: Path
    frontend_event_executable: Path
    ground_executable: Path
    output_dir: Path
    scan_count: int = 21

    def validate(self) -> None:
        for name, path in (
            ("canonical dataset", self.dataset),
            ("GRIL config", self.gril_config),
            ("frontend event executable", self.frontend_event_executable),
            ("ground executable", self.ground_executable),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{name} not found: {path}")
        if self.scan_count <= 0:
            raise ValueError("scan_count must be positive")
        available = len(load_dataset(self.dataset).lidar.scan_timestamps_ns)
        if self.scan_count > available:
            raise ValueError(
                f"scan_count {self.scan_count} exceeds {available} available scans"
            )


def build_native_frontend_commands(
    config: NativeFrontendRunConfig,
    event_input: Path,
    ground_input: Path,
) -> tuple[list[str], list[str]]:
    config.validate()
    return (
        [
            str(config.frontend_event_executable.resolve()),
            str(event_input.resolve()),
            str((config.output_dir / "sync_trace.txt").resolve()),
            "--all",
        ],
        [
            str(config.ground_executable.resolve()),
            str(ground_input.resolve()),
            str((config.output_dir / "ground_trace.txt").resolve()),
        ],
    )


def run_native_frontend(config: NativeFrontendRunConfig) -> Path:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    event_input = write_frontend_event_input(
        config.dataset,
        config.gril_config,
        config.output_dir / "frontend_events.txt",
        lidar_scan_count=config.scan_count,
    )
    ground_input = write_ground_input(
        config.dataset,
        config.gril_config,
        config.output_dir / "ground_input.txt",
        scan_count=config.scan_count,
    )
    commands = build_native_frontend_commands(config, event_input, ground_input)
    for command in commands:
        subprocess.run(command, check=True)

    sync_trace = config.output_dir / "sync_trace.txt"
    ground_trace = config.output_dir / "ground_trace.txt"
    for path in (sync_trace, ground_trace):
        if not path.is_file():
            raise RuntimeError(f"Native GRIL frontend did not create {path}")

    algorithm_config = load_algorithm_config(config.gril_config)
    manifest = {
        "engine": "gril_native_frontend_partial",
        "ros_runtime_required": False,
        "input": {
            "dataset": str(config.dataset.resolve()),
            "scan_count": config.scan_count,
        },
        "algorithm_config": {
            "path": str(config.gril_config.resolve()),
            "digest": config_digest(algorithm_config),
        },
        "completed_stages": [
            "velodyne_preprocessing",
            "fifo_synchronization",
            "patchworkpp_ground_segmentation",
        ],
        "pending_stages": [
            "ikd_tree_lidar_odometry_event_alignment",
            "odometry_feedback_cv_propagation",
            "ground_constraint_alignment",
            "batch_calibration",
        ],
        "final_calibration_produced": False,
        "artifacts": {
            "sync_trace": {
                "path": str(sync_trace.resolve()),
                "sha256": file_sha256(sync_trace),
            },
            "ground_trace": {
                "path": str(ground_trace.resolve()),
                "sha256": file_sha256(ground_trace),
            },
        },
    }
    manifest_path = config.output_dir / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest_path
