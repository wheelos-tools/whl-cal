"""Execution boundary for the ROS-free GRIL batch core."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from gril.config import config_digest, load_algorithm_config
from gril.dataset_io import file_sha256

_NATIVE_DEFAULTS = {
    "data_accum_length": 300.0,
    "x_accumulate": 0.1,
    "y_accumulate": 0.1,
    "z_accumulate": 0.1,
    "svd_threshold": 0.01,
    "imu_sensor_height": 0.1,
    "trans_IL_x": 0.0,
    "trans_IL_y": 0.0,
    "trans_IL_z": 0.0,
    "bound_th": 0.1,
    "set_boundary": False,
    "verbose": False,
    "gyro_factor": 1.0,
    "acc_factor": 1.0,
    "ground_factor": 1.0,
}


@dataclass(frozen=True)
class NativeBatchRunConfig:
    executable: Path
    trace: Path
    gril_config: Path
    output_dir: Path

    def validate(self) -> None:
        for name, path in (
            ("native executable", self.executable),
            ("batch trace", self.trace),
            ("GRIL config", self.gril_config),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{name} not found: {path}")


def native_config_values(gril_config_path: Path) -> dict[str, Any]:
    algorithm = load_algorithm_config(gril_config_path)
    calibration = algorithm["calibration"]
    values = dict(_NATIVE_DEFAULTS)
    for key in values:
        if key in calibration:
            values[key] = calibration[key]
    return values


def write_native_config(gril_config_path: Path, output_path: Path) -> Path:
    values = native_config_values(gril_config_path)
    lines = ["# Generated from the frozen GRIL algorithm configuration."]
    for key, value in values.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        else:
            rendered = str(value)
        lines.append(f"{key} {rendered}")
    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def build_native_command(
    config: NativeBatchRunConfig, native_config_path: Path
) -> list[str]:
    config.validate()
    return [
        str(config.executable.resolve()),
        "--trace",
        str(config.trace.resolve()),
        "--config",
        str(native_config_path.resolve()),
        "--output",
        str((config.output_dir / "GRIL_Calib_result.txt").resolve()),
    ]


def run_native_batch(config: NativeBatchRunConfig) -> Path:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    native_config_path = write_native_config(
        config.gril_config, config.output_dir / "gril_native.conf"
    )
    subprocess.run(
        build_native_command(config, native_config_path),
        check=True,
        cwd=config.output_dir,
    )
    result_path = config.output_dir / "GRIL_Calib_result.txt"
    if not result_path.is_file():
        raise RuntimeError(f"Native GRIL did not create its result: {result_path}")
    algorithm_config = load_algorithm_config(config.gril_config)
    manifest = {
        "engine": "gril_native_batch",
        "ros_runtime_required": False,
        "scope": "pre-LI_Calibration batch trace replay",
        "executable": {
            "path": str(config.executable.resolve()),
            "sha256": file_sha256(config.executable),
        },
        "trace": {
            "path": str(config.trace.resolve()),
            "sha256": file_sha256(config.trace),
            "schema": "GRIL_BATCH_TRACE 1",
        },
        "algorithm_config": {
            "path": str(config.gril_config.resolve()),
            "digest": config_digest(algorithm_config),
            "native_path": str(native_config_path.resolve()),
        },
        "result": {
            "path": str(result_path.resolve()),
            "sha256": file_sha256(result_path),
        },
    }
    manifest_path = config.output_dir / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return result_path
