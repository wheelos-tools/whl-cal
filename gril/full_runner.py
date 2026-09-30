"""Complete canonical-dataset to ROS-free GRIL execution boundary."""

from __future__ import annotations

import math
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
import yaml

from gril.config import config_digest, load_algorithm_config
from gril.dataset_io import file_sha256, load_dataset
from gril.evaluation.customer_summary import write_customer_summary
from gril.models import CanonicalDataset
from gril.native_runner import write_native_config

_EVENT_COUNT = struct.Struct("<Q")
_IMU_EVENT = struct.Struct("<Bq6d")
_LIDAR_EVENT = struct.Struct("<BIqQ")
_POINT_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("intensity", "<f4"),
        ("time_s", "<f4"),
        ("ring", "<u2"),
    ],
    align=False,
)


@dataclass(frozen=True)
class NativeFullRunConfig:
    dataset: Path
    gril_config: Path
    executable: Path
    output_dir: Path
    batch_executable: Path | None = None
    scan_count: int | None = None
    gap_policy: str = "golden"
    forward_gap_s: float | None = None
    configured_time_lag_s: float = 0.0

    def resolved_batch_executable(self) -> Path:
        return (
            self.batch_executable
            if self.batch_executable is not None
            else self.executable.with_name("gril_native_batch")
        )

    def validate(self) -> None:
        for name, path in (
            ("canonical dataset", self.dataset),
            ("GRIL config", self.gril_config),
            ("full frontend executable", self.executable),
            ("native batch executable", self.resolved_batch_executable()),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{name} not found: {path}")
        if self.scan_count is not None and self.scan_count <= 0:
            raise ValueError("scan_count must be positive when specified")
        if self.gap_policy not in {"golden", "reset"}:
            raise ValueError("gap_policy must be golden or reset")
        if self.gap_policy == "reset" and (
            self.forward_gap_s is None or self.forward_gap_s <= 0.0
        ):
            raise ValueError("reset gap policy requires positive forward_gap_s")
        if self.gap_policy == "golden" and self.forward_gap_s is not None:
            raise ValueError("forward_gap_s is only valid with reset gap policy")
        if not math.isfinite(self.configured_time_lag_s):
            raise ValueError("configured_time_lag_s must be finite")


def _required(section: dict[str, Any], names: tuple[str, ...], label: str) -> None:
    missing = [name for name in names if name not in section]
    if missing:
        raise ValueError(f"GRIL {label} config is missing: {', '.join(missing)}")


def write_full_native_config(
    gril_config_path: Path,
    output_path: Path,
    *,
    configured_time_lag_s: float = 0.0,
) -> Path:
    algorithm = load_algorithm_config(gril_config_path)
    launch = algorithm["launch"]
    preprocess = algorithm["preprocess"]
    calibration = algorithm["calibration"]
    mapping = algorithm["mapping"]
    patchwork = algorithm["patchworkpp"]
    czm = patchwork.get("czm")
    if not isinstance(czm, dict):
        raise ValueError("GRIL Patchwork++ config is missing czm")

    _required(launch, ("max_iteration", "cube_side_length"), "launch")
    _required(
        preprocess,
        (
            "lidar_type",
            "scan_line",
            "blind",
            "point_filter_num",
            "feature_extract_en",
        ),
        "preprocess",
    )
    _required(
        calibration,
        (
            "cut_frame",
            "cut_frame_num",
            "orig_odom_freq",
            "mean_acc_norm",
            "data_accum_length",
            "x_accumulate",
            "y_accumulate",
            "z_accumulate",
            "imu_sensor_height",
            "trans_IL_x",
            "trans_IL_y",
            "trans_IL_z",
            "bound_th",
            "set_boundary",
            "verbose",
            "gyro_factor",
            "acc_factor",
            "ground_factor",
        ),
        "calibration",
    )
    _required(
        mapping,
        (
            "filter_size_surf",
            "filter_size_map",
            "gyr_cov",
            "acc_cov",
            "det_range",
            "ground_cov",
        ),
        "mapping",
    )
    _required(
        patchwork,
        (
            "sensor_height",
            "verbose",
            "num_iter",
            "num_lpr",
            "num_min_pts",
            "max_flatness_storage",
            "max_elevation_storage",
            "th_seeds",
            "th_dist",
            "th_seeds_v",
            "th_dist_v",
            "max_r",
            "min_r",
            "uprightness_thr",
            "adaptive_seed_selection_margin",
            "RNR_ver_angle_thr",
            "RNR_intensity_thr",
            "enable_RNR",
            "enable_RVPF",
            "enable_TGR",
        ),
        "Patchwork++",
    )
    _required(
        czm,
        (
            "num_zones",
            "num_sectors_each_zone",
            "mum_rings_each_zone",
            "elevation_thresholds",
            "flatness_thresholds",
        ),
        "Patchwork++ CZM",
    )
    if patchwork.get("mode") != "czm":
        raise ValueError("full native GRIL requires Patchwork++ mode=czm")
    if int(czm["num_zones"]) != 4:
        raise ValueError("full native GRIL requires four Patchwork++ CZM zones")
    for name in (
        "num_sectors_each_zone",
        "mum_rings_each_zone",
        "elevation_thresholds",
        "flatness_thresholds",
    ):
        if len(czm[name]) != 4:
            raise ValueError(f"Patchwork++ CZM {name} must contain four zones")
    if not math.isfinite(configured_time_lag_s):
        raise ValueError("configured_time_lag_s must be finite")

    def boolean(value: Any) -> int:
        return 1 if bool(value) else 0

    def vector(label: str, values: Any) -> str:
        items = list(values)
        return f"{label} {len(items)} " + " ".join(map(str, items))

    svd_threshold = calibration.get("svd_threshold", 0.01)
    lines = [
        "GRIL_NATIVE_FULL_CONFIG 1",
        "preprocess "
        + " ".join(
            map(
                str,
                (
                    int(preprocess["lidar_type"]),
                    int(preprocess["scan_line"]),
                    float(preprocess["blind"]),
                    int(preprocess["point_filter_num"]),
                    boolean(preprocess["feature_extract_en"]),
                    boolean(calibration["cut_frame"]),
                    int(calibration["cut_frame_num"]),
                ),
            )
        ),
        "calibration "
        + " ".join(
            map(
                str,
                (
                    int(calibration["orig_odom_freq"]),
                    float(calibration["mean_acc_norm"]),
                    float(calibration["data_accum_length"]),
                    float(calibration["x_accumulate"]),
                    float(calibration["y_accumulate"]),
                    float(calibration["z_accumulate"]),
                    float(svd_threshold),
                    float(calibration["imu_sensor_height"]),
                    float(calibration["trans_IL_x"]),
                    float(calibration["trans_IL_y"]),
                    float(calibration["trans_IL_z"]),
                    float(calibration["bound_th"]),
                    boolean(calibration["set_boundary"]),
                    boolean(calibration["verbose"]),
                    float(calibration["gyro_factor"]),
                    float(calibration["acc_factor"]),
                    float(calibration["ground_factor"]),
                ),
            )
        ),
        "mapping "
        + " ".join(
            map(
                str,
                (
                    int(launch["max_iteration"]),
                    float(launch["cube_side_length"]),
                    float(mapping["filter_size_surf"]),
                    float(mapping["filter_size_map"]),
                    float(mapping["gyr_cov"]),
                    float(mapping["acc_cov"]),
                    float(mapping["det_range"]),
                    float(mapping["ground_cov"]),
                ),
            )
        ),
        "patchwork "
        + " ".join(
            map(
                str,
                (
                    float(patchwork["sensor_height"]),
                    int(patchwork["num_iter"]),
                    int(patchwork["num_lpr"]),
                    int(patchwork["num_min_pts"]),
                    int(patchwork["max_flatness_storage"]),
                    int(patchwork["max_elevation_storage"]),
                    float(patchwork["th_seeds"]),
                    float(patchwork["th_dist"]),
                    float(patchwork["th_seeds_v"]),
                    float(patchwork["th_dist_v"]),
                    float(patchwork["max_r"]),
                    float(patchwork["min_r"]),
                    float(patchwork["uprightness_thr"]),
                    float(patchwork["adaptive_seed_selection_margin"]),
                    float(patchwork["RNR_ver_angle_thr"]),
                    float(patchwork["RNR_intensity_thr"]),
                    boolean(patchwork["verbose"]),
                    boolean(patchwork["enable_RNR"]),
                    boolean(patchwork["enable_RVPF"]),
                    boolean(patchwork["enable_TGR"]),
                ),
            )
        ),
        vector("sectors", czm["num_sectors_each_zone"]),
        vector("rings", czm["mum_rings_each_zone"]),
        vector("elevation", czm["elevation_thresholds"]),
        vector("flatness", czm["flatness_thresholds"]),
        f"runtime {float(configured_time_lag_s)}",
        "END",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def write_full_native_input(
    dataset_path: Union[Path, CanonicalDataset],
    output_path: Path,
    *,
    scan_count: int | None = None,
) -> Path:
    dataset = (
        dataset_path
        if isinstance(dataset_path, CanonicalDataset)
        else load_dataset(dataset_path)
    )
    lidar = dataset.lidar.normalized()
    imu = dataset.imu.normalized()
    selected_scans = len(lidar.scan_timestamps_ns) if scan_count is None else scan_count
    if selected_scans <= 0 or selected_scans > len(lidar.scan_timestamps_ns):
        raise ValueError(
            f"scan_count {selected_scans} exceeds "
            f"{len(lidar.scan_timestamps_ns)} available scans"
        )

    events: list[tuple[int, int, int]] = [
        (int(lidar.scan_timestamps_ns[index]), 1, index)
        for index in range(selected_scans)
    ]
    # Preserve the complete IMU stream.  A source may require fallback
    # azimuth timing or discover a hard sensor-clock offset only after the
    # first LiDAR event, so a raw point-time-derived cutoff can omit samples
    # needed to synchronize the final selected scan.
    events.extend(
        (int(imu.timestamps_ns[index]), 0, index)
        for index in range(len(imu.timestamps_ns))
    )
    events.sort()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output:
        output.write(b"GRIL_NATIVE_DATASET 1\n")
        output.write(_EVENT_COUNT.pack(len(events)))
        for timestamp_ns, event_type, index in events:
            if event_type == 0:
                output.write(
                    _IMU_EVENT.pack(
                        0,
                        timestamp_ns,
                        *imu.angular_velocity[index],
                        *imu.linear_acceleration[index],
                    )
                )
                continue

            start = int(lidar.scan_offsets[index])
            end = int(lidar.scan_offsets[index + 1])
            output.write(_LIDAR_EVENT.pack(1, index + 1, timestamp_ns, end - start))
            points = np.empty(end - start, dtype=_POINT_DTYPE)
            points["x"] = lidar.xyz[start:end, 0]
            points["y"] = lidar.xyz[start:end, 1]
            points["z"] = lidar.xyz[start:end, 2]
            points["intensity"] = lidar.intensity[start:end]
            points["time_s"] = lidar.point_time_s[start:end]
            points["ring"] = lidar.ring[start:end]
            output.write(points.tobytes(order="C"))
    return output_path


def build_full_native_command(
    config: NativeFullRunConfig,
    input_path: Path,
    native_config_path: Path,
) -> list[str]:
    config.validate()
    command = [
        str(config.executable.resolve()),
        "--input",
        str(input_path.resolve()),
        "--config",
        str(native_config_path.resolve()),
        "--output",
        str((config.output_dir / "GRIL_Calib_result.txt").resolve()),
        "--trace",
        str((config.output_dir / "GRIL_full_frontend_trace_v1.txt").resolve()),
        "--batch-trace",
        str((config.output_dir / "GRIL_batch_trace_v1.txt").resolve()),
        "--batch-executable",
        str(config.resolved_batch_executable().resolve()),
        "--batch-config",
        str((config.output_dir / "gril_native.conf").resolve()),
        "--gap-policy",
        config.gap_policy,
    ]
    if config.gap_policy == "reset":
        command.extend(["--forward-gap-s", str(config.forward_gap_s)])
    return command


def run_full_native(config: NativeFullRunConfig) -> Path:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(config.dataset)
    input_path = write_full_native_input(
        dataset,
        config.output_dir / "native_dataset_v1.bin",
        scan_count=config.scan_count,
    )
    native_config_path = write_full_native_config(
        config.gril_config,
        config.output_dir / "gril_native_full.conf",
        configured_time_lag_s=config.configured_time_lag_s,
    )
    write_native_config(
        config.gril_config,
        config.output_dir / "gril_native.conf",
    )
    result_path = config.output_dir / "GRIL_Calib_result.txt"
    trace_path = config.output_dir / "GRIL_full_frontend_trace_v1.txt"
    batch_trace_path = config.output_dir / "GRIL_batch_trace_v1.txt"
    manifest_path = config.output_dir / "manifest.yaml"
    for stale in (result_path, trace_path, batch_trace_path, manifest_path):
        stale.unlink(missing_ok=True)

    subprocess.run(
        build_full_native_command(config, input_path, native_config_path),
        check=True,
        cwd=config.output_dir,
    )
    for name, path in (
        ("GRIL result", result_path),
        ("full frontend trace", trace_path),
        ("batch handoff trace", batch_trace_path),
    ):
        if not path.is_file():
            raise RuntimeError(f"Native full frontend did not create {name}: {path}")

    algorithm = load_algorithm_config(config.gril_config)
    dataset_manifest = (
        config.dataset if config.dataset.is_file() else config.dataset / "dataset.yaml"
    )
    manifest = {
        "engine": "gril_native_full_frontend",
        "ros_runtime_required": False,
        "status": "completed",
        "input": {
            "dataset": str(dataset_manifest.resolve()),
            "dataset_sha256": file_sha256(dataset_manifest),
            "native_boundary": str(input_path.resolve()),
            "native_boundary_sha256": file_sha256(input_path),
            "native_boundary_schema": "GRIL_NATIVE_DATASET 1",
            "scan_count": config.scan_count,
        },
        "algorithm_config": {
            "path": str(config.gril_config.resolve()),
            "digest": config_digest(algorithm),
            "native_path": str(native_config_path.resolve()),
            "native_sha256": file_sha256(native_config_path),
            "batch_native_path": str(
                (config.output_dir / "gril_native.conf").resolve()
            ),
            "batch_native_sha256": file_sha256(config.output_dir / "gril_native.conf"),
        },
        "runtime_semantics": {
            "gap_policy": config.gap_policy,
            "forward_gap_s": config.forward_gap_s,
            "configured_time_lag_s": config.configured_time_lag_s,
            "quality_gating_applied": False,
            "batch_handoff": "live_trace_to_clean_native_process",
        },
        "executable": {
            "path": str(config.executable.resolve()),
            "sha256": file_sha256(config.executable),
            "batch_path": str(config.resolved_batch_executable().resolve()),
            "batch_sha256": file_sha256(config.resolved_batch_executable()),
        },
        "completed_stages": [
            "velodyne_preprocessing",
            "event_fifo_synchronization",
            "constant_velocity_propagation",
            "patchworkpp_ground_segmentation",
            "fusion_ahrs",
            "ikd_tree_lidar_odometry",
            "lidar_only_ekf",
            "ground_constraint_alignment",
            "data_sufficiency_assess",
            "live_li_calibration",
        ],
        "artifacts": {
            "result": {
                "path": str(result_path.resolve()),
                "sha256": file_sha256(result_path),
            },
            "full_frontend_trace": {
                "path": str(trace_path.resolve()),
                "sha256": file_sha256(trace_path),
                "schema": "GRIL_FULL_FRONTEND_TRACE 1",
            },
            "batch_trace": {
                "path": str(batch_trace_path.resolve()),
                "sha256": file_sha256(batch_trace_path),
                "schema": "GRIL_BATCH_TRACE 1",
            },
        },
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    customer_summary_path = write_customer_summary(
        config.output_dir / "customer_summary.yaml",
        result_path,
        dataset,
        developer_diagnostics={
            "manifest": str(manifest_path.resolve()),
            "full_frontend_trace": str(trace_path.resolve()),
            "batch_trace": str(batch_trace_path.resolve()),
        },
    )
    manifest["artifacts"]["customer_summary"] = {
        "path": str(customer_summary_path.resolve()),
        "sha256": file_sha256(customer_summary_path),
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest_path
