#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

SKILL_DIR = Path(__file__).resolve().parents[1]
PINNED_REVISION = "c09b01a05ec83bc0a361941acf897109aaecf0a6"
DEFAULT_REPOSITORY = "https://github.com/Taeyoung96/GRIL-Calib.git"
DEFAULT_IMAGE = "gril-calib-validation:2026-08"


def run(command: list[str], **kwargs) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def records(arguments) -> list[str]:
    if not arguments.record_file:
        raise ValueError("At least one --record-file is required")
    return [str(Path(path).resolve()) for path in arguments.record_file]


def repeated(flag: str, values: list[str]) -> list[str]:
    result = []
    for value in values:
        result.extend([flag, value])
    return result


def paths(workspace: Path) -> tuple[Path, Path]:
    return workspace / "GRIL-Calib", workspace / "catkin_ws"


def setup(arguments) -> None:
    workspace = arguments.workspace.resolve()
    repository, catkin_ws = paths(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    if not repository.exists():
        run(["git", "clone", arguments.repository, str(repository)])
    run(["git", "-C", str(repository), "fetch", "--all", "--tags"])
    run(["git", "-C", str(repository), "checkout", "--detach", PINNED_REVISION])

    patch = SKILL_DIR / "patches/gril-validation.patch"
    check = subprocess.run(
        ["git", "-C", str(repository), "apply", "--check", str(patch)]
    )
    if check.returncode == 0:
        run(["git", "-C", str(repository), "apply", str(patch)])
    else:
        run(
            [
                "git",
                "-C",
                str(repository),
                "apply",
                "--reverse",
                "--check",
                str(patch),
            ]
        )
    shutil.copy2(
        SKILL_DIR / "resources/vanjeelidar16.yaml",
        repository / "config/velodyne16.yaml",
    )
    (repository / "Log").mkdir(exist_ok=True)
    (repository / "result").mkdir(exist_ok=True)
    (catkin_ws / "src").mkdir(parents=True, exist_ok=True)

    run(
        [
            "docker",
            "build",
            "-t",
            arguments.image,
            "-f",
            str(SKILL_DIR / "resources/Dockerfile"),
            str(SKILL_DIR / "resources"),
        ]
    )
    run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{catkin_ws}:/root/catkin_ws",
            "-v",
            f"{repository}:/root/catkin_ws/src/gril_calib",
            arguments.image,
            "bash",
            "-lc",
            (
                "source /opt/ros/noetic/setup.bash && "
                "source /root/livox_ws/devel/setup.bash && "
                "cd /root/catkin_ws && "
                "catkin_make -j4 --quiet "
                "-DGRIL_DETERMINISTIC_FRONTEND=ON"
            ),
        ]
    )


def prepare(arguments) -> None:
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_records = records(arguments)
    bag = output / "gril_input.bag"
    common = repeated("--record-file", source_records)
    run(
        [
            sys.executable,
            str(SKILL_DIR / "scripts/export_apollo_rosbag.py"),
            *common,
            "--output",
            str(bag),
            "--lidar-topic",
            arguments.lidar_topic,
            "--imu-topic",
            arguments.imu_topic,
            "--scan-lines",
            str(arguments.scan_lines),
        ]
    )
    run(
        [
            sys.executable,
            str(SKILL_DIR / "scripts/audit_conversion.py"),
            *common,
            "--bag",
            str(bag),
            "--output",
            str(output / "input_contract.yaml"),
            "--lidar-topic",
            arguments.lidar_topic,
            "--imu-topic",
            arguments.imu_topic,
            "--tf-parent",
            arguments.tf_parent,
            "--tf-child",
            arguments.tf_child,
        ]
    )
    contract = yaml.safe_load((output / "input_contract.yaml").read_text())
    if contract["verdict"] != "accepted":
        raise RuntimeError("Input contract rejected; GRIL run was not started")


def calibrate(arguments) -> None:
    output = arguments.output_dir.resolve()
    bag = output / "gril_input.bag"
    if not bag.exists():
        raise FileNotFoundError(f"Run prepare first: {bag}")
    repository, catkin_ws = paths(arguments.workspace.resolve())
    run(
        [
            "docker",
            "run",
            "--rm",
            "-e",
            "QT_QPA_PLATFORM=offscreen",
            "-v",
            f"{catkin_ws}:/root/catkin_ws",
            "-v",
            f"{repository}:/root/catkin_ws/src/gril_calib",
            "-v",
            f"{output}:/data",
            "-v",
            (
                f"{SKILL_DIR / 'resources/run_gril_container.sh'}:"
                "/run_gril_container.sh:ro"
            ),
            arguments.image,
            "bash",
            "/run_gril_container.sh",
            "/data/gril_input.bag",
            str(arguments.runs),
        ]
    )


def parse_result(path: Path) -> tuple[list[float], list[float], float]:
    text = path.read_text()

    def values(label: str) -> list[float]:
        match = re.search(rf"{label}[^=]*=\s*([^\n]+)", text)
        if match is None:
            raise RuntimeError(f"Missing {label} in {path}")
        return [float(value) for value in match.group(1).split()]

    return (
        values("Rotation LiDAR to IMU"),
        values("Translation LiDAR to IMU"),
        values("Time Lag IMU to LiDAR")[0],
    )


def diagnose(arguments) -> None:
    output = arguments.output_dir.resolve()
    source_records = records(arguments)
    run_dir = output / "run_1"
    rotation, _, time_offset = parse_result(run_dir / "GRIL_Calib_result.txt")
    diagnostic_dir = output / "diagnostics"
    diagnostic_dir.mkdir(exist_ok=True)
    run_results = []
    for result_path in sorted(output.glob("run_*/GRIL_Calib_result.txt")):
        run_rotation, run_translation, run_time = parse_result(result_path)
        run_results.append(
            {
                "path": str(result_path),
                "rotation": run_rotation,
                "translation": run_translation,
                "time": run_time,
            }
        )
    rotation_spread = 0.0
    translation_spread = 0.0
    time_spread = 0.0
    for first in range(len(run_results)):
        for second in range(first + 1, len(run_results)):
            first_rotation = Rotation.from_euler(
                "xyz", run_results[first]["rotation"], degrees=True
            )
            second_rotation = Rotation.from_euler(
                "xyz", run_results[second]["rotation"], degrees=True
            )
            rotation_spread = max(
                rotation_spread,
                float(np.degrees((first_rotation.inv() * second_rotation).magnitude())),
            )
            translation_spread = max(
                translation_spread,
                float(
                    np.linalg.norm(
                        np.asarray(run_results[first]["translation"])
                        - np.asarray(run_results[second]["translation"])
                    )
                ),
            )
            time_spread = max(
                time_spread,
                abs(run_results[first]["time"] - run_results[second]["time"]),
            )
    repeatability = {
        "verdict": (
            "accepted"
            if rotation_spread <= 0.2
            and translation_spread <= 0.03
            and time_spread <= 0.001
            else "rejected"
        ),
        "run_count": len(run_results),
        "maximum_rotation_difference_deg": rotation_spread,
        "maximum_translation_difference_m": translation_spread,
        "maximum_time_difference_s": time_spread,
        "gates": {
            "rotation_deg": 0.2,
            "translation_m": 0.03,
            "time_s": 0.001,
        },
        "runs": run_results,
    }
    with (diagnostic_dir / "repeatability.yaml").open("w") as stream:
        yaml.safe_dump(repeatability, stream, sort_keys=False)
    common = repeated("--record-file", source_records)
    run(
        [
            sys.executable,
            str(SKILL_DIR / "scripts/evaluate_trajectories.py"),
            *common,
            "--trajectory",
            str(run_dir / "lidar_trajectory.txt"),
            "--output-dir",
            str(diagnostic_dir),
            "--pose-topic",
            arguments.pose_topic,
        ]
    )
    environment = os.environ.copy()
    environment.update(
        {
            "GRIL_ALIGNED_STATES": str(run_dir / "log/aligned_accel_states.txt"),
            "GRIL_VALIDATION_OUTPUT": str(diagnostic_dir),
            "GRIL_ROTATION_XYZ_DEG": ",".join(map(str, rotation)),
            "GRIL_TIME_OFFSET_S": str(time_offset),
        }
    )
    run(
        [sys.executable, str(SKILL_DIR / "scripts/validate_dynamics.py")],
        env=environment,
    )
    run(
        [sys.executable, str(SKILL_DIR / "scripts/yaw_time_grid.py")],
        env=environment,
    )
    if not arguments.skip_submap:
        submap_command = [
            sys.executable,
            str(SKILL_DIR / "scripts/build_imu_submap.py"),
            *common,
            "--lidar-topic",
            arguments.lidar_topic,
            "--pose-topic",
            arguments.pose_topic,
        ]
        candidate_submap = diagnostic_dir / "imu_submap"
        run(
            submap_command
            + [
                "--result",
                str(run_dir / "GRIL_Calib_result.txt"),
                "--output-dir",
                str(candidate_submap),
            ]
        )
        input_contract = yaml.safe_load((output / "input_contract.yaml").read_text())
        static_transform = np.asarray(
            input_contract["static_transform_lidar_to_imu"]["matrix"]
        )
        static_rotation = Rotation.from_matrix(static_transform[:3, :3]).as_euler(
            "xyz", degrees=True
        )
        static_translation = static_transform[:3, 3]
        static_result = output / "static_tf_baseline_result.txt"
        static_result.write_text(
            "LiDAR-IMU calibration result:\n"
            "Rotation LiDAR to IMU (degree)     = "
            + " ".join(map(str, static_rotation))
            + "\nTranslation LiDAR to IMU (meter)   = "
            + " ".join(map(str, static_translation))
            + "\nTime Lag IMU to LiDAR (second)     = 0.0\n"
        )
        baseline_submap = diagnostic_dir / "imu_submap_static_tf"
        run(
            submap_command
            + [
                "--result",
                str(static_result),
                "--output-dir",
                str(baseline_submap),
            ]
        )
        candidate_thickness = yaml.safe_load(
            (candidate_submap / "submap_metrics.yaml").read_text()
        )["local_planar_thickness_m"]
        baseline_thickness = yaml.safe_load(
            (baseline_submap / "submap_metrics.yaml").read_text()
        )["local_planar_thickness_m"]
        submap_comparison = {
            "candidate": candidate_thickness,
            "static_tf_baseline": baseline_thickness,
            "thickness_improvement_percent": {
                quantile: 100.0
                * (baseline_thickness[quantile] - candidate_thickness[quantile])
                / baseline_thickness[quantile]
                for quantile in ("p50", "p95", "p99")
            },
            "lower_is_better": True,
            "settings_identical": True,
        }
        with (diagnostic_dir / "submap_comparison.yaml").open("w") as stream:
            yaml.safe_dump(submap_comparison, stream, sort_keys=False)
    dynamics = yaml.safe_load(
        (diagnostic_dir / "gril_first_principles_validation.yaml").read_text()
    )
    yaw_time = yaml.safe_load(
        (diagnostic_dir / "gril_yaw_time_holdout_grid.yaml").read_text()
    )
    metrics = {
        "verdict": (
            "review_only"
            if repeatability["verdict"] == "accepted"
            and dynamics["verdict"] not in {"not_self_consistent", "rejected"}
            and yaw_time["verdict"] != "no_unique_generalizing_solution"
            else "rejected_for_full_extrinsic_acceptance"
        ),
        "source_revision": PINNED_REVISION,
        "candidate": {
            "rotation_lidar_to_imu_deg": rotation,
            "translation_lidar_to_imu_m": parse_result(
                run_dir / "GRIL_Calib_result.txt"
            )[1],
            "time_lag_imu_to_lidar_s": time_offset,
        },
        "input_contract": yaml.safe_load((output / "input_contract.yaml").read_text())[
            "verdict"
        ],
        "repeatability": repeatability["verdict"],
        "dynamics_holdout": dynamics["verdict"],
        "yaw_time_holdout": yaw_time["verdict"],
        "artifacts": {
            "trajectory": "diagnostics/frontend_trajectory.png",
            "repeatability": "diagnostics/repeatability.yaml",
            "dynamics": "diagnostics/gril_first_principles_validation.yaml",
            "yaw_time": "diagnostics/gril_yaw_time_holdout_grid.yaml",
            "submap": (
                None
                if arguments.skip_submap
                else "diagnostics/imu_submap/imu_extrinsic_submap.ply"
            ),
            "submap_comparison": (
                None if arguments.skip_submap else "diagnostics/submap_comparison.yaml"
            ),
        },
    }
    with (output / "metrics.yaml").open("w") as stream:
        yaml.safe_dump(metrics, stream, sort_keys=False)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace", type=Path, default=Path(".cache/gril-validation")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--record-file", action="append")
    parser.add_argument(
        "--lidar-topic",
        default="/apollo/sensor/vanjeelidar/up/PointCloud2",
    )
    parser.add_argument("--imu-topic", default="/apollo/sensor/gnss/imu")
    parser.add_argument("--pose-topic", default="/apollo/sensor/gnss/odometry")
    parser.add_argument("--tf-parent", default="imu")
    parser.add_argument("--tf-child", default="vanjeelidar_up")
    parser.add_argument("--scan-lines", type=int, default=16)
    parser.add_argument("--image", default=DEFAULT_IMAGE)


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="command", required=True)
setup_parser = subparsers.add_parser("setup")
setup_parser.add_argument(
    "--workspace", type=Path, default=Path(".cache/gril-validation")
)
setup_parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
setup_parser.add_argument("--image", default=DEFAULT_IMAGE)
setup_parser.set_defaults(function=setup)

prepare_parser = subparsers.add_parser("prepare")
add_common(prepare_parser)
prepare_parser.set_defaults(function=prepare)

run_parser = subparsers.add_parser("run")
add_common(run_parser)
run_parser.add_argument("--runs", type=int, default=2)
run_parser.set_defaults(function=calibrate)

diagnose_parser = subparsers.add_parser("diagnose")
add_common(diagnose_parser)
diagnose_parser.add_argument("--skip-submap", action="store_true")
diagnose_parser.set_defaults(function=diagnose)

all_parser = subparsers.add_parser("all")
add_common(all_parser)
all_parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
all_parser.add_argument("--runs", type=int, default=2)
all_parser.add_argument("--skip-submap", action="store_true")


def all_steps(arguments) -> None:
    setup(arguments)
    prepare(arguments)
    calibrate(arguments)
    diagnose(arguments)


all_parser.set_defaults(function=all_steps)
arguments = parser.parse_args()
arguments.function(arguments)
