"""Point cloud registration via native PCL Generalized ICP."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from lidar2lidar.lidar2lidar import preprocess_point_cloud

PACKAGE_ROOT = Path(__file__).resolve().parent
PCL_GICP_BINARY = PACKAGE_ROOT / "bin" / "pcl_gicp_align"
PCL_GICP_SOURCE = PACKAGE_ROOT / "native" / "pcl_gicp_align.cpp"


@dataclass
class PclRegistrationResult:
    fitness: float
    inlier_rmse: float
    transformation: np.ndarray


def pcl_gicp_available() -> bool:
    return _ensure_pcl_binary() is not None


def _ensure_pcl_binary() -> Path | None:
    if PCL_GICP_BINARY.exists():
        return PCL_GICP_BINARY
    try:
        PCL_GICP_BINARY.parent.mkdir(parents=True, exist_ok=True)
        compile_cmd = [
            "g++",
            "-O3",
            "-std=c++17",
            str(PCL_GICP_SOURCE),
            "-o",
            str(PCL_GICP_BINARY),
        ]
        pkg_config = subprocess.run(
            [
                "pkg-config",
                "--cflags",
                "--libs",
                "pcl_common",
                "pcl_io",
                "pcl_registration",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        compile_cmd.extend(pkg_config.stdout.strip().split())
        logging.info("Compiling PCL GICP helper: %s", " ".join(compile_cmd))
        subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        logging.warning("PCL GICP binary unavailable: %s", exc)
        return None
    if not PCL_GICP_BINARY.exists():
        return None
    PCL_GICP_BINARY.chmod(0o755)
    return PCL_GICP_BINARY


def _write_matrix(path: Path, matrix: np.ndarray) -> None:
    values = np.asarray(matrix, dtype=float).reshape(4, 4)
    lines = [" ".join(f"{value:.10f}" for value in row) for row in values]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evaluate_registration(
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    transform: np.ndarray,
    max_correspondence_distance: float,
) -> tuple[float, float]:
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source_cloud,
        target_cloud,
        float(max_correspondence_distance),
        np.asarray(transform, dtype=float),
    )
    return float(evaluation.fitness), float(evaluation.inlier_rmse)


def register_generalized_icp(
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    *,
    initial_transform: np.ndarray | None = None,
    preprocessing_params: dict | None = None,
    max_correspondence_distance: float | None = None,
    max_iterations: int = 120,
) -> tuple[np.ndarray | None, PclRegistrationResult | None]:
    """Run PCL Generalized ICP on preprocessed point clouds."""
    binary = _ensure_pcl_binary()
    if binary is None:
        return None, None

    params = dict(preprocessing_params or {})
    params.setdefault("voxel_size", 0.04)
    params.setdefault("nb_neighbors", 20)
    params.setdefault("std_ratio", 2.0)
    params.setdefault("plane_dist_thresh", 0.05)
    params.setdefault("height_range", None)
    params.setdefault("remove_ground", False)
    params.setdefault("remove_walls", False)

    source_preprocessed = preprocess_point_cloud(source_cloud, **params)
    target_preprocessed = preprocess_point_cloud(target_cloud, **params)
    if len(source_preprocessed.points) == 0 or len(target_preprocessed.points) == 0:
        return None, None

    voxel_size = float(params["voxel_size"])
    effective_max_corr = float(
        max_correspondence_distance
        if max_correspondence_distance is not None
        else max(voxel_size * 5, 0.02)
    )
    initial = (
        np.eye(4, dtype=float)
        if initial_transform is None
        else np.asarray(initial_transform, dtype=float).reshape(4, 4)
    )

    with tempfile.TemporaryDirectory(prefix="whl_cal_pcl_gicp_") as temp_dir:
        temp_path = Path(temp_dir)
        source_path = temp_path / "source.pcd"
        target_path = temp_path / "target.pcd"
        initial_path = temp_path / "initial.txt"
        output_path = temp_path / "result.json"
        o3d.io.write_point_cloud(str(source_path), source_preprocessed)
        o3d.io.write_point_cloud(str(target_path), target_preprocessed)
        _write_matrix(initial_path, initial)

        command = [
            str(binary),
            "--source",
            str(source_path),
            "--target",
            str(target_path),
            "--initial",
            str(initial_path),
            "--max-correspondence-distance",
            f"{effective_max_corr:.6f}",
            "--max-iterations",
            str(int(max_iterations)),
            "--output",
            str(output_path),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode not in {0, 1} or not output_path.exists():
            logging.error(
                "PCL GICP failed (code=%s): %s",
                completed.returncode,
                completed.stderr.strip(),
            )
            return None, None

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if not payload.get("success"):
            return None, None

        transform = np.asarray(payload["transform"], dtype=float).reshape(4, 4)
        fitness, inlier_rmse = _evaluate_registration(
            source_preprocessed,
            target_preprocessed,
            transform,
            effective_max_corr,
        )
        result = PclRegistrationResult(
            fitness=fitness,
            inlier_rmse=inlier_rmse,
            transformation=transform,
        )
        return transform, result
