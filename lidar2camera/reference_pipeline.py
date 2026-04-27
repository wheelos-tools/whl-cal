from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from lidar2camera.io import write_outputs
from lidar2camera.metrics import build_metrics_output, transform_delta_metrics
from lidar2camera.models import (
    ReferenceCalibrationConfig,
    ReferenceCalibrationDataset,
    ReferencePoseObservation,
)


def default_reference_config_payload() -> dict[str, Any]:
    return {
        "camera": {
            "intrinsics": [
                [1000.0, 0.0, 960.0],
                [0.0, 1000.0, 540.0],
                [0.0, 0.0, 1.0],
            ],
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "checkerboard": {
            "pattern_size": [8, 6],
            "square_size": 0.05,
        },
        "point_cloud": {
            "plane_dist_thresh": 0.02,
            "min_plane_points": 500,
        },
        "frames": {
            "parent": "camera",
            "child": "lidar",
        },
        "data_directory": "calibration_data",
        "optimization": {
            "min_poses": 5,
            "loss": "huber",
            "f_scale": 1.0,
            "max_nfev": 200,
        },
        "metrics": {
            "warning_final_rms_px": 1.0,
            "warning_pose_rms_p95_px": 1.5,
            "warning_holdout_rms_px": 1.5,
            "warning_repeatability_translation_m": 0.05,
            "warning_repeatability_rotation_deg": 1.0,
            "min_leave_one_out_pose_count": 5,
        },
        "output": {"directory": "calibration_output"},
    }


def _load_config(config_path: str) -> tuple[dict[str, Any], ReferenceCalibrationConfig]:
    with open(config_path, "r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

    checkerboard = payload.get("checkerboard", {}) or {}
    point_cloud = payload.get("point_cloud", {}) or {}
    optimization = payload.get("optimization", {}) or {}
    metrics = payload.get("metrics", {}) or {}
    frames = payload.get("frames", {}) or {}
    config = ReferenceCalibrationConfig(
        data_directory=str(payload.get("data_directory", "calibration_data")),
        parent_frame=str(frames.get("parent", payload.get("parent_frame", "camera"))),
        child_frame=str(frames.get("child", payload.get("child_frame", "lidar"))),
        board_pattern_size=tuple(checkerboard.get("pattern_size", [8, 6])),
        board_square_size_m=float(checkerboard.get("square_size", 0.05)),
        plane_distance_threshold_m=float(point_cloud.get("plane_dist_thresh", 0.02)),
        min_plane_points=int(point_cloud.get("min_plane_points", 500)),
        min_pose_count=int(optimization.get("min_poses", 5)),
        optimization_loss=str(optimization.get("loss", "huber")),
        optimization_f_scale=float(optimization.get("f_scale", 1.0)),
        optimization_max_nfev=int(optimization.get("max_nfev", 200)),
        metrics_warning_final_rms_px=float(metrics.get("warning_final_rms_px", 1.0)),
        metrics_warning_pose_rms_p95_px=float(
            metrics.get("warning_pose_rms_p95_px", 1.5)
        ),
        metrics_warning_holdout_rms_px=float(
            metrics.get("warning_holdout_rms_px", 1.5)
        ),
        metrics_warning_repeatability_translation_m=float(
            metrics.get("warning_repeatability_translation_m", 0.05)
        ),
        metrics_warning_repeatability_rotation_deg=float(
            metrics.get("warning_repeatability_rotation_deg", 1.0)
        ),
        metrics_min_leave_one_out_pose_count=int(
            metrics.get("min_leave_one_out_pose_count", 5)
        ),
    )
    return payload, config


def _build_board_template(
    pattern_size: tuple[int, int], square_size: float
) -> np.ndarray:
    object_points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    object_points[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(
        -1, 2
    )
    object_points *= float(square_size)
    return object_points


def _pair_data_files(data_directory: Path) -> tuple[list[tuple[str, Path, Path]], dict]:
    image_candidates: dict[str, Path] = {}
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        for path in sorted(data_directory.glob(pattern)):
            image_candidates[path.stem] = path
    pcd_candidates = {path.stem: path for path in sorted(data_directory.glob("*.pcd"))}
    common_stems = sorted(set(image_candidates) & set(pcd_candidates))
    pairs = [
        (stem, image_candidates[stem], pcd_candidates[stem]) for stem in common_stems
    ]
    return pairs, {
        "image_file_count": len(image_candidates),
        "pcd_file_count": len(pcd_candidates),
        "paired_count": len(pairs),
        "missing_image_stems": sorted(set(pcd_candidates) - set(image_candidates)),
        "missing_pcd_stems": sorted(set(image_candidates) - set(pcd_candidates)),
    }


def _find_image_corners(
    image: np.ndarray, pattern_size: tuple[int, int]
) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK,
    )
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners.reshape(-1, 2)


def _extract_lidar_board_points(
    pcd: o3d.geometry.PointCloud,
    board_template: np.ndarray,
    config: ReferenceCalibrationConfig,
) -> tuple[np.ndarray | None, dict]:
    diagnostics: dict[str, Any] = {
        "plane_inlier_count": 0,
        "plane_inlier_ratio": 0.0,
        "plane_residual_rmse_m": None,
        "orientation_source": "gravity_projected_vertical",
    }
    try:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=config.plane_distance_threshold_m,
            ransac_n=3,
            num_iterations=1000,
        )
    except RuntimeError:
        diagnostics["skip_reason"] = "plane_segmentation_failed"
        return None, diagnostics
    if len(inliers) < int(config.min_plane_points):
        diagnostics["skip_reason"] = "insufficient_plane_points"
        diagnostics["plane_inlier_count"] = int(len(inliers))
        return None, diagnostics

    plane_points = np.asarray(pcd.select_by_index(inliers).points, dtype=float)
    centroid = np.mean(plane_points, axis=0)
    normal = np.asarray(plane_model[:3], dtype=float)
    normal /= max(np.linalg.norm(normal), 1e-12)
    if float(np.dot(normal, centroid)) > 0.0:
        normal = -normal

    global_up = np.array([0.0, 0.0, 1.0], dtype=float)
    y_axis = global_up - float(np.dot(global_up, normal)) * normal
    if np.linalg.norm(y_axis) < 1e-6:
        global_up = np.array([0.0, 1.0, 0.0], dtype=float)
        y_axis = global_up - float(np.dot(global_up, normal)) * normal
    y_axis /= max(np.linalg.norm(y_axis), 1e-12)
    x_axis = np.cross(y_axis, normal)
    x_axis /= max(np.linalg.norm(x_axis), 1e-12)
    y_axis = np.cross(normal, x_axis)
    y_axis /= max(np.linalg.norm(y_axis), 1e-12)

    residuals = np.abs((plane_points - centroid) @ normal)
    projected_x = (plane_points - centroid) @ x_axis
    projected_y = (plane_points - centroid) @ y_axis
    diagnostics.update(
        {
            "plane_inlier_count": int(len(inliers)),
            "plane_inlier_ratio": float(len(inliers) / max(len(pcd.points), 1)),
            "plane_residual_rmse_m": float(np.sqrt(np.mean(residuals**2))),
            "board_extent_xy_m": {
                "x": float(np.ptp(projected_x)),
                "y": float(np.ptp(projected_y)),
            },
        }
    )

    centered_template = board_template.astype(float) - np.mean(board_template, axis=0)
    object_points = (
        np.outer(centered_template[:, 0], x_axis)
        + np.outer(centered_template[:, 1], y_axis)
        + centroid
    )
    return object_points, diagnostics


def _load_reference_dataset(
    config_path: str,
) -> tuple[ReferenceCalibrationDataset, ReferenceCalibrationConfig, dict, dict]:
    config_payload, config = _load_config(config_path)
    camera_payload = config_payload.get("camera", {}) or {}
    camera_matrix = np.asarray(camera_payload.get("intrinsics"), dtype=float)
    camera_distortion = np.asarray(
        camera_payload.get("distortion", [0.0, 0.0, 0.0, 0.0, 0.0]), dtype=float
    )
    board_template = _build_board_template(
        config.board_pattern_size, config.board_square_size_m
    )
    data_directory = Path(config.data_directory).expanduser().resolve()
    data_directory.mkdir(parents=True, exist_ok=True)
    pairs, pairing_summary = _pair_data_files(data_directory)

    extraction_entries = []
    observations = []
    for pose_id, image_path, pcd_path in pairs:
        entry: dict[str, Any] = {
            "pose_id": pose_id,
            "image_path": str(image_path),
            "pcd_path": str(pcd_path),
        }
        image = cv2.imread(str(image_path))
        if image is None:
            entry["status"] = "skipped"
            entry["skip_reason"] = "image_load_failed"
            extraction_entries.append(entry)
            continue
        image_points = _find_image_corners(image, config.board_pattern_size)
        if image_points is None:
            entry["status"] = "skipped"
            entry["skip_reason"] = "image_corners_not_found"
            extraction_entries.append(entry)
            continue

        pcd = o3d.io.read_point_cloud(str(pcd_path))
        if not pcd.has_points():
            entry["status"] = "skipped"
            entry["skip_reason"] = "empty_point_cloud"
            extraction_entries.append(entry)
            continue
        object_points, board_diagnostics = _extract_lidar_board_points(
            pcd, board_template, config
        )
        entry["board_diagnostics"] = board_diagnostics
        if object_points is None:
            entry["status"] = "skipped"
            entry["skip_reason"] = board_diagnostics.get(
                "skip_reason", "board_geometry_unavailable"
            )
            extraction_entries.append(entry)
            continue

        entry["status"] = "accepted"
        entry["image_corner_count"] = int(len(image_points))
        entry["object_point_count"] = int(len(object_points))
        observations.append(
            ReferencePoseObservation(
                pose_id=pose_id,
                image_path=str(image_path),
                pcd_path=str(pcd_path),
                image_size_wh=(int(image.shape[1]), int(image.shape[0])),
                image_points=np.asarray(image_points, dtype=float),
                object_points=np.asarray(object_points, dtype=float),
                metadata={"board_diagnostics": board_diagnostics},
            )
        )
        extraction_entries.append(entry)

    initial_transform = None
    if "initial_transform" in config_payload:
        initial_transform = np.asarray(config_payload["initial_transform"], dtype=float)
    dataset = ReferenceCalibrationDataset(
        parent_frame=config.parent_frame,
        child_frame=config.child_frame,
        camera_matrix=camera_matrix,
        camera_distortion=camera_distortion,
        observations=observations,
        initial_transform=initial_transform,
        metadata={
            "config_path": str(Path(config_path).expanduser().resolve()),
            "data_directory": str(data_directory),
            "extractor": "checkerboard_reference_based",
            "orientation_assumption": "gravity_projected_vertical",
        },
    )
    extraction_report = {
        "pairing_summary": pairing_summary,
        "accepted_pose_count": len(observations),
        "rejected_pose_count": int(
            sum(1 for item in extraction_entries if item["status"] != "accepted")
        ),
        "entries": extraction_entries,
    }
    return dataset, config, config_payload, extraction_report


def _transform_from_params(params: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = R.from_rotvec(np.asarray(params[:3], dtype=float)).as_matrix()
    transform[:3, 3] = np.asarray(params[3:], dtype=float)
    return transform


def _params_from_transform(transform: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [R.from_matrix(transform[:3, :3]).as_rotvec(), np.asarray(transform[:3, 3])]
    )


def _project_points(
    transform: np.ndarray,
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    camera_distortion: np.ndarray,
) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    tvec = transform[:3, 3].reshape(3, 1)
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, camera_distortion
    )
    return projected.reshape(-1, 2)


def _per_pose_rms(
    transform: np.ndarray,
    dataset: ReferenceCalibrationDataset,
) -> list[dict[str, Any]]:
    rows = []
    for observation in dataset.observations:
        projected = _project_points(
            transform,
            observation.object_points,
            dataset.camera_matrix,
            dataset.camera_distortion,
        )
        residuals = projected - observation.image_points
        point_errors = np.linalg.norm(residuals, axis=1)
        rows.append(
            {
                "pose_id": observation.pose_id,
                "rms_px": float(np.sqrt(np.mean(np.sum(residuals**2, axis=1)))),
                "p95_px": float(np.percentile(point_errors, 95)),
                "max_px": float(np.max(point_errors)),
                "image_path": observation.image_path,
                "pcd_path": observation.pcd_path,
            }
        )
    return rows


def _objective_function(
    params: np.ndarray, dataset: ReferenceCalibrationDataset
) -> np.ndarray:
    transform = _transform_from_params(params)
    residuals = []
    for observation in dataset.observations:
        projected = _project_points(
            transform,
            observation.object_points,
            dataset.camera_matrix,
            dataset.camera_distortion,
        )
        residuals.append((projected - observation.image_points).reshape(-1))
    return np.concatenate(residuals) if residuals else np.zeros(0, dtype=float)


def _compute_rms(params: np.ndarray, dataset: ReferenceCalibrationDataset) -> float:
    residuals = _objective_function(params, dataset)
    if residuals.size == 0:
        return float("inf")
    return float(np.sqrt(np.mean(residuals**2)))


def _select_initial_transform(
    dataset: ReferenceCalibrationDataset,
) -> tuple[np.ndarray, dict]:
    if dataset.initial_transform is not None:
        initial_transform = np.asarray(dataset.initial_transform, dtype=float)
        return initial_transform, {
            "source": "config_initial_transform",
            "pose_id": None,
            "rms_px": _compute_rms(_params_from_transform(initial_transform), dataset),
        }

    best_transform = None
    best_rms = float("inf")
    best_pose_id = None
    for observation in dataset.observations:
        success, rvec, tvec = cv2.solvePnP(
            observation.object_points,
            observation.image_points,
            dataset.camera_matrix,
            dataset.camera_distortion,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if not success:
            continue
        candidate_transform = np.eye(4, dtype=float)
        candidate_transform[:3, :3], _ = cv2.Rodrigues(rvec)
        candidate_transform[:3, 3] = tvec.reshape(3)
        candidate_rms = _compute_rms(
            _params_from_transform(candidate_transform), dataset
        )
        if candidate_rms < best_rms:
            best_rms = candidate_rms
            best_transform = candidate_transform
            best_pose_id = observation.pose_id
    if best_transform is None:
        raise RuntimeError("Could not find a valid initial transform from any pose.")
    return best_transform, {
        "source": "best_single_pose_ippe",
        "pose_id": best_pose_id,
        "rms_px": float(best_rms),
    }


def _optimize_dataset(
    dataset: ReferenceCalibrationDataset,
    config: ReferenceCalibrationConfig,
    initial_transform: np.ndarray,
) -> tuple[np.ndarray, dict]:
    initial_params = _params_from_transform(initial_transform)
    result = least_squares(
        _objective_function,
        initial_params,
        args=(dataset,),
        method="trf",
        loss=config.optimization_loss,
        f_scale=config.optimization_f_scale,
        jac="2-point",
        max_nfev=config.optimization_max_nfev,
    )
    final_transform = _transform_from_params(result.x)
    return final_transform, {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
        "jacobian_rank": int(np.linalg.matrix_rank(result.jac)),
        "jacobian_shape": [int(result.jac.shape[0]), int(result.jac.shape[1])],
        "optimization_loss": config.optimization_loss,
        "optimization_f_scale": float(config.optimization_f_scale),
        "result": result,
    }


def _build_leave_one_out_repeatability(
    dataset: ReferenceCalibrationDataset,
    config: ReferenceCalibrationConfig,
    primary_transform: np.ndarray,
) -> dict | None:
    if len(dataset.observations) < int(config.metrics_min_leave_one_out_pose_count):
        return None

    trials = []
    transforms = []
    for holdout_index, holdout_observation in enumerate(dataset.observations):
        subset_observations = [
            observation
            for index, observation in enumerate(dataset.observations)
            if index != holdout_index
        ]
        if len(subset_observations) < max(3, int(config.min_pose_count) - 1):
            continue
        subset_dataset = ReferenceCalibrationDataset(
            parent_frame=dataset.parent_frame,
            child_frame=dataset.child_frame,
            camera_matrix=np.asarray(dataset.camera_matrix, dtype=float),
            camera_distortion=np.asarray(dataset.camera_distortion, dtype=float),
            observations=subset_observations,
            initial_transform=primary_transform,
            metadata=dict(dataset.metadata),
        )
        subset_transform, subset_optimization = _optimize_dataset(
            subset_dataset, config, primary_transform
        )
        transforms.append(subset_transform)
        holdout_projected = _project_points(
            subset_transform,
            holdout_observation.object_points,
            dataset.camera_matrix,
            dataset.camera_distortion,
        )
        holdout_residuals = holdout_projected - holdout_observation.image_points
        holdout_rms = float(np.sqrt(np.mean(holdout_residuals**2)))
        trials.append(
            {
                "holdout_pose_id": holdout_observation.pose_id,
                "train_pose_count": len(subset_observations),
                "holdout_rms_px": holdout_rms,
                "delta_to_primary": transform_delta_metrics(
                    primary_transform, subset_transform
                ),
                "optimization_success": bool(subset_optimization["success"]),
            }
        )

    if not trials:
        return None

    holdout_rms_values = [float(trial["holdout_rms_px"]) for trial in trials]
    translation_threshold = float(config.metrics_warning_repeatability_translation_m)
    rotation_threshold = float(config.metrics_warning_repeatability_rotation_deg)
    distinct_solution_count = 1
    if transforms:
        distinct_solution_count = 0
        representatives: list[np.ndarray] = []
        for transform in transforms:
            matched = False
            for representative in representatives:
                delta = transform_delta_metrics(representative, transform)
                if (
                    delta["translation_norm_m"] <= translation_threshold
                    and delta["rotation_deg"] <= rotation_threshold
                ):
                    matched = True
                    break
            if not matched:
                representatives.append(transform)
        distinct_solution_count = len(representatives)

    translation_deltas = [
        float(trial["delta_to_primary"]["translation_norm_m"]) for trial in trials
    ]
    rotation_deltas = [
        float(trial["delta_to_primary"]["rotation_deg"]) for trial in trials
    ]
    if (
        max(holdout_rms_values) <= float(config.metrics_warning_holdout_rms_px)
        and distinct_solution_count <= 1
    ):
        status = "pass"
        primary_cause = "stable_leave_one_out"
        recommendations: list[str] = []
    elif distinct_solution_count > 1:
        status = "warning"
        primary_cause = "leave_one_out_solution_instability"
        recommendations = [
            "Leaving one pose out changes the final solution family; do not treat one optimization result as a production release yet.",
            "Collect wider board pose diversity or improve LiDAR-side board geometry extraction before promoting this run.",
        ]
    else:
        status = "warning"
        primary_cause = "leave_one_out_holdout_gap"
        recommendations = [
            "At least one held-out pose reprojection is materially worse than the release threshold; require better pose diversity before accepting the result.",
            "Prefer board captures that cover left/right/up/down image regions and multiple depth / tilt states.",
        ]

    rotvecs = [R.from_matrix(transform[:3, :3]).as_rotvec() for transform in transforms]
    translations = [transform[:3, 3] for transform in transforms]
    uncertainty_summary = None
    if transforms:
        translation_array = np.asarray(translations, dtype=float)
        rotvec_array = np.asarray(rotvecs, dtype=float)
        uncertainty_summary = {
            "source": "leave_one_out_pose_repeatability",
            "trial_count": len(transforms),
            "translation_std_m": {
                "x": float(np.std(translation_array[:, 0])),
                "y": float(np.std(translation_array[:, 1])),
                "z": float(np.std(translation_array[:, 2])),
            },
            "rotation_vector_std_rad": {
                "x": float(np.std(rotvec_array[:, 0])),
                "y": float(np.std(rotvec_array[:, 1])),
                "z": float(np.std(rotvec_array[:, 2])),
            },
        }

    return {
        "status": status,
        "primary_cause": primary_cause,
        "trial_count": len(trials),
        "distinct_solution_count": distinct_solution_count,
        "holdout_rms_summary": {
            "mean": float(np.mean(holdout_rms_values)),
            "p95": float(
                np.percentile(np.asarray(holdout_rms_values, dtype=float), 95)
            ),
            "max": float(np.max(holdout_rms_values)),
        },
        "delta_to_primary_summary": {
            "translation_norm_m": {
                "mean": float(np.mean(translation_deltas)),
                "p95": float(
                    np.percentile(np.asarray(translation_deltas, dtype=float), 95)
                ),
                "max": float(np.max(translation_deltas)),
            },
            "rotation_deg": {
                "mean": float(np.mean(rotation_deltas)),
                "p95": float(
                    np.percentile(np.asarray(rotation_deltas, dtype=float), 95)
                ),
                "max": float(np.max(rotation_deltas)),
            },
        },
        "uncertainty_summary": uncertainty_summary,
        "recommendations": recommendations,
        "trials": trials,
    }


def _build_overlay_artifact(
    dataset: ReferenceCalibrationDataset,
    transform: np.ndarray,
    output_dir: Path,
) -> str | None:
    if not dataset.observations:
        return None
    representative = dataset.observations[0]
    image = cv2.imread(representative.image_path)
    if image is None:
        return None
    pcd = o3d.io.read_point_cloud(representative.pcd_path)
    if not pcd.has_points():
        return None
    points = np.asarray(pcd.points, dtype=float)
    camera_points = (transform[:3, :3] @ points.T + transform[:3, 3:]).T
    projected, _ = cv2.projectPoints(
        camera_points,
        np.zeros(3),
        np.zeros(3),
        dataset.camera_matrix,
        dataset.camera_distortion,
    )
    projected = projected.reshape(-1, 2)
    valid = (
        (camera_points[:, 2] > 1e-3)
        & (projected[:, 0] >= 0)
        & (projected[:, 0] < image.shape[1])
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < image.shape[0])
    )
    if not np.any(valid):
        return None
    overlay = image.copy()
    valid_points = projected[valid].astype(np.int32)
    depths = camera_points[valid, 2]
    colors = cv2.applyColorMap(
        np.clip((depths / max(np.percentile(depths, 95), 1e-6)) * 255.0, 0, 255).astype(
            np.uint8
        ),
        cv2.COLORMAP_JET,
    )
    for index, (u, v) in enumerate(valid_points):
        cv2.circle(
            overlay,
            (int(u), int(v)),
            1,
            tuple(int(channel) for channel in colors[index, 0]),
            thickness=-1,
        )
    blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    artifact_path = output_dir / "diagnostics" / "reference_overlay.png"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(artifact_path), blended)
    return str(artifact_path)


def run_reference_calibration_from_config(
    config_path: str,
    *,
    output_dir_override: str | None = None,
) -> dict:
    dataset, config, raw_config, extraction_report = _load_reference_dataset(
        config_path
    )
    if len(dataset.observations) < int(config.min_pose_count):
        raise RuntimeError(
            f"Insufficient valid poses ({len(dataset.observations)}). Need at least {config.min_pose_count}."
        )

    initial_transform, initial_guess_summary = _select_initial_transform(dataset)
    initial_rms_px = float(initial_guess_summary["rms_px"])
    final_transform, optimization = _optimize_dataset(
        dataset, config, initial_transform
    )
    optimization_result = optimization.pop("result")
    final_rms_px = _compute_rms(_params_from_transform(final_transform), dataset)
    per_pose_rms_px = _per_pose_rms(final_transform, dataset)
    leave_one_out = _build_leave_one_out_repeatability(dataset, config, final_transform)
    uncertainty_summary = None
    if leave_one_out is not None:
        uncertainty_summary = leave_one_out.get("uncertainty_summary")

    try:
        covariance = np.linalg.inv(optimization_result.jac.T @ optimization_result.jac)
        parameter_uncertainty = (
            2.0 * final_rms_px * np.sqrt(np.maximum(np.diag(covariance), 0.0))
        )
        jacobian_uncertainty = {
            "rotation_vector_rad": [
                float(value) for value in parameter_uncertainty[:3]
            ],
            "translation_m": [float(value) for value in parameter_uncertainty[3:]],
        }
    except np.linalg.LinAlgError:
        jacobian_uncertainty = None

    optimization_report = {
        "initial_guess": initial_guess_summary,
        "optimization": optimization,
        "jacobian_uncertainty": jacobian_uncertainty,
        "raw_config": copy.deepcopy(raw_config),
    }
    output_dir = Path(
        output_dir_override
        or str(
            (raw_config.get("output", {}) or {}).get("directory", "calibration_output")
        )
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_artifact = _build_overlay_artifact(dataset, final_transform, output_dir)
    metrics_output = build_metrics_output(
        dataset,
        config,
        initial_transform=initial_transform,
        final_transform=final_transform,
        initial_rms_px=initial_rms_px,
        final_rms_px=final_rms_px,
        per_pose_rms_px=per_pose_rms_px,
        leave_one_out=leave_one_out,
        uncertainty_summary=uncertainty_summary,
        extraction_report=extraction_report,
        optimization_report=optimization_report,
        output_dir=str(output_dir),
    )
    evaluation_report = {
        "per_pose_reprojection": per_pose_rms_px,
        "leave_one_out_repeatability": leave_one_out,
        "uncertainty_summary": uncertainty_summary,
        "delta_to_initial": transform_delta_metrics(initial_transform, final_transform),
    }
    manifest = write_outputs(
        output_dir=output_dir,
        dataset=dataset,
        initial_transform=initial_transform,
        final_transform=final_transform,
        metrics_output=metrics_output,
        extraction_report=extraction_report,
        optimization_report=optimization_report,
        evaluation_report=evaluation_report,
        overlay_artifact=overlay_artifact,
    )
    return {
        "dataset": dataset,
        "config": config,
        "initial_transform": initial_transform,
        "final_transform": final_transform,
        "metrics": metrics_output,
        "extraction_report": extraction_report,
        "optimization_report": optimization_report,
        "evaluation_report": evaluation_report,
        "manifest": manifest,
    }
