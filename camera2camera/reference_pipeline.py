from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from camera2camera.io import write_failure_outputs, write_outputs
from camera2camera.metrics import (build_metrics_output, float_list_summary,
                                   transform_delta_metrics)
from camera2camera.models import (StereoCalibrationConfig,
                                  StereoCalibrationDataset,
                                  StereoCalibrationObservation)
from lidar2lidar.extrinsic_io import load_extrinsics_file


def _resolve_path(path_value: str, *, base_directory: Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    return (base_directory / path).resolve()


def default_reference_config_payload() -> dict[str, Any]:
    return {
        "cameras": {
            "parent": {
                "frame_id": "camera_parent",
                "image_directory": "calibration_data/parent",
                "intrinsics_path": "parent_intrinsics.yaml",
            },
            "child": {
                "frame_id": "camera_child",
                "image_directory": "calibration_data/child",
                "intrinsics_path": "child_intrinsics.yaml",
            },
        },
        "target": {
            "type": "checkerboard",
            "pattern_size": [11, 8],
            "square_size_m": 0.025,
        },
        "extraction": {
            "min_bbox_area_ratio": 0.003,
            "min_edge_margin_px": 16.0,
            "max_pnp_reprojection_rms_px": 1.5,
            "max_candidate_translation_delta_m": 0.25,
            "max_candidate_rotation_delta_deg": 3.0,
        },
        "optimization": {
            "min_pairs": 8,
            "loss": "huber",
            "f_scale": 1.0,
            "max_nfev": 300,
            "max_refinement_rounds": 2,
            "outlier_pair_rms_px": 2.5,
        },
        "metrics": {
            "warning_final_rms_px": 1.0,
            "warning_pair_rms_p95_px": 1.5,
            "warning_holdout_rms_px": 1.5,
            "warning_epipolar_p95_px": 1.0,
            "warning_repeatability_translation_m": 0.02,
            "warning_repeatability_rotation_deg": 0.3,
            "min_leave_one_out_pair_count": 6,
            "warning_image_coverage_min_cells": 4,
            "warning_image_horizontal_span_ratio": 0.35,
            "warning_image_vertical_span_ratio": 0.35,
            "warning_depth_span_m": 0.3,
            "warning_tilt_span_deg": 8.0,
            "warning_accepted_pair_ratio": 0.5,
        },
        "initial_transform_path": None,
        "output": {"directory": "outputs/camera2camera/reference"},
    }


def _matrix_from_payload(
    value: Any, *, expected_shape: tuple[int, ...] | None = None
) -> np.ndarray:
    if isinstance(value, dict) and "data" in value:
        rows = int(value.get("rows", 0))
        cols = int(value.get("cols", 0))
        array = np.asarray(value.get("data", []), dtype=float)
        if rows > 0 and cols > 0:
            array = array.reshape(rows, cols)
        return array
    array = np.asarray(value, dtype=float)
    if expected_shape is not None and tuple(array.shape) != tuple(expected_shape):
        raise ValueError(f"Expected shape {expected_shape}, got {array.shape}")
    return array


def _parse_camera_calibration_payload(
    payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    matrix_source = payload.get("camera_matrix", payload.get("intrinsics"))
    distortion_source = payload.get(
        "distortion_coefficients", payload.get("distortion", [0, 0, 0, 0, 0])
    )
    if matrix_source is None:
        raise ValueError(
            "Camera calibration payload is missing camera_matrix/intrinsics."
        )
    camera_matrix = _matrix_from_payload(matrix_source)
    if tuple(camera_matrix.shape) != (3, 3):
        raise ValueError(f"Camera matrix must be 3x3, got {camera_matrix.shape}")
    distortion = _matrix_from_payload(distortion_source).reshape(-1)
    metadata = {
        "distortion_model": payload.get("distortion_model"),
        "image_size_wh": {
            "width": payload.get("image_width"),
            "height": payload.get("image_height"),
        },
    }
    return camera_matrix, distortion, metadata


def _load_camera_config(
    camera_payload: dict[str, Any],
    *,
    base_directory: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if camera_payload.get("intrinsics_path"):
        calibration_path = _resolve_path(
            str(camera_payload["intrinsics_path"]),
            base_directory=base_directory,
        )
        with calibration_path.open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}
        camera_matrix, distortion, metadata = _parse_camera_calibration_payload(payload)
        metadata["source_path"] = str(calibration_path)
        return camera_matrix, distortion, metadata
    camera_matrix, distortion, metadata = _parse_camera_calibration_payload(
        camera_payload
    )
    metadata["source_path"] = None
    return camera_matrix, distortion, metadata


def _load_config(
    config_path: str,
    *,
    prepared_payload: dict[str, Any] | None = None,
    output_dir_override: str | None = None,
) -> tuple[dict[str, Any], StereoCalibrationConfig, dict[str, Any], Path]:
    if prepared_payload is None:
        resolved_config_path = Path(config_path).expanduser().resolve()
        with resolved_config_path.open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}
        base_directory = resolved_config_path.parent
    else:
        payload = copy.deepcopy(prepared_payload)
        base_directory = Path.cwd()

    cameras = payload.get("cameras", {}) or {}
    parent_camera_payload = cameras.get("parent", {}) or {}
    child_camera_payload = cameras.get("child", {}) or {}
    target = payload.get("target", {}) or {}
    extraction = payload.get("extraction", {}) or {}
    optimization = payload.get("optimization", {}) or {}
    metrics = payload.get("metrics", {}) or {}
    output = payload.get("output", {}) or {}

    parent_camera_matrix, parent_distortion, parent_camera_metadata = (
        _load_camera_config(
            parent_camera_payload,
            base_directory=base_directory,
        )
    )
    child_camera_matrix, child_distortion, child_camera_metadata = _load_camera_config(
        child_camera_payload,
        base_directory=base_directory,
    )
    initial_transform = None
    if payload.get("initial_transform_path"):
        initial_transform, _, _, _, _ = load_extrinsics_file(
            str(
                _resolve_path(
                    str(payload["initial_transform_path"]),
                    base_directory=base_directory,
                )
            )
        )

    config = StereoCalibrationConfig(
        parent_image_directory=str(
            _resolve_path(
                str(
                    parent_camera_payload.get(
                        "image_directory", "calibration_data/parent"
                    )
                ),
                base_directory=base_directory,
            )
        ),
        child_image_directory=str(
            _resolve_path(
                str(
                    child_camera_payload.get(
                        "image_directory", "calibration_data/child"
                    )
                ),
                base_directory=base_directory,
            )
        ),
        parent_frame=str(parent_camera_payload.get("frame_id", "camera_parent")),
        child_frame=str(child_camera_payload.get("frame_id", "camera_child")),
        board_pattern_size=tuple(target.get("pattern_size", [11, 8])),
        board_square_size_m=float(target.get("square_size_m", 0.025)),
        extraction_min_bbox_area_ratio=float(
            extraction.get("min_bbox_area_ratio", 0.003)
        ),
        extraction_min_edge_margin_px=float(extraction.get("min_edge_margin_px", 16.0)),
        extraction_max_pnp_reprojection_rms_px=float(
            extraction.get("max_pnp_reprojection_rms_px", 1.5)
        ),
        extraction_max_candidate_translation_delta_m=float(
            extraction.get("max_candidate_translation_delta_m", 0.25)
        ),
        extraction_max_candidate_rotation_delta_deg=float(
            extraction.get("max_candidate_rotation_delta_deg", 3.0)
        ),
        min_pair_count=int(optimization.get("min_pairs", 8)),
        optimization_loss=str(optimization.get("loss", "huber")),
        optimization_f_scale=float(optimization.get("f_scale", 1.0)),
        optimization_max_nfev=int(optimization.get("max_nfev", 300)),
        optimization_max_refinement_rounds=int(
            optimization.get("max_refinement_rounds", 2)
        ),
        optimization_outlier_pair_rms_px=float(
            optimization.get("outlier_pair_rms_px", 2.5)
        ),
        metrics_warning_final_rms_px=float(metrics.get("warning_final_rms_px", 1.0)),
        metrics_warning_pair_rms_p95_px=float(
            metrics.get("warning_pair_rms_p95_px", 1.5)
        ),
        metrics_warning_holdout_rms_px=float(
            metrics.get("warning_holdout_rms_px", 1.5)
        ),
        metrics_warning_epipolar_p95_px=float(
            metrics.get("warning_epipolar_p95_px", 1.0)
        ),
        metrics_warning_repeatability_translation_m=float(
            metrics.get("warning_repeatability_translation_m", 0.02)
        ),
        metrics_warning_repeatability_rotation_deg=float(
            metrics.get("warning_repeatability_rotation_deg", 0.3)
        ),
        metrics_min_leave_one_out_pair_count=int(
            metrics.get("min_leave_one_out_pair_count", 6)
        ),
        metrics_warning_image_coverage_min_cells=int(
            metrics.get("warning_image_coverage_min_cells", 4)
        ),
        metrics_warning_image_horizontal_span_ratio=float(
            metrics.get("warning_image_horizontal_span_ratio", 0.35)
        ),
        metrics_warning_image_vertical_span_ratio=float(
            metrics.get("warning_image_vertical_span_ratio", 0.35)
        ),
        metrics_warning_depth_span_m=float(metrics.get("warning_depth_span_m", 0.3)),
        metrics_warning_tilt_span_deg=float(metrics.get("warning_tilt_span_deg", 8.0)),
        metrics_warning_accepted_pair_ratio=float(
            metrics.get("warning_accepted_pair_ratio", 0.5)
        ),
    )
    resolved_output_dir = _resolve_path(
        str(
            output_dir_override
            or output.get("directory", "outputs/camera2camera/reference")
        ),
        base_directory=base_directory,
    )
    return (
        payload,
        config,
        {
            "parent_camera_matrix": parent_camera_matrix,
            "parent_distortion": parent_distortion,
            "parent_camera_metadata": parent_camera_metadata,
            "child_camera_matrix": child_camera_matrix,
            "child_distortion": child_distortion,
            "child_camera_metadata": child_camera_metadata,
            "initial_transform": initial_transform,
        },
        resolved_output_dir,
    )


def _build_board_template(
    pattern_size: tuple[int, int], square_size: float
) -> np.ndarray:
    object_points = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    object_points[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(
        -1, 2
    )
    object_points *= float(square_size)
    return object_points


def _pair_image_files(
    parent_directory: Path, child_directory: Path
) -> tuple[list[tuple[str, Path, Path]], dict[str, Any]]:
    parent_candidates: dict[str, Path] = {}
    child_candidates: dict[str, Path] = {}
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        for path in sorted(parent_directory.glob(pattern)):
            parent_candidates[path.stem] = path
        for path in sorted(child_directory.glob(pattern)):
            child_candidates[path.stem] = path
    common_stems = sorted(set(parent_candidates) & set(child_candidates))
    pairs = [
        (stem, parent_candidates[stem], child_candidates[stem]) for stem in common_stems
    ]
    return pairs, {
        "parent_image_count": len(parent_candidates),
        "child_image_count": len(child_candidates),
        "paired_count": len(pairs),
        "missing_parent_stems": sorted(set(child_candidates) - set(parent_candidates)),
        "missing_child_stems": sorted(set(parent_candidates) - set(child_candidates)),
    }


def _find_checkerboard_corners(
    image: np.ndarray, pattern_size: tuple[int, int]
) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
            flags |= cv2.CALIB_CB_EXHAUSTIVE
        found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)
        if found:
            return np.asarray(corners, dtype=float).reshape(-1, 2)
    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None
    refined = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )
    return np.asarray(refined, dtype=float).reshape(-1, 2)


def _image_point_metrics(
    image_points: np.ndarray, image_size_wh: tuple[int, int]
) -> dict[str, Any]:
    width, height = image_size_wh
    bbox_min = np.min(image_points, axis=0)
    bbox_max = np.max(image_points, axis=0)
    center = np.mean(image_points, axis=0)
    area_ratio = float(
        max((bbox_max[0] - bbox_min[0]), 0.0)
        * max((bbox_max[1] - bbox_min[1]), 0.0)
        / max(width * height, 1)
    )
    edge_margin_px = float(
        min(
            bbox_min[0],
            bbox_min[1],
            max(width - bbox_max[0], 0.0),
            max(height - bbox_max[1], 0.0),
        )
    )
    return {
        "center_xy_normalized": {
            "x": float(center[0] / max(width, 1)),
            "y": float(center[1] / max(height, 1)),
        },
        "bbox_area_ratio": area_ratio,
        "edge_margin_px": edge_margin_px,
    }


def _corner_permutations(
    corners: np.ndarray, pattern_size: tuple[int, int]
) -> dict[str, np.ndarray]:
    cols, rows = pattern_size
    grid = np.asarray(corners, dtype=float).reshape(rows, cols, 2)
    return {
        "identity": grid.reshape(-1, 2),
        "flip_x": np.flip(grid, axis=1).reshape(-1, 2),
        "flip_y": np.flip(grid, axis=0).reshape(-1, 2),
        "flip_xy": np.flip(np.flip(grid, axis=0), axis=1).reshape(-1, 2),
    }


def _transform_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3], _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
    transform[:3, 3] = np.asarray(tvec, dtype=float).reshape(3)
    return transform


def _solve_board_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> dict[str, Any] | None:
    flags = (
        cv2.SOLVEPNP_IPPE if hasattr(cv2, "SOLVEPNP_IPPE") else cv2.SOLVEPNP_ITERATIVE
    )
    success, rvec, tvec = cv2.solvePnP(
        np.asarray(object_points, dtype=np.float32),
        np.asarray(image_points, dtype=np.float32).reshape(-1, 1, 2),
        np.asarray(camera_matrix, dtype=float),
        np.asarray(distortion, dtype=float),
        flags=flags,
    )
    if not success:
        return None
    transform = _transform_from_rvec_tvec(rvec, tvec)
    projected, _ = cv2.projectPoints(
        np.asarray(object_points, dtype=float),
        rvec,
        tvec,
        np.asarray(camera_matrix, dtype=float),
        np.asarray(distortion, dtype=float),
    )
    residuals = np.asarray(projected, dtype=float).reshape(-1, 2) - np.asarray(
        image_points, dtype=float
    ).reshape(-1, 2)
    rms_px = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
    object_points_camera = (
        transform[:3, :3] @ np.asarray(object_points, dtype=float).T
    ).T + transform[:3, 3]
    board_center = np.mean(object_points_camera, axis=0)
    centered = object_points_camera - board_center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = np.asarray(vh[-1], dtype=float)
    normal /= max(np.linalg.norm(normal), 1e-12)
    tilt_deg = float(np.degrees(np.arccos(np.clip(np.abs(normal[2]), 0.0, 1.0))))
    return {
        "transform": transform,
        "rvec": np.asarray(rvec, dtype=float).reshape(3),
        "tvec": np.asarray(tvec, dtype=float).reshape(3),
        "reprojection_rms_px": rms_px,
        "board_center_camera_m": {
            "x": float(board_center[0]),
            "y": float(board_center[1]),
            "z": float(board_center[2]),
        },
        "board_tilt_deg": tilt_deg,
    }


def _transform_cost(
    reference_transform: np.ndarray, candidate_transform: np.ndarray
) -> float:
    delta = transform_delta_metrics(reference_transform, candidate_transform)
    return delta["translation_norm_m"] / 0.05 + delta["rotation_deg"] / 1.0


def _medoid_transform(transforms: list[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("Cannot compute medoid of an empty transform list.")
    best_index = 0
    best_cost = float("inf")
    for index, transform in enumerate(transforms):
        cost = 0.0
        for other in transforms:
            cost += _transform_cost(transform, other)
        if cost < best_cost:
            best_cost = cost
            best_index = index
    return np.asarray(transforms[best_index], dtype=float)


def _cluster_transform_count(
    transforms: list[np.ndarray],
    *,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
) -> int:
    representatives: list[np.ndarray] = []
    for transform in transforms:
        matched = False
        for representative in representatives:
            delta = transform_delta_metrics(representative, transform)
            if (
                delta["translation_norm_m"] <= translation_threshold_m
                and delta["rotation_deg"] <= rotation_threshold_deg
            ):
                matched = True
                break
        if not matched:
            representatives.append(transform)
    return len(representatives)


def _resolve_ordering_candidates(
    preliminary_pairs: list[dict[str, Any]],
    *,
    seed_transform: np.ndarray | None,
    config: StereoCalibrationConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    if not preliminary_pairs:
        raise ValueError(
            "No preliminary stereo pairs were available for ordering resolution."
        )
    if seed_transform is None:
        all_candidates = [
            candidate["relative_transform"]
            for pair in preliminary_pairs
            for candidate in pair["child_candidates"]
        ]
        current_seed = _medoid_transform(all_candidates)
    else:
        current_seed = np.asarray(seed_transform, dtype=float)
    changed_pair_count = 0
    selected_pairs: list[dict[str, Any]] = []
    for _ in range(5):
        previous_labels = {
            pair["pose_id"]: pair.get("selected_permutation") for pair in selected_pairs
        }
        selected_pairs = []
        selected_transforms = []
        for pair in preliminary_pairs:
            candidate = min(
                pair["child_candidates"],
                key=lambda item: _transform_cost(
                    current_seed, np.asarray(item["relative_transform"], dtype=float)
                ),
            )
            selected_pair = dict(pair)
            selected_pair["selected_child_candidate"] = candidate
            selected_pair["selected_permutation"] = candidate["permutation"]
            selected_pairs.append(selected_pair)
            selected_transforms.append(
                np.asarray(candidate["relative_transform"], dtype=float)
            )
        new_seed = _medoid_transform(selected_transforms)
        changed_pair_count = sum(
            1
            for pair in selected_pairs
            if previous_labels.get(pair["pose_id"])
            not in (None, pair["selected_permutation"])
        )
        if np.allclose(new_seed, current_seed, atol=1e-9):
            current_seed = new_seed
            break
        current_seed = new_seed

    resolved_pairs = []
    rejected_pair_count = 0
    for pair in selected_pairs:
        candidate_transform = np.asarray(
            pair["selected_child_candidate"]["relative_transform"], dtype=float
        )
        delta = transform_delta_metrics(current_seed, candidate_transform)
        pair_copy = dict(pair)
        pair_copy["selected_child_candidate"]["delta_to_consensus"] = delta
        pair_copy["accepted"] = (
            delta["translation_norm_m"]
            <= config.extraction_max_candidate_translation_delta_m
            and delta["rotation_deg"]
            <= config.extraction_max_candidate_rotation_delta_deg
        )
        if not pair_copy["accepted"]:
            pair_copy["skip_reason"] = "inconsistent_relative_transform"
            rejected_pair_count += 1
        resolved_pairs.append(pair_copy)
    return (
        resolved_pairs,
        {
            "seed_transform": current_seed,
            "changed_pair_count": int(changed_pair_count),
            "rejected_pair_count": int(rejected_pair_count),
            "resolved_pairs": [
                {
                    "pose_id": pair["pose_id"],
                    "selected_permutation": pair["selected_permutation"],
                    "accepted": bool(pair["accepted"]),
                    "delta_to_consensus": pair["selected_child_candidate"].get(
                        "delta_to_consensus"
                    ),
                }
                for pair in resolved_pairs
            ],
        },
        current_seed,
    )


def _build_dataset(
    *,
    config: StereoCalibrationConfig,
    camera_data: dict[str, Any],
) -> tuple[StereoCalibrationDataset, dict[str, Any], np.ndarray]:
    parent_directory = Path(config.parent_image_directory).expanduser()
    child_directory = Path(config.child_image_directory).expanduser()
    pairs, pairing_summary = _pair_image_files(parent_directory, child_directory)
    object_points = _build_board_template(
        config.board_pattern_size, config.board_square_size_m
    )
    preliminary_pairs = []
    extraction_entries: list[dict[str, Any]] = []
    skip_reason_counts: dict[str, int] = {}
    dataset_metadata = {
        "board_template_extent_xy_m": {
            "x": float((config.board_pattern_size[0] - 1) * config.board_square_size_m),
            "y": float((config.board_pattern_size[1] - 1) * config.board_square_size_m),
        },
        "target": {
            "type": "checkerboard",
            "pattern_size": list(config.board_pattern_size),
            "square_size_m": float(config.board_square_size_m),
        },
        "camera_sources": {
            "parent": camera_data["parent_camera_metadata"],
            "child": camera_data["child_camera_metadata"],
        },
    }
    fallback_initial_transform = (
        np.asarray(camera_data["initial_transform"], dtype=float)
        if camera_data.get("initial_transform") is not None
        else np.eye(4, dtype=float)
    )

    for pose_id, parent_path, child_path in pairs:
        entry: dict[str, Any] = {
            "pose_id": pose_id,
            "parent_image_path": str(parent_path),
            "child_image_path": str(child_path),
        }
        parent_image = cv2.imread(str(parent_path))
        child_image = cv2.imread(str(child_path))
        if parent_image is None or child_image is None:
            entry["skip_reason"] = "image_read_failed"
            extraction_entries.append(entry)
            skip_reason_counts["image_read_failed"] = (
                skip_reason_counts.get("image_read_failed", 0) + 1
            )
            continue
        parent_size = (int(parent_image.shape[1]), int(parent_image.shape[0]))
        child_size = (int(child_image.shape[1]), int(child_image.shape[0]))
        entry["parent_image_size_wh"] = {
            "width": parent_size[0],
            "height": parent_size[1],
        }
        entry["child_image_size_wh"] = {"width": child_size[0], "height": child_size[1]}

        expected_parent_size = (
            camera_data["parent_camera_metadata"].get("image_size_wh") or {}
        )
        expected_child_size = (
            camera_data["child_camera_metadata"].get("image_size_wh") or {}
        )
        if (
            expected_parent_size.get("width")
            and expected_parent_size.get("height")
            and (
                parent_size[0] != int(expected_parent_size["width"])
                or parent_size[1] != int(expected_parent_size["height"])
            )
        ):
            entry["skip_reason"] = "parent_image_size_mismatch"
            extraction_entries.append(entry)
            skip_reason_counts["parent_image_size_mismatch"] = (
                skip_reason_counts.get("parent_image_size_mismatch", 0) + 1
            )
            continue
        if (
            expected_child_size.get("width")
            and expected_child_size.get("height")
            and (
                child_size[0] != int(expected_child_size["width"])
                or child_size[1] != int(expected_child_size["height"])
            )
        ):
            entry["skip_reason"] = "child_image_size_mismatch"
            extraction_entries.append(entry)
            skip_reason_counts["child_image_size_mismatch"] = (
                skip_reason_counts.get("child_image_size_mismatch", 0) + 1
            )
            continue

        parent_corners = _find_checkerboard_corners(
            parent_image, config.board_pattern_size
        )
        child_corners = _find_checkerboard_corners(
            child_image, config.board_pattern_size
        )
        if parent_corners is None:
            entry["skip_reason"] = "parent_checkerboard_not_found"
            extraction_entries.append(entry)
            skip_reason_counts["parent_checkerboard_not_found"] = (
                skip_reason_counts.get("parent_checkerboard_not_found", 0) + 1
            )
            continue
        if child_corners is None:
            entry["skip_reason"] = "child_checkerboard_not_found"
            extraction_entries.append(entry)
            skip_reason_counts["child_checkerboard_not_found"] = (
                skip_reason_counts.get("child_checkerboard_not_found", 0) + 1
            )
            continue
        parent_metrics = _image_point_metrics(parent_corners, parent_size)
        child_metrics = _image_point_metrics(child_corners, child_size)
        entry["parent_image_metrics"] = parent_metrics
        entry["child_image_metrics"] = child_metrics
        if (
            parent_metrics["bbox_area_ratio"] < config.extraction_min_bbox_area_ratio
            or child_metrics["bbox_area_ratio"] < config.extraction_min_bbox_area_ratio
        ):
            entry["skip_reason"] = "board_too_small"
            extraction_entries.append(entry)
            skip_reason_counts["board_too_small"] = (
                skip_reason_counts.get("board_too_small", 0) + 1
            )
            continue
        if (
            parent_metrics["edge_margin_px"] < config.extraction_min_edge_margin_px
            or child_metrics["edge_margin_px"] < config.extraction_min_edge_margin_px
        ):
            entry["skip_reason"] = "board_too_close_to_edge"
            extraction_entries.append(entry)
            skip_reason_counts["board_too_close_to_edge"] = (
                skip_reason_counts.get("board_too_close_to_edge", 0) + 1
            )
            continue

        parent_pose = _solve_board_pose(
            object_points,
            parent_corners,
            camera_data["parent_camera_matrix"],
            camera_data["parent_distortion"],
        )
        if parent_pose is None:
            entry["skip_reason"] = "parent_pnp_failed"
            extraction_entries.append(entry)
            skip_reason_counts["parent_pnp_failed"] = (
                skip_reason_counts.get("parent_pnp_failed", 0) + 1
            )
            continue
        if (
            parent_pose["reprojection_rms_px"]
            > config.extraction_max_pnp_reprojection_rms_px
        ):
            entry["skip_reason"] = "parent_pnp_reprojection_too_high"
            extraction_entries.append(entry)
            skip_reason_counts["parent_pnp_reprojection_too_high"] = (
                skip_reason_counts.get("parent_pnp_reprojection_too_high", 0) + 1
            )
            continue

        child_candidates = []
        for permutation, candidate_corners in _corner_permutations(
            child_corners, config.board_pattern_size
        ).items():
            child_pose = _solve_board_pose(
                object_points,
                candidate_corners,
                camera_data["child_camera_matrix"],
                camera_data["child_distortion"],
            )
            if child_pose is None:
                continue
            if (
                child_pose["reprojection_rms_px"]
                > config.extraction_max_pnp_reprojection_rms_px
            ):
                continue
            relative_transform = child_pose["transform"] @ np.linalg.inv(
                parent_pose["transform"]
            )
            child_candidates.append(
                {
                    "permutation": permutation,
                    "image_points": np.asarray(candidate_corners, dtype=float),
                    "child_pose": child_pose,
                    "relative_transform": relative_transform,
                }
            )
        if not child_candidates:
            entry["skip_reason"] = "child_pnp_failed"
            extraction_entries.append(entry)
            skip_reason_counts["child_pnp_failed"] = (
                skip_reason_counts.get("child_pnp_failed", 0) + 1
            )
            continue
        preliminary_pairs.append(
            {
                "pose_id": pose_id,
                "parent_image_path": str(parent_path),
                "child_image_path": str(child_path),
                "parent_image_size_wh": parent_size,
                "child_image_size_wh": child_size,
                "parent_image_points": np.asarray(parent_corners, dtype=float),
                "raw_child_image_points": np.asarray(child_corners, dtype=float),
                "parent_pose": parent_pose,
                "child_candidates": child_candidates,
                "image_entry": entry,
            }
        )

    if not preliminary_pairs:
        dataset = StereoCalibrationDataset(
            parent_frame=config.parent_frame,
            child_frame=config.child_frame,
            parent_camera_matrix=np.asarray(
                camera_data["parent_camera_matrix"], dtype=float
            ),
            parent_camera_distortion=np.asarray(
                camera_data["parent_distortion"], dtype=float
            ),
            child_camera_matrix=np.asarray(
                camera_data["child_camera_matrix"], dtype=float
            ),
            child_camera_distortion=np.asarray(
                camera_data["child_distortion"], dtype=float
            ),
            observations=[],
            initial_transform=fallback_initial_transform,
            metadata=dataset_metadata,
        )
        extraction_report = {
            "pairing_summary": pairing_summary,
            "entries": extraction_entries,
            "skip_reason_counts": skip_reason_counts,
            "pre_optimization_accepted_pair_count": 0,
            "pre_optimization_accepted_pair_ratio": 0.0,
            "rejected_pair_count": int(
                sum(
                    1
                    for row in extraction_entries
                    if row.get("skip_reason") is not None
                )
            ),
            "ordering_resolution": {},
            "ready_for_optimization": False,
            "failure_message": "No valid stereo pairs survived extraction.",
        }
        return dataset, extraction_report, fallback_initial_transform

    resolved_pairs, ordering_resolution, consensus_seed = _resolve_ordering_candidates(
        preliminary_pairs,
        seed_transform=camera_data.get("initial_transform"),
        config=config,
    )

    observations = []
    for pair in resolved_pairs:
        selected = pair["selected_child_candidate"]
        entry = dict(pair["image_entry"])
        entry["selected_permutation"] = pair["selected_permutation"]
        entry["parent_pnp_reprojection_rms_px"] = float(
            pair["parent_pose"]["reprojection_rms_px"]
        )
        entry["child_pnp_reprojection_rms_px"] = float(
            selected["child_pose"]["reprojection_rms_px"]
        )
        entry["relative_transform_delta_to_consensus"] = selected.get(
            "delta_to_consensus"
        )
        if not pair["accepted"]:
            entry["skip_reason"] = str(pair.get("skip_reason", "rejected"))
            extraction_entries.append(entry)
            reason = entry["skip_reason"]
            skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + 1
            continue
        entry["skip_reason"] = None
        extraction_entries.append(entry)
        metadata = {
            "parent_image_metrics": pair["image_entry"]["parent_image_metrics"],
            "child_image_metrics": pair["image_entry"]["child_image_metrics"],
            "parent_pose_initial": pair["parent_pose"],
            "child_pose_initial": selected["child_pose"],
            "initial_parent_board_transform_matrix": np.asarray(
                pair["parent_pose"]["transform"], dtype=float
            ).tolist(),
            "initial_child_board_transform_matrix": np.asarray(
                selected["child_pose"]["transform"], dtype=float
            ).tolist(),
            "selected_permutation": pair["selected_permutation"],
            "delta_to_consensus": selected.get("delta_to_consensus"),
        }
        observations.append(
            StereoCalibrationObservation(
                pose_id=pair["pose_id"],
                parent_image_path=pair["parent_image_path"],
                child_image_path=pair["child_image_path"],
                parent_image_size_wh=pair["parent_image_size_wh"],
                child_image_size_wh=pair["child_image_size_wh"],
                parent_image_points=np.asarray(
                    pair["parent_image_points"], dtype=float
                ),
                child_image_points=np.asarray(selected["image_points"], dtype=float),
                object_points=np.asarray(object_points, dtype=float),
                metadata=metadata,
            )
        )

    dataset = StereoCalibrationDataset(
        parent_frame=config.parent_frame,
        child_frame=config.child_frame,
        parent_camera_matrix=np.asarray(
            camera_data["parent_camera_matrix"], dtype=float
        ),
        parent_camera_distortion=np.asarray(
            camera_data["parent_distortion"], dtype=float
        ),
        child_camera_matrix=np.asarray(camera_data["child_camera_matrix"], dtype=float),
        child_camera_distortion=np.asarray(
            camera_data["child_distortion"], dtype=float
        ),
        observations=observations,
        initial_transform=np.asarray(consensus_seed, dtype=float),
        metadata=dataset_metadata,
    )
    rejected_pair_count = int(
        sum(1 for row in extraction_entries if row.get("skip_reason") is not None)
    )
    pre_optimization_accepted_pair_count = int(len(observations))
    extraction_report = {
        "pairing_summary": pairing_summary,
        "entries": extraction_entries,
        "skip_reason_counts": skip_reason_counts,
        "pre_optimization_accepted_pair_count": pre_optimization_accepted_pair_count,
        "pre_optimization_accepted_pair_ratio": float(
            pre_optimization_accepted_pair_count
            / max(pairing_summary.get("paired_count", 1), 1)
        ),
        "rejected_pair_count": rejected_pair_count,
        "ordering_resolution": ordering_resolution,
        "ready_for_optimization": len(observations) >= config.min_pair_count,
        "failure_message": (
            None
            if len(observations) >= config.min_pair_count
            else (
                f"Only {len(observations)} stereo pairs survived extraction; "
                f"need at least {config.min_pair_count}."
            )
        ),
    }
    return dataset, extraction_report, np.asarray(consensus_seed, dtype=float)


def _pack_params(
    global_transform: np.ndarray, board_transforms: list[np.ndarray]
) -> np.ndarray:
    params = []
    params.extend(R.from_matrix(global_transform[:3, :3]).as_rotvec().tolist())
    params.extend(np.asarray(global_transform[:3, 3], dtype=float).tolist())
    for transform in board_transforms:
        params.extend(R.from_matrix(transform[:3, :3]).as_rotvec().tolist())
        params.extend(np.asarray(transform[:3, 3], dtype=float).tolist())
    return np.asarray(params, dtype=float)


def _unpack_params(
    params: np.ndarray, pair_count: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = R.from_rotvec(np.asarray(params[:3], dtype=float)).as_matrix()
    transform[:3, 3] = np.asarray(params[3:6], dtype=float)
    board_transforms = []
    offset = 6
    for _ in range(pair_count):
        board_transform = np.eye(4, dtype=float)
        board_transform[:3, :3] = R.from_rotvec(
            np.asarray(params[offset : offset + 3], dtype=float)
        ).as_matrix()
        board_transform[:3, 3] = np.asarray(
            params[offset + 3 : offset + 6], dtype=float
        )
        board_transforms.append(board_transform)
        offset += 6
    return transform, board_transforms


def _project_points_with_transform(
    object_points: np.ndarray,
    transform: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    rvec = R.from_matrix(transform[:3, :3]).as_rotvec().reshape(3, 1)
    tvec = np.asarray(transform[:3, 3], dtype=float).reshape(3, 1)
    projected, _ = cv2.projectPoints(
        np.asarray(object_points, dtype=float),
        rvec,
        tvec,
        np.asarray(camera_matrix, dtype=float),
        np.asarray(distortion, dtype=float),
    )
    return np.asarray(projected, dtype=float).reshape(-1, 2)


def _undistort_pixel_points(
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    undistorted = cv2.undistortPoints(
        np.asarray(image_points, dtype=float).reshape(-1, 1, 2),
        np.asarray(camera_matrix, dtype=float),
        np.asarray(distortion, dtype=float),
        P=np.asarray(camera_matrix, dtype=float),
    )
    return np.asarray(undistorted, dtype=float).reshape(-1, 2)


def _bundle_adjustment_residuals(
    params: np.ndarray,
    dataset: StereoCalibrationDataset,
) -> np.ndarray:
    global_transform, board_transforms = _unpack_params(
        params, len(dataset.observations)
    )
    residuals = []
    for observation, board_transform in zip(dataset.observations, board_transforms):
        parent_projected = _project_points_with_transform(
            observation.object_points,
            board_transform,
            dataset.parent_camera_matrix,
            dataset.parent_camera_distortion,
        )
        child_projected = _project_points_with_transform(
            observation.object_points,
            global_transform @ board_transform,
            dataset.child_camera_matrix,
            dataset.child_camera_distortion,
        )
        residuals.append(
            (
                parent_projected
                - np.asarray(observation.parent_image_points, dtype=float)
            ).reshape(-1)
        )
        residuals.append(
            (
                child_projected
                - np.asarray(observation.child_image_points, dtype=float)
            ).reshape(-1)
        )
    return np.concatenate(residuals, axis=0) if residuals else np.zeros(0, dtype=float)


def _rms_from_residual_vector(residual_vector: np.ndarray) -> float:
    residual_vector = np.asarray(residual_vector, dtype=float)
    if residual_vector.size <= 0:
        return 0.0
    return float(np.sqrt(np.mean(residual_vector**2)))


def _evaluate_pair_rows(
    dataset: StereoCalibrationDataset,
    global_transform: np.ndarray,
    board_transforms: list[np.ndarray],
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    if not dataset.observations:
        return [], None
    rotation = np.asarray(global_transform[:3, :3], dtype=float)
    translation = np.asarray(global_transform[:3, 3], dtype=float)
    skew_t = np.array(
        [
            [0.0, -translation[2], translation[1]],
            [translation[2], 0.0, -translation[0]],
            [-translation[1], translation[0], 0.0],
        ],
        dtype=float,
    )
    fundamental = (
        np.linalg.inv(dataset.child_camera_matrix).T
        @ skew_t
        @ rotation
        @ np.linalg.inv(dataset.parent_camera_matrix)
    )
    rows = []
    for observation, board_transform in zip(dataset.observations, board_transforms):
        parent_projected = _project_points_with_transform(
            observation.object_points,
            board_transform,
            dataset.parent_camera_matrix,
            dataset.parent_camera_distortion,
        )
        child_transform = global_transform @ board_transform
        child_projected = _project_points_with_transform(
            observation.object_points,
            child_transform,
            dataset.child_camera_matrix,
            dataset.child_camera_distortion,
        )
        parent_observed = np.asarray(observation.parent_image_points, dtype=float)
        child_observed = np.asarray(observation.child_image_points, dtype=float)
        parent_residual = parent_projected - parent_observed
        child_residual = child_projected - child_observed
        parent_rms = float(np.sqrt(np.mean(np.sum(parent_residual**2, axis=1))))
        child_rms = float(np.sqrt(np.mean(np.sum(child_residual**2, axis=1))))
        combined_residual = np.vstack([parent_residual, child_residual])
        combined_rms = float(np.sqrt(np.mean(np.sum(combined_residual**2, axis=1))))

        parent_undistorted = _undistort_pixel_points(
            parent_observed,
            dataset.parent_camera_matrix,
            dataset.parent_camera_distortion,
        )
        child_undistorted = _undistort_pixel_points(
            child_observed,
            dataset.child_camera_matrix,
            dataset.child_camera_distortion,
        )
        parent_h = np.concatenate(
            [parent_undistorted, np.ones((parent_undistorted.shape[0], 1))], axis=1
        )
        child_h = np.concatenate(
            [child_undistorted, np.ones((child_undistorted.shape[0], 1))], axis=1
        )
        child_lines = (fundamental @ parent_h.T).T
        parent_lines = (fundamental.T @ child_h.T).T
        child_distance = np.abs(np.sum(child_lines * child_h, axis=1)) / np.maximum(
            np.linalg.norm(child_lines[:, :2], axis=1), 1e-12
        )
        parent_distance = np.abs(np.sum(parent_lines * parent_h, axis=1)) / np.maximum(
            np.linalg.norm(parent_lines[:, :2], axis=1), 1e-12
        )
        epipolar_distance = 0.5 * (child_distance + parent_distance)

        object_points_parent = (
            board_transform[:3, :3]
            @ np.asarray(observation.object_points, dtype=float).T
        ).T + board_transform[:3, 3]
        board_center = np.mean(object_points_parent, axis=0)
        centered = object_points_parent - board_center
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = np.asarray(vh[-1], dtype=float)
        normal /= max(np.linalg.norm(normal), 1e-12)
        tilt_deg = float(np.degrees(np.arccos(np.clip(np.abs(normal[2]), 0.0, 1.0))))
        rows.append(
            {
                "pose_id": observation.pose_id,
                "parent_rms_px": parent_rms,
                "child_rms_px": child_rms,
                "combined_rms_px": combined_rms,
                "epipolar_mean_px": float(np.mean(epipolar_distance)),
                "epipolar_p95_px": float(np.percentile(epipolar_distance, 95)),
                "board_center_parent_camera_m": {
                    "x": float(board_center[0]),
                    "y": float(board_center[1]),
                    "z": float(board_center[2]),
                },
                "board_tilt_deg": tilt_deg,
                "selected_permutation": observation.metadata.get(
                    "selected_permutation"
                ),
            }
        )
    return rows, fundamental


def _run_bundle_adjustment(
    dataset: StereoCalibrationDataset,
    config: StereoCalibrationConfig,
    initial_transform: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    initial_board_transforms = [
        np.asarray(
            observation.metadata["initial_parent_board_transform_matrix"], dtype=float
        )
        for observation in dataset.observations
    ]
    initial_params = _pack_params(initial_transform, initial_board_transforms)
    initial_residuals = _bundle_adjustment_residuals(initial_params, dataset)
    result = least_squares(
        _bundle_adjustment_residuals,
        initial_params,
        args=(dataset,),
        method="trf",
        jac="2-point",
        loss=config.optimization_loss,
        f_scale=config.optimization_f_scale,
        max_nfev=config.optimization_max_nfev,
    )
    final_transform, board_transforms = _unpack_params(
        result.x, len(dataset.observations)
    )
    final_residuals = _bundle_adjustment_residuals(result.x, dataset)
    return (
        final_transform,
        board_transforms,
        {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
            "jacobian_shape": (
                list(result.jac.shape) if result.jac is not None else None
            ),
            "jacobian_rank": (
                int(np.linalg.matrix_rank(result.jac))
                if result.jac is not None
                else None
            ),
            "initial_rms_px": _rms_from_residual_vector(initial_residuals),
            "final_rms_px": _rms_from_residual_vector(final_residuals),
        },
    )


def _subset_dataset(
    dataset: StereoCalibrationDataset, observations: list[StereoCalibrationObservation]
) -> StereoCalibrationDataset:
    return StereoCalibrationDataset(
        parent_frame=dataset.parent_frame,
        child_frame=dataset.child_frame,
        parent_camera_matrix=np.asarray(dataset.parent_camera_matrix, dtype=float),
        parent_camera_distortion=np.asarray(
            dataset.parent_camera_distortion, dtype=float
        ),
        child_camera_matrix=np.asarray(dataset.child_camera_matrix, dtype=float),
        child_camera_distortion=np.asarray(
            dataset.child_camera_distortion, dtype=float
        ),
        observations=observations,
        initial_transform=(
            np.asarray(dataset.initial_transform, dtype=float)
            if dataset.initial_transform is not None
            else None
        ),
        metadata=dict(dataset.metadata),
    )


def _optimize_dataset(
    dataset: StereoCalibrationDataset,
    config: StereoCalibrationConfig,
    initial_transform: np.ndarray,
) -> tuple[StereoCalibrationDataset, np.ndarray, list[np.ndarray], dict[str, Any]]:
    current_dataset = dataset
    current_initial_transform = np.asarray(initial_transform, dtype=float)
    rounds = []
    final_transform = current_initial_transform
    final_board_transforms: list[np.ndarray] = []
    final_state: dict[str, Any] = {"success": False}
    for round_index in range(max(1, config.optimization_max_refinement_rounds)):
        optimized_transform, board_transforms, run_state = _run_bundle_adjustment(
            current_dataset,
            config,
            current_initial_transform,
        )
        pair_rows, _ = _evaluate_pair_rows(
            current_dataset,
            optimized_transform,
            board_transforms,
        )
        outlier_pose_ids = [
            row["pose_id"]
            for row in pair_rows
            if float(row["combined_rms_px"]) > config.optimization_outlier_pair_rms_px
        ]
        round_report = {
            "round_index": int(round_index),
            "pair_count": int(len(current_dataset.observations)),
            "initial_rms_px": float(run_state["initial_rms_px"]),
            "final_rms_px": float(run_state["final_rms_px"]),
            "outlier_pair_rms_px": float(config.optimization_outlier_pair_rms_px),
            "outlier_pose_ids": list(outlier_pose_ids),
            "optimization": run_state,
        }
        rounds.append(round_report)
        final_transform = optimized_transform
        final_board_transforms = board_transforms
        final_state = run_state
        if (
            round_index + 1 >= config.optimization_max_refinement_rounds
            or not outlier_pose_ids
        ):
            break
        remaining_observations = [
            observation
            for observation in current_dataset.observations
            if observation.pose_id not in set(outlier_pose_ids)
        ]
        if len(remaining_observations) < config.min_pair_count:
            break
        current_dataset = _subset_dataset(current_dataset, remaining_observations)
        current_initial_transform = np.asarray(final_transform, dtype=float)

    final_dataset = _subset_dataset(current_dataset, list(current_dataset.observations))
    return (
        final_dataset,
        final_transform,
        final_board_transforms,
        {
            "success": bool(final_state.get("success")),
            "initial_rms_px": float(rounds[0]["initial_rms_px"]) if rounds else None,
            "final_rms_px": float(rounds[-1]["final_rms_px"]) if rounds else None,
            "rounds": rounds,
            "active_pose_ids": [
                observation.pose_id for observation in final_dataset.observations
            ],
            "message": str(final_state.get("message", "")),
        },
    )


def _image_coverage_metrics(
    observations: list[StereoCalibrationObservation], image_key: str
) -> dict[str, Any] | None:
    centers_x = []
    centers_y = []
    area_ratios = []
    edge_margins = []
    grid_counts = [[0, 0, 0] for _ in range(3)]
    per_pair = []
    for observation in observations:
        points = np.asarray(
            getattr(observation, f"{image_key}_image_points"), dtype=float
        )
        image_size = getattr(observation, f"{image_key}_image_size_wh")
        width, height = image_size
        metrics = _image_point_metrics(points, image_size)
        center = metrics["center_xy_normalized"]
        cell_x = min(2, max(0, int(float(center["x"]) * 3.0)))
        cell_y = min(2, max(0, int(float(center["y"]) * 3.0)))
        grid_counts[cell_y][cell_x] += 1
        centers_x.append(float(center["x"]))
        centers_y.append(float(center["y"]))
        area_ratios.append(float(metrics["bbox_area_ratio"]))
        edge_margins.append(float(metrics["edge_margin_px"]))
        per_pair.append(
            {
                "pose_id": observation.pose_id,
                "grid_cell": {"x": cell_x, "y": cell_y},
                "center_xy_normalized": center,
                "bbox_area_ratio": float(metrics["bbox_area_ratio"]),
                "edge_margin_px": float(metrics["edge_margin_px"]),
                "image_size_wh": {"width": int(width), "height": int(height)},
            }
        )
    if not per_pair:
        return None
    return {
        "occupied_cell_count": int(
            sum(1 for row in grid_counts for count in row if int(count) > 0)
        ),
        "grid_counts": grid_counts,
        "horizontal_span_ratio": float(max(centers_x) - min(centers_x)),
        "vertical_span_ratio": float(max(centers_y) - min(centers_y)),
        "bbox_area_ratio": float_list_summary(area_ratios),
        "edge_margin_px": float_list_summary(edge_margins),
        "per_pair": per_pair,
    }


def _pose_diversity_metrics(
    per_pair_rows: list[dict[str, Any]]
) -> dict[str, Any] | None:
    if not per_pair_rows:
        return None
    depths = []
    tilts = []
    for row in per_pair_rows:
        center = row.get("board_center_parent_camera_m", {}) or {}
        if center.get("z") is None or row.get("board_tilt_deg") is None:
            continue
        depths.append(float(center["z"]))
        tilts.append(float(row["board_tilt_deg"]))
    if not depths or not tilts:
        return None
    return {
        "board_center_depth_m": float_list_summary(depths),
        "board_tilt_deg": float_list_summary(tilts),
        "depth_span_m": float(max(depths) - min(depths)),
        "tilt_span_deg": float(max(tilts) - min(tilts)),
        "per_pair": per_pair_rows,
    }


def _optimize_holdout_board_pose(
    observation: StereoCalibrationObservation,
    dataset: StereoCalibrationDataset,
    global_transform: np.ndarray,
) -> float:
    initial_board_transform = np.asarray(
        observation.metadata["initial_parent_board_transform_matrix"], dtype=float
    )
    initial_params = np.concatenate(
        [
            R.from_matrix(initial_board_transform[:3, :3]).as_rotvec(),
            initial_board_transform[:3, 3],
        ]
    )

    def residuals(params: np.ndarray) -> np.ndarray:
        board_transform = np.eye(4, dtype=float)
        board_transform[:3, :3] = R.from_rotvec(
            np.asarray(params[:3], dtype=float)
        ).as_matrix()
        board_transform[:3, 3] = np.asarray(params[3:6], dtype=float)
        parent_projected = _project_points_with_transform(
            observation.object_points,
            board_transform,
            dataset.parent_camera_matrix,
            dataset.parent_camera_distortion,
        )
        child_projected = _project_points_with_transform(
            observation.object_points,
            global_transform @ board_transform,
            dataset.child_camera_matrix,
            dataset.child_camera_distortion,
        )
        return np.concatenate(
            [
                (
                    parent_projected
                    - np.asarray(observation.parent_image_points, dtype=float)
                ).reshape(-1),
                (
                    child_projected
                    - np.asarray(observation.child_image_points, dtype=float)
                ).reshape(-1),
            ]
        )

    result = least_squares(
        residuals,
        initial_params,
        method="trf",
        jac="2-point",
        loss="huber",
        f_scale=1.0,
        max_nfev=100,
    )
    return _rms_from_residual_vector(residuals(result.x))


def _leave_one_out_repeatability(
    dataset: StereoCalibrationDataset,
    config: StereoCalibrationConfig,
    primary_transform: np.ndarray,
) -> dict[str, Any] | None:
    if len(dataset.observations) < config.metrics_min_leave_one_out_pair_count:
        return None
    trials = []
    trial_transforms = []
    for holdout_index, holdout_observation in enumerate(dataset.observations):
        train_observations = [
            observation
            for index, observation in enumerate(dataset.observations)
            if index != holdout_index
        ]
        train_dataset = _subset_dataset(dataset, train_observations)
        trial_dataset, trial_transform, _, _ = _optimize_dataset(
            train_dataset,
            config,
            primary_transform,
        )
        if len(trial_dataset.observations) < config.min_pair_count:
            continue
        holdout_rms = _optimize_holdout_board_pose(
            holdout_observation,
            dataset,
            trial_transform,
        )
        delta = transform_delta_metrics(primary_transform, trial_transform)
        trials.append(
            {
                "holdout_pose_id": holdout_observation.pose_id,
                "train_pair_count": len(train_dataset.observations),
                "holdout_rms_px": float(holdout_rms),
                "delta_to_primary": delta,
            }
        )
        trial_transforms.append(np.asarray(trial_transform, dtype=float))
    if not trials:
        return None
    return {
        "trials": trials,
        "holdout_rms_px": float_list_summary(
            [float(trial["holdout_rms_px"]) for trial in trials]
        ),
        "delta_translation_norm_m": float_list_summary(
            [float(trial["delta_to_primary"]["translation_norm_m"]) for trial in trials]
        ),
        "delta_rotation_deg": float_list_summary(
            [float(trial["delta_to_primary"]["rotation_deg"]) for trial in trials]
        ),
        "distinct_solution_count": (
            _cluster_transform_count(
                trial_transforms,
                translation_threshold_m=(
                    config.metrics_warning_repeatability_translation_m
                ),
                rotation_threshold_deg=(
                    config.metrics_warning_repeatability_rotation_deg
                ),
            )
            if trial_transforms
            else 0
        ),
    }


def _evaluate_dataset(
    dataset: StereoCalibrationDataset,
    config: StereoCalibrationConfig,
    *,
    initial_transform: np.ndarray,
    final_transform: np.ndarray,
    board_transforms: list[np.ndarray],
) -> dict[str, Any]:
    initial_board_transforms = [
        np.asarray(
            observation.metadata["initial_parent_board_transform_matrix"], dtype=float
        )
        for observation in dataset.observations
    ]
    initial_params = _pack_params(initial_transform, initial_board_transforms)
    initial_rms = _rms_from_residual_vector(
        _bundle_adjustment_residuals(initial_params, dataset)
    )
    final_params = _pack_params(final_transform, board_transforms)
    final_rms = _rms_from_residual_vector(
        _bundle_adjustment_residuals(final_params, dataset)
    )
    per_pair_rows, fundamental = _evaluate_pair_rows(
        dataset, final_transform, board_transforms
    )
    return {
        "initial_rms_px": float(initial_rms),
        "final_rms_px": float(final_rms),
        "per_pair_rows": per_pair_rows,
        "parent_image_coverage": _image_coverage_metrics(
            dataset.observations, "parent"
        ),
        "child_image_coverage": _image_coverage_metrics(dataset.observations, "child"),
        "pose_diversity": _pose_diversity_metrics(per_pair_rows),
        "leave_one_out_repeatability": _leave_one_out_repeatability(
            dataset,
            config,
            final_transform,
        ),
        "fundamental_matrix": None if fundamental is None else fundamental.tolist(),
    }


def run_reference_calibration_from_config(
    config_path: str,
    *,
    output_dir_override: str | None = None,
) -> dict[str, Any]:
    _, config, camera_data, output_dir = _load_config(
        config_path,
        output_dir_override=output_dir_override,
    )
    dataset, extraction_report, initial_transform = _build_dataset(
        config=config,
        camera_data=camera_data,
    )
    if not bool(extraction_report.get("ready_for_optimization", True)):
        write_failure_outputs(
            output_dir,
            dataset=dataset,
            extraction_report=extraction_report,
            failure_message=str(extraction_report.get("failure_message")),
        )
        raise SystemExit(str(extraction_report.get("failure_message")))
    optimized_dataset, final_transform, board_transforms, optimization_report = (
        _optimize_dataset(
            dataset,
            config,
            initial_transform,
        )
    )
    extraction_report["final_inlier_pair_count"] = int(
        len(optimized_dataset.observations)
    )
    extraction_report["final_inlier_pair_ratio"] = float(
        len(optimized_dataset.observations)
        / max(
            (extraction_report.get("pairing_summary") or {}).get("paired_count", 1), 1
        )
    )
    evaluation = _evaluate_dataset(
        optimized_dataset,
        config,
        initial_transform=initial_transform,
        final_transform=final_transform,
        board_transforms=board_transforms,
    )
    metrics_output = build_metrics_output(
        dataset=optimized_dataset,
        config=config,
        initial_transform=initial_transform,
        final_transform=final_transform,
        extraction_report=extraction_report,
        optimization_report=optimization_report,
        evaluation=evaluation,
    )
    manifest = write_outputs(
        output_dir,
        dataset=optimized_dataset,
        initial_transform=initial_transform,
        final_transform=final_transform,
        extraction_report=extraction_report,
        optimization_report=optimization_report,
        evaluation=evaluation,
        metrics_output=metrics_output,
    )
    return {
        "dataset": optimized_dataset,
        "initial_transform": initial_transform,
        "final_transform": final_transform,
        "extraction_report": extraction_report,
        "optimization_report": optimization_report,
        "evaluation": evaluation,
        "metrics_output": metrics_output,
        "manifest": manifest,
    }
