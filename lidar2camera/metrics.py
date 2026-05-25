from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from calibration_common.evaluation import build_final_acceptance
from lidar2camera.models import ReferenceCalibrationConfig, ReferenceCalibrationDataset


def _float_list_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    series = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(series)),
        "std": float(np.std(series)),
        "min": float(np.min(series)),
        "p50": float(np.percentile(series, 50)),
        "p95": float(np.percentile(series, 95)),
        "max": float(np.max(series)),
    }


def transform_delta_metrics(
    reference_transform: np.ndarray, candidate_transform: np.ndarray
) -> dict[str, Any]:
    delta_rotation = R.from_matrix(reference_transform[:3, :3]).inv() * R.from_matrix(
        candidate_transform[:3, :3]
    )
    delta_translation = candidate_transform[:3, 3] - reference_transform[:3, 3]
    return {
        "translation_norm_m": float(np.linalg.norm(delta_translation)),
        "translation_xyz_m": {
            "x": float(delta_translation[0]),
            "y": float(delta_translation[1]),
            "z": float(delta_translation[2]),
        },
        "rotation_deg": float(np.degrees(delta_rotation.magnitude())),
    }


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


def _image_coverage_metrics(
    dataset: ReferenceCalibrationDataset,
) -> dict[str, Any] | None:
    if not dataset.observations:
        return None
    center_x = []
    center_y = []
    edge_margins_px = []
    bbox_area_ratio = []
    grid_counts = [[0, 0, 0] for _ in range(3)]
    per_pose = []
    for observation in dataset.observations:
        image_points = np.asarray(observation.image_points, dtype=float)
        if image_points.size == 0:
            continue
        width, height = observation.image_size_wh
        bbox_min = np.min(image_points, axis=0)
        bbox_max = np.max(image_points, axis=0)
        center = np.mean(image_points, axis=0)
        normalized_x = float(center[0] / max(width, 1))
        normalized_y = float(center[1] / max(height, 1))
        cell_x = min(2, max(0, int(normalized_x * 3.0)))
        cell_y = min(2, max(0, int(normalized_y * 3.0)))
        grid_counts[cell_y][cell_x] += 1
        margin = float(
            min(
                bbox_min[0],
                bbox_min[1],
                max(width - bbox_max[0], 0.0),
                max(height - bbox_max[1], 0.0),
            )
        )
        area_ratio = float(
            max((bbox_max[0] - bbox_min[0]), 0.0)
            * max((bbox_max[1] - bbox_min[1]), 0.0)
            / max(width * height, 1)
        )
        center_x.append(normalized_x)
        center_y.append(normalized_y)
        edge_margins_px.append(margin)
        bbox_area_ratio.append(area_ratio)
        per_pose.append(
            {
                "pose_id": observation.pose_id,
                "center_xy_normalized": {
                    "x": normalized_x,
                    "y": normalized_y,
                },
                "grid_cell": {"x": cell_x, "y": cell_y},
                "edge_margin_px": margin,
                "bbox_area_ratio": area_ratio,
            }
        )
    if not per_pose:
        return None
    occupied_cell_count = sum(
        1 for row in grid_counts for count in row if int(count) > 0
    )
    return {
        "occupied_cell_count": int(occupied_cell_count),
        "grid_counts": grid_counts,
        "horizontal_span_ratio": float(max(center_x) - min(center_x)),
        "vertical_span_ratio": float(max(center_y) - min(center_y)),
        "edge_margin_px": _float_list_summary(edge_margins_px),
        "bbox_area_ratio": _float_list_summary(bbox_area_ratio),
        "per_pose": per_pose,
    }


def _image_coverage_status(
    coverage: dict[str, Any] | None,
    config: ReferenceCalibrationConfig,
) -> str:
    if coverage is None:
        return "warning"
    if int(coverage["occupied_cell_count"]) < int(
        config.metrics_warning_image_coverage_min_cells
    ):
        return "warning"
    if float(coverage["horizontal_span_ratio"]) < float(
        config.metrics_warning_image_horizontal_span_ratio
    ):
        return "warning"
    if float(coverage["vertical_span_ratio"]) < float(
        config.metrics_warning_image_vertical_span_ratio
    ):
        return "warning"
    return "pass"


def _pose_diversity_metrics(
    final_transform: np.ndarray,
    dataset: ReferenceCalibrationDataset,
) -> dict[str, Any] | None:
    if not dataset.observations:
        return None
    depths = []
    tilts = []
    per_pose = []
    rotation = final_transform[:3, :3]
    translation = final_transform[:3, 3]
    for observation in dataset.observations:
        camera_points = (rotation @ observation.object_points.T).T + translation
        if camera_points.size == 0:
            continue
        board_center = np.mean(camera_points, axis=0)
        centered = camera_points - board_center
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = np.asarray(vh[-1], dtype=float)
        normal /= max(np.linalg.norm(normal), 1e-12)
        tilt_deg = float(np.degrees(np.arccos(np.clip(np.abs(normal[2]), 0.0, 1.0))))
        depth_m = float(board_center[2])
        depths.append(depth_m)
        tilts.append(tilt_deg)
        per_pose.append(
            {
                "pose_id": observation.pose_id,
                "board_center_camera_m": {
                    "x": float(board_center[0]),
                    "y": float(board_center[1]),
                    "z": depth_m,
                },
                "board_tilt_deg": tilt_deg,
            }
        )
    if not per_pose:
        return None
    return {
        "board_center_depth_m": _float_list_summary(depths),
        "board_tilt_deg": _float_list_summary(tilts),
        "depth_span_m": float(max(depths) - min(depths)),
        "tilt_span_deg": float(max(tilts) - min(tilts)),
        "per_pose": per_pose,
    }


def _pose_diversity_status(
    diversity: dict[str, Any] | None,
    config: ReferenceCalibrationConfig,
) -> str:
    if diversity is None:
        return "warning"
    if float(diversity["depth_span_m"]) < float(config.metrics_warning_depth_span_m):
        return "warning"
    if float(diversity["tilt_span_deg"]) < float(config.metrics_warning_tilt_span_deg):
        return "warning"
    return "pass"


def _board_geometry_metrics(
    dataset: ReferenceCalibrationDataset,
) -> dict[str, Any] | None:
    expected_extent = dataset.metadata.get("board_template_extent_xy_m", {}) or {}
    expected_x = float(expected_extent.get("x", 0.0) or 0.0)
    expected_y = float(expected_extent.get("y", 0.0) or 0.0)
    plane_residuals = []
    plane_inliers = []
    ratio_x = []
    ratio_y = []
    quality_warning_count = 0
    per_pose = []
    for observation in dataset.observations:
        diagnostics = observation.metadata.get("board_diagnostics", {}) or {}
        residual = diagnostics.get("plane_residual_rmse_m")
        if residual is not None:
            plane_residuals.append(float(residual))
        inlier_count = diagnostics.get("plane_inlier_count")
        if inlier_count is not None:
            plane_inliers.append(float(inlier_count))
        extents = diagnostics.get("board_extent_xy_m", {}) or {}
        observed_x = extents.get("x")
        observed_y = extents.get("y")
        ratio_value_x = None
        ratio_value_y = None
        if observed_x is not None and expected_x > 1e-9:
            ratio_value_x = float(observed_x) / expected_x
            ratio_x.append(ratio_value_x)
        if observed_y is not None and expected_y > 1e-9:
            ratio_value_y = float(observed_y) / expected_y
            ratio_y.append(ratio_value_y)
        warnings = diagnostics.get("quality_warnings", []) or []
        quality_warning_count += len(warnings)
        per_pose.append(
            {
                "pose_id": observation.pose_id,
                "plane_residual_rmse_m": residual,
                "plane_inlier_count": inlier_count,
                "board_extent_ratio_xy": {
                    "x": ratio_value_x,
                    "y": ratio_value_y,
                },
                "quality_warnings": list(warnings),
            }
        )
    if not per_pose:
        return None
    return {
        "plane_residual_rmse_m": _float_list_summary(plane_residuals),
        "plane_inlier_count": _float_list_summary(plane_inliers),
        "board_extent_ratio_x": _float_list_summary(ratio_x),
        "board_extent_ratio_y": _float_list_summary(ratio_y),
        "quality_warning_count": int(quality_warning_count),
        "per_pose": per_pose,
    }


def _board_geometry_status(
    geometry: dict[str, Any] | None,
    config: ReferenceCalibrationConfig,
) -> str:
    if geometry is None:
        return "warning"
    residual_summary = geometry.get("plane_residual_rmse_m") or {}
    if residual_summary and float(residual_summary.get("p95", 0.0)) > float(
        config.metrics_warning_plane_residual_rmse_m
    ):
        return "warning"
    for key in ("board_extent_ratio_x", "board_extent_ratio_y"):
        summary = geometry.get(key) or {}
        if not summary:
            continue
        if float(summary.get("max", 0.0)) > float(
            config.metrics_warning_board_extent_ratio_max
        ):
            return "warning"
        if float(summary.get("min", 1.0)) < float(
            config.metrics_warning_board_extent_ratio_min
        ):
            return "warning"
    if int(geometry.get("quality_warning_count", 0)) > 0:
        return "warning"
    return "pass"


def _build_final_acceptance(
    *,
    pose_count: int,
    config: ReferenceCalibrationConfig,
    coarse_metrics: dict[str, Any],
    assessment: dict[str, Any],
    extraction_report: dict[str, Any],
    optimization_report: dict[str, Any],
) -> dict[str, Any]:
    statuses = coarse_metrics["statuses"]
    pairing_summary = extraction_report.get("pairing_summary", {}) or {}
    paired_count = int(pairing_summary.get("paired_count", 0))
    accepted_pose_count = int(extraction_report.get("accepted_pose_count", pose_count))
    rejected_pose_count = int(extraction_report.get("rejected_pose_count", 0))
    accepted_pair_ratio = float(extraction_report.get("accepted_pair_ratio", 0.0))
    optimization = optimization_report.get("optimization", {}) or {}
    gates = [
        {
            "name": "paired_pose_count",
            "status": "pass" if paired_count >= int(config.min_pose_count) else "fail",
            "severity": "required",
            "evidence": f"paired_count={paired_count}",
            "action": "Collect more synchronized image/PCD pairs before attempting release calibration.",
        },
        {
            "name": "accepted_pose_count",
            "status": "pass" if pose_count >= int(config.min_pose_count) else "fail",
            "severity": "required",
            "evidence": f"accepted_pose_count={accepted_pose_count}, rejected_pose_count={rejected_pose_count}",
            "action": "Recollect more valid board poses or reduce extraction failures before promotion.",
        },
        {
            "name": "accepted_pair_ratio",
            "status": statuses["extraction_yield"],
            "severity": "required",
            "evidence": f"accepted_pair_ratio={accepted_pair_ratio}",
            "action": "If most paired samples are rejected, fix the collection workflow before trusting the run.",
        },
        {
            "name": "optimization_success",
            "status": "pass" if bool(optimization.get("success")) else "fail",
            "severity": "required",
            "evidence": f"solver_success={optimization.get('success')}, message={optimization.get('message')}",
            "action": "Do not trust the extrinsic if the nonlinear solver failed; fix initialization or data quality first.",
        },
        {
            "name": "final_reprojection",
            "status": statuses["reprojection"],
            "severity": "required",
            "evidence": f"final_rms_px={coarse_metrics['final_rms_px']}",
            "action": "Inspect intrinsics, target geometry, and point-to-corner correspondences before release.",
        },
        {
            "name": "per_pose_reprojection",
            "status": statuses["pose_reprojection"],
            "severity": "required",
            "evidence": (
                "pose_reprojection_rms_p95_px="
                f"{coarse_metrics['pose_reprojection_rms_p95_px']}"
            ),
            "action": "Reject runs with weak pose tails; improve pose diversity and outlier filtering.",
        },
        {
            "name": "holdout_reprojection",
            "status": statuses["holdout_pose"],
            "severity": "required",
            "evidence": (
                "holdout_reprojection_rms_p95_px="
                f"{coarse_metrics['holdout_reprojection_rms_p95_px']}"
            ),
            "action": "Require held-out pose agreement before treating the result as production-ready.",
        },
        {
            "name": "pose_repeatability",
            "status": statuses["pose_repeatability"],
            "severity": "required",
            "evidence": f"leave_one_out_status={statuses['pose_repeatability']}",
            "action": "Collect wider board coverage or improve LiDAR-side board geometry extraction before release.",
        },
        {
            "name": "image_coverage",
            "status": statuses["image_coverage"],
            "severity": "required",
            "evidence": (
                "occupied_cells="
                f"{coarse_metrics['image_coverage_occupied_cell_count']}, "
                "horizontal_span_ratio="
                f"{coarse_metrics['image_coverage_horizontal_span_ratio']}, "
                "vertical_span_ratio="
                f"{coarse_metrics['image_coverage_vertical_span_ratio']}"
            ),
            "action": "Collect board views that cover left/right/up/down image regions before release.",
        },
        {
            "name": "pose_diversity",
            "status": statuses["pose_diversity"],
            "severity": "required",
            "evidence": (
                f"depth_span_m={coarse_metrics['board_depth_span_m']}, "
                f"tilt_span_deg={coarse_metrics['board_tilt_span_deg']}"
            ),
            "action": "Increase board depth and tilt diversity so the solution is not dominated by one pose family.",
        },
        {
            "name": "board_geometry",
            "status": statuses["board_geometry"],
            "severity": "required",
            "evidence": (
                f"plane_residual_rmse_p95_m={coarse_metrics['plane_residual_rmse_p95_m']}, "
                f"board_extent_ratio_x_p95={coarse_metrics['board_extent_ratio_x_p95']}, "
                f"board_extent_ratio_y_p95={coarse_metrics['board_extent_ratio_y_p95']}"
            ),
            "action": "Do not release runs where LiDAR board support looks like a wall or an under-constrained plane patch.",
        },
    ]
    return build_final_acceptance(
        module="lidar2camera",
        gates=gates,
        pass_recommendation="release_reference_extrinsics",
        review_recommendation="review_metrics_and_overlay",
        fail_recommendation="reject_and_recollect_reference_data",
    )


def build_metrics_output(
    dataset: ReferenceCalibrationDataset,
    config: ReferenceCalibrationConfig,
    *,
    initial_transform: np.ndarray,
    final_transform: np.ndarray,
    initial_rms_px: float,
    final_rms_px: float,
    per_pose_rms_px: list[dict[str, Any]],
    leave_one_out: dict | None,
    uncertainty_summary: dict | None,
    extraction_report: dict,
    optimization_report: dict,
    output_dir: str,
) -> dict:
    pose_rms_values = [float(item["rms_px"]) for item in per_pose_rms_px]
    pose_count = len(dataset.observations)
    coverage = _image_coverage_metrics(dataset)
    pose_diversity = _pose_diversity_metrics(final_transform, dataset)
    board_geometry = _board_geometry_metrics(dataset)
    leave_one_out_status = "unknown"
    holdout_rms_p95 = None
    paired_count = int(
        (extraction_report.get("pairing_summary", {}) or {}).get("paired_count", 0)
    )
    accepted_pair_ratio = float(
        extraction_report.get(
            "accepted_pair_ratio",
            0.0 if paired_count <= 0 else pose_count / paired_count,
        )
    )
    if leave_one_out is not None:
        leave_one_out_status = str(leave_one_out.get("status", "unknown"))
        holdout_rms_p95 = (leave_one_out.get("holdout_rms_summary") or {}).get("p95")

    statuses = {
        "pose_count": (
            "pass" if pose_count >= int(config.min_pose_count) else "warning"
        ),
        "reprojection": (
            "pass"
            if float(final_rms_px) <= float(config.metrics_warning_final_rms_px)
            else "warning"
        ),
        "pose_reprojection": (
            "pass"
            if pose_rms_values
            and float(np.percentile(np.asarray(pose_rms_values, dtype=float), 95))
            <= float(config.metrics_warning_pose_rms_p95_px)
            else "warning"
        ),
        "holdout_pose": (
            "unknown"
            if holdout_rms_p95 is None
            else (
                "pass"
                if float(holdout_rms_p95)
                <= float(config.metrics_warning_holdout_rms_px)
                else "warning"
            )
        ),
        "pose_repeatability": leave_one_out_status,
        "extraction_yield": (
            "pass"
            if accepted_pair_ratio >= float(config.metrics_warning_accepted_pair_ratio)
            else "warning"
        ),
        "image_coverage": _image_coverage_status(coverage, config),
        "pose_diversity": _pose_diversity_status(pose_diversity, config),
        "board_geometry": _board_geometry_status(board_geometry, config),
    }

    if (
        all(
            statuses[key] == "pass"
            for key in (
                "pose_count",
                "extraction_yield",
                "reprojection",
                "pose_reprojection",
                "image_coverage",
                "pose_diversity",
                "board_geometry",
            )
        )
        and leave_one_out_status == "pass"
    ):
        recommendation = "accepted_reference_candidate"
    elif statuses["pose_count"] == "warning":
        recommendation = "recollect_data"
    elif (
        statuses["extraction_yield"] == "warning"
        or statuses["image_coverage"] == "warning"
        or statuses["pose_diversity"] == "warning"
        or statuses["board_geometry"] == "warning"
    ):
        recommendation = "recollect_data"
    elif (
        statuses["holdout_pose"] == "unknown"
        or statuses["pose_repeatability"] == "unknown"
    ):
        recommendation = "recollect_data"
    elif leave_one_out_status == "warning":
        recommendation = "repeatability_review"
    else:
        recommendation = "reference_quality_review"

    delta_to_initial = transform_delta_metrics(initial_transform, final_transform)
    final_euler = R.from_matrix(final_transform[:3, :3]).as_euler("ZYX", degrees=True)
    coarse_metrics = {
        "pose_count": pose_count,
        "initial_rms_px": float(initial_rms_px),
        "final_rms_px": float(final_rms_px),
        "pose_reprojection_rms_p95_px": (
            None
            if not pose_rms_values
            else float(np.percentile(np.asarray(pose_rms_values, dtype=float), 95))
        ),
        "holdout_reprojection_rms_p95_px": holdout_rms_p95,
        "accepted_pair_ratio": accepted_pair_ratio,
        "image_coverage_occupied_cell_count": (
            None if coverage is None else int(coverage["occupied_cell_count"])
        ),
        "image_coverage_horizontal_span_ratio": (
            None if coverage is None else float(coverage["horizontal_span_ratio"])
        ),
        "image_coverage_vertical_span_ratio": (
            None if coverage is None else float(coverage["vertical_span_ratio"])
        ),
        "board_depth_span_m": (
            None if pose_diversity is None else float(pose_diversity["depth_span_m"])
        ),
        "board_tilt_span_deg": (
            None if pose_diversity is None else float(pose_diversity["tilt_span_deg"])
        ),
        "plane_residual_rmse_p95_m": (
            None
            if board_geometry is None
            or board_geometry.get("plane_residual_rmse_m") is None
            else float(board_geometry["plane_residual_rmse_m"]["p95"])
        ),
        "board_extent_ratio_x_p95": (
            None
            if board_geometry is None
            or board_geometry.get("board_extent_ratio_x") is None
            else float(board_geometry["board_extent_ratio_x"]["p95"])
        ),
        "board_extent_ratio_y_p95": (
            None
            if board_geometry is None
            or board_geometry.get("board_extent_ratio_y") is None
            else float(board_geometry["board_extent_ratio_y"]["p95"])
        ),
        "statuses": statuses,
    }
    assessment = {
        "recommendation": recommendation,
        "pose_count": statuses["pose_count"],
        "extraction_yield": statuses["extraction_yield"],
        "reprojection": statuses["reprojection"],
        "pose_reprojection": statuses["pose_reprojection"],
        "holdout_pose": statuses["holdout_pose"],
        "pose_repeatability": leave_one_out_status,
        "delta_to_initial": delta_to_initial,
        "leave_one_out_details": leave_one_out,
        "uncertainty_summary": uncertainty_summary,
        "image_coverage": coverage,
        "pose_diversity": pose_diversity,
        "board_geometry": board_geometry,
    }
    final_acceptance = _build_final_acceptance(
        pose_count=pose_count,
        config=config,
        coarse_metrics=coarse_metrics,
        assessment=assessment,
        extraction_report=extraction_report,
        optimization_report=optimization_report,
    )
    metrics_output = {
        "summary": {
            "parent_frame": dataset.parent_frame,
            "child_frame": dataset.child_frame,
            "final_translation_m": {
                "x": float(final_transform[0, 3]),
                "y": float(final_transform[1, 3]),
                "z": float(final_transform[2, 3]),
            },
            "final_euler_deg": {
                "yaw": float(final_euler[0]),
                "pitch": float(final_euler[1]),
                "roll": float(final_euler[2]),
            },
            "delta_to_initial": delta_to_initial,
            "pose_count": pose_count,
            "initial_rms_px": float(initial_rms_px),
            "final_rms_px": float(final_rms_px),
            "final_acceptance_status": final_acceptance["status"],
            "release_ready": final_acceptance["release_ready"],
        },
        "coarse_metrics": coarse_metrics,
        "final_acceptance": final_acceptance,
        "camera_calibration_assessment": assessment,
        "fine_metrics": {
            "per_pose_reprojection": per_pose_rms_px,
            "leave_one_out_repeatability": leave_one_out,
            "uncertainty_summary": uncertainty_summary,
            "image_coverage": coverage,
            "pose_diversity": pose_diversity,
            "board_geometry": board_geometry,
            "extraction": extraction_report,
            "optimization": optimization_report,
            "artifacts": {
                "output_dir": output_dir,
                "diagnostics_dir": f"{output_dir}/diagnostics",
            },
        },
    }
    return metrics_output
