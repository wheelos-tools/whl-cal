from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

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
    leave_one_out_status = "unknown"
    holdout_rms_p95 = None
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
    }

    if all(
        statuses[key] == "pass"
        for key in ("pose_count", "reprojection", "pose_reprojection")
    ) and leave_one_out_status in {"pass", "unknown"}:
        recommendation = "accepted_reference_candidate"
    elif statuses["pose_count"] == "warning":
        recommendation = "recollect_data"
    elif leave_one_out_status == "warning":
        recommendation = "repeatability_review"
    else:
        recommendation = "reference_quality_review"

    delta_to_initial = transform_delta_metrics(initial_transform, final_transform)
    final_euler = R.from_matrix(final_transform[:3, :3]).as_euler("ZYX", degrees=True)
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
        },
        "coarse_metrics": {
            "pose_count": pose_count,
            "initial_rms_px": float(initial_rms_px),
            "final_rms_px": float(final_rms_px),
            "pose_reprojection_rms_p95_px": (
                None
                if not pose_rms_values
                else float(np.percentile(np.asarray(pose_rms_values, dtype=float), 95))
            ),
            "holdout_reprojection_rms_p95_px": holdout_rms_p95,
            "statuses": statuses,
        },
        "camera_calibration_assessment": {
            "recommendation": recommendation,
            "pose_count": statuses["pose_count"],
            "reprojection": statuses["reprojection"],
            "pose_reprojection": statuses["pose_reprojection"],
            "holdout_pose": statuses["holdout_pose"],
            "pose_repeatability": leave_one_out_status,
            "delta_to_initial": delta_to_initial,
            "leave_one_out_details": leave_one_out,
            "uncertainty_summary": uncertainty_summary,
        },
        "fine_metrics": {
            "per_pose_reprojection": per_pose_rms_px,
            "leave_one_out_repeatability": leave_one_out,
            "uncertainty_summary": uncertainty_summary,
            "extraction": extraction_report,
            "optimization": optimization_report,
            "artifacts": {
                "output_dir": output_dir,
                "diagnostics_dir": f"{output_dir}/diagnostics",
            },
        },
    }
    return metrics_output
