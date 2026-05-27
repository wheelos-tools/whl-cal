from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from calibration_common.evaluation import build_final_acceptance
from camera2camera.models import (StereoCalibrationConfig,
                                  StereoCalibrationDataset)


def float_list_summary(values: list[float]) -> dict[str, float] | None:
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


def _image_coverage_status(
    coverage: dict[str, Any] | None, config: StereoCalibrationConfig
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


def _pose_diversity_status(
    diversity: dict[str, Any] | None, config: StereoCalibrationConfig
) -> str:
    if diversity is None:
        return "warning"
    if float(diversity["depth_span_m"]) < float(config.metrics_warning_depth_span_m):
        return "warning"
    if float(diversity["tilt_span_deg"]) < float(config.metrics_warning_tilt_span_deg):
        return "warning"
    return "pass"


def _repeatability_status(
    holdout_report: dict[str, Any] | None, config: StereoCalibrationConfig
) -> str:
    if not holdout_report or not holdout_report.get("trials"):
        return "unknown"
    translation_summary = holdout_report.get("delta_translation_norm_m") or {}
    rotation_summary = holdout_report.get("delta_rotation_deg") or {}
    if float(translation_summary.get("p95", float("inf"))) > float(
        config.metrics_warning_repeatability_translation_m
    ):
        return "warning"
    if float(rotation_summary.get("p95", float("inf"))) > float(
        config.metrics_warning_repeatability_rotation_deg
    ):
        return "warning"
    return "pass"


def _holdout_status(
    holdout_report: dict[str, Any] | None, config: StereoCalibrationConfig
) -> str:
    if not holdout_report or not holdout_report.get("holdout_rms_px"):
        return "unknown"
    if float(
        (holdout_report.get("holdout_rms_px") or {}).get("p95", float("inf"))
    ) > float(config.metrics_warning_holdout_rms_px):
        return "warning"
    return "pass"


def build_metrics_output(
    *,
    dataset: StereoCalibrationDataset,
    config: StereoCalibrationConfig,
    initial_transform: np.ndarray,
    final_transform: np.ndarray,
    extraction_report: dict[str, Any],
    optimization_report: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    per_pair_rows = list(evaluation.get("per_pair_rows", []))
    per_pair_rms = [float(row["combined_rms_px"]) for row in per_pair_rows]
    epipolar_means = [float(row["epipolar_mean_px"]) for row in per_pair_rows]
    final_rms = float(evaluation.get("final_rms_px", 0.0))
    initial_rms = float(evaluation.get("initial_rms_px", 0.0))
    accepted_pair_ratio = float(extraction_report.get("final_inlier_pair_ratio", 0.0))
    holdout_report = evaluation.get("leave_one_out_repeatability")
    parent_coverage = evaluation.get("parent_image_coverage")
    child_coverage = evaluation.get("child_image_coverage")
    pose_diversity = evaluation.get("pose_diversity")
    paired_count_value = int(
        (extraction_report.get("pairing_summary") or {}).get("paired_count", 0)
    )
    holdout_translation_summary = (
        holdout_report.get("delta_translation_norm_m") if holdout_report else None
    )
    holdout_rotation_summary = (
        holdout_report.get("delta_rotation_deg") if holdout_report else None
    )
    repeatability_status = _repeatability_status(holdout_report, config)
    holdout_status = _holdout_status(holdout_report, config)
    parent_coverage_status = _image_coverage_status(parent_coverage, config)
    child_coverage_status = _image_coverage_status(child_coverage, config)
    pose_diversity_status = _pose_diversity_status(pose_diversity, config)
    ordering_resolution = extraction_report.get("ordering_resolution") or {}
    ordering_changed_count = int(ordering_resolution.get("changed_pair_count", 0))
    initial_to_final = transform_delta_metrics(initial_transform, final_transform)
    statuses = {
        "pair_count": (
            "pass"
            if len(dataset.observations) >= int(config.min_pair_count)
            else "warning"
        ),
        "accepted_pair_ratio": (
            "pass"
            if accepted_pair_ratio >= float(config.metrics_warning_accepted_pair_ratio)
            else "warning"
        ),
        "optimization_success": (
            "pass" if bool(optimization_report.get("success")) else "fail"
        ),
        "reprojection": (
            "pass"
            if final_rms <= float(config.metrics_warning_final_rms_px)
            else "warning"
        ),
        "pair_reprojection": (
            (
                "pass"
                if float(np.percentile(per_pair_rms, 95))
                <= float(config.metrics_warning_pair_rms_p95_px)
                else "warning"
            )
            if per_pair_rms
            else "warning"
        ),
        "holdout_reprojection": holdout_status,
        "repeatability": repeatability_status,
        "epipolar": (
            "pass"
            if epipolar_means
            and float(np.percentile(epipolar_means, 95))
            <= float(config.metrics_warning_epipolar_p95_px)
            else "warning"
        ),
        "parent_image_coverage": parent_coverage_status,
        "child_image_coverage": child_coverage_status,
        "pose_diversity": pose_diversity_status,
        "ordering_consistency": "pass" if ordering_changed_count <= 0 else "warning",
    }
    coarse_metrics = {
        "pair_count": int(len(dataset.observations)),
        "accepted_pair_ratio": float(accepted_pair_ratio),
        "initial_rms_px": float(initial_rms),
        "final_rms_px": float(final_rms),
        "pair_reprojection_rms_px": float_list_summary(per_pair_rms),
        "epipolar_error_px": float_list_summary(epipolar_means),
        "holdout_reprojection_rms_px": (
            (holdout_report or {}).get("holdout_rms_px") if holdout_report else None
        ),
        "parent_image_coverage": parent_coverage,
        "child_image_coverage": child_coverage,
        "pose_diversity": pose_diversity,
        "statuses": statuses,
    }
    gates = [
        {
            "name": "paired_pair_count",
            "status": (
                "pass"
                if int(
                    (extraction_report.get("pairing_summary") or {}).get(
                        "paired_count", 0
                    )
                )
                > 0
                else "fail"
            ),
            "severity": "required",
            "evidence": f"paired_count={paired_count_value}",
            "action": "Verify both camera directories contain matched image stems.",
        },
        {
            "name": "accepted_pair_count",
            "status": statuses["pair_count"],
            "severity": "required",
            "evidence": (
                f"accepted_pairs={len(dataset.observations)} "
                f"min_pairs={config.min_pair_count}"
            ),
            "action": (
                "Collect more multi-pose stereo board pairs before trusting the result."
            ),
        },
        {
            "name": "optimization_success",
            "status": statuses["optimization_success"],
            "severity": "required",
            "evidence": f"success={bool(optimization_report.get('success'))}",
            "action": (
                "Fix the extraction or initialization before retrying optimization."
            ),
        },
        {
            "name": "final_reprojection",
            "status": statuses["reprojection"],
            "severity": "required",
            "evidence": (
                f"final_rms_px={final_rms:.3f} "
                f"threshold={config.metrics_warning_final_rms_px:.3f}"
            ),
            "action": (
                "Inspect the highest-residual pairs and recollect cleaner board "
                "observations."
            ),
        },
        {
            "name": "per_pair_reprojection",
            "status": statuses["pair_reprojection"],
            "severity": "required",
            "evidence": (
                "pair_rms_p95_px="
                f"{float((float_list_summary(per_pair_rms) or {}).get('p95', 0.0)):.3f}"
            ),
            "action": (
                "Inspect per_pair_reprojection.csv for tail samples and reject weak "
                "views."
            ),
        },
        {
            "name": "holdout_reprojection",
            "status": statuses["holdout_reprojection"],
            "severity": "required",
            "evidence": (
                "holdout_rms="
                f"{holdout_report.get('holdout_rms_px') if holdout_report else None}"
            ),
            "action": (
                "Treat the run as review-only until leave-one-out holdout error is "
                "stable."
            ),
        },
        {
            "name": "pose_repeatability",
            "status": statuses["repeatability"],
            "severity": "required",
            "evidence": (
                "translation="
                f"{holdout_translation_summary}, "
                "rotation="
                f"{holdout_rotation_summary}"
            ),
            "action": (
                "Recollect more diverse stereo poses when leave-one-out extrinsics "
                "disagree."
            ),
        },
        {
            "name": "accepted_pair_ratio",
            "status": statuses["accepted_pair_ratio"],
            "severity": "required",
            "evidence": (
                f"accepted_pair_ratio={accepted_pair_ratio:.3f} "
                f"threshold={config.metrics_warning_accepted_pair_ratio:.3f}"
            ),
            "action": (
                "Improve acquisition discipline instead of trusting a run with many "
                "rejected pairs."
            ),
        },
        {
            "name": "epipolar_error",
            "status": statuses["epipolar"],
            "severity": "required",
            "evidence": (
                "epipolar_error_p95_px="
                f"{(float_list_summary(epipolar_means) or {}).get('p95')}"
            ),
            "action": (
                "Inspect epipolar previews and per-pair residuals before promoting "
                "the extrinsic."
            ),
        },
        {
            "name": "parent_image_coverage",
            "status": statuses["parent_image_coverage"],
            "severity": "required",
            "evidence": str(parent_coverage),
            "action": "Move the board across more parent-camera image regions.",
        },
        {
            "name": "child_image_coverage",
            "status": statuses["child_image_coverage"],
            "severity": "required",
            "evidence": str(child_coverage),
            "action": "Move the board across more child-camera image regions.",
        },
        {
            "name": "pose_diversity",
            "status": statuses["pose_diversity"],
            "severity": "required",
            "evidence": str(pose_diversity),
            "action": (
                "Collect more depth and tilt variation instead of only fronto-parallel "
                "board views."
            ),
        },
        {
            "name": "ordering_consistency",
            "status": statuses["ordering_consistency"],
            "severity": "advisory",
            "evidence": f"changed_pair_count={ordering_changed_count}",
            "action": (
                "Prefer ChArUco or add more asymmetric board poses if checkerboard "
                "ordering frequently flips."
            ),
        },
    ]
    final_acceptance = build_final_acceptance(
        module="camera2camera",
        gates=gates,
        pass_recommendation="release_stereo_extrinsics",
        review_recommendation="review_camera2camera_diagnostics",
        fail_recommendation="reject_and_recollect_camera2camera_pairs",
    )
    summary = {
        "pair_count": int(len(dataset.observations)),
        "initial_rms_px": float(initial_rms),
        "final_rms_px": float(final_rms),
        "delta_to_initial": initial_to_final,
        "final_translation_m": {
            "x": float(final_transform[0, 3]),
            "y": float(final_transform[1, 3]),
            "z": float(final_transform[2, 3]),
        },
        "final_euler_deg": {
            "roll": float(
                R.from_matrix(final_transform[:3, :3]).as_euler("xyz", degrees=True)[0]
            ),
            "pitch": float(
                R.from_matrix(final_transform[:3, :3]).as_euler("xyz", degrees=True)[1]
            ),
            "yaw": float(
                R.from_matrix(final_transform[:3, :3]).as_euler("xyz", degrees=True)[2]
            ),
        },
        "final_acceptance_status": final_acceptance["status"],
        "release_ready": final_acceptance["release_ready"],
    }
    return {
        "summary": summary,
        "coarse_metrics": coarse_metrics,
        "final_acceptance": final_acceptance,
        "stereo_calibration_assessment": {
            "recommendation": final_acceptance["recommendation"],
            "statuses": statuses,
            "ordering_resolution": ordering_resolution,
            "leave_one_out_details": holdout_report,
        },
        "fine_metrics": {
            "per_pair_reprojection": per_pair_rows,
            "leave_one_out_repeatability": holdout_report,
            "image_coverage": {
                "parent": parent_coverage,
                "child": child_coverage,
            },
            "pose_diversity": pose_diversity,
            "extraction": extraction_report,
            "optimization": optimization_report,
            "artifacts": {},
        },
    }
