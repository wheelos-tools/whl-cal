from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from calibration_common.evaluation import (build_final_acceptance,
                                           write_acceptance_artifacts,
                                           write_paradigm_artifacts,
                                           write_table_csv)
from camera2camera.models import (StereoCalibrationDataset,
                                  StereoCalibrationObservation)
from lidar2lidar.extrinsic_io import extrinsics_filename, save_extrinsics_yaml


def _sanitize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_payload(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_sanitize_payload(item) for item in value]
    return value


def _observation_to_payload(
    observation: StereoCalibrationObservation,
) -> dict[str, Any]:
    return {
        "pose_id": observation.pose_id,
        "parent_image_path": observation.parent_image_path,
        "child_image_path": observation.child_image_path,
        "parent_image_size_wh": {
            "width": int(observation.parent_image_size_wh[0]),
            "height": int(observation.parent_image_size_wh[1]),
        },
        "child_image_size_wh": {
            "width": int(observation.child_image_size_wh[0]),
            "height": int(observation.child_image_size_wh[1]),
        },
        "parent_image_points": np.asarray(
            observation.parent_image_points, dtype=float
        ).tolist(),
        "child_image_points": np.asarray(
            observation.child_image_points, dtype=float
        ).tolist(),
        "object_points": np.asarray(observation.object_points, dtype=float).tolist(),
        "metadata": _sanitize_payload(dict(observation.metadata)),
    }


def dataset_to_payload(dataset: StereoCalibrationDataset) -> dict[str, Any]:
    return {
        "parent_frame": dataset.parent_frame,
        "child_frame": dataset.child_frame,
        "parent_camera_matrix": np.asarray(
            dataset.parent_camera_matrix, dtype=float
        ).tolist(),
        "parent_camera_distortion": np.asarray(
            dataset.parent_camera_distortion, dtype=float
        ).tolist(),
        "child_camera_matrix": np.asarray(
            dataset.child_camera_matrix, dtype=float
        ).tolist(),
        "child_camera_distortion": np.asarray(
            dataset.child_camera_distortion, dtype=float
        ).tolist(),
        "initial_transform": (
            None
            if dataset.initial_transform is None
            else np.asarray(dataset.initial_transform, dtype=float).tolist()
        ),
        "metadata": _sanitize_payload(dict(dataset.metadata)),
        "observations": [
            _observation_to_payload(observation) for observation in dataset.observations
        ],
    }


def _build_grid_heatmap_artifact(
    output_path: Path, grid_counts: list[list[int]], *, title: str
) -> str | None:
    if not grid_counts:
        return None
    rows = len(grid_counts)
    cols = max((len(row) for row in grid_counts), default=0)
    if rows <= 0 or cols <= 0:
        return None

    cell_size = 120
    left_margin = 80
    top_margin = 100
    width = left_margin + cols * cell_size + 40
    height = top_margin + rows * cell_size + 60
    image = np.full((height, width, 3), 245, dtype=np.uint8)

    max_count = max(max(int(value) for value in row) for row in grid_counts)
    max_count = max(max_count, 1)
    cv2.putText(
        image,
        title,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (30, 30, 30),
        2,
    )
    for row_index, row in enumerate(grid_counts):
        for col_index, count in enumerate(row):
            x0 = left_margin + col_index * cell_size
            y0 = top_margin + row_index * cell_size
            x1 = x0 + cell_size - 10
            y1 = y0 + cell_size - 10
            intensity = int(255 * float(count) / max_count)
            color = (255 - intensity, 200 - intensity // 3, 80 + intensity // 2)
            cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness=-1)
            cv2.rectangle(image, (x0, y0), (x1, y1), (60, 60, 60), thickness=2)
            cv2.putText(
                image,
                str(int(count)),
                (x0 + 38, y0 + 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (20, 20, 20),
                2,
            )
    for col_index in range(cols):
        cv2.putText(
            image,
            str(col_index),
            (left_margin + col_index * cell_size + 42, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (70, 70, 70),
            2,
        )
    for row_index in range(rows):
        cv2.putText(
            image,
            str(row_index),
            (35, top_margin + row_index * cell_size + 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (70, 70, 70),
            2,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return str(output_path)


def _build_pose_diversity_artifact(
    output_path: Path, per_pair_rows: list[dict[str, Any]]
) -> str | None:
    if not per_pair_rows:
        return None
    points = []
    for row in per_pair_rows:
        center = row.get("board_center_parent_camera_m", {}) or {}
        depth = center.get("z")
        tilt = row.get("board_tilt_deg")
        if depth is None or tilt is None:
            continue
        points.append((str(row.get("pose_id")), float(depth), float(tilt)))
    if not points:
        return None

    width = 720
    height = 480
    margin_left = 80
    margin_bottom = 60
    margin_top = 40
    margin_right = 40
    image = np.full((height, width, 3), 250, dtype=np.uint8)
    cv2.putText(
        image,
        "Pose diversity: depth vs tilt",
        (30, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (30, 30, 30),
        2,
    )
    x0 = margin_left
    y0 = height - margin_bottom
    x1 = width - margin_right
    y1 = margin_top
    cv2.line(image, (x0, y0), (x1, y0), (70, 70, 70), 2)
    cv2.line(image, (x0, y0), (x0, y1), (70, 70, 70), 2)

    depths = [point[1] for point in points]
    tilts = [point[2] for point in points]
    min_depth = min(depths)
    max_depth = max(depths)
    min_tilt = min(tilts)
    max_tilt = max(tilts)
    depth_span = max(max_depth - min_depth, 1e-6)
    tilt_span = max(max_tilt - min_tilt, 1e-6)

    for pose_id, depth, tilt in points:
        px = int(x0 + ((depth - min_depth) / depth_span) * (x1 - x0))
        py = int(y0 - ((tilt - min_tilt) / tilt_span) * (y0 - y1))
        cv2.circle(image, (px, py), 6, (0, 102, 204), thickness=-1)
        cv2.putText(
            image,
            pose_id,
            (px + 8, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 40, 40),
            1,
        )
    cv2.putText(
        image,
        f"depth [{min_depth:.2f}, {max_depth:.2f}] m",
        (x0, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (70, 70, 70),
        2,
    )
    cv2.putText(
        image,
        f"tilt [{min_tilt:.1f}, {max_tilt:.1f}] deg",
        (width - 250, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (70, 70, 70),
        2,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return str(output_path)


def _draw_epipolar_line(
    image: np.ndarray, line: np.ndarray, color: tuple[int, int, int]
) -> None:
    a, b, c = [float(value) for value in line]
    height, width = image.shape[:2]
    if abs(b) > 1e-8:
        y0 = int(np.clip(-(c / b), 0, height - 1))
        y1 = int(np.clip(-(c + a * (width - 1)) / b, 0, height - 1))
        cv2.line(image, (0, y0), (width - 1, y1), color, 1, cv2.LINE_AA)
        return
    if abs(a) > 1e-8:
        x = int(np.clip(-(c / a), 0, width - 1))
        cv2.line(image, (x, 0), (x, height - 1), color, 1, cv2.LINE_AA)


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


def _build_epipolar_previews(
    diagnostics_dir: Path,
    dataset: StereoCalibrationDataset,
    metrics_output: dict[str, Any],
    fundamental_matrix: np.ndarray | None,
) -> list[dict[str, str]]:
    if fundamental_matrix is None:
        return []
    preview_dir = diagnostics_dir / "epipolar_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    per_pair_rows = list(
        (metrics_output.get("fine_metrics", {}) or {}).get("per_pair_reprojection", [])
    )
    sorted_rows = sorted(
        per_pair_rows,
        key=lambda row: float(row.get("combined_rms_px", 0.0)),
    )
    preview_rows = []
    seen_pose_ids: set[str] = set()
    for row in list(reversed(sorted_rows[:]))[:3] + sorted_rows[:3]:
        pose_id = str(row.get("pose_id"))
        if pose_id in seen_pose_ids:
            continue
        seen_pose_ids.add(pose_id)
        preview_rows.append(row)
    previews = []
    for row in preview_rows[: min(6, len(preview_rows))]:
        pose_id = str(row.get("pose_id"))
        observation = next(
            (item for item in dataset.observations if item.pose_id == pose_id),
            None,
        )
        if observation is None:
            continue
        parent = cv2.imread(observation.parent_image_path)
        child = cv2.imread(observation.child_image_path)
        if parent is None or child is None:
            continue
        parent_points = _undistort_pixel_points(
            np.asarray(observation.parent_image_points, dtype=float).reshape(-1, 2),
            dataset.parent_camera_matrix,
            dataset.parent_camera_distortion,
        )
        child_points = _undistort_pixel_points(
            np.asarray(observation.child_image_points, dtype=float).reshape(-1, 2),
            dataset.child_camera_matrix,
            dataset.child_camera_distortion,
        )
        canvas_parent = cv2.undistort(
            parent,
            np.asarray(dataset.parent_camera_matrix, dtype=float),
            np.asarray(dataset.parent_camera_distortion, dtype=float),
        )
        canvas_child = cv2.undistort(
            child,
            np.asarray(dataset.child_camera_matrix, dtype=float),
            np.asarray(dataset.child_camera_distortion, dtype=float),
        )
        if parent_points.shape[0] <= 0:
            continue
        step = max(1, parent_points.shape[0] // 12)
        subset_indices = list(range(0, parent_points.shape[0], step))
        parent_h = np.concatenate(
            [parent_points[subset_indices], np.ones((len(subset_indices), 1))], axis=1
        )
        child_lines = (fundamental_matrix @ parent_h.T).T
        for line, parent_point, child_point in zip(
            child_lines, parent_points[subset_indices], child_points[subset_indices]
        ):
            color = (0, 220, 255)
            _draw_epipolar_line(canvas_child, line, color)
            cv2.circle(
                canvas_parent,
                tuple(np.round(parent_point).astype(int)),
                5,
                color,
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                canvas_child,
                tuple(np.round(child_point).astype(int)),
                5,
                (0, 255, 0),
                -1,
                cv2.LINE_AA,
            )
        target_height = max(canvas_parent.shape[0], canvas_child.shape[0])
        if canvas_parent.shape[0] != target_height:
            scale = target_height / float(canvas_parent.shape[0])
            canvas_parent = cv2.resize(
                canvas_parent,
                (int(round(canvas_parent.shape[1] * scale)), target_height),
            )
        if canvas_child.shape[0] != target_height:
            scale = target_height / float(canvas_child.shape[0])
            canvas_child = cv2.resize(
                canvas_child,
                (int(round(canvas_child.shape[1] * scale)), target_height),
            )
        gap = np.full((target_height, 20, 3), 30, dtype=np.uint8)
        canvas = np.concatenate([canvas_parent, gap, canvas_child], axis=1)
        cv2.putText(
            canvas,
            (
                f"{pose_id} | undistorted epipolar | "
                f"epipolar_mean_px={float(row.get('epipolar_mean_px', 0.0)):.3f}"
            ),
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )
        out_path = preview_dir / f"{pose_id}.png"
        cv2.imwrite(str(out_path), canvas)
        previews.append({"pose_id": pose_id, "path": str(out_path)})
    return previews


def write_outputs(
    output_dir: Path,
    *,
    dataset: StereoCalibrationDataset,
    initial_transform: np.ndarray,
    final_transform: np.ndarray,
    extraction_report: dict[str, Any],
    optimization_report: dict[str, Any],
    evaluation: dict[str, Any],
    metrics_output: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    initial_guess_dir = output_dir / "initial_guess"
    calibrated_dir = output_dir / "calibrated"
    initial_guess_dir.mkdir(parents=True, exist_ok=True)
    calibrated_dir.mkdir(parents=True, exist_ok=True)

    filename = extrinsics_filename(dataset.parent_frame, dataset.child_frame)
    initial_guess_file = initial_guess_dir / filename
    calibrated_file = calibrated_dir / filename
    save_extrinsics_yaml(
        str(initial_guess_file),
        dataset.parent_frame,
        dataset.child_frame,
        initial_transform,
        metrics={"stage": "initial_guess"},
    )
    extrinsic_payload = save_extrinsics_yaml(
        str(calibrated_file),
        dataset.parent_frame,
        dataset.child_frame,
        final_transform,
        metrics={
            "final_rms_px": float(metrics_output["summary"]["final_rms_px"]),
            "release_ready": bool(metrics_output["summary"]["release_ready"]),
        },
    )
    with (output_dir / "calibrated_tf.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(extrinsic_payload, file, sort_keys=False)

    with (diagnostics_dir / "reference_dataset.yaml").open(
        "w", encoding="utf-8"
    ) as file:
        yaml.safe_dump(dataset_to_payload(dataset), file, sort_keys=False)
    with (diagnostics_dir / "extraction.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(_sanitize_payload(extraction_report), file, sort_keys=False)
    with (diagnostics_dir / "optimization.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(_sanitize_payload(optimization_report), file, sort_keys=False)
    with (diagnostics_dir / "evaluation.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(_sanitize_payload(evaluation), file, sort_keys=False)

    extraction_entries_csv = write_table_csv(
        diagnostics_dir / "extraction_entries.csv",
        list(extraction_report.get("entries", [])),
    )
    per_pair_reprojection_csv = write_table_csv(
        diagnostics_dir / "per_pair_reprojection.csv",
        list(evaluation.get("per_pair_rows", [])),
    )
    leave_one_out_trials_csv = write_table_csv(
        diagnostics_dir / "leave_one_out_trials.csv",
        list((evaluation.get("leave_one_out_repeatability") or {}).get("trials", [])),
    )

    parent_heatmap = _build_grid_heatmap_artifact(
        diagnostics_dir / "parent_image_coverage_heatmap.png",
        list(
            (
                (evaluation.get("parent_image_coverage") or {}).get("grid_counts", [])
                or []
            )
        ),
        title="Parent image coverage heatmap",
    )
    child_heatmap = _build_grid_heatmap_artifact(
        diagnostics_dir / "child_image_coverage_heatmap.png",
        list(
            (
                (evaluation.get("child_image_coverage") or {}).get("grid_counts", [])
                or []
            )
        ),
        title="Child image coverage heatmap",
    )
    pose_diversity_plot = _build_pose_diversity_artifact(
        diagnostics_dir / "pose_diversity_plot.png",
        list(evaluation.get("per_pair_rows", [])),
    )
    epipolar_previews = _build_epipolar_previews(
        diagnostics_dir,
        dataset,
        metrics_output,
        (
            np.asarray(evaluation.get("fundamental_matrix"), dtype=float)
            if evaluation.get("fundamental_matrix") is not None
            else None
        ),
    )

    final_acceptance = _sanitize_payload(metrics_output["final_acceptance"])
    acceptance_artifacts = write_acceptance_artifacts(diagnostics_dir, final_acceptance)
    standardized_data = {
        "schema_version": 1,
        "module": "camera2camera",
        "representation": "checkerboard_image_pair_observations",
        "frames": {
            "parent_frame": dataset.parent_frame,
            "child_frame": dataset.child_frame,
        },
        "sample_counts": {
            "paired_candidates": int(
                (extraction_report.get("pairing_summary") or {}).get("paired_count", 0)
            ),
            "pre_optimization_accepted_pairs": int(
                extraction_report.get("pre_optimization_accepted_pair_count", 0)
            ),
            "optimization_inlier_pairs": int(len(dataset.observations)),
            "rejected_pairs": int(extraction_report.get("rejected_pair_count", 0)),
            "accepted_pair_ratio": float(
                extraction_report.get("final_inlier_pair_ratio", 0.0)
            ),
        },
        "camera_calibrations": {
            "parent": {
                "camera_matrix": np.asarray(
                    dataset.parent_camera_matrix, dtype=float
                ).tolist(),
                "distortion": np.asarray(
                    dataset.parent_camera_distortion, dtype=float
                ).tolist(),
            },
            "child": {
                "camera_matrix": np.asarray(
                    dataset.child_camera_matrix, dtype=float
                ).tolist(),
                "distortion": np.asarray(
                    dataset.child_camera_distortion, dtype=float
                ).tolist(),
            },
        },
        "metadata": _sanitize_payload(dict(dataset.metadata)),
    }
    data_quality = {
        "schema_version": 1,
        "module": "camera2camera",
        "status": final_acceptance["status"],
        "release_ready": final_acceptance["release_ready"],
        "quality_gates": final_acceptance["gates"],
        "coarse_statuses": _sanitize_payload(
            metrics_output["coarse_metrics"]["statuses"]
        ),
        "pairing_summary": _sanitize_payload(
            extraction_report.get("pairing_summary", {})
        ),
        "accepted_pair_ratio": float(
            extraction_report.get("final_inlier_pair_ratio", 0.0)
        ),
        "pre_optimization_accepted_pair_ratio": float(
            extraction_report.get("pre_optimization_accepted_pair_ratio", 0.0)
        ),
        "skip_reason_counts": _sanitize_payload(
            extraction_report.get("skip_reason_counts", {})
        ),
        "ordering_resolution": _sanitize_payload(
            extraction_report.get("ordering_resolution", {})
        ),
        "optimization_rounds": _sanitize_payload(optimization_report.get("rounds", [])),
        "recommendation": final_acceptance["recommendation"],
    }
    visual_review = [
        str(output_dir / "metrics.yaml"),
        str(diagnostics_dir / "reference_dataset.yaml"),
        str(diagnostics_dir / "extraction.yaml"),
        str(diagnostics_dir / "optimization.yaml"),
        str(diagnostics_dir / "evaluation.yaml"),
    ]
    for artifact in (parent_heatmap, child_heatmap, pose_diversity_plot):
        if artifact is not None:
            visual_review.append(artifact)
    visual_review.extend(
        preview["path"]
        for preview in epipolar_previews[: min(6, len(epipolar_previews))]
    )
    detail_metrics = [
        str(output_dir / "metrics.yaml"),
        str(diagnostics_dir / "reference_dataset.yaml"),
        str(diagnostics_dir / "extraction.yaml"),
        str(diagnostics_dir / "optimization.yaml"),
        str(diagnostics_dir / "evaluation.yaml"),
        extraction_entries_csv,
        per_pair_reprojection_csv,
        leave_one_out_trials_csv,
    ]
    visualization_index = {
        "schema_version": 1,
        "module": "camera2camera",
        "layers": {
            "conclusion": [
                acceptance_artifacts["acceptance_report"],
                acceptance_artifacts["status_summary_csv"],
            ],
            "detail_metrics": [item for item in detail_metrics if item is not None],
            "visual_review": [item for item in visual_review if item is not None],
        },
        "manual_review": [
            (
                "Read diagnostics/standardized_data.yaml to confirm paired, accepted, "
                "and optimization-inlier counts before trusting the extrinsic."
            ),
            (
                "Read diagnostics/data_quality.yaml to inspect accepted_pair_ratio, "
                "ordering_resolution, and optimization round decisions."
            ),
            (
                "Inspect per_pair_reprojection.csv and leave_one_out_trials.csv for "
                "tail residuals and repeatability outliers, not only the mean RMS."
            ),
            (
                "Inspect both image coverage heatmaps and pose_diversity_plot.png to "
                "confirm the board moved across both cameras and multiple depths/tilts."
            ),
            (
                "Inspect epipolar_previews to confirm corresponding corners sit near "
                "their predicted epipolar lines across both views."
            ),
            (
                "Treat missing holdout or warning-level repeatability as review-only, "
                "not release-ready."
            ),
        ],
    }
    paradigm_artifacts = write_paradigm_artifacts(
        diagnostics_dir,
        standardized_data=standardized_data,
        data_quality=data_quality,
        visualization_index=visualization_index,
    )
    metrics_output["fine_metrics"]["artifacts"].update(acceptance_artifacts)
    metrics_output["fine_metrics"]["artifacts"].update(paradigm_artifacts)
    with (output_dir / "metrics.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(_sanitize_payload(metrics_output), file, sort_keys=False)

    return {
        "artifacts": {
            "calibrated_tf": str(output_dir / "calibrated_tf.yaml"),
            "metrics": str(output_dir / "metrics.yaml"),
            "initial_guess": str(initial_guess_file),
            "calibrated_extrinsics": str(calibrated_file),
            "diagnostics": {
                "reference_dataset": str(diagnostics_dir / "reference_dataset.yaml"),
                "extraction": str(diagnostics_dir / "extraction.yaml"),
                "optimization": str(diagnostics_dir / "optimization.yaml"),
                "evaluation": str(diagnostics_dir / "evaluation.yaml"),
                "acceptance_report": acceptance_artifacts["acceptance_report"],
                "status_summary_csv": acceptance_artifacts["status_summary_csv"],
                "standardized_data": paradigm_artifacts["standardized_data"],
                "data_quality": paradigm_artifacts["data_quality"],
                "visualization_index": paradigm_artifacts["visualization_index"],
                "extraction_entries_csv": extraction_entries_csv,
                "per_pair_reprojection_csv": per_pair_reprojection_csv,
                "leave_one_out_trials_csv": leave_one_out_trials_csv,
                "parent_image_coverage_heatmap": parent_heatmap,
                "child_image_coverage_heatmap": child_heatmap,
                "pose_diversity_plot": pose_diversity_plot,
                "epipolar_previews": epipolar_previews,
            },
        }
    }


def write_failure_outputs(
    output_dir: Path,
    *,
    dataset: StereoCalibrationDataset,
    extraction_report: dict[str, Any],
    failure_message: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    with (diagnostics_dir / "reference_dataset.yaml").open(
        "w", encoding="utf-8"
    ) as file:
        yaml.safe_dump(dataset_to_payload(dataset), file, sort_keys=False)
    with (diagnostics_dir / "extraction.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(_sanitize_payload(extraction_report), file, sort_keys=False)
    with (diagnostics_dir / "optimization.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            {
                "skipped": True,
                "reason": "extraction_failed",
                "message": failure_message,
            },
            file,
            sort_keys=False,
        )
    with (diagnostics_dir / "evaluation.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            {
                "skipped": True,
                "reason": "extraction_failed",
                "message": failure_message,
            },
            file,
            sort_keys=False,
        )

    extraction_entries_csv = write_table_csv(
        diagnostics_dir / "extraction_entries.csv",
        list(extraction_report.get("entries", [])),
    )
    paired_candidate_count = int(
        (extraction_report.get("pairing_summary") or {}).get("paired_count", 0)
    )

    final_acceptance = build_final_acceptance(
        module="camera2camera",
        gates=[
            {
                "name": "extraction_ready",
                "status": "fail",
                "severity": "required",
                "evidence": failure_message,
                "action": (
                    "Read diagnostics/extraction.yaml and recollect cleaner stereo "
                    "board pairs."
                ),
            },
            {
                "name": "accepted_pair_count",
                "status": "fail",
                "severity": "required",
                "evidence": (
                    f"accepted_pairs={len(dataset.observations)} "
                    f"paired_candidates={paired_candidate_count}"
                ),
                "action": (
                    "Increase usable paired captures before retrying calibration."
                ),
            },
        ],
        pass_recommendation="release_stereo_extrinsics",
        review_recommendation="review_camera2camera_diagnostics",
        fail_recommendation="reject_and_recollect_camera2camera_pairs",
    )
    acceptance_artifacts = write_acceptance_artifacts(diagnostics_dir, final_acceptance)
    standardized_data = {
        "schema_version": 1,
        "module": "camera2camera",
        "representation": "checkerboard_image_pair_observations",
        "frames": {
            "parent_frame": dataset.parent_frame,
            "child_frame": dataset.child_frame,
        },
        "sample_counts": {
            "paired_candidates": int(
                (extraction_report.get("pairing_summary") or {}).get("paired_count", 0)
            ),
            "accepted_pairs": int(len(dataset.observations)),
            "rejected_pairs": int(extraction_report.get("rejected_pair_count", 0)),
        },
        "camera_calibrations": {
            "parent": {
                "camera_matrix": np.asarray(
                    dataset.parent_camera_matrix, dtype=float
                ).tolist(),
                "distortion": np.asarray(
                    dataset.parent_camera_distortion, dtype=float
                ).tolist(),
            },
            "child": {
                "camera_matrix": np.asarray(
                    dataset.child_camera_matrix, dtype=float
                ).tolist(),
                "distortion": np.asarray(
                    dataset.child_camera_distortion, dtype=float
                ).tolist(),
            },
        },
        "metadata": {
            **_sanitize_payload(dict(dataset.metadata)),
            "failure_stage": "extraction",
            "failure_message": failure_message,
        },
    }
    data_quality = {
        "schema_version": 1,
        "module": "camera2camera",
        "status": final_acceptance["status"],
        "release_ready": final_acceptance["release_ready"],
        "quality_gates": final_acceptance["gates"],
        "pairing_summary": _sanitize_payload(
            extraction_report.get("pairing_summary", {})
        ),
        "skip_reason_counts": _sanitize_payload(
            extraction_report.get("skip_reason_counts", {})
        ),
        "failure_stage": "extraction",
        "failure_message": failure_message,
        "recommendation": final_acceptance["recommendation"],
    }
    visualization_index = {
        "schema_version": 1,
        "module": "camera2camera",
        "layers": {
            "conclusion": [
                acceptance_artifacts["acceptance_report"],
                acceptance_artifacts["status_summary_csv"],
            ],
            "detail_metrics": [
                str(diagnostics_dir / "reference_dataset.yaml"),
                str(diagnostics_dir / "extraction.yaml"),
                str(diagnostics_dir / "optimization.yaml"),
                str(diagnostics_dir / "evaluation.yaml"),
                extraction_entries_csv,
            ],
            "visual_review": [],
        },
        "manual_review": [
            (
                "Read diagnostics/extraction.yaml for the failure reason and "
                "skip_reason_counts."
            ),
            (
                "Read diagnostics/standardized_data.yaml to confirm how many pairs "
                "were actually usable."
            ),
            (
                "Inspect diagnostics/extraction_entries.csv before recollecting the "
                "stereo board session."
            ),
        ],
    }
    paradigm_artifacts = write_paradigm_artifacts(
        diagnostics_dir,
        standardized_data=standardized_data,
        data_quality=data_quality,
        visualization_index=visualization_index,
    )
    metrics_output = {
        "summary": {
            "pair_count": int(len(dataset.observations)),
            "final_acceptance_status": final_acceptance["status"],
            "release_ready": final_acceptance["release_ready"],
            "failure_stage": "extraction",
            "failure_message": failure_message,
        },
        "coarse_metrics": {
            "pair_count": int(len(dataset.observations)),
            "accepted_pair_ratio": float(
                extraction_report.get("pre_optimization_accepted_pair_ratio", 0.0)
            ),
            "statuses": {
                "optimization_success": "fail",
                "reprojection": "unknown",
                "pair_reprojection": "unknown",
            },
        },
        "final_acceptance": final_acceptance,
        "fine_metrics": {
            "extraction": extraction_report,
            "optimization": {"skipped": True, "message": failure_message},
            "artifacts": {
                **acceptance_artifacts,
                **paradigm_artifacts,
            },
        },
    }
    with (output_dir / "metrics.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(_sanitize_payload(metrics_output), file, sort_keys=False)
    return {
        "artifacts": {
            "metrics": str(output_dir / "metrics.yaml"),
            "diagnostics": {
                "reference_dataset": str(diagnostics_dir / "reference_dataset.yaml"),
                "extraction": str(diagnostics_dir / "extraction.yaml"),
                "optimization": str(diagnostics_dir / "optimization.yaml"),
                "evaluation": str(diagnostics_dir / "evaluation.yaml"),
                "acceptance_report": acceptance_artifacts["acceptance_report"],
                "status_summary_csv": acceptance_artifacts["status_summary_csv"],
                "standardized_data": paradigm_artifacts["standardized_data"],
                "data_quality": paradigm_artifacts["data_quality"],
                "visualization_index": paradigm_artifacts["visualization_index"],
                "extraction_entries_csv": extraction_entries_csv,
            },
        }
    }
