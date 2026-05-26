from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from calibration_common.evaluation import (
    write_acceptance_artifacts,
    write_paradigm_artifacts,
    write_table_csv,
)
from lidar2camera.models import ReferenceCalibrationDataset, ReferencePoseObservation
from lidar2lidar.extrinsic_io import (
    build_extrinsics_payload,
    extrinsics_filename,
    save_extrinsics_yaml,
)


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


def _observation_to_payload(observation: ReferencePoseObservation) -> dict[str, Any]:
    return {
        "pose_id": observation.pose_id,
        "image_path": observation.image_path,
        "pcd_path": observation.pcd_path,
        "image_size_wh": {
            "width": int(observation.image_size_wh[0]),
            "height": int(observation.image_size_wh[1]),
        },
        "image_points": np.asarray(observation.image_points, dtype=float).tolist(),
        "object_points": np.asarray(observation.object_points, dtype=float).tolist(),
        "metadata": _sanitize_payload(dict(observation.metadata)),
    }


def dataset_to_payload(dataset: ReferenceCalibrationDataset) -> dict[str, Any]:
    return {
        "parent_frame": dataset.parent_frame,
        "child_frame": dataset.child_frame,
        "camera_matrix": np.asarray(dataset.camera_matrix, dtype=float).tolist(),
        "camera_distortion": np.asarray(
            dataset.camera_distortion, dtype=float
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
    output_path: Path, per_pose_rows: list[dict[str, Any]]
) -> str | None:
    if not per_pose_rows:
        return None
    points = []
    for row in per_pose_rows:
        center = row.get("board_center_camera_m", {}) or {}
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


def _build_checkerboard_alignment_previews(
    diagnostics_dir: Path,
    dataset: ReferenceCalibrationDataset,
    initial_transform: np.ndarray,
    final_transform: np.ndarray,
) -> list[dict[str, str]]:
    preview_dir = diagnostics_dir / "checkerboard_alignment_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    initial_rvec, _ = cv2.Rodrigues(np.asarray(initial_transform[:3, :3], dtype=float))
    initial_tvec = np.asarray(initial_transform[:3, 3], dtype=float).reshape(3, 1)
    final_rvec, _ = cv2.Rodrigues(np.asarray(final_transform[:3, :3], dtype=float))
    final_tvec = np.asarray(final_transform[:3, 3], dtype=float).reshape(3, 1)
    previews: list[dict[str, str]] = []
    for observation in dataset.observations:
        image = cv2.imread(observation.image_path)
        if image is None:
            continue
        image_points = np.asarray(observation.image_points, dtype=float)
        object_points = np.asarray(observation.object_points, dtype=float)
        initial_projected, _ = cv2.projectPoints(
            object_points,
            initial_rvec,
            initial_tvec,
            dataset.camera_matrix,
            dataset.camera_distortion,
        )
        final_projected, _ = cv2.projectPoints(
            object_points,
            final_rvec,
            final_tvec,
            dataset.camera_matrix,
            dataset.camera_distortion,
        )
        overlay = image.copy()
        for u, v in np.round(image_points).astype(np.int32):
            cv2.circle(overlay, (int(u), int(v)), 5, (0, 255, 0), thickness=-1)
        for u, v in np.round(initial_projected.reshape(-1, 2)).astype(np.int32):
            cv2.circle(overlay, (int(u), int(v)), 3, (0, 215, 255), thickness=-1)
        for u, v in np.round(final_projected.reshape(-1, 2)).astype(np.int32):
            cv2.circle(overlay, (int(u), int(v)), 2, (0, 0, 255), thickness=-1)
        legend = [
            ("detected corners", (0, 255, 0)),
            ("initial projection", (0, 215, 255)),
            ("final projection", (0, 0, 255)),
        ]
        for index, (label, color) in enumerate(legend):
            y = 30 + index * 24
            cv2.circle(overlay, (24, y - 5), 6, color, thickness=-1)
            cv2.putText(
                overlay,
                label,
                (40, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                label,
                (40, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )
        output_path = preview_dir / f"{observation.pose_id}.png"
        cv2.imwrite(str(output_path), overlay)
        previews.append({"pose_id": observation.pose_id, "path": str(output_path)})
    return previews


def prepare_output_layout(output_dir: Path) -> tuple[Path, Path, Path]:
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    for file_path in (
        output_dir / "calibrated_tf.yaml",
        output_dir / "metrics.yaml",
        diagnostics_dir / "manifest.yaml",
        diagnostics_dir / "acceptance_report.yaml",
        diagnostics_dir / "status_summary.csv",
        diagnostics_dir / "standardized_data.yaml",
        diagnostics_dir / "data_quality.yaml",
        diagnostics_dir / "visualization_index.yaml",
        diagnostics_dir / "reference_dataset.yaml",
        diagnostics_dir / "extraction.yaml",
        diagnostics_dir / "optimization.yaml",
        diagnostics_dir / "evaluation.yaml",
    ):
        if file_path.exists() and file_path.is_file():
            file_path.unlink()

    initial_guess_dir = output_dir / "initial_guess"
    calibrated_dir = output_dir / "calibrated"
    initial_guess_dir.mkdir(parents=True, exist_ok=True)
    calibrated_dir.mkdir(parents=True, exist_ok=True)
    return initial_guess_dir, calibrated_dir, diagnostics_dir


def write_outputs(
    output_dir: Path,
    dataset: ReferenceCalibrationDataset,
    initial_transform: np.ndarray,
    final_transform: np.ndarray,
    metrics_output: dict,
    extraction_report: dict,
    optimization_report: dict,
    evaluation_report: dict,
    overlay_artifact: str | None = None,
) -> dict:
    initial_guess_dir, calibrated_dir, diagnostics_dir = prepare_output_layout(
        output_dir
    )
    calibrated_file = calibrated_dir / extrinsics_filename(
        dataset.parent_frame, dataset.child_frame
    )
    initial_guess_file = initial_guess_dir / extrinsics_filename(
        dataset.parent_frame, dataset.child_frame
    )

    save_extrinsics_yaml(
        str(initial_guess_file),
        parent_frame=dataset.parent_frame,
        child_frame=dataset.child_frame,
        matrix=initial_transform,
        metadata={"source": "lidar2camera_reference_initial"},
    )
    save_extrinsics_yaml(
        str(calibrated_file),
        parent_frame=dataset.parent_frame,
        child_frame=dataset.child_frame,
        matrix=final_transform,
        metrics=metrics_output["coarse_metrics"],
        metadata={"source": "lidar2camera-calibrate"},
    )

    tf_payload = {
        "base_frame": dataset.parent_frame,
        "extrinsics": [
            build_extrinsics_payload(
                parent_frame=dataset.parent_frame,
                child_frame=dataset.child_frame,
                matrix=final_transform,
                metrics=metrics_output["coarse_metrics"],
                metadata={"source": "lidar2camera-calibrate"},
            )
        ],
    }
    with (output_dir / "calibrated_tf.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(tf_payload, file, sort_keys=False)
    with (diagnostics_dir / "reference_dataset.yaml").open(
        "w", encoding="utf-8"
    ) as file:
        yaml.safe_dump(dataset_to_payload(dataset), file, sort_keys=False)
    with (diagnostics_dir / "extraction.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(_sanitize_payload(extraction_report), file, sort_keys=False)
    with (diagnostics_dir / "optimization.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(_sanitize_payload(optimization_report), file, sort_keys=False)
    with (diagnostics_dir / "evaluation.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(_sanitize_payload(evaluation_report), file, sort_keys=False)

    extraction_entries_csv = write_table_csv(
        diagnostics_dir / "extraction_entries.csv",
        list(extraction_report.get("entries", []) or []),
    )
    per_pose_reprojection_csv = write_table_csv(
        diagnostics_dir / "per_pose_reprojection.csv",
        list(evaluation_report.get("per_pose_reprojection", []) or []),
    )
    leave_one_out_trials = (
        evaluation_report.get("leave_one_out_repeatability", {}) or {}
    ).get("trials", []) or []
    leave_one_out_trials_csv = (
        write_table_csv(
            diagnostics_dir / "leave_one_out_trials.csv", leave_one_out_trials
        )
        if leave_one_out_trials
        else None
    )
    geometry_resolution_rows = []
    geometry_resolution = extraction_report.get("geometry_resolution", {}) or {}
    for iteration in geometry_resolution.get("iterations", []) or []:
        for row in iteration.get("selected", []) or []:
            geometry_resolution_rows.append(
                {
                    "iteration": int(iteration.get("iteration", 0) or 0),
                    "changed_pose_count": int(
                        iteration.get("changed_pose_count", 0) or 0
                    ),
                    **dict(row),
                }
            )
    geometry_resolution_csv = (
        write_table_csv(
            diagnostics_dir / "geometry_resolution.csv", geometry_resolution_rows
        )
        if geometry_resolution_rows
        else None
    )

    image_coverage = metrics_output.get("fine_metrics", {}).get("image_coverage") or {}
    pose_diversity = metrics_output.get("fine_metrics", {}).get("pose_diversity") or {}
    image_coverage_heatmap = _build_grid_heatmap_artifact(
        diagnostics_dir / "image_coverage_heatmap.png",
        list(image_coverage.get("grid_counts", []) or []),
        title="Image coverage heatmap",
    )
    pose_diversity_plot = _build_pose_diversity_artifact(
        diagnostics_dir / "pose_diversity_plot.png",
        list(pose_diversity.get("per_pose", []) or []),
    )
    checkerboard_alignment_previews = _build_checkerboard_alignment_previews(
        diagnostics_dir,
        dataset,
        initial_transform,
        final_transform,
    )

    final_acceptance = _sanitize_payload(metrics_output["final_acceptance"])
    acceptance_artifacts = write_acceptance_artifacts(diagnostics_dir, final_acceptance)
    standardized_data = {
        "schema_version": 1,
        "module": "lidar2camera",
        "representation": "reference_checkerboard_image_pcd_pairs",
        "frames": {
            "parent_frame": dataset.parent_frame,
            "child_frame": dataset.child_frame,
        },
        "sample_counts": {
            "accepted_pose_observations": len(dataset.observations),
            "paired_candidates": int(
                (extraction_report.get("pairing_summary", {}) or {}).get(
                    "paired_count", 0
                )
            ),
            "rejected_pose_observations": int(
                extraction_report.get("rejected_pose_count", 0)
            ),
            "accepted_pair_ratio": float(
                extraction_report.get("accepted_pair_ratio", 0.0)
            ),
        },
        "metadata": _sanitize_payload(dict(dataset.metadata)),
        "camera": {
            "intrinsics": np.asarray(dataset.camera_matrix, dtype=float).tolist(),
            "distortion": np.asarray(dataset.camera_distortion, dtype=float).tolist(),
        },
    }
    data_quality = {
        "schema_version": 1,
        "module": "lidar2camera",
        "status": final_acceptance["status"],
        "release_ready": final_acceptance["release_ready"],
        "quality_gates": final_acceptance["gates"],
        "coarse_statuses": _sanitize_payload(
            metrics_output["coarse_metrics"]["statuses"]
        ),
        "pairing_summary": _sanitize_payload(
            extraction_report.get("pairing_summary", {})
        ),
        "accepted_pair_ratio": _sanitize_payload(
            extraction_report.get("accepted_pair_ratio")
        ),
        "skip_reason_counts": _sanitize_payload(
            extraction_report.get("skip_reason_counts", {})
        ),
        "geometry_resolution": _sanitize_payload(geometry_resolution),
        "optimization_stages": _sanitize_payload(optimization_report.get("stages", [])),
        "recommendation": final_acceptance["recommendation"],
    }
    visual_review = [
        str(output_dir / "metrics.yaml"),
        str(diagnostics_dir / "reference_dataset.yaml"),
        str(diagnostics_dir / "extraction.yaml"),
        str(diagnostics_dir / "optimization.yaml"),
        str(diagnostics_dir / "evaluation.yaml"),
    ]
    if image_coverage_heatmap is not None:
        visual_review.append(image_coverage_heatmap)
    if pose_diversity_plot is not None:
        visual_review.append(pose_diversity_plot)
    if overlay_artifact is not None:
        visual_review.append(overlay_artifact)
    else:
        visual_review.append(
            "No overlay artifact was generated; keep this run review-only until "
            "visual evidence is available."
        )
    visual_review.extend(
        preview["path"]
        for preview in checkerboard_alignment_previews[
            : min(6, len(checkerboard_alignment_previews))
        ]
    )
    detail_metrics = [
        str(output_dir / "metrics.yaml"),
        str(diagnostics_dir / "reference_dataset.yaml"),
        str(diagnostics_dir / "extraction.yaml"),
        str(diagnostics_dir / "optimization.yaml"),
        str(diagnostics_dir / "evaluation.yaml"),
        extraction_entries_csv,
        per_pose_reprojection_csv,
        leave_one_out_trials_csv,
        geometry_resolution_csv,
    ]
    visualization_index = {
        "schema_version": 1,
        "module": "lidar2camera",
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
                "Read diagnostics/standardized_data.yaml to confirm "
                "accepted/rejected sample counts and capture assumptions."
            ),
            (
                "Read diagnostics/data_quality.yaml to inspect extraction_yield, "
                "image_coverage, pose_diversity, and board_geometry gates."
            ),
            (
                "Read diagnostics/optimization.yaml stages plus "
                "diagnostics/geometry_resolution.csv before trusting a "
                "candidate that required multiple board-geometry resolution rounds."
            ),
            (
                "Inspect the checkerboard corner coverage and extraction skip "
                "reasons before promoting a run."
            ),
            (
                "Inspect checkerboard_alignment_previews to confirm detected "
                "corners and final projected board corners agree pose by pose."
            ),
            (
                "Inspect the overlay for board-edge alignment and depth "
                "consistency, not only aggregate RMS."
            ),
            (
                "Treat missing holdout or repeatability evidence as review-only, "
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

    manifest = {
        "parent_frame": dataset.parent_frame,
        "child_frame": dataset.child_frame,
        "sample_counts": {
            "pose_observations": len(dataset.observations),
        },
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
                "per_pose_reprojection_csv": per_pose_reprojection_csv,
                "leave_one_out_trials_csv": leave_one_out_trials_csv,
                "geometry_resolution_csv": geometry_resolution_csv,
                "image_coverage_heatmap": image_coverage_heatmap,
                "pose_diversity_plot": pose_diversity_plot,
                "checkerboard_alignment_previews": checkerboard_alignment_previews,
                "overlay": overlay_artifact,
            },
        },
    }
    with (diagnostics_dir / "manifest.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(manifest, file, sort_keys=False)
    return manifest
