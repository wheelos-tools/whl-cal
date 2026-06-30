#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from lidar2imu.models import CalibrationDataset
from lidar2imu.review.charts import (
    cost_scan_series as _cost_scan_series,
    series_from_rows as _series_from_rows,
    svg_line_chart as _svg_line_chart,
    svg_xy_path_chart as _svg_xy_path_chart,
    svg_yaw_cost_chart as _svg_yaw_cost_chart,
)
from lidar2imu.review.front_end import (
    build_registration_review_artifacts as _build_registration_review_artifacts,
    build_stitched_cloud as _build_stitched_cloud,
    load_node_clouds as _load_node_clouds,
)
from lidar2imu.review.io_utils import write_point_cloud as _write_point_cloud
from lidar2imu.review.io_utils import write_svg as _write_svg
from lidar2imu.review.report import write_review_html as _write_review_html
from lidar2imu.review.trajectory import (
    review_motion_candidates as _review_motion_candidates,
    review_trajectory_nodes as _review_trajectory_nodes,
    trajectory_cloud_nodes as _trajectory_cloud_nodes,
    trajectory_gap_points as _trajectory_gap_points,
    trajectory_nodes as _trajectory_nodes,
    trajectory_xy_points as _trajectory_xy_points,
)


def build_visualization_artifacts(
    diagnostics_dir: Path,
    *,
    dataset: CalibrationDataset,
    final_transform: np.ndarray,
    summary: dict[str, Any],
    final_acceptance: dict[str, Any],
    motion_assessment: dict[str, Any],
    ground_rows: list[dict[str, Any]],
    motion_rows: list[dict[str, Any]],
    holdout_motion_rows: list[dict[str, Any]],
    observability: dict[str, Any],
    raw_payload: dict[str, Any],
    artifact_links: dict[str, str],
    solver_window_ids: list[int] | None = None,
) -> dict[str, str]:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    ground_plot = _write_svg(
        diagnostics_dir / "ground_residuals_plot.svg",
        _svg_line_chart(
            "Ground residual angle by sample",
            x_label="sample index",
            y_label="normal angle (deg)",
            points=_series_from_rows(ground_rows, "normal_angle_deg"),
            color="#2563eb",
        ),
    )
    artifacts["ground_residuals_plot"] = ground_plot
    artifacts["ground_height_residuals_plot"] = _write_svg(
        diagnostics_dir / "ground_height_residuals_plot.svg",
        _svg_line_chart(
            "Ground height residual by sample",
            x_label="sample index",
            y_label="height residual (m)",
            points=_series_from_rows(ground_rows, "height_residual_m"),
            color="#ea580c",
        ),
    )
    artifacts["motion_rotation_residuals_plot"] = _write_svg(
        diagnostics_dir / "motion_rotation_residuals_plot.svg",
        _svg_line_chart(
            "Motion rotation residual by sample",
            x_label="sample index",
            y_label="rotation residual (deg)",
            points=_series_from_rows(motion_rows, "rotation_residual_deg"),
            color="#2563eb",
        ),
    )
    motion_plot = _write_svg(
        diagnostics_dir / "motion_residuals_plot.svg",
        _svg_line_chart(
            "Motion translation residual by sample",
            x_label="sample index",
            y_label="translation residual (m)",
            points=_series_from_rows(motion_rows, "translation_residual_m"),
            color="#dc2626",
        ),
    )
    artifacts["motion_residuals_plot"] = motion_plot
    artifacts["motion_registration_fitness_plot"] = _write_svg(
        diagnostics_dir / "motion_registration_fitness_plot.svg",
        _svg_line_chart(
            "Motion registration fitness by sample",
            x_label="sample index",
            y_label="registration fitness",
            points=_series_from_rows(motion_rows, "registration_fitness"),
            color="#16a34a",
        ),
    )

    review_candidates = _review_motion_candidates(
        raw_payload,
        selected_window_ids=solver_window_ids,
    )
    (
        registration_artifacts,
        _registration_review_rows,
        dense_review_nodes,
    ) = _build_registration_review_artifacts(
        diagnostics_dir,
        dataset=dataset,
        final_transform=np.asarray(final_transform, dtype=float),
        raw_payload=raw_payload,
        selected_window_ids=solver_window_ids,
    )
    artifacts.update(registration_artifacts)
    trajectory_nodes = dense_review_nodes
    if not trajectory_nodes and review_candidates:
        trajectory_nodes = _review_trajectory_nodes(
            raw_payload,
            np.asarray(final_transform, dtype=float),
            selected_window_ids=solver_window_ids,
        )
    if not trajectory_nodes:
        trajectory_nodes = _trajectory_nodes(
            dataset, np.asarray(final_transform, dtype=float)
        )
    if trajectory_nodes:
        artifacts["trajectory_overlay_plot"] = _write_svg(
            diagnostics_dir / "trajectory_overlay.svg",
            _svg_xy_path_chart(
                (
                    "IMU vs LiDAR BEV review trajectory"
                    if review_candidates
                    else "IMU vs LiDAR relative trajectory"
                ),
                series=[
                    {
                        "name": (
                            "LiDAR review chain"
                            if review_candidates
                            else "LiDAR odometry"
                        ),
                        "points": _trajectory_xy_points(
                            trajectory_nodes, source="lidar"
                        ),
                        "color": "#dc2626",
                        "stroke_width": 2.5,
                        "opacity": 0.95,
                    },
                    {
                        "name": (
                            "IMU review chain" if review_candidates else "IMU odometry"
                        ),
                        "points": _trajectory_xy_points(trajectory_nodes, source="imu"),
                        "color": "#2563eb",
                        "stroke_width": 5.0,
                        "dasharray": "10 7",
                        "opacity": 0.72,
                    },
                ],
            ),
        )
        artifacts["trajectory_position_gap_plot"] = _write_svg(
            diagnostics_dir / "trajectory_position_gap_plot.svg",
            _svg_line_chart(
                (
                    "BEV IMU vs LiDAR position gap"
                    if review_candidates
                    else "Cumulative IMU vs LiDAR position gap"
                ),
                x_label="trajectory node index",
                y_label="position gap (m)",
                points=_trajectory_gap_points(trajectory_nodes),
                color="#7c3aed",
                x_digits=0,
                note_text=(
                    (
                        "window-best review chain built from registered candidates "
                        "across the sequence"
                    )
                    if review_candidates
                    else (
                        "single purple curve = cumulative relative-position gap "
                        "between the IMU chain and LiDAR chain"
                    )
                ),
            ),
        )
        if not registration_artifacts:
            cloud_nodes = _trajectory_cloud_nodes(
                dataset,
                np.asarray(final_transform, dtype=float),
            )
            node_clouds = _load_node_clouds(dataset, cloud_nodes)
            if node_clouds:
                imu_cloud = _build_stitched_cloud(
                    node_clouds,
                    pose_source="imu",
                    final_transform=np.asarray(final_transform, dtype=float),
                    color=(0.145, 0.388, 0.922),
                )
                lidar_cloud = _build_stitched_cloud(
                    node_clouds,
                    pose_source="lidar",
                    final_transform=np.asarray(final_transform, dtype=float),
                    color=(0.862, 0.239, 0.176),
                )
                if not imu_cloud.is_empty():
                    artifacts["imu_trajectory_cloud"] = _write_point_cloud(
                        diagnostics_dir / "imu_trajectory_cloud.ply",
                        imu_cloud,
                    )
                if not lidar_cloud.is_empty():
                    artifacts["lidar_trajectory_cloud"] = _write_point_cloud(
                        diagnostics_dir / "lidar_trajectory_cloud.ply",
                        lidar_cloud,
                    )
                if not imu_cloud.is_empty() and not lidar_cloud.is_empty():
                    overlay_cloud = imu_cloud + lidar_cloud
                    artifacts["trajectory_overlay_cloud"] = _write_point_cloud(
                        diagnostics_dir / "trajectory_overlay_cloud.ply",
                        overlay_cloud.voxel_down_sample(0.20),
                    )

    holdout_plot = _write_svg(
        diagnostics_dir / "holdout_motion_residuals_plot.svg",
        _svg_line_chart(
            "Holdout motion translation residual by sample",
            x_label="sample index",
            y_label="translation residual (m)",
            points=_series_from_rows(holdout_motion_rows, "translation_residual_m"),
            color="#7c3aed",
        ),
    )
    if holdout_motion_rows:
        artifacts["holdout_motion_residuals_plot"] = holdout_plot
    cost_scan = (observability.get("motion_rotation") or {}).get("cost_scan") or {}
    yaw_points = _cost_scan_series(cost_scan)
    if yaw_points:
        artifacts["yaw_cost_scan"] = _write_svg(
            diagnostics_dir / "yaw_cost_scan.svg",
            _svg_yaw_cost_chart(cost_scan),
        )
    artifacts["review_report"] = _write_review_html(
        diagnostics_dir / "review_report.html",
        summary=summary,
        final_acceptance=final_acceptance,
        motion_assessment=motion_assessment,
        artifacts={**artifact_links, **artifacts},
    )
    return artifacts
