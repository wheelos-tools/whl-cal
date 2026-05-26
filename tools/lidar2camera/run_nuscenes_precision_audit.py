#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from calibration_common.evaluation import write_table_csv
from lidar2camera.metrics import transform_delta_metrics
from lidar2camera.nuscenes_benchmark import (
    EdgeRefinementConfig,
    _delta_transform,
    _edge_alignment_cost,
    _projection_stats,
    _silhouette_alignment_cost,
    build_edge_alignment_context,
    load_nuscenes_camera_samples,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _axis_cases(
    translation_magnitudes_m: tuple[float, ...],
    rotation_magnitudes_deg: tuple[float, ...],
) -> list[tuple[str, str, float, np.ndarray, np.ndarray]]:
    cases = []
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        for magnitude in translation_magnitudes_m:
            translation = np.zeros(3, dtype=float)
            translation[axis_index] = float(magnitude)
            cases.append(
                ("translation", axis_name, float(magnitude), np.zeros(3), translation)
            )
    for axis_index, axis_name in enumerate(("roll", "pitch", "yaw")):
        for magnitude in rotation_magnitudes_deg:
            rotvec = np.zeros(3, dtype=float)
            rotvec[axis_index] = np.deg2rad(float(magnitude))
            cases.append(("rotation", axis_name, float(magnitude), rotvec, np.zeros(3)))
    return cases


def _summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["kind"], row["axis"], float(row["magnitude"]))].append(row)

    summary = []
    for (kind, axis, magnitude), group in sorted(groups.items()):
        summary.append(
            {
                "kind": kind,
                "axis": axis,
                "magnitude": magnitude,
                "sample_count": len(group),
                "edge_gt_better_rate": (
                    sum(bool(row["edge_gt_correction_better"]) for row in group)
                    / max(len(group), 1)
                ),
                "silhouette_gt_better_rate": (
                    sum(bool(row["silhouette_gt_correction_better"]) for row in group)
                    / max(len(group), 1)
                ),
                "both_gt_better_rate": (
                    sum(bool(row["both_gt_correction_better"]) for row in group)
                    / max(len(group), 1)
                ),
                "mean_projected_point_count": float(
                    np.mean([row["projected_point_count"] for row in group])
                ),
                "mean_projected_bbox_area_ratio": float(
                    np.mean([row["projected_bbox_area_ratio"] for row in group])
                ),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit targetless lidar2camera initial-value tolerance on nuScenes "
            "with axis-aligned GT perturbations."
        )
    )
    parser.add_argument(
        "--info-path",
        default=(
            "/mnt/synology/nuScenes/OpenDataLab___nuScenes/raw/Trainval/train/"
            "nuscenes_infos_val.pkl"
        ),
        help="Path to the nuScenes info pickle file.",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--camera-names", default="CAM_FRONT")
    parser.add_argument("--sample-limit", type=int, default=4)
    parser.add_argument("--sample-tokens", default="")
    parser.add_argument("--max-sensor-time-delta-ms", type=float, default=40.0)
    parser.add_argument(
        "--translation-magnitudes-m",
        default="0.01,0.02,0.05,0.10",
        help="Comma-separated axis translation perturbations in meters.",
    )
    parser.add_argument(
        "--rotation-magnitudes-deg",
        default="0.1,0.3,0.5,1.0,2.0",
        help="Comma-separated roll/pitch/yaw perturbations in degrees.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/lidar2camera/targetless_initial_precision_audit",
    )
    parser.add_argument("--intensity-percentile", type=float, default=75.0)
    parser.add_argument("--max-points", type=int, default=12000)
    parser.add_argument("--visualization-max-range-m", type=float, default=80.0)
    parser.add_argument("--visualization-max-points", type=int, default=60000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    samples, manifest = load_nuscenes_camera_samples(
        info_path=args.info_path,
        data_root=args.data_root,
        camera_names=_parse_csv_strings(args.camera_names),
        sample_limit=args.sample_limit,
        sample_tokens=_parse_csv_strings(args.sample_tokens),
        max_sensor_time_delta_ms=args.max_sensor_time_delta_ms,
    )
    config = EdgeRefinementConfig(
        intensity_percentile=args.intensity_percentile,
        max_points=args.max_points,
        visualization_max_point_xy_range_m=args.visualization_max_range_m,
        visualization_max_points=args.visualization_max_points,
    )
    cases = _axis_cases(
        _parse_csv_floats(args.translation_magnitudes_m),
        _parse_csv_floats(args.rotation_magnitudes_deg),
    )

    rows: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(samples):
        context = build_edge_alignment_context(sample, config)
        reference_transform = np.asarray(sample.rigid_lidar_to_camera, dtype=float)
        for kind, axis, magnitude, rotvec, translation in cases:
            initial_transform = reference_transform @ _delta_transform(
                rotvec, translation
            )
            correction = np.linalg.inv(initial_transform) @ reference_transform
            correction_params = np.r_[
                R.from_matrix(correction[:3, :3]).as_rotvec(),
                correction[:3, 3],
            ]
            initial_delta = transform_delta_metrics(
                reference_transform, initial_transform
            )
            projection = _projection_stats(
                context=context,
                transform=initial_transform,
                min_depth_m=config.min_camera_depth_m,
            )
            edge_initial = _edge_alignment_cost(
                np.zeros(6),
                initial_transform=initial_transform,
                context=context,
                config=config,
            )
            edge_gt = _edge_alignment_cost(
                correction_params,
                initial_transform=initial_transform,
                context=context,
                config=config,
            )
            silhouette_initial = _silhouette_alignment_cost(
                np.zeros(6),
                initial_transform=initial_transform,
                context=context,
                config=config,
            )
            silhouette_gt = _silhouette_alignment_cost(
                correction_params,
                initial_transform=initial_transform,
                context=context,
                config=config,
            )
            edge_better = bool(edge_gt < edge_initial)
            silhouette_better = bool(silhouette_gt < silhouette_initial)
            rows.append(
                {
                    "sample_index": sample_index,
                    "sample_token": sample.token,
                    "camera_name": sample.camera_name,
                    "time_delta_ms": sample.time_delta_ms,
                    "kind": kind,
                    "axis": axis,
                    "magnitude": magnitude,
                    "initial_rotation_error_deg": initial_delta["rotation_deg"],
                    "initial_translation_error_m": initial_delta["translation_norm_m"],
                    "projected_point_count": projection["projected_point_count"],
                    "projected_point_ratio": projection["projected_point_ratio"],
                    "projected_bbox_area_ratio": projection["bbox_px"]["area_ratio"],
                    "edge_gt_cost_improvement": edge_initial - edge_gt,
                    "edge_gt_correction_better": edge_better,
                    "silhouette_gt_cost_improvement": silhouette_initial
                    - silhouette_gt,
                    "silhouette_gt_correction_better": silhouette_better,
                    "both_gt_correction_better": edge_better and silhouette_better,
                }
            )

    summary_rows = _summarize_rows(rows)
    write_table_csv(diagnostics_dir / "objective_landscape.csv", rows)
    write_table_csv(diagnostics_dir / "objective_summary.csv", summary_rows)
    overall = {
        "sample_count": len(samples),
        "case_count": len(cases),
        "row_count": len(rows),
        "edge_gt_better_rate": sum(row["edge_gt_correction_better"] for row in rows)
        / max(len(rows), 1),
        "silhouette_gt_better_rate": sum(
            row["silhouette_gt_correction_better"] for row in rows
        )
        / max(len(rows), 1),
        "both_gt_better_rate": sum(row["both_gt_correction_better"] for row in rows)
        / max(len(rows), 1),
        "manifest": manifest,
    }
    with (diagnostics_dir / "precision_audit_summary.yaml").open(
        "w", encoding="utf-8"
    ) as file:
        yaml.safe_dump(overall, file, sort_keys=False)
    logging.info("precision audit outputs: %s", output_dir)
    logging.info(
        "GT correction better rates: edge=%.3f silhouette=%.3f both=%.3f",
        overall["edge_gt_better_rate"],
        overall["silhouette_gt_better_rate"],
        overall["both_gt_better_rate"],
    )


if __name__ == "__main__":
    main()
