#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import yaml

from lidar2camera.nuscenes_benchmark import (
    EdgeRefinementConfig,
    NuScenesBenchmarkConfig,
    run_nuscenes_benchmark,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _to_float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_method_summary(metrics_path: Path) -> dict[str, dict[str, Any]]:
    csv_path = metrics_path.parent / "diagnostics" / "per_method_summary.csv"
    if not csv_path.exists():
        return {}
    summaries: dict[str, dict[str, Any]] = {}
    with csv_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            method = str(row.get("method", "")).strip()
            if not method:
                continue
            summaries[method] = row
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run nuScenes benchmark in split directories: baseline methods vs "
            "SensorsCalibration-style methods."
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
    parser.add_argument(
        "--data-root",
        default=None,
        help=(
            "Optional nuScenes train root containing samples/ and sweeps/. "
            "Defaults to the directory containing --info-path."
        ),
    )
    parser.add_argument(
        "--camera-names",
        default="CAM_FRONT",
        help="Comma-separated camera names, for example CAM_FRONT,CAM_FRONT_LEFT.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=4,
        help="Maximum number of camera samples to benchmark.",
    )
    parser.add_argument(
        "--sample-tokens",
        default="",
        help="Optional comma-separated sample tokens to benchmark explicitly.",
    )
    parser.add_argument(
        "--reference-transform-mode",
        default="rigid_sensor",
        help="Reference transform mode: rigid_sensor or sample_pair.",
    )
    parser.add_argument(
        "--max-sensor-time-delta-ms",
        type=float,
        default=40.0,
        help="Maximum allowed camera/lidar timestamp delta in milliseconds.",
    )
    parser.add_argument(
        "--rotation-perturb-deg",
        default="0.5,1.0,2.0",
        help="Comma-separated rotation perturbation magnitudes in degrees.",
    )
    parser.add_argument(
        "--translation-perturb-m",
        default="0.02,0.05,0.10",
        help="Comma-separated translation perturbation magnitudes in meters.",
    )
    parser.add_argument(
        "--perturbations-per-level",
        type=int,
        default=2,
        help="How many random perturbations to sample for each difficulty level.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=7,
        help="Random seed used for perturbation generation.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/lidar2camera/nuscenes_split_benchmark",
        help=(
            "Split benchmark root. Two subdirectories will be written: "
            "baseline_algorithms/ and sensorscalibration_algorithms/."
        ),
    )
    parser.add_argument(
        "--baseline-methods",
        default=(
            "identity,edge_refine,direct_visual_refine,"
            "silhouette_refine,batch_hybrid_refine,oracle_gt"
        ),
        help="Comma-separated baseline method list.",
    )
    parser.add_argument(
        "--sensorscalib-methods",
        default="identity,sensorscalib_line_refine,oracle_gt",
        help="Comma-separated SensorsCalibration-style method list.",
    )
    parser.add_argument(
        "--image-downscale",
        type=float,
        default=2.0,
        help="Optional image downscale factor for faster refinement.",
    )
    parser.add_argument(
        "--intensity-percentile",
        type=float,
        default=75.0,
        help="LiDAR intensity percentile used by the optimization objective.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=12000,
        help="Maximum LiDAR points used by the optimization objective.",
    )
    parser.add_argument(
        "--visualization-max-range-m",
        type=float,
        default=80.0,
        help="Maximum XY range for dense visual point-cloud projection overlays.",
    )
    parser.add_argument(
        "--visualization-max-points",
        type=int,
        default=60000,
        help="Maximum LiDAR points used by visual projection overlays.",
    )
    parser.add_argument(
        "--overlay-point-radius-px",
        type=int,
        default=4,
        help="Point radius for depth-colored projection overlays.",
    )
    parser.add_argument(
        "--search-rotation-deg",
        type=float,
        default=1.5,
        help="Local search bound around the initial guess in degrees per axis.",
    )
    parser.add_argument(
        "--search-translation-m",
        type=float,
        default=0.08,
        help="Local search bound around the initial guess in meters per axis.",
    )
    parser.add_argument(
        "--optimizer-maxiter",
        type=int,
        default=60,
        help="Maximum Powell iterations for local refinement methods.",
    )
    parser.add_argument(
        "--disable-guard-methods",
        default="",
        help=(
            "Comma-separated method names whose local-update guard should be disabled, "
            "for white-box debugging (for example sensorscalib_line_refine)."
        ),
    )
    parser.add_argument(
        "--sensorscalib-line-dilation-px",
        type=int,
        default=3,
        help="Line-mask dilation radius in pixels for sensorscalib_line_refine.",
    )
    parser.add_argument(
        "--max-overlay-artifacts",
        type=int,
        default=6,
        help="Maximum overlay images to write per run.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir).expanduser().resolve()
    baseline_output = output_root / "baseline_algorithms"
    sensors_output = output_root / "sensorscalibration_algorithms"

    shared_kwargs = dict(
        info_path=args.info_path,
        data_root=args.data_root,
        camera_names=_parse_csv_strings(args.camera_names),
        sample_limit=args.sample_limit,
        sample_tokens=_parse_csv_strings(args.sample_tokens),
        reference_transform_mode=args.reference_transform_mode,
        max_sensor_time_delta_ms=args.max_sensor_time_delta_ms,
        rotation_perturb_deg=_parse_csv_floats(args.rotation_perturb_deg),
        translation_perturb_m=_parse_csv_floats(args.translation_perturb_m),
        perturbations_per_level=args.perturbations_per_level,
        random_seed=args.random_seed,
        max_overlay_artifacts=args.max_overlay_artifacts,
        edge_refinement=EdgeRefinementConfig(
            image_downscale=args.image_downscale,
            intensity_percentile=args.intensity_percentile,
            max_points=args.max_points,
            visualization_max_point_xy_range_m=args.visualization_max_range_m,
            visualization_max_points=args.visualization_max_points,
            overlay_point_radius_px=args.overlay_point_radius_px,
            search_rotation_deg=args.search_rotation_deg,
            search_translation_m=args.search_translation_m,
            optimizer_maxiter=args.optimizer_maxiter,
            disable_update_guard_methods=_parse_csv_strings(args.disable_guard_methods),
            sensorscalib_line_dilation_px=args.sensorscalib_line_dilation_px,
        ),
    )

    baseline_config = NuScenesBenchmarkConfig(
        methods=_parse_csv_strings(args.baseline_methods),
        output_dir=str(baseline_output),
        **shared_kwargs,
    )
    sensors_config = NuScenesBenchmarkConfig(
        methods=_parse_csv_strings(args.sensorscalib_methods),
        output_dir=str(sensors_output),
        **shared_kwargs,
    )

    baseline_result = run_nuscenes_benchmark(baseline_config)
    sensors_result = run_nuscenes_benchmark(sensors_config)

    baseline_metrics = Path(baseline_result["artifact_paths"]["metrics"])
    sensors_metrics = Path(sensors_result["artifact_paths"]["metrics"])
    baseline_summary = _load_method_summary(baseline_metrics)
    sensors_summary = _load_method_summary(sensors_metrics)
    identity_baseline = baseline_summary.get("identity", {})
    sensors_candidate = sensors_summary.get("sensorscalib_line_refine", {})

    compare_manifest = {
        "baseline_output_dir": str(baseline_output),
        "sensorscalibration_output_dir": str(sensors_output),
        "baseline_metrics": str(baseline_metrics),
        "sensorscalibration_metrics": str(sensors_metrics),
        "baseline_methods": list(_parse_csv_strings(args.baseline_methods)),
        "sensorscalibration_methods": list(
            _parse_csv_strings(args.sensorscalib_methods)
        ),
        "quick_compare": {
            "identity_mean_rotation_error_deg": _to_float_or_none(
                identity_baseline.get("mean_final_rotation_error_deg")
            ),
            "identity_mean_translation_error_m": _to_float_or_none(
                identity_baseline.get("mean_final_translation_error_m")
            ),
            "sensorscalib_mean_rotation_error_deg": _to_float_or_none(
                sensors_candidate.get("mean_final_rotation_error_deg")
            ),
            "sensorscalib_mean_translation_error_m": _to_float_or_none(
                sensors_candidate.get("mean_final_translation_error_m")
            ),
            "sensorscalib_accepted_update_rate": _to_float_or_none(
                sensors_candidate.get("accepted_update_rate")
            ),
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    compare_manifest_path = output_root / "split_compare_summary.yaml"
    with compare_manifest_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(compare_manifest, file, sort_keys=False)

    logging.info("baseline outputs: %s", baseline_output)
    logging.info("sensorscalibration outputs: %s", sensors_output)
    logging.info("split compare summary: %s", compare_manifest_path)


if __name__ == "__main__":
    main()
