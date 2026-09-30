#!/usr/bin/env python3
"""Run a paired full-FOV versus shared empirical-FOV LiDAR registration study."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# noqa is needed because the repository root is added to sys.path immediately above.
from lidar2lidar.extrinsic_io import load_extrinsics_file  # noqa: E402
from lidar2lidar.fov import estimate_angular_support, shared_fov_clouds  # noqa: E402
from lidar2lidar.lidar2lidar import (  # noqa: E402
    calibrate_lidar_extrinsic,
    compute_fpfh_features,
    perform_coarse_registration,
    preprocess_point_cloud,
)
from lidar2lidar.prepared_dataset import collect_record_bundle  # noqa: E402
from lidar2lidar.record_utils import (  # noqa: E402
    build_transform_graph,
    discover_record_files,
    find_synchronized_pairs,
    lookup_transform,
    prefetch_pointcloud_cache,
)

LOGGER = logging.getLogger("shared_fov_ablation")
VARIANTS = ("full_fov", "shared_fov")
MIN_CROPPED_POINTS = 100


def configure_logging() -> None:
    logging.basicConfig(level=logging.WARNING)
    LOGGER.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.propagate = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the existing full-view registration baseline against "
            "registration restricted to the pair's shared empirical azimuth FOV."
        )
    )
    parser.add_argument(
        "--record-path",
        required=True,
        help="A record file (used alone) or a directory of record files.",
    )
    parser.add_argument("--source-topic", required=True)
    parser.add_argument("--target-topic", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--include-record-family",
        action="store_true",
        help="When record-path is a file, include adjacent numbered record shards.",
    )
    parser.add_argument("--initial-transform", help="Optional source-to-target YAML.")
    parser.add_argument("--max-pairs", type=int, default=24)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--sync-threshold-ms", type=float, default=20.0)
    parser.add_argument("--voxel-size", type=float, default=0.1)
    parser.add_argument("--fov-coverage", type=float, default=0.995)
    parser.add_argument("--fov-margin-deg", type=float, default=1.0)
    parser.add_argument("--method", type=int, default=1)
    return parser.parse_args()


def _select_record_files(record_path: Path, include_family: bool) -> list[str]:
    if record_path.is_dir():
        return discover_record_files(str(record_path))
    if not record_path.is_file():
        raise FileNotFoundError(f"Record path does not exist: {record_path}")
    if include_family:
        return discover_record_files(str(record_path))
    return [str(record_path)]


def _load_initial_transform(args, bundle) -> tuple[np.ndarray, str]:
    if args.initial_transform:
        transform, *_ = load_extrinsics_file(args.initial_transform)
        transform = np.asarray(transform, dtype=float).reshape(4, 4)
        return transform, f"file:{args.initial_transform}"

    graph = build_transform_graph(bundle.tf_edges)
    source_frame = bundle.topic_frame_ids[args.source_topic]
    target_frame = bundle.topic_frame_ids[args.target_topic]
    transform = lookup_transform(graph, source_frame, target_frame)
    if transform is not None:
        return np.asarray(transform, dtype=float).reshape(4, 4), "record_tf"
    return None, "fpfh_ransac_from_first_training_pair"


def _estimate_global_seed(
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    voxel_size: float,
) -> tuple[np.ndarray, dict]:
    params = {
        "voxel_size": voxel_size,
        "nb_neighbors": 20,
        "std_ratio": 2.0,
        "plane_dist_thresh": 0.05,
        "remove_ground": False,
        "remove_walls": False,
    }
    source_processed = preprocess_point_cloud(source_cloud, **params)
    target_processed = preprocess_point_cloud(target_cloud, **params)
    if len(source_processed.points) < 10 or len(target_processed.points) < 10:
        raise RuntimeError("Too few points for FPFH/RANSAC initialization.")
    source_features = compute_fpfh_features(source_processed, voxel_size)
    target_features = compute_fpfh_features(target_processed, voxel_size)
    coarse_result = perform_coarse_registration(
        source_processed,
        target_processed,
        source_features,
        target_features,
        voxel_size,
    )
    transform = np.asarray(coarse_result.transformation, dtype=float).reshape(4, 4)
    if not np.isfinite(transform).all() or len(coarse_result.correspondence_set) < 3:
        raise RuntimeError(
            "FPFH/RANSAC did not produce a usable common initialization "
            f"(fitness={coarse_result.fitness}, "
            f"correspondences={len(coarse_result.correspondence_set)})."
        )
    return transform, {
        "fitness": float(coarse_result.fitness),
        "inlier_rmse": float(coarse_result.inlier_rmse),
        "correspondences": int(len(coarse_result.correspondence_set)),
    }


def _rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _transform_delta(first: np.ndarray, second: np.ndarray) -> dict:
    return {
        "translation_m": float(np.linalg.norm(first[:3, 3] - second[:3, 3])),
        "rotation_deg": _rotation_distance_deg(first, second),
    }


def _training_medoid(results: list[dict]) -> tuple[np.ndarray | None, int | None]:
    successful = [
        (index, item["transform"])
        for index, item in enumerate(results)
        if item["split"] == "train" and item["status"] == "success"
    ]
    if not successful:
        return None, None
    transforms = [np.asarray(transform, dtype=float) for _, transform in successful]
    distances = np.zeros((len(transforms), len(transforms)), dtype=float)
    for first_index, first in enumerate(transforms):
        for second_index, second in enumerate(transforms):
            delta = _transform_delta(first, second)
            distances[first_index, second_index] = (
                delta["translation_m"] / 0.1 + delta["rotation_deg"] / 1.0
            )
    medoid_local_index = int(np.argmin(np.sum(distances, axis=1)))
    result_index = successful[medoid_local_index][0]
    return transforms[medoid_local_index], result_index


def _evaluate(
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    transform: np.ndarray,
    distance: float,
) -> dict:
    if source_cloud.is_empty() or target_cloud.is_empty():
        return {"fitness": None, "inlier_rmse": None, "correspondences": 0}
    result = o3d.pipelines.registration.evaluate_registration(
        source_cloud, target_cloud, distance, transform
    )
    return {
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
        "correspondences": int(len(result.correspondence_set)),
    }


def _registration_metrics(result) -> dict:
    if result is None:
        return {
            "status": "no_result",
            "fitness": None,
            "inlier_rmse": None,
            "correspondences": 0,
        }
    return {
        "status": "success",
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
        "correspondences": int(len(result.correspondence_set)),
    }


def _colored_alignment(
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    transform: np.ndarray,
    output_path: Path,
    voxel_size: float,
) -> None:
    source = copy.deepcopy(source_cloud)
    target = copy.deepcopy(target_cloud)
    source.transform(transform)
    source.paint_uniform_color([0.95, 0.25, 0.12])
    target.paint_uniform_color([0.10, 0.65, 0.95])
    merged = source + target
    if voxel_size > 0:
        merged = merged.voxel_down_sample(voxel_size)
    if not o3d.io.write_point_cloud(str(output_path), merged):
        raise RuntimeError(f"Failed to write alignment preview: {output_path}")


def _summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "median": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def run_ablation(args: argparse.Namespace) -> dict:
    record_path = Path(args.record_path).expanduser().resolve()
    record_files = _select_record_files(record_path, args.include_record_family)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; choose a new path: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = collect_record_bundle(
        record_path=str(record_path),
        lidar_topics=[args.source_topic, args.target_topic],
        pose_topic="",
        imu_topic=None,
        parent_frame="base_link",
        record_files=record_files,
    )
    source_metas = bundle.metadata_by_topic[args.source_topic]
    target_metas = bundle.metadata_by_topic[args.target_topic]
    pairs = find_synchronized_pairs(
        source_metas,
        target_metas,
        max_delta_ns=int(args.sync_threshold_ms * 1e6),
        max_pairs=args.max_pairs,
    )
    if len(pairs) < 4:
        raise RuntimeError(
            f"Only {len(pairs)} synchronized pairs; at least 4 are needed "
            "for train/holdout comparison."
        )
    train_count = min(
        len(pairs) - 1,
        max(1, int(np.floor(len(pairs) * args.train_fraction))),
    )
    holdout_pairs = pairs[train_count:]

    cached_clouds = prefetch_pointcloud_cache(
        [
            meta
            for source_meta, target_meta, _ in pairs
            for meta in (source_meta, target_meta)
        ]
    )
    cloud_pairs = [
        (
            source_meta,
            target_meta,
            delta_ns,
            cached_clouds[(source_meta.topic, source_meta.timestamp_ns)],
            cached_clouds[(target_meta.topic, target_meta.timestamp_ns)],
        )
        for source_meta, target_meta, delta_ns in pairs
    ]
    train_cloud_pairs = cloud_pairs[:train_count]

    source_support = estimate_angular_support(
        [np.asarray(source.points) for _, _, _, source, _ in train_cloud_pairs],
        coverage_ratio=args.fov_coverage,
        margin_deg=args.fov_margin_deg,
    )
    target_support = estimate_angular_support(
        [np.asarray(target.points) for _, _, _, _, target in train_cloud_pairs],
        coverage_ratio=args.fov_coverage,
        margin_deg=args.fov_margin_deg,
    )

    initial_transform, initial_source = _load_initial_transform(args, bundle)
    coarse_metrics = None
    if initial_transform is None:
        first = train_cloud_pairs[0]
        initial_transform, coarse_metrics = _estimate_global_seed(
            first[3], first[4], args.voxel_size
        )
    if initial_transform.shape != (4, 4) or not np.isfinite(initial_transform).all():
        raise RuntimeError("The shared source-to-target initial transform is invalid.")

    preprocessing_params = {
        "voxel_size": float(args.voxel_size),
        "nb_neighbors": 20,
        "std_ratio": 2.0,
        "plane_dist_thresh": 0.05,
        "height_range": None,
        "remove_ground": False,
        "remove_walls": False,
    }
    preprocessing_cache = {}
    per_window = []
    for window_index, (
        source_meta,
        target_meta,
        delta_ns,
        source_cloud,
        target_cloud,
    ) in enumerate(cloud_pairs):
        split = "train" if window_index < train_count else "holdout"
        clipped_source, clipped_target, crop_metrics = shared_fov_clouds(
            source_cloud,
            target_cloud,
            initial_transform,
            source_support,
            target_support,
        )
        result_row = {
            "window_index": window_index,
            "split": split,
            "source_timestamp_ns": int(source_meta.timestamp_ns),
            "target_timestamp_ns": int(target_meta.timestamp_ns),
            "sync_delta_ms": float(delta_ns / 1e6),
            "variants": {},
        }
        for variant, source_input, target_input in (
            ("full_fov", source_cloud, target_cloud),
            ("shared_fov", clipped_source, clipped_target),
        ):
            variant_result = {
                "input_source_points": int(len(source_input.points)),
                "input_target_points": int(len(target_input.points)),
                "crop": crop_metrics if variant == "shared_fov" else None,
            }
            if (
                len(source_input.points) < MIN_CROPPED_POINTS
                or len(target_input.points) < MIN_CROPPED_POINTS
            ):
                variant_result.update(
                    {
                        "status": "no_result",
                        "failure_reason": "fewer_than_100_input_points",
                        "fitness": None,
                        "inlier_rmse": None,
                        "correspondences": 0,
                        "transform": None,
                    }
                )
            else:
                final_transform, _, registration_result = calibrate_lidar_extrinsic(
                    source_input,
                    target_input,
                    preprocessing_params=preprocessing_params,
                    method=args.method,
                    initial_transform=initial_transform,
                    preprocessing_cache=preprocessing_cache,
                )
                variant_result.update(_registration_metrics(registration_result))
                if (
                    final_transform is None
                    or not np.isfinite(final_transform).all()
                    or registration_result is None
                ):
                    variant_result.update(
                        {
                            "status": "no_result",
                            "failure_reason": "registration_returned_no_result",
                            "transform": None,
                        }
                    )
                else:
                    variant_result["transform"] = np.asarray(
                        final_transform, dtype=float
                    ).tolist()
            result_row["variants"][variant] = variant_result
        per_window.append(result_row)
        LOGGER.info(
            "Window %d/%d (%s): full=%s shared=%s",
            window_index + 1,
            len(cloud_pairs),
            split,
            result_row["variants"]["full_fov"]["status"],
            result_row["variants"]["shared_fov"]["status"],
        )

    train_results = {}
    for variant in VARIANTS:
        result_views = [
            {
                "split": row["split"],
                "status": row["variants"][variant]["status"],
                "transform": row["variants"][variant]["transform"],
            }
            for row in per_window
        ]
        medoid, medoid_index = _training_medoid(result_views)
        deviations = []
        if medoid is not None:
            for row in per_window:
                result = row["variants"][variant]
                if result["status"] == "success":
                    delta = _transform_delta(
                        medoid, np.asarray(result["transform"], dtype=float)
                    )
                    result["delta_from_training_medoid"] = delta
                    if row["split"] == "holdout":
                        deviations.append(delta)
        train_results[variant] = {
            "training_medoid": (medoid.tolist() if medoid is not None else None),
            "medoid_window_index": medoid_index,
            "training_success_count": sum(
                row["split"] == "train"
                and row["variants"][variant]["status"] == "success"
                for row in per_window
            ),
            "holdout_solve_success_count": sum(
                row["split"] == "holdout"
                and row["variants"][variant]["status"] == "success"
                for row in per_window
            ),
            "holdout_transform_deviation": {
                "translation_m": _summary(
                    [item["translation_m"] for item in deviations]
                ),
                "rotation_deg": _summary([item["rotation_deg"] for item in deviations]),
            },
        }

    evaluation_threshold = max(args.voxel_size * 5.0, 0.1)
    holdout_evaluation = []
    for row, pair in zip(per_window, cloud_pairs):
        if row["split"] != "holdout":
            continue
        source_cloud, target_cloud = pair[3], pair[4]
        clipped_source, clipped_target, crop_metrics = shared_fov_clouds(
            source_cloud,
            target_cloud,
            initial_transform,
            source_support,
            target_support,
        )
        evaluation_row = {
            "window_index": row["window_index"],
            "source_timestamp_ns": row["source_timestamp_ns"],
            "target_timestamp_ns": row["target_timestamp_ns"],
            "crop": crop_metrics,
            "variants": {},
        }
        for variant in VARIANTS:
            medoid_data = train_results[variant]["training_medoid"]
            if medoid_data is None:
                evaluation_row["variants"][variant] = {"status": "no_training_medoid"}
                continue
            medoid = np.asarray(medoid_data, dtype=float)
            evaluation_row["variants"][variant] = {
                "full_scene": _evaluate(
                    source_cloud,
                    target_cloud,
                    medoid,
                    evaluation_threshold,
                ),
                "fixed_shared_support": _evaluate(
                    clipped_source,
                    clipped_target,
                    medoid,
                    evaluation_threshold,
                ),
            }
        holdout_evaluation.append(evaluation_row)

    if holdout_evaluation:
        preview = holdout_evaluation[len(holdout_evaluation) // 2]
        preview_pair = cloud_pairs[preview["window_index"]]
        preview_source, preview_target = preview_pair[3], preview_pair[4]
        clipped_source, clipped_target, _ = shared_fov_clouds(
            preview_source,
            preview_target,
            initial_transform,
            source_support,
            target_support,
        )
        for variant, filename in (
            ("full_fov", "holdout_full_scene_baseline.ply"),
            ("shared_fov", "holdout_shared_support_candidate.ply"),
        ):
            medoid_data = train_results[variant]["training_medoid"]
            if medoid_data is None:
                continue
            source_preview = preview_source if variant == "full_fov" else clipped_source
            target_preview = preview_target if variant == "full_fov" else clipped_target
            _colored_alignment(
                source_preview,
                target_preview,
                np.asarray(medoid_data, dtype=float),
                output_dir / filename,
                args.voxel_size,
            )
        full_medoid = train_results["full_fov"]["training_medoid"]
        if full_medoid is not None:
            _colored_alignment(
                clipped_source,
                clipped_target,
                np.asarray(full_medoid, dtype=float),
                output_dir / "holdout_shared_support_baseline.ply",
                args.voxel_size,
            )

    report = {
        "experiment": "full_fov_vs_shared_empirical_fov",
        "interpretation": (
            "Empirical angular support is estimated from training scans only; "
            "it is not a manufacturer-rated field of view. The shared crop is "
            "fixed from the common initialization for both variants' comparison."
        ),
        "input": {
            "record_files": record_files,
            "source_topic": args.source_topic,
            "target_topic": args.target_topic,
            "source_frame": bundle.topic_frame_ids[args.source_topic],
            "target_frame": bundle.topic_frame_ids[args.target_topic],
            "source_message_count": len(source_metas),
            "target_message_count": len(target_metas),
            "synchronized_pair_count": len(pairs),
        },
        "configuration": {
            "max_pairs": args.max_pairs,
            "train_fraction": args.train_fraction,
            "training_pair_count": train_count,
            "holdout_pair_count": len(holdout_pairs),
            "sync_threshold_ms": args.sync_threshold_ms,
            "voxel_size": args.voxel_size,
            "method": args.method,
            "evaluation_correspondence_distance_m": evaluation_threshold,
            "fov_coverage": args.fov_coverage,
            "fov_margin_deg": args.fov_margin_deg,
            "medoid_translation_scale_m": 0.1,
            "medoid_rotation_scale_deg": 1.0,
        },
        "initialization": {
            "source": initial_source,
            "source_to_target": initial_transform.tolist(),
            "coarse_registration": coarse_metrics,
        },
        "empirical_fov": {
            "source": source_support.as_dict(),
            "target": target_support.as_dict(),
        },
        "variant_summary": train_results,
        "holdout_fixed_transform_evaluation": holdout_evaluation,
        "per_window": per_window,
        "decision": (
            "INCONCLUSIVE_WITHOUT_INDEPENDENT_EXTRINSIC_REFERENCE; compare "
            "repeatability and holdout behavior, but do not interpret registration "
            "fitness or RMSE alone as extrinsic accuracy."
        ),
    }

    report_path = output_dir / "report.yaml"
    with report_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(report, stream, sort_keys=False, allow_unicode=True)

    with (output_dir / "per_window.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = [
            "window_index",
            "split",
            "source_timestamp_ns",
            "target_timestamp_ns",
            "sync_delta_ms",
            "variant",
            "status",
            "fitness",
            "inlier_rmse",
            "correspondences",
            "input_source_points",
            "input_target_points",
            "source_retained_ratio",
            "target_retained_ratio",
            "transform",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in per_window:
            for variant in VARIANTS:
                result = row["variants"][variant]
                crop = result.get("crop") or {}
                writer.writerow(
                    {
                        "window_index": row["window_index"],
                        "split": row["split"],
                        "source_timestamp_ns": row["source_timestamp_ns"],
                        "target_timestamp_ns": row["target_timestamp_ns"],
                        "sync_delta_ms": row["sync_delta_ms"],
                        "variant": variant,
                        "status": result["status"],
                        "fitness": result.get("fitness"),
                        "inlier_rmse": result.get("inlier_rmse"),
                        "correspondences": result.get("correspondences"),
                        "input_source_points": result["input_source_points"],
                        "input_target_points": result["input_target_points"],
                        "source_retained_ratio": crop.get("source_retained_ratio"),
                        "target_retained_ratio": crop.get("target_retained_ratio"),
                        "transform": json.dumps(result.get("transform")),
                    }
                )
    LOGGER.info("Ablation report: %s", report_path)
    return report


def main() -> None:
    configure_logging()
    args = parse_args()
    if args.max_pairs < 4:
        raise ValueError("--max-pairs must be at least 4.")
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be in (0, 1).")
    if args.sync_threshold_ms <= 0.0 or args.voxel_size <= 0.0:
        raise ValueError("Sync threshold and voxel size must be positive.")
    run_ablation(args)


if __name__ == "__main__":
    main()
