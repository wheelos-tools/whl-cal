from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from lidar2camera.models import ReferenceCalibrationDataset, ReferencePoseObservation
from lidar2lidar.extrinsic_io import (
    build_extrinsics_payload,
    extrinsics_filename,
    save_extrinsics_yaml,
)


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
        "metadata": dict(observation.metadata),
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
        "metadata": dict(dataset.metadata),
        "observations": [
            _observation_to_payload(observation) for observation in dataset.observations
        ],
    }


def prepare_output_layout(output_dir: Path) -> tuple[Path, Path, Path]:
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    for file_path in (
        output_dir / "calibrated_tf.yaml",
        output_dir / "metrics.yaml",
        diagnostics_dir / "manifest.yaml",
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
    with (output_dir / "metrics.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(metrics_output, file, sort_keys=False)
    with (diagnostics_dir / "reference_dataset.yaml").open(
        "w", encoding="utf-8"
    ) as file:
        yaml.safe_dump(dataset_to_payload(dataset), file, sort_keys=False)
    with (diagnostics_dir / "extraction.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(extraction_report, file, sort_keys=False)
    with (diagnostics_dir / "optimization.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(optimization_report, file, sort_keys=False)
    with (diagnostics_dir / "evaluation.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(evaluation_report, file, sort_keys=False)

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
                "overlay": overlay_artifact,
            },
        },
    }
    with (diagnostics_dir / "manifest.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(manifest, file, sort_keys=False)
    return manifest
