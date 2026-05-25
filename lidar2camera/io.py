from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from calibration_common.evaluation import (
    write_acceptance_artifacts,
    write_paradigm_artifacts,
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
    if isinstance(value, list):
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
        yaml.safe_dump(extraction_report, file, sort_keys=False)
    with (diagnostics_dir / "optimization.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(optimization_report, file, sort_keys=False)
    with (diagnostics_dir / "evaluation.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(evaluation_report, file, sort_keys=False)

    acceptance_artifacts = write_acceptance_artifacts(
        diagnostics_dir, metrics_output["final_acceptance"]
    )
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
        "status": metrics_output["final_acceptance"]["status"],
        "release_ready": metrics_output["final_acceptance"]["release_ready"],
        "quality_gates": metrics_output["final_acceptance"]["gates"],
        "coarse_statuses": metrics_output["coarse_metrics"]["statuses"],
        "pairing_summary": extraction_report.get("pairing_summary", {}),
        "accepted_pair_ratio": extraction_report.get("accepted_pair_ratio"),
        "skip_reason_counts": extraction_report.get("skip_reason_counts", {}),
        "recommendation": metrics_output["final_acceptance"]["recommendation"],
    }
    visual_review = [
        str(output_dir / "metrics.yaml"),
        str(diagnostics_dir / "reference_dataset.yaml"),
        str(diagnostics_dir / "extraction.yaml"),
        str(diagnostics_dir / "optimization.yaml"),
        str(diagnostics_dir / "evaluation.yaml"),
    ]
    if overlay_artifact is not None:
        visual_review.append(overlay_artifact)
    else:
        visual_review.append(
            "No overlay artifact was generated; keep this run review-only until visual evidence is available."
        )
    visualization_index = {
        "schema_version": 1,
        "module": "lidar2camera",
        "layers": {
            "conclusion": [
                acceptance_artifacts["acceptance_report"],
                acceptance_artifacts["status_summary_csv"],
            ],
            "detail_metrics": [
                str(output_dir / "metrics.yaml"),
                str(diagnostics_dir / "reference_dataset.yaml"),
                str(diagnostics_dir / "extraction.yaml"),
                str(diagnostics_dir / "optimization.yaml"),
                str(diagnostics_dir / "evaluation.yaml"),
            ],
            "visual_review": visual_review,
        },
        "manual_review": [
            "Read diagnostics/standardized_data.yaml to confirm accepted/rejected sample counts and capture assumptions.",
            "Read diagnostics/data_quality.yaml to inspect extraction_yield, image_coverage, pose_diversity, and board_geometry gates.",
            "Inspect the checkerboard corner coverage and extraction skip reasons before promoting a run.",
            "Inspect the overlay for board-edge alignment and depth consistency, not only aggregate RMS.",
            "Treat missing holdout or repeatability evidence as review-only, not release-ready.",
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
        yaml.safe_dump(metrics_output, file, sort_keys=False)

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
                "overlay": overlay_artifact,
            },
        },
    }
    with (diagnostics_dir / "manifest.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(manifest, file, sort_keys=False)
    return manifest
