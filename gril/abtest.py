"""A/B report assembly for frozen and ROS-free GRIL runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from gril.comparison import compare_datasets, compare_results, parse_gril_result
from gril.config import compare_configs
from gril.dataset_io import file_sha256, load_dataset
from gril.reference import REFERENCE


def build_abtest_report(
    reference_dataset_path: Path,
    candidate_dataset_path: Path,
    reference_result_path: Path,
    candidate_result_path: Path,
    reference_config_path: Path,
    candidate_config_path: Path,
    reference_trace_path: Path,
    candidate_trace_path: Path,
    *,
    point_tolerance: float = 1e-6,
    point_time_tolerance_s: float = 1e-8,
    imu_tolerance: float = 1e-12,
    rotation_tolerance_deg: float = 0.2,
    translation_tolerance_m: float = 0.03,
    time_tolerance_s: float = 0.001,
) -> dict[str, Any]:
    if Path(reference_trace_path).resolve() == Path(candidate_trace_path).resolve():
        raise ValueError(
            "Reference and candidate batch traces must be distinct run artifacts"
        )
    reference_trace_hash = file_sha256(reference_trace_path)
    candidate_trace_hash = file_sha256(candidate_trace_path)
    trace_comparison = {
        "verdict": (
            "equivalent"
            if reference_trace_hash == candidate_trace_hash
            else "different"
        ),
        "schema": "GRIL_BATCH_TRACE 1",
        "reference_path": str(Path(reference_trace_path).resolve()),
        "candidate_path": str(Path(candidate_trace_path).resolve()),
        "reference_sha256": reference_trace_hash,
        "candidate_sha256": candidate_trace_hash,
    }
    config_comparison = compare_configs(reference_config_path, candidate_config_path)
    input_comparison = compare_datasets(
        load_dataset(reference_dataset_path),
        load_dataset(candidate_dataset_path),
        point_tolerance=point_tolerance,
        point_time_tolerance_s=point_time_tolerance_s,
        imu_tolerance=imu_tolerance,
    )
    result_comparison = compare_results(
        parse_gril_result(reference_result_path),
        parse_gril_result(candidate_result_path),
        rotation_tolerance_deg=rotation_tolerance_deg,
        translation_tolerance_m=translation_tolerance_m,
        time_tolerance_s=time_tolerance_s,
    )
    checks = {
        "batch_trace_equivalence": trace_comparison["verdict"] == "equivalent",
        "config_equivalence": config_comparison["verdict"] == "equivalent",
        "input_equivalence": input_comparison["verdict"] == "equivalent",
        "result_equivalence": result_comparison["verdict"] == "equivalent",
    }
    return {
        "verdict": "equivalent" if all(checks.values()) else "different",
        "checks": checks,
        "methods": {
            "reference": REFERENCE,
            "candidate": {
                "name": "gril_native",
                "runtime": "ROS-free native C++",
                "algorithm": "GRIL migration candidate",
            },
        },
        "batch_trace": trace_comparison,
        "configuration": config_comparison,
        "inputs": input_comparison,
        "results": result_comparison,
        "limitations": [
            "Final-result equivalence does not establish frontend-state equivalence.",
            "Repeatability and independent physical holdout remain mandatory.",
        ],
    }


def write_abtest_report(report: dict[str, Any], output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "comparison.yaml"
    output_path.write_text(yaml.safe_dump(report, sort_keys=False))
    return output_path
