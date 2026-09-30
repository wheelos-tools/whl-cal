"""Post-run GRIL migration review assembly.

This module only reads completed-run artifacts.  It deliberately does not
invoke a frontend, change its configuration, select a preferred result, or
turn a diagnostic outcome into an execution-time gate.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from gril.comparison import compare_results, parse_gril_result
from gril.config import config_digest, load_algorithm_config
from gril.dataset_io import file_sha256
from gril.reference import REFERENCE

_REQUIRED_FULL_EVENTS = (
    "package",
    "propagated",
    "updated",
    "motion_start",
    "calibration_push",
)
_REFERENCE_PATCHES = (
    ("validation_patch", "validation_patch_sha256"),
    ("batch_trace_patch", "batch_trace_patch_sha256"),
    ("preprocess_trace_patch", "preprocess_trace_patch_sha256"),
    ("frontend_cv_trace_patch", "frontend_cv_trace_patch_sha256"),
    ("ground_trace_patch", "ground_trace_patch_sha256"),
    (
        "full_frontend_reference_trace_patch",
        "full_frontend_reference_trace_patch_sha256",
    ),
)


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a YAML mapping: {path}")
    return value


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _artifact_path(manifest: dict[str, Any], name: str) -> Path:
    artifacts = _require_mapping(manifest.get("artifacts"), "run manifest artifacts")
    artifact = _require_mapping(artifacts.get(name), f"run manifest artifact {name}")
    path = artifact.get("path")
    if not isinstance(path, str):
        raise ValueError(f"run manifest artifact {name} has no path")
    return Path(path)


def _artifact(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    artifacts = _require_mapping(manifest.get("artifacts"), "run manifest artifacts")
    return _require_mapping(artifacts.get(name), f"run manifest artifact {name}")


def _trace_counts(path: Path) -> Counter[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Full frontend trace not found: {path}")
    return Counter(
        line.split(maxsplit=1)[0]
        for line in path.read_text().splitlines()
        if line.strip()
    )


def _reference_identity(
    archive: dict[str, Any], evidence: dict[str, Any]
) -> dict[str, Any]:
    recorded_patches = archive.get("patches")
    if not isinstance(recorded_patches, list):
        raise ValueError("reference trace archive has no patch list")
    by_path = {
        item.get("path"): item
        for item in recorded_patches
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    patches = []
    patch_matches = []
    for path_key, hash_key in _REFERENCE_PATCHES:
        path = Path(REFERENCE[path_key]).resolve()
        expected = REFERENCE[hash_key]
        recorded = by_path.get(str(path), {})
        recorded_hash = recorded.get("sha256")
        actual = file_sha256(path)
        matches = actual == expected and recorded_hash == expected
        patches.append(
            {
                "path": str(path),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matches": matches,
            }
        )
        patch_matches.append(matches)

    distribution = _require_mapping(
        evidence.get("reference_distribution"), "reference_distribution"
    )
    distribution_actual = distribution.get("sha256")
    distribution_matches = distribution_actual == REFERENCE["source_archive_sha256"]
    source_revision = archive.get("source_revision")
    trace_archive_sha = archive.get("source_archive_sha256")
    return {
        "verdict": (
            "passed"
            if source_revision == REFERENCE["revision"]
            and distribution_matches
            and all(patch_matches)
            else "failed"
        ),
        "revision_expected": REFERENCE["revision"],
        "revision_actual": source_revision,
        "upstream_distribution_tarball_sha256_expected": REFERENCE[
            "source_archive_sha256"
        ],
        "upstream_distribution_tarball_sha256_actual": distribution_actual,
        "reference_trace_git_archive_sha256": trace_archive_sha,
        "patches": patches,
    }


def _input_identity(
    candidate: dict[str, Any],
    repeat: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    input_evidence = _require_mapping(evidence.get("inputs"), "inputs")
    candidate_input = _require_mapping(candidate.get("input"), "candidate input")
    repeat_input = _require_mapping(repeat.get("input"), "repeat input")
    dataset_path = Path(candidate_input["dataset"])
    dataset_hash = file_sha256(dataset_path)
    source_bag = Path(input_evidence["frozen_reference_bag"])
    source_bag_hash = file_sha256(source_bag)
    expected_bag_hash = input_evidence["frozen_reference_bag_sha256"]
    expected_dataset_hash = input_evidence["canonical_manifest_sha256"]
    checks = {
        "frozen_reference_bag": source_bag_hash == expected_bag_hash,
        "candidate_dataset": candidate_input.get("dataset_sha256") == dataset_hash,
        "canonical_dataset": dataset_hash == expected_dataset_hash,
        "repeat_dataset": repeat_input.get("dataset_sha256") == dataset_hash,
        "canonical_arrays": bool(input_evidence["canonical_array_hashes_verified"]),
        "canonical_contract": input_evidence["canonical_input_contract"] == "accepted",
        "record_to_bag_contract": str(
            input_evidence["record_to_bag_contract"]
        ).startswith("accepted"),
    }
    return {
        "verdict": "passed" if all(checks.values()) else "failed",
        "frozen_reference_bag_sha256": source_bag_hash,
        "canonical_manifest_sha256": dataset_hash,
        "canonical_counts": input_evidence["canonical_counts"],
        "canonical_array_hashes_verified": bool(
            input_evidence["canonical_array_hashes_verified"]
        ),
        "canonical_input_contract": input_evidence["canonical_input_contract"],
        "record_to_bag_contract": input_evidence["record_to_bag_contract"],
        "checks": checks,
    }


def _configuration_identity(
    reference_config: Path, candidate: dict[str, Any], repeat: dict[str, Any]
) -> dict[str, Any]:
    expected = config_digest(load_algorithm_config(reference_config))
    candidate_config = _require_mapping(
        candidate.get("algorithm_config"), "candidate algorithm_config"
    )
    repeat_config = _require_mapping(
        repeat.get("algorithm_config"), "repeat algorithm_config"
    )
    candidate_digest = candidate_config.get("digest")
    repeat_digest = repeat_config.get("digest")
    candidate_runtime = _require_mapping(
        candidate.get("runtime_semantics"), "candidate runtime_semantics"
    )
    repeat_runtime = _require_mapping(
        repeat.get("runtime_semantics"), "repeat runtime_semantics"
    )
    checks = {
        "candidate_matches_reference": candidate_digest == expected,
        "repeat_matches_reference": repeat_digest == expected,
        "candidate_has_no_execution_quality_gate": candidate_runtime.get(
            "quality_gating_applied"
        )
        is False,
        "repeat_has_no_execution_quality_gate": repeat_runtime.get(
            "quality_gating_applied"
        )
        is False,
    }
    return {
        "verdict": "passed" if all(checks.values()) else "failed",
        "effective_reference_digest": expected,
        "native_algorithm_digest": candidate_digest,
        "repeat_native_algorithm_digest": repeat_digest,
        "explicit_native_config_sha256": candidate_config.get("native_sha256"),
        "repeat_native_config_sha256": repeat_config.get("native_sha256"),
        "checks": checks,
    }


def _full_trace_coverage(
    reference_archive: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    traces = reference_archive.get("traces")
    if not isinstance(traces, list) or not traces or not isinstance(traces[0], dict):
        raise ValueError("reference trace archive has no trace entry")
    reference_trace = traces[0]
    reference_counts = _require_mapping(
        reference_trace.get("event_counts"), "reference trace event_counts"
    )
    reference_trace_path = Path(reference_trace["path"])
    candidate_trace = _artifact(candidate, "full_frontend_trace")
    candidate_trace_path = Path(candidate_trace["path"])
    reference_hash = file_sha256(reference_trace_path)
    candidate_hash = file_sha256(candidate_trace_path)
    candidate_counts = _trace_counts(candidate_trace_path)
    required = {
        event: {
            "reference": reference_counts.get(event),
            "candidate": candidate_counts.get(event, 0),
            "matches": reference_counts.get(event) == candidate_counts.get(event, 0),
        }
        for event in _REQUIRED_FULL_EVENTS
    }
    return {
        "verdict": (
            "complete_execution_coverage"
            if all(entry["matches"] for entry in required.values())
            and reference_hash == reference_trace.get("sha256")
            and candidate_hash == candidate_trace.get("sha256")
            else "incomplete_execution_coverage"
        ),
        "reference_trace_sha256": reference_hash,
        "candidate_trace_sha256": candidate_hash,
        "reference_trace_hash_verified": reference_hash
        == reference_trace.get("sha256"),
        "candidate_trace_hash_verified": candidate_hash
        == candidate_trace.get("sha256"),
        "required_event_counts": required,
        "reference_total_packages": reference_counts.get("package"),
        "candidate_total_packages": candidate_counts.get("package", 0),
        "native_terminal_records": {
            name: candidate_counts.get(name, 0)
            for name in ("end_package", "batch_handoff", "batch_complete", "END")
        },
    }


def _component_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    components = _require_mapping(
        evidence.get("proven_components"), "proven_components"
    )
    required = (
        "preprocess",
        "fifo_synchronization",
        "constant_velocity_propagation",
        "patchwork_ground",
        "isolated_odometry_ekf",
    )
    missing = [name for name in required if name not in components]
    if missing:
        raise ValueError("proven_components is missing: " + ", ".join(missing))
    return {name: components[name] for name in required}


def build_full_ab_review(
    *,
    reference_trace_archive_path: Path,
    reference_result_path: Path,
    reference_config_path: Path,
    candidate_manifest_path: Path,
    repeat_manifest_path: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    """Build a read-only review of completed full GRIL runs.

    ``evidence_path`` carries immutable conclusions from component traces and
    physical validation that are not derivable from a final result file.  It
    must not be generated by, or consumed by, the calibration executable.
    """

    archive = _load_yaml(reference_trace_archive_path, "reference trace archive")
    candidate = _load_yaml(candidate_manifest_path, "candidate run manifest")
    repeat = _load_yaml(repeat_manifest_path, "repeat run manifest")
    evidence = _load_yaml(evidence_path, "review evidence")
    if evidence.get("schema") != "GRIL_FULL_REVIEW_EVIDENCE 1":
        raise ValueError("Unsupported review evidence schema")

    reference_identity = _reference_identity(archive, evidence)
    inputs = _input_identity(candidate, repeat, evidence)
    configuration = _configuration_identity(reference_config_path, candidate, repeat)
    components = _component_evidence(evidence)
    coverage = _full_trace_coverage(archive, candidate)
    candidate_result = _artifact_path(candidate, "result")
    repeat_result = _artifact_path(repeat, "result")
    result_comparison = compare_results(
        parse_gril_result(reference_result_path), parse_gril_result(candidate_result)
    )
    repeatability = compare_results(
        parse_gril_result(repeat_result), parse_gril_result(candidate_result)
    )
    limitation = _require_mapping(
        evidence.get("full_frontend_non_bitwise_limitation"),
        "full_frontend_non_bitwise_limitation",
    ).copy()
    limitation["is_acceptance_gate"] = False
    physical = _require_mapping(
        evidence.get("physical_validation_separate_from_migration"),
        "physical_validation_separate_from_migration",
    )

    gates = {
        "reference_identity": reference_identity["verdict"] == "passed",
        "input_identity_and_contract": inputs["verdict"] == "passed",
        "effective_config_identity": configuration["verdict"] == "passed",
        "proven_component_equivalence": all(
            _require_mapping(value, name).get("verdict")
            in {"byte_identical", "byte_identical_on_frozen_inputs", "equivalent"}
            for name, value in components.items()
        ),
        "full_trace_coverage": coverage["verdict"] == "complete_execution_coverage",
        "complete_result_comparison": result_comparison["verdict"] == "equivalent",
        "native_repeatability": repeatability["verdict"] == "equivalent",
        "exact_full_state_equality_required": False,
    }
    runtime_ready = all(
        value
        for name, value in gates.items()
        if name != "exact_full_state_equality_required"
    )
    return {
        "schema": "GRIL_FULL_AB_REVIEW 1",
        "reviewed_at": evidence.get("reviewed_at"),
        "verdict": {
            "full_runtime_reproduction": (
                "review_ready" if runtime_ready else "not_review_ready"
            ),
            "calibration_physical_quality": physical["native_physical_holdout"],
            "overall": (
                "completed_review_runtime_reproduction_supported"
                if runtime_ready
                else "completed_review_not_accepted_for_runtime_reproduction"
            ),
        },
        "decision": (
            "The ROS-free full execution completed, but it is not review-ready "
            "against the frozen full reference because the rotation result differs "
            f"by {result_comparison['errors']['rotation_deg']:.15g} deg, exceeding "
            f"the {result_comparison['tolerances']['rotation_deg']:.15g} deg "
            "migration tolerance."
            if not result_comparison["checks"]["rotation"]
            else "The review reflects completed evidence only; physical calibration "
            "acceptance remains a separate verdict."
        ),
        "reference_identity": reference_identity,
        "inputs": inputs,
        "effective_algorithm_configuration": configuration,
        "proven_components": components,
        "full_trace_coverage": coverage,
        "full_frontend_non_bitwise_limitation": limitation,
        "result_comparison_current_frozen_reference": result_comparison,
        "native_repeatability": repeatability,
        "physical_validation_separate_from_migration": physical,
        "gates": gates,
        "next_action": (
            "Investigate the final rotation discrepancy without changing pinned "
            "ikd-tree scheduling, migration tolerances, or algorithm mechanics."
            if not result_comparison["checks"]["rotation"]
            else "Obtain an independent native physical holdout before acceptance."
        ),
    }


def write_full_ab_review(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write stable review and concise equivalence artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    review_path = output_dir / "full_ab_review.yaml"
    review_path.write_text(yaml.safe_dump(report, sort_keys=False))
    result = report["result_comparison_current_frozen_reference"]
    repeatability = report["native_repeatability"]
    summary = {
        "schema": "GRIL_FULL_AB_EQUIVALENCE_SUMMARY 1",
        "verdict": report["verdict"]["full_runtime_reproduction"],
        "report": str(review_path.resolve()),
        "result_comparison_current_frozen_reference": result,
        "native_repeatability": repeatability,
        "exact_full_frontend_equality_required": False,
        "non_bitwise_frontend_limitation": report[
            "full_frontend_non_bitwise_limitation"
        ].get("source_behavior"),
    }
    summary_path = output_dir / "equivalence.yaml"
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))
    return review_path, summary_path
