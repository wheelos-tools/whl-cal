#!/usr/bin/env python3

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml

PASS_STATUSES = {"pass", "ok", "healthy"}
FAIL_STATUSES = {"fail", "failed", "error", "blocked"}


def _normalize_status(status: Any) -> str:
    if status is None:
        return "unknown"
    return str(status).strip().lower()


def build_final_acceptance(
    *,
    module: str,
    gates: list[dict[str, Any]],
    pass_recommendation: str,
    review_recommendation: str,
    fail_recommendation: str,
) -> dict[str, Any]:
    normalized_gates = []
    required_problems = []
    advisory_problems = []
    failures = []
    for gate in gates:
        normalized = dict(gate)
        status = _normalize_status(gate.get("status"))
        severity = str(gate.get("severity", "required"))
        normalized["status"] = status
        normalized["severity"] = severity
        normalized_gates.append(normalized)
        if status in FAIL_STATUSES:
            failures.append(normalized)
        elif status not in PASS_STATUSES:
            if severity == "required":
                required_problems.append(normalized)
            else:
                advisory_problems.append(normalized)

    if failures:
        status = "fail"
        recommendation = fail_recommendation
    elif required_problems or advisory_problems:
        status = "warning"
        recommendation = review_recommendation
    else:
        status = "pass"
        recommendation = pass_recommendation

    return {
        "module": module,
        "status": status,
        "release_ready": status == "pass",
        "recommendation": recommendation,
        "gate_counts": {
            "total": len(normalized_gates),
            "required_problem": len(required_problems),
            "advisory_problem": len(advisory_problems),
            "failure": len(failures),
        },
        "gates": normalized_gates,
    }


def write_acceptance_artifacts(
    diagnostics_dir: Path, final_acceptance: dict[str, Any]
) -> dict[str, str]:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    acceptance_path = diagnostics_dir / "acceptance_report.yaml"
    status_csv_path = diagnostics_dir / "status_summary.csv"

    with open(acceptance_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(final_acceptance, file, sort_keys=False)

    with open(status_csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "module",
                "gate",
                "status",
                "severity",
                "evidence",
                "action",
            ],
        )
        writer.writeheader()
        for gate in final_acceptance.get("gates", []):
            writer.writerow(
                {
                    "module": final_acceptance.get("module"),
                    "gate": gate.get("name"),
                    "status": gate.get("status"),
                    "severity": gate.get("severity"),
                    "evidence": gate.get("evidence"),
                    "action": gate.get("action"),
                }
            )

    return {
        "acceptance_report": str(acceptance_path),
        "status_summary_csv": str(status_csv_path),
    }


def _flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten(child_prefix, child, output)
    elif isinstance(value, (list, tuple)):
        output[prefix] = "|".join(str(item) for item in value)
    else:
        output[prefix] = value


def write_table_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows: list[dict[str, Any]] = []
    fieldnames: list[str] = []
    for row in rows:
        flat_row: dict[str, Any] = {}
        _flatten("", row, flat_row)
        flat_rows.append(flat_row)
        for key in flat_row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)
    return str(path)


def write_paradigm_artifacts(
    diagnostics_dir: Path,
    *,
    standardized_data: dict[str, Any],
    data_quality: dict[str, Any],
    visualization_index: dict[str, Any],
) -> dict[str, str]:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "standardized_data": diagnostics_dir / "standardized_data.yaml",
        "data_quality": diagnostics_dir / "data_quality.yaml",
        "visualization_index": diagnostics_dir / "visualization_index.yaml",
    }
    payloads = {
        "standardized_data": standardized_data,
        "data_quality": data_quality,
        "visualization_index": visualization_index,
    }
    written: dict[str, str] = {}
    for key, path in artifacts.items():
        with open(path, "w", encoding="utf-8") as file:
            yaml.safe_dump(payloads[key], file, sort_keys=False)
        written[key] = str(path)
    return written
