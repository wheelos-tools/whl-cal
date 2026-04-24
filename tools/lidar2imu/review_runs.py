#!/usr/bin/env python3

# Copyright 2026 The WheelOS Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Created Date: 2026-04-24

"""Review lidar2imu run outputs and generate a compact HTML/CSV report."""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def _resolve_metrics_path(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if path.is_file():
        return path
    direct = path / "metrics.yaml"
    if direct.exists():
        return direct
    nested = path / "calibration" / "metrics.yaml"
    if nested.exists():
        return nested
    raise FileNotFoundError(
        f"Could not find metrics.yaml under {path}. Expected metrics.yaml or calibration/metrics.yaml."
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object at {path}")
    return payload


def _dig(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_float(value: Any, digits: int = 3) -> str:
    numeric = _float_or_none(value)
    if numeric is None:
        return "-"
    return f"{numeric:.{digits}f}"


def _format_ratio(numerator: Any, denominator: Any) -> str:
    left = _int_or_none(numerator)
    right = _int_or_none(denominator)
    if left is None or right in (None, 0):
        return "-"
    return f"{left}/{right}"


def _join_list(values: Any) -> str:
    if not isinstance(values, list):
        return "-"
    return ",".join(str(item) for item in values)


def _status_badge(value: str | None) -> str:
    normalized = (value or "unknown").strip() or "unknown"
    css = {
        "pass": "pass",
        "warning": "warning",
        "unknown": "unknown",
        "recoverable": "warning",
        "full_6dof_candidate": "pass",
        "z_roll_pitch_priority": "warning",
        "holdout_review": "warning",
        "prior_sensitivity_review": "warning",
        "reference_conflict_review": "danger",
        "reextract_review": "danger",
        "basin_sensitivity_review": "danger",
        "recollect_data": "danger",
    }.get(normalized, "unknown")
    return f'<span class="badge {css}">{html.escape(normalized)}</span>'


def _recommendation_rank(value: str | None) -> int:
    return {
        "full_6dof_candidate": 0,
        "z_roll_pitch_priority": 1,
        "holdout_review": 2,
        "prior_sensitivity_review": 3,
        "reextract_review": 4,
        "reference_conflict_review": 5,
        "basin_sensitivity_review": 6,
        "recollect_data": 7,
        None: 8,
    }.get(value, 8)


def _row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        _recommendation_rank(row.get("recommendation")),
        row.get("trusted_reference_consistency") != "pass",
        row.get("extraction_consistency") != "pass",
        row.get("planar_basin_stability") != "pass",
        row.get("full_prior_robustness") != "pass",
        row.get("holdout_generalization") != "pass",
        _float_or_none(row.get("delta_to_reference_rotation_deg")) or 1e9,
        _float_or_none(row.get("motion_translation_residual_p95_m")) or 1e9,
        row.get("run_name") or "",
    )


def _load_run_row(metrics_path: Path) -> dict[str, Any]:
    payload = _load_yaml(metrics_path)
    summary = payload.get("summary", {}) or {}
    coarse = payload.get("coarse_metrics", {}) or {}
    statuses = coarse.get("statuses", {}) or {}
    assessment = payload.get("vehicle_motion_assessment", {}) or {}
    yaw_diagnostic = assessment.get("yaw_diagnostic", {}) or {}
    evidence = yaw_diagnostic.get("evidence", {}) or {}
    basin = assessment.get("planar_basin_stability_details", {}) or {}
    metrics_dir = metrics_path.parent
    run_dir = metrics_dir.parent if metrics_dir.name == "calibration" else metrics_dir

    delta_to_initial = summary.get("delta_to_initial", {}) or {}
    delta_to_extraction = summary.get("delta_to_extraction", {}) or {}
    delta_to_reference = summary.get("delta_to_reference", {}) or {}

    row = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
        "run_profile": summary.get("run_profile"),
        "recommendation": assessment.get("recommendation"),
        "applied_solver_planar_motion_policy": assessment.get(
            "applied_solver_planar_motion_policy"
        ),
        "ground_support": assessment.get("ground_support"),
        "motion_registration_quality": assessment.get("motion_registration_quality"),
        "yaw_observability": assessment.get("yaw_observability"),
        "joint_observability": assessment.get("joint_observability"),
        "initial_prior_assessment": assessment.get("initial_prior_assessment"),
        "extraction_consistency": assessment.get("extraction_consistency"),
        "trusted_reference_consistency": assessment.get(
            "trusted_reference_consistency"
        ),
        "planar_basin_stability": assessment.get("planar_basin_stability"),
        "full_prior_robustness": assessment.get("full_prior_robustness"),
        "holdout_generalization": assessment.get("holdout_generalization"),
        "planar_basin_primary_cause": basin.get("primary_cause"),
        "full_prior_primary_cause": _dig(
            assessment, "full_prior_robustness_details", "primary_cause"
        ),
        "holdout_primary_cause": _dig(
            assessment, "holdout_validation_details", "primary_cause"
        ),
        "distinct_solution_count": basin.get("distinct_solution_count"),
        "basin_trial_count": basin.get("trial_count"),
        "reference_consistent_trial_count": basin.get(
            "reference_consistent_trial_count"
        ),
        "holdout_motion_sample_count": _dig(
            summary, "dataset_partition", "holdout_motion_samples"
        ),
        "holdout_rotation_ratio": _dig(
            assessment, "holdout_validation_details", "ratios", "rotation_residual_p95"
        ),
        "holdout_translation_ratio": _dig(
            assessment,
            "holdout_validation_details",
            "ratios",
            "translation_residual_p95",
        ),
        "holdout_fitness_ratio": _dig(
            assessment,
            "holdout_validation_details",
            "ratios",
            "registration_fitness_p05",
        ),
        "holdout_rmse_ratio": _dig(
            assessment,
            "holdout_validation_details",
            "ratios",
            "registration_inlier_rmse_p95",
        ),
        "yaw_primary_cause": yaw_diagnostic.get("primary_cause"),
        "yaw_max_cost_ratio": evidence.get("max_cost_ratio"),
        "yaw_within_5pct_span_deg": evidence.get("within_5pct_span_deg"),
        "yaw_scalar_sensitivity": evidence.get("scalar_sensitivity"),
        "motion_registration_mode": evidence.get("motion_registration_mode"),
        "motion_selection_strategy": evidence.get("motion_selection_strategy"),
        "selected_frame_strides": _join_list(evidence.get("selected_frame_strides")),
        "translation_heading_span_deg": evidence.get("translation_heading_span_deg"),
        "imu_rotation_axis_abs_mean_z": evidence.get("imu_rotation_axis_abs_mean_z"),
        "ground_normal_angle_p95_deg": coarse.get("ground_normal_angle_p95_deg"),
        "ground_height_residual_p95_m": coarse.get("ground_height_residual_p95_m"),
        "motion_rotation_residual_p95_deg": coarse.get(
            "motion_rotation_residual_p95_deg"
        ),
        "motion_translation_residual_p95_m": coarse.get(
            "motion_translation_residual_p95_m"
        ),
        "motion_registration_fitness_p05": coarse.get(
            "motion_registration_fitness_p05"
        ),
        "motion_registration_inlier_rmse_p95": coarse.get(
            "motion_registration_inlier_rmse_p95"
        ),
        "turn_balance_ratio": coarse.get("turn_balance_ratio"),
        "left_turn_count": coarse.get("left_turn_count"),
        "right_turn_count": coarse.get("right_turn_count"),
        "joint_condition_number": coarse.get("joint_condition_number"),
        "status_ground_orientation": statuses.get("ground_orientation"),
        "status_ground_height": statuses.get("ground_height"),
        "status_motion_rotation": statuses.get("motion_rotation"),
        "status_motion_translation": statuses.get("motion_translation"),
        "status_motion_registration": statuses.get("motion_registration"),
        "status_turn_balance": statuses.get("turn_balance"),
        "status_observability": statuses.get("observability"),
        "status_initial_prior_nominal_range": statuses.get(
            "initial_prior_nominal_range"
        ),
        "status_extraction_geometry": statuses.get("extraction_geometry"),
        "status_trusted_reference": statuses.get("trusted_reference"),
        "status_planar_basin_stability": statuses.get("planar_basin_stability"),
        "status_full_prior_robustness": statuses.get("full_prior_robustness"),
        "status_holdout_generalization": statuses.get("holdout_generalization"),
        "delta_to_initial_translation_m": delta_to_initial.get("translation_norm_m"),
        "delta_to_initial_z_m": _dig(delta_to_initial, "translation_xyz_m", "z"),
        "delta_to_initial_rotation_deg": delta_to_initial.get("rotation_deg"),
        "delta_to_extraction_translation_m": delta_to_extraction.get(
            "translation_norm_m"
        ),
        "delta_to_extraction_z_m": _dig(delta_to_extraction, "translation_xyz_m", "z"),
        "delta_to_extraction_rotation_deg": delta_to_extraction.get("rotation_deg"),
        "delta_to_reference_translation_m": delta_to_reference.get(
            "translation_norm_m"
        ),
        "delta_to_reference_z_m": _dig(delta_to_reference, "translation_xyz_m", "z"),
        "delta_to_reference_rotation_deg": delta_to_reference.get("rotation_deg"),
        "initial_transform_source": _dig(
            payload,
            "fine_metrics",
            "algorithm_stages",
            "solver_policy",
            "initial_transform_source",
        ),
        "extraction_transform_source": assessment.get("extraction_transform_source"),
        "reference_transform_source": assessment.get("reference_transform_source"),
        "final_x_m": _dig(summary, "final_translation_m", "x"),
        "final_y_m": _dig(summary, "final_translation_m", "y"),
        "final_z_m": _dig(summary, "final_translation_m", "z"),
        "final_yaw_deg": _dig(summary, "final_euler_deg", "yaw"),
        "final_roll_deg": _dig(summary, "final_euler_deg", "roll"),
        "final_pitch_deg": _dig(summary, "final_euler_deg", "pitch"),
        "reference_consistency_recommendations": assessment.get(
            "reference_consistency_recommendations", []
        ),
        "extraction_consistency_recommendations": assessment.get(
            "extraction_consistency_recommendations", []
        ),
        "planar_basin_recommendations": basin.get("recommendations", []),
        "initial_prior_primary_cause": _dig(
            assessment, "initial_prior_assessment_details", "primary_cause"
        ),
        "initial_prior_failure_reasons": _join_list(
            _dig(assessment, "initial_prior_assessment_details", "failure_reasons")
        ),
    }
    return row


def _write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _svg_bar_chart(
    rows: list[dict[str, Any]],
    *,
    title: str,
    value_key: str,
    value_label: str,
    width: int = 820,
    bar_color: str = "#4f7cff",
) -> str:
    values = [
        (_float_or_none(row.get(value_key)), str(row.get("run_name") or f"run-{index}"))
        for index, row in enumerate(rows)
    ]
    values = [(value, label) for value, label in values if value is not None]
    if not values:
        return f"<p>No data for {html.escape(title)}.</p>"
    max_value = max(value for value, _ in values)
    max_value = max(max_value, 1e-9)
    bar_area_height = 220
    chart_height = 290
    margin_left = 56
    plot_width = width - margin_left - 20
    bar_step = plot_width / max(len(values), 1)
    bar_width = max(bar_step * 0.6, 16.0)
    parts = [
        f'<svg viewBox="0 0 {width} {chart_height}" class="chart" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{width / 2:.1f}" y="18" text-anchor="middle" class="chart-title">{html.escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{bar_area_height}" x2="{width - 12}" y2="{bar_area_height}" class="axis" />',
        f'<line x1="{margin_left}" y1="28" x2="{margin_left}" y2="{bar_area_height}" class="axis" />',
        f'<text x="14" y="24" class="axis-label">{html.escape(value_label)}</text>',
    ]
    for index, (value, label) in enumerate(values):
        x = margin_left + (index + 0.5) * bar_step - (bar_width / 2.0)
        height = 0.0 if max_value <= 0 else (value / max_value) * (bar_area_height - 40)
        y = bar_area_height - height
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{height:.2f}" fill="{bar_color}" rx="3" />'
        )
        parts.append(
            f'<text x="{x + (bar_width / 2.0):.2f}" y="{y - 6:.2f}" text-anchor="middle" class="value-label">{html.escape(_format_float(value, 2))}</text>'
        )
        parts.append(
            f'<text x="{x + (bar_width / 2.0):.2f}" y="{bar_area_height + 16:.2f}" text-anchor="end" transform="rotate(-35 {x + (bar_width / 2.0):.2f},{bar_area_height + 16:.2f})" class="tick-label">{html.escape(label)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _svg_scatter_plot(
    rows: list[dict[str, Any]],
    *,
    title: str,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    width: int = 820,
) -> str:
    points = []
    for index, row in enumerate(rows):
        x = _float_or_none(row.get(x_key))
        y = _float_or_none(row.get(y_key))
        if x is None or y is None:
            continue
        points.append((x, y, str(row.get("run_name") or f"run-{index}"), row))
    if not points:
        return f"<p>No data for {html.escape(title)}.</p>"
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    if math.isclose(min_x, max_x):
        max_x = min_x + 1.0
    if math.isclose(min_y, max_y):
        max_y = min_y + 1.0
    height = 320
    margin_left = 64
    margin_bottom = 46
    plot_width = width - margin_left - 18
    plot_height = height - 44 - margin_bottom
    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{width / 2:.1f}" y="18" text-anchor="middle" class="chart-title">{html.escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{plot_height + 26}" x2="{width - 12}" y2="{plot_height + 26}" class="axis" />',
        f'<line x1="{margin_left}" y1="34" x2="{margin_left}" y2="{plot_height + 26}" class="axis" />',
        f'<text x="{width / 2:.1f}" y="{height - 10}" text-anchor="middle" class="axis-label">{html.escape(x_label)}</text>',
        f'<text x="18" y="{height / 2:.1f}" text-anchor="middle" transform="rotate(-90 18,{height / 2:.1f})" class="axis-label">{html.escape(y_label)}</text>',
    ]
    color_map = {
        "full_6dof_candidate": "#27ae60",
        "z_roll_pitch_priority": "#f39c12",
        "holdout_review": "#d97706",
        "prior_sensitivity_review": "#b45309",
        "reextract_review": "#e74c3c",
        "reference_conflict_review": "#c0392b",
        "basin_sensitivity_review": "#8e44ad",
    }
    for x, y, label, row in points:
        px = margin_left + ((x - min_x) / (max_x - min_x)) * plot_width
        py = 34 + plot_height - ((y - min_y) / (max_y - min_y)) * plot_height
        color = color_map.get(row.get("recommendation"), "#4f7cff")
        parts.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="5.5" fill="{color}" opacity="0.92"><title>{html.escape(label)}: {html.escape(x_label)}={_format_float(x, 3)}, {html.escape(y_label)}={_format_float(y, 3)}</title></circle>'
        )
        parts.append(
            f'<text x="{px + 7:.2f}" y="{py - 7:.2f}" class="tick-label">{html.escape(label)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _render_summary_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "<tr>"
        "<th>run</th><th>recommendation</th><th>policy</th><th>extract</th>"
        "<th>reference</th><th>basin</th><th>6dof prior</th><th>holdout</th><th>prior</th><th>yaw ratio</th><th>plateau</th>"
        "<th>trans p95</th><th>fit p05</th><th>delta ref</th><th>delta ext</th>"
        "</tr>"
    )
    body_rows = []
    for row in rows:
        delta_ref = (
            f'{_format_float(row.get("delta_to_reference_translation_m"), 3)} m / '
            f'{_format_float(row.get("delta_to_reference_rotation_deg"), 3)} deg '
            f'(z={_format_float(row.get("delta_to_reference_z_m"), 3)} m)'
        )
        delta_ext = (
            f'{_format_float(row.get("delta_to_extraction_translation_m"), 3)} m / '
            f'{_format_float(row.get("delta_to_extraction_rotation_deg"), 3)} deg '
            f'(z={_format_float(row.get("delta_to_extraction_z_m"), 3)} m)'
        )
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('run_name') or '-'))}</td>"
            f"<td>{_status_badge(row.get('recommendation'))}</td>"
            f"<td>{html.escape(str(row.get('applied_solver_planar_motion_policy') or '-'))}</td>"
            f"<td>{_status_badge(row.get('extraction_consistency'))}</td>"
            f"<td>{_status_badge(row.get('trusted_reference_consistency'))}</td>"
            f"<td>{_status_badge(row.get('planar_basin_stability'))}</td>"
            f"<td>{_status_badge(row.get('full_prior_robustness'))}</td>"
            f"<td>{_status_badge(row.get('holdout_generalization'))}</td>"
            f"<td>{_status_badge(row.get('initial_prior_assessment'))}</td>"
            f"<td>{html.escape(_format_float(row.get('yaw_max_cost_ratio'), 2))}</td>"
            f"<td>{html.escape(_format_float(row.get('yaw_within_5pct_span_deg'), 1))}</td>"
            f"<td>{html.escape(_format_float(row.get('motion_translation_residual_p95_m'), 3))}</td>"
            f"<td>{html.escape(_format_float(row.get('motion_registration_fitness_p05'), 3))}</td>"
            f"<td>{html.escape(delta_ref)}</td>"
            f"<td>{html.escape(delta_ext)}</td>"
            "</tr>"
        )
    return "<table>" + header + "".join(body_rows) + "</table>"


def _render_detail_cards(rows: list[dict[str, Any]]) -> str:
    cards = []
    for row in rows:
        cards.append(
            "<div class='card'>"
            f"<h3>{html.escape(str(row.get('run_name') or '-'))}</h3>"
            f"<p><strong>recommendation:</strong> {html.escape(str(row.get('recommendation') or '-'))}</p>"
            f"<p><strong>profile:</strong> {html.escape(str(row.get('run_profile') or '-'))}</p>"
            f"<p><strong>sources:</strong> initial={html.escape(str(row.get('initial_transform_source') or '-'))}<br>"
            f"extraction={html.escape(str(row.get('extraction_transform_source') or '-'))}<br>"
            f"reference={html.escape(str(row.get('reference_transform_source') or '-'))}</p>"
            f"<p><strong>turns:</strong> L={html.escape(str(row.get('left_turn_count') or '-'))}, "
            f"R={html.escape(str(row.get('right_turn_count') or '-'))}, "
            f"balance={html.escape(_format_float(row.get('turn_balance_ratio'), 3))}</p>"
            f"<p><strong>selected strides:</strong> {html.escape(str(row.get('selected_frame_strides') or '-'))}</p>"
            f"<p><strong>basin trials:</strong> {html.escape(_format_ratio(row.get('reference_consistent_trial_count'), row.get('basin_trial_count')))} reference-consistent, "
            f"{html.escape(str(row.get('distinct_solution_count') or '-'))} distinct solutions</p>"
            f"<p><strong>6DoF prior robustness:</strong> "
            f"{html.escape(str(row.get('full_prior_robustness') or '-'))}, "
            f"cause={html.escape(str(row.get('full_prior_primary_cause') or '-'))}</p>"
            f"<p><strong>holdout:</strong> {html.escape(str(row.get('holdout_generalization') or '-'))}, "
            f"rot ratio={html.escape(_format_float(row.get('holdout_rotation_ratio'), 2))}, "
            f"trans ratio={html.escape(_format_float(row.get('holdout_translation_ratio'), 2))}</p>"
            f"<p><strong>initial prior:</strong> "
            f"{html.escape(str(row.get('initial_prior_assessment') or '-'))}, "
            f"delta={html.escape(_format_float(row.get('delta_to_initial_translation_m'), 3))} m / "
            f"{html.escape(_format_float(row.get('delta_to_initial_rotation_deg'), 3))} deg, "
            f"z={html.escape(_format_float(row.get('delta_to_initial_z_m'), 3))} m</p>"
            f"<p><strong>reference drift:</strong> "
            f"{html.escape(_format_float(row.get('delta_to_reference_translation_m'), 3))} m / "
            f"{html.escape(_format_float(row.get('delta_to_reference_rotation_deg'), 3))} deg, "
            f"z={html.escape(_format_float(row.get('delta_to_reference_z_m'), 3))} m</p>"
            f"<p><strong>extraction drift:</strong> "
            f"{html.escape(_format_float(row.get('delta_to_extraction_translation_m'), 3))} m / "
            f"{html.escape(_format_float(row.get('delta_to_extraction_rotation_deg'), 3))} deg, "
            f"z={html.escape(_format_float(row.get('delta_to_extraction_z_m'), 3))} m</p>"
            "</div>"
        )
    return "<div class='card-grid'>" + "".join(cards) + "</div>"


def _render_html_report(rows: list[dict[str, Any]], report_path: Path) -> None:
    recommendation_counts = Counter(
        str(row.get("recommendation") or "unknown") for row in rows
    )
    summary_items = "".join(
        f"<li><strong>{html.escape(name)}:</strong> {count}</li>"
        for name, count in sorted(recommendation_counts.items())
    )
    html_payload = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>lidar2imu review report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    h1, h2, h3 {{ margin-bottom: 8px; }}
    .muted {{ color: #6b7280; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; font-size: 13px; }}
    th {{ background: #f3f4f6; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; }}
    .pass {{ background: #dcfce7; color: #166534; }}
    .warning {{ background: #fef3c7; color: #92400e; }}
    .danger {{ background: #fee2e2; color: #991b1b; }}
    .unknown {{ background: #e5e7eb; color: #374151; }}
    .chart {{ width: 100%; max-width: 900px; background: #fff; border: 1px solid #e5e7eb; margin: 12px 0 28px; }}
    .chart-title {{ font-size: 14px; font-weight: bold; fill: #111827; }}
    .axis {{ stroke: #9ca3af; stroke-width: 1; }}
    .axis-label, .tick-label, .value-label {{ fill: #374151; font-size: 11px; }}
    .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px 14px; background: #fafafa; }}
    ul {{ margin-top: 4px; }}
  </style>
</head>
<body>
  <h1>lidar2imu review report</h1>
  <p class="muted">Generated from {len(rows)} run(s).</p>
  <h2>Recommendation counts</h2>
  <ul>{summary_items}</ul>
  <h2>Acceptance summary</h2>
  {_render_summary_table(rows)}
  <h2>Reference drift vs motion translation residual</h2>
  {_svg_scatter_plot(rows, title="reference drift vs motion translation residual", x_key="delta_to_reference_rotation_deg", y_key="motion_translation_residual_p95_m", x_label="delta_to_reference rotation (deg)", y_label="motion translation residual p95 (m)")}
  <h2>Yaw support</h2>
  {_svg_bar_chart(rows, title="yaw max cost ratio", value_key="yaw_max_cost_ratio", value_label="ratio", bar_color="#2563eb")}
  {_svg_bar_chart(rows, title="yaw 5% plateau span", value_key="yaw_within_5pct_span_deg", value_label="deg", bar_color="#0f766e")}
  <h2>Basin stability</h2>
  {_svg_bar_chart(rows, title="distinct solution count", value_key="distinct_solution_count", value_label="count", bar_color="#7c3aed")}
  {_svg_bar_chart(rows, title="reference-consistent trial count", value_key="reference_consistent_trial_count", value_label="count", bar_color="#16a34a")}
  <h2>Run details</h2>
  {_render_detail_cards(rows)}
</body>
</html>
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html_payload, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review lidar2imu run outputs and generate a compact CSV/HTML report."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Run directories or metrics.yaml files. Directories may contain metrics.yaml or calibration/metrics.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for review_summary.csv and review_report.html.",
    )
    args = parser.parse_args()

    rows = []
    for raw_path in args.paths:
        metrics_path = _resolve_metrics_path(raw_path)
        rows.append(_load_run_row(metrics_path))
    rows.sort(key=_row_sort_key)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "review_summary.csv"
    html_path = output_dir / "review_report.html"
    _write_csv(rows, csv_path)
    _render_html_report(rows, html_path)
    print(f"Saved review CSV to {csv_path}")
    print(f"Saved review HTML to {html_path}")


if __name__ == "__main__":
    main()
