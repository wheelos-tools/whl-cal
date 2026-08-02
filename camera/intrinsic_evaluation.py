#!/usr/bin/env python3

"""Evaluation and artifact writing for camera intrinsic calibration."""

from pathlib import Path

import cv2
import numpy as np
from calibration_common.evaluation import (
    build_final_acceptance,
    write_acceptance_artifacts,
    write_paradigm_artifacts,
    write_table_csv,
)

from camera.intrinsic_sampling import edge_corner_coverage
from camera.intrinsic_solver import normalize_distortion_model


def float_list_summary(values):
    if not values:
        return None
    series = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(series)),
        "std": float(np.std(series)),
        "min": float(np.min(series)),
        "p50": float(np.percentile(series, 50)),
        "p95": float(np.percentile(series, 95)),
        "max": float(np.max(series)),
    }


def coverage_metrics(
    sample_records,
    grid_shape=None,
    samples_per_grid=1,
    edge_corner_min_radius_ratio=0.7,
):
    if not sample_records:
        return None
    if grid_shape is None:
        max_cell_x = 0
        max_cell_y = 0
        for record in sample_records:
            occupied_grid_cells = record.get("occupied_grid_cells") or [
                record["grid_cell"]
            ]
            for grid_cell in occupied_grid_cells:
                max_cell_x = max(max_cell_x, int(grid_cell["x"]))
                max_cell_y = max(max_cell_y, int(grid_cell["y"]))
        rows = max_cell_y + 1
        cols = max_cell_x + 1
    else:
        rows = max(int(grid_shape[0]), 1)
        cols = max(int(grid_shape[1]), 1)
    grid_counts = [[0 for _ in range(cols)] for _ in range(rows)]
    center_x = []
    center_y = []
    margins = []
    areas = []
    for record in sample_records:
        occupied_grid_cells = record.get("occupied_grid_cells") or [record["grid_cell"]]
        seen_cells = set()
        for grid_cell in occupied_grid_cells:
            cell_key = (int(grid_cell["x"]), int(grid_cell["y"]))
            if cell_key in seen_cells:
                continue
            seen_cells.add(cell_key)
            if 0 <= cell_key[1] < rows and 0 <= cell_key[0] < cols:
                grid_counts[cell_key[1]][cell_key[0]] += 1
        bbox = record["image_bbox"]
        center_x.append(float(bbox["center_xy_normalized"]["x"]))
        center_y.append(float(bbox["center_xy_normalized"]["y"]))
        margins.append(float(bbox["edge_margin_px"]))
        areas.append(float(bbox["bbox_area_ratio"]))
    occupied = sum(1 for row in grid_counts for count in row if int(count) > 0)
    return {
        "occupied_cell_count": int(occupied),
        "grid_counts": grid_counts,
        "minimum_cell_count": int(
            min((count for row in grid_counts for count in row), default=0)
        ),
        "required_samples_per_cell": int(samples_per_grid),
        "horizontal_span_ratio": float(max(center_x) - min(center_x)),
        "vertical_span_ratio": float(max(center_y) - min(center_y)),
        "edge_margin_px": float_list_summary(margins),
        "bbox_area_ratio": float_list_summary(areas),
        "edge_corner_coverage": edge_corner_coverage(
            sample_records,
            min_radius_ratio=edge_corner_min_radius_ratio,
        ),
        "per_sample": list(sample_records),
    }


def sample_image_size_report(sample_records, capture_runtime_info):
    if not sample_records:
        return None
    sample_sizes = []
    for record in sample_records:
        image_size = record.get("image_size_wh") or {}
        sample_sizes.append(
            (int(image_size.get("width", 0)), int(image_size.get("height", 0)))
        )
    unique_sizes = sorted(
        {size for size in sample_sizes if size[0] > 0 and size[1] > 0}
    )
    actual_capture = (capture_runtime_info or {}).get("actual_capture_resolution") or {}
    actual_size = (
        int(actual_capture.get("width", 0)),
        int(actual_capture.get("height", 0)),
    )
    has_actual_size = actual_size[0] > 0 and actual_size[1] > 0
    report = {
        "unique_sample_sizes": [
            {"width": int(width), "height": int(height)}
            for width, height in unique_sizes
        ],
        "unique_sample_size_count": int(len(unique_sizes)),
        "consistent": bool(len(unique_sizes) == 1),
        "matches_actual_capture_resolution": None,
    }
    if unique_sizes:
        report["primary_sample_size"] = {
            "width": int(unique_sizes[0][0]),
            "height": int(unique_sizes[0][1]),
        }
    if has_actual_size and len(unique_sizes) == 1:
        report["matches_actual_capture_resolution"] = bool(
            unique_sizes[0] == actual_size
        )
        report["actual_capture_resolution"] = {
            "width": int(actual_size[0]),
            "height": int(actual_size[1]),
        }
    return report


def per_view_reprojection_report(residual_views, sample_records):
    rows = []
    for index, residuals in enumerate(residual_views):
        residuals = np.asarray(residuals, dtype=float).reshape(-1, 2)
        point_errors = np.linalg.norm(residuals, axis=1)
        record = sample_records[index] if index < len(sample_records) else {}
        rows.append(
            {
                "sample_id": int(record.get("sample_id", index + 1)),
                "source": record.get("source"),
                "source_path": record.get("source_path"),
                "grid_cell": record.get("grid_cell"),
                "point_count": int(point_errors.size),
                "rms_px": float(np.sqrt(np.mean(np.sum(residuals**2, axis=1)))),
                "p95_px": float(np.percentile(point_errors, 95)),
                "max_px": float(np.max(point_errors)),
            }
        )
    return rows


def distortion_monotonicity_report(
    mtx,
    dist,
    image_size_wh,
    distortion_model="plumb_bob",
):
    model = normalize_distortion_model(distortion_model)
    coeffs = np.asarray(dist, dtype=float).reshape(-1)
    width, height = int(image_size_wh[0]), int(image_size_wh[1])
    fx = float(mtx[0, 0]) if mtx is not None else 1.0
    fy = float(mtx[1, 1]) if mtx is not None else 1.0
    cx = float(mtx[0, 2]) if mtx is not None else width / 2.0
    cy = float(mtx[1, 2]) if mtx is not None else height / 2.0
    corner_radii = []
    for px, py in ((0.0, 0.0), (width, 0.0), (0.0, height), (width, height)):
        xn = (px - cx) / max(fx, 1e-6)
        yn = (py - cy) / max(fy, 1e-6)
        corner_radii.append(float(np.sqrt(xn**2 + yn**2)))
    if model == "fisheye":
        k1 = float(coeffs[0]) if coeffs.size > 0 else 0.0
        k2 = float(coeffs[1]) if coeffs.size > 1 else 0.0
        k3 = float(coeffs[2]) if coeffs.size > 2 else 0.0
        k4 = float(coeffs[3]) if coeffs.size > 3 else 0.0
        max_theta = float(max(np.arctan(corner_radii), default=1.0))
        max_theta = max(max_theta, 1e-6)
        sample_axis = np.linspace(0.0, max_theta, 256)
        derivative = (
            1.0
            + 3.0 * k1 * sample_axis**2
            + 5.0 * k2 * sample_axis**4
            + 7.0 * k3 * sample_axis**6
            + 9.0 * k4 * sample_axis**8
        )
        max_axis = max_theta
        axis_name = "max_theta_rad"
    else:
        k1 = float(coeffs[0]) if coeffs.size > 0 else 0.0
        k2 = float(coeffs[1]) if coeffs.size > 1 else 0.0
        k3 = float(coeffs[4]) if coeffs.size > 4 else 0.0
        max_radius = max(max(corner_radii), 1.0)
        sample_axis = np.linspace(0.0, max_radius, 256)
        derivative = (
            1.0
            + 3.0 * k1 * sample_axis**2
            + 5.0 * k2 * sample_axis**4
            + 7.0 * k3 * sample_axis**6
        )
        max_axis = max_radius
        axis_name = "max_normalized_radius"
    min_derivative = float(np.min(derivative))
    return {
        "distortion_model": model,
        "status": "pass" if min_derivative > 0.0 else "warning",
        axis_name: float(max_axis),
        "min_radial_derivative": min_derivative,
        "sample_count": int(sample_axis.size),
    }


def build_heatmap_artifact(diagnostics_dir, coverage):
    if not coverage:
        return None
    grid_counts = coverage.get("grid_counts", [])
    if not grid_counts:
        return None
    rows = len(grid_counts)
    cols = max((len(row) for row in grid_counts), default=0)
    if rows <= 0 or cols <= 0:
        return None
    cell_size = 120
    image = np.full((rows * cell_size + 170, cols * cell_size + 120, 3), 245, np.uint8)
    cv2.putText(
        image,
        "Intrinsic sample coverage",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (30, 30, 30),
        2,
    )
    max_count = max(max(int(v) for v in row) for row in grid_counts)
    max_count = max(max_count, 1)
    for row_index, row in enumerate(grid_counts):
        for col_index, count in enumerate(row):
            x0 = 70 + col_index * cell_size
            y0 = 80 + row_index * cell_size
            x1 = x0 + cell_size - 10
            y1 = y0 + cell_size - 10
            intensity = int(255 * float(count) / max_count)
            color = (255 - intensity, 210 - intensity // 4, 80 + intensity // 2)
            cv2.rectangle(image, (x0, y0), (x1, y1), color, -1)
            cv2.rectangle(image, (x0, y0), (x1, y1), (50, 50, 50), 2)
            cv2.putText(
                image,
                str(int(count)),
                (x0 + 40, y0 + 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (20, 20, 20),
                2,
            )
    edge = coverage.get("edge_corner_coverage") or {}
    edge_text = (
        "Outer corner quadrants: "
        f"{edge.get('covered_quadrant_count', 0)}/{edge.get('required_quadrant_count', 4)}"
        f" | max radius: {float(edge.get('max_observed_radius_ratio', 0.0)):.2f}"
    )
    cv2.putText(
        image,
        edge_text,
        (30, rows * cell_size + 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (30, 30, 30),
        2,
    )
    artifact = diagnostics_dir / "image_coverage_heatmap.png"
    cv2.imwrite(str(artifact), image)
    return str(artifact)


def build_intrinsic_acceptance(
    min_total_samples,
    sample_records,
    capture_runtime_info,
    global_reprojection_rms,
    solver_reported_rms,
    per_view_report,
    coverage,
    monotonicity_report,
    distortion_model="plumb_bob",
):
    model = normalize_distortion_model(distortion_model)
    per_view_rms = [float(row["rms_px"]) for row in per_view_report]
    per_view_point_counts = [int(row["point_count"]) for row in per_view_report]
    recomputed_global_rms = None
    if per_view_rms and sum(per_view_point_counts) > 0:
        recomputed_global_rms = float(
            np.sqrt(
                np.average(
                    np.square(np.asarray(per_view_rms, dtype=float)),
                    weights=np.asarray(per_view_point_counts, dtype=float),
                )
            )
        )
    solver_rms_delta = (
        None
        if solver_reported_rms is None or recomputed_global_rms is None
        else abs(float(solver_reported_rms) - float(recomputed_global_rms))
    )
    solver_rms_tolerance = (
        None
        if recomputed_global_rms is None
        else max(0.05, 0.1 * float(recomputed_global_rms))
    )
    image_size_report = sample_image_size_report(sample_records, capture_runtime_info)
    coverage_ready = bool(
        coverage is not None
        and len(sample_records) >= int(min_total_samples)
        and int(coverage["minimum_cell_count"])
        >= int(coverage["required_samples_per_cell"])
        and int(coverage["edge_corner_coverage"]["covered_quadrant_count"])
        >= int(coverage["edge_corner_coverage"]["required_quadrant_count"])
    )
    reprojection_fit_ready = bool(
        per_view_rms
        and float(global_reprojection_rms) <= 1.0
        and float(np.percentile(np.asarray(per_view_rms, dtype=float), 95)) <= 1.5
    )
    capture_mode_status = "pass"
    if image_size_report is None or not bool(image_size_report.get("consistent")):
        capture_mode_status = "fail"
    elif image_size_report.get("matches_actual_capture_resolution") is False:
        capture_mode_status = "fail"
    elif bool((capture_runtime_info or {}).get("force_capture_resolution")):
        capture_mode_status = "warning"
    gates = [
        {
            "name": "sample_sufficiency",
            "status": "pass" if coverage_ready else "fail",
            "severity": "required",
            "evidence": (
                f"samples={len(sample_records)}/{min_total_samples}, "
                "minimum_cell_count="
                f"{None if coverage is None else coverage['minimum_cell_count']}/"
                f"{None if coverage is None else coverage['required_samples_per_cell']}, "
                "outer_quadrants="
                f"{None if coverage is None else coverage['edge_corner_coverage']['covered_quadrant_count']}/4"
            ),
            "action": "Recollect the documented 36-view pattern: three views per grid cell, all outer quadrants, then novel poses.",
        },
        {
            "name": "reprojection_fit",
            "status": "pass" if reprojection_fit_ready else "warning",
            "severity": "required",
            "evidence": (
                "global_reprojection_rms_px="
                f"{float(global_reprojection_rms)}"
                ", per_view_rms_p95_px="
                f"{None if not per_view_rms else float(np.percentile(np.asarray(per_view_rms, dtype=float), 95))}"
            ),
            "action": "Inspect the per-view residual table; recollect blurred, unstable, or poorly posed views instead of tuning the solver.",
        },
        {
            "name": "reprojection_consistency",
            "status": (
                "pass"
                if solver_rms_delta is not None
                and solver_rms_tolerance is not None
                and solver_rms_delta <= solver_rms_tolerance
                else "fail"
            ),
            "severity": "required",
            "evidence": (
                f"solver_reported_rms_px={solver_reported_rms}, "
                f"recomputed_global_rms_px={recomputed_global_rms}, "
                f"tolerance_px={solver_rms_tolerance}"
            ),
            "action": "Reject the result when the solver-reported RMS and the residual-derived RMS disagree; inspect the lens model and calibration inputs.",
        },
        {
            "name": "capture_mode",
            "status": capture_mode_status,
            "severity": "required",
            "evidence": (
                "unique_sample_sizes="
                f"{None if image_size_report is None else image_size_report.get('unique_sample_sizes')}, "
                "matches_capture="
                f"{None if image_size_report is None else image_size_report.get('matches_actual_capture_resolution')}, "
                "forced="
                f"{bool((capture_runtime_info or {}).get('force_capture_resolution'))}"
            ),
            "action": "Use one native capture mode for collection and deployment; do not mix image sizes.",
        },
        {
            "name": "projection_validity",
            "status": ("fail" if monotonicity_report["status"] != "pass" else "pass"),
            "severity": "required",
            "evidence": (
                f"distortion_model={model}, "
                "min_radial_derivative="
                f"{float(monotonicity_report['min_radial_derivative'])}"
            ),
            "action": (
                "Treat non-monotonic radial distortion as calibration failure; "
                "verify lens-model choice, capture mode, and recollect broader views."
            ),
        },
    ]
    return build_final_acceptance(
        module="camera_intrinsic",
        gates=gates,
        pass_recommendation="release_intrinsics",
        review_recommendation="review_intrinsic_diagnostics",
        fail_recommendation="reject_and_recollect_intrinsic_samples",
    )


def write_review_artifacts(
    output_yaml_path,
    *,
    min_total_samples,
    sample_records,
    capture_runtime_info,
    calibration_target,
    comparison_view_path,
    global_reprojection_rms,
    solver_reported_rms,
    per_view_report,
    coverage,
    monotonicity_report,
    distortion_model="plumb_bob",
):
    output_path = Path(output_yaml_path)
    diagnostics_dir = output_path.with_name(f"{output_path.stem}_diagnostics")
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    per_view_csv = write_table_csv(
        diagnostics_dir / "per_view_reprojection.csv", per_view_report
    )
    sample_records_csv = write_table_csv(
        diagnostics_dir / "sample_records.csv", sample_records
    )
    heatmap_path = build_heatmap_artifact(diagnostics_dir, coverage)
    final_acceptance = build_intrinsic_acceptance(
        min_total_samples,
        sample_records,
        capture_runtime_info,
        global_reprojection_rms,
        solver_reported_rms,
        per_view_report,
        coverage,
        monotonicity_report,
        distortion_model=distortion_model,
    )
    acceptance_artifacts = write_acceptance_artifacts(diagnostics_dir, final_acceptance)
    standardized_data = {
        "schema_version": 1,
        "module": "camera_intrinsic",
        "representation": f"{str((calibration_target or {}).get('type', 'target'))}_image_samples",
        "sample_counts": {
            "accepted_samples": int(len(sample_records)),
            "required_samples": int(min_total_samples),
        },
        "capture_runtime": capture_runtime_info,
        "sample_records": list(sample_records),
        "calibration_target": calibration_target,
    }
    data_quality = {
        "schema_version": 1,
        "module": "camera_intrinsic",
        "status": final_acceptance["status"],
        "release_ready": final_acceptance["release_ready"],
        "quality_gates": final_acceptance["gates"],
        "global_reprojection_rms_px": float(global_reprojection_rms),
        "solver_reported_rms_px": float(solver_reported_rms),
        "per_view_reprojection_summary": float_list_summary(
            [float(row["rms_px"]) for row in per_view_report]
        ),
        "sample_image_sizes": sample_image_size_report(
            sample_records,
            capture_runtime_info,
        ),
        "image_coverage": coverage,
        "radial_monotonicity": monotonicity_report,
        "calibration_target": calibration_target,
    }
    visualization_index = {
        "schema_version": 1,
        "module": "camera_intrinsic",
        "layers": {
            "conclusion": [
                acceptance_artifacts["acceptance_report"],
                acceptance_artifacts["status_summary_csv"],
            ],
            "detail_metrics": [
                str(output_path),
                per_view_csv,
                sample_records_csv,
            ],
            "visual_review": [
                item
                for item in (
                    comparison_view_path,
                    heatmap_path,
                )
                if item is not None
            ],
        },
        "manual_review": [
            "Read data_quality.yaml and acceptance_report.yaml before inspecting the preview.",
            "Use per_view_reprojection.csv only to identify the images behind a failed fit gate.",
            "Confirm image_coverage_heatmap.png has three views per cell and four outer quadrants.",
            "Reject solver/residual disagreement or non-monotonic projection.",
        ],
    }
    paradigm_artifacts = write_paradigm_artifacts(
        diagnostics_dir,
        standardized_data=standardized_data,
        data_quality=data_quality,
        visualization_index=visualization_index,
    )
    return {
        "diagnostics_dir": str(diagnostics_dir),
        "acceptance": acceptance_artifacts,
        "release_ready": bool(final_acceptance.get("release_ready", False)),
        "final_acceptance": final_acceptance,
        "paradigm": paradigm_artifacts,
        "per_view_reprojection_csv": per_view_csv,
        "sample_records_csv": sample_records_csv,
        "image_coverage_heatmap": heatmap_path,
    }
