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


def coverage_metrics(sample_records, grid_shape=None):
    if not sample_records:
        return None
    if grid_shape is None:
        max_cell_x = 0
        max_cell_y = 0
        for record in sample_records:
            occupied_grid_cells = record.get("occupied_grid_cells") or [record["grid_cell"]]
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
        "horizontal_span_ratio": float(max(center_x) - min(center_x)),
        "vertical_span_ratio": float(max(center_y) - min(center_y)),
        "edge_margin_px": float_list_summary(margins),
        "bbox_area_ratio": float_list_summary(areas),
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
    unique_sizes = sorted({size for size in sample_sizes if size[0] > 0 and size[1] > 0})
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
        report["matches_actual_capture_resolution"] = bool(unique_sizes[0] == actual_size)
        report["actual_capture_resolution"] = {
            "width": int(actual_size[0]),
            "height": int(actual_size[1]),
        }
    return report


def per_view_reprojection_report(objpoints, imgpoints, mtx, dist, sample_records, rvecs, tvecs):
    rows = []
    for index in range(len(objpoints)):
        imgpts2, _ = cv2.projectPoints(
            objpoints[index],
            rvecs[index],
            tvecs[index],
            mtx,
            dist,
        )
        observed = np.asarray(imgpoints[index], dtype=float).reshape(-1, 2)
        predicted = np.asarray(imgpts2, dtype=float).reshape(-1, 2)
        residuals = predicted - observed
        point_errors = np.linalg.norm(residuals, axis=1)
        record = sample_records[index] if index < len(sample_records) else {}
        rows.append(
            {
                "sample_id": int(record.get("sample_id", index + 1)),
                "source": record.get("source"),
                "source_path": record.get("source_path"),
                "grid_cell": record.get("grid_cell"),
                "rms_px": float(np.sqrt(np.mean(np.sum(residuals**2, axis=1)))),
                "p95_px": float(np.percentile(point_errors, 95)),
                "max_px": float(np.max(point_errors)),
            }
        )
    return rows


def distortion_monotonicity_report(mtx, dist, image_size_wh):
    coeffs = np.asarray(dist, dtype=float).reshape(-1)
    k1 = float(coeffs[0]) if coeffs.size > 0 else 0.0
    k2 = float(coeffs[1]) if coeffs.size > 1 else 0.0
    k3 = float(coeffs[4]) if coeffs.size > 4 else 0.0
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
    max_radius = max(max(corner_radii), 1.0)
    sample_r = np.linspace(0.0, max_radius, 256)
    derivative = 1.0 + 3.0 * k1 * sample_r**2 + 5.0 * k2 * sample_r**4 + 7.0 * k3 * sample_r**6
    min_derivative = float(np.min(derivative))
    return {
        "status": "pass" if min_derivative > 0.0 else "warning",
        "max_normalized_radius": float(max_radius),
        "min_radial_derivative": min_derivative,
        "sample_count": int(sample_r.size),
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
    image = np.full((rows * cell_size + 120, cols * cell_size + 120, 3), 245, np.uint8)
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
    artifact = diagnostics_dir / "image_coverage_heatmap.png"
    cv2.imwrite(str(artifact), image)
    return str(artifact)


def build_intrinsic_acceptance(
    min_total_samples,
    sample_records,
    capture_runtime_info,
    calibration_target,
    imgpoints,
    avg_error,
    per_view_report,
    coverage,
    monotonicity_report,
):
    target_type = str((calibration_target or {}).get("type", "chessboard"))
    per_view_rms = [float(row["rms_px"]) for row in per_view_report]
    occupied_cell_target = max(4, min(6, int(min_total_samples)))
    target_points_per_sample = [int(np.asarray(points).shape[0]) for points in imgpoints]
    image_size_report = sample_image_size_report(sample_records, capture_runtime_info)
    if target_type == "aprilgrid":
        min_points_per_sample = int(
            (calibration_target or {}).get("min_points_per_frame", 16)
        )
    elif target_type == "charuco":
        min_points_per_sample = int(
            (calibration_target or {}).get("min_corners_per_frame", 12)
        )
    else:
        min_points_per_sample = 0
    gates = [
        {
            "name": "sample_count",
            "status": "pass" if len(sample_records) >= int(min_total_samples) else "fail",
            "severity": "required",
            "evidence": f"samples={len(sample_records)}, required={min_total_samples}",
            "action": "Collect more valid calibration-target views before trusting the intrinsic result.",
        },
        {
            "name": "image_coverage",
            "status": (
                "pass"
                if coverage is not None
                and int(coverage["occupied_cell_count"]) >= occupied_cell_target
                and float(coverage["horizontal_span_ratio"]) >= 0.35
                and float(coverage["vertical_span_ratio"]) >= 0.35
                else "warning"
            ),
            "severity": "required",
            "evidence": (
                "occupied_cells="
                f"{None if coverage is None else coverage['occupied_cell_count']}, "
                "horizontal_span_ratio="
                f"{None if coverage is None else coverage['horizontal_span_ratio']}, "
                "vertical_span_ratio="
                f"{None if coverage is None else coverage['vertical_span_ratio']}"
            ),
            "action": "Collect target views across more image regions instead of clustering near the center.",
        },
        {
            "name": "feature_count_per_sample",
            "status": (
                "pass"
                if target_type not in ("aprilgrid", "charuco")
                else (
                    "pass"
                    if target_points_per_sample
                    and float(np.percentile(np.asarray(target_points_per_sample, dtype=float), 20))
                    >= float(min_points_per_sample)
                    else "warning"
                )
            ),
            "severity": "required" if target_type in ("aprilgrid", "charuco") else "advisory",
            "evidence": (
                "target_type="
                f"{target_type}, points_per_sample_p20="
                f"{None if not target_points_per_sample else float(np.percentile(np.asarray(target_points_per_sample, dtype=float), 20))}, "
                f"min_required={min_points_per_sample}"
            ),
            "action": "Increase visible target features per frame and avoid heavy occlusion/crop during capture.",
        },
        {
            "name": "avg_reprojection",
            "status": "pass" if float(avg_error) <= 1.0 else "warning",
            "severity": "required",
            "evidence": f"avg_reprojection_error_px={float(avg_error)}",
            "action": "Recheck board dimensions, image sharpness, and capture mode if average reprojection remains high.",
        },
        {
            "name": "sample_image_size_consistency",
            "status": (
                "pass"
                if image_size_report is not None and bool(image_size_report.get("consistent"))
                else "fail"
            ),
            "severity": "required",
            "evidence": (
                "unique_sample_sizes="
                f"{None if image_size_report is None else image_size_report.get('unique_sample_sizes')}"
            ),
            "action": "Use one native camera mode per calibration run; do not mix images captured at different resolutions.",
        },
        {
            "name": "sample_vs_capture_resolution",
            "status": (
                "pass"
                if image_size_report is None
                or image_size_report.get("matches_actual_capture_resolution") in (None, True)
                else "warning"
            ),
            "severity": "required",
            "evidence": (
                "matches_actual_capture_resolution="
                f"{None if image_size_report is None else image_size_report.get('matches_actual_capture_resolution')}"
            ),
            "action": "Review whether saved accepted frames still match the camera's actual capture resolution before trusting the intrinsics.",
        },
        {
            "name": "per_view_reprojection",
            "status": (
                "pass"
                if per_view_rms
                and float(np.percentile(np.asarray(per_view_rms, dtype=float), 95)) <= 1.5
                else "warning"
            ),
            "severity": "required",
            "evidence": (
                "per_view_rms_p95_px="
                f"{None if not per_view_rms else float(np.percentile(np.asarray(per_view_rms, dtype=float), 95))}"
            ),
            "action": "Remove weak captures and recollect views with better feature sharpness and pose diversity.",
        },
        {
            "name": "radial_monotonicity",
            "status": monotonicity_report["status"],
            "severity": "required",
            "evidence": (
                "min_radial_derivative=" f"{float(monotonicity_report['min_radial_derivative'])}"
            ),
            "action": "Treat non-monotonic radial distortion as calibration failure; verify capture mode and recollect broader views.",
        },
        {
            "name": "capture_mode_review",
            "status": (
                "pass"
                if not (capture_runtime_info or {}).get("force_capture_resolution")
                else "warning"
            ),
            "severity": "advisory",
            "evidence": (
                "force_capture_resolution="
                f"{bool((capture_runtime_info or {}).get('force_capture_resolution'))}"
            ),
            "action": "Prefer native capture mode for intrinsic calibration to avoid hidden ISP crop before the 3x3 grid.",
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
    imgpoints,
    comparison_view_path,
    avg_error,
    per_view_report,
    coverage,
    monotonicity_report,
):
    output_path = Path(output_yaml_path)
    diagnostics_dir = output_path.with_name(f"{output_path.stem}_diagnostics")
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    per_view_csv = write_table_csv(diagnostics_dir / "per_view_reprojection.csv", per_view_report)
    sample_records_csv = write_table_csv(diagnostics_dir / "sample_records.csv", sample_records)
    heatmap_path = build_heatmap_artifact(diagnostics_dir, coverage)
    final_acceptance = build_intrinsic_acceptance(
        min_total_samples,
        sample_records,
        capture_runtime_info,
        calibration_target,
        imgpoints,
        avg_error,
        per_view_report,
        coverage,
        monotonicity_report,
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
        "avg_reprojection_error_px": float(avg_error),
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
            "Read diagnostics/data_quality.yaml before trusting average reprojection alone.",
            "Inspect per_view_reprojection.csv for tail samples instead of only the mean.",
            "Inspect image_coverage_heatmap.png to confirm the calibration target covered multiple image regions.",
            "Confirm sample_image_sizes stays consistent and matches the actual capture resolution.",
            "Treat radial_monotonicity warnings as calibration failure, not a cosmetic issue.",
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
