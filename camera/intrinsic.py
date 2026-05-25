#!/usr/bin/env python3

# Copyright 2025 WheelOS. All Rights Reserved.
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

# Created Date: 2025-11-06
# Author: daohu527


"""Camera intrinsic calibrator.

Added headless mode: process a directory of images (--images-dir) to perform
calibration without any GUI. This enables CI and smoke tests.
"""

import glob
from pathlib import Path
import cv2
import numpy as np
import yaml
import time
import os
from datetime import datetime

from calibration_common.evaluation import (
    build_final_acceptance,
    write_acceptance_artifacts,
    write_paradigm_artifacts,
    write_table_csv,
)


def _float_list_summary(values):
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


class CameraCalibrator:
    def __init__(self, cfg_path):
        """Initialize and load configuration"""
        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.pattern_size = tuple(self.cfg["pattern_size"])
        self.square_size = self.cfg["square_size"]
        self.ac_cfg = self.cfg["auto_capture_settings"]
        self.window_name = self.cfg.get("window_name", "Camera Calibrator")

        # Camera calibration data container
        self.objpoints, self.imgpoints = [], []
        self.objp = np.zeros(
            (self.pattern_size[0] * self.pattern_size[1], 3), np.float32
        )
        self.objp[:, :2] = np.mgrid[
            0 : self.pattern_size[0], 0 : self.pattern_size[1]
        ].T.reshape(-1, 2)
        self.objp *= self.square_size

        # state
        self.state = "CAPTURING"
        self.mtx, self.dist = None, None
        self.feedback_text = "Searching for board..."
        self.result_canvas = None
        self.last_raw_frame = None
        self.capture_runtime_info = None
        self.sample_records = []
        self.comparison_view_path = None

        self._reset_auto_capture_state()

    def _reset_auto_capture_state(self):
        ac = self.ac_cfg
        self.grid_shape = tuple(ac["grid_shape"])
        self.samples_per_grid = ac["samples_per_grid"]
        self.min_total_samples = (
            self.grid_shape[0] * self.grid_shape[1] * self.samples_per_grid
        )
        self.grid_coverage = np.zeros(self.grid_shape, dtype=int)
        self.stability_counter = 0
        self.last_corners_center = None
        self.last_capture_time = 0
        self.objpoints, self.imgpoints = [], []
        self.sample_records = []
        self.state = "CAPTURING"
        print("\n[INFO] Session reset and ready.")

    def _display_size(self):
        return (
            int(self.cfg.get("window_width", 1280)),
            int(self.cfg.get("window_height", 720)),
        )

    def _capture_config(self):
        capture_cfg = self.cfg.get("capture", {}) or {}
        width = capture_cfg.get("width", self.cfg.get("capture_width"))
        height = capture_cfg.get("height", self.cfg.get("capture_height"))
        return {
            "force_resolution": bool(
                capture_cfg.get(
                    "force_resolution",
                    self.cfg.get("force_capture_resolution", False),
                )
            ),
            "width": None if width in (None, "") else int(width),
            "height": None if height in (None, "") else int(height),
            "fourcc": capture_cfg.get("fourcc"),
        }

    def _apply_capture_settings(self, cap):
        capture_cfg = self._capture_config()
        fourcc = capture_cfg.get("fourcc")
        if isinstance(fourcc, str) and len(fourcc) == 4:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if (
            capture_cfg["force_resolution"]
            and capture_cfg["width"] is not None
            and capture_cfg["height"] is not None
        ):
            print(
                "[INFO] Requesting capture resolution "
                f"{capture_cfg['width']}x{capture_cfg['height']}."
            )
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_cfg["width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_cfg["height"])
        else:
            print(
                "[INFO] Capture resolution is not forced. "
                "Using the camera's native/as-is output avoids accidental FOV crop."
            )

    def _build_capture_runtime_info(self, cap, frame):
        capture_cfg = self._capture_config()
        display_w, display_h = self._display_size()
        actual_h, actual_w = frame.shape[:2]
        reported_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        reported_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        warnings = []
        if capture_cfg["force_resolution"]:
            warnings.append(
                "Forced capture resolution can crop some sensors before the 3x3 grid is drawn."
            )
            warnings.append(
                "If the live grid looks clipped, disable capture.force_resolution or switch to a native mode such as 4:3."
            )
        if abs((display_w / max(display_h, 1)) - (actual_w / max(actual_h, 1))) > 0.05:
            warnings.append(
                "Display window aspect differs from capture; the app letterboxes for display, but that is not sensor crop."
            )
        return {
            "camera_index": int(self.cfg["camera_index"]),
            "requested_capture_resolution": (
                None
                if capture_cfg["width"] is None or capture_cfg["height"] is None
                else {
                    "width": int(capture_cfg["width"]),
                    "height": int(capture_cfg["height"]),
                }
            ),
            "force_capture_resolution": bool(capture_cfg["force_resolution"]),
            "fourcc": capture_cfg.get("fourcc"),
            "actual_capture_resolution": {
                "width": int(actual_w),
                "height": int(actual_h),
            },
            "reported_capture_resolution": {
                "width": int(reported_w),
                "height": int(reported_h),
            },
            "display_resolution": {
                "width": int(display_w),
                "height": int(display_h),
            },
            "warnings": warnings,
        }

    def _log_capture_runtime_info(self, runtime_info):
        actual = runtime_info["actual_capture_resolution"]
        display = runtime_info["display_resolution"]
        requested = runtime_info.get("requested_capture_resolution")
        print(
            "[INFO] Live capture resolution:",
            f"{actual['width']}x{actual['height']}",
            "| display window:",
            f"{display['width']}x{display['height']}",
        )
        if requested is not None:
            print(
                "[INFO] Requested capture resolution:",
                f"{requested['width']}x{requested['height']}",
            )
        for warning in runtime_info.get("warnings", []):
            print("[WARN]", warning)

    def _draw_capture_runtime_info(self, image):
        runtime_info = self.capture_runtime_info or {}
        actual = runtime_info.get("actual_capture_resolution", {})
        display = runtime_info.get("display_resolution", {})
        requested = runtime_info.get("requested_capture_resolution")
        lines = [
            (
                "Capture "
                f"{actual.get('width', 0)}x{actual.get('height', 0)}"
                f" | Display {display.get('width', 0)}x{display.get('height', 0)}"
            ),
            (
                "Forced capture: "
                + (
                    "ON"
                    if runtime_info.get("force_capture_resolution")
                    else "OFF (preferred for full FOV)"
                )
            ),
        ]
        if requested is not None:
            lines.append(
                "Requested capture "
                f"{requested.get('width', 0)}x{requested.get('height', 0)}"
            )
        warnings = runtime_info.get("warnings", [])
        if warnings:
            lines.extend(warnings[:2])

        start_y = max(30, image.shape[0] - 140)
        for index, line in enumerate(lines):
            color = (0, 180, 255) if index >= 2 else (255, 255, 255)
            self._draw_text(
                image,
                line,
                (30, start_y + index * 30),
                color=color,
                scale=0.7,
                thickness=2,
            )

    def _build_sample_record(
        self,
        refined,
        image_size_wh,
        *,
        source,
        source_path=None,
        grid_cell=None,
    ):
        width, height = int(image_size_wh[0]), int(image_size_wh[1])
        corners = np.asarray(refined, dtype=float).reshape(-1, 2)
        bbox_min = np.min(corners, axis=0)
        bbox_max = np.max(corners, axis=0)
        center = np.mean(corners, axis=0)
        if grid_cell is None:
            cell_x = min(2, max(0, int((center[0] / max(width, 1)) * 3.0)))
            cell_y = min(2, max(0, int((center[1] / max(height, 1)) * 3.0)))
        else:
            cell_x = int(grid_cell[0])
            cell_y = int(grid_cell[1])
        return {
            "sample_id": len(self.sample_records) + 1,
            "source": str(source),
            "source_path": source_path,
            "grid_cell": {"x": cell_x, "y": cell_y},
            "image_size_wh": {"width": width, "height": height},
            "image_bbox": {
                "min_xy_px": {"x": float(bbox_min[0]), "y": float(bbox_min[1])},
                "max_xy_px": {"x": float(bbox_max[0]), "y": float(bbox_max[1])},
                "center_xy_normalized": {
                    "x": float(center[0] / max(width, 1)),
                    "y": float(center[1] / max(height, 1)),
                },
                "edge_margin_px": float(
                    min(
                        bbox_min[0],
                        bbox_min[1],
                        max(width - bbox_max[0], 0.0),
                        max(height - bbox_max[1], 0.0),
                    )
                ),
                "bbox_area_ratio": float(
                    max((bbox_max[0] - bbox_min[0]), 0.0)
                    * max((bbox_max[1] - bbox_min[1]), 0.0)
                    / max(width * height, 1)
                ),
            },
        }

    def _append_sample(
        self,
        refined,
        image_size_wh,
        *,
        source,
        source_path=None,
        grid_cell=None,
    ):
        self.objpoints.append(self.objp.copy())
        self.imgpoints.append(np.asarray(refined, dtype=np.float32).copy())
        self.sample_records.append(
            self._build_sample_record(
                refined,
                image_size_wh,
                source=source,
                source_path=source_path,
                grid_cell=grid_cell,
            )
        )

    def _coverage_metrics(self):
        if not self.sample_records:
            return None
        grid_counts = [[0, 0, 0] for _ in range(3)]
        center_x = []
        center_y = []
        margins = []
        areas = []
        for record in self.sample_records:
            grid_cell = record["grid_cell"]
            grid_counts[int(grid_cell["y"])][int(grid_cell["x"])] += 1
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
            "edge_margin_px": _float_list_summary(margins),
            "bbox_area_ratio": _float_list_summary(areas),
            "per_sample": list(self.sample_records),
        }

    def _per_view_reprojection_report(self, rvecs, tvecs):
        rows = []
        for index in range(len(self.objpoints)):
            imgpts2, _ = cv2.projectPoints(
                self.objpoints[index],
                rvecs[index],
                tvecs[index],
                self.mtx,
                self.dist,
            )
            observed = np.asarray(self.imgpoints[index], dtype=float).reshape(-1, 2)
            predicted = np.asarray(imgpts2, dtype=float).reshape(-1, 2)
            residuals = predicted - observed
            point_errors = np.linalg.norm(residuals, axis=1)
            record = (
                self.sample_records[index] if index < len(self.sample_records) else {}
            )
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

    def _distortion_monotonicity_report(self, image_size_wh):
        coeffs = np.asarray(self.dist, dtype=float).reshape(-1)
        k1 = float(coeffs[0]) if coeffs.size > 0 else 0.0
        k2 = float(coeffs[1]) if coeffs.size > 1 else 0.0
        k3 = float(coeffs[4]) if coeffs.size > 4 else 0.0
        width, height = int(image_size_wh[0]), int(image_size_wh[1])
        fx = float(self.mtx[0, 0]) if self.mtx is not None else 1.0
        fy = float(self.mtx[1, 1]) if self.mtx is not None else 1.0
        cx = float(self.mtx[0, 2]) if self.mtx is not None else width / 2.0
        cy = float(self.mtx[1, 2]) if self.mtx is not None else height / 2.0
        corner_radii = []
        for px, py in ((0.0, 0.0), (width, 0.0), (0.0, height), (width, height)):
            xn = (px - cx) / max(fx, 1e-6)
            yn = (py - cy) / max(fy, 1e-6)
            corner_radii.append(float(np.sqrt(xn**2 + yn**2)))
        max_radius = max(max(corner_radii), 1.0)
        sample_r = np.linspace(0.0, max_radius, 256)
        derivative = (
            1.0
            + 3.0 * k1 * sample_r**2
            + 5.0 * k2 * sample_r**4
            + 7.0 * k3 * sample_r**6
        )
        min_derivative = float(np.min(derivative))
        return {
            "status": "pass" if min_derivative > 0.0 else "warning",
            "max_normalized_radius": float(max_radius),
            "min_radial_derivative": min_derivative,
            "sample_count": int(sample_r.size),
        }

    def _build_heatmap_artifact(self, diagnostics_dir, coverage):
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
        image = np.full(
            (rows * cell_size + 120, cols * cell_size + 120, 3), 245, np.uint8
        )
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

    def _build_intrinsic_acceptance(
        self, avg_error, per_view_report, coverage, monotonicity_report
    ):
        per_view_rms = [float(row["rms_px"]) for row in per_view_report]
        occupied_cell_target = max(4, min(6, int(self.min_total_samples)))
        gates = [
            {
                "name": "sample_count",
                "status": (
                    "pass"
                    if len(self.sample_records) >= int(self.min_total_samples)
                    else "fail"
                ),
                "severity": "required",
                "evidence": f"samples={len(self.sample_records)}, required={self.min_total_samples}",
                "action": "Collect more valid checkerboard views before trusting the intrinsic result.",
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
                "action": "Collect checkerboard views across more image regions instead of clustering near the center.",
            },
            {
                "name": "avg_reprojection",
                "status": "pass" if float(avg_error) <= 1.0 else "warning",
                "severity": "required",
                "evidence": f"avg_reprojection_error_px={float(avg_error)}",
                "action": "Recheck board dimensions, image sharpness, and capture mode if average reprojection remains high.",
            },
            {
                "name": "per_view_reprojection",
                "status": (
                    "pass"
                    if per_view_rms
                    and float(np.percentile(np.asarray(per_view_rms, dtype=float), 95))
                    <= 1.5
                    else "warning"
                ),
                "severity": "required",
                "evidence": (
                    "per_view_rms_p95_px="
                    f"{None if not per_view_rms else float(np.percentile(np.asarray(per_view_rms, dtype=float), 95))}"
                ),
                "action": "Remove weak captures and recollect views with better corner sharpness and pose diversity.",
            },
            {
                "name": "radial_monotonicity",
                "status": monotonicity_report["status"],
                "severity": "required",
                "evidence": (
                    "min_radial_derivative="
                    f"{float(monotonicity_report['min_radial_derivative'])}"
                ),
                "action": "Treat non-monotonic radial distortion as calibration failure; verify capture mode and recollect broader views.",
            },
            {
                "name": "capture_mode_review",
                "status": (
                    "pass"
                    if not (self.capture_runtime_info or {}).get(
                        "force_capture_resolution"
                    )
                    else "warning"
                ),
                "severity": "advisory",
                "evidence": (
                    "force_capture_resolution="
                    f"{bool((self.capture_runtime_info or {}).get('force_capture_resolution'))}"
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

    def _write_review_artifacts(
        self,
        output_yaml_path,
        *,
        avg_error,
        per_view_report,
        coverage,
        monotonicity_report,
    ):
        output_path = Path(output_yaml_path)
        diagnostics_dir = output_path.with_name(f"{output_path.stem}_diagnostics")
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        per_view_csv = write_table_csv(
            diagnostics_dir / "per_view_reprojection.csv", per_view_report
        )
        sample_records_csv = write_table_csv(
            diagnostics_dir / "sample_records.csv", self.sample_records
        )
        heatmap_path = self._build_heatmap_artifact(diagnostics_dir, coverage)
        final_acceptance = self._build_intrinsic_acceptance(
            avg_error, per_view_report, coverage, monotonicity_report
        )
        acceptance_artifacts = write_acceptance_artifacts(
            diagnostics_dir, final_acceptance
        )
        standardized_data = {
            "schema_version": 1,
            "module": "camera_intrinsic",
            "representation": "checkerboard_image_samples",
            "sample_counts": {
                "accepted_samples": int(len(self.sample_records)),
                "required_samples": int(self.min_total_samples),
            },
            "capture_runtime": self.capture_runtime_info,
            "sample_records": list(self.sample_records),
        }
        data_quality = {
            "schema_version": 1,
            "module": "camera_intrinsic",
            "status": final_acceptance["status"],
            "release_ready": final_acceptance["release_ready"],
            "quality_gates": final_acceptance["gates"],
            "avg_reprojection_error_px": float(avg_error),
            "per_view_reprojection_summary": _float_list_summary(
                [float(row["rms_px"]) for row in per_view_report]
            ),
            "image_coverage": coverage,
            "radial_monotonicity": monotonicity_report,
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
                        self.comparison_view_path,
                        heatmap_path,
                    )
                    if item is not None
                ],
            },
            "manual_review": [
                "Read diagnostics/data_quality.yaml before trusting average reprojection alone.",
                "Inspect per_view_reprojection.csv for tail samples instead of only the mean.",
                "Inspect image_coverage_heatmap.png to confirm the checkerboard covered multiple image regions.",
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
            "paradigm": paradigm_artifacts,
            "per_view_reprojection_csv": per_view_csv,
            "sample_records_csv": sample_records_csv,
            "image_coverage_heatmap": heatmap_path,
        }

    def run(self):
        """Interactive GUI capture mode (existing behaviour)."""
        print("[INFO] Industrial Calibration Tool (Grid Overlay Stable Edition)")
        cap = cv2.VideoCapture(self.cfg["camera_index"], cv2.CAP_V4L2)
        if not cap.isOpened():
            print("[ERROR] Cannot open camera index", self.cfg["camera_index"])
            return

        self._apply_capture_settings(cap)

        # Force window size
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        win_w, win_h = self._display_size()
        cv2.resizeWindow(self.window_name, win_w, win_h)

        h, w, grid_overlay = None, None, None
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Capture the last frame of the original image for result comparison.
            self.last_raw_frame = frame.copy()

            if h is None:
                h, w = frame.shape[:2]
                self.capture_runtime_info = self._build_capture_runtime_info(cap, frame)
                self._log_capture_runtime_info(self.capture_runtime_info)
                grid_overlay = self._generate_grid_overlay(w, h)

            display = frame.copy()

            if self.state == "CAPTURING":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found, corners = self._find_corners(gray, frame_count)

                # 1. Overlay static blue grid
                display = cv2.addWeighted(display, 1.0, grid_overlay, 0.8, 0)

                # 2. Run corner detection and automatic data acquisition logic
                if found:
                    cv2.drawChessboardCorners(display, self.pattern_size, corners, True)
                    self._run_auto_capture(gray, corners, w, h)
                else:
                    self.feedback_text = "Searching..."

                # 3. Design the dynamic UI (green completed grid and text).
                self._draw_dynamic_ui(display, w, h)
                self._draw_capture_runtime_info(display)

            elif self.state == "SHOWING_RESULT":
                if self.result_canvas is not None:
                    display = self.result_canvas

            elif self.state == "VALIDATING":
                if self.mtx is not None:
                    undist, preview_info = self._undistort_for_preview(frame)
                    undist = self._draw_valid_roi(undist, preview_info)
                    display = np.hstack((frame, undist))
                    self._draw_text(
                        display,
                        f"Left: Distorted | Right: Undistorted alpha={preview_info['alpha']:.2f} (green ROI = valid crop)",
                        (50, 50),
                        (0, 255, 0),
                    )

            if h is not None:
                self._draw_text(
                    display, "R: Restart | V: Validate | ESC: Exit", (50, h - 40)
                )

            # Render to window: preserve aspect ratio and center-pad to avoid distortion
            src_h, src_w = display.shape[:2]
            if src_w == 0 or src_h == 0:
                # fallback
                render_frame = display
            else:
                scale = min(win_w / src_w, win_h / src_h)
                nw, nh = max(1, int(src_w * scale)), max(1, int(src_h * scale))
                resized = cv2.resize(display, (nw, nh))
                canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                x0 = (win_w - nw) // 2
                y0 = (win_h - nh) // 2
                canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
                render_frame = canvas

            cv2.imshow(self.window_name, render_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("[INFO] Exit by ESC key.")
                break
            elif key == ord("r"):
                print("[INFO] Restart calibration.")
                self._reset_auto_capture_state()
            elif key == ord("v"):
                if self.mtx is not None:
                    print("[INFO] Entering validation view.")
                    self.state = "VALIDATING"

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

    def run_headless(
        self, images_dir: str, patterns=("*.png", "*.jpg", "*.jpeg")
    ) -> int:
        """Process a directory of images to run calibration without GUI.

        Returns 0 on success, 1 if no images found, 2 on calibration failure.
        """
        if not images_dir or not os.path.isdir(images_dir):
            print("[ERROR] images_dir must point to an existing directory.")
            return 1

        # collect images
        img_paths = []
        for p in patterns:
            img_paths.extend(sorted(glob.glob(os.path.join(images_dir, p))))
        if not img_paths:
            print(f"[ERROR] No images found in {images_dir} with patterns {patterns}")
            return 1

        print(f"[INFO] Found {len(img_paths)} images; processing...")
        processed = 0
        h = w = None
        for ip in img_paths:
            img = cv2.imread(ip)
            if img is None:
                print(f"[WARN] Could not read image {ip}, skipping")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # direct detection (no resizing) for headless runs
            found, corners = cv2.findChessboardCorners(
                gray,
                self.pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if not found:
                continue
            refined = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            self._append_sample(
                refined,
                (int(img.shape[1]), int(img.shape[0])),
                source="headless",
                source_path=str(ip),
            )
            processed += 1
            if h is None:
                h, w = img.shape[:2]
            print(f"[OK] Captured sample #{processed} from {os.path.basename(ip)}")
            if len(self.objpoints) >= self.min_total_samples:
                break

        if len(self.objpoints) < self.min_total_samples:
            print(
                f"[ERROR] Not enough valid samples ({len(self.objpoints)}/{self.min_total_samples})"
            )
            return 2

        # run calibration
        self._calibrate(w, h)
        # success if self.mtx set
        if self.mtx is not None:
            print("[PASS] Headless calibration completed.")
            return 0
        else:
            print("[FAIL] Headless calibration failed.")
            return 2

    def _undistortion_preview_config(self):
        return self.cfg.get("undistortion_preview", {}) or {}

    def _build_undistortion_model(self, image_size_wh, alpha=None):
        if self.mtx is None or self.dist is None:
            raise RuntimeError(
                "Camera must be calibrated before building undistortion preview."
            )
        preview_cfg = self._undistortion_preview_config()
        preview_alpha = float(preview_cfg.get("alpha", 1.0) if alpha is None else alpha)
        width, height = map(int, image_size_wh)
        center_principal_point = bool(preview_cfg.get("center_principal_point", False))
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.mtx,
            self.dist,
            (width, height),
            preview_alpha,
            (width, height),
            centerPrincipalPoint=center_principal_point,
        )
        x, y, roi_w, roi_h = [int(value) for value in roi]
        preview_info = {
            "alpha": float(preview_alpha),
            "center_principal_point": center_principal_point,
            "optimized_camera_matrix": np.asarray(
                new_camera_matrix, dtype=float
            ).tolist(),
            "valid_roi": {
                "x": x,
                "y": y,
                "width": roi_w,
                "height": roi_h,
            },
        }
        return new_camera_matrix, preview_info

    def _undistort_for_preview(self, image, alpha=None):
        new_camera_matrix, preview_info = self._build_undistortion_model(
            (image.shape[1], image.shape[0]), alpha=alpha
        )
        undistorted = cv2.undistort(image, self.mtx, self.dist, None, new_camera_matrix)
        return undistorted, preview_info

    def _draw_valid_roi(self, image, preview_info):
        roi = (preview_info or {}).get("valid_roi") or {}
        x = int(roi.get("x", 0))
        y = int(roi.get("y", 0))
        roi_w = int(roi.get("width", 0))
        roi_h = int(roi.get("height", 0))
        annotated = image.copy()
        if roi_w > 0 and roi_h > 0:
            cv2.rectangle(
                annotated,
                (x, y),
                (x + roi_w - 1, y + roi_h - 1),
                (0, 255, 0),
                2,
            )
        return annotated

    def _draw_dynamic_ui(self, display, w, h):
        """Draw dynamically updating UI elements: completed green grid and status text."""
        # 1. Draw the highlighted green grid.
        gh, gw = self.grid_shape
        for r in range(gh):
            for c in range(gw):
                if self.grid_coverage[r, c] > 0:
                    y0, x0 = int(r * h / gh), int(c * w / gw)
                    y1, x1 = int((r + 1) * h / gh), int((c + 1) * w / gw)

                    # Create a temporary transparent layer the same size as the displayed image.
                    overlay = display.copy()
                    # Draw a solid green rectangle on this temporary layer.
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), -1)
                    # Blend this temporary layer with the green squares with the main display image.
                    cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)

        # 2. Draw status text
        self._draw_text(display, self.feedback_text, (50, 60), (0, 255, 255))
        self._draw_text(
            display,
            f"Samples: {len(self.objpoints)}/{self.min_total_samples}",
            (50, 110),
            (0, 255, 255),
        )

    def _generate_grid_overlay(self, w, h):
        """Generate a transparent grid layer of fixed size to avoid flickering."""
        overlay = np.zeros((h, w, 3), np.uint8)
        gh, gw = self.ac_cfg["grid_shape"]
        color = (255, 100, 100)
        thickness = 2
        for r in range(gh + 1):
            y = int(round(r * h / gh))
            cv2.line(overlay, (0, y), (w, y), color, thickness)
        for c in range(gw + 1):
            x = int(round(c * w / gw))
            cv2.line(overlay, (x, 0), (x, h), color, thickness)
        return overlay

    def _find_corners(self, gray, frame_counter):
        opt = self.cfg["optimization"]
        if frame_counter % opt["detection_interval"] == 0:
            factor = opt["resize_factor"]
            small = cv2.resize(
                gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA
            )
            found, small_corners = cv2.findChessboardCorners(
                small,
                self.pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK,
            )
            self.last_found = found
            if found:
                self.last_corners = small_corners / factor
        return getattr(self, "last_found", False), getattr(self, "last_corners", None)

    def _run_auto_capture(self, gray, corners, w, h):
        if time.time() - self.last_capture_time < self.ac_cfg["delay_between_captures"]:
            return
        center = np.mean(corners, axis=0)[0]
        if self.last_corners_center is not None:
            dist = np.linalg.norm(center - self.last_corners_center)
            if dist < self.ac_cfg["stability_threshold"]:
                self.stability_counter += 1
                self.feedback_text = f"Hold steady ({self.stability_counter})"
            else:
                self.stability_counter = 0
        self.last_corners_center = center

        if self.stability_counter >= self.ac_cfg["stability_frames"]:
            gx = int(center[0] * self.grid_shape[1] / w)
            gy = int(center[1] * self.grid_shape[0] / h)
            gx, gy = np.clip(gx, 0, self.grid_shape[1] - 1), np.clip(
                gy, 0, self.grid_shape[0] - 1
            )
            if self.grid_coverage[gy, gx] < self.samples_per_grid:
                self._save_sample(gray, corners)
                self.grid_coverage[gy, gx] += 1
            self.stability_counter = 0
            if len(self.objpoints) >= self.min_total_samples:
                self._calibrate(w, h)

    def _save_sample(self, gray, corners):
        print(f"[OK] Captured sample #{len(self.objpoints)+1}")
        refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        center = np.mean(np.asarray(refined, dtype=float).reshape(-1, 2), axis=0)
        width = int(gray.shape[1])
        height = int(gray.shape[0])
        grid_cell = (
            min(2, max(0, int((center[0] / max(width, 1)) * 3.0))),
            min(2, max(0, int((center[1] / max(height, 1)) * 3.0))),
        )
        self._append_sample(
            refined,
            (width, height),
            source="interactive",
            grid_cell=grid_cell,
        )
        self.last_capture_time = time.time()

    def _calibrate(self, w, h):
        print(f"[INFO] Calibrating ({len(self.objpoints)} samples)...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, (w, h), None, None
        )
        if not ret:
            print("[ERROR] Calibration failed.")
            return
        self.mtx, self.dist = mtx, dist
        per_view_report = self._per_view_reprojection_report(rvecs, tvecs)
        err = self._reprojection_error(rvecs, tvecs)
        print(f"[REPORT] Avg Reprojection Error: {err:.4f}px")
        self._build_result_canv(w, h)
        self._save_results(
            w,
            h,
            err,
            per_view_report=per_view_report,
        )
        self.state = "SHOWING_RESULT"

    def _reprojection_error(self, rvecs, tvecs):
        total_err = 0
        for i in range(len(self.objpoints)):
            imgpts2, _ = cv2.projectPoints(
                self.objpoints[i], rvecs[i], tvecs[i], self.mtx, self.dist
            )
            total_err += cv2.norm(self.imgpoints[i], imgpts2, cv2.NORM_L2) / len(
                imgpts2
            )
        return total_err / len(self.objpoints)

    def _build_result_canv(self, w, h):
        print("[INFO] Generating Distortion Comparison View...")
        canvas = np.full((h, w * 2 + 60, 3), 40, np.uint8)

        if self.last_raw_frame is None:
            self.last_raw_frame = np.zeros((h, w, 3), dtype=np.uint8)

        dist_img = self.last_raw_frame.copy()
        und, preview_info = self._undistort_for_preview(dist_img)
        und = self._draw_valid_roi(und, preview_info)
        canvas[:, 20 : 20 + w] = dist_img
        canvas[:, 40 + w : 40 + 2 * w] = und
        self._draw_text(canvas, "Distorted", (50, 50), (200, 200, 255))
        self._draw_text(
            canvas,
            f"Undistorted alpha={preview_info['alpha']:.2f}",
            (w + 80, 50),
            (180, 255, 180),
        )
        self._draw_text(
            canvas,
            "Green ROI shows the all-valid crop window",
            (w + 80, 95),
            (180, 255, 180),
        )
        self.comparison_view_path = "comparison_view.png"
        cv2.imwrite(self.comparison_view_path, canvas)
        print(f"[SAVED] {self.comparison_view_path}")
        self.result_canvas = canvas

    def _draw_text(
        self,
        img,
        text,
        pos,
        color=(255, 255, 255),
        scale=1.0,
        thickness=2,
    ):
        cv2.putText(
            img,
            text,
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
        )

    def _save_results(self, w, h, error, *, per_view_report):
        fname = f"calibration_{datetime.now():%Y%m%d_%H%M%S}.yaml"
        _, preview_info = self._build_undistortion_model((w, h))
        distortion = np.asarray(self.dist, dtype=float).reshape(-1)
        coverage = self._coverage_metrics()
        monotonicity_report = self._distortion_monotonicity_report((w, h))
        data = dict(
            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            image_width=w,
            image_height=h,
            camera_matrix=dict(rows=3, cols=3, data=self.mtx.tolist()),
            distortion_model=str(self.cfg.get("distortion_model", "plumb_bob")),
            distortion_coefficients=dict(
                rows=1,
                cols=int(distortion.size),
                data=distortion.tolist(),
            ),
            capture_runtime=self.capture_runtime_info,
            undistortion_preview=preview_info,
            sample_quality={
                "accepted_sample_count": int(len(self.sample_records)),
                "required_sample_count": int(self.min_total_samples),
                "image_coverage": coverage,
                "radial_monotonicity": monotonicity_report,
            },
            per_view_reprojection_summary=_float_list_summary(
                [float(row["rms_px"]) for row in per_view_report]
            ),
            avg_reprojection_error=float(error),
        )
        with open(fname, "w") as f:
            yaml.dump(data, f, indent=4)
        print(f"[SAVED] Calibration file: {fname}")
        review_artifacts = self._write_review_artifacts(
            fname,
            avg_error=error,
            per_view_report=per_view_report,
            coverage=coverage,
            monotonicity_report=monotonicity_report,
        )
        print(
            "[SAVED] Intrinsic diagnostics:",
            review_artifacts["diagnostics_dir"],
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WheelOS Industrial Camera Calibrator")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Directory of images for headless calibration",
    )
    parser.add_argument(
        "--pattern-size", default=None, help="Override pattern size as W,H (optional)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        default_cfg = dict(
            camera_index=0,
            window_name="Industrial Calibration Tool",
            window_width=1280,
            window_height=720,
            capture={
                "force_resolution": False,
                "width": None,
                "height": None,
                "fourcc": None,
            },
            distortion_model="plumb_bob",
            pattern_size=[11, 8],
            square_size=0.025,
            optimization={"resize_factor": 0.5, "detection_interval": 2},
            undistortion_preview={"alpha": 1.0, "center_principal_point": False},
            auto_capture_settings=dict(
                grid_shape=[3, 3],
                samples_per_grid=1,
                delay_between_captures=1.0,
                stability_frames=5,
                stability_threshold=2.0,
            ),
        )
        with open(args.config, "w") as f:
            yaml.dump(default_cfg, f, indent=4)
        print("[INFO] Default config.yaml created.")

    calibrator = CameraCalibrator(args.config)

    if args.pattern_size:
        try:
            w, h = map(int, args.pattern_size.split(","))
            calibrator.pattern_size = (w, h)
            calibrator.objp = np.zeros((w * h, 3), np.float32)
            calibrator.objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
            calibrator.objp *= calibrator.square_size
        except Exception:
            print("[WARN] invalid --pattern-size, ignoring")

    if args.images_dir:
        rc = calibrator.run_headless(args.images_dir)
        raise SystemExit(rc)
    else:
        calibrator.run()
