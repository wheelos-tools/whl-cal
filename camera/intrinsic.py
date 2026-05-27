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
import cv2
import numpy as np
import yaml
import time
import os
from datetime import datetime

from camera.intrinsic_capture import (
    apply_capture_settings,
    build_capture_runtime_info,
    capture_config,
    display_size,
    is_visually_empty_frame,
    log_capture_runtime_info,
    open_managed_capture,
    resolve_capture_source,
)
from camera.intrinsic_evaluation import (
    coverage_metrics,
    distortion_monotonicity_report,
    float_list_summary,
    per_view_reprojection_report,
    write_review_artifacts,
)
from camera.intrinsic_sampling import IntrinsicSamplingState
from camera.intrinsic_solver import (
    build_undistortion_model,
    calibrate_camera,
    mean_reprojection_error,
    undistort_for_preview,
)
from camera.intrinsic_targets import CalibrationTargetDetector
from camera.intrinsic_visualization import (
    build_comparison_canvas,
    draw_aprilgrid_debug,
    draw_capture_runtime_info,
    draw_dynamic_ui,
    draw_text,
    draw_valid_roi,
    generate_grid_overlay,
    render_preserving_aspect_ratio,
)
from camera.intrinsic_workspace import IntrinsicWorkspace


def _float_list_summary(values):
    return float_list_summary(values)


class CameraCalibrator:
    def __init__(self, cfg_path, session_name=None, capture_only=False):
        """Initialize and load configuration"""
        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.ac_cfg = self.cfg["auto_capture_settings"]
        self.window_name = self.cfg.get("window_name", "Camera Calibrator")
        self.target_type = str(
            self.cfg.get("target_type", self.cfg.get("pattern_type", "chessboard"))
        ).lower()
        self.pattern_size = None
        self.square_size = None
        if self.target_type == "chessboard":
            pattern_size = self.cfg.get("pattern_size")
            square_size = self.cfg.get("square_size")
            if pattern_size is None or square_size is None:
                raise ValueError(
                    "Chessboard calibration requires pattern_size and square_size in the config."
                )
            self.pattern_size = tuple(pattern_size)
            self.square_size = float(square_size)
        self.capture_only = bool(capture_only)
        self.require_release_ready = bool(self.cfg.get("require_release_ready", False))
        self.last_release_ready = None
        self.session_name = session_name
        self.workspace = IntrinsicWorkspace(
            self.cfg,
            self.target_type,
            session_name=session_name,
        )

        # Camera calibration data container
        self.sampling = IntrinsicSamplingState(self.ac_cfg)
        self.target_detector = CalibrationTargetDetector(
            self.cfg,
            self.pattern_size,
            self.square_size,
        )

        # state
        self.state = "CAPTURING"
        self.mtx, self.dist = None, None
        self.feedback_text = "Searching for board..."
        self.result_canvas = None
        self.last_raw_frame = None
        self.capture_runtime_info = None
        self.comparison_view_path = None
        self.capture_session = None
        self.run_session = None
        self.dataset_images_dir = None
        self.live_capture_handle = None
        self.preexisting_capture_sample_count = 0
        self.last_detection_debug = None
        self._last_aprilgrid_debug_signature = None
        self.last_sampling_debug = None
        self._last_sampling_debug_signature = None
        self.last_sampling_progress = self.sampling.progress_snapshot()
        self.capture_source, self.capture_source_meta = resolve_capture_source(self.cfg)

        self._reset_auto_capture_state()

    def _reset_auto_capture_state(self):
        self.sampling.reset()
        self.state = "CAPTURING"
        self.last_detection_debug = None
        self._last_aprilgrid_debug_signature = None
        self.last_sampling_debug = None
        self._last_sampling_debug_signature = None
        self.last_sampling_progress = self.sampling.progress_snapshot()
        self.preexisting_capture_sample_count = 0
        print("\n[INFO] Session reset and ready.")

    @property
    def objpoints(self):
        return self.sampling.objpoints

    @property
    def imgpoints(self):
        return self.sampling.imgpoints

    @property
    def sample_records(self):
        return self.sampling.sample_records

    @property
    def grid_shape(self):
        return self.sampling.grid_shape

    @property
    def samples_per_grid(self):
        return self.sampling.samples_per_grid

    @property
    def min_total_samples(self):
        return self.sampling.min_total_samples

    @property
    def grid_coverage(self):
        return self.sampling.grid_coverage

    def _prepare_live_capture_session(self):
        if self.capture_session is None:
            self.capture_session = self.workspace.prepare_capture_session()
            self.preexisting_capture_sample_count = int(
                len(list(self.capture_session.accepted_dir.glob("sample_*.jpg")))
            )
            self.dataset_images_dir = str(self.capture_session.accepted_dir)
            print(
                "[INFO] Live accepted samples directory:",
                self.capture_session.accepted_dir,
            )
            if self.preexisting_capture_sample_count > 0:
                print(
                    "[WARN] Reusing a capture directory that already contains accepted samples:",
                    f"existing_samples={self.preexisting_capture_sample_count}.",
                    "Use a new --session-name or clear the directory if you want an isolated run.",
                )
            self._write_capture_session_manifest(status="collecting")
        return self.capture_session

    def _prepare_run_session(self, dataset_label=None):
        if self.run_session is None:
            self.run_session = self.workspace.prepare_run_session(dataset_label=dataset_label)
            print(
                "[INFO] Calibration artifacts directory:",
                self.run_session.session_dir,
            )
        return self.run_session

    def _base_capture_runtime_snapshot(self):
        display_w, display_h = display_size(self.cfg)
        cap_cfg = capture_config(self.cfg, self.capture_source_meta)
        snapshot = {
            "capture_source": str(self.capture_source),
            "capture_source_type": self.capture_source_meta.get("source_type"),
            "selected_camera_index": self.capture_source_meta.get("selected_camera_index"),
            "requested_capture_resolution": (
                None
                if cap_cfg["width"] is None or cap_cfg["height"] is None
                else {
                    "width": int(cap_cfg["width"]),
                    "height": int(cap_cfg["height"]),
                }
            ),
            "force_capture_resolution": bool(cap_cfg["force_resolution"]),
            "fourcc": cap_cfg.get("fourcc"),
            "fps": cap_cfg.get("fps"),
            "codec": cap_cfg.get("codec"),
            "requested_buffersize": cap_cfg.get("buffersize"),
            "display_resolution": {
                "width": int(display_w),
                "height": int(display_h),
            },
        }
        if self.live_capture_handle is not None:
            snapshot["reported_capture_resolution"] = {
                "width": int(self.live_capture_handle.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.live_capture_handle.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
            if hasattr(self.live_capture_handle, "diagnostics"):
                snapshot["stream_health"] = self.live_capture_handle.diagnostics()
        return snapshot

    def _capture_runtime_snapshot(self):
        snapshot = (
            dict(self.capture_runtime_info)
            if self.capture_runtime_info is not None
            else self._base_capture_runtime_snapshot()
        )
        if self.live_capture_handle is not None and hasattr(self.live_capture_handle, "diagnostics"):
            snapshot["stream_health"] = self.live_capture_handle.diagnostics()
        return snapshot

    def _freeze_capture_runtime_info(self):
        self.capture_runtime_info = self._capture_runtime_snapshot()
        return self.capture_runtime_info

    def _draw_capture_runtime_info(self, image):
        draw_capture_runtime_info(image, self._capture_runtime_snapshot())

    def _write_capture_session_manifest(self, status):
        if self.capture_session is None:
            return
        accepted_total = len(list(self.capture_session.accepted_dir.glob("sample_*.jpg")))
        capture_runtime = self._capture_runtime_snapshot()
        data = {
            "schema_version": 1,
            "status": str(status),
            "capture_only": bool(self.capture_only),
            "target_type": self.target_type,
            "calibration_target": self.target_detector.target_config(),
            "capture_source": str(self.capture_source),
            "capture_runtime": capture_runtime,
            "accepted_dir": str(self.capture_session.accepted_dir),
            "accepted_sample_count": int(accepted_total),
            "preexisting_accepted_sample_count": int(self.preexisting_capture_sample_count),
            "accepted_sample_count_current_run": int(len(self.sample_records)),
            "required_sample_count": int(self.min_total_samples),
            "latest_detection_debug": self.last_detection_debug,
            "latest_sampling_debug": self.last_sampling_debug,
            "latest_sampling_progress": self.last_sampling_progress,
            "sample_records": list(self.sample_records),
        }
        with open(self.capture_session.manifest_path, "w") as f:
            yaml.dump(data, f, indent=4)

    def _record_detection_debug(self, detection_result, frame_counter, source):
        debug_info = dict(getattr(detection_result, "debug_info", None) or {})
        if not debug_info:
            return
        debug_info["frame_counter"] = int(frame_counter)
        debug_info["source"] = str(source)
        self.last_detection_debug = debug_info
        if self.target_type != "aprilgrid" or debug_info.get("skipped"):
            return

        signature = (
            bool(debug_info.get("found")),
            debug_info.get("failure_stage"),
            int(debug_info.get("detected_marker_count", 0)),
            int(debug_info.get("matched_point_count", 0)),
            tuple(debug_info.get("selected_marker_ids", [])[:10]),
            debug_info.get("selected_scale"),
        )
        should_print = (
            self._last_aprilgrid_debug_signature != signature
            or frame_counter % 90 == 0
        )
        if should_print:
            attempts = " ".join(
                f"{float(item.get('scale', 0.0)):.2f}x:{int(item.get('detected_marker_count', 0))}"
                for item in (debug_info.get("attempts") or [])[:6]
            )
            print(
                "[DEBUG] AprilGrid:",
                f"found={bool(debug_info.get('found'))}",
                f"markers={int(debug_info.get('detected_marker_count', 0))}/{int(debug_info.get('min_tags_per_frame', 0))}",
                f"points={int(debug_info.get('matched_point_count', 0))}",
                f"scale={debug_info.get('selected_scale')}",
                f"stage={debug_info.get('failure_stage') or 'ok'}",
                f"ids={list(debug_info.get('selected_marker_ids') or [])[:10]}",
                f"attempts={attempts}",
            )
            self._last_aprilgrid_debug_signature = signature

    def _record_sampling_debug(self, sampling_debug, frame_counter, source):
        sampling_debug = dict(sampling_debug or {})
        if not sampling_debug:
            return
        self.last_sampling_progress = self.sampling.progress_snapshot()
        sampling_debug.update(
            {
                "frame_counter": int(frame_counter),
                "source": str(source),
                "samples_collected": int(len(self.objpoints)),
                "required_sample_count": int(self.min_total_samples),
                "sampling_progress": dict(self.last_sampling_progress),
            }
        )
        self.last_sampling_debug = sampling_debug
        signature = (
            int(sampling_debug.get("stability_counter", 0)),
            None
            if sampling_debug.get("motion_px") is None
            else round(float(sampling_debug.get("motion_px")), 2),
            round(float(sampling_debug.get("effective_threshold_px", 0.0)), 2),
            int(len(self.objpoints)),
            bool(sampling_debug.get("accept")) if "accept" in sampling_debug else None,
            sampling_debug.get("capture_reason"),
        )
        should_print = (
            self._last_sampling_debug_signature != signature
            and sampling_debug.get("motion_px") is not None
        )
        if should_print and (
            int(sampling_debug.get("stability_counter", 0)) > 0
            or float(sampling_debug.get("motion_px") or 0.0)
            > float(sampling_debug.get("effective_threshold_px") or 0.0)
        ):
            print(
                "[DEBUG] Sampling:",
                f"stable={int(sampling_debug.get('stability_counter', 0))}/{self.sampling.stability_frames}",
                f"motion_px={sampling_debug.get('motion_px')}",
                f"threshold_px={sampling_debug.get('effective_threshold_px')}",
                f"bbox_diag_px={sampling_debug.get('bbox_diagonal_px')}",
                f"accept={sampling_debug.get('accept')}",
                f"reason={sampling_debug.get('capture_reason')}",
                f"coverage_complete={sampling_debug.get('coverage_complete')}",
                f"remaining={sampling_debug.get('remaining_required_samples')}",
                f"closest_sample_id={sampling_debug.get('closest_sample_id')}",
            )
        self._last_sampling_debug_signature = signature

    def _draw_target_detection(self, image, detection_result):
        if detection_result is None or not detection_result.found:
            return
        if self.target_type == "aprilgrid":
            if detection_result.marker_ids is not None:
                cv2.aruco.drawDetectedMarkers(
                    image,
                    detection_result.marker_corners,
                    detection_result.marker_ids,
                )
        elif self.target_type == "charuco":
            if detection_result.image_points is not None:
                cv2.aruco.drawDetectedCornersCharuco(
                    image,
                    detection_result.image_points,
                    detection_result.feature_ids,
                )
            if detection_result.marker_ids is not None and detection_result.marker_corners:
                cv2.aruco.drawDetectedMarkers(
                    image,
                    detection_result.marker_corners,
                    detection_result.marker_ids,
                )
        else:
            cv2.drawChessboardCorners(
                image,
                self.pattern_size,
                detection_result.image_points,
                True,
            )

    def _build_pose_rejection_feedback(self, capture_decision):
        remaining_samples = int(capture_decision.get("remaining_required_samples") or 0)
        guidance_parts = []
        area_delta = capture_decision.get("closest_area_delta")
        aspect_delta = capture_decision.get("closest_aspect_delta")
        center_delta = capture_decision.get("closest_center_distance_ratio")
        if area_delta is not None and float(area_delta) < float(self.sampling.pose_novelty_area_delta):
            guidance_parts.append("move closer or farther")
        if aspect_delta is not None and float(aspect_delta) < float(self.sampling.pose_novelty_aspect_delta):
            guidance_parts.append("tilt the board more")
        if center_delta is not None and float(center_delta) < float(self.sampling.pose_novelty_center_distance_ratio):
            guidance_parts.append("shift the board center")
        if not guidance_parts:
            guidance_parts.append("make a clearly different pose")
        if len(guidance_parts) == 1:
            action_text = guidance_parts[0]
        elif len(guidance_parts) == 2:
            action_text = f"{guidance_parts[0]} and {guidance_parts[1]}"
        else:
            action_text = ", ".join(guidance_parts[:-1]) + f", and {guidance_parts[-1]}"
        closest_sample_id = capture_decision.get("closest_sample_id")
        if closest_sample_id:
            return (
                f"Too similar to sample #{int(closest_sample_id)}: {action_text} "
                f"({remaining_samples} novel poses left)"
            )
        return f"Coverage done: {action_text} ({remaining_samples} novel poses left)"

    def _coverage_metrics(self):
        return coverage_metrics(self.sample_records, grid_shape=self.grid_shape)

    def _per_view_reprojection_report(self, rvecs, tvecs):
        return per_view_reprojection_report(
            self.objpoints,
            self.imgpoints,
            self.mtx,
            self.dist,
            self.sample_records,
            rvecs,
            tvecs,
        )

    def _distortion_monotonicity_report(self, image_size_wh):
        return distortion_monotonicity_report(self.mtx, self.dist, image_size_wh)

    def _write_review_artifacts(
        self,
        output_yaml_path,
        *,
        avg_error,
        per_view_report,
        coverage,
        monotonicity_report,
    ):
        return write_review_artifacts(
            output_yaml_path,
            min_total_samples=self.min_total_samples,
            sample_records=self.sample_records,
            capture_runtime_info=self.capture_runtime_info,
            calibration_target=self.target_detector.target_config(),
            imgpoints=self.imgpoints,
            comparison_view_path=self.comparison_view_path,
            avg_error=avg_error,
            per_view_report=per_view_report,
            coverage=coverage,
            monotonicity_report=monotonicity_report,
        )

    def run(self):
        """Interactive GUI capture mode (existing behaviour)."""
        print("[INFO] Industrial Calibration Tool (Grid Overlay Stable Edition)")
        print(f"[INFO] Capture source: {self.capture_source}")
        self._prepare_live_capture_session()
        cap, _backend_name = open_managed_capture(
            self.capture_source,
            self.cfg,
            self.capture_source_meta,
        )
        if cap is None:
            print("[ERROR] Cannot open capture source", self.capture_source)
            return 1
        self.live_capture_handle = cap

        apply_capture_settings(cap, self.cfg, self.capture_source_meta)

        # Force window size
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        win_w, win_h = display_size(self.cfg)
        cv2.resizeWindow(self.window_name, win_w, win_h)

        h, w, grid_overlay = None, None, None
        frame_count = 0
        empty_frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                if getattr(cap, "managed_capture", False) and hasattr(cap, "is_ready") and not cap.is_ready():
                    if frame_count % 30 == 0:
                        print("[WARN] Waiting for network stream recovery/warm-up...")
                    frame_count += 1
                    continue
                empty_frame_count += 1
                if empty_frame_count % 30 == 0:
                    print("[WARN] Waiting for a fresh frame from capture source...")
                if empty_frame_count > 240:
                    print(
                        "[ERROR] Too many empty frame waits from capture source. "
                        "Check RTSP stability, codec settings, or camera connectivity."
                    )
                    break
                frame_count += 1
                continue
            if is_visually_empty_frame(frame):
                print(
                    "[WARN] Received invalid image frame (blank or near-uniform, e.g. solid green); waiting for a valid frame."
                )
                frame_count += 1
                if frame_count > 90:
                    print(
                        "[ERROR] Too many invalid frames from capture source. "
                        "Check whether the selected /dev/video node is wrong, the camera is returning a bogus ISP stream, or you should switch to RTSP/camera_uri."
                    )
                    break
                continue
            empty_frame_count = 0

            # Capture the last frame of the original image for result comparison.
            self.last_raw_frame = frame.copy()

            if h is None:
                h, w = frame.shape[:2]
                self.capture_runtime_info = build_capture_runtime_info(
                    self.cfg,
                    self.capture_source,
                    self.capture_source_meta,
                    cap,
                    frame,
                )
                log_capture_runtime_info(self.capture_runtime_info)
                grid_overlay = generate_grid_overlay((w, h), self.grid_shape)

            display = frame.copy()

            if self.state == "CAPTURING":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detection = self._find_target(gray, frame_count)
                self._record_detection_debug(detection, frame_count, "interactive")

                # 1. Overlay static blue grid
                display = cv2.addWeighted(display, 1.0, grid_overlay, 0.8, 0)

                # 2. Run corner detection and automatic data acquisition logic
                if detection.found:
                    self._draw_target_detection(display, detection)
                    capture_complete = self._run_auto_capture(
                        gray,
                        detection,
                        w,
                        h,
                        frame_bgr=frame,
                        source="interactive",
                        frame_counter=frame_count,
                    )
                    if capture_complete and self.capture_only:
                        print("[PASS] Capture-only session completed.")
                        break
                else:
                    self.feedback_text = "Searching..."

                # 3. Design the dynamic UI (green completed grid and text).
                draw_dynamic_ui(
                    display,
                    self.grid_coverage,
                    self.grid_shape,
                    self.feedback_text,
                    self.sampling.progress_snapshot(),
                )
                draw_aprilgrid_debug(display, self.last_detection_debug)
                self._draw_capture_runtime_info(display)

            elif self.state == "SHOWING_RESULT":
                if self.result_canvas is not None:
                    display = self.result_canvas

            elif self.state == "VALIDATING":
                if self.mtx is not None:
                    undist, preview_info = self._undistort_for_preview(frame)
                    undist = draw_valid_roi(undist, preview_info)
                    display = np.hstack((frame, undist))
                    draw_text(
                        display,
                        f"Left: Distorted | Right: Undistorted alpha={preview_info['alpha']:.2f} (green ROI = valid crop)",
                        (50, 50),
                        (0, 255, 0),
                    )

            if h is not None:
                draw_text(
                    display, "R: Restart | V: Validate | ESC: Exit", (50, h - 40)
                )

            # Render to window: preserve aspect ratio and center-pad to avoid distortion
            src_h, src_w = display.shape[:2]
            if src_w == 0 or src_h == 0:
                # fallback
                render_frame = display
            else:
                render_frame = render_preserving_aspect_ratio(display, (win_w, win_h))

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

        self._freeze_capture_runtime_info()
        cap.release()
        self.live_capture_handle = None
        cv2.destroyAllWindows()
        if self.capture_only:
            if len(self.objpoints) >= self.min_total_samples:
                self._write_capture_session_manifest(status="capture_complete")
                return 0
            self._write_capture_session_manifest(status="capture_incomplete")
            print(
                "[ERROR] Capture-only session did not finish.",
                f"samples={len(self.objpoints)}/{self.min_total_samples}",
            )
            return 2
        if self.mtx is not None and self.require_release_ready and not bool(
            self.last_release_ready
        ):
            print("[FAIL] Calibration finished but quality gates are not release-ready.")
            return 3
        return 0

    def run_live_headless(self, max_seconds=0):
        """Live camera capture without GUI; safe on servers without DISPLAY."""
        print("[INFO] Headless live mode: GUI disabled, running auto capture loop.")
        print(f"[INFO] Capture source: {self.capture_source}")
        self._prepare_live_capture_session()
        cap, _backend_name = open_managed_capture(
            self.capture_source,
            self.cfg,
            self.capture_source_meta,
        )
        if cap is None:
            print("[ERROR] Cannot open capture source", self.capture_source)
            return 1
        self.live_capture_handle = cap

        apply_capture_settings(cap, self.cfg, self.capture_source_meta)
        start_ts = time.time()
        h = w = None
        frame_count = 0
        empty_frame_count = 0
        first_frame_saved = False

        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                if getattr(cap, "managed_capture", False) and hasattr(cap, "is_ready") and not cap.is_ready():
                    if frame_count % 30 == 0:
                        print("[WARN] Waiting for network stream recovery/warm-up...")
                    frame_count += 1
                    continue
                print("[WARN] Failed to read frame from camera; retrying...")
                frame_count += 1
                empty_frame_count += 1
                if empty_frame_count > 90:
                    print(
                        "[ERROR] Too many empty frames. Check stream URI/codec/network and OpenCV ffmpeg support."
                    )
                    break
                continue
            if is_visually_empty_frame(frame):
                empty_frame_count += 1
                if empty_frame_count % 15 == 0:
                    print(
                        "[WARN] Received invalid image frames repeatedly (blank or near-uniform, e.g. solid green); "
                        "continuing to wait for a valid decoded frame."
                    )
                if empty_frame_count > 120:
                    print(
                        "[ERROR] Too many invalid frames. Check whether the selected /dev/video node is wrong, "
                        "the camera is returning a bogus ISP stream, or the RTSP/codec path is misconfigured."
                    )
                    break
                continue
            empty_frame_count = 0

            self.last_raw_frame = frame.copy()

            if h is None:
                h, w = frame.shape[:2]
                self.capture_runtime_info = build_capture_runtime_info(
                    self.cfg,
                    self.capture_source,
                    self.capture_source_meta,
                    cap,
                    frame,
                )
                log_capture_runtime_info(self.capture_runtime_info)

            if not first_frame_saved:
                debug_frame_path = self.workspace.debug_image_path(
                    self.capture_session,
                    "headless_first_frame.jpg",
                )
                if cv2.imwrite(str(debug_frame_path), frame):
                    print(f"[SAVED] First headless frame: {debug_frame_path}")
                first_frame_saved = True

            if self.state == "CAPTURING":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detection = self._find_target(gray, frame_count)
                self._record_detection_debug(detection, frame_count, "headless_live")
                if detection.found:
                    capture_complete = self._run_auto_capture(
                        gray,
                        detection,
                        w,
                        h,
                        frame_bgr=frame,
                        source="headless_live",
                        frame_counter=frame_count,
                    )
                    if capture_complete and self.capture_only:
                        self._write_capture_session_manifest(status="capture_complete")
                        print("[PASS] Headless capture-only session completed.")
                        self._freeze_capture_runtime_info()
                        cap.release()
                        self.live_capture_handle = None
                        return 0

                if frame_count % 30 == 0:
                    progress = self.sampling.progress_snapshot()
                    print(
                        "[INFO] Headless progress:",
                        f"stage={progress['stage']}",
                        f"samples={progress['sample_count']}/{progress['required_sample_count']}",
                        f"coverage={progress['coverage_cell_count']}/{progress['coverage_target_cell_count']}",
                        f"remaining={progress['remaining_required_samples']}",
                        f"next={progress.get('guidance_summary')}",
                    )

            if self.state == "SHOWING_RESULT" and self.mtx is not None:
                if self.require_release_ready and not bool(self.last_release_ready):
                    print(
                        "[FAIL] Calibration finished but quality gates are not release-ready."
                    )
                    self._freeze_capture_runtime_info()
                    cap.release()
                    self.live_capture_handle = None
                    return 3
                print("[PASS] Headless live calibration completed.")
                self._freeze_capture_runtime_info()
                cap.release()
                self.live_capture_handle = None
                return 0

            if max_seconds > 0 and (time.time() - start_ts) >= float(max_seconds):
                progress = self.sampling.progress_snapshot()
                print(
                    "[WARN] Headless live mode timed out before collecting enough samples.",
                    f"stage={progress['stage']}",
                    f"samples={progress['sample_count']}/{progress['required_sample_count']}",
                    f"coverage={progress['coverage_cell_count']}/{progress['coverage_target_cell_count']}",
                )
                break

            frame_count += 1

        self._freeze_capture_runtime_info()
        cap.release()
        self.live_capture_handle = None
        if self.capture_only:
            self._write_capture_session_manifest(status="capture_incomplete")
            progress = self.sampling.progress_snapshot()
            print(
                "[ERROR] Headless capture-only session did not finish.",
                f"stage={progress['stage']}",
                f"samples={progress['sample_count']}/{progress['required_sample_count']}",
                f"coverage={progress['coverage_cell_count']}/{progress['coverage_target_cell_count']}",
            )
            return 2
        if self.mtx is not None:
            if self.require_release_ready and not bool(self.last_release_ready):
                print(
                    "[FAIL] Calibration finished but quality gates are not release-ready."
                )
                return 3
            print("[PASS] Headless live calibration completed.")
            return 0
        print(
            "[ERROR] Headless live calibration did not finish.",
            f"samples={len(self.objpoints)}/{self.min_total_samples}",
        )
        return 2

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

        self.dataset_images_dir = os.path.abspath(images_dir)
        dataset_label = self.workspace.dataset_label_from_images_dir(images_dir)
        self._prepare_run_session(dataset_label=dataset_label)
        matched_preview = ", ".join(
            os.path.basename(path) for path in img_paths[: min(5, len(img_paths))]
        )
        print(f"[INFO] Found {len(img_paths)} images; processing...")
        if len(img_paths) < self.min_total_samples:
            print(
                "[WARN] images_dir contains fewer candidate images than the minimum sample target:",
                f"images={len(img_paths)} required_samples={self.min_total_samples}",
            )
            print(
                "[HINT] Point --images-dir to a folder of raw calibration captures from multiple poses,",
                f"not a results folder. Matched files: {matched_preview}",
            )
        processed = 0
        h = w = None
        for frame_index, ip in enumerate(img_paths):
            img = cv2.imread(ip)
            if img is None:
                print(f"[WARN] Could not read image {ip}, skipping")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detection = self._find_target(gray, frame_index, detect_every_frame=True)
            self._record_detection_debug(detection, frame_index, "headless_dataset")
            if not detection.found:
                continue
            img_h, img_w = img.shape[:2]
            if h is None:
                h, w = img_h, img_w
            elif (img_h, img_w) != (h, w):
                print(
                    "[WARN] Skipping image with mismatched size:",
                    f"{ip} -> got {img_w}x{img_h}, expected {w}x{h}",
                )
                continue
            self.last_raw_frame = img.copy()
            self._save_sample(
                gray,
                detection,
                source="headless",
                source_path=str(ip),
            )
            processed += 1
            print(f"[OK] Captured sample #{processed} from {os.path.basename(ip)}")

        if len(self.objpoints) < self.min_total_samples:
            print(
                f"[ERROR] Not enough valid samples ({len(self.objpoints)}/{self.min_total_samples})"
            )
            print(
                "[HINT] The detector only uses frames where the full calibration target is recognized.",
                f"Candidate images scanned={len(img_paths)}. Example matches: {matched_preview}",
            )
            return 2

        # run calibration
        self._calibrate(w, h)
        # success if self.mtx set
        if self.mtx is not None:
            if self.require_release_ready and not bool(self.last_release_ready):
                print("[FAIL] Calibration finished but quality gates are not release-ready.")
                return 3
            print("[PASS] Headless calibration completed.")
            return 0
        else:
            print("[FAIL] Headless calibration failed.")
            return 2

    def _build_undistortion_model(self, image_size_wh, alpha=None):
        if self.mtx is None or self.dist is None:
            raise RuntimeError(
                "Camera must be calibrated before building undistortion preview."
            )
        return build_undistortion_model(
            self.mtx,
            self.dist,
            image_size_wh,
            self.cfg.get("undistortion_preview", {}) or {},
            alpha=alpha,
        )

    def _undistort_for_preview(self, image, alpha=None):
        return undistort_for_preview(
            image,
            self.mtx,
            self.dist,
            self.cfg.get("undistortion_preview", {}) or {},
            alpha=alpha,
        )

    def _find_target(self, gray, frame_counter, detect_every_frame=False):
        optimization = dict(self.cfg["optimization"])
        if detect_every_frame:
            optimization["detection_interval"] = 1
        return self.target_detector.detect(gray, frame_counter, optimization)

    def _run_auto_capture(
        self,
        gray,
        detection,
        w,
        h,
        frame_bgr=None,
        source="interactive",
        frame_counter=0,
    ):
        if not self.sampling.can_capture_now():
            return False
        sampling_debug = self.sampling.note_detection(detection.image_points)
        stability_counter = int(sampling_debug.get("stability_counter", 0))
        self._record_sampling_debug(sampling_debug, frame_counter, source)
        if stability_counter > 0:
            motion_px = sampling_debug.get("motion_px")
            threshold_px = sampling_debug.get("effective_threshold_px")
            if motion_px is not None:
                self.feedback_text = (
                    f"Hold steady ({stability_counter}/{self.sampling.stability_frames}, "
                    f"motion {float(motion_px):.1f}px < {float(threshold_px):.1f}px)"
                )
            else:
                self.feedback_text = f"Hold steady ({stability_counter})"
        elif sampling_debug.get("motion_px") is not None:
            motion_px = float(sampling_debug.get("motion_px") or 0.0)
            threshold_px = float(sampling_debug.get("effective_threshold_px") or 0.0)
            if motion_px >= threshold_px:
                self.feedback_text = (
                    f"Stabilize board (motion {motion_px:.1f}px > {threshold_px:.1f}px)"
                )

        if stability_counter >= self.sampling.stability_frames:
            capture_decision = self.sampling.evaluate_capture_candidate(
                detection.image_points,
                (w, h),
            )
            sampling_debug.update(capture_decision)
            self._record_sampling_debug(sampling_debug, frame_counter, source)
            remaining_samples = int(capture_decision.get("remaining_required_samples") or 0)
            remaining_cells = int(self.sampling.remaining_coverage_cells)
            if bool(capture_decision.get("accept")):
                self._save_sample(
                    gray,
                    detection,
                    source=source,
                    frame_bgr=frame_bgr,
                    capture_reason=str(capture_decision.get("capture_reason", "")),
                )
            else:
                capture_reason = str(capture_decision.get("capture_reason", ""))
                if capture_reason == "move_to_uncovered_cells":
                    self.feedback_text = (
                        f"Move target to uncovered cells ({remaining_cells} cells left)"
                    )
                elif capture_reason == "pose_not_novel":
                    self.feedback_text = self._build_pose_rejection_feedback(
                        capture_decision
                    )
                elif capture_reason == "sample_target_met":
                    self.feedback_text = "Capture complete"
            self.sampling.reset_stability()
            if len(self.objpoints) >= self.min_total_samples:
                if self.capture_only:
                    self.feedback_text = "Capture complete"
                    return True
                self._prepare_run_session(dataset_label=self.capture_session.label if self.capture_session else None)
                self._calibrate(w, h)
        return False

    def _save_sample(
        self,
        gray,
        detection,
        source="interactive",
        source_path=None,
        frame_bgr=None,
        capture_reason="",
    ):
        sample_index = len(self.objpoints) + 1
        capture_reason = str(capture_reason or "coverage_needed")
        if capture_reason == "pose_novel":
            print(f"[OK] Captured sample #{sample_index} (diverse pose)")
        elif capture_reason == "coverage_needed":
            print(f"[OK] Captured sample #{sample_index} (coverage)")
        else:
            print(f"[OK] Captured sample #{sample_index}")
        refined = np.asarray(detection.image_points, dtype=np.float32).reshape(-1, 1, 2)
        object_points = np.asarray(detection.object_points, dtype=np.float32).reshape(-1, 3)
        width = int(gray.shape[1])
        height = int(gray.shape[0])
        saved_source_path = source_path
        if (
            saved_source_path is None
            and frame_bgr is not None
            and self.capture_session is not None
            and self.workspace.save_live_accepted_frames
        ):
            sample_path = self.workspace.accepted_sample_path(
                self.capture_session,
                len(self.objpoints) + 1,
            )
            if cv2.imwrite(str(sample_path), frame_bgr):
                saved_source_path = str(sample_path)
                print(f"[SAVED] Accepted sample image: {saved_source_path}")
        self.sampling.append_sample(
            refined,
            (width, height),
            object_points=object_points,
            source=source,
            source_path=saved_source_path,
        )
        if self.capture_session is not None:
            self._write_capture_session_manifest(status="collecting")

    def _calibrate(self, w, h):
        print(f"[INFO] Calibrating ({len(self.objpoints)} samples)...")
        ret, mtx, dist, rvecs, tvecs = calibrate_camera(
            self.objpoints,
            self.imgpoints,
            (w, h),
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
        return mean_reprojection_error(
            self.objpoints,
            self.imgpoints,
            self.mtx,
            self.dist,
            rvecs,
            tvecs,
        )

    def _build_result_canv(self, w, h):
        print("[INFO] Generating Distortion Comparison View...")
        self._prepare_run_session(dataset_label=self.capture_session.label if self.capture_session else None)

        if self.last_raw_frame is None:
            self.last_raw_frame = np.zeros((h, w, 3), dtype=np.uint8)

        dist_img = self.last_raw_frame.copy()
        und, preview_info = self._undistort_for_preview(dist_img)
        und = draw_valid_roi(und, preview_info)
        canvas = build_comparison_canvas(dist_img, und, preview_info)
        self.comparison_view_path = str(self.run_session.comparison_view_path)
        cv2.imwrite(self.comparison_view_path, canvas)
        print(f"[SAVED] {self.comparison_view_path}")
        self.result_canvas = canvas

    def _save_results(self, w, h, error, *, per_view_report):
        self._prepare_run_session(dataset_label=self.capture_session.label if self.capture_session else None)
        fname = str(self.run_session.calibration_yaml_path)
        self.capture_runtime_info = self._capture_runtime_snapshot()
        _, preview_info = self._build_undistortion_model((w, h))
        distortion = np.asarray(self.dist, dtype=float).reshape(-1)
        coverage = self._coverage_metrics()
        monotonicity_report = self._distortion_monotonicity_report((w, h))
        data = dict(
            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            workflow={
                "run_dir": str(self.run_session.session_dir),
                "dataset_images_dir": self.dataset_images_dir,
                "capture_session_dir": (
                    str(self.capture_session.session_dir)
                    if self.capture_session is not None
                    else None
                ),
            },
            image_width=w,
            image_height=h,
            calibration_target=self.target_detector.target_config(),
            camera_matrix=dict(rows=3, cols=3, data=self.mtx.tolist()),
            distortion_model=str(self.cfg.get("distortion_model", "plumb_bob")),
            distortion_coefficients=dict(
                rows=1,
                cols=int(distortion.size),
                data=distortion.tolist(),
            ),
            capture_runtime=self.capture_runtime_info,
            undistortion_preview=preview_info,
            latest_detection_debug=self.last_detection_debug,
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
        self.last_release_ready = bool(review_artifacts.get("release_ready", False))
        print(
            "[SAVED] Intrinsic diagnostics:",
            review_artifacts["diagnostics_dir"],
        )


if __name__ == "__main__":
    from camera.cli import main

    main()
