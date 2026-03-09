#!/usr/bin/env python

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


import cv2
import numpy as np
import yaml
import time
import os
from datetime import datetime


class CameraCalibrator:
    def __init__(self, cfg_path):
        """Initialize and load configuration"""
        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.pattern_size = tuple(self.cfg["pattern_size"])
        self.square_size = self.cfg["square_size"]
        self.ac_cfg = self.cfg["auto_capture_settings"]
        self.window_name = self.cfg["window_name"]

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
        self.state = "CAPTURING"
        print("\n[INFO] Session reset and ready.")

    def run(self):
        print("[INFO] Industrial Calibration Tool (Grid Overlay Stable Edition)")
        cap = cv2.VideoCapture(self.cfg["camera_index"], cv2.CAP_V4L2)
        if not cap.isOpened():
            print("[ERROR] Cannot open camera index", self.cfg["camera_index"])
            return

        # Try to request the display resolution from the camera (may be ignored by some drivers)
        req_w = int(self.cfg.get("window_width", 1280))
        req_h = int(self.cfg.get("window_height", 720))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)

        # Force window size
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self.window_name, self.cfg["window_width"], self.cfg["window_height"]
        )

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
                # Log the actual captured frame size (helps debug capture vs window size)
                print("[DEBUG] first captured frame.shape:", frame.shape)
                print(
                    "[DEBUG] cap reported (w,h):",
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                )
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

            elif self.state == "SHOWING_RESULT":
                if self.result_canvas is not None:
                    display = self.result_canvas

            elif self.state == "VALIDATING":
                if self.mtx is not None:
                    undist = cv2.undistort(frame, self.mtx, self.dist, None)
                    display = np.hstack((frame, undist))
                    self._draw_text(
                        display,
                        "Left: Distorted | Right: Undistorted",
                        (50, 50),
                        (0, 255, 0),
                    )

            if h is not None:
                self._draw_text(
                    display, "R: Restart | V: Validate | ESC: Exit", (50, h - 40)
                )

            # Render to window: preserve aspect ratio and center-pad to avoid distortion
            win_w = int(self.cfg.get("window_width", 1280))
            win_h = int(self.cfg.get("window_height", 720))

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
        self.objpoints.append(self.objp)
        self.imgpoints.append(refined)
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
        err = self._reprojection_error(rvecs, tvecs)
        print(f"[REPORT] Avg Reprojection Error: {err:.4f}px")
        self._build_result_canv(w, h)
        self._save_results(w, h, err)
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
        und = cv2.undistort(dist_img, self.mtx, self.dist, None)
        canvas[:, 20 : 20 + w] = dist_img
        canvas[:, 40 + w : 40 + 2 * w] = und
        self._draw_text(canvas, "Distorted", (50, 50), (200, 200, 255))
        self._draw_text(canvas, "Undistorted", (w + 80, 50), (180, 255, 180))
        cv2.imwrite("comparison_view.png", canvas)
        print("[SAVED] comparison_view.png")
        self.result_canvas = canvas

    def _draw_text(self, img, text, pos, color=(255, 255, 255)):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def _save_results(self, w, h, error):
        fname = f"calibration_{datetime.now():%Y%m%d_%H%M%S}.yaml"
        data = dict(
            time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            image_width=w,
            image_height=h,
            camera_matrix=dict(rows=3, cols=3, data=self.mtx.tolist()),
            distortion_coefficients=dict(rows=1, cols=5, data=self.dist.tolist()[0]),
            avg_reprojection_error=float(error),
        )
        with open(fname, "w") as f:
            yaml.dump(data, f, indent=4)
        print(f"[SAVED] Calibration file: {fname}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WheelOS Industrial Camera Calibrator")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        default_cfg = dict(
            camera_index=0,
            window_name="Industrial Calibration Tool",
            window_width=1280,
            window_height=720,
            pattern_size=[11, 8],
            square_size=0.025,
            optimization={"resize_factor": 0.5, "detection_interval": 2},
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
    calibrator.run()
