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
    def __init__(self, config_path):
        """初始化，加载配置文件"""
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # 核心配置
        self.pattern_size = tuple(self.cfg["pattern_size"])
        self.square_size = self.cfg["square_size"]
        self.ac_cfg = self.cfg["auto_capture_settings"]
        self.window_name = self.cfg["window_name"]

        # 标定数据容器
        self.objpoints, self.imgpoints = [], []
        self.objp = np.zeros(
            (self.pattern_size[0] * self.pattern_size[1], 3), np.float32
        )
        self.objp[:, :2] = np.mgrid[
            0 : self.pattern_size[0], 0 : self.pattern_size[1]
        ].T.reshape(-1, 2)
        self.objp *= self.square_size

        # 状态控制
        self.state = "CAPTURING"
        self.mtx, self.dist = None, None
        self.buttons = {}
        self.result_canvas = None
        self.feedback_text = "Searching for board..."

        self._reset_auto_capture_state()

    def _reset_auto_capture_state(self):
        """重置自动采集状态"""
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

    def run(self):
        print("[INFO] Industrial GUI Calibration Tool Started")
        cap = cv2.VideoCapture(self.cfg["camera_index"], cv2.CAP_V4L2)
        if not cap.isOpened():
            print("[ERROR] Cannot open camera index", self.cfg["camera_index"])
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self.window_name, self.cfg["window_width"], self.cfg["window_height"]
        )

        h, w = None, None
        frame_count = 0
        cv2.setMouseCallback(self.window_name, self._on_mouse_click)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if h is None:
                h, w = frame.shape[:2]

            display = frame.copy()
            self.last_raw_frame = frame.copy()

            if self.state == "CAPTURING":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found, corners = self._find_corners(gray, frame_count)
                if found:
                    cv2.drawChessboardCorners(display, self.pattern_size, corners, True)
                    self._run_auto_capture_logic(gray, corners, w, h)
                else:
                    self.feedback_text = "Searching..."
                self._draw_capture_ui(display, w, h)

            elif self.state == "SHOWING_RESULT":
                display = self.result_canvas

            elif self.state == "VALIDATING":
                if self.mtx is not None:
                    undist = cv2.undistort(frame, self.mtx, self.dist, None)
                    display = np.hstack((frame, undist))
                    cv2.putText(
                        display,
                        "Left:Distorted | Right:Undistorted",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

            cv2.imshow(self.window_name, display)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC emergency exit
                break
            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

    def _find_corners(self, gray, frame_count):
        opt_cfg = self.cfg["optimization"]
        if frame_count % opt_cfg["detection_interval"] == 0:
            factor = opt_cfg["resize_factor"]
            small = cv2.resize(gray, None, fx=factor, fy=factor)
            found, corners_small = cv2.findChessboardCorners(
                small,
                self.pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK,
            )
            self.last_found_state = found
            if found:
                self.last_corners = corners_small / factor
        return getattr(self, "last_found_state", False), getattr(
            self, "last_corners", None
        )

    def _run_auto_capture_logic(self, gray, corners, w, h):
        elapsed = time.time() - self.last_capture_time
        if elapsed < self.ac_cfg["delay_between_captures"]:
            return
        current_center = np.mean(corners, axis=0)[0]
        if self.last_corners_center is not None:
            dist = np.linalg.norm(current_center - self.last_corners_center)
            if dist < self.ac_cfg["stability_threshold"]:
                self.stability_counter += 1
                self.feedback_text = f"Hold steady... ({self.stability_counter})"
            else:
                self.stability_counter = 0
        self.last_corners_center = current_center

        if self.stability_counter >= self.ac_cfg["stability_frames"]:
            gx = int(current_center[0] * self.grid_shape[1] / w)
            gy = int(current_center[1] * self.grid_shape[0] / h)
            gx, gy = np.clip(gx, 0, self.grid_shape[1] - 1), np.clip(
                gy, 0, self.grid_shape[0] - 1
            )
            if self.grid_coverage[gy, gx] < self.samples_per_grid:
                self._save_sample(gray, corners)
                self.grid_coverage[gy, gx] += 1
            self.stability_counter = 0
            if len(self.objpoints) >= self.min_total_samples:
                self.calibrate(w, h)

    def _save_sample(self, gray, corners):
        print(f"[OK] Captured sample #{len(self.objpoints)+1}")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        self.objpoints.append(self.objp)
        self.imgpoints.append(refined)
        self.last_capture_time = time.time()

    def calibrate(self, w, h):
        print(f"\n[INFO] Calibrating with {len(self.objpoints)} samples...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, (w, h), None, None
        )
        if not ret:
            print("[ERROR] Calibration failed.")
            return
        self.mtx, self.dist = mtx, dist
        err = self._compute_reprojection_error(rvecs, tvecs)
        print(f"[REPORT] Avg Reprojection Error: {err:.4f}px")
        self._generate_reprojection_canvas(w, h, rvecs, tvecs)
        self._save_results(w, h, err)
        self.state = "SHOWING_RESULT"

    def _compute_reprojection_error(self, rvecs, tvecs):
        total_err = 0
        for i in range(len(self.objpoints)):
            imgpts2, _ = cv2.projectPoints(
                self.objpoints[i], rvecs[i], tvecs[i], self.mtx, self.dist
            )
            err = cv2.norm(self.imgpoints[i], imgpts2, cv2.NORM_L2) / len(imgpts2)
            total_err += err
        return total_err / len(self.objpoints)

    def _generate_reprojection_canvas(self, w, h, rvecs, tvecs):
        """Dual-view (distorted vs undistorted) + clickable GUI."""
        print("[INFO] Generating Distortion Comparison View...")
        # Base canvas
        canvas = np.full((h, w * 2 + 60, 3), 40, np.uint8)
        # Left: last frame (distorted)
        dist_img = self.last_raw_frame.copy()
        if hasattr(self, "last_corners"):
            cv2.drawChessboardCorners(
                dist_img, self.pattern_size, self.last_corners, True
            )
        # Right: undistorted
        undist_img = cv2.undistort(dist_img, self.mtx, self.dist, None)
        canvas[:, 20 : 20 + w] = dist_img
        canvas[:, 40 + w : 40 + 2 * w] = undist_img

        cv2.putText(
            canvas,
            "Distorted",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 200, 200),
            2,
        )
        cv2.putText(
            canvas,
            "Undistorted",
            (w + 80, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (200, 255, 200),
            2,
        )

        self.result_canvas = canvas
        # save result snapshot
        cv2.imwrite("comparison_view.png", canvas)
        print("[SAVED] Comparison image: comparison_view.png")

        # draw GUI buttons
        self.buttons = self._draw_buttons(canvas, w, h)
        cv2.setMouseCallback(self.window_name, self._on_mouse_click)

    def _draw_buttons(self, canvas, w, h):
        """Draw GUI control buttons."""
        yb = h - 80
        buttons = {
            "validate": (80, yb, 380, yb + 60),
            "restart": (w - 220, yb, w + 80, yb + 60),
            "quit": (w + 300, yb, w + 600, yb + 60),
        }
        for name, (x1, y1, x2, y2) in buttons.items():
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 60, 60), -1)
            color = (255, 255, 255)
            cv2.putText(
                canvas,
                name.upper(),
                (x1 + 20, y2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )
        return buttons

    def _on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.state == "SHOWING_RESULT":
            for name, (x1, y1, x2, y2) in self.buttons.items():
                if x1 < x < x2 and y1 < y < y2:
                    if name == "quit":
                        print("[INFO] Exit requested.")
                        cv2.destroyAllWindows()
                        os._exit(0)
                    elif name == "restart":
                        print("[INFO] Restart calibration session.")
                        self._reset_auto_capture_state()
                        self.state = "CAPTURING"
                    elif name == "validate":
                        print("[INFO] Switching to Live Validation View.")
                        self.state = "VALIDATING"

    def _draw_capture_ui(self, display, w, h):
        """绘制采集状态与网格覆盖"""
        gh, gw = self.grid_shape
        for r in range(gh + 1):
            cv2.line(
                display, (0, int(r * h / gh)), (w, int(r * h / gh)), (255, 100, 100), 1
            )
        for c in range(gw + 1):
            cv2.line(
                display, (int(c * w / gw), 0), (int(c * w / gw), h), (255, 100, 100), 1
            )
        for r in range(gh):
            for c in range(gw):
                if self.grid_coverage[r, c] > 0:
                    y0, x0 = int(r * h / gh), int(c * w / gw)
                    y1, x1 = int((r + 1) * h / gh), int((c + 1) * w / gw)
                    overlay = display.copy()
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), -1)
                    cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
        cv2.putText(
            display,
            self.feedback_text,
            (50, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            display,
            f"Samples: {len(self.objpoints)}/{self.min_total_samples}",
            (50, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

    def _save_results(self, w, h, error):
        fname = f"calibration_{datetime.now():%Y%m%d_%H%M%S}.yaml"
        data = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_width": w,
            "image_height": h,
            "camera_matrix": {"rows": 3, "cols": 3, "data": self.mtx.tolist()},
            "distortion_coefficients": {
                "rows": 1,
                "cols": 5,
                "data": self.dist.tolist()[0],
            },
            "avg_reprojection_error": float(error),
        }
        with open(fname, "w") as f:
            yaml.dump(data, f, sort_keys=False)
        print(f"[SAVED] Calibration results: {fname}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Industrial GUI Camera Calibrator")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    # if config missing, create default
    if not os.path.exists(args.config):
        default = {
            "camera_index": 0,
            "window_name": "Industrial Calibration Tool",
            "window_width": 1280,
            "window_height": 720,
            "pattern_size": [11, 8],
            "square_size": 0.025,
            "output_path": "calibration_output.yaml",
            "optimization": {"resize_factor": 0.5, "detection_interval": 2},
            "auto_capture_settings": {
                "grid_shape": [3, 3],
                "samples_per_grid": 1,
                "delay_between_captures": 1.0,
                "stability_frames": 5,
                "stability_threshold": 2.0,
            },
        }
        print(f"[INFO] Created default config.yaml")
        with open(args.config, "w") as f:
            yaml.dump(default, f, sort_keys=False, indent=4)

    calibrator = CameraCalibrator(args.config)
    calibrator.run()
