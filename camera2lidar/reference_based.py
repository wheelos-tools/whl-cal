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
import open3d as o3d
import yaml
from pathlib import Path
from scipy.optimize import least_squares


class ReferenceBasedCalibrator:
    """
    Calibrates LiDAR-to-Camera extrinsics by jointly optimizing over multiple poses.
    This class is designed for robustness and high precision, reflecting industry best practices.
    """

    def __init__(self, config_path):
        print("--- Production-Grade LiDAR-Camera Calibrator v2.0 ---")
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # Load all parameters from config, ensuring correct data types
        self.cam_intrinsics = np.array(
            self.cfg["camera"]["intrinsics"], dtype=np.float64
        )
        self.cam_distortion = np.array(
            self.cfg["camera"]["distortion"], dtype=np.float64
        )
        self.board_pattern = tuple(self.cfg["checkerboard"]["pattern_size"])
        self.board_square_size = float(self.cfg["checkerboard"]["square_size"])

        # Create the ideal 3D object points for the checkerboard in its own coordinate frame (Z=0)
        self.objp_board_frame = np.zeros(
            (self.board_pattern[0] * self.board_pattern[1], 3), np.float32
        )
        self.objp_board_frame[:, :2] = np.mgrid[
            0 : self.board_pattern[0], 0 : self.board_pattern[1]
        ].T.reshape(-1, 2)
        self.objp_board_frame *= self.board_square_size
        print("[CONFIG] System configured successfully.")

    def _find_2d_corners(self, image):
        """Finds checkerboard corners in the 2D image with subpixel refinement."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray,
            self.board_pattern,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK,
        )
        if not found:
            return None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return corners

    def _find_3d_corners(self, pcd):
        """Segments the board plane and calculates precise 3D corner coordinates via geometric construction."""
        try:
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=self.cfg["point_cloud"]["plane_dist_thresh"],
                ransac_n=3,
                num_iterations=1000,
            )
        except RuntimeError:
            return None  # segment_plane can fail on empty point clouds

        if len(inliers) < self.cfg["point_cloud"]["min_plane_points"]:
            return None

        [a, b, c, d] = plane_model
        normal_z = np.array([a, b, c]) / np.linalg.norm([a, b, c])

        plane_pcd = pcd.select_by_index(inliers)
        plane_centroid = plane_pcd.get_center()

        # Create a robust coordinate system on the plane, avoiding gimbal lock
        global_up = np.array([0, 0, 1])
        if np.abs(np.dot(normal_z, global_up)) > 0.95:
            global_up = np.array([0, 1, 0])
        normal_x = np.cross(
            global_up, normal_z
        )  # Swapped order for right-handed system
        normal_x /= np.linalg.norm(normal_x)
        normal_y = np.cross(normal_z, normal_x)

        R_board_to_lidar = np.stack([normal_x, normal_y, normal_z], axis=1)
        centered_objp = self.objp_board_frame - np.mean(self.objp_board_frame, axis=0)
        objp_lidar_frame = centered_objp @ R_board_to_lidar.T + plane_centroid

        return objp_lidar_frame

    def _process_one_pose(self, image_path, pcd_path):
        """Extracts 2D-3D correspondences for a single pose."""
        image = cv2.imread(str(image_path))
        pcd = o3d.io.read_point_cloud(str(pcd_path))

        if not pcd.has_points():
            print(f"[WARN] Point cloud file {pcd_path.name} is empty or invalid.")
            return None

        img_points = self._find_2d_corners(image)
        if img_points is None:
            print(f"[WARN] Failed to find 2D corners in {image_path.name}.")
            return None

        obj_points = self._find_3d_corners(pcd)
        if obj_points is None:
            print(f"[WARN] Failed to find 3D plane in {pcd_path.name}.")
            return None

        return {
            "obj_points": obj_points,
            "img_points": img_points,
            "image_path": image_path,
            "pcd_path": pcd_path,
        }

    def _objective_function(self, params, all_obj_points, all_img_points):
        """Calculates total reprojection error given extrinsic parameters."""
        rvec, tvec = params[:3], params[3:]
        all_errors = []
        for obj_points, img_points in zip(all_obj_points, all_img_points):
            proj_points, _ = cv2.projectPoints(
                obj_points, rvec, tvec, self.cam_intrinsics, self.cam_distortion
            )
            errors = (proj_points.squeeze() - img_points.squeeze()).flatten()
            all_errors.extend(errors)
        return np.array(all_errors)

    def _get_best_initial_guess(self, all_pose_data, all_obj_points, all_img_points):
        """
        Expert Practice: Find the single pose that provides the most globally
        consistent result to use as the initial guess for the optimizer.
        """
        print("[INFO] Searching for the best initial guess among all poses...")
        best_rms = float("inf")
        best_params = None
        best_pose_idx = -1

        for i, pose in enumerate(all_pose_data):
            try:
                _, rvec_i, tvec_i = cv2.solvePnP(
                    pose["obj_points"],
                    pose["img_points"],
                    self.cam_intrinsics,
                    self.cam_distortion,
                    flags=cv2.SOLVEPNP_IPPE,
                )
            except cv2.error:
                continue

            params_i = np.concatenate((rvec_i.flatten(), tvec_i.flatten()))
            errors_i = self._objective_function(
                params_i, all_obj_points, all_img_points
            )
            rms_i = np.sqrt(np.mean(errors_i**2))

            if rms_i < best_rms:
                best_rms = rms_i
                best_params = params_i
                best_pose_idx = i

        if best_pose_idx != -1:
            print(
                f"[INFO] Best initial guess found from pose #{best_pose_idx} with RMS Error: {best_rms:.4f} pixels"
            )
        return best_params, best_rms

    def calibrate(self):
        """Main calibration entry point: loads data, runs optimization, and saves results."""
        data_dir = Path(self.cfg["data_directory"])
        image_files = sorted(
            list(data_dir.glob("*.png")) + list(data_dir.glob("*.jpg"))
        )
        pcd_files = sorted(list(data_dir.glob("*.pcd")))

        if len(image_files) != len(pcd_files) or len(image_files) == 0:
            raise ValueError(
                f"Mismatch in image/pcd files or directory '{data_dir}' is empty."
            )

        print(f"[INFO] Found {len(image_files)} potential poses. Processing...")

        all_pose_data = [
            self._process_one_pose(img_f, pcd_f)
            for img_f, pcd_f in zip(image_files, pcd_files)
        ]
        all_pose_data = [p for p in all_pose_data if p is not None]

        min_poses = self.cfg["optimization"]["min_poses"]
        if len(all_pose_data) < min_poses:
            raise RuntimeError(
                f"Insufficient valid poses ({len(all_pose_data)}). Need at least {min_poses}."
            )

        print(
            f"\n[STEP 1] Feature Extraction: Success from {len(all_pose_data)} valid poses."
        )
        all_obj_points = [p["obj_points"] for p in all_pose_data]
        all_img_points = [p["img_points"] for p in all_pose_data]

        initial_params, initial_rms = self._get_best_initial_guess(
            all_pose_data, all_obj_points, all_img_points
        )
        if initial_params is None:
            raise RuntimeError(
                "Could not find a valid initial guess. Check data quality or PnP solver."
            )

        print(
            f"\n[STEP 2] Joint Optimization: Optimizing over {len(all_pose_data)} poses..."
        )
        result = least_squares(
            self._objective_function,
            initial_params,
            args=(all_obj_points, all_img_points),
            method="lm",
            jac="2-point",
            verbose=2,
        )

        print("\n[STEP 3] Analysis and Reporting:")
        if not result.success:
            print(
                "[WARN] Optimization may not have converged. Review results carefully."
            )

        final_params = result.x
        final_errors = result.fun
        final_rms = np.sqrt(np.mean(final_errors**2))

        rvec_final, tvec_final = final_params[:3], final_params[3:]
        R_final, _ = cv2.Rodrigues(rvec_final)

        transformation_matrix = np.eye(4)
        transformation_matrix[0:3, 0:3] = R_final
        transformation_matrix[0:3, 3] = tvec_final

        J = result.jac
        try:
            cov = np.linalg.inv(J.T @ J)
            uncertainty_factor = 2 * final_rms
            param_uncertainties = uncertainty_factor * np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            param_uncertainties = np.full(6, np.nan)
            print(
                "[WARN] Could not compute uncertainty (Jacobian is singular). This may indicate poor pose diversity."
            )

        print("\n--- CALIBRATION REPORT ---")
        print(f"Initial RMS Reprojection Error: {initial_rms:.4f} pixels")
        print(f"Final RMS Reprojection Error:   {final_rms:.4f} pixels")
        print("\nFinal Optimized Extrinsic Matrix (LiDAR to Camera):")
        print(np.round(transformation_matrix, 5))
        print("\nParameter Uncertainty (approx. 95% confidence interval):")
        print(f"  - Rotation Vector (rad): {param_uncertainties[:3]}")
        print(f"  - Translation Vector (m):  {param_uncertainties[3:]}")
        print("--------------------------\n")

        self.save_results(
            transformation_matrix, initial_rms, final_rms, param_uncertainties
        )
        self.visualize(
            all_pose_data[0]["image_path"],
            all_pose_data,
            transformation_matrix,
            final_rms,
        )

    def save_results(self, matrix, initial_rms, final_rms, uncertainties):
        output_dir = Path(self.cfg["output"]["directory"])
        output_dir.mkdir(exist_ok=True)
        data = {
            "extrinsic_matrix": matrix.tolist(),
            "quality_metrics": {
                "rms_error_initial_pixels": float(initial_rms),
                "rms_error_final_pixels": float(final_rms),
                "improvement_percent": (
                    100 * (initial_rms - final_rms) / initial_rms
                    if initial_rms > 0
                    else 0
                ),
            },
            "parameter_uncertainty_approx_95_confidence": {
                "rotation_rad": uncertainties[:3].tolist(),
                "translation_meters": uncertainties[3:].tolist(),
            },
        }
        with open(output_dir / "extrinsics_optimized_report.yaml", "w") as f:
            yaml.dump(data, f, indent=4, sort_keys=False)
        print(
            f"[SAVED] Full report saved to '{output_dir / 'extrinsics_optimized_report.yaml'}'"
        )

    def visualize(self, image_path, all_pose_data, T, final_rms):
        """
        [PERFORMANCE OPTIMIZED] Generates a verification image by projecting points
        from all poses using a vectorized approach with a depth buffer.
        """
        print("[INFO] Generating final verification image (optimized)...")
        output_dir = Path(self.cfg["output"]["directory"])
        vis_image = cv2.imread(str(image_path))
        h, w, _ = vis_image.shape

        # Create an overlay image and a depth buffer for correct occlusion
        vis_overlay = np.zeros_like(vis_image)
        depth_buffer = np.full((h, w), float("inf"), dtype=np.float32)

        for i, pose_data in enumerate(all_pose_data):
            pcd = o3d.io.read_point_cloud(str(pose_data["pcd_path"]))
            if not pcd.has_points():
                continue
            pcd_points = np.asarray(pcd.points)

            pcd_cam_frame = (T[:3, :3] @ pcd_points.T + T[:3, 3:]).T
            img_proj_points, _ = cv2.projectPoints(
                pcd_cam_frame,
                np.zeros(3),
                np.zeros(3),
                self.cam_intrinsics,
                self.cam_distortion,
            )
            img_proj_points = img_proj_points.squeeze()

            depths = pcd_cam_frame[:, 2]

            # --- Vectorized Filtering and Drawing ---
            # 1. Filter points that are in front of the camera and within image bounds
            valid_idx = (
                (depths > 1e-3)
                & (img_proj_points[:, 0] >= 0)
                & (img_proj_points[:, 0] < w)
                & (img_proj_points[:, 1] >= 0)
                & (img_proj_points[:, 1] < h)
            )

            if not np.any(valid_idx):
                continue

            u, v = img_proj_points[valid_idx, :2].astype(int).T
            d = depths[valid_idx]

            # 2. Use a depth buffer to handle occlusions correctly
            #    Only draw a point if it's closer than what's already there
            closer_mask = d < depth_buffer[v, u]
            u, v, d = u[closer_mask], v[closer_mask], d[closer_mask]

            # 3. Update the depth buffer and draw the points on the overlay
            depth_buffer[v, u] = d
            color = cv2.applyColorMap(
                np.array([[i * 255 / len(all_pose_data)]], dtype=np.uint8),
                cv2.COLORMAP_JET,
            )[0][0]
            vis_overlay[v, u] = color

        # Combine the overlay with the original image where points were drawn
        mask = np.any(vis_overlay > 0, axis=-1)
        vis_image[mask] = vis_overlay[mask]

        title_text = f"Verification - Final RMS: {final_rms:.4f} pixels"
        cv2.putText(
            vis_image, title_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5
        )
        cv2.putText(
            vis_image,
            title_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            2,
        )

        save_path = output_dir / "verification_all_poses_final.png"
        cv2.imwrite(str(save_path), vis_image)
        print(f"[SAVED] Final verification image saved to '{save_path}'")


if __name__ == "__main__":
    CONFIG_PATH = "config.yaml"

    if not Path(CONFIG_PATH).exists():
        print(f"'{CONFIG_PATH}' not found. Creating a default configuration file.")
        config_data = {
            "camera": {
                "intrinsics": [
                    [1000.0, 0.0, 960.0],
                    [0.0, 1000.0, 540.0],
                    [0.0, 0.0, 1.0],
                ],
                "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
            },
            "checkerboard": {
                "pattern_size": [8, 6],
                "square_size": 0.05,
            },
            "point_cloud": {
                "plane_dist_thresh": 0.02,
                "min_plane_points": 500,
            },
            "data_directory": "calibration_data",
            "optimization": {"min_poses": 5},
            "output": {"directory": "calibration_output"},
        }
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(config_data, f, indent=4, sort_keys=False)

    try:
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[FATAL] Config file '{CONFIG_PATH}' not found.")
        exit()

    data_dir = Path(config["data_directory"])
    data_dir.mkdir(exist_ok=True)

    print("\n--- ACTION REQUIRED ---")
    print(
        f"1. Please review and edit '{CONFIG_PATH}' with your specific sensor parameters."
    )
    print(f"2. Place synchronized image/pcd pairs in the '{data_dir}' directory.")
    print("   - Naming: 001.png, 001.pcd; 002.jpg, 002.pcd; ... (supports png/jpg)")
    print("   - For BEST results, capture at least 10-15 POSES with high DIVERSITY.")
    print("-----------------------\n")

    image_files = list(data_dir.glob("*.png")) + list(data_dir.glob("*.jpg"))
    if len(image_files) >= config["optimization"]["min_poses"]:
        try:
            calibrator = ReferenceBasedCalibrator(CONFIG_PATH)
            calibrator.calibrate()
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            print(f"\n[FATAL ERROR] Calibration failed: {e}")
    else:
        print(f"[STOP] Not enough data found in '{data_dir}'. Halting.")
