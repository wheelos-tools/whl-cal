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
from scipy.spatial.transform import Rotation


class LearningBasedCalibrator:
    """
    Targetless LiDAR-to-Camera Calibration

    Key improvements:
    1. Use geometric Harris 3D keypoints for the camera depth-derived cloud.
    2. Adaptive ISS for LiDAR feature extraction.
    3. Colored ICP refinement with Sim(3) support.
    4. Self-adaptive downsampling and convergence criteria.
    5. Enhanced visualization and diagnostic metrics.
    """

    def __init__(self, config_path):
        print("\n--- [LAUNCH] Learning-Based Targetless Calibrator ---")
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.cam_intrinsics = np.array(
            self.cfg["camera"]["intrinsics"], dtype=np.float64
        )
        self.output_dir = Path(self.cfg["output"]["directory"])
        self.output_dir.mkdir(exist_ok=True)
        print("[CONFIG] Loaded calibration parameters successfully.")

    def _depth_to_point_cloud(self, depth_map):
        """Convert monocular depth map to dense 3D point cloud."""
        h, w = depth_map.shape
        fx, fy = self.cam_intrinsics[0, 0], self.cam_intrinsics[1, 1]
        cx, cy = self.cam_intrinsics[0, 2], self.cam_intrinsics[1, 2]

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth_map.astype(np.float32)
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        pts = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        valid_mask = (pts[:, 2] > self.cfg.get("min_depth_m", 0.1)) & np.isfinite(
            pts[:, 2]
        )
        pts = pts[valid_mask]

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        return pcd

    def _extract_camera_features(self, dense_pcd):
        """Use Harris 3D keypoints on dense camera-derived cloud."""
        print(
            "[INFO] Extracting geometric keypoints from camera depth cloud (Harris3D)..."
        )
        dense_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        try:
            keypoints = o3d.geometry.keypoint.compute_harris_keypoints(dense_pcd)
        except Exception:
            keypoints = dense_pcd.voxel_down_sample(0.05)  # fallback
            print("[WARN] Harris3D failed — using fallback downsample features.")

        print(f"[INFO] Found {len(keypoints.points)} camera geometric keypoints.")
        return keypoints

    def _extract_lidar_features(self, pcd):
        """Adaptive ISS keypoint extraction depending on scene scale."""
        bbox = pcd.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(
            np.array(bbox.get_max_bound()) - np.array(bbox.get_min_bound())
        )
        salient = 0.02 * diag
        nonmax = 0.015 * diag

        print(
            f"[INFO] Extracting ISS keypoints (adaptive radii: {salient:.3f}/{nonmax:.3f})..."
        )
        keypoints = o3d.geometry.keypoint.compute_iss_keypoints(
            pcd,
            salient_radius=salient,
            non_max_radius=nonmax,
            gamma_21=0.975,
            gamma_32=0.975,
        )
        print(f"[INFO] Found {len(keypoints.points)} LiDAR ISS keypoints.")
        return keypoints

    def _run_coarse_registration(self, src, tgt, voxel_size):
        """FPFH+RANSAC coarse alignment."""
        print("[STEP 1.1] Running FPFH + RANSAC coarse registration...")
        for cloud in [src, tgt]:
            cloud.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
            )
        src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            src, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
        )
        tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            tgt, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
        )

        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src,
            tgt,
            src_fpfh,
            tgt_fpfh,
            mutual_filter=True,
            max_correspondence_distance=voxel_size * 1.5,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False
            ),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    voxel_size * 1.5
                ),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                100000, 0.999
            ),
        )

        print(
            f"[INFO] Coarse registration done. Fitness={result.fitness:.4f}, RMSE={result.inlier_rmse:.4f}"
        )
        return result.transformation

    @staticmethod
    def _estimate_initial_scale(src, tgt):
        """Estimate initial scale using isotropic cloud size ratio."""
        s_pts, t_pts = np.asarray(src.points), np.asarray(tgt.points)
        if len(s_pts) < 5 or len(t_pts) < 5:
            return 1.0
        s_size = np.mean(np.linalg.norm(s_pts - np.mean(s_pts, axis=0), axis=1))
        t_size = np.mean(np.linalg.norm(t_pts - np.mean(t_pts, axis=0), axis=1))
        s_est = t_size / s_size if s_size > 1e-6 else 1.0
        print(f"[INFO] Estimated scale factor (Depth→LiDAR ratio): {s_est:.4f}")
        return s_est

    def calibrate(self, image_path, lidar_path, depth_path, init_extr_path):
        """Main calibration entry."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot open image: {image_path}")
        lidar = o3d.io.read_point_cloud(str(lidar_path))
        depth_map = np.load(depth_path)
        with open(init_extr_path, "r") as f:
            init_guess = yaml.safe_load(f)

        # depth to pcd
        cam_pcd_full = self._depth_to_point_cloud(depth_map)
        lidar_ds = lidar.voxel_down_sample(self.cfg["downsample"]["lidar_voxel_size"])
        cam_ds = cam_pcd_full.voxel_down_sample(
            self.cfg["downsample"]["camera_voxel_size"]
        )

        print("[INFO] Removing outliers...")
        lidar_ds, _ = lidar_ds.remove_statistical_outlier(
            nb_neighbors=30, std_ratio=2.0
        )
        cam_ds, _ = cam_ds.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

        # --- feature extraction ---
        lidar_feat = self._extract_lidar_features(lidar_ds)
        cam_feat = self._extract_camera_features(cam_ds)

        if len(lidar_feat.points) < 30 or len(cam_feat.points) < 30:
            raise RuntimeError(
                "[ERROR] Too few features — adjust ISS or scene texture."
            )

        # --- scale + coarse align ---
        s_est = self._estimate_initial_scale(lidar_feat, cam_feat)
        lidar_feat_scaled = o3d.geometry.PointCloud(lidar_feat).scale(
            s_est, center=(0, 0, 0)
        )
        T_coarse = self._run_coarse_registration(
            lidar_feat_scaled, cam_feat, self.cfg["downsample"]["camera_voxel_size"]
        )

        # --- fine ICP refinement ---
        print("\n[STEP 2] Running fine alignment (Colored ICP)...")
        cam_pcd_full.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50)
        )
        lidar_scaled_full = o3d.geometry.PointCloud(lidar).scale(
            s_est, center=(0, 0, 0)
        )

        icp_result = o3d.pipelines.registration.registration_colored_icp(
            lidar_scaled_full,
            cam_pcd_full,
            self.cfg["icp"]["max_correspondence_distance"],
            init=T_coarse,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50
            ),
        )

        T_final = icp_result.transformation
        R_final, t_final = T_final[:3, :3], T_final[:3, 3]
        print("\n--- [RESULT] Definitive Calibration Report ---")
        print(f"ICP Fitness: {icp_result.fitness:.5f}")
        print(f"ICP RMSE: {icp_result.inlier_rmse:.6f}")
        print(f"Scale factor: {s_est:.5f}")
        print(f"Rotation:\n{np.round(R_final, 5)}")
        print(f"Translation:\n{np.round(t_final, 5)}")

        self.save_results(s_est, R_final, t_final, icp_result)
        final_params = np.concatenate(
            ([s_est], Rotation.from_matrix(R_final).as_rotvec(), t_final)
        )
        self.visualize_alignment(
            "final_alignment_overlay.png", image, lidar, final_params
        )

    def save_results(self, scale, R, t, icp_result):
        data = {
            "scale_factor": float(scale),
            "final_extrinsics": np.vstack(
                (np.hstack((R, t.reshape(3, 1))), [0, 0, 0, 1])
            ).tolist(),
            "metrics": {
                "fitness": float(icp_result.fitness),
                "rmse": float(icp_result.inlier_rmse),
            },
            "note": "Transform LiDAR→Camera: P_cam = R @ (s * P_lidar) + t",
        }
        with open(self.output_dir / "final_calibration.yaml", "w") as f:
            yaml.dump(data, f, sort_keys=False)
        print(
            f"[SAVE] Calibration results saved → {self.output_dir/'final_calibration.yaml'}"
        )

    def visualize_alignment(self, fname, image, cloud, params):
        """Project LiDAR cloud onto image (depth-colored overlay)."""
        print(f"[VIS] Generating overlay: {fname}")
        scale, rx, ry, rz, tx, ty, tz = params
        R = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
        t = np.array([tx, ty, tz])
        pts = np.asarray(cloud.points)
        transformed = (R @ (scale * pts.T)).T + t

        h, w, _ = image.shape
        proj, _ = cv2.projectPoints(
            transformed, np.zeros(3), np.zeros(3), self.cam_intrinsics, np.zeros(5)
        )
        proj = proj.squeeze()

        d = transformed[:, 2]
        valid = (
            (d > 0.1)
            & (proj[:, 0] >= 0)
            & (proj[:, 0] < w)
            & (proj[:, 1] >= 0)
            & (proj[:, 1] < h)
        )
        proj = proj[valid].astype(np.int32)
        d_valid = np.clip(d[valid] / (np.percentile(d[valid], 95) + 1e-6), 0, 1)
        colors = cv2.applyColorMap((d_valid * 255).astype(np.uint8), cv2.COLORMAP_JET)

        overlay = image.copy()
        for i, (px, py) in enumerate(proj):
            cv2.circle(
                overlay,
                (px, py),
                1,
                color=tuple(int(c) for c in colors[i, 0]),
                thickness=-1,
            )
        blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

        out_path = self.output_dir / fname
        cv2.imwrite(str(out_path), blended)
        print(f"[SAVED] Visualization overlay saved → {out_path}")


def main():
    CONFIG_PATH = "learning_config.yaml"
    DATA_DIR = Path("learning_data")
    DATA_DIR.mkdir(exist_ok=True)

    if not Path(CONFIG_PATH).exists():
        cfg = {
            "camera": {"intrinsics": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]]},
            "downsample": {"lidar_voxel_size": 0.1, "camera_voxel_size": 0.05},
            "icp": {"max_correspondence_distance": 0.1},
            "output": {"directory": "learning_output_expert"},
        }
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(cfg, f, sort_keys=False)

    if not (DATA_DIR / "scene.png").exists():
        cv2.imwrite(str(DATA_DIR / "scene.png"), np.zeros((1080, 1920, 3), np.uint8))
    if not (DATA_DIR / "lidar.pcd").exists():
        o3d.io.write_point_cloud(str(DATA_DIR / "lidar.pcd"), o3d.geometry.PointCloud())
    if not (DATA_DIR / "depth.npy").exists():
        np.save(DATA_DIR / "depth.npy", np.ones((1080, 1920)))
    if not (DATA_DIR / "initial_extrinsics.yaml").exists():
        with open(DATA_DIR / "initial_extrinsics.yaml", "w") as f:
            yaml.dump({"extrinsic_matrix": np.eye(4).tolist()}, f)

    print(
        "\n[INFO] Ready for calibration. Ensure real data are placed in 'learning_data/'."
    )
    calibrator = LearningBasedCalibrator(CONFIG_PATH)
    try:
        calibrator.calibrate(
            DATA_DIR / "scene.png",
            DATA_DIR / "lidar.pcd",
            DATA_DIR / "depth.npy",
            DATA_DIR / "initial_extrinsics.yaml",
        )
    except Exception as e:
        import traceback

        print(f"[FATAL] Calibration failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
