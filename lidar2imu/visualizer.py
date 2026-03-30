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


import open3d as o3d
import matplotlib.pyplot as plt
from src import data_loader, motion_estimator, calibrator, validator, visualizer
import yaml


def plot_residuals(initial_residuals, final_residuals):
    plt.figure()
    plt.hist(initial_residuals, bins=50, alpha=0.5, label="Initial Residuals")
    plt.hist(final_residuals, bins=50, alpha=0.5, label="Final Residuals")
    plt.xlabel("Residual (radians)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Calibration Optimization Residuals")
    plt.savefig("results/residual_plot.png")
    plt.show()


def visualize_stitching(pcd):
    print("Visualizing stitched point cloud. Close the window to exit.")
    o3d.visualization.draw_geometries([pcd])


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 1. Loading data
    motion_segments = data_loader.load_calibration_data(config)

    # 2. Estimate the relative motion of each segment.
    imu_motions = []
    lidar_motions = []
    for frame_k, frame_k1, imu_seq in motion_segments:
        T_lidar = motion_estimator.estimate_lidar_motion(
            frame_k.points, frame_k1.points, config
        )
        R_imu = motion_estimator.integrate_imu_motion(imu_seq)
        lidar_motions.append(T_lidar)
        imu_motions.append(R_imu)

    # 3. Solving for extrinsic parameters
    T_L_I, opt_result = calibrator.solve_extrinsics(imu_motions, lidar_motions, config)

    print("\n----- Calibration Finished -----")
    print("Final Extrinsics (T_Lidar_to_IMU):")
    print(T_L_I)

    # 4. Validation and Quantification
    rms_error = validator.calculate_reprojection_error(
        T_L_I, imu_motions, lidar_motions
    )
    print(f"\nFinal RMS Reprojection Error: {rms_error:.6f} radians")

    # 5. Visualization
    initial_residuals = (
        opt_result.fun
    )  # The initial residuals are calculated before optimization.
    final_residuals = calibrator.calibration_residual(
        opt_result.x, imu_motions, lidar_motions
    )
    visualizer.plot_residuals(initial_residuals, final_residuals)

    # Visual verification by splicing point clouds
    all_lidar_frames = [seg[0] for seg in motion_segments]
    all_imu_sequences = [seg[2] for seg in motion_segments]
    stitched_pcd = validator.stitch_point_clouds(
        all_lidar_frames,
        all_imu_sequences,
        T_L_I,
        config["validation"]["stitching_num_frames"],
    )
    o3d.io.write_point_cloud("results/stitched_cloud.pcd", stitched_pcd)
    visualizer.visualize_stitching(stitched_pcd)


if __name__ == "__main__":
    main()
