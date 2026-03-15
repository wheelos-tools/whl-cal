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


import numpy as np


def calculate_reprojection_error(T_L_I, imu_motions, lidar_motions):
    """Calculate the final reprojection error (RMS)."""
    R_L_I = T_L_I[:3, :3]
    residuals = []
    for R_imu, T_lidar in zip(imu_motions, lidar_motions):
        R_lidar = T_lidar[:3, :3]
        residual_R = R_imu @ R_L_I @ np.linalg.inv(R_lidar) @ np.linalg.inv(R_L_I)
        residual_angle_rad = np.linalg.norm(to_so3(residual_R))
        residuals.append(residual_angle_rad)

    rms_error_rad = np.sqrt(np.mean(np.square(residuals)))
    return rms_error_rad


def stitch_point_clouds(lidar_frames, imu_data_list, T_L_I, num_frames):
    """
    Point clouds are stitched together using IMU motion and calibration extrinsic parameters for visualization verification.
    """
    stitched_pcd = o3d.geometry.PointCloud()

    # Transform all point clouds to the coordinate system of the first frame of LiDAR.
    T_world_L0 = np.identity(4)

    current_T_world_I = np.identity(4)

    for i in range(min(num_frames, len(lidar_frames) - 1)):
        # 1. Calculate the motion of the IMU from 0 to k
        R_I0_Ik = integrate_imu_motion(
            imu_data_list[i]
        )  # Simplified: It should be IMU integration starting from 0.

        # 2. Calculate the transformation from the LiDAR k-system to the world (L0) system.
        # T_L0_Lk = T_L0_I0 * T_I0_Ik * T_Ik_Lk
        # T_L0_Lk = (T_I_L) * T_I0_Ik * (T_I_L)^-1
        T_I_L = T_L_I
        T_L_I_inv = np.linalg.inv(T_I_L)

        T_I0_Ik_4x4 = np.identity(4)
        T_I0_Ik_4x4[:3, :3] = R_I0_Ik

        T_L0_Lk = T_L_I_inv @ T_I0_Ik_4x4 @ T_I_L

        # 3. Transform the current point cloud and stitch it to the global point cloud
        pcd_k = lidar_frames[i]
        pcd_k.transform(T_L0_Lk)
        stitched_pcd += pcd_k

    return stitched_pcd.voxel_down_sample(0.05)
