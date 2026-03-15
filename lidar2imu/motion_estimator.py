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
import numpy as np
from scipy.spatial.transform import Rotation


def estimate_lidar_motion(pcd_k, pcd_k1, config):
    """ICP is used to estimate the relative motion between two frames of point clouds."""
    voxel_size = config["preprocessing"]["lidar_voxel_size"]
    pcd_k_down = pcd_k.voxel_down_sample(voxel_size)
    pcd_k1_down = pcd_k1.voxel_down_sample(voxel_size)

    # Estimate normals to support point-to-plane ICP
    if config["motion_estimation"]["icp_method"] == "point_to_plane":
        pcd_k_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
        pcd_k1_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
        )
        loss = o3d.pipelines.registration.TukeyLoss(k=0.1)
        estimation_method = (
            o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        )
    else:
        estimation_method = (
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

    reg_result = o3d.pipelines.registration.registration_icp(
        pcd_k_down,
        pcd_k1_down,
        config["motion_estimation"]["icp_max_correspondence_distance"],
        np.identity(4),
        estimation_method,
    )
    return reg_result.transformation


def integrate_imu_motion(imu_data):
    """Relative rotation is calculated by integrating the angular velocity of the IMU."""
    # TODO: The integrator will be more complex (e.g., considering bias).
    delta_rotation = Rotation.identity()
    for i in range(len(imu_data) - 1):
        dt = imu_data[i + 1].timestamp - imu_data[i].timestamp
        avg_ang_vel = (
            imu_data[i].angular_velocity + imu_data[i + 1].angular_velocity
        ) / 2.0
        # Rotation vector
        rot_vec = avg_ang_vel * dt
        # Incremental rotation
        inc_rotation = Rotation.from_rotvec(rot_vec)
        # Cumulative rotation
        delta_rotation = inc_rotation * delta_rotation

    return delta_rotation.as_matrix()
