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

# Created Date: 2025-11-14
# Author: daohu527


import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def to_so3(R):
    """Convert the rotation matrix into a Lie algebra so(3) vector."""
    return Rotation.from_matrix(R).as_rotvec()


def calibration_residual(params, imu_motions, lidar_motions):
    """
    The residual function is the core of the optimization.

    params: [roll, pitch, yaw, x, y, z] - Current extrinsic parameter estimates
    """
    rpy_deg = params[:3]
    # Note: SciPy's Rotation.from_euler uses 'xyz' order and radians.
    R_L_I = Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()

    residuals = []
    for R_imu, T_lidar in zip(imu_motions, lidar_motions):
        R_lidar = T_lidar[:3, :3]

        # AX = XB -> AX(XB)^-1 = I

        # We minimize the difference between the residual rotation matrix and the identity matrix
        residual_R = R_imu @ R_L_I @ np.linalg.inv(R_lidar) @ np.linalg.inv(R_L_I)

        # Convert the residual rotation matrix to angle-axis form; its magnitude (angle) represents the error.
        residual_angle_rad = np.linalg.norm(to_so3(residual_R))
        residuals.append(residual_angle_rad)

    return np.array(residuals)


def solve_extrinsics(imu_motions, lidar_motions, config):
    """Solve for the external parameters."""
    print("Solving for extrinsics...")
    initial_guess_rpy = config["calibration"]["initial_guess"]["rpy"]
    initial_guess_xyz = config["calibration"]["initial_guess"]["xyz"]
    initial_params = np.array(initial_guess_rpy)  # Simplified to only 3-DOF rotation

    # **Important**: In this classic hand-eye calibration model A_X=X_B, we primarily need to solve for the rotation R.
    # The translation t requires different constraints (such as gravity alignment or point-to-surface constraints) to solve.
    # Here we will first solve for the rotation precisely.

    result = least_squares(
        calibration_residual,
        initial_params,
        args=(imu_motions, lidar_motions),
        verbose=2,
    )

    final_rpy_deg = result.x
    final_R = Rotation.from_euler("xyz", final_rpy_deg, degrees=True).as_matrix()

    # TODO (daohu527): The solution logic for translation t should be added.
    # t_L_I = solve_translation(final_R, imu_motions, lidar_motions)
    final_t = np.array(initial_guess_xyz)  # Use initial values ​​for now

    # Construct the final 4x4 transformation matrix
    T_L_I = np.identity(4)
    T_L_I[:3, :3] = final_R
    T_L_I[:3, 3] = final_t

    return T_L_I, result
