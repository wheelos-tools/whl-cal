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


class LidarFrame:
    def __init__(self, timestamp, points):
        self.timestamp = timestamp
        self.points = points  # open3d.geometry.PointCloud


class ImuMeasurement:
    def __init__(self, timestamp, angular_velocity, linear_acceleration):
        self.timestamp = timestamp
        self.angular_velocity = angular_velocity
        self.linear_acceleration = linear_acceleration


def load_calibration_data(config):
    """
    Load the data and organize it into a pair list of (LiDAR frames, [IMU measurement sequences]).
    """
    print("Loading and synchronizing data...")
    # 1. Read all LiDAR and IMU messages
    # 2. Iterate through the LiDAR messages
    # 3. For every two consecutive LiDAR frames (L_k, L_{k+1})
    # 4. Find all IMU messages with timestamps between [L_k.ts, L_{k+1}.ts]
    # 5. Treat (L_k, L_{k+1}, [IMU_sub_sequence]) as a motion segment

    motion_segments = []  # List of (lidar_frame_k, lidar_frame_k+1, imu_data_between)
    print(f"Found {len(motion_segments)} motion segments for calibration.")
    return motion_segments
