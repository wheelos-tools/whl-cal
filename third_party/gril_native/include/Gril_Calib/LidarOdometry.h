/*
 * ROS-free LiDAR-only odometry adapted from GRIL-Calib.
 * Original implementation: TaeYoung Kim and GRIL-Calib contributors.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License, version 2.
 * See ../../LICENSE.
 */

#ifndef GRIL_NATIVE_LIDAR_ODOMETRY_H
#define GRIL_NATIVE_LIDAR_ODOMETRY_H

#include "FrontendCore.h"

#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <memory>
#include <vector>

struct LidarOdometryConfig {
  int max_iterations = 5;
  double cube_side_length = 1000.0;
  double filter_size_surf = 0.5;
  double filter_size_map = 0.5;
  float detection_range = 100.0F;
  double ground_covariance = 100.0;
};

struct LidarGroundEstimate {
  Eigen::Quaterniond lidar_ground_rotation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d normal_lidar = Eigen::Vector3d(0.0, 0.0, 1.0);
};

using LidarNeighborSet =
    std::vector<PointType, Eigen::aligned_allocator<PointType>>;

struct LidarOdometryResult {
  bool map_initialized_this_frame = false;
  bool update_performed = false;
  int iterations = 0;
  int rematch_count = 0;
  int effective_feature_count = 0;
  int deleted_point_count = 0;
  int added_point_count = 0;
  int map_size = 0;
  int map_valid_points = 0;
  double residual_mean = 0.05;
  double delta_rotation_deg = 0.0;
  double delta_translation_cm = 0.0;
  double total_distance = 0.0;
  FrontendState propagated_state;
  FrontendState updated_state;
  PointCloudXYZI downsampled_body;
  PointCloudXYZI downsampled_world;
  PointCloudXYZI effective_body;
  PointCloudXYZI effective_normals;
  std::vector<LidarNeighborSet> nearest_points;
  std::vector<bool> selected;
  std::vector<float> residuals;
};

class LidarOdometryCore {
public:
  explicit LidarOdometryCore(
      const LidarOdometryConfig &config = LidarOdometryConfig());
  ~LidarOdometryCore();

  LidarOdometryCore(const LidarOdometryCore &) = delete;
  LidarOdometryCore &operator=(const LidarOdometryCore &) = delete;

  LidarOdometryResult process(const PointCloudXYZI &undistorted,
                              const LidarGroundEstimate &ground,
                              FrontendState &state);

  const LidarOdometryConfig &config() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

#endif
