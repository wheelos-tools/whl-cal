/*
 * Complete ROS-free GRIL frontend and live calibration execution.
 * Copyright (C) 2026 whl-cal contributors.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License, version 2.
 * See ../../LICENSE.
 */

#ifndef GRIL_NATIVE_FULL_FRONTEND_H
#define GRIL_NATIVE_FULL_FRONTEND_H

#include "LidarOdometry.h"

#include <string>
#include <vector>

struct FullPatchworkConfig {
  int num_iter = 3;
  int num_lpr = 20;
  int num_min_pts = 10;
  int max_flatness_storage = 1000;
  int max_elevation_storage = 1000;
  double sensor_height = 1.723;
  double th_seeds = 0.4;
  double th_dist = 0.3;
  double th_seeds_v = 0.4;
  double th_dist_v = 0.3;
  double max_range = 80.0;
  double min_range = 2.7;
  double uprightness_thr = 0.5;
  double adaptive_seed_selection_margin = -1.1;
  double rnr_ver_angle_thr = -15.0;
  double rnr_intensity_thr = 0.2;
  bool verbose = false;
  bool enable_rnr = true;
  bool enable_rvpf = true;
  bool enable_tgr = true;
  std::vector<int> num_sectors_each_zone;
  std::vector<int> num_rings_each_zone;
  std::vector<double> elevation_thresholds;
  std::vector<double> flatness_thresholds;
};

struct FullFrontendConfig {
  int lidar_type = 2;
  int scan_line = 16;
  double blind = 1.0;
  int point_filter_num = 2;
  bool feature_extract_enabled = false;
  bool cut_frame = true;
  int cut_frame_num = 1;

  int orig_odom_freq = 10;
  double mean_acc_norm = 9.81;
  double data_accum_length = 300.0;
  double x_accumulate = 0.1;
  double y_accumulate = 0.1;
  double z_accumulate = 0.1;
  double svd_threshold = 0.01;
  double imu_sensor_height = 0.1;
  double trans_IL_x = 0.0;
  double trans_IL_y = 0.0;
  double trans_IL_z = 0.0;
  double bound_th = 0.1;
  bool set_boundary = false;
  bool verbose = false;
  double gyro_factor = 1.0;
  double acc_factor = 1.0;
  double ground_factor = 1.0;

  LidarOdometryConfig odometry;
  Eigen::Vector3d gyr_cov = Eigen::Vector3d::Constant(0.1);
  Eigen::Vector3d acc_cov = Eigen::Vector3d::Constant(0.1);
  FullPatchworkConfig patchwork;
  double configured_time_lag_s = 0.0;
};

enum class ForwardGapPolicy {
  GoldenEquivalence,
  Reset,
};

struct FullFrontendRunConfig {
  std::string input_path;
  std::string config_path;
  std::string result_path;
  std::string trace_path;
  std::string batch_trace_path;
  std::string batch_executable_path;
  std::string batch_config_path;
  ForwardGapPolicy forward_gap_policy = ForwardGapPolicy::GoldenEquivalence;
  double forward_gap_s = 0.0;
};

class HardTimeCompensator {
public:
  explicit HardTimeCompensator(double configured_time_lag_s = 0.0);

  bool lidar_rolled_back(double raw_timestamp_s) const;
  bool observe_lidar(double raw_timestamp_s, bool imu_queue_nonempty);
  double compensate_imu(double raw_timestamp_s) const;
  bool imu_rolled_back(double compensated_timestamp_s) const;
  void observe_imu(double compensated_timestamp_s);

  double hard_offset_s() const;
  bool hard_offset_locked() const;
  double last_lidar_timestamp_s() const;
  double last_imu_timestamp_s() const;

private:
  double configured_time_lag_s_ = 0.0;
  double hard_offset_s_ = 0.0;
  double last_lidar_timestamp_s_ = 0.0;
  double last_imu_timestamp_s_ = 0.0;
  bool hard_offset_locked_ = false;
};

FullFrontendConfig read_full_frontend_config(const std::string &path);
void run_full_frontend(const FullFrontendRunConfig &config);

#endif
