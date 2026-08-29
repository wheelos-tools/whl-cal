/*
 * ROS-free synchronization and constant-velocity propagation for GRIL-Calib.
 * Original implementation: TaeYoung Kim and GRIL-Calib contributors.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License, version 2.
 * See ../../LICENSE.
 */

#ifndef GRIL_NATIVE_FRONTEND_CORE_H
#define GRIL_NATIVE_FRONTEND_CORE_H

#include "VelodynePreprocess.h"

#include <Eigen/Core>

#include <cstddef>
#include <deque>

struct FrontendImuSample {
    double timestamp_s = 0.0;
    Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d linear_acceleration = Eigen::Vector3d::Zero();
};

struct FrontendMeasureGroup {
    double lidar_beg_time_s = 0.0;
    std::size_t source_scan_index = 0;
    PointCloudXYZI lidar;
    std::deque<FrontendImuSample> imu;
};

struct FrontendState {
    Eigen::Matrix3d rot_end = Eigen::Matrix3d::Identity();
    Eigen::Vector3d pos_end = Eigen::Vector3d::Zero();
    Eigen::Matrix3d offset_R_L_I = Eigen::Matrix3d::Identity();
    Eigen::Vector3d offset_T_L_I = Eigen::Vector3d::Zero();
    Eigen::Vector3d vel_end = Eigen::Vector3d::Zero();
    Eigen::Vector3d bias_g = Eigen::Vector3d::Zero();
    Eigen::Vector3d bias_a = Eigen::Vector3d::Zero();
    Eigen::Vector3d gravity = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 24, 24> cov =
        Eigen::Matrix<double, 24, 24>::Identity();

    FrontendState();
};

class FrontendSynchronizer {
  public:
    void push_lidar(
        const PointCloudXYZI &cloud,
        double timestamp_s,
        std::size_t source_scan_index = 0);
    void push_imu(const FrontendImuSample &sample);
    bool try_sync(FrontendMeasureGroup &measure);
    void clear_lidar();
    void clear_imu();
    void clear();

    std::size_t lidar_buffer_size() const;
    std::size_t imu_buffer_size() const;
    double lidar_end_time_s() const;

  private:
    std::deque<PointCloudXYZI> lidar_buffer_;
    std::deque<double> time_buffer_;
    std::deque<std::size_t> source_scan_buffer_;
    std::deque<FrontendImuSample> imu_buffer_;
    double last_timestamp_imu_s_ = 0.0;
    double lidar_end_time_s_ = 0.0;
    bool lidar_pushed_ = false;
    FrontendMeasureGroup latched_;
};

class ConstantVelocityPropagator {
  public:
    ConstantVelocityPropagator(
        const Eigen::Vector3d &gyr_cov,
        const Eigen::Vector3d &acc_cov);

    PointCloudXYZI process(
        const FrontendMeasureGroup &measure,
        FrontendState &state);

    double last_dt_s() const;
    double time_last_scan_s() const;

  private:
    Eigen::Vector3d cov_gyr_scale_;
    Eigen::Vector3d cov_acc_scale_;
    double time_last_scan_s_ = 0.0;
    double last_dt_s_ = 0.0;
    bool first_frame_ = true;
};

#endif
