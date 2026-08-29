/*
 * ROS-free synchronization and constant-velocity propagation for GRIL-Calib.
 * Original implementation: TaeYoung Kim and GRIL-Calib contributors.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License, version 2.
 * See ../../LICENSE.
 */

#include "FrontendCore.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

Eigen::Matrix3d exp_so3(
    const Eigen::Vector3d &angular_velocity,
    double dt) {
    const double norm = angular_velocity.norm();
    const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
    if (norm <= 0.0000001)
        return identity;

    const Eigen::Vector3d axis = angular_velocity / norm;
    Eigen::Matrix3d skew;
    skew << 0.0, -axis.z(), axis.y(),
        axis.z(), 0.0, -axis.x(),
        -axis.y(), axis.x(), 0.0;
    const double angle = norm * dt;
    return identity + std::sin(angle) * skew +
           (1.0 - std::cos(angle)) * skew * skew;
}

bool point_time_less(const PointType &left, const PointType &right) {
    return left.curvature < right.curvature;
}

}  // namespace

FrontendState::FrontendState() {
    cov.block<9, 9>(15, 15) =
        Eigen::Matrix<double, 9, 9>::Identity() * 0.00001;
}

void FrontendSynchronizer::push_lidar(
    const PointCloudXYZI &cloud,
    double timestamp_s,
    std::size_t source_scan_index) {
    lidar_buffer_.push_back(cloud);
    time_buffer_.push_back(timestamp_s);
    source_scan_buffer_.push_back(source_scan_index);
}

void FrontendSynchronizer::push_imu(const FrontendImuSample &sample) {
    imu_buffer_.push_back(sample);
    last_timestamp_imu_s_ = sample.timestamp_s;
}

bool FrontendSynchronizer::try_sync(FrontendMeasureGroup &measure) {
    if (lidar_buffer_.empty() || imu_buffer_.empty())
        return false;

    if (!lidar_pushed_) {
        latched_.lidar = lidar_buffer_.front();
        if (latched_.lidar.size() <= 1) {
            lidar_buffer_.pop_front();
            time_buffer_.pop_front();
            source_scan_buffer_.pop_front();
            return false;
        }
        latched_.lidar_beg_time_s = time_buffer_.front();
        latched_.source_scan_index = source_scan_buffer_.front();
        lidar_end_time_s_ =
            latched_.lidar_beg_time_s +
            latched_.lidar.back().curvature / 1000.0;
        lidar_pushed_ = true;
    }

    if (last_timestamp_imu_s_ < lidar_end_time_s_)
        return false;

    double imu_time = imu_buffer_.front().timestamp_s;
    latched_.imu.clear();
    while (!imu_buffer_.empty() && imu_time < lidar_end_time_s_) {
        imu_time = imu_buffer_.front().timestamp_s;
        if (imu_time > lidar_end_time_s_)
            break;
        latched_.imu.push_back(imu_buffer_.front());
        imu_buffer_.pop_front();
    }
    lidar_buffer_.pop_front();
    time_buffer_.pop_front();
    source_scan_buffer_.pop_front();
    lidar_pushed_ = false;
    measure = latched_;
    return true;
}

void FrontendSynchronizer::clear_lidar() {
    lidar_buffer_.clear();
    time_buffer_.clear();
    source_scan_buffer_.clear();
    lidar_pushed_ = false;
}

void FrontendSynchronizer::clear_imu() {
    imu_buffer_.clear();
    last_timestamp_imu_s_ = 0.0;
}

void FrontendSynchronizer::clear() {
    clear_lidar();
    clear_imu();
    lidar_end_time_s_ = 0.0;
    latched_ = FrontendMeasureGroup();
}

std::size_t FrontendSynchronizer::lidar_buffer_size() const {
    return lidar_buffer_.size();
}

std::size_t FrontendSynchronizer::imu_buffer_size() const {
    return imu_buffer_.size();
}

double FrontendSynchronizer::lidar_end_time_s() const {
    return lidar_end_time_s_;
}

ConstantVelocityPropagator::ConstantVelocityPropagator(
    const Eigen::Vector3d &gyr_cov,
    const Eigen::Vector3d &acc_cov)
    : cov_gyr_scale_(gyr_cov), cov_acc_scale_(acc_cov) {}

PointCloudXYZI ConstantVelocityPropagator::process(
    const FrontendMeasureGroup &measure,
    FrontendState &state) {
    if (measure.lidar.empty())
        throw std::invalid_argument("GRIL propagation requires a nonempty cloud");

    PointCloudXYZI output = measure.lidar;
    std::sort(
        output.points.begin(),
        output.points.end(),
        point_time_less);
    const double end_offset_s = output.back().curvature / 1000.0;

    if (first_frame_) {
        last_dt_s_ = 0.1;
        time_last_scan_s_ = measure.lidar_beg_time_s;
        first_frame_ = false;
    } else {
        last_dt_s_ = measure.lidar_beg_time_s - time_last_scan_s_;
        time_last_scan_s_ = measure.lidar_beg_time_s;
    }

    Eigen::Matrix<double, 24, 24> transition =
        Eigen::Matrix<double, 24, 24>::Identity();
    Eigen::Matrix<double, 24, 24> process_noise =
        Eigen::Matrix<double, 24, 24>::Zero();
    const Eigen::Matrix3d rotation_step =
        exp_so3(state.bias_g, last_dt_s_);
    transition.block<3, 3>(0, 0) =
        exp_so3(state.bias_g, -last_dt_s_);
    transition.block<3, 3>(0, 15) =
        Eigen::Matrix3d::Identity() * last_dt_s_;
    transition.block<3, 3>(3, 12) =
        Eigen::Matrix3d::Identity() * last_dt_s_;
    process_noise.block<3, 3>(15, 15).diagonal() =
        cov_gyr_scale_ * last_dt_s_ * last_dt_s_;
    process_noise.block<3, 3>(12, 12).diagonal() =
        cov_acc_scale_ * last_dt_s_ * last_dt_s_;

    state.cov =
        transition * state.cov * transition.transpose() +
        process_noise;
    state.rot_end = state.rot_end * rotation_step;
    state.pos_end += state.vel_end * last_dt_s_;

    auto point = output.points.end() - 1;
    for (; point != output.points.begin(); --point) {
        const double point_dt_s =
            end_offset_s - point->curvature / 1000.0;
        const Eigen::Matrix3d point_rotation =
            exp_so3(state.bias_g, -point_dt_s);
        const Eigen::Vector3d input(point->x, point->y, point->z);
        const Eigen::Vector3d translation =
            -state.rot_end.transpose() * state.vel_end * point_dt_s;
        const Eigen::Vector3d compensated =
            point_rotation * input + translation;
        point->x = compensated.x();
        point->y = compensated.y();
        point->z = compensated.z();
    }
    return output;
}

double ConstantVelocityPropagator::last_dt_s() const {
    return last_dt_s_;
}

double ConstantVelocityPropagator::time_last_scan_s() const {
    return time_last_scan_s_;
}
