/*
 * Deterministic synchronization and CV propagation tests.
 * Copyright (C) 2026 whl-cal contributors.
 * GPL-2.0-only; see ../LICENSE.
 */

#include <Gril_Calib/FrontendCore.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

void require_near(
    double actual,
    double expected,
    double tolerance,
    const char *message) {
    if (std::abs(actual - expected) > tolerance)
        throw std::runtime_error(message);
}

PointCloudXYZI cloud(double end_offset_ms) {
    PointCloudXYZI value;
    for (int index = 0; index < 3; ++index) {
        PointType point;
        point.x = static_cast<float>(index + 1);
        point.y = 0.0F;
        point.z = 0.0F;
        point.curvature =
            static_cast<float>(index) * end_offset_ms / 2.0F;
        value.push_back(point);
    }
    return value;
}

FrontendImuSample imu(double timestamp_s) {
    FrontendImuSample value;
    value.timestamp_s = timestamp_s;
    return value;
}

void synchronization() {
    FrontendSynchronizer synchronizer;
    FrontendMeasureGroup measure;
    synchronizer.push_lidar(cloud(100.0), 1.0);
    synchronizer.push_imu(imu(0.95));
    synchronizer.push_imu(imu(1.05));
    require(!synchronizer.try_sync(measure),
            "scan released before IMU reached its end");
    synchronizer.push_imu(imu(1.10));
    require(synchronizer.try_sync(measure),
            "scan did not release when IMU reached its end");
    require(measure.imu.size() == 3,
            "upstream stale-loop timing behavior changed");
    require(synchronizer.imu_buffer_size() == 0,
            "equal-time IMU after earlier samples must be consumed");

    synchronizer.push_lidar(cloud(100.0), 2.0);
    synchronizer.push_imu(imu(2.10));
    require(synchronizer.try_sync(measure),
            "equal-time-only IMU did not release scan");
    require(measure.imu.empty(),
            "equal-time-only IMU must remain buffered");
    require(synchronizer.imu_buffer_size() == 1,
            "equal-time-only IMU was unexpectedly consumed");
}

void propagation() {
    ConstantVelocityPropagator propagator(
        Eigen::Vector3d::Constant(0.5),
        Eigen::Vector3d::Constant(0.5));
    FrontendState state;
    state.vel_end = Eigen::Vector3d(1.0, 0.0, 0.0);
    FrontendMeasureGroup measure;
    measure.lidar_beg_time_s = 10.0;
    measure.lidar = cloud(100.0);

    const auto first = propagator.process(measure, state);
    require_near(propagator.last_dt_s(), 0.1, 1e-15,
                 "first-frame dt changed");
    require_near(propagator.time_last_scan_s(), 10.0, 1e-15,
                 "first-frame scan time was not initialized");
    require_near(state.pos_end.x(), 0.1, 1e-15,
                 "first position propagation mismatch");
    require_near(first.front().x, 1.0, 1e-6,
                 "upstream first point must remain unchanged");
    require_near(first.back().x, 3.0, 1e-6,
                 "scan-end point compensation mismatch");
    require_near(first[1].x, 1.95, 1e-6,
                 "intermediate point compensation mismatch");
    require_near(state.cov(3, 3), 1.01, 1e-12,
                 "position covariance propagation mismatch");

    measure.lidar_beg_time_s = 10.25;
    propagator.process(measure, state);
    require_near(propagator.last_dt_s(), 0.25, 1e-15,
                 "subsequent dt changed");
    require_near(state.pos_end.x(), 0.35, 1e-15,
                 "subsequent position propagation mismatch");
}

}  // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("expected one test mode");
        const std::string mode(argv[1]);
        if (mode == "synchronization")
            synchronization();
        else if (mode == "propagation")
            propagation();
        else
            throw std::runtime_error("unknown test mode");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
