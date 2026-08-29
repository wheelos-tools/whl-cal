/*
 * GRIL batch trace contract tests.
 * Copyright (C) 2026 whl-cal contributors.
 * GPL-2.0-only; see ../LICENSE.
 */

#include <Gril_Calib/BatchTrace.h>

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

BatchTrace sample_trace() {
    BatchTrace trace;
    trace.orig_odom_freq = 10;
    trace.cut_frame_num = 2;
    trace.timediff_imu_wrt_lidar = -0.012345678901234;
    trace.move_start_time = 42.25;
    for (int index = 0; index < 2; index++) {
        CalibState imu;
        imu.timeStamp = 40.0 + index * 0.01;
        imu.rot_end =
            Eigen::AngleAxisd(0.01 * index, V3D::UnitZ()).matrix();
        imu.pos_end = V3D(index, index + 1, index + 2);
        imu.ang_vel = V3D(0.1, 0.2, 0.3 + index);
        imu.linear_vel = V3D(1.1, 1.2, 1.3);
        imu.ang_acc = V3D(2.1, 2.2, 2.3);
        imu.linear_acc = V3D(0.0, 0.0, 9.81 + index);
        trace.normalized_imu_states.push_back(imu);

        CalibState lidar(imu);
        lidar.timeStamp += 0.001;
        trace.lidar_states.push_back(lidar);

        GroundConstraint ground;
        ground.lidar_wrt_ground =
            QD(Eigen::AngleAxisd(0.02 * index, V3D::UnitX()));
        ground.imu_wrt_ground =
            QD(Eigen::AngleAxisd(0.03 * index, V3D::UnitY()));
        ground.normal_lidar = V3D(0.0, 0.0, 1.0);
        ground.distance_lidar = 1.25 + index;
        trace.ground_constraints.push_back(ground);
    }
    return trace;
}

void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

void round_trip() {
    const BatchTrace original = sample_trace();
    std::ostringstream output;
    write_batch_trace(output, original);
    std::istringstream input(output.str());
    const BatchTrace parsed = read_batch_trace(input);

    require(parsed.orig_odom_freq == original.orig_odom_freq,
            "orig_odom_freq mismatch");
    require(parsed.cut_frame_num == original.cut_frame_num,
            "cut_frame_num mismatch");
    require(parsed.timediff_imu_wrt_lidar ==
                original.timediff_imu_wrt_lidar,
            "time diff mismatch");
    require(parsed.move_start_time == original.move_start_time,
            "move_start_time mismatch");
    require(parsed.normalized_imu_states.size() == 2,
            "IMU count mismatch");
    require(parsed.lidar_states.size() == 2,
            "LiDAR count mismatch");
    require(parsed.ground_constraints.size() == 2,
            "ground count mismatch");
    require(
        parsed.normalized_imu_states[1].linear_acc.isApprox(
            original.normalized_imu_states[1].linear_acc, 0.0),
        "IMU state mismatch");
    require(
        parsed.lidar_states[1].rot_end.isApprox(
            original.lidar_states[1].rot_end, 0.0),
        "LiDAR state mismatch");
    require(
        parsed.ground_constraints[1].lidar_wrt_ground.coeffs().isApprox(
            original.ground_constraints[1]
                .lidar_wrt_ground.coeffs(), 0.0),
        "ground quaternion mismatch");
}

void rejection() {
    {
        std::istringstream malformed(
            "GRIL_BATCH_TRACE 2\n");
        bool rejected = false;
        try {
            (void)read_batch_trace(malformed);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        require(rejected, "unsupported version was accepted");
    }
    {
        BatchTrace trace = sample_trace();
        trace.ground_constraints.pop_back();
        std::ostringstream output;
        write_batch_trace(output, trace);
        std::istringstream input(output.str());
        bool rejected = false;
        try {
            (void)read_batch_trace(input);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        require(rejected, "mismatched paired counts were accepted");
    }
    {
        BatchTrace trace = sample_trace();
        std::string reason;
        require(!validate_batch_trace_for_calibration(trace, &reason),
                "insufficient trace was accepted for calibration");
        require(!reason.empty(), "insufficient rejection lacks reason");
    }
}

}  // namespace

int main(int argc, char **argv) {
    try {
        require(argc == 2, "expected one test mode");
        const std::string mode(argv[1]);
        if (mode == "round_trip")
            round_trip();
        else if (mode == "rejection")
            rejection();
        else
            throw std::runtime_error("unknown test mode");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
