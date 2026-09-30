/*
 * Focused tests for the pinned GRIL LiDAR-only odometry layer.
 * Copyright (C) 2026 whl-cal contributors.
 * GPL-2.0-only; see ../LICENSE.
 */

#include <Gril_Calib/LidarOdometry.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

PointCloudXYZI plane_cloud(float z) {
    PointCloudXYZI cloud;
    for (int x = -4; x <= 4; ++x) {
        for (int y = -4; y <= 4; ++y) {
            PointType point;
            point.x = static_cast<float>(x) * 0.6F;
            point.y = static_cast<float>(y) * 0.6F;
            point.z = z;
            point.intensity =
                static_cast<float>((x + 4) * 9 + y + 4);
            cloud.push_back(point);
        }
    }
    return cloud;
}

std::uint64_t mix(std::uint64_t hash, std::uint64_t field) {
    return (hash ^ field) * UINT64_C(1099511628211);
}

std::uint32_t float_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::uint64_t cloud_identity(const PointCloudXYZI &cloud) {
    std::uint64_t hash = UINT64_C(1469598103934665603);
    hash = mix(hash, cloud.size());
    for (const PointType &point : cloud) {
        hash = mix(hash, float_bits(point.x));
        hash = mix(hash, float_bits(point.y));
        hash = mix(hash, float_bits(point.z));
        hash = mix(hash, float_bits(point.intensity));
        hash = mix(hash, float_bits(point.curvature));
    }
    return hash;
}

PointCloudXYZI voxel_order_cloud() {
    PointCloudXYZI cloud;
    for (int voxel = 0; voxel < 8; ++voxel) {
        for (int index = 0; index < 33; ++index) {
            PointType point;
            point.x = voxel * 0.5F + 0.01F + (index % 11) * 0.013F;
            point.y = 0.01F + (index % 7) * 0.019F;
            point.z = 0.02F + (index % 5) * 0.023F;
            point.intensity = (index % 9) * 1.3F + voxel * 0.2F;
            point.curvature = (index % 13) * 0.37F + voxel * 0.11F;
            cloud.push_back(point);
        }
    }
    return cloud;
}

void pinned_config() {
    const LidarOdometryConfig config;
    require(config.max_iterations == 5,
            "launch max_iteration was not pinned to 5");
    require(config.cube_side_length == 1000.0,
            "launch cube_side_length was not pinned to 1000");
}

void map_initialization() {
    LidarOdometryCore odometry;
    FrontendState state;
    const FrontendState before = state;
    const LidarOdometryResult result =
        odometry.process(
            plane_cloud(1.0F), LidarGroundEstimate(), state);
    require(result.map_initialized_this_frame,
            "first frame did not initialize ikd-tree");
    require(!result.update_performed,
            "first frame unexpectedly entered EKF update");
    require(result.map_size > 5,
            "initialized ikd-tree contains too few points");
    require((state.pos_end - before.pos_end).norm() == 0.0,
            "map initialization changed the propagated state");
}

void ekf_update() {
    LidarOdometryCore odometry;
    FrontendState state;
    odometry.process(
        plane_cloud(1.0F), LidarGroundEstimate(), state);
    const LidarOdometryResult result =
        odometry.process(
            plane_cloud(1.0F), LidarGroundEstimate(), state);
    require(result.update_performed,
            "second frame did not enter EKF update");
    require(result.iterations > 0 && result.iterations <= 5,
            "EKF iteration order changed");
    require(result.effective_feature_count > 0,
            "synthetic plane produced no correspondences");
    require(result.rematch_count >= 1,
            "upstream rematch schedule did not execute");
    require(std::isfinite(state.pos_end.z()),
            "EKF update produced a non-finite pose");
    require(result.map_size >= result.map_valid_points,
            "ikd-tree map counters are inconsistent");
}

void residual_mean() {
    LidarOdometryCore odometry;
    FrontendState state;
    odometry.process(plane_cloud(1.0F), LidarGroundEstimate(), state);
    const LidarOdometryResult result =
        odometry.process(
            plane_cloud(1.02F), LidarGroundEstimate(), state);
    require(result.effective_feature_count > 0,
            "shifted plane produced no correspondences");
    double sum = 0.0;
    int count = 0;
    for (std::size_t index = 0; index < result.selected.size(); ++index) {
        if (result.selected[index]) {
            sum += result.residuals[index];
            ++count;
        }
    }
    require(count == result.effective_feature_count,
            "residual selection does not match correspondence count");
    require(std::isfinite(result.residual_mean) &&
                result.residual_mean > 0.0 &&
                std::abs(result.residual_mean - sum / count) < 1e-6,
            "residual mean does not match accepted point-to-plane distances");

    LidarOdometryCore empty_matches;
    FrontendState other_state;
    empty_matches.process(
        plane_cloud(1.0F), LidarGroundEstimate(), other_state);
    const LidarOdometryResult no_match =
        empty_matches.process(
            plane_cloud(100.0F), LidarGroundEstimate(), other_state);
    require(no_match.effective_feature_count == 0 &&
                std::isnan(no_match.residual_mean),
            "no correspondences must report an undefined residual");
}

void determinism() {
    LidarOdometryCore left;
    LidarOdometryCore right;
    FrontendState left_state;
    FrontendState right_state;
    const PointCloudXYZI cloud = plane_cloud(1.0F);
    left.process(cloud, LidarGroundEstimate(), left_state);
    right.process(cloud, LidarGroundEstimate(), right_state);
    const LidarOdometryResult left_result =
        left.process(cloud, LidarGroundEstimate(), left_state);
    const LidarOdometryResult right_result =
        right.process(cloud, LidarGroundEstimate(), right_state);
    require(
        (left_state.rot_end - right_state.rot_end).cwiseAbs().maxCoeff() ==
            0.0,
        "repeated odometry rotation differs");
    require((left_state.pos_end - right_state.pos_end).norm() == 0.0,
            "repeated odometry translation differs");
    require(
        (left_state.cov - right_state.cov).cwiseAbs().maxCoeff() == 0.0,
        "repeated odometry covariance differs");
    require(
        left_result.effective_feature_count ==
            right_result.effective_feature_count,
        "repeated correspondence count differs");
    require(left_result.map_size == right_result.map_size,
            "repeated map size differs");
}

void pcl_1_10_voxel_order() {
    LidarOdometryCore odometry;
    FrontendState state;
    const LidarOdometryResult result =
        odometry.process(
            voxel_order_cloud(), LidarGroundEstimate(), state);
    const std::uint64_t identity =
        cloud_identity(result.downsampled_body);
    if (identity != UINT64_C(5027915533257193198))
        throw std::runtime_error(
            "unexpected PCL 1.10 voxel identity: " +
            std::to_string(identity));
}

}  // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 2)
            throw std::runtime_error("expected one test mode");
        const std::string mode(argv[1]);
        if (mode == "pinned_config")
            pinned_config();
        else if (mode == "map_initialization")
            map_initialization();
        else if (mode == "ekf_update")
            ekf_update();
        else if (mode == "residual_mean")
            residual_mean();
        else if (mode == "determinism")
            determinism();
        else if (mode == "pcl_1_10_voxel_order")
            pcl_1_10_voxel_order();
        else
            throw std::runtime_error("unknown test mode");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
