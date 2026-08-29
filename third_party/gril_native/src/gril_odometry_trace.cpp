/*
 * Replay the native GRIL LiDAR-only odometry boundary without ROS.
 * Copyright (C) 2026 whl-cal contributors.
 * GPL-2.0-only; see ../LICENSE.
 */

#include <Gril_Calib/LidarOdometry.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void expect(std::istream &input, const std::string &expected) {
    std::string actual;
    if (!(input >> actual) || actual != expected)
        throw std::runtime_error("expected token: " + expected);
}

FrontendState read_state(std::istream &input) {
    expect(input, "state");
    FrontendState state;
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            input >> state.rot_end(row, column);
    for (int index = 0; index < 3; ++index)
        input >> state.pos_end(index);
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            input >> state.offset_R_L_I(row, column);
    for (int index = 0; index < 3; ++index)
        input >> state.offset_T_L_I(index);
    for (int index = 0; index < 3; ++index)
        input >> state.vel_end(index);
    for (int index = 0; index < 3; ++index)
        input >> state.bias_g(index);
    for (int index = 0; index < 3; ++index)
        input >> state.bias_a(index);
    for (int index = 0; index < 3; ++index)
        input >> state.gravity(index);
    for (int row = 0; row < 24; ++row)
        for (int column = 0; column < 24; ++column)
            input >> state.cov(row, column);
    if (!input)
        throw std::runtime_error("invalid odometry state");
    return state;
}

void write_state(
    std::ostream &output,
    const std::string &label,
    const FrontendState &state) {
    output << label;
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            output << " " << state.rot_end(row, column);
    for (int index = 0; index < 3; ++index)
        output << " " << state.pos_end(index);
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            output << " " << state.offset_R_L_I(row, column);
    for (int index = 0; index < 3; ++index)
        output << " " << state.offset_T_L_I(index);
    for (int index = 0; index < 3; ++index)
        output << " " << state.vel_end(index);
    for (int index = 0; index < 3; ++index)
        output << " " << state.bias_g(index);
    for (int index = 0; index < 3; ++index)
        output << " " << state.bias_a(index);
    for (int index = 0; index < 3; ++index)
        output << " " << state.gravity(index);
    for (int row = 0; row < 24; ++row)
        for (int column = 0; column < 24; ++column)
            output << " " << state.cov(row, column);
    output << "\n";
}

PointCloudXYZI read_cloud(std::istream &input) {
    expect(input, "cloud");
    std::size_t count = 0;
    input >> count;
    PointCloudXYZI cloud;
    cloud.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        expect(input, "point");
        PointType point;
        input >> point.x >> point.y >> point.z
              >> point.intensity >> point.curvature
              >> point.normal_x >> point.normal_y >> point.normal_z;
        cloud.push_back(point);
    }
    if (!input)
        throw std::runtime_error("invalid odometry cloud");
    return cloud;
}

void write_cloud(
    std::ostream &output,
    const std::string &label,
    const PointCloudXYZI &cloud) {
    output << label << " " << cloud.size() << "\n";
    for (const PointType &point : cloud.points) {
        output << "point "
               << point.x << " " << point.y << " " << point.z << " "
               << point.intensity << " " << point.curvature << " "
               << point.normal_x << " " << point.normal_y << " "
               << point.normal_z << "\n";
    }
}

}  // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 3)
            throw std::runtime_error(
                "usage: gril_native_odometry_trace INPUT OUTPUT");
        std::ifstream input(argv[1]);
        std::ofstream output(argv[2]);
        if (!input || !output)
            throw std::runtime_error("could not open odometry trace");

        expect(input, "GRIL_ODOMETRY_INPUT");
        int version = 0;
        input >> version;
        if (version != 1)
            throw std::runtime_error(
                "unsupported odometry input version");
        expect(input, "config");
        LidarOdometryConfig config;
        input >> config.max_iterations
              >> config.cube_side_length
              >> config.filter_size_surf
              >> config.filter_size_map
              >> config.detection_range
              >> config.ground_covariance;
        LidarOdometryCore odometry(config);

        expect(input, "frames");
        std::size_t frame_count = 0;
        input >> frame_count;
        output << std::setprecision(17);
        output << "GRIL_ODOMETRY_TRACE 1\n";
        output << "config "
               << config.max_iterations << " "
               << config.cube_side_length << " "
               << config.filter_size_surf << " "
               << config.filter_size_map << " "
               << config.detection_range << " "
               << config.ground_covariance << "\n";
        for (std::size_t frame = 0; frame < frame_count; ++frame) {
            expect(input, "frame");
            int frame_number = 0;
            double timestamp = 0.0;
            input >> frame_number >> timestamp;
            FrontendState state = read_state(input);
            expect(input, "ground");
            LidarGroundEstimate ground;
            input >> ground.lidar_ground_rotation.w()
                  >> ground.lidar_ground_rotation.x()
                  >> ground.lidar_ground_rotation.y()
                  >> ground.lidar_ground_rotation.z()
                  >> ground.normal_lidar.x()
                  >> ground.normal_lidar.y()
                  >> ground.normal_lidar.z();
            const PointCloudXYZI cloud = read_cloud(input);
            expect(input, "end_frame");

            const LidarOdometryResult result =
                odometry.process(cloud, ground, state);
            output << "frame " << frame_number << " "
                   << timestamp << "\n";
            write_state(
                output, "propagated_state",
                result.propagated_state);
            output << "ground "
                   << ground.lidar_ground_rotation.w() << " "
                   << ground.lidar_ground_rotation.x() << " "
                   << ground.lidar_ground_rotation.y() << " "
                   << ground.lidar_ground_rotation.z() << " "
                   << ground.normal_lidar.x() << " "
                   << ground.normal_lidar.y() << " "
                   << ground.normal_lidar.z() << "\n";
            write_cloud(output, "input", cloud);
            output << "metrics "
                   << (result.map_initialized_this_frame ? 1 : 0)
                   << " " << (result.update_performed ? 1 : 0)
                   << " " << result.iterations
                   << " " << result.rematch_count
                   << " " << result.effective_feature_count
                   << " " << result.deleted_point_count
                   << " " << result.added_point_count
                   << " " << result.map_size
                   << " " << result.map_valid_points
                   << " " << result.residual_mean
                   << " " << result.delta_rotation_deg
                   << " " << result.delta_translation_cm
                   << " " << result.total_distance << "\n";
            write_state(
                output, "updated_state",
                result.updated_state);
            write_cloud(
                output, "downsampled_body",
                result.downsampled_body);
            write_cloud(
                output, "downsampled_world",
                result.downsampled_world);
            write_cloud(
                output, "effective_body",
                result.effective_body);
            write_cloud(
                output, "effective_normals",
                result.effective_normals);
            output << "correspondences "
                   << result.nearest_points.size() << "\n";
            for (std::size_t index = 0;
                 index < result.nearest_points.size();
                 ++index) {
                output << "match " << index << " "
                       << (result.selected[index] ? 1 : 0) << " "
                       << result.residuals[index] << " "
                       << result.nearest_points[index].size() << "\n";
                for (const PointType &point :
                     result.nearest_points[index]) {
                    output << "near "
                           << point.x << " " << point.y << " "
                           << point.z << " " << point.intensity << " "
                           << point.curvature << "\n";
                }
            }
            output << "end_frame\n";
        }
        expect(input, "END");
        output << "END\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
