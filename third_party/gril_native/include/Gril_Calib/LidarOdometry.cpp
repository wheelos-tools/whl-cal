/*
 * ROS-free LiDAR-only odometry adapted from GRIL-Calib.
 * Original implementation: TaeYoung Kim and GRIL-Calib contributors.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License, version 2.
 * See ../../LICENSE.
 */

#include "LidarOdometry.h"

#include <ikd-Tree/ikd_Tree.h>

#include <Eigen/Dense>
#include <pcl/common/centroid.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace {

constexpr int kStateDimension = 24;
constexpr int kMatchPointCount = 5;
constexpr float kMovementThreshold = 1.5F;

struct LegacyVoxelIndex {
    unsigned int voxel;
    unsigned int point;

    bool operator<(const LegacyVoxelIndex &other) const {
        return voxel < other.voxel;
    }
};

bool point_xyz_finite(const PointType &point) {
    return std::isfinite(point.x) &&
           std::isfinite(point.y) &&
           std::isfinite(point.z);
}

PointCloudXYZI legacy_voxel_grid(
    const PointCloudXYZI &input,
    float leaf_size) {
    PointCloudXYZI output;
    if (input.empty())
        return output;

    Eigen::Array4f minimum =
        Eigen::Array4f::Constant(std::numeric_limits<float>::max());
    Eigen::Array4f maximum =
        Eigen::Array4f::Constant(std::numeric_limits<float>::lowest());
    for (const PointType &point : input) {
        if (!input.is_dense && !point_xyz_finite(point))
            continue;
        const Eigen::Array4f coordinates = point.getArray4fMap();
        minimum = minimum.min(coordinates);
        maximum = maximum.max(coordinates);
    }

    const Eigen::Array4f inverse_leaf(
        1.0F / leaf_size,
        1.0F / leaf_size,
        1.0F / leaf_size,
        1.0F);
    const Eigen::Vector4i minimum_bin =
        (minimum * inverse_leaf).floor().cast<int>();
    const Eigen::Vector4i maximum_bin =
        (maximum * inverse_leaf).floor().cast<int>();
    const Eigen::Vector4i divisions =
        maximum_bin - minimum_bin + Eigen::Vector4i::Ones();
    const std::int64_t voxel_count =
        static_cast<std::int64_t>(divisions[0]) * divisions[1] *
        divisions[2];
    if (voxel_count > std::numeric_limits<std::int32_t>::max())
        return input;
    const Eigen::Vector4i multipliers(
        1, divisions[0], divisions[0] * divisions[1], 0);

    std::vector<LegacyVoxelIndex> indices;
    indices.reserve(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        const PointType &point = input[index];
        if (!input.is_dense && !point_xyz_finite(point))
            continue;
        const int x = static_cast<int>(
            std::floor(point.x * inverse_leaf[0]) -
            static_cast<float>(minimum_bin[0]));
        const int y = static_cast<int>(
            std::floor(point.y * inverse_leaf[1]) -
            static_cast<float>(minimum_bin[1]));
        const int z = static_cast<int>(
            std::floor(point.z * inverse_leaf[2]) -
            static_cast<float>(minimum_bin[2]));
        indices.push_back({
            static_cast<unsigned int>(
                x * multipliers[0] + y * multipliers[1] +
                z * multipliers[2]),
            static_cast<unsigned int>(index),
        });
    }
    std::sort(indices.begin(), indices.end());

    std::vector<std::pair<std::size_t, std::size_t>> groups;
    groups.reserve(indices.size());
    std::size_t first = 0;
    while (first < indices.size()) {
        std::size_t last = first + 1;
        while (last < indices.size() &&
               indices[last].voxel == indices[first].voxel)
            ++last;
        groups.emplace_back(first, last);
        first = last;
    }

    output.resize(groups.size());
    for (std::size_t group = 0; group < groups.size(); ++group) {
        pcl::CentroidPoint<PointType> centroid;
        for (std::size_t index = groups[group].first;
             index < groups[group].second;
             ++index)
            centroid.add(input[indices[index].point]);
        centroid.get(output[group]);
    }
    return output;
}

Eigen::Matrix3d exp_so3(
    double value_1,
    double value_2,
    double value_3) {
    const double norm =
        std::sqrt(value_1 * value_1 + value_2 * value_2 +
                  value_3 * value_3);
    const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
    if (norm <= 0.00001)
        return identity;
    const double axis[3] = {
        value_1 / norm,
        value_2 / norm,
        value_3 / norm,
    };
    Eigen::Matrix3d skew;
    skew << 0.0, -axis[2], axis[1],
        axis[2], 0.0, -axis[0],
        -axis[1], axis[0], 0.0;
    return identity + std::sin(norm) * skew +
           (1.0 - std::cos(norm)) * skew * skew;
}

Eigen::Vector3d log_so3(const Eigen::Matrix3d &rotation) {
    const double theta =
        rotation.trace() > 3.0 - 1e-6
            ? 0.0
            : std::acos(0.5 * (rotation.trace() - 1.0));
    const Eigen::Vector3d skew(
        rotation(2, 1) - rotation(1, 2),
        rotation(0, 2) - rotation(2, 0),
        rotation(1, 0) - rotation(0, 1));
    return std::abs(theta) < 0.001
               ? 0.5 * skew
               : 0.5 * theta / std::sin(theta) * skew;
}

Eigen::Matrix<double, kStateDimension, 1> state_difference(
    const FrontendState &left,
    const FrontendState &right) {
    Eigen::Matrix<double, kStateDimension, 1> difference;
    difference.block<3, 1>(0, 0) =
        log_so3(right.rot_end.transpose() * left.rot_end);
    difference.block<3, 1>(3, 0) = left.pos_end - right.pos_end;
    difference.block<3, 1>(6, 0) =
        log_so3(
            right.offset_R_L_I.transpose() * left.offset_R_L_I);
    difference.block<3, 1>(9, 0) =
        left.offset_T_L_I - right.offset_T_L_I;
    difference.block<3, 1>(12, 0) =
        left.vel_end - right.vel_end;
    difference.block<3, 1>(15, 0) =
        left.bias_g - right.bias_g;
    difference.block<3, 1>(18, 0) =
        left.bias_a - right.bias_a;
    difference.block<3, 1>(21, 0) =
        left.gravity - right.gravity;
    return difference;
}

void add_state(
    FrontendState &state,
    const Eigen::Matrix<double, kStateDimension, 1> &addition) {
    state.rot_end =
        state.rot_end *
        exp_so3(addition(0, 0), addition(1, 0), addition(2, 0));
    state.pos_end += addition.block<3, 1>(3, 0);
    state.offset_R_L_I =
        state.offset_R_L_I *
        exp_so3(addition(6, 0), addition(7, 0), addition(8, 0));
    state.offset_T_L_I += addition.block<3, 1>(9, 0);
    state.vel_end += addition.block<3, 1>(12, 0);
    state.bias_g += addition.block<3, 1>(15, 0);
    state.bias_a += addition.block<3, 1>(18, 0);
    state.gravity += addition.block<3, 1>(21, 0);
}

float squared_distance(const PointType &left, const PointType &right) {
    return (left.x - right.x) * (left.x - right.x) +
           (left.y - right.y) * (left.y - right.y) +
           (left.z - right.z) * (left.z - right.z);
}

bool estimate_plane(
    Eigen::Matrix<double, 4, 1> &plane,
    const PointVector &points,
    double threshold) {
    Eigen::Matrix<double, kMatchPointCount, 3> matrix;
    Eigen::Matrix<double, kMatchPointCount, 1> values;
    matrix.setZero();
    values.setOnes();
    values *= -1.0F;
    for (int index = 0; index < kMatchPointCount; ++index) {
        matrix(index, 0) = points[index].x;
        matrix(index, 1) = points[index].y;
        matrix(index, 2) = points[index].z;
    }
    const Eigen::Vector3d normal =
        matrix.colPivHouseholderQr().solve(values);
    const double norm = normal.norm();
    plane(0) = normal(0) / norm;
    plane(1) = normal(1) / norm;
    plane(2) = normal(2) / norm;
    plane(3) = 1.0 / norm;
    for (int index = 0; index < kMatchPointCount; ++index) {
        if (std::fabs(
                plane(0) * points[index].x +
                plane(1) * points[index].y +
                plane(2) * points[index].z + plane(3)) >
            threshold)
            return false;
    }
    return true;
}

void calculate_body_variance(
    Eigen::Vector3d &point,
    float range_increment,
    float degree_increment,
    Eigen::Matrix3d &variance) {
    const float range = std::sqrt(
        point[0] * point[0] + point[1] * point[1] +
        point[2] * point[2]);
    const float range_variance = range_increment * range_increment;
    Eigen::Matrix2d direction_variance;
    const double angular_variance =
        std::pow(std::sin(degree_increment * M_PI / 180.0), 2);
    direction_variance << angular_variance, 0.0, 0.0,
        angular_variance;
    Eigen::Vector3d direction(point);
    direction.normalize();
    Eigen::Matrix3d direction_hat;
    direction_hat << 0.0, -direction(2), direction(1),
        direction(2), 0.0, -direction(0),
        -direction(1), direction(0), 0.0;
    Eigen::Vector3d base_vector_1(
        1.0,
        1.0,
        -(direction(0) + direction(1)) / direction(2));
    base_vector_1.normalize();
    Eigen::Vector3d base_vector_2 =
        base_vector_1.cross(direction);
    base_vector_2.normalize();
    Eigen::Matrix<double, 3, 2> basis;
    basis << base_vector_1(0), base_vector_2(0),
        base_vector_1(1), base_vector_2(1),
        base_vector_1(2), base_vector_2(2);
    const Eigen::Matrix<double, 3, 2> angular =
        range * direction_hat * basis;
    variance =
        direction * range_variance * direction.transpose() +
        angular * direction_variance * angular.transpose();
}

Eigen::Matrix3d skew(const Eigen::Vector3d &value) {
    Eigen::Matrix3d result;
    result << 0.0, -value[2], value[1],
        value[2], 0.0, -value[0],
        -value[1], value[0], 0.0;
    return result;
}

}  // namespace

struct LidarOdometryCore::Impl {
    explicit Impl(const LidarOdometryConfig &value) : config(value) {
        gain_product.setZero();
        hessian.setZero();
        identity.setIdentity();
    }

    PointType point_body_to_world(
        const PointType &input,
        const FrontendState &state) const {
        const Eigen::Vector3d body(input.x, input.y, input.z);
        const Eigen::Vector3d world =
            state.rot_end *
                (state.offset_R_L_I * body + state.offset_T_L_I) +
            state.pos_end;
        PointType output = input;
        output.x = world(0);
        output.y = world(1);
        output.z = world(2);
        return output;
    }

    void collect_removed_points() {
        PointVector history;
        tree.acquire_removed_points(history);
        removed_points.insert(
            removed_points.end(), history.begin(), history.end());
    }

    int segment_local_map(const FrontendState &state) {
        boxes_to_remove.clear();
        const Eigen::Vector3d position = state.pos_end;
        if (!local_map_initialized) {
            for (int axis = 0; axis < 3; ++axis) {
                local_map.vertex_min[axis] =
                    position(axis) - config.cube_side_length / 2.0;
                local_map.vertex_max[axis] =
                    position(axis) + config.cube_side_length / 2.0;
            }
            local_map_initialized = true;
            return 0;
        }

        float edge_distance[3][2];
        bool need_move = false;
        for (int axis = 0; axis < 3; ++axis) {
            edge_distance[axis][0] =
                std::fabs(position(axis) - local_map.vertex_min[axis]);
            edge_distance[axis][1] =
                std::fabs(position(axis) - local_map.vertex_max[axis]);
            if (edge_distance[axis][0] <=
                    kMovementThreshold * config.detection_range ||
                edge_distance[axis][1] <=
                    kMovementThreshold * config.detection_range)
                need_move = true;
        }
        if (!need_move)
            return 0;

        BoxPointType new_local_map = local_map;
        BoxPointType removed_box;
        const float movement_distance = std::max(
            (config.cube_side_length -
             2.0 * kMovementThreshold * config.detection_range) *
                0.5 * 0.9,
            static_cast<double>(
                config.detection_range *
                (kMovementThreshold - 1.0F)));
        for (int axis = 0; axis < 3; ++axis) {
            removed_box = local_map;
            if (edge_distance[axis][0] <=
                kMovementThreshold * config.detection_range) {
                new_local_map.vertex_max[axis] -= movement_distance;
                new_local_map.vertex_min[axis] -= movement_distance;
                removed_box.vertex_min[axis] =
                    local_map.vertex_max[axis] - movement_distance;
                boxes_to_remove.push_back(removed_box);
            } else if (
                edge_distance[axis][1] <=
                kMovementThreshold * config.detection_range) {
                new_local_map.vertex_max[axis] += movement_distance;
                new_local_map.vertex_min[axis] += movement_distance;
                removed_box.vertex_max[axis] =
                    local_map.vertex_min[axis] + movement_distance;
                boxes_to_remove.push_back(removed_box);
            }
        }
        local_map = new_local_map;
        collect_removed_points();
        if (!boxes_to_remove.empty())
            return tree.Delete_Point_Boxes(boxes_to_remove);
        return 0;
    }

    int increment_map(
        const PointCloudXYZI &body,
        PointCloudXYZI &world,
        const FrontendState &state) {
        PointVector add_with_downsample;
        PointVector add_without_downsample;
        add_with_downsample.reserve(body.size());
        add_without_downsample.reserve(body.size());
        for (std::size_t index = 0; index < body.size(); ++index) {
            world.points[index] =
                point_body_to_world(body.points[index], state);
            if (!nearest_points[index].empty()) {
                const PointVector &near = nearest_points[index];
                bool need_add = true;
                PointType midpoint;
                midpoint.x =
                    std::floor(
                        world.points[index].x /
                        config.filter_size_map) *
                        config.filter_size_map +
                    0.5 * config.filter_size_map;
                midpoint.y =
                    std::floor(
                        world.points[index].y /
                        config.filter_size_map) *
                        config.filter_size_map +
                    0.5 * config.filter_size_map;
                midpoint.z =
                    std::floor(
                        world.points[index].z /
                        config.filter_size_map) *
                        config.filter_size_map +
                    0.5 * config.filter_size_map;
                const float distance =
                    squared_distance(world.points[index], midpoint);
                if (std::fabs(near[0].x - midpoint.x) >
                        0.5 * config.filter_size_map &&
                    std::fabs(near[0].y - midpoint.y) >
                        0.5 * config.filter_size_map &&
                    std::fabs(near[0].z - midpoint.z) >
                        0.5 * config.filter_size_map) {
                    add_without_downsample.push_back(
                        world.points[index]);
                    continue;
                }
                for (int near_index = 0;
                     near_index < kMatchPointCount;
                     ++near_index) {
                    if (near.size() < kMatchPointCount)
                        break;
                    if (squared_distance(near[near_index], midpoint) <
                        distance) {
                        need_add = false;
                        break;
                    }
                }
                if (need_add)
                    add_with_downsample.push_back(world.points[index]);
            } else {
                add_with_downsample.push_back(world.points[index]);
            }
        }
        tree.Add_Points(add_with_downsample, true);
        tree.Add_Points(add_without_downsample, false);
        return static_cast<int>(
            add_with_downsample.size() +
            add_without_downsample.size());
    }

    LidarOdometryConfig config;
    KD_TREE tree;
    bool local_map_initialized = false;
    BoxPointType local_map;
    std::vector<BoxPointType> boxes_to_remove;
    PointVector removed_points;
    std::vector<PointVector> nearest_points;
    std::vector<unsigned char> point_selected_surf;
    std::vector<float> residual_last;
    Eigen::Matrix<double, kStateDimension, kStateDimension>
        gain_product;
    Eigen::Matrix<double, kStateDimension, kStateDimension> hessian;
    Eigen::Matrix<double, kStateDimension, kStateDimension> identity;
    Eigen::Vector3d position_last = Eigen::Vector3d::Zero();
    double total_distance = 0.0;
    double residual_mean = 0.05;
};

LidarOdometryCore::LidarOdometryCore(
    const LidarOdometryConfig &config)
    : impl_(new Impl(config)) {}

LidarOdometryCore::~LidarOdometryCore() = default;

const LidarOdometryConfig &LidarOdometryCore::config() const {
    return impl_->config;
}

LidarOdometryResult LidarOdometryCore::process(
    const PointCloudXYZI &undistorted,
    const LidarGroundEstimate &ground,
    FrontendState &state) {
    Impl &core = *impl_;
    LidarOdometryResult result;
    result.propagated_state = state;
    const FrontendState propagated_state = state;

    result.deleted_point_count = core.segment_local_map(state);

    PointCloudXYZI::Ptr downsampled(new PointCloudXYZI(
        legacy_voxel_grid(
            undistorted,
            static_cast<float>(core.config.filter_size_surf))));
    result.downsampled_body = *downsampled;
    const int feature_count =
        static_cast<int>(downsampled->points.size());
    core.point_selected_surf.assign(feature_count, true);
    core.residual_last.assign(feature_count, -1000.0F);

    PointCloudXYZI world;
    world.resize(feature_count);
    if (core.tree.Root_Node == nullptr) {
        if (feature_count > kMatchPointCount) {
            core.tree.set_downsample_param(
                core.config.filter_size_map);
            for (int index = 0; index < feature_count; ++index)
                world.points[index] = core.point_body_to_world(
                    downsampled->points[index], state);
            core.tree.Build(world.points);
            result.map_initialized_this_frame = true;
        }
        result.downsampled_world = world;
        result.map_size = core.tree.size();
        result.map_valid_points = core.tree.validnum();
        result.total_distance = core.total_distance;
        result.updated_state = state;
        return result;
    }

    PointCloudXYZI normals(feature_count, 1);
    PointCloudXYZI effective_body(feature_count, 1);
    PointCloudXYZI effective_normals(feature_count, 1);
    core.nearest_points.resize(feature_count);
    int rematch_count = 0;
    bool nearest_search_enabled = true;
    int effective_feature_count = 0;
    double delta_rotation = 0.0;
    double delta_translation = 0.0;

    std::vector<Eigen::Matrix3d> body_variance;
    std::vector<Eigen::Matrix3d> cross_matrices;
    body_variance.reserve(feature_count);
    cross_matrices.reserve(feature_count);

    for (int iteration = 0;
         iteration < core.config.max_iterations;
         ++iteration) {
        effective_body.clear();
        effective_normals.clear();
        double total_residual = 0.0;

        for (int index = 0; index < feature_count; ++index) {
            PointType &point_body = downsampled->points[index];
            PointType &point_world = world.points[index];
            const Eigen::Vector3d body(
                point_body.x, point_body.y, point_body.z);
            point_world = core.point_body_to_world(point_body, state);
            std::vector<float> squared_distances(kMatchPointCount);
            PointVector &near = core.nearest_points[index];
            if (nearest_search_enabled) {
                core.tree.Nearest_Search(
                    point_world,
                    kMatchPointCount,
                    near,
                    squared_distances,
                    5);
                if (near.size() < kMatchPointCount)
                    core.point_selected_surf[index] = false;
                else
                    core.point_selected_surf[index] =
                        !(squared_distances[kMatchPointCount - 1] >
                          5);
            }

            core.residual_last[index] = -1000.0F;
            if (!core.point_selected_surf[index] ||
                near.size() < kMatchPointCount) {
                core.point_selected_surf[index] = false;
                continue;
            }

            core.point_selected_surf[index] = false;
            Eigen::Matrix<double, 4, 1> plane;
            plane.setZero();
            if (estimate_plane(plane, near, 0.1)) {
                const float distance =
                    plane(0) * point_world.x +
                    plane(1) * point_world.y +
                    plane(2) * point_world.z + plane(3);
                const float score =
                    1.0F -
                    0.9F * std::fabs(distance) /
                        std::sqrt(body.norm());
                if (score > 0.9F) {
                    core.point_selected_surf[index] = true;
                    normals.points[index].x = plane(0);
                    normals.points[index].y = plane(1);
                    normals.points[index].z = plane(2);
                    normals.points[index].intensity = distance;
                    core.residual_last[index] =
                        std::abs(distance);
                }
            }
        }

        effective_feature_count = 0;
        for (int index = 0; index < feature_count; ++index) {
            if (core.point_selected_surf[index]) {
                effective_body.points[effective_feature_count] =
                    downsampled->points[index];
                effective_normals.points[effective_feature_count] =
                    normals.points[index];
                ++effective_feature_count;
            }
        }
        core.residual_mean =
            total_residual / effective_feature_count;

        const int residual_dimension =
            effective_feature_count + 3;
        Eigen::MatrixXd measurement_jacobian(
            residual_dimension, 12);
        Eigen::MatrixXd weighted_jacobian_transpose(
            12, residual_dimension);
        Eigen::VectorXd inverse_covariance(residual_dimension);
        Eigen::VectorXd measurements(residual_dimension);
        measurement_jacobian.setZero();
        weighted_jacobian_transpose.setZero();
        measurements.setZero();

        for (int index = 0;
             index < effective_feature_count;
             ++index) {
            const PointType &laser_point =
                effective_body.points[index];
            const Eigen::Vector3d point_lidar(
                laser_point.x, laser_point.y, laser_point.z);
            Eigen::Vector3d point =
                state.offset_R_L_I * point_lidar +
                state.offset_T_L_I;
            Eigen::Matrix3d variance;
            calculate_body_variance(point, 0.02F, 0.05F, variance);
            variance =
                state.rot_end * variance * state.rot_end.transpose();
            const Eigen::Matrix3d point_cross = skew(point);
            const PointType &normal =
                effective_normals.points[index];
            const Eigen::Vector3d normal_vector(
                normal.x, normal.y, normal.z);
            inverse_covariance(index) = 1000;
            effective_body.points[index].intensity =
                std::sqrt(inverse_covariance(index));
            const Eigen::Vector3d rotation_jacobian =
                point_cross * state.rot_end.transpose() *
                normal_vector;
            measurement_jacobian.row(index)
                << rotation_jacobian(0),
                rotation_jacobian(1),
                rotation_jacobian(2),
                normal.x,
                normal.y,
                normal.z,
                0,
                0,
                0,
                0,
                0,
                0;
            weighted_jacobian_transpose.col(index) =
                measurement_jacobian.row(index).transpose() *
                1000;
            measurements(index) = -normal.intensity;
        }

        const Eigen::Matrix3d lidar_ground =
            ground.lidar_ground_rotation.toRotationMatrix();
        const Eigen::Matrix3d ground_lidar =
            lidar_ground.transpose();
        const Eigen::Vector3d axis_x(1.0, 0.0, 0.0);
        const Eigen::Vector3d axis_y(0.0, 1.0, 0.0);
        const Eigen::Vector3d axis_z(0.0, 0.0, 1.0);
        const Eigen::Matrix3d ground_normal_cross =
            skew(ground.normal_lidar);
        const Eigen::Vector3d ground_jacobian_x =
            ground_normal_cross * ground_lidar *
            state.rot_end * axis_x;
        const Eigen::Vector3d ground_jacobian_y =
            ground_normal_cross * ground_lidar *
            state.rot_end * axis_y;
        const Eigen::Vector3d ground_jacobian_z =
            ground_lidar * axis_z;

        measurement_jacobian.row(effective_feature_count)
            << ground_jacobian_x(0),
            ground_jacobian_x(1),
            ground_jacobian_x(2),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0;
        weighted_jacobian_transpose.col(
            effective_feature_count) =
            measurement_jacobian.row(
                effective_feature_count).transpose() *
            core.config.ground_covariance;
        inverse_covariance(effective_feature_count) =
            core.config.ground_covariance;

        measurement_jacobian.row(effective_feature_count + 1)
            << ground_jacobian_y(0),
            ground_jacobian_y(1),
            ground_jacobian_y(2),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0;
        weighted_jacobian_transpose.col(
            effective_feature_count + 1) =
            measurement_jacobian.row(
                effective_feature_count + 1).transpose() *
            core.config.ground_covariance;
        inverse_covariance(effective_feature_count + 1) =
            core.config.ground_covariance;

        measurement_jacobian.row(effective_feature_count + 2)
            << 0,
            0,
            0,
            ground_jacobian_z(0),
            ground_jacobian_z(1),
            ground_jacobian_z(2),
            0,
            0,
            0,
            0,
            0,
            0;
        weighted_jacobian_transpose.col(
            effective_feature_count + 2) =
            measurement_jacobian.row(
                effective_feature_count + 2).transpose() *
            core.config.ground_covariance;
        inverse_covariance(effective_feature_count + 2) =
            core.config.ground_covariance;

        const Eigen::Vector3d ground_rotation_measurement =
            ground_lidar * state.rot_end *
            ground.normal_lidar;
        const Eigen::Vector3d ground_position_measurement =
            ground_lidar * state.pos_end;
        measurements(effective_feature_count) =
            axis_x.transpose() * ground_rotation_measurement;
        measurements(effective_feature_count + 1) =
            axis_y.transpose() * ground_rotation_measurement;
        measurements(effective_feature_count + 2) =
            axis_z.transpose() * ground_position_measurement -
            1.0;

        Eigen::MatrixXd kalman_gain(
            kStateDimension, residual_dimension);
        bool stop = false;
        bool converged = false;
        core.hessian.block<12, 12>(0, 0) =
            weighted_jacobian_transpose *
            measurement_jacobian;
        Eigen::Matrix<double, kStateDimension, kStateDimension>
            inverse =
                (core.hessian + state.cov.inverse()).inverse();
        kalman_gain =
            inverse.block<kStateDimension, 12>(0, 0) *
            weighted_jacobian_transpose;
        const Eigen::Matrix<double, kStateDimension, 1>
            propagated_difference =
                state_difference(propagated_state, state);
        const Eigen::Matrix<double, kStateDimension, 1> solution =
            kalman_gain * measurements +
            propagated_difference -
            kalman_gain * measurement_jacobian *
                propagated_difference.block<12, 1>(0, 0);
        add_state(state, solution);

        const Eigen::Vector3d rotation_addition =
            solution.block<3, 1>(0, 0);
        const Eigen::Vector3d translation_addition =
            solution.block<3, 1>(3, 0);
        if (rotation_addition.norm() * 57.3 < 0.01 &&
            translation_addition.norm() * 100 < 0.015)
            converged = true;
        delta_rotation = rotation_addition.norm() * 57.3;
        delta_translation =
            translation_addition.norm() * 100;

        nearest_search_enabled = false;
        if (converged ||
            (rematch_count == 0 &&
             iteration == core.config.max_iterations - 2)) {
            nearest_search_enabled = true;
            ++rematch_count;
        }

        if (!stop &&
            (rematch_count >= 2 ||
             iteration == core.config.max_iterations - 1)) {
            core.gain_product.setZero();
            core.gain_product.block<kStateDimension, 12>(0, 0) =
                kalman_gain * measurement_jacobian;
            state.cov =
                (core.identity - core.gain_product) * state.cov;
            core.total_distance +=
                (state.pos_end - core.position_last).norm();
            core.position_last = state.pos_end;
            const Eigen::Matrix<double, kStateDimension, 1>
                gain_sum = kalman_gain.rowwise().sum();
            const Eigen::Matrix<double, kStateDimension, 1>
                covariance_diagonal = state.cov.diagonal();
            (void)gain_sum;
            (void)covariance_diagonal;
            stop = true;
        }

        result.iterations = iteration + 1;
        result.rematch_count = rematch_count;
        if (stop)
            break;
    }

    result.update_performed = true;
    result.effective_feature_count = effective_feature_count;
    result.residual_mean = core.residual_mean;
    result.delta_rotation_deg = delta_rotation;
    result.delta_translation_cm = delta_translation;
    result.effective_body.reserve(effective_feature_count);
    result.effective_normals.reserve(effective_feature_count);
    for (int index = 0; index < effective_feature_count; ++index) {
        result.effective_body.push_back(
            effective_body.points[index]);
        result.effective_normals.push_back(
            effective_normals.points[index]);
    }
    result.added_point_count =
        core.increment_map(*downsampled, world, state);
    result.downsampled_world = world;
    result.map_size = core.tree.size();
    result.map_valid_points = core.tree.validnum();
    result.total_distance = core.total_distance;
    result.updated_state = state;
    result.nearest_points.assign(
        core.nearest_points.begin(), core.nearest_points.end());
    result.selected.resize(feature_count);
    result.residuals.resize(feature_count);
    for (int index = 0; index < feature_count; ++index) {
        result.selected[index] =
            core.point_selected_surf[index];
        result.residuals[index] = core.residual_last[index];
    }
    return result;
}
