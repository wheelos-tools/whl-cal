/*
 * ROS-free VELO/Velodyne preprocessing adapted from GRIL-Calib.
 * Original implementation: TaeYoung Kim and GRIL-Calib contributors.
 *
 * Modified 2026-08-28 to accept native typed points and timestamps.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License, version 2.
 * See ../../LICENSE.
 */

#ifndef GRIL_NATIVE_VELODYNE_PREPROCESS_H
#define GRIL_NATIVE_VELODYNE_PREPROCESS_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cstdint>
#include <deque>
#include <vector>

using PointType = pcl::PointXYZINormal;
using PointCloudXYZI = pcl::PointCloud<PointType>;

struct VelodynePoint {
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
    float intensity = 0.0F;
    float time_s = 0.0F;
    std::uint16_t ring = 0;
};

struct VelodynePreprocessConfig {
    double blind = 1.0;
    int point_filter_num = 1;
    int n_scans = 16;
    int required_frame_num = 1;
    int scan_count = 0;
};

struct VelodynePreprocessResult {
    PointCloudXYZI surface;
    std::deque<PointCloudXYZI> cut_clouds;
    std::deque<double> cut_timestamps_ms;
    bool given_offset_time = false;
};

void validate_velodyne_preprocess_input(
    const std::vector<VelodynePoint> &points,
    double scan_timestamp_s,
    const VelodynePreprocessConfig &config);

VelodynePreprocessResult preprocess_velodyne_scan(
    const std::vector<VelodynePoint> &points,
    double scan_timestamp_s,
    const VelodynePreprocessConfig &config);

#endif
