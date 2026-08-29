/*
 * ROS-free VELO/Velodyne preprocessing adapted from GRIL-Calib.
 * Original implementation: TaeYoung Kim and GRIL-Calib contributors.
 *
 * Modified 2026-08-28 to replace PointCloud2 conversion with typed input.
 * Valid-input filtering, timing, sorting, and cut order follow the patched
 * upstream process_cut_frame_pcl2 VELO branch.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License, version 2.
 * See ../../LICENSE.
 */

#include "VelodynePreprocess.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

constexpr int kMaxLineNum = 128;

bool time_list_cut_frame(const PointType &x, const PointType &y) {
    return x.curvature < y.curvature;
}

VelodynePreprocessResult process_validated_velodyne_scan(
    const std::vector<VelodynePoint> &pl_orig,
    double scan_timestamp_s,
    const VelodynePreprocessConfig &config) {
    VelodynePreprocessResult result;
    const int plsize = static_cast<int>(pl_orig.size());
    result.surface.reserve(plsize);

    bool is_first[kMaxLineNum];
    double yaw_fp[kMaxLineNum] = {0};
    double omega_l = 3.61;
    float yaw_last[kMaxLineNum] = {0.0F};
    float time_last[kMaxLineNum] = {0.0F};

    if (pl_orig[plsize - 1].time_s > 0) {
        result.given_offset_time = true;
    } else {
        result.given_offset_time = false;
        std::memset(is_first, true, sizeof(is_first));
        double yaw_first =
            std::atan2(pl_orig[0].y, pl_orig[0].x) * 57.29578;
        double yaw_end = yaw_first;
        int layer_first = pl_orig[0].ring;
        for (unsigned int i = plsize - 1; i > 0; i--) {
            if (pl_orig[i].ring == layer_first) {
                yaw_end =
                    std::atan2(pl_orig[i].y, pl_orig[i].x) * 57.29578;
                break;
            }
        }
        (void)yaw_end;
    }

    for (int i = 0; i < plsize; i++) {
        PointType added_pt;
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.x = pl_orig[i].x;
        added_pt.y = pl_orig[i].y;
        added_pt.z = pl_orig[i].z;
        added_pt.intensity = pl_orig[i].intensity;
        added_pt.curvature = pl_orig[i].time_s * 1000.0;

        double dist = added_pt.x * added_pt.x +
                      added_pt.y * added_pt.y +
                      added_pt.z * added_pt.z;
        if (dist < config.blind * config.blind ||
            !std::isfinite(added_pt.x) ||
            !std::isfinite(added_pt.y) ||
            !std::isfinite(added_pt.z))
            continue;

        if (!result.given_offset_time) {
            int layer = pl_orig[i].ring;
            double yaw_angle =
                std::atan2(added_pt.y, added_pt.x) * 57.2957;

            if (is_first[layer]) {
                yaw_fp[layer] = yaw_angle;
                is_first[layer] = false;
                added_pt.curvature = 0.0;
                yaw_last[layer] = yaw_angle;
                time_last[layer] = added_pt.curvature;
                continue;
            }

            if (yaw_angle <= yaw_fp[layer]) {
                added_pt.curvature =
                    (yaw_fp[layer] - yaw_angle) / omega_l;
            } else {
                added_pt.curvature =
                    (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
            }
            if (added_pt.curvature < time_last[layer])
                added_pt.curvature += 360.0 / omega_l;

            yaw_last[layer] = yaw_angle;
            time_last[layer] = added_pt.curvature;
        }

        if (i % config.point_filter_num == 0 &&
            pl_orig[i].ring < config.n_scans) {
            result.surface.points.push_back(added_pt);
        }
    }

    std::sort(
        result.surface.points.begin(),
        result.surface.points.end(),
        time_list_cut_frame);

    double last_frame_end_time = scan_timestamp_s * 1000;
    unsigned int valid_num = 0;
    unsigned int cut_num = 0;
    unsigned int valid_pcl_size = result.surface.points.size();

    int required_cut_num = config.required_frame_num;
    if (config.scan_count < 20)
        required_cut_num = 1;

    PointCloudXYZI pcl_cut;
    for (unsigned int i = 1; i < valid_pcl_size; i++) {
        valid_num++;
        result.surface[i].curvature +=
            scan_timestamp_s * 1000 - last_frame_end_time;
        pcl_cut.push_back(result.surface[i]);

        if (valid_num ==
            (int((cut_num + 1) * valid_pcl_size / required_cut_num) - 1)) {
            cut_num++;
            result.cut_timestamps_ms.push_back(last_frame_end_time);
            result.cut_clouds.push_back(pcl_cut);
            last_frame_end_time += result.surface[i].curvature;
            pcl_cut.clear();
            pcl_cut.reserve(
                valid_pcl_size * 2 / config.required_frame_num);
        }
    }
    return result;
}

}  // namespace

void validate_velodyne_preprocess_input(
    const std::vector<VelodynePoint> &points,
    double scan_timestamp_s,
    const VelodynePreprocessConfig &config) {
    if (points.empty())
        throw std::invalid_argument("Velodyne scan must not be empty");
    if (!std::isfinite(scan_timestamp_s))
        throw std::invalid_argument("scan timestamp must be finite");
    if (!std::isfinite(config.blind) || config.blind < 0.0)
        throw std::invalid_argument("blind distance must be finite and nonnegative");
    if (config.point_filter_num <= 0)
        throw std::invalid_argument("point_filter_num must be positive");
    if (config.n_scans <= 0 || config.n_scans > kMaxLineNum)
        throw std::invalid_argument("n_scans must be in [1, 128]");
    if (config.required_frame_num <= 0)
        throw std::invalid_argument("required_frame_num must be positive");
    if (config.scan_count < 0)
        throw std::invalid_argument("scan_count must be nonnegative");
    for (const auto &point : points) {
        if (point.ring >= kMaxLineNum)
            throw std::invalid_argument("ring must be less than 128");
        if (!std::isfinite(point.time_s))
            throw std::invalid_argument("point time must be finite");
    }
}

VelodynePreprocessResult preprocess_velodyne_scan(
    const std::vector<VelodynePoint> &points,
    double scan_timestamp_s,
    const VelodynePreprocessConfig &config) {
    validate_velodyne_preprocess_input(
        points, scan_timestamp_s, config);
    return process_validated_velodyne_scan(
        points, scan_timestamp_s, config);
}
