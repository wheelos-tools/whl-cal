/*
 * ROS-free Patchwork++ construction and planar-ground smoke test.
 * Copyright (C) 2026 whl-cal contributors.
 * GPL-2.0-only; see ../LICENSE.
 */

#include <GroundSegmentation/PatchworkppNative.h>

#include <iostream>
#include <stdexcept>

int main() {
    try {
        PatchworkppConfig config;
        config.sensor_height = 1.117;
        config.min_range = 1.0;
        config.max_range = 5.0;
        config.uprightness_thr = 0.707;
        config.num_sectors_each_zone = {16, 32, 54, 32};
        config.num_rings_each_zone = {2, 4, 4, 4};
        config.elevation_thresholds = {0.1, 0.2, 0.4, 0.6};
        config.flatness_thresholds = {0.0, 0.0, 0.0, 0.0};
        PatchWorkpp<pcl::PointXYZI> patchwork(config);

        pcl::PointCloud<pcl::PointXYZI> input;
        for (int x = -20; x <= 20; ++x) {
            for (int y = -20; y <= 20; ++y) {
                pcl::PointXYZI point;
                point.x = static_cast<float>(x) * 0.2F;
                point.y = static_cast<float>(y) * 0.2F;
                point.z = -1.117F;
                point.intensity = 1.0F;
                input.push_back(point);
            }
        }
        pcl::PointCloud<pcl::PointXYZI> ground;
        pcl::PointCloud<pcl::PointXYZI> nonground;
        double elapsed = 0.0;
        patchwork.estimate_ground(input, ground, nonground, elapsed);
        if (ground.empty())
            throw std::runtime_error("planar ground was not detected");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
