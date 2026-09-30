/*
 * Replay canonical scans through the ROS-free pinned Patchwork++ algorithm.
 * Copyright (C) 2026 whl-cal contributors.
 * GPL-2.0-only; see ../LICENSE.
 */

#include <GroundSegmentation/PatchworkppNative.h>

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(std::istream &input, const std::string &expected) {
  std::string actual;
  if (!(input >> actual) || actual != expected)
    throw std::runtime_error("expected token: " + expected);
}

template <typename T>
std::vector<T> read_vector(std::istream &input, const std::string &label) {
  expect(input, label);
  std::size_t count = 0;
  input >> count;
  std::vector<T> values(count);
  for (auto &value : values)
    input >> value;
  return values;
}

void write_cloud(std::ostream &output, const std::string &label,
                 const pcl::PointCloud<pcl::PointXYZI> &cloud) {
  output << label << " " << cloud.size() << "\n";
  for (const auto &point : cloud.points)
    output << "point " << point.x << " " << point.y << " " << point.z << " "
           << point.intensity << "\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error("usage: gril_native_ground_trace INPUT OUTPUT");
    std::ifstream input(argv[1]);
    std::ofstream output(argv[2]);
    if (!input || !output)
      throw std::runtime_error("could not open ground trace");

    expect(input, "GRIL_GROUND_INPUT");
    int version = 0;
    input >> version;
    if (version != 1)
      throw std::runtime_error("unsupported ground input version");
    expect(input, "config");
    PatchworkppConfig config;
    input >> config.sensor_height >> config.num_iter >> config.num_lpr >>
        config.num_min_pts >> config.max_flatness_storage >>
        config.max_elevation_storage >> config.th_seeds >> config.th_dist >>
        config.th_seeds_v >> config.th_dist_v >> config.max_range >>
        config.min_range >> config.uprightness_thr >>
        config.adaptive_seed_selection_margin >> config.rnr_ver_angle_thr >>
        config.rnr_intensity_thr >> config.enable_rnr >> config.enable_rvpf >>
        config.enable_tgr;
    config.num_sectors_each_zone = read_vector<int>(input, "sectors");
    config.num_rings_each_zone = read_vector<int>(input, "rings");
    config.elevation_thresholds = read_vector<double>(input, "elevation");
    config.flatness_thresholds = read_vector<double>(input, "flatness");
    PatchWorkpp<pcl::PointXYZI> patchwork(config);

    expect(input, "scans");
    std::size_t scan_count = 0;
    input >> scan_count;
    output << std::setprecision(17);
    output << "GRIL_GROUND_TRACE 1\n";
    for (std::size_t scan_index = 0; scan_index < scan_count; ++scan_index) {
      expect(input, "scan");
      int scan_number = 0;
      std::int64_t timestamp_ns = 0;
      std::size_t point_count = 0;
      input >> scan_number >> timestamp_ns >> point_count;
      pcl::PointCloud<pcl::PointXYZI> cloud;
      cloud.reserve(point_count);
      for (std::size_t point_index = 0; point_index < point_count;
           ++point_index) {
        expect(input, "point");
        pcl::PointXYZI point;
        input >> point.x >> point.y >> point.z >> point.intensity;
        cloud.push_back(point);
      }
      pcl::PointCloud<pcl::PointXYZI> ground;
      pcl::PointCloud<pcl::PointXYZI> nonground;
      double elapsed = 0.0;
      patchwork.estimate_ground(cloud, ground, nonground, elapsed);
      output << "scan " << scan_number << " " << timestamp_ns << "\n";
      write_cloud(output, "ground", ground);
      write_cloud(output, "nonground", nonground);
      output << "end_scan\n";
    }
    expect(input, "END");
    output << "END\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
