/*
 * ROS-free preprocessing trace runner for GRIL migration equivalence.
 * Copyright (C) 2026 whl-cal contributors.
 * GPL-2.0-only; see ../LICENSE.
 */

#include <Gril_Calib/VelodynePreprocess.h>

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

void write_cloud(std::ostream &output, const PointCloudXYZI &cloud) {
  output << cloud.size() << "\n";
  for (const auto &point : cloud.points) {
    output << "point " << point.x << " " << point.y << " " << point.z << " "
           << point.intensity << " " << point.curvature << "\n";
  }
}

void write_trace(std::ostream &output, int scan_count,
                 std::int64_t timestamp_ns,
                 const VelodynePreprocessResult &result) {
  output << "scan " << scan_count << " " << timestamp_ns << "\n";
  output << "given_offset_time " << (result.given_offset_time ? 1 : 0) << "\n";
  output << "surface ";
  write_cloud(output, result.surface);
  output << "cuts " << result.cut_clouds.size() << "\n";
  for (std::size_t i = 0; i < result.cut_clouds.size(); ++i) {
    output << "cut " << result.cut_timestamps_ms[i] << " ";
    write_cloud(output, result.cut_clouds[i]);
  }
  output << "end_scan\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error(
          "usage: gril_native_preprocess_trace INPUT OUTPUT");

    std::ifstream input(argv[1]);
    if (!input)
      throw std::runtime_error("could not open preprocessing input");
    std::ofstream output(argv[2]);
    if (!output)
      throw std::runtime_error("could not open preprocessing output");

    expect(input, "GRIL_PREPROCESS_INPUT");
    int version = 0;
    input >> version;
    if (version != 1)
      throw std::runtime_error("unsupported preprocessing input version");

    VelodynePreprocessConfig config;
    expect(input, "config");
    input >> config.blind >> config.point_filter_num >> config.n_scans >>
        config.required_frame_num;
    expect(input, "scans");
    std::size_t scan_total = 0;
    input >> scan_total;

    output << std::setprecision(17);
    output << "GRIL_PREPROCESS_TRACE 1\n";
    for (std::size_t scan_index = 0; scan_index < scan_total; ++scan_index) {
      expect(input, "scan");
      int scan_count = 0;
      std::int64_t timestamp_ns = 0;
      std::size_t point_count = 0;
      input >> scan_count >> timestamp_ns >> point_count;
      std::vector<VelodynePoint> points;
      points.reserve(point_count);
      for (std::size_t point_index = 0; point_index < point_count;
           ++point_index) {
        expect(input, "point");
        VelodynePoint point;
        input >> point.x >> point.y >> point.z >> point.intensity >>
            point.time_s >> point.ring;
        points.push_back(point);
      }
      config.scan_count = scan_count;
      const double timestamp_s =
          static_cast<double>(timestamp_ns) / 1000000000.0;
      write_trace(output, scan_count, timestamp_ns,
                  preprocess_velodyne_scan(points, timestamp_s, config));
    }
    expect(input, "END");
    output << "END\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
