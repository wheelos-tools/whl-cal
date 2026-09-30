/*
 * Deterministic ROS-free Velodyne preprocessing tests.
 * Copyright (C) 2026 whl-cal contributors.
 * GPL-2.0-only; see ../LICENSE.
 */

#include <Gril_Calib/VelodynePreprocess.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

void require_near(double actual, double expected, double tolerance,
                  const char *message) {
  if (std::abs(actual - expected) > tolerance)
    throw std::runtime_error(message);
}

VelodynePoint point(double radius, double yaw_degrees, float time_s,
                    std::uint16_t ring, float intensity = 0.0F) {
  const double yaw = yaw_degrees * kPi / 180.0;
  VelodynePoint value;
  value.x = static_cast<float>(radius * std::cos(yaw));
  value.y = static_cast<float>(radius * std::sin(yaw));
  value.z = 0.0F;
  value.intensity = intensity;
  value.time_s = time_s;
  value.ring = ring;
  return value;
}

VelodynePreprocessConfig base_config() {
  VelodynePreprocessConfig config;
  config.blind = 0.0;
  config.point_filter_num = 1;
  config.n_scans = 16;
  config.required_frame_num = 1;
  config.scan_count = 20;
  return config;
}

void supplied_zero_first() {
  std::vector<VelodynePoint> points{
      point(2.0, 30.0, 0.00F, 0, 0.0F), point(2.0, 20.0, 0.01F, 0, 1.0F),
      point(2.0, 10.0, 0.02F, 0, 2.0F), point(2.0, 0.0, 0.03F, 0, 3.0F)};
  const auto result = preprocess_velodyne_scan(points, 1.0, base_config());

  require(result.given_offset_time,
          "first-zero supplied timing was not accepted");
  require(result.surface.size() == 4, "surface size mismatch");
  require_near(result.surface[0].curvature, 0.0, 1e-6,
               "first supplied offset changed");
  require_near(result.surface[3].curvature, 30.0, 1e-5,
               "last supplied offset mismatch");
  require(result.cut_clouds.size() == 1, "expected one cut");
  require(result.cut_clouds[0].size() == 3,
          "upstream first sorted point was not omitted");
  require_near(result.cut_timestamps_ms[0], 1000.0, 1e-12,
               "cut timestamp mismatch");
}

void fallback_azimuth() {
  std::vector<VelodynePoint> points{point(2.0, 10.0, 0.0F, 0),
                                    point(2.0, 0.0, 0.0F, 0),
                                    point(2.0, -10.0, 0.0F, 0)};
  const auto result = preprocess_velodyne_scan(points, 2.0, base_config());

  require(!result.given_offset_time,
          "zero-ending scan did not use fallback timing");
  require(result.surface.size() == 2,
          "fallback must omit the first point of each ring");
  require_near(result.surface[0].curvature, 10.0 / 3.61, 2e-4,
               "first fallback azimuth offset mismatch");
  require_near(result.surface[1].curvature, 20.0 / 3.61, 2e-4,
               "second fallback azimuth offset mismatch");
}

void nonfinite() {
  std::vector<VelodynePoint> points{
      point(2.0, 30.0, 0.00F, 0, 0.0F), point(2.0, 20.0, 0.01F, 0, 1.0F),
      point(2.0, 10.0, 0.02F, 0, 2.0F), point(2.0, 0.0, 0.03F, 0, 3.0F)};
  points[1].x = std::numeric_limits<float>::quiet_NaN();
  points[2].y = std::numeric_limits<float>::infinity();

  const auto result = preprocess_velodyne_scan(points, 3.0, base_config());
  require(result.given_offset_time, "supplied timing decision changed");
  require(result.surface.size() == 2,
          "non-finite coordinates were not filtered");
  require(result.surface[0].intensity == 0.0F,
          "wrong first finite point retained");
  require(result.surface[1].intensity == 3.0F,
          "wrong last finite point retained");
}

void ring_stride_blind() {
  std::vector<VelodynePoint> points{
      point(2.0, 60.0, 0.00F, 0, 0.0F), point(2.0, 50.0, 0.01F, 0, 1.0F),
      point(0.5, 40.0, 0.02F, 0, 2.0F), point(2.0, 30.0, 0.03F, 0, 3.0F),
      point(2.0, 20.0, 0.04F, 4, 4.0F), point(2.0, 10.0, 0.05F, 0, 5.0F),
      point(2.0, 0.0, 0.06F, 1, 6.0F)};
  auto config = base_config();
  config.blind = 1.0;
  config.point_filter_num = 2;
  config.n_scans = 4;

  const auto result = preprocess_velodyne_scan(points, 4.0, config);
  require(result.surface.size() == 2,
          "blind/ring/stride filter count mismatch");
  require(result.surface[0].intensity == 0.0F,
          "original-index stride did not retain point zero");
  require(result.surface[1].intensity == 6.0F,
          "ring/stride filtering retained the wrong point");
}

std::vector<VelodynePoint> cut_points() {
  std::vector<VelodynePoint> points;
  for (int i = 0; i < 9; i++) {
    points.push_back(point(2.0, 80.0 - i * 10.0, static_cast<float>(i * 0.01),
                           0, static_cast<float>(i)));
  }
  return points;
}

void cut_frame() {
  auto config = base_config();
  config.required_frame_num = 3;
  config.scan_count = 19;
  const auto warmup = preprocess_velodyne_scan(cut_points(), 5.0, config);
  require(warmup.cut_clouds.size() == 1, "scan 19 must force one cut");
  require(warmup.cut_clouds[0].size() == 8, "warmup cut size mismatch");
  require_near(warmup.cut_timestamps_ms[0], 5000.0, 1e-12,
               "warmup timestamp mismatch");

  config.scan_count = 20;
  const auto steady = preprocess_velodyne_scan(cut_points(), 5.0, config);
  require(steady.cut_clouds.size() == 3,
          "scan 20 must honor requested cut count");
  require(steady.cut_clouds[0].size() == 2, "first steady cut size mismatch");
  require(steady.cut_clouds[1].size() == 3, "second steady cut size mismatch");
  require(steady.cut_clouds[2].size() == 3, "third steady cut size mismatch");
  require_near(steady.cut_timestamps_ms[0], 5000.0, 1e-12,
               "first steady timestamp mismatch");
  require_near(steady.cut_timestamps_ms[1], 5020.0, 1e-4,
               "second steady timestamp mismatch");
  require_near(steady.cut_timestamps_ms[2], 5050.0, 1e-4,
               "third steady timestamp mismatch");
  require_near(steady.cut_clouds[1][0].curvature, 10.0, 1e-4,
               "second cut point time was not rebased");
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "expected one test mode");
    const std::string mode(argv[1]);
    if (mode == "supplied_zero_first")
      supplied_zero_first();
    else if (mode == "fallback_azimuth")
      fallback_azimuth();
    else if (mode == "nonfinite")
      nonfinite();
    else if (mode == "ring_stride_blind")
      ring_stride_blind();
    else if (mode == "cut_frame")
      cut_frame();
    else
      throw std::runtime_error("unknown test mode");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
