/*
 * Replay frozen GRIL constant-velocity frontend traces without ROS.
 * Copyright (C) 2026 whl-cal contributors.
 * GPL-2.0-only; see ../LICENSE.
 */

#include <Gril_Calib/FrontendCore.h>

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

FrontendState read_state(std::istream &input, const std::string &label) {
  expect(input, label);
  FrontendState state;
  for (int row = 0; row < 3; ++row)
    for (int col = 0; col < 3; ++col)
      input >> state.rot_end(row, col);
  for (int index = 0; index < 3; ++index)
    input >> state.pos_end(index);
  for (int row = 0; row < 3; ++row)
    for (int col = 0; col < 3; ++col)
      input >> state.offset_R_L_I(row, col);
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
    for (int col = 0; col < 24; ++col)
      input >> state.cov(row, col);
  if (!input)
    throw std::runtime_error("invalid frontend state");
  return state;
}

void write_state(std::ostream &output, const std::string &label,
                 const FrontendState &state) {
  output << label;
  for (int row = 0; row < 3; ++row)
    for (int col = 0; col < 3; ++col)
      output << " " << state.rot_end(row, col);
  for (int index = 0; index < 3; ++index)
    output << " " << state.pos_end(index);
  for (int row = 0; row < 3; ++row)
    for (int col = 0; col < 3; ++col)
      output << " " << state.offset_R_L_I(row, col);
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
    for (int col = 0; col < 24; ++col)
      output << " " << state.cov(row, col);
  output << "\n";
}

PointCloudXYZI read_cloud(std::istream &input, const std::string &label) {
  expect(input, label);
  std::size_t count = 0;
  input >> count;
  PointCloudXYZI cloud;
  cloud.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    expect(input, "point");
    PointType point;
    input >> point.x >> point.y >> point.z >> point.intensity >>
        point.curvature;
    point.normal_x = 0.0F;
    point.normal_y = 0.0F;
    point.normal_z = 0.0F;
    cloud.push_back(point);
  }
  if (!input)
    throw std::runtime_error("invalid frontend cloud");
  return cloud;
}

void write_cloud(std::ostream &output, const std::string &label,
                 const PointCloudXYZI &cloud) {
  output << label << " " << cloud.size() << "\n";
  for (const auto &point : cloud.points) {
    output << "point " << point.x << " " << point.y << " " << point.z << " "
           << point.intensity << " " << point.curvature << "\n";
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error(
          "usage: gril_native_frontend_cv_trace INPUT OUTPUT");
    std::ifstream input(argv[1]);
    std::ofstream output(argv[2]);
    if (!input || !output)
      throw std::runtime_error("could not open frontend trace");

    expect(input, "GRIL_FRONTEND_CV_TRACE");
    int version = 0;
    input >> version;
    if (version != 1)
      throw std::runtime_error("unsupported frontend trace version");
    expect(input, "config");
    double gyr_cov = 0.0;
    double acc_cov = 0.0;
    input >> gyr_cov >> acc_cov;
    ConstantVelocityPropagator propagator(Eigen::Vector3d::Constant(gyr_cov),
                                          Eigen::Vector3d::Constant(acc_cov));

    output << std::setprecision(17);
    output << "GRIL_FRONTEND_CV_TRACE 1\n";
    output << "config " << gyr_cov << " " << acc_cov << "\n";
    std::string token;
    while (input >> token) {
      if (token == "END") {
        output << "END\n";
        return 0;
      }
      if (token != "package")
        throw std::runtime_error("expected package or END");
      int package_index = 0;
      FrontendMeasureGroup measure;
      double lidar_end_time_s = 0.0;
      input >> package_index >> measure.lidar_beg_time_s >> lidar_end_time_s;
      expect(input, "imu");
      std::size_t imu_count = 0;
      input >> imu_count;
      for (std::size_t index = 0; index < imu_count; ++index) {
        expect(input, "imu_sample");
        FrontendImuSample sample;
        input >> sample.timestamp_s;
        measure.imu.push_back(sample);
      }
      const FrontendState before = read_state(input, "before");
      FrontendState state = before;
      measure.lidar = read_cloud(input, "input");
      read_state(input, "after");
      read_cloud(input, "output");
      expect(input, "end_package");

      const PointCloudXYZI cloud = propagator.process(measure, state);
      output << "package " << package_index << " " << measure.lidar_beg_time_s
             << " " << lidar_end_time_s << "\n";
      output << "imu " << measure.imu.size() << "\n";
      for (const auto &sample : measure.imu)
        output << "imu_sample " << sample.timestamp_s << "\n";
      write_state(output, "before", before);
      write_cloud(output, "input", measure.lidar);
      write_state(output, "after", state);
      write_cloud(output, "output", cloud);
      output << "end_package\n";
    }
    throw std::runtime_error("frontend trace is missing END");
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
