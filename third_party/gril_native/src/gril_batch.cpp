/*
 * ROS-free GRIL batch trace runner.
 * Copyright (C) 2026 whl-cal contributors.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License, version 2.
 * See ../LICENSE.
 */

#include <Gril_Calib/BatchTrace.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

struct NativeConfig {
  double data_accum_length;
  double x_accumulate;
  double y_accumulate;
  double z_accumulate;
  double svd_threshold;
  double imu_sensor_height;
  double trans_IL_x;
  double trans_IL_y;
  double trans_IL_z;
  double bound_th;
  bool set_boundary;
  bool verbose;
  double gyro_factor;
  double acc_factor;
  double ground_factor;
  QD rotation_LI_seed = QD::Identity();
  bool rotation_LI_seed_provided = false;
  double max_xcorr_lag_s = -1.0;
  double max_time_lag_2_s = -1.0;
  int rot_init_max_iters = 50;
  bool use_imu_angular_in_acc_cost = false;
  bool skip_secondary_acc_norm = false;
  int tail_trim_states = 20;
  bool normalize_butterworth_gain = false;
  bool world_frame_gravity_in_acc_cost = false;
  bool ground_plane_imu_to_ground = false;
  double acc_bias_bound_m_s2 = 0.01;
};

double parse_double(const std::string &text, const std::string &key) {
  std::size_t consumed = 0;
  const double value = std::stod(text, &consumed);
  if (consumed != text.size() || !std::isfinite(value))
    throw std::runtime_error("invalid value for " + key);
  return value;
}

int parse_int(const std::string &text, const std::string &key) {
  std::size_t consumed = 0;
  const int value = std::stoi(text, &consumed);
  if (consumed != text.size())
    throw std::runtime_error("invalid integer for " + key);
  return value;
}

bool parse_bool(const std::string &text, const std::string &key) {
  if (text == "0" || text == "false")
    return false;
  if (text == "1" || text == "true")
    return true;
  throw std::runtime_error("invalid boolean for " + key);
}

NativeConfig read_config(const std::string &path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("cannot open native config: " + path);

  std::map<std::string, std::string> values;
  std::string raw;
  int line_number = 0;
  while (std::getline(input, raw)) {
    line_number++;
    const std::size_t comment = raw.find('#');
    if (comment != std::string::npos)
      raw.erase(comment);
    std::istringstream line(raw);
    std::string key;
    std::string value;
    if (!(line >> key))
      continue;
    if (!(line >> value))
      throw std::runtime_error("missing config value at line " +
                               std::to_string(line_number));
    std::string extra;
    if (line >> extra)
      throw std::runtime_error("extra config field at line " +
                               std::to_string(line_number));
    if (!values.emplace(key, value).second)
      throw std::runtime_error("duplicate config key: " + key);
  }

  const auto take = [&](const char *key) {
    const auto it = values.find(key);
    if (it == values.end())
      throw std::runtime_error(std::string("missing config key: ") + key);
    const std::string value = it->second;
    values.erase(it);
    return value;
  };

  NativeConfig config;
  config.data_accum_length =
      parse_double(take("data_accum_length"), "data_accum_length");
  config.x_accumulate = parse_double(take("x_accumulate"), "x_accumulate");
  config.y_accumulate = parse_double(take("y_accumulate"), "y_accumulate");
  config.z_accumulate = parse_double(take("z_accumulate"), "z_accumulate");
  config.svd_threshold = parse_double(take("svd_threshold"), "svd_threshold");
  config.imu_sensor_height =
      parse_double(take("imu_sensor_height"), "imu_sensor_height");
  config.trans_IL_x = parse_double(take("trans_IL_x"), "trans_IL_x");
  config.trans_IL_y = parse_double(take("trans_IL_y"), "trans_IL_y");
  config.trans_IL_z = parse_double(take("trans_IL_z"), "trans_IL_z");
  config.bound_th = parse_double(take("bound_th"), "bound_th");
  config.set_boundary = parse_bool(take("set_boundary"), "set_boundary");
  config.verbose = parse_bool(take("verbose"), "verbose");
  config.gyro_factor = parse_double(take("gyro_factor"), "gyro_factor");
  config.acc_factor = parse_double(take("acc_factor"), "acc_factor");
  config.ground_factor = parse_double(take("ground_factor"), "ground_factor");
  if (values.count("rotation_LI_qx") || values.count("rotation_LI_qy") ||
      values.count("rotation_LI_qz") || values.count("rotation_LI_qw")) {
    const double x = parse_double(take("rotation_LI_qx"), "rotation_LI_qx");
    const double y = parse_double(take("rotation_LI_qy"), "rotation_LI_qy");
    const double z = parse_double(take("rotation_LI_qz"), "rotation_LI_qz");
    const double w = parse_double(take("rotation_LI_qw"), "rotation_LI_qw");
    config.rotation_LI_seed = QD(w, x, y, z);
    config.rotation_LI_seed_provided = true;
  }
  if (values.count("max_xcorr_lag_s"))
    config.max_xcorr_lag_s =
        parse_double(take("max_xcorr_lag_s"), "max_xcorr_lag_s");
  if (values.count("max_time_lag_2_s"))
    config.max_time_lag_2_s =
        parse_double(take("max_time_lag_2_s"), "max_time_lag_2_s");
  if (values.count("rot_init_max_iters"))
    config.rot_init_max_iters =
        parse_int(take("rot_init_max_iters"), "rot_init_max_iters");
  if (values.count("use_imu_angular_in_acc_cost"))
    config.use_imu_angular_in_acc_cost = parse_bool(
        take("use_imu_angular_in_acc_cost"), "use_imu_angular_in_acc_cost");
  if (values.count("skip_secondary_acc_norm"))
    config.skip_secondary_acc_norm =
        parse_bool(take("skip_secondary_acc_norm"), "skip_secondary_acc_norm");
  if (values.count("tail_trim_states"))
    config.tail_trim_states =
        parse_int(take("tail_trim_states"), "tail_trim_states");
  if (values.count("normalize_butterworth_gain"))
    config.normalize_butterworth_gain = parse_bool(
        take("normalize_butterworth_gain"), "normalize_butterworth_gain");
  if (values.count("world_frame_gravity_in_acc_cost"))
    config.world_frame_gravity_in_acc_cost =
        parse_bool(take("world_frame_gravity_in_acc_cost"),
                   "world_frame_gravity_in_acc_cost");
  if (values.count("ground_plane_imu_to_ground"))
    config.ground_plane_imu_to_ground = parse_bool(
        take("ground_plane_imu_to_ground"), "ground_plane_imu_to_ground");
  if (values.count("acc_bias_bound_m_s2")) {
    config.acc_bias_bound_m_s2 =
        parse_double(take("acc_bias_bound_m_s2"), "acc_bias_bound_m_s2");
    if (!(config.acc_bias_bound_m_s2 > 0.0))
      throw std::runtime_error("acc_bias_bound_m_s2 must be positive");
  }
  if (!values.empty())
    throw std::runtime_error("unknown config key: " + values.begin()->first);
  return config;
}

void write_result(const std::string &path, Gril_Calib &calibration) {
  std::ofstream output(path);
  if (!output)
    throw std::runtime_error("cannot create result file: " + path);

  const M3D rotation = calibration.get_R_LI();
  const V3D translation = calibration.get_T_LI();
  const V3D gyro_bias = calibration.get_gyro_bias();
  const V3D acc_bias = calibration.get_acc_bias();
  output << "LiDAR-IMU calibration result:\n";
  output.setf(std::ios::fixed);
  output << std::setprecision(6) << "Rotation LiDAR to IMU (degree)     = "
         << (RotMtoEuler(rotation) * 57.3).transpose() << "\n"
         << "Translation LiDAR to IMU (meter)   = " << translation.transpose()
         << "\n"
         << "Time Lag IMU to LiDAR (second)     = "
         << calibration.get_time_result() << "\n"
         << "Bias of Gyroscope  (rad/s)         = " << gyro_bias.transpose()
         << "\n"
         << "Bias of Accelerometer (meters/s^2) = " << acc_bias.transpose()
         << "\n\n";

  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform.block<3, 3>(0, 0) = rotation;
  transform.block<3, 1>(0, 3) = translation;
  output << "Homogeneous Transformation Matrix from LiDAR frmae L "
            "to IMU frame I:\n"
         << transform << "\n\n\n";
}

void usage(const char *program) {
  std::cerr << "Usage: " << program
            << " --trace TRACE --config CONFIG --output "
               "GRIL_Calib_result.txt\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::string trace_path;
    std::string config_path;
    std::string output_path;
    for (int i = 1; i < argc; i++) {
      const std::string argument(argv[i]);
      if ((argument == "--trace" || argument == "--config" ||
           argument == "--output") &&
          i + 1 < argc) {
        const std::string value(argv[++i]);
        if (argument == "--trace")
          trace_path = value;
        else if (argument == "--config")
          config_path = value;
        else
          output_path = value;
      } else {
        usage(argv[0]);
        return 2;
      }
    }
    if (trace_path.empty() || config_path.empty() || output_path.empty()) {
      usage(argv[0]);
      return 2;
    }

    BatchTrace trace = read_batch_trace_file(trace_path);
    std::string reason;
    if (!validate_batch_trace_for_calibration(trace, &reason))
      throw std::runtime_error(
          "batch trace is insufficient for LI_Calibration: " + reason);
    const NativeConfig config = read_config(config_path);

    Gril_Calib calibration;
    calibration.data_accum_length = config.data_accum_length;
    calibration.x_accumulate = config.x_accumulate;
    calibration.y_accumulate = config.y_accumulate;
    calibration.z_accumulate = config.z_accumulate;
    calibration.svd_threshold = config.svd_threshold;
    calibration.imu_sensor_height = config.imu_sensor_height;
    calibration.trans_IL_x = config.trans_IL_x;
    calibration.trans_IL_y = config.trans_IL_y;
    calibration.trans_IL_z = config.trans_IL_z;
    calibration.bound_th = config.bound_th;
    calibration.set_boundary = config.set_boundary;
    calibration.verbose = config.verbose;
    calibration.max_xcorr_lag_s = config.max_xcorr_lag_s;
    calibration.max_time_lag_2_s = config.max_time_lag_2_s;
    calibration.rot_init_max_iters = config.rot_init_max_iters;
    calibration.use_imu_angular_in_acc_cost =
        config.use_imu_angular_in_acc_cost;
    calibration.skip_secondary_acc_norm = config.skip_secondary_acc_norm;
    calibration.tail_trim_states = config.tail_trim_states;
    calibration.normalize_butterworth_gain = config.normalize_butterworth_gain;
    calibration.world_frame_gravity_in_acc_cost =
        config.world_frame_gravity_in_acc_cost;
    calibration.ground_plane_imu_to_ground = config.ground_plane_imu_to_ground;
    calibration.acc_bias_bound_m_s2 = config.acc_bias_bound_m_s2;
    if (config.rotation_LI_seed_provided) {
      calibration.set_rotation_initial_guess(config.rotation_LI_seed);
      std::cout << "[Rotation initialization] LiDAR to IMU quaternion xyzw = "
                << std::setprecision(17)
                << config.rotation_LI_seed.coeffs().transpose() << "\n";
    }
    GYRO_FACTOR_ = config.gyro_factor;
    ACC_FACTOR_ = config.acc_factor;
    GROUND_FACTOR_ = config.ground_factor;

    std::deque<QD> lidar_ground;
    std::deque<QD> imu_ground;
    std::deque<V3D> normals;
    std::deque<double> distances;
    for (const auto &constraint : trace.ground_constraints) {
      lidar_ground.push_back(constraint.lidar_wrt_ground);
      imu_ground.push_back(constraint.imu_wrt_ground);
      normals.push_back(constraint.normal_lidar);
      distances.push_back(constraint.distance_lidar);
    }
    calibration.set_batch_inputs(trace.normalized_imu_states,
                                 trace.lidar_states, lidar_ground, imu_ground,
                                 normals, distances);

    calibration.LI_Calibration(trace.orig_odom_freq, trace.cut_frame_num,
                               trace.timediff_imu_wrt_lidar,
                               trace.move_start_time);
    write_result(output_path, calibration);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "gril_native_batch: " << error.what() << "\n";
    return 1;
  }
}
