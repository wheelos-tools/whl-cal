/*
 * Complete ROS-free GRIL frontend executable.
 * Copyright (C) 2026 whl-cal contributors.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License, version 2.
 * See ../LICENSE.
 */

#include <Gril_Calib/FullFrontend.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void usage(const char *program) {
  std::cerr
      << "Usage: " << program
      << " --input DATASET --config CONFIG --output RESULT"
         " --trace TRACE --batch-trace BATCH_TRACE"
         " --batch-executable EXECUTABLE --batch-config CONFIG"
         " [--gap-policy golden|reset --forward-gap-s SECONDS]"
         " [--ikd-rebuild async|sync]"
         " [--cut-mode equal_count|equal_time --min-cut-points N]"
         " [--cut-frame-num N]"
         " [--odom-ground-mode golden|disabled --min-effective-features N]"
         " [--ahrs-imu-hz HZ --ahrs-dt-mode package_reset|continuous]"
         " [--motion-start-min-speed-mps V --motion-start-min-omega-rad-s W]"
         " [--min-accum-duration-s S --max-accum-duration-s S]"
         " [--eof-policy fail|calibrate]\n";
}

double parse_double(const std::string &text, const std::string &name) {
  std::size_t consumed = 0;
  const double value = std::stod(text, &consumed);
  if (consumed != text.size() || !std::isfinite(value))
    throw std::runtime_error("invalid " + name);
  return value;
}

int parse_int(const std::string &text, const std::string &name) {
  std::size_t consumed = 0;
  const int value = std::stoi(text, &consumed);
  if (consumed != text.size())
    throw std::runtime_error("invalid " + name);
  return value;
}

} // namespace

int main(int argc, char **argv) {
  try {
    FullFrontendRunConfig config;
    for (int index = 1; index < argc; ++index) {
      const std::string argument(argv[index]);
      if (index + 1 >= argc) {
        usage(argv[0]);
        return 2;
      }
      const std::string value(argv[++index]);
      if (argument == "--input")
        config.input_path = value;
      else if (argument == "--config")
        config.config_path = value;
      else if (argument == "--output")
        config.result_path = value;
      else if (argument == "--trace")
        config.trace_path = value;
      else if (argument == "--batch-trace")
        config.batch_trace_path = value;
      else if (argument == "--batch-executable")
        config.batch_executable_path = value;
      else if (argument == "--batch-config")
        config.batch_config_path = value;
      else if (argument == "--gap-policy") {
        if (value == "golden")
          config.forward_gap_policy = ForwardGapPolicy::GoldenEquivalence;
        else if (value == "reset")
          config.forward_gap_policy = ForwardGapPolicy::Reset;
        else
          throw std::runtime_error("gap policy must be golden or reset");
      } else if (argument == "--forward-gap-s") {
        config.forward_gap_s = parse_double(value, "forward gap");
      } else if (argument == "--ikd-rebuild") {
        if (value == "async")
          config.sync_ikd_rebuild = false;
        else if (value == "sync")
          config.sync_ikd_rebuild = true;
        else
          throw std::runtime_error("ikd rebuild mode must be async or sync");
      } else if (argument == "--cut-mode") {
        if (value == "equal_count")
          config.cut_mode = VelodyneCutMode::EqualPointCount;
        else if (value == "equal_time")
          config.cut_mode = VelodyneCutMode::EqualTime;
        else
          throw std::runtime_error(
              "cut mode must be equal_count or equal_time");
      } else if (argument == "--min-cut-points") {
        config.min_cut_points = parse_int(value, "min cut points");
      } else if (argument == "--cut-frame-num") {
        config.cut_frame_num_override = parse_int(value, "cut frame num");
      } else if (argument == "--odom-ground-mode") {
        if (value == "golden")
          config.disable_odom_ground_constraint = false;
        else if (value == "disabled")
          config.disable_odom_ground_constraint = true;
        else
          throw std::runtime_error(
              "odom ground mode must be golden or disabled");
      } else if (argument == "--min-effective-features") {
        config.min_effective_features =
            parse_int(value, "min effective features");
      } else if (argument == "--ahrs-imu-hz") {
        const int hz = parse_int(value, "ahrs imu hz");
        if (hz <= 0)
          throw std::runtime_error("ahrs imu hz must be positive");
        config.ahrs_imu_hz = static_cast<unsigned int>(hz);
      } else if (argument == "--ahrs-dt-mode") {
        if (value == "package_reset")
          config.ahrs_preserve_sample_dt = false;
        else if (value == "continuous")
          config.ahrs_preserve_sample_dt = true;
        else
          throw std::runtime_error(
              "ahrs dt mode must be package_reset or continuous");
      } else if (argument == "--motion-start-min-speed-mps") {
        config.motion_start_min_speed_mps =
            parse_double(value, "motion start min speed");
      } else if (argument == "--motion-start-min-omega-rad-s") {
        config.motion_start_min_imu_omega_rad_s =
            parse_double(value, "motion start min omega");
      } else if (argument == "--min-accum-duration-s") {
        config.min_accum_duration_s =
            parse_double(value, "min accum duration");
      } else if (argument == "--max-accum-duration-s") {
        config.max_accum_duration_s =
            parse_double(value, "max accum duration");
      } else if (argument == "--eof-policy") {
        if (value == "fail")
          config.calibrate_on_eof = false;
        else if (value == "calibrate")
          config.calibrate_on_eof = true;
        else
          throw std::runtime_error("eof policy must be fail or calibrate");
      } else {
        usage(argv[0]);
        return 2;
      }
    }
    run_full_frontend(config);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "gril_native_full_frontend: " << error.what() << "\n";
    return 1;
  }
}
