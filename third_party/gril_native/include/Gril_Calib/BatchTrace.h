/*
 * Versioned GRIL batch trace I/O.
 * Copyright (C) 2026 whl-cal contributors.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License, version 2.
 * See ../../LICENSE.
 */

#ifndef GRIL_NATIVE_BATCH_TRACE_H
#define GRIL_NATIVE_BATCH_TRACE_H

#include "Gril_Calib.h"

#include <deque>
#include <iosfwd>
#include <string>

struct GroundConstraint {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  QD lidar_wrt_ground = QD::Identity();
  QD imu_wrt_ground = QD::Identity();
  V3D normal_lidar = V3D::Zero();
  double distance_lidar = 0.0;
};

struct BatchTrace {
  int orig_odom_freq = 0;
  int cut_frame_num = 0;
  double timediff_imu_wrt_lidar = 0.0;
  double move_start_time = 0.0;
  std::deque<CalibState> normalized_imu_states;
  std::deque<CalibState> lidar_states;
  std::deque<GroundConstraint> ground_constraints;
};

BatchTrace read_batch_trace(std::istream &input);
BatchTrace read_batch_trace_file(const std::string &path);
void write_batch_trace(std::ostream &output, const BatchTrace &trace);
void write_batch_trace_file(const std::string &path, const BatchTrace &trace);

bool validate_batch_trace_for_calibration(const BatchTrace &trace,
                                          std::string *reason);

#endif
