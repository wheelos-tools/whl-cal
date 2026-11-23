/*
 * GRIL batch trace contract tests.
 * Copyright (C) 2026 whl-cal contributors.
 * GPL-2.0-only; see ../LICENSE.
 */

#include <Gril_Calib/BatchTrace.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

BatchTrace sample_trace() {
  BatchTrace trace;
  trace.orig_odom_freq = 10;
  trace.cut_frame_num = 2;
  trace.timediff_imu_wrt_lidar = -0.012345678901234;
  trace.move_start_time = 42.25;
  for (int index = 0; index < 2; index++) {
    CalibState imu;
    imu.timeStamp = 40.0 + index * 0.01;
    imu.rot_end = Eigen::AngleAxisd(0.01 * index, V3D::UnitZ()).matrix();
    imu.pos_end = V3D(index, index + 1, index + 2);
    imu.ang_vel = V3D(0.1, 0.2, 0.3 + index);
    imu.linear_vel = V3D(1.1, 1.2, 1.3);
    imu.ang_acc = V3D(2.1, 2.2, 2.3);
    imu.linear_acc = V3D(0.0, 0.0, 9.81 + index);
    trace.normalized_imu_states.push_back(imu);

    CalibState lidar(imu);
    lidar.timeStamp += 0.001;
    trace.lidar_states.push_back(lidar);

    GroundConstraint ground;
    ground.lidar_wrt_ground = QD(Eigen::AngleAxisd(0.02 * index, V3D::UnitX()));
    ground.imu_wrt_ground = QD(Eigen::AngleAxisd(0.03 * index, V3D::UnitY()));
    ground.normal_lidar = V3D(0.0, 0.0, 1.0);
    ground.distance_lidar = 1.25 + index;
    trace.ground_constraints.push_back(ground);
  }
  return trace;
}

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

void round_trip() {
  const BatchTrace original = sample_trace();
  std::ostringstream output;
  write_batch_trace(output, original);
  std::istringstream input(output.str());
  const BatchTrace parsed = read_batch_trace(input);

  require(parsed.orig_odom_freq == original.orig_odom_freq,
          "orig_odom_freq mismatch");
  require(parsed.cut_frame_num == original.cut_frame_num,
          "cut_frame_num mismatch");
  require(parsed.timediff_imu_wrt_lidar == original.timediff_imu_wrt_lidar,
          "time diff mismatch");
  require(parsed.move_start_time == original.move_start_time,
          "move_start_time mismatch");
  require(parsed.normalized_imu_states.size() == 2, "IMU count mismatch");
  require(parsed.lidar_states.size() == 2, "LiDAR count mismatch");
  require(parsed.ground_constraints.size() == 2, "ground count mismatch");
  require(parsed.normalized_imu_states[1].linear_acc.isApprox(
              original.normalized_imu_states[1].linear_acc, 0.0),
          "IMU state mismatch");
  require(parsed.lidar_states[1].rot_end.isApprox(
              original.lidar_states[1].rot_end, 0.0),
          "LiDAR state mismatch");
  require(parsed.ground_constraints[1].lidar_wrt_ground.coeffs().isApprox(
              original.ground_constraints[1].lidar_wrt_ground.coeffs(), 0.0),
          "ground quaternion mismatch");
}

void rejection() {
  {
    std::istringstream malformed("GRIL_BATCH_TRACE 2\n");
    bool rejected = false;
    try {
      (void)read_batch_trace(malformed);
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    require(rejected, "unsupported version was accepted");
  }
  {
    BatchTrace trace = sample_trace();
    trace.ground_constraints.pop_back();
    std::ostringstream output;
    write_batch_trace(output, trace);
    std::istringstream input(output.str());
    bool rejected = false;
    try {
      (void)read_batch_trace(input);
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    require(rejected, "mismatched paired counts were accepted");
  }
  {
    BatchTrace trace = sample_trace();
    std::string reason;
    require(!validate_batch_trace_for_calibration(trace, &reason),
            "insufficient trace was accepted for calibration");
    require(!reason.empty(), "insufficient rejection lacks reason");
  }
}

void rotation_seed() {
  const QD seed(Eigen::AngleAxisd(136.0 * std::acos(-1.0) / 180.0,
                                V3D::UnitZ()));
  std::deque<CalibState> states;
  for (int index = 0; index < 5; ++index) {
    CalibState state;
    state.ang_vel = V3D(0.0, 0.0, 0.1 * (index + 1));
    states.push_back(state);
  }
  Gril_Calib calibration;
  calibration.set_IMU_state(states);
  calibration.set_Lidar_state(states);
  calibration.solve_Rotation_only();
  require(calibration.get_R_LI().isApprox(M3D::Identity(), 1e-12),
          "default rotation initialization changed");
  calibration.set_rotation_initial_guess(seed);
  calibration.solve_Rotation_only();
  require(calibration.get_R_LI().isApprox(seed.matrix(), 1e-12),
          "yaw-only data did not retain the supplied rotation seed");
  for (const QD &invalid :
       {QD(0.0, 0.0, 0.0, 0.0), QD(2.0, 0.0, 0.0, 0.0),
        QD(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0)}) {
    bool rejected = false;
    try {
      calibration.set_rotation_initial_guess(invalid);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected, "invalid rotation initial guess was accepted");
  }
}

void butterworth_unity_gain() {
  std::deque<CalibState> states;
  for (int i = 0; i < 200; ++i) {
    CalibState s;
    s.timeStamp = 100.0 + i * (1.0 / 30.0);
    s.linear_acc = V3D(0.0, 0.0, 9.81);
    s.ang_vel = V3D(0.0, 0.0, 0.2);
    states.push_back(s);
  }

  Gril_Calib legacy;
  std::deque<CalibState> out_legacy;
  legacy.zero_phase_filt(states, out_legacy);
  const double legacy_mid_z = out_legacy[100].linear_acc.z();
  require(std::abs(legacy_mid_z - 9.81) > 0.35,
          "expected unnormalized Butterworth DC gain inflation by default");

  Gril_Calib unity;
  unity.normalize_butterworth_gain = true;
  std::deque<CalibState> out_unity;
  unity.zero_phase_filt(states, out_unity);
  const double unity_mid_z = out_unity[100].linear_acc.z();
  require(std::abs(unity_mid_z - 9.81) < 1e-10,
          "normalized Butterworth DC gain must preserve constant signal");
}

void gravity_and_ground_frames() {
  const M3D R_LI =
      Eigen::AngleAxisd(0.5 * std::acos(-1.0), V3D::UnitZ()).matrix();
  const M3D R_IL = R_LI.transpose();
  const QD q_IL(R_IL);
  const double q_IL_arr[4] = {q_IL.w(), q_IL.x(), q_IL.y(), q_IL.z()};
  const double b_a[3] = {0.0, 0.0, 0.0};
  const double trans[3] = {0.0, 0.0, 0.0};

  // Tilted ground normal in current LiDAR frame + 180 deg vehicle yaw in L0.
  const V3D normal_L = V3D(0.02, -0.01, 1.0).normalized();
  const QD q_GL = QD::FromTwoVectors(V3D::UnitZ(), normal_L);
  const M3D R_LL0 =
      Eigen::AngleAxisd(std::acos(-1.0), V3D::UnitZ()).matrix();
  CalibState lidar_state;
  lidar_state.rot_end = R_LL0;
  lidar_state.linear_acc = V3D::Zero();
  const V3D imu_acc = R_LI * (q_GL.toRotationMatrix() * V3D(0.0, 0.0, 9.81));

  double res_default[3] = {0.0, 0.0, 0.0};
  Linear_acc_Rot_Cost_without_Gravity cost_default(
      lidar_state, imu_acc, q_GL, false, V3D::Zero(), V3D::Zero(), false);
  cost_default(q_IL_arr, b_a, trans, res_default);
  require(V3D(res_default[0], res_default[1], res_default[2]).norm() > 0.2,
          "expected default gravity frame mismatch under rotated R_LL0");

  double res_world[3] = {0.0, 0.0, 0.0};
  Linear_acc_Rot_Cost_without_Gravity cost_world(
      lidar_state, imu_acc, q_GL, false, V3D::Zero(), V3D::Zero(), true);
  cost_world(q_IL_arr, b_a, trans, res_world);
  require(V3D(res_world[0], res_world[1], res_world[2]).norm() < 1e-10,
          "world_frame_gravity must vanish under rotated R_LL0");

  // Ground plane: R_GI maps I -> G so R_GI^T * e3 = R_LI * normal_L.
  const V3D normal_I = R_LI * normal_L;
  const QD q_IG = QD::FromTwoVectors(V3D::UnitZ(), normal_I);
  const QD q_GI = q_IG.conjugate();
  double g_default[3] = {0.0, 0.0, 0.0};
  Ground_Plane_Cost_IL gp_default(q_GL, q_GI, 1.9, 0.7, false);
  gp_default(q_IL_arr, trans, g_default);
  require(std::hypot(g_default[0], g_default[1]) > 0.01,
          "expected default Ground_Plane_Cost_IL order mismatch");

  double g_fixed[3] = {0.0, 0.0, 0.0};
  Ground_Plane_Cost_IL gp_fixed(q_GL, q_GI, 1.9, 0.7, true);
  gp_fixed(q_IL_arr, trans, g_fixed);
  require(std::hypot(g_fixed[0], g_fixed[1]) < 1e-10,
          "fixed Ground_Plane_Cost_IL order must vanish");
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "expected one test mode");
    const std::string mode(argv[1]);
    if (mode == "round_trip")
      round_trip();
    else if (mode == "rejection")
      rejection();
    else if (mode == "rotation_seed")
      rotation_seed();
    else if (mode == "butterworth_unity_gain")
      butterworth_unity_gain();
    else if (mode == "gravity_and_ground_frames")
      gravity_and_ground_frames();
    else
      throw std::runtime_error("unknown test mode");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
