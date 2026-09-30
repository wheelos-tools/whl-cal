/*
 * GRIL-Calib batch calibration core.
 * Original implementation: TaeYoung Kim and GRIL-Calib contributors.
 * Heavily adapted upstream from LI-Init by Fangcheng Zhu and contributors.
 *
 * Modified 2026-08-28 for whl-cal:
 *   - removed ROS/catkin, generated messages, common_lib, and matplotlib;
 *   - replaced sensor_msgs IMU ingestion with Eigen vectors and a timestamp;
 *   - retained upstream batch calibration math/order and validated deque fixes.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License, version 2.
 * See ../../LICENSE.
 */

#ifndef GRIL_NATIVE_GRIL_CALIB_H
#define GRIL_NATIVE_GRIL_CALIB_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <ceres/ceres.h>
#include <ceres/covariance.h>
#include <ceres/local_parameterization.h>
#include <ceres/rotation.h>

#include <cmath>
#include <cstddef>
#include <deque>
#include <fstream>
#include <string>

#define FILE_DIR(name) (std::string(std::string(ROOT_DIR) + "Log/" + (name)))
#define SKEW_SYM_MATRX(v)                                                      \
  0.0, -(v)[2], (v)[1], (v)[2], 0.0, -(v)[0], -(v)[1], (v)[0], 0.0

using V3D = Eigen::Vector3d;
using M3D = Eigen::Matrix3d;
using QD = Eigen::Quaterniond;

constexpr double G_m_s2 = 9.81;
extern const V3D STD_GRAV;
extern double GYRO_FACTOR_;
extern double ACC_FACTOR_;
extern double GROUND_FACTOR_;

template <typename T>
Eigen::Matrix<T, 3, 1> RotMtoEuler(const Eigen::Matrix<T, 3, 3> &rot) {
  const T sy = std::sqrt(rot(0, 0) * rot(0, 0) + rot(1, 0) * rot(1, 0));
  const bool singular = sy < T(1e-6);
  T x;
  T y;
  T z;
  if (!singular) {
    x = std::atan2(rot(2, 1), rot(2, 2));
    y = std::atan2(-rot(2, 0), sy);
    z = std::atan2(rot(1, 0), rot(0, 0));
  } else {
    x = std::atan2(-rot(1, 2), rot(1, 1));
    y = std::atan2(-rot(2, 0), sy);
    z = T(0);
  }
  return Eigen::Matrix<T, 3, 1>(x, y, z);
}

struct CalibState {
  M3D rot_end;
  V3D pos_end;
  V3D ang_vel;
  V3D linear_vel;
  V3D ang_acc;
  V3D linear_acc;
  double timeStamp;

  CalibState() {
    rot_end = M3D::Identity();
    pos_end = V3D::Zero();
    ang_vel = V3D::Zero();
    linear_vel = V3D::Zero();
    ang_acc = V3D::Zero();
    linear_acc = V3D::Zero();
    timeStamp = 0.0;
  }

  CalibState(const CalibState &other) {
    rot_end = other.rot_end;
    pos_end = other.pos_end;
    ang_vel = other.ang_vel;
    ang_acc = other.ang_acc;
    linear_vel = other.linear_vel;
    linear_acc = other.linear_acc;
    timeStamp = other.timeStamp;
  }

  CalibState operator*(const double &coeff) {
    CalibState state;
    state.ang_vel = ang_vel * coeff;
    state.ang_acc = ang_acc * coeff;
    state.linear_vel = linear_vel * coeff;
    state.linear_acc = linear_acc * coeff;
    return state;
  }

  CalibState &operator+=(const CalibState &other) {
    ang_vel += other.ang_vel;
    ang_acc += other.ang_acc;
    linear_vel += other.linear_vel;
    linear_acc += other.linear_acc;
    return *this;
  }

  CalibState &operator-=(const CalibState &other) {
    ang_vel -= other.ang_vel;
    ang_acc -= other.ang_acc;
    linear_vel -= other.linear_vel;
    linear_acc -= other.linear_acc;
    return *this;
  }

  CalibState &operator=(const CalibState &other) {
    ang_vel = other.ang_vel;
    ang_acc = other.ang_acc;
    linear_vel = other.linear_vel;
    linear_acc = other.linear_acc;
    return *this;
  }
};

struct Angular_Vel_Cost_only_Rot {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Angular_Vel_Cost_only_Rot(V3D imu_ang_vel, V3D lidar_ang_vel)
      : IMU_ang_vel(imu_ang_vel), Lidar_ang_vel(lidar_ang_vel) {}

  template <typename T> bool operator()(const T *q, T *residual) const {
    Eigen::Matrix<T, 3, 1> IMU_ang_vel_T = IMU_ang_vel.cast<T>();
    Eigen::Matrix<T, 3, 1> Lidar_ang_vel_T = Lidar_ang_vel.cast<T>();
    Eigen::Quaternion<T> q_LI{q[0], q[1], q[2], q[3]};
    Eigen::Matrix<T, 3, 3> R_LI = q_LI.toRotationMatrix();
    Eigen::Matrix<T, 3, 1> resi = R_LI * Lidar_ang_vel_T - IMU_ang_vel_T;
    residual[0] = resi[0];
    residual[1] = resi[1];
    residual[2] = resi[2];
    return true;
  }

  static ceres::CostFunction *Create(const V3D imu, const V3D lidar) {
    return new ceres::AutoDiffCostFunction<Angular_Vel_Cost_only_Rot, 3, 4>(
        new Angular_Vel_Cost_only_Rot(imu, lidar));
  }

  V3D IMU_ang_vel;
  V3D Lidar_ang_vel;
};

struct Angular_Vel_IL_Cost {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Angular_Vel_IL_Cost(V3D imu_ang_vel, V3D imu_ang_acc, V3D lidar_ang_vel,
                      double delta_t)
      : IMU_ang_vel(imu_ang_vel), IMU_ang_acc(imu_ang_acc),
        Lidar_ang_vel(lidar_ang_vel), deltaT_LI(delta_t) {}

  template <typename T>
  bool operator()(const T *q, const T *b_g, const T *t, T *residual) const {
    Eigen::Matrix<T, 3, 1> IMU_ang_vel_T = IMU_ang_vel.cast<T>();
    Eigen::Matrix<T, 3, 1> IMU_ang_acc_T = IMU_ang_acc.cast<T>();
    Eigen::Matrix<T, 3, 1> Lidar_ang_vel_T = Lidar_ang_vel.cast<T>();
    T deltaT_LI_T{deltaT_LI};
    Eigen::Quaternion<T> q_IL{q[0], q[1], q[2], q[3]};
    Eigen::Matrix<T, 3, 3> R_IL = q_IL.toRotationMatrix();
    Eigen::Matrix<T, 3, 1> bias_g{b_g[0], b_g[1], b_g[2]};
    T td{t[0]};
    Eigen::Matrix<T, 3, 1> resi = R_IL.transpose() * Lidar_ang_vel_T -
                                  IMU_ang_vel_T -
                                  (deltaT_LI_T + td) * IMU_ang_acc_T + bias_g;
    residual[0] = T(GYRO_FACTOR_) * resi[0];
    residual[1] = T(GYRO_FACTOR_) * resi[1];
    residual[2] = T(GYRO_FACTOR_) * resi[2];
    return true;
  }

  static ceres::CostFunction *Create(const V3D imu_ang_vel,
                                     const V3D imu_ang_acc,
                                     const V3D lidar_ang_vel,
                                     const double delta_t) {
    return new ceres::AutoDiffCostFunction<Angular_Vel_IL_Cost, 3, 4, 3, 1>(
        new Angular_Vel_IL_Cost(imu_ang_vel, imu_ang_acc, lidar_ang_vel,
                                delta_t));
  }

  V3D IMU_ang_vel;
  V3D IMU_ang_acc;
  V3D Lidar_ang_vel;
  double deltaT_LI;
};

struct Ground_Plane_Cost_IL {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Ground_Plane_Cost_IL(QD lidar_wrt_ground, QD imu_wrt_ground,
                       double lidar_height, double imu_height)
      : Lidar_wrt_ground(lidar_wrt_ground), IMU_wrt_ground(imu_wrt_ground),
        distance_Lidar_wrt_ground(lidar_height), imu_sensor_height(imu_height) {
  }

  template <typename T>
  bool operator()(const T *q, const T *trans, T *residual) const {
    Eigen::Quaternion<T> Lidar_wrt_ground_T = Lidar_wrt_ground.cast<T>();
    Eigen::Matrix<T, 3, 3> R_GL = Lidar_wrt_ground_T.toRotationMatrix();
    Eigen::Quaternion<T> IMU_wrt_ground_T = IMU_wrt_ground.cast<T>();
    Eigen::Matrix<T, 3, 3> R_GI = IMU_wrt_ground_T.toRotationMatrix();
    T distance_Lidar_wrt_ground_T = T(distance_Lidar_wrt_ground);
    T imu_sensor_height_T = T(imu_sensor_height);

    Eigen::Quaternion<T> q_IL{q[0], q[1], q[2], q[3]};
    Eigen::Matrix<T, 3, 3> R_IL = q_IL.toRotationMatrix();
    Eigen::Matrix<T, 3, 1> T_IL{trans[0], trans[1], trans[2]};

    Eigen::Matrix<T, 3, 3> R_plane = R_IL.transpose() * R_GI.transpose() * R_GL;
    Eigen::Matrix<T, 3, 1> e3 = Eigen::Matrix<T, 3, 1>::UnitZ();
    Eigen::Matrix<T, 3, 1> resi_plane = R_plane * e3;

    Eigen::Matrix<T, 3, 1> imu_height_vec = imu_sensor_height_T * e3;
    Eigen::Matrix<T, 3, 1> lidar_height_vec = distance_Lidar_wrt_ground_T * e3;
    Eigen::Matrix<T, 3, 1> tmp_distance =
        R_IL * R_GI * imu_height_vec - R_GL * lidar_height_vec;
    T resi_distance = T_IL[2] - tmp_distance[2];

    residual[0] = T(GROUND_FACTOR_) * resi_plane[0];
    residual[1] = T(GROUND_FACTOR_) * resi_plane[1];
    residual[2] = T(GROUND_FACTOR_) * resi_distance;
    return true;
  }

  static ceres::CostFunction *Create(const QD lidar_wrt_ground,
                                     const QD imu_wrt_ground,
                                     const double lidar_height,
                                     const double imu_height) {
    return new ceres::AutoDiffCostFunction<Ground_Plane_Cost_IL, 3, 4, 3>(
        new Ground_Plane_Cost_IL(lidar_wrt_ground, imu_wrt_ground, lidar_height,
                                 imu_height));
  }

  QD Lidar_wrt_ground;
  QD IMU_wrt_ground;
  double distance_Lidar_wrt_ground;
  double imu_sensor_height;
};

struct Linear_acc_Rot_Cost_without_Gravity {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Linear_acc_Rot_Cost_without_Gravity(CalibState lidar_state,
                                      V3D imu_linear_acc, QD lidar_wrt_ground)
      : LidarState(lidar_state), IMU_linear_acc(imu_linear_acc),
        Lidar_wrt_ground(lidar_wrt_ground) {}

  template <typename T>
  bool operator()(const T *q, const T *b_a, const T *trans, T *residual) const {
    Eigen::Matrix<T, 3, 3> R_LL0_T = LidarState.rot_end.cast<T>();
    Eigen::Matrix<T, 3, 1> IMU_linear_acc_T = IMU_linear_acc.cast<T>();
    Eigen::Matrix<T, 3, 1> Lidar_linear_acc_T = LidarState.linear_acc.cast<T>();
    Eigen::Quaternion<T> Lidar_wrt_ground_T = Lidar_wrt_ground.cast<T>();
    Eigen::Matrix<T, 3, 3> R_GL = Lidar_wrt_ground_T.toRotationMatrix();

    Eigen::Matrix<T, 3, 1> bias_aL{b_a[0], b_a[1], b_a[2]};
    Eigen::Matrix<T, 3, 1> T_IL{trans[0], trans[1], trans[2]};
    Eigen::Quaternion<T> q_IL{q[0], q[1], q[2], q[3]};
    Eigen::Matrix<T, 3, 3> R_IL = q_IL.toRotationMatrix();

    M3D Lidar_omg_SKEW;
    M3D Lidar_angacc_SKEW;
    Lidar_omg_SKEW << SKEW_SYM_MATRX(LidarState.ang_vel);
    Lidar_angacc_SKEW << SKEW_SYM_MATRX(LidarState.ang_acc);
    M3D Jacob_trans = Lidar_omg_SKEW * Lidar_omg_SKEW + Lidar_angacc_SKEW;
    Eigen::Matrix<T, 3, 3> Jacob_trans_T = Jacob_trans.cast<T>();

    Eigen::Matrix<T, 3, 1> resi =
        R_LL0_T * R_IL * IMU_linear_acc_T - R_LL0_T * bias_aL +
        R_GL * STD_GRAV - Lidar_linear_acc_T - R_LL0_T * Jacob_trans_T * T_IL;
    residual[0] = T(ACC_FACTOR_) * resi[0];
    residual[1] = T(ACC_FACTOR_) * resi[1];
    residual[2] = T(ACC_FACTOR_) * resi[2];
    return true;
  }

  static ceres::CostFunction *Create(const CalibState lidar_state,
                                     const V3D imu_acc,
                                     const QD lidar_wrt_ground) {
    return new ceres::AutoDiffCostFunction<Linear_acc_Rot_Cost_without_Gravity,
                                           3, 4, 3, 3>(
        new Linear_acc_Rot_Cost_without_Gravity(lidar_state, imu_acc,
                                                lidar_wrt_ground));
  }

  CalibState LidarState;
  V3D IMU_linear_acc;
  QD Lidar_wrt_ground;
};

class Gril_Calib {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  std::ofstream fout_LiDAR_meas;
  std::ofstream fout_IMU_meas;
  std::ofstream fout_before_filt_IMU;
  std::ofstream fout_before_filt_Lidar;
  std::ofstream fout_acc_cost;
  std::ofstream fout_after_rot;
  std::ofstream fout_LiDAR_ang_vel;
  std::ofstream fout_IMU_ang_vel;
  std::ofstream fout_Jacob_trans;
  std::ofstream fout_LiDAR_meas_after;

  Gril_Calib();
  ~Gril_Calib();

  struct Butterworth {
    double Coeff_b[7] = {0.0001, 0.0005, 0.0011, 0.0015,
                         0.0011, 0.0005, 0.0001};
    double Coeff_a[7] = {1.0,    -4.1824, 7.4916, -7.3136,
                         4.0893, -1.2385, 0.1584};
    int Coeff_size = 7;
    int extend_num = 0;
  };

  void plot_result();

  void push_ALL_IMU_CalibState(const V3D &angular_velocity,
                               const V3D &linear_acceleration,
                               const double &timestamp,
                               const double &mean_acc_norm);
  void push_IMU_CalibState(const V3D &omg, const V3D &acc,
                           const double &timestamp);
  void push_Lidar_CalibState(const M3D &rot, const V3D &pos, const V3D &omg,
                             const V3D &linear_vel, const double &timestamp);
  void push_Plane_Constraint(const QD &q_lidar, const QD &q_imu,
                             const V3D &normal_lidar,
                             const double &distance_lidar);
  void set_batch_inputs(const std::deque<CalibState> &normalized_imu_states,
                        const std::deque<CalibState> &lidar_states,
                        const std::deque<QD> &lidar_wrt_ground,
                        const std::deque<QD> &imu_wrt_ground,
                        const std::deque<V3D> &normal_lidar,
                        const std::deque<double> &distance_lidar);

  std::size_t all_imu_sample_count() const;
  std::size_t imu_state_count() const;
  std::size_t lidar_state_count() const;
  std::size_t plane_constraint_count() const;

  void downsample_interpolate_IMU(const double &move_start_time);
  void central_diff();
  void xcorr_temporal_init(const double &odom_freq);
  void IMU_time_compensate(const double &lag_time, const bool &is_discard);
  void acc_interpolate();
  void Butter_filt(const std::deque<CalibState> &signal_in,
                   std::deque<CalibState> &signal_out);
  void zero_phase_filt(const std::deque<CalibState> &signal_in,
                       std::deque<CalibState> &signal_out);
  void cut_sequence_tail();
  void set_IMU_state(const std::deque<CalibState> &imu_states);
  void set_Lidar_state(const std::deque<CalibState> &lidar_states);
  void set_states_2nd_filter(const std::deque<CalibState> &imu_states,
                             const std::deque<CalibState> &lidar_states);
  void solve_Rot_Trans_calib(double &timediff_imu_wrt_lidar,
                             const double &imu_height);
  void normalize_acc(std::deque<CalibState> &signal_in);
  void align_Group(const std::deque<CalibState> &imu_states,
                   std::deque<QD> &lidar_wrt_ground_states,
                   std::deque<QD> &imu_wrt_ground_states,
                   std::deque<V3D> &normal_vectors,
                   std::deque<double> &ground_distances);
  bool data_sufficiency_assess(Eigen::MatrixXd &jacobian_rot, int &frame_num,
                               V3D &lidar_omg, int &orig_odom_freq,
                               int &cut_frame_num, QD &lidar_q, QD &imu_q,
                               double &lidar_estimate_height);
  void solve_Rotation_only();
  void printProgress(double percentage, int axis_ascii);
  void clear();
  void fout_before_filter();
  void fout_check_lidar();
  void dump_batch_trace_v1(const std::string &path, const int &orig_odom_freq,
                           const int &cut_frame_num,
                           const double &timediff_imu_wrt_lidar,
                           const double &move_start_time) const;
  void LI_Calibration(int &orig_odom_freq, int &cut_frame_num,
                      double &timediff_imu_wrt_lidar,
                      const double &move_start_time);
  void print_calibration_result(double &time_L_I, M3D &R_L_I, V3D &p_L_I,
                                V3D &bias_g, V3D &bias_a, V3D gravity);

  double get_lag_time_1() { return time_lag_1; }
  double get_lag_time_2() { return time_lag_2; }
  double get_total_time_lag() { return time_delay_IMU_wtr_Lidar; }
  double get_time_result() { return time_offset_result; }
  V3D get_Grav_L0() { return Grav_L0; }
  M3D get_R_LI() { return Rot_Lidar_wrt_IMU; }
  V3D get_T_LI() { return Trans_Lidar_wrt_IMU; }
  V3D get_gyro_bias() { return gyro_bias; }
  V3D get_acc_bias() { return acc_bias; }
  void IMU_buffer_clear() { IMU_state_group_ALL.clear(); }
  std::deque<CalibState> get_IMU_state() { return IMU_state_group; }
  std::deque<CalibState> get_Lidar_state() { return Lidar_state_group; }

  double data_accum_length = 300.0;
  double x_accumulate = 0.0;
  double y_accumulate = 0.0;
  double z_accumulate = 0.0;
  double svd_threshold = 0.01;
  double imu_sensor_height = 0.0;
  double trans_IL_x = 0.0;
  double trans_IL_y = 0.0;
  double trans_IL_z = 0.0;
  double bound_th = 0.1;
  bool set_boundary = false;
  bool verbose = false;

private:
  std::deque<CalibState> IMU_state_group;
  std::deque<CalibState> Lidar_state_group;
  std::deque<CalibState> IMU_state_group_ALL;
  std::deque<QD> Lidar_wrt_ground_group;
  std::deque<QD> IMU_wrt_ground_group;
  std::deque<V3D> normal_vector_wrt_lidar_group;
  std::deque<double> distance_Lidar_wrt_ground_group;

  V3D Grav_L0;
  M3D Rot_Grav_wrt_Init_Lidar;
  M3D Rot_Lidar_wrt_IMU;
  V3D Trans_Lidar_wrt_IMU;
  V3D gyro_bias;
  V3D acc_bias;

  double time_delay_IMU_wtr_Lidar;
  double time_offset_result;
  double time_lag_1;
  double time_lag_2;
  int lag_IMU_wtr_Lidar;

  M3D R_IL_prev;
  V3D T_IL_prev;
  V3D gyro_bias_prev;
  V3D acc_bias_prev;
  double time_lag_2_prev;
};

#endif
