/*
 * GRIL-Calib batch calibration core.
 * Original implementation: TaeYoung Kim and GRIL-Calib contributors.
 * Heavily adapted upstream from LI-Init by Fangcheng Zhu and contributors.
 *
 * Modified 2026-08-28 for whl-cal:
 *   - removed ROS/catkin, generated messages, common_lib, and matplotlib;
 *   - replaced sensor_msgs IMU ingestion with Eigen vectors and a timestamp;
 *   - retained upstream batch calibration math/order;
 *   - retained validated synchronized removal of LiDAR states and their
 *     paired ground constraints.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License, version 2.
 * See ../../LICENSE.
 */

#include "Gril_Calib.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

const V3D STD_GRAV(0.0, 0.0, -G_m_s2);
double GYRO_FACTOR_ = 1.0;
double ACC_FACTOR_ = 1.0;
double GROUND_FACTOR_ = 1.0;

namespace {

M3D skew(const V3D &vector) {
    M3D result;
    result << 0.0, -vector[2], vector[1],
        vector[2], 0.0, -vector[0],
        -vector[1], vector[0], 0.0;
    return result;
}

}  // namespace

Gril_Calib::Gril_Calib()
        : time_delay_IMU_wtr_Lidar(0.0), time_lag_1(0.0),
          time_lag_2(0.0), lag_IMU_wtr_Lidar(0) {
    fout_LiDAR_meas.open(FILE_DIR("LiDAR_meas.txt"), std::ios::out);
    fout_IMU_meas.open(FILE_DIR("IMU_meas.txt"), std::ios::out);
    fout_before_filt_IMU.open(
        FILE_DIR("IMU_before_filter.txt"), std::ios::out);
    fout_before_filt_Lidar.open(
        FILE_DIR("Lidar_before_filter.txt"), std::ios::out);
    fout_acc_cost.open(FILE_DIR("acc_cost.txt"), std::ios::out);
    fout_after_rot.open(FILE_DIR("Lidar_omg_after_rot.txt"), std::ios::out);
    fout_LiDAR_ang_vel.open(FILE_DIR("Lidar_ang_vel.txt"), std::ios::out);
    fout_IMU_ang_vel.open(FILE_DIR("IMU_ang_vel.txt"), std::ios::out);
    fout_Jacob_trans.open(FILE_DIR("Jacob_trans.txt"), std::ios::out);
    fout_LiDAR_meas_after.open(
        FILE_DIR("LiDAR_meas_after.txt"), std::ios::out);

    data_accum_length = 300;
    trans_IL_x = 0.0;
    trans_IL_y = 0.0;
    trans_IL_z = 0.0;
    bound_th = 0.1;
    set_boundary = false;
    Rot_Grav_wrt_Init_Lidar = M3D::Identity();
    Trans_Lidar_wrt_IMU = V3D::Zero();
    Rot_Lidar_wrt_IMU = M3D::Identity();
    gyro_bias = V3D::Zero();
    acc_bias = V3D::Zero();
}

Gril_Calib::~Gril_Calib() = default;

void Gril_Calib::set_IMU_state(
    const std::deque<CalibState> &imu_states) {
    IMU_state_group.assign(imu_states.begin(), imu_states.end() - 1);
}

void Gril_Calib::set_Lidar_state(
    const std::deque<CalibState> &lidar_states) {
    Lidar_state_group.assign(lidar_states.begin(), lidar_states.end() - 1);
}

void Gril_Calib::set_states_2nd_filter(
    const std::deque<CalibState> &imu_states,
    const std::deque<CalibState> &lidar_states) {
    for (int i = 0; i < IMU_state_group.size(); i++) {
        IMU_state_group[i].ang_acc = imu_states[i].ang_acc;
        Lidar_state_group[i].ang_acc = lidar_states[i].ang_acc;
        Lidar_state_group[i].linear_acc = lidar_states[i].linear_acc;
    }
}

void Gril_Calib::fout_before_filter() {
    for (auto it = IMU_state_group.begin();
         it != IMU_state_group.end() - 1; ++it) {
        fout_before_filt_IMU
            << std::setprecision(15) << it->ang_vel.transpose() << " "
            << it->ang_vel.norm() << " " << it->linear_acc.transpose()
            << " " << it->timeStamp << std::endl;
    }
    for (auto it = Lidar_state_group.begin();
         it != Lidar_state_group.end() - 1; ++it) {
        fout_before_filt_Lidar
            << std::setprecision(15) << it->ang_vel.transpose() << " "
            << it->ang_vel.norm() << " " << it->timeStamp << std::endl;
    }
}

void Gril_Calib::fout_check_lidar() {
    for (auto it = Lidar_state_group.begin() + 1;
         it != Lidar_state_group.end() - 2; ++it) {
        fout_LiDAR_meas_after
            << std::setprecision(12) << it->ang_vel.transpose() << " "
            << it->ang_vel.norm() << " " << it->linear_acc.transpose() << " "
            << it->ang_acc.transpose() << " " << it->timeStamp << std::endl;
    }
}

void Gril_Calib::push_ALL_IMU_CalibState(
    const V3D &angular_velocity, const V3D &linear_acceleration,
    const double &timestamp, const double &mean_acc_norm) {
    CalibState state;
    state.ang_vel = angular_velocity;
    state.linear_acc = linear_acceleration / mean_acc_norm * G_m_s2;
    state.timeStamp = timestamp;
    IMU_state_group_ALL.push_back(state);
}

void Gril_Calib::push_IMU_CalibState(
    const V3D &omg, const V3D &acc, const double &timestamp) {
    CalibState state;
    state.ang_vel = omg;
    state.linear_acc = acc;
    state.timeStamp = timestamp;
    IMU_state_group.push_back(state);
}

void Gril_Calib::push_Lidar_CalibState(
    const M3D &rot, const V3D &pos, const V3D &omg,
    const V3D &linear_vel, const double &timestamp) {
    CalibState state;
    state.rot_end = rot;
    state.pos_end = pos;
    state.ang_vel = omg;
    state.linear_vel = linear_vel;
    state.timeStamp = timestamp;
    Lidar_state_group.push_back(state);
}

void Gril_Calib::push_Plane_Constraint(
    const QD &q_lidar, const QD &q_imu, const V3D &normal_lidar,
    const double &distance_lidar) {
    Lidar_wrt_ground_group.push_back(q_lidar);
    IMU_wrt_ground_group.push_back(q_imu);
    normal_vector_wrt_lidar_group.push_back(normal_lidar);
    distance_Lidar_wrt_ground_group.push_back(distance_lidar);
}

void Gril_Calib::set_batch_inputs(
    const std::deque<CalibState> &normalized_imu_states,
    const std::deque<CalibState> &lidar_states,
    const std::deque<QD> &lidar_wrt_ground,
    const std::deque<QD> &imu_wrt_ground,
    const std::deque<V3D> &normal_lidar,
    const std::deque<double> &distance_lidar) {
    IMU_state_group_ALL = normalized_imu_states;
    Lidar_state_group = lidar_states;
    Lidar_wrt_ground_group = lidar_wrt_ground;
    IMU_wrt_ground_group = imu_wrt_ground;
    normal_vector_wrt_lidar_group = normal_lidar;
    distance_Lidar_wrt_ground_group = distance_lidar;
}

std::size_t Gril_Calib::all_imu_sample_count() const {
    return IMU_state_group_ALL.size();
}

std::size_t Gril_Calib::imu_state_count() const {
    return IMU_state_group.size();
}

std::size_t Gril_Calib::lidar_state_count() const {
    return Lidar_state_group.size();
}

std::size_t Gril_Calib::plane_constraint_count() const {
    return Lidar_wrt_ground_group.size();
}

void Gril_Calib::downsample_interpolate_IMU(
    const double &move_start_time) {
    while (IMU_state_group_ALL.front().timeStamp < move_start_time - 3.0)
        IMU_state_group_ALL.pop_front();
    while (Lidar_state_group.front().timeStamp < move_start_time - 3.0) {
        Lidar_state_group.pop_front();
        Lidar_wrt_ground_group.pop_front();
        IMU_wrt_ground_group.pop_front();
        normal_vector_wrt_lidar_group.pop_front();
        distance_Lidar_wrt_ground_group.pop_front();
    }

    std::deque<CalibState> original(
        IMU_state_group_ALL.begin(), IMU_state_group_ALL.end() - 1);

    int mean_filter_size = 3;
    for (int i = mean_filter_size;
         i < IMU_state_group_ALL.size() - mean_filter_size; i++) {
        V3D acc_real = V3D::Zero();
        for (int k = -mean_filter_size; k < mean_filter_size + 1; k++) {
            acc_real +=
                (original[i + k].linear_acc - acc_real) /
                (k + mean_filter_size + 1);
        }
        IMU_state_group_ALL[i].linear_acc = acc_real;
    }

    for (int i = 0; i < Lidar_state_group.size(); i++) {
        for (int j = 1; j < IMU_state_group_ALL.size(); j++) {
            if (IMU_state_group_ALL[j - 1].timeStamp <=
                    Lidar_state_group[i].timeStamp &&
                IMU_state_group_ALL[j].timeStamp >
                    Lidar_state_group[i].timeStamp) {
                CalibState imu_state_interpolation;
                double delta_t =
                    IMU_state_group_ALL[j].timeStamp -
                    IMU_state_group_ALL[j - 1].timeStamp;
                double delta_t_right =
                    IMU_state_group_ALL[j].timeStamp -
                    Lidar_state_group[i].timeStamp;
                double s = delta_t_right / delta_t;
                imu_state_interpolation.ang_vel =
                    s * IMU_state_group_ALL[j - 1].ang_vel +
                    (1.0 - s) * IMU_state_group_ALL[j].ang_vel;
                imu_state_interpolation.linear_acc =
                    s * IMU_state_group_ALL[j - 1].linear_acc +
                    (1.0 - s) * IMU_state_group_ALL[j].linear_acc;
                push_IMU_CalibState(
                    imu_state_interpolation.ang_vel,
                    imu_state_interpolation.linear_acc,
                    Lidar_state_group[i].timeStamp);
                break;
            }
        }
    }
}

void Gril_Calib::central_diff() {
    for (auto it = IMU_state_group.begin() + 1;
         it != IMU_state_group.end() - 2; ++it) {
        const auto previous = it - 1;
        const auto next = it + 1;
        const double dt = next->timeStamp - previous->timeStamp;
        it->ang_acc = (next->ang_vel - previous->ang_vel) / dt;
        fout_IMU_meas
            << std::setprecision(12) << it->ang_vel.transpose() << " "
            << it->ang_vel.norm() << " " << it->linear_acc.transpose() << " "
            << it->ang_acc.transpose() << " " << it->timeStamp << std::endl;
    }

    for (auto it = Lidar_state_group.begin() + 1;
         it != Lidar_state_group.end() - 2; ++it) {
        const auto previous = it - 1;
        const auto next = it + 1;
        const double dt = next->timeStamp - previous->timeStamp;
        it->ang_acc = (next->ang_vel - previous->ang_vel) / dt;
        it->linear_acc = (next->linear_vel - previous->linear_vel) / dt;
        fout_LiDAR_meas
            << std::setprecision(12) << it->ang_vel.transpose() << " "
            << it->ang_vel.norm() << " "
            << (it->linear_acc - STD_GRAV).transpose() << " "
            << it->ang_acc.transpose() << " " << it->timeStamp << std::endl;
    }
}

void Gril_Calib::xcorr_temporal_init(const double &odom_freq) {
    const int count = static_cast<int>(IMU_state_group.size());
    double mean_imu = 0.0;
    double mean_lidar = 0.0;
    for (int i = 0; i < count; ++i) {
        mean_imu +=
            (IMU_state_group[i].ang_vel.norm() - mean_imu) / (i + 1);
        mean_lidar +=
            (Lidar_state_group[i].ang_vel.norm() - mean_lidar) / (i + 1);
    }

    double max_correlation = -DBL_MAX;
    for (int lag = -count + 1; lag < count; lag++) {
        double correlation = 0.0;
        int cnt = 0;
        for (int i = 0; i < count; i++) {
            const int j = i + lag;
            if (j < 0 || j > count - 1) {
                continue;
            }
            cnt++;
            correlation +=
                (IMU_state_group[i].ang_vel.norm() - mean_imu) *
                (Lidar_state_group[j].ang_vel.norm() - mean_lidar);
        }
        if (correlation > max_correlation) {
            max_correlation = correlation;
            lag_IMU_wtr_Lidar = -lag;
        }
    }

    time_lag_1 = lag_IMU_wtr_Lidar / odom_freq;
    std::cout << "Max Cross-correlation: IMU lag wtr Lidar : "
              << -lag_IMU_wtr_Lidar << '\n'
              << "Time lag 1: IMU lag wtr Lidar : " << time_lag_1
              << std::endl;
}

void Gril_Calib::IMU_time_compensate(
    const double &lag_time, const bool &is_discard) {
    if (is_discard) {
        int i = 0;
        while (i < 10) {
            Lidar_state_group.pop_front();
            IMU_state_group.pop_front();
            Lidar_wrt_ground_group.pop_front();
            IMU_wrt_ground_group.pop_front();
            normal_vector_wrt_lidar_group.pop_front();
            distance_Lidar_wrt_ground_group.pop_front();
            i++;
        }
    }

    for (auto it = IMU_state_group.begin();
         it != IMU_state_group.end() - 1; ++it) {
        it->timeStamp -= lag_time;
    }

    while (Lidar_state_group.front().timeStamp <
           IMU_state_group.front().timeStamp) {
        Lidar_state_group.pop_front();
        Lidar_wrt_ground_group.pop_front();
        IMU_wrt_ground_group.pop_front();
        normal_vector_wrt_lidar_group.pop_front();
        distance_Lidar_wrt_ground_group.pop_front();
    }
    while (Lidar_state_group.front().timeStamp >
           IMU_state_group[1].timeStamp)
        IMU_state_group.pop_front();

    while (IMU_state_group.size() > Lidar_state_group.size())
        IMU_state_group.pop_back();
    while (IMU_state_group.size() < Lidar_state_group.size())
        Lidar_state_group.pop_back();
}

void Gril_Calib::cut_sequence_tail() {
    for (int i = 0; i < 20; i++) {
        Lidar_state_group.pop_back();
        IMU_state_group.pop_back();
    }
    while (Lidar_state_group.front().timeStamp <
           IMU_state_group.front().timeStamp) {
        Lidar_state_group.pop_front();
        Lidar_wrt_ground_group.pop_front();
        IMU_wrt_ground_group.pop_front();
        normal_vector_wrt_lidar_group.pop_front();
        distance_Lidar_wrt_ground_group.pop_front();
    }
    while (Lidar_state_group.front().timeStamp >
           IMU_state_group[1].timeStamp)
        IMU_state_group.pop_front();

    while (IMU_state_group.size() > Lidar_state_group.size())
        IMU_state_group.pop_back();
    while (IMU_state_group.size() < Lidar_state_group.size())
        Lidar_state_group.pop_back();
}

void Gril_Calib::acc_interpolate() {
    for (int i = 1; i < Lidar_state_group.size() - 1; i++) {
        const double delta_t =
            Lidar_state_group[i].timeStamp - IMU_state_group[i].timeStamp;
        if (delta_t > 0.0) {
            const double interval =
                IMU_state_group[i + 1].timeStamp -
                IMU_state_group[i].timeStamp;
            const double s = delta_t / interval;
            IMU_state_group[i].linear_acc =
                s * IMU_state_group[i + 1].linear_acc +
                (1.0 - s) * IMU_state_group[i].linear_acc;
        } else {
            const double interval =
                IMU_state_group[i].timeStamp -
                IMU_state_group[i - 1].timeStamp;
            const double s = -delta_t / interval;
            IMU_state_group[i].linear_acc =
                s * IMU_state_group[i - 1].linear_acc +
                (1.0 - s) * IMU_state_group[i].linear_acc;
        }
        IMU_state_group[i].timeStamp += delta_t;
    }
}

void Gril_Calib::Butter_filt(
    const std::deque<CalibState> &signal_in,
    std::deque<CalibState> &signal_out) {
    Butterworth butter;
    butter.extend_num = 10 * (butter.Coeff_size - 1);
    auto front = signal_in.begin() + butter.extend_num;
    auto back = signal_in.end() - 1 - butter.extend_num;
    std::deque<CalibState> extend_front;
    std::deque<CalibState> extend_back;

    for (int index = 0; index < butter.extend_num; index++) {
        extend_front.push_back(*front);
        extend_back.push_front(*back);
        --front;
        ++back;
    }

    std::deque<CalibState> extended(signal_in);
    while (!extend_front.empty()) {
        extended.push_front(extend_front.back());
        extend_front.pop_back();
    }
    while (!extend_back.empty()) {
        extended.push_back(extend_back.front());
        extend_back.pop_front();
    }

    std::deque<CalibState> output(extended);
    for (int i = butter.Coeff_size;
         i < extended.size() - butter.extend_num; i++) {
        CalibState temporary;
        for (int j = 0; j < butter.Coeff_size; j++) {
            temporary += extended[i - j] * butter.Coeff_b[j];
        }
        for (int j = 1; j < butter.Coeff_size; j++) {
            temporary -= output[i - j] * butter.Coeff_a[j];
        }
        output[i] = temporary;
    }

    for (auto it = output.begin() + butter.extend_num;
         it != output.end() - butter.extend_num; ++it) {
        signal_out.push_back(*it);
    }
}

void Gril_Calib::zero_phase_filt(
    const std::deque<CalibState> &signal_in,
    std::deque<CalibState> &signal_out) {
    std::deque<CalibState> first_pass;
    Butter_filt(signal_in, first_pass);
    std::deque<CalibState> reversed(first_pass);
    std::reverse(reversed.begin(), reversed.end());
    Butter_filt(reversed, signal_out);
    std::reverse(signal_out.begin(), signal_out.end());
}

void Gril_Calib::solve_Rotation_only() {
    double R_LI_quat[4] = {1.0, 0.0, 0.0, 0.0};
    auto *quaternion_parameterization =
        new ceres::QuaternionParameterization();
    ceres::Problem problem;
    problem.AddParameterBlock(R_LI_quat, 4, quaternion_parameterization);
    for (int i = 0; i < IMU_state_group.size(); i++) {
        problem.AddResidualBlock(
            Angular_Vel_Cost_only_Rot::Create(
                IMU_state_group[i].ang_vel,
                Lidar_state_group[i].ang_vel),
            nullptr, R_LI_quat);
    }
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    Rot_Lidar_wrt_IMU =
        QD(R_LI_quat[0], R_LI_quat[1], R_LI_quat[2], R_LI_quat[3])
            .matrix();
}

void Gril_Calib::solve_Rot_Trans_calib(
    double &timediff_imu_wrt_lidar, const double &imu_height) {
    (void)timediff_imu_wrt_lidar;
    const M3D R_IL_init = Rot_Lidar_wrt_IMU.transpose();
    const QD initial_quaternion(R_IL_init);
    double R_IL_quat[4] = {
        initial_quaternion.w(), initial_quaternion.x(),
        initial_quaternion.y(), initial_quaternion.z()};
    double Trans_IL[3] = {trans_IL_x, trans_IL_y, trans_IL_z};
    double bias_g[3] = {0.0, 0.0, 0.0};
    double bias_aL[3] = {0.0, 0.0, 0.0};
    double time_lag2 = 0.0;

    ceres::Problem problem;
    auto *quaternion_parameterization =
        new ceres::QuaternionParameterization();
    auto *angular_loss = new ceres::ScaledLoss(
        new ceres::CauchyLoss(0.5), 0.5, ceres::TAKE_OWNERSHIP);
    auto *acceleration_loss = new ceres::ScaledLoss(
        new ceres::CauchyLoss(0.5), 0.2, ceres::TAKE_OWNERSHIP);
    auto *ground_loss = new ceres::ScaledLoss(
        new ceres::HuberLoss(0.5), 0.3, ceres::TAKE_OWNERSHIP);

    problem.AddParameterBlock(R_IL_quat, 4, quaternion_parameterization);
    problem.AddParameterBlock(Trans_IL, 3);
    problem.AddParameterBlock(bias_g, 3);
    problem.AddParameterBlock(bias_aL, 3);

    const int jacobian_rows =
        3 * static_cast<int>(Lidar_state_group.size());
    Eigen::MatrixXd Jacobian(jacobian_rows, 9);
    Jacobian.setZero();
    Eigen::MatrixXd Jaco_Trans(jacobian_rows, 3);
    Jaco_Trans.setZero();

    for (int i = 0; i < IMU_state_group.size(); i++) {
        const double delta_t =
            Lidar_state_group[i].timeStamp - IMU_state_group[i].timeStamp;
        problem.AddResidualBlock(
            Ground_Plane_Cost_IL::Create(
                Lidar_wrt_ground_group[i], IMU_wrt_ground_group[i],
                distance_Lidar_wrt_ground_group[i], imu_height),
            ground_loss, R_IL_quat, Trans_IL);
        problem.AddResidualBlock(
            Angular_Vel_IL_Cost::Create(
                IMU_state_group[i].ang_vel,
                IMU_state_group[i].ang_acc,
                Lidar_state_group[i].ang_vel, delta_t),
            angular_loss, R_IL_quat, bias_g, &time_lag2);
        problem.AddResidualBlock(
            Linear_acc_Rot_Cost_without_Gravity::Create(
                Lidar_state_group[i], IMU_state_group[i].linear_acc,
                Lidar_wrt_ground_group[i]),
            acceleration_loss, R_IL_quat, bias_aL, Trans_IL);

        Jacobian.block<3, 3>(3 * i, 0) =
            -Lidar_state_group[i].rot_end;
        Jacobian.block<3, 3>(3 * i, 3) = skew(STD_GRAV);
        const M3D translation_jacobian =
            -skew(Lidar_state_group[i].ang_vel) *
                skew(Lidar_state_group[i].ang_vel) -
            skew(Lidar_state_group[i].ang_acc);
        Jaco_Trans.block<3, 3>(3 * i, 0) = translation_jacobian;
        Jacobian.block<3, 3>(3 * i, 6) = translation_jacobian;
    }

    for (int index = 0; index < 3; index++) {
        problem.SetParameterUpperBound(bias_aL, index, 0.01);
        problem.SetParameterLowerBound(bias_aL, index, -0.01);
        problem.SetParameterUpperBound(bias_g, index, 0.01);
        problem.SetParameterLowerBound(bias_g, index, -0.01);
    }
    if (set_boundary) {
        for (int index = 0; index < 3; index++) {
            problem.SetParameterUpperBound(
                Trans_IL, index, Trans_IL[index] + bound_th);
            problem.SetParameterLowerBound(
                Trans_IL, index, Trans_IL[index] - bound_th);
        }
    }

    ceres::Solver::Options options;
    options.num_threads = 1;
    options.use_explicit_schur_complement = true;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.preconditioner_type = ceres::SCHUR_JACOBI;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    const QD q_IL_final(
        R_IL_quat[0], R_IL_quat[1], R_IL_quat[2], R_IL_quat[3]);
    Rot_Lidar_wrt_IMU = q_IL_final.matrix().transpose();
    const V3D Trans_IL_vec(Trans_IL[0], Trans_IL[1], Trans_IL[2]);
    Trans_Lidar_wrt_IMU =
        -1.0 * Rot_Lidar_wrt_IMU * Trans_IL_vec;
    Grav_L0 = Lidar_wrt_ground_group[0].matrix() * STD_GRAV;
    const V3D bias_a_lidar(bias_aL[0], bias_aL[1], bias_aL[2]);
    acc_bias = Rot_Lidar_wrt_IMU * bias_a_lidar;
    gyro_bias = V3D(bias_g[0], bias_g[1], bias_g[2]);
    time_lag_2 = time_lag2;
    time_delay_IMU_wtr_Lidar = time_lag_1 + time_lag_2;
    time_offset_result = time_delay_IMU_wtr_Lidar;

    IMU_time_compensate(get_lag_time_2(), false);

    for (int i = 0; i < IMU_state_group.size(); i++) {
        const V3D gravity_lidar =
            Lidar_wrt_ground_group[i].matrix() * STD_GRAV;
        const V3D acceleration_imu =
            Lidar_state_group[i].rot_end *
                Rot_Lidar_wrt_IMU.transpose() *
                IMU_state_group[i].linear_acc -
            Lidar_state_group[i].rot_end * bias_a_lidar;
        const V3D acceleration_lidar =
            Lidar_state_group[i].linear_acc +
            Lidar_state_group[i].rot_end *
                Jaco_Trans.block<3, 3>(3 * i, 0) * Trans_IL_vec -
            gravity_lidar;
        fout_acc_cost
            << std::setprecision(10) << acceleration_imu.transpose() << " "
            << acceleration_lidar.transpose() << " "
            << IMU_state_group[i].timeStamp << " "
            << Lidar_state_group[i].timeStamp << std::endl;
    }
}

void Gril_Calib::normalize_acc(std::deque<CalibState> &signal_in) {
    V3D mean_acceleration = V3D::Zero();
    for (int i = 1; i < 10; i++) {
        mean_acceleration +=
            (signal_in[i].linear_acc - mean_acceleration) / i;
    }
    for (int i = 0; i < signal_in.size(); i++) {
        signal_in[i].linear_acc =
            signal_in[i].linear_acc / mean_acceleration.norm() * G_m_s2;
    }
}

void Gril_Calib::align_Group(
    const std::deque<CalibState> &imu_states,
    std::deque<QD> &lidar_wrt_ground_states,
    std::deque<QD> &imu_wrt_ground_states,
    std::deque<V3D> &normal_vectors,
    std::deque<double> &ground_distances) {
    while (imu_states.size() < lidar_wrt_ground_states.size()) {
        lidar_wrt_ground_states.pop_back();
        imu_wrt_ground_states.pop_back();
        normal_vectors.pop_back();
        ground_distances.pop_back();
    }
}

bool Gril_Calib::data_sufficiency_assess(
    Eigen::MatrixXd &jacobian_rot, int &frame_num, V3D &lidar_omg,
    int &orig_odom_freq, int &cut_frame_num, QD &lidar_q, QD &imu_q,
    double &lidar_estimate_height) {
    jacobian_rot.block<3, 3>(3 * frame_num, 0) = skew(lidar_omg);
    bool data_sufficient = false;

    if (frame_num % orig_odom_freq * cut_frame_num == 0) {
        const M3D hessian = jacobian_rot.transpose() * jacobian_rot;
        Eigen::EigenSolver<M3D> eigen_solver(hessian);
        const V3D eigenvalues = eigen_solver.eigenvalues().real();
        const M3D eigenvectors = eigen_solver.eigenvectors().real();
        const M3D squared_eigenvectors =
            eigenvectors.cwiseProduct(eigenvectors);
        std::vector<double> column_1{
            squared_eigenvectors(0, 0), squared_eigenvectors(1, 0),
            squared_eigenvectors(2, 0)};
        std::vector<double> column_2{
            squared_eigenvectors(0, 1), squared_eigenvectors(1, 1),
            squared_eigenvectors(2, 1)};
        std::vector<double> column_3{
            squared_eigenvectors(0, 2), squared_eigenvectors(1, 2),
            squared_eigenvectors(2, 2)};
        int max_position[3] = {
            static_cast<int>(
                std::max_element(column_1.begin(), column_1.end()) -
                column_1.begin()),
            static_cast<int>(
                std::max_element(column_2.begin(), column_2.end()) -
                column_2.begin()),
            static_cast<int>(
                std::max_element(column_3.begin(), column_3.end()) -
                column_3.begin())};

        const V3D scaled_eigenvalues = eigenvalues / data_accum_length;
        const V3D rotation_percent(
            scaled_eigenvalues[1] * scaled_eigenvalues[2],
            scaled_eigenvalues[0] * scaled_eigenvalues[2],
            scaled_eigenvalues[0] * scaled_eigenvalues[1]);
        int axis[3];
        axis[2] = static_cast<int>(
            std::max_element(max_position, max_position + 3) -
            max_position);
        axis[0] = static_cast<int>(
            std::min_element(max_position, max_position + 3) -
            max_position);
        axis[1] = 3 - (axis[0] + axis[2]);

        const double percentage_x =
            rotation_percent[axis[0]] < x_accumulate
                ? rotation_percent[axis[0]]
                : 1.0;
        const double percentage_y =
            rotation_percent[axis[1]] < y_accumulate
                ? rotation_percent[axis[1]]
                : 1.0;
        const double percentage_z =
            rotation_percent[axis[2]] < z_accumulate
                ? rotation_percent[axis[2]]
                : 1.0;
        clear();
        printProgress(percentage_x, 88);
        printProgress(percentage_y, 89);
        printProgress(percentage_z, 90);

        if (verbose) {
            std::cout << "[Rotation matrix Ground to LiDAR (euler)] "
                      << std::setprecision(4)
                      << (RotMtoEuler(lidar_q.toRotationMatrix()) * 57.3)
                             .transpose()
                      << " deg\n"
                      << "[Rotation matrix Ground to IMU (euler)] "
                      << (RotMtoEuler(imu_q.toRotationMatrix()) * 57.3)
                             .transpose()
                      << " deg\n"
                      << "[Estimated LiDAR sensor height] "
                      << lidar_estimate_height << " m\n";
        }
        if (rotation_percent[axis[0]] > x_accumulate &&
            rotation_percent[axis[1]] > y_accumulate &&
            rotation_percent[axis[2]] > z_accumulate) {
            std::cout << "[calibration] Data accumulation finished, "
                         "LiDAR-IMU calibration begins.\n\n";
            data_sufficient = true;
        }
    }
    return data_sufficient;
}

void Gril_Calib::printProgress(double percentage, int axis_ascii) {
    constexpr int width = 30;
    constexpr const char *bar =
        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||";
    const int value = static_cast<int>(percentage * 100.0);
    const int left = static_cast<int>(percentage * width);
    const int right = width - left;
    if (percentage < 1.0) {
        std::printf(
            "[Data accumulation] Rotation around Lidar %c Axis: "
            "%3d%% [%.*s%*s]\n",
            static_cast<char>(axis_ascii), value, left, bar, right, "");
    } else {
        std::printf(
            "[Data accumulation] Rotation around Lidar %c Axis complete!\n",
            static_cast<char>(axis_ascii));
    }
}

void Gril_Calib::clear() {
    std::cout << "\x1B[2J\x1B[H";
}

void Gril_Calib::dump_batch_trace_v1(
    const std::string &path,
    const int &orig_odom_freq,
    const int &cut_frame_num,
    const double &timediff_imu_wrt_lidar,
    const double &move_start_time) const {
    std::ofstream trace(path, std::ios::out);
    if (!trace)
        throw std::runtime_error("cannot create GRIL batch trace: " + path);
    trace << std::setprecision(17);
    trace << "GRIL_BATCH_TRACE 1\n"
          << "orig_odom_freq " << orig_odom_freq << "\n"
          << "cut_frame_num " << cut_frame_num << "\n"
          << "timediff_imu_wrt_lidar "
          << timediff_imu_wrt_lidar << "\n"
          << "move_start_time " << move_start_time << "\n"
          << "imu_states " << IMU_state_group_ALL.size() << "\n";

    const auto write_state = [&trace](
                                 const char *label,
                                 const CalibState &state) {
        trace << label << " " << state.timeStamp;
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 3; ++column)
                trace << " " << state.rot_end(row, column);
        trace << " " << state.pos_end.transpose()
              << " " << state.ang_vel.transpose()
              << " " << state.linear_vel.transpose()
              << " " << state.ang_acc.transpose()
              << " " << state.linear_acc.transpose() << "\n";
    };
    for (const auto &state : IMU_state_group_ALL)
        write_state("imu", state);

    trace << "lidar_states " << Lidar_state_group.size() << "\n";
    for (const auto &state : Lidar_state_group)
        write_state("lidar", state);

    trace << "ground_constraints "
          << Lidar_wrt_ground_group.size() << "\n";
    for (std::size_t index = 0;
         index < Lidar_wrt_ground_group.size();
         ++index) {
        trace << "ground "
              << Lidar_wrt_ground_group[index].w() << " "
              << Lidar_wrt_ground_group[index].x() << " "
              << Lidar_wrt_ground_group[index].y() << " "
              << Lidar_wrt_ground_group[index].z() << " "
              << IMU_wrt_ground_group[index].w() << " "
              << IMU_wrt_ground_group[index].x() << " "
              << IMU_wrt_ground_group[index].y() << " "
              << IMU_wrt_ground_group[index].z() << " "
              << normal_vector_wrt_lidar_group[index].transpose() << " "
              << distance_Lidar_wrt_ground_group[index] << "\n";
    }
    trace << "END\n";
    if (!trace)
        throw std::runtime_error("failed to write GRIL batch trace: " + path);
}

void Gril_Calib::LI_Calibration(
    int &orig_odom_freq, int &cut_frame_num,
    double &timediff_imu_wrt_lidar, const double &move_start_time) {
    downsample_interpolate_IMU(move_start_time);
    fout_before_filter();
    IMU_time_compensate(0.0, true);

    std::deque<CalibState> imu_first_filter;
    std::deque<CalibState> lidar_first_filter;
    zero_phase_filt(get_IMU_state(), imu_first_filter);
    normalize_acc(imu_first_filter);
    zero_phase_filt(get_Lidar_state(), lidar_first_filter);
    set_IMU_state(imu_first_filter);
    set_Lidar_state(lidar_first_filter);
    cut_sequence_tail();

    xcorr_temporal_init(orig_odom_freq * cut_frame_num);
    IMU_time_compensate(get_lag_time_1(), false);
    central_diff();

    std::deque<CalibState> imu_second_filter;
    std::deque<CalibState> lidar_second_filter;
    zero_phase_filt(get_IMU_state(), imu_second_filter);
    zero_phase_filt(get_Lidar_state(), lidar_second_filter);
    set_states_2nd_filter(imu_second_filter, lidar_second_filter);
    fout_check_lidar();

    solve_Rotation_only();
    acc_interpolate();
    align_Group(
        IMU_state_group, Lidar_wrt_ground_group, IMU_wrt_ground_group,
        normal_vector_wrt_lidar_group,
        distance_Lidar_wrt_ground_group);
    solve_Rot_Trans_calib(timediff_imu_wrt_lidar, imu_sensor_height);

    double time_L_I =
        timediff_imu_wrt_lidar + time_delay_IMU_wtr_Lidar;
    time_offset_result = time_L_I;
    print_calibration_result(
        time_L_I, Rot_Lidar_wrt_IMU, Trans_Lidar_wrt_IMU,
        gyro_bias, acc_bias, Grav_L0);
    std::cout << "GRIL-Calib: Ground Robot IMU-LiDAR calibration done.\n";
}

void Gril_Calib::print_calibration_result(
    double &time_L_I, M3D &R_L_I, V3D &p_L_I, V3D &bias_g,
    V3D &bias_a, V3D gravity) {
    (void)gravity;
    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(6)
              << "[Calibration Result] Rotation matrix from LiDAR frame "
                 "to IMU frame = "
              << (RotMtoEuler(R_L_I) * 57.3).transpose() << " deg\n"
              << "[Calibration Result] Translation vector from LiDAR frame "
                 "to IMU frame = "
              << p_L_I.transpose() << " m\n";
    std::printf(
        "[Calibration Result] Time Lag IMU to LiDAR = %.8lf s\n",
        time_L_I);
    std::cout << "[Calibration Result] Bias of Gyroscope = "
              << bias_g.transpose() << " rad/s\n"
              << "[Calibration Result] Bias of Accelerometer = "
              << bias_a.transpose() << " m/s^2\n";
}

void Gril_Calib::plot_result() {}
