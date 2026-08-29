/*
 * Complete ROS-free GRIL frontend and live calibration execution.
 * Copyright (C) 2026 whl-cal contributors.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License, version 2.
 * See ../../LICENSE.
 */

#include "FullFrontend.h"

#include "Gril_Calib.h"
#include "VelodynePreprocess.h"

#include <Fusion/Fusion.h>
#include <GroundSegmentation/PatchworkppNative.h>

#include <pcl/common/point_tests.h>
#include <pcl/features/normal_3d.h>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kPiM = 3.14159265358;
constexpr unsigned int kImuHz = 200;
constexpr std::size_t kJacobianRows = 30000;
constexpr std::uint8_t kImuEvent = 0;
constexpr std::uint8_t kLidarEvent = 1;
constexpr std::int64_t kNanosecondsPerSecond = 1000000000;

double ros_time_to_seconds(std::int64_t timestamp_ns) {
  std::int64_t seconds = timestamp_ns / kNanosecondsPerSecond;
  std::int64_t nanoseconds = timestamp_ns % kNanosecondsPerSecond;
  if (nanoseconds < 0) {
    --seconds;
    nanoseconds += kNanosecondsPerSecond;
  }
  return static_cast<double>(seconds) +
         1e-9 * static_cast<double>(nanoseconds);
}

void expect(std::istream &input, const std::string &expected) {
  std::string actual;
  if (!(input >> actual) || actual != expected)
    throw std::runtime_error("expected config token: " + expected);
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

void require_finite(double value, const std::string &name) {
  if (!std::isfinite(value))
    throw std::runtime_error("non-finite full frontend config: " + name);
}

void validate_config(const FullFrontendConfig &config) {
  if (config.lidar_type != 2)
    throw std::runtime_error(
        "full native GRIL currently implements only pinned VELO lidar_type=2");
  if (!config.cut_frame)
    throw std::runtime_error(
        "full native GRIL requires the pinned cut_frame=true path");
  if (config.feature_extract_enabled)
    throw std::runtime_error(
        "full native GRIL does not implement feature_extract_en=true");
  if (config.scan_line <= 0 || config.point_filter_num <= 0 ||
      config.cut_frame_num <= 0 || config.orig_odom_freq <= 0)
    throw std::runtime_error("invalid positive full frontend configuration");
  if (config.mean_acc_norm == 0.0)
    throw std::runtime_error("mean_acc_norm must be nonzero");
  if (config.patchwork.num_sectors_each_zone.size() != 4 ||
      config.patchwork.num_rings_each_zone.size() != 4 ||
      config.patchwork.elevation_thresholds.size() != 4 ||
      config.patchwork.flatness_thresholds.size() != 4)
    throw std::runtime_error(
        "Patchwork++ full frontend requires four values per CZM zone");
  const double values[] = {
      config.blind,
      config.mean_acc_norm,
      config.data_accum_length,
      config.x_accumulate,
      config.y_accumulate,
      config.z_accumulate,
      config.svd_threshold,
      config.imu_sensor_height,
      config.trans_IL_x,
      config.trans_IL_y,
      config.trans_IL_z,
      config.bound_th,
      config.gyro_factor,
      config.acc_factor,
      config.ground_factor,
      config.odometry.cube_side_length,
      config.odometry.filter_size_surf,
      config.odometry.filter_size_map,
      config.odometry.detection_range,
      config.odometry.ground_covariance,
      config.gyr_cov.x(),
      config.acc_cov.x(),
      config.patchwork.sensor_height,
      config.patchwork.th_seeds,
      config.patchwork.th_dist,
      config.patchwork.th_seeds_v,
      config.patchwork.th_dist_v,
      config.patchwork.max_range,
      config.patchwork.min_range,
      config.patchwork.uprightness_thr,
      config.patchwork.adaptive_seed_selection_margin,
      config.patchwork.rnr_ver_angle_thr,
      config.patchwork.rnr_intensity_thr,
      config.configured_time_lag_s,
  };
  for (double value : values)
    require_finite(value, "value");
}

template <typename T>
void read_binary(std::istream &input, T &value, const char *name) {
  input.read(reinterpret_cast<char *>(&value), sizeof(T));
  if (!input)
    throw std::runtime_error(std::string("truncated native dataset at ") +
                             name);
}

std::uint64_t trace_mix(std::uint64_t hash, std::uint64_t field) {
  return (hash ^ field) * UINT64_C(1099511628211);
}

std::uint32_t float_bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::uint64_t cloud_hash(const PointCloudXYZI &cloud) {
  std::uint64_t hash = UINT64_C(1469598103934665603);
  hash = trace_mix(hash, cloud.size());
  for (const auto &point : cloud.points) {
    hash = trace_mix(hash, float_bits(point.x));
    hash = trace_mix(hash, float_bits(point.y));
    hash = trace_mix(hash, float_bits(point.z));
    hash = trace_mix(hash, float_bits(point.intensity));
    hash = trace_mix(hash, float_bits(point.curvature));
  }
  return hash;
}

bool compute_point_normal_pcl_1_10(const pcl::PointCloud<pcl::PointXYZI> &cloud,
                                   Eigen::Vector4f &plane_parameters,
                                   float &curvature) {
  if (cloud.size() < 3) {
    plane_parameters.setConstant(std::numeric_limits<float>::quiet_NaN());
    curvature = std::numeric_limits<float>::quiet_NaN();
    return false;
  }
  Eigen::Matrix<float, 1, 9, Eigen::RowMajor> accumulated =
      Eigen::Matrix<float, 1, 9, Eigen::RowMajor>::Zero();
  std::size_t point_count;
  if (cloud.is_dense) {
    point_count = cloud.size();
    for (const auto &point : cloud) {
      accumulated[0] += point.x * point.x;
      accumulated[1] += point.x * point.y;
      accumulated[2] += point.x * point.z;
      accumulated[3] += point.y * point.y;
      accumulated[4] += point.y * point.z;
      accumulated[5] += point.z * point.z;
      accumulated[6] += point.x;
      accumulated[7] += point.y;
      accumulated[8] += point.z;
    }
  } else {
    point_count = 0;
    for (const auto &point : cloud) {
      if (!pcl::isFinite(point))
        continue;
      accumulated[0] += point.x * point.x;
      accumulated[1] += point.x * point.y;
      accumulated[2] += point.x * point.z;
      accumulated[3] += point.y * point.y;
      accumulated[4] += point.y * point.z;
      accumulated[5] += point.z * point.z;
      accumulated[6] += point.x;
      accumulated[7] += point.y;
      accumulated[8] += point.z;
      ++point_count;
    }
  }
  accumulated /= static_cast<float>(point_count);
  if (point_count == 0) {
    plane_parameters.setConstant(std::numeric_limits<float>::quiet_NaN());
    curvature = std::numeric_limits<float>::quiet_NaN();
    return false;
  }

  Eigen::Vector4f centroid;
  centroid[0] = accumulated[6];
  centroid[1] = accumulated[7];
  centroid[2] = accumulated[8];
  centroid[3] = 1.0F;
  Eigen::Matrix3f covariance;
  covariance.coeffRef(0) = accumulated[0] - accumulated[6] * accumulated[6];
  covariance.coeffRef(1) = accumulated[1] - accumulated[6] * accumulated[7];
  covariance.coeffRef(2) = accumulated[2] - accumulated[6] * accumulated[8];
  covariance.coeffRef(4) = accumulated[3] - accumulated[7] * accumulated[7];
  covariance.coeffRef(5) = accumulated[4] - accumulated[7] * accumulated[8];
  covariance.coeffRef(8) = accumulated[5] - accumulated[8] * accumulated[8];
  covariance.coeffRef(3) = covariance.coeff(1);
  covariance.coeffRef(6) = covariance.coeff(2);
  covariance.coeffRef(7) = covariance.coeff(5);
  pcl::solvePlaneParameters(covariance, centroid, plane_parameters, curvature);
  return true;
}

void write_state(std::ostream &output, const char *label,
                 const FrontendState &state) {
  output << label;
  for (int row = 0; row < 3; ++row)
    for (int column = 0; column < 3; ++column)
      output << " " << state.rot_end(row, column);
  output << " " << state.pos_end.transpose();
  for (int row = 0; row < 3; ++row)
    for (int column = 0; column < 3; ++column)
      output << " " << state.offset_R_L_I(row, column);
  output << " " << state.offset_T_L_I.transpose() << " "
         << state.vel_end.transpose() << " " << state.bias_g.transpose() << " "
         << state.bias_a.transpose() << " " << state.gravity.transpose();
  for (int row = 0; row < 24; ++row)
    for (int column = 0; column < 24; ++column)
      output << " " << state.cov(row, column);
  output << "\n";
}

void run_batch_process(const FullFrontendRunConfig &run) {
  const pid_t child = fork();
  if (child < 0)
    throw std::runtime_error("could not fork native GRIL batch process");
  if (child == 0) {
    execl(run.batch_executable_path.c_str(), run.batch_executable_path.c_str(),
          "--trace", run.batch_trace_path.c_str(), "--config",
          run.batch_config_path.c_str(), "--output", run.result_path.c_str(),
          static_cast<char *>(nullptr));
    _exit(127);
  }
  int status = 0;
  if (waitpid(child, &status, 0) != child)
    throw std::runtime_error("could not wait for native GRIL batch process");
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
    throw std::runtime_error("native GRIL batch process failed");
  std::ifstream result(run.result_path);
  if (!result)
    throw std::runtime_error(
        "native GRIL batch process did not create its result");
}

struct RawGroundState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Quaterniond lidar_wrt_ground = Eigen::Quaterniond::Identity();
  Eigen::Vector3d normal_lidar = Eigen::Vector3d(0.0, 0.0, 1.0);
  double lidar_height = 0.0;
  std::size_t ground_count = 0;
  std::size_t nonground_count = 0;
};

class AhrsEstimator {
public:
  AhrsEstimator() {
    FusionOffsetInitialise(&offset_, kImuHz);
    FusionAhrsInitialise(&ahrs_);
    FusionAhrsSettings settings;
    settings.gain = 0.5F;
    settings.accelerationRejection = 10.0F;
    settings.magneticRejection = 0.0F;
    settings.rejectionTimeout = 5 * kImuHz;
    FusionAhrsSetSettings(&ahrs_, &settings);

    std::memset(&identity_, 0, sizeof(identity_));
    identity_.array[0][0] = 1.0F;
    identity_.array[1][1] = 1.0F;
    identity_.array[2][2] = 1.0F;
    sensitivity_.axis.x = 1.0F;
    sensitivity_.axis.y = 1.0F;
    sensitivity_.axis.z = 1.0F;
    zero_.axis.x = 0.0F;
    zero_.axis.y = 0.0F;
    zero_.axis.z = 0.0F;
  }

  bool update(const std::deque<FrontendImuSample> &samples,
              Eigen::Quaterniond &imu_wrt_ground) {
    if (samples.empty())
      throw std::runtime_error(
          "synchronized GRIL package contains no IMU samples");
    double estimate_timestamp_s = samples.front().timestamp_s;
    bool initialized = false;
    for (const auto &sample : samples) {
      const float delta_time =
          static_cast<float>(sample.timestamp_s - estimate_timestamp_s);

      FusionVector gyroscope;
      gyroscope.axis.x =
          static_cast<float>(sample.angular_velocity.x() * 180.0 / kPiM);
      gyroscope.axis.y =
          static_cast<float>(sample.angular_velocity.y() * 180.0 / kPiM);
      gyroscope.axis.z =
          static_cast<float>(sample.angular_velocity.z() * 180.0 / kPiM);
      gyroscope =
          FusionCalibrationInertial(gyroscope, identity_, sensitivity_, zero_);
      gyroscope = FusionOffsetUpdate(&offset_, gyroscope);

      FusionVector accelerometer;
      accelerometer.axis.x =
          static_cast<float>(sample.linear_acceleration.x() / 9.81);
      accelerometer.axis.y =
          static_cast<float>(sample.linear_acceleration.y() / 9.81);
      accelerometer.axis.z =
          static_cast<float>(sample.linear_acceleration.z() / 9.81);
      accelerometer = FusionCalibrationInertial(accelerometer, identity_,
                                                sensitivity_, zero_);

      FusionAhrsUpdateNoMagnetometer(&ahrs_, gyroscope, accelerometer,
                                     delta_time);
      const FusionAhrsFlags flags = FusionAhrsGetFlags(&ahrs_);
      const FusionAhrsInternalStates internal =
          FusionAhrsGetInternalStates(&ahrs_);
      (void)internal;
      if (!flags.initialising) {
        const FusionQuaternion quaternion = FusionAhrsGetQuaternion(&ahrs_);
        const FusionEuler euler = FusionQuaternionToEuler(quaternion);
        (void)euler;
        imu_wrt_ground =
            Eigen::Quaterniond(quaternion.element.w, quaternion.element.x,
                               quaternion.element.y, quaternion.element.z);
        initialized = true;
      }
      estimate_timestamp_s = sample.timestamp_s;
    }
    return initialized;
  }

private:
  FusionOffset offset_;
  FusionAhrs ahrs_;
  FusionMatrix identity_;
  FusionVector sensitivity_;
  FusionVector zero_;
};

void configure_calibration(Gril_Calib &calibration,
                           const FullFrontendConfig &config) {
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
  GYRO_FACTOR_ = config.gyro_factor;
  ACC_FACTOR_ = config.acc_factor;
  GROUND_FACTOR_ = config.ground_factor;
}

PatchworkppConfig patchwork_config(const FullPatchworkConfig &input) {
  PatchworkppConfig output;
  output.num_iter = input.num_iter;
  output.num_lpr = input.num_lpr;
  output.num_min_pts = input.num_min_pts;
  output.max_flatness_storage = input.max_flatness_storage;
  output.max_elevation_storage = input.max_elevation_storage;
  output.sensor_height = input.sensor_height;
  output.th_seeds = input.th_seeds;
  output.th_dist = input.th_dist;
  output.th_seeds_v = input.th_seeds_v;
  output.th_dist_v = input.th_dist_v;
  output.max_range = input.max_range;
  output.min_range = input.min_range;
  output.uprightness_thr = input.uprightness_thr;
  output.adaptive_seed_selection_margin = input.adaptive_seed_selection_margin;
  output.rnr_ver_angle_thr = input.rnr_ver_angle_thr;
  output.rnr_intensity_thr = input.rnr_intensity_thr;
  output.verbose = input.verbose;
  output.enable_rnr = input.enable_rnr;
  output.enable_rvpf = input.enable_rvpf;
  output.enable_tgr = input.enable_tgr;
  output.num_sectors_each_zone = input.num_sectors_each_zone;
  output.num_rings_each_zone = input.num_rings_each_zone;
  output.elevation_thresholds = input.elevation_thresholds;
  output.flatness_thresholds = input.flatness_thresholds;
  return output;
}

class PipelineState {
public:
  explicit PipelineState(const FullFrontendConfig &value)
      : config(value), propagator(value.gyr_cov, value.acc_cov),
        odometry(new LidarOdometryCore(value.odometry)),
        patchwork(patchwork_config(value.patchwork)),
        hard_time(value.configured_time_lag_s),
        jacobian(Eigen::MatrixXd::Zero(kJacobianRows, 3)) {
    configure_calibration(calibration, config);
  }

  RawGroundState segment_ground(const std::vector<VelodynePoint> &points) {
    pcl::PointCloud<pcl::PointXYZI> cloud;
    cloud.reserve(points.size());
    for (const auto &input : points) {
      pcl::PointXYZI point;
      point.x = input.x;
      point.y = input.y;
      point.z = input.z;
      point.intensity = input.intensity;
      cloud.push_back(point);
    }

    pcl::PointCloud<pcl::PointXYZI> ground;
    pcl::PointCloud<pcl::PointXYZI> nonground;
    double elapsed = 0.0;
    patchwork.estimate_ground(cloud, ground, nonground, elapsed);

    Eigen::Vector4f plane_parameters;
    float curvature = 0.0F;
    compute_point_normal_pcl_1_10(ground, plane_parameters, curvature);
    (void)curvature;
    RawGroundState result;
    result.normal_lidar = Eigen::Vector3d(
        plane_parameters[0], plane_parameters[1], plane_parameters[2]);
    result.normal_lidar.normalize();
    result.lidar_wrt_ground = Eigen::Quaterniond::FromTwoVectors(
        Eigen::Vector3d(0.0, 0.0, 1.0), result.normal_lidar);
    result.lidar_height = plane_parameters[3];
    result.ground_count = ground.size();
    result.nonground_count = nonground.size();
    return result;
  }

  FullFrontendConfig config;
  FrontendSynchronizer synchronizer;
  ConstantVelocityPropagator propagator;
  std::unique_ptr<LidarOdometryCore> odometry;
  PatchWorkpp<pcl::PointXYZI> patchwork;
  AhrsEstimator ahrs;
  Gril_Calib calibration;
  FrontendState state;
  HardTimeCompensator hard_time;
  Eigen::MatrixXd jacobian;
  std::map<std::uint32_t, RawGroundState> ground_by_scan;
  std::map<std::uint32_t, std::size_t> pending_cuts_by_scan;
  Eigen::Quaterniond imu_wrt_ground = Eigen::Quaterniond::Identity();
  Eigen::Quaterniond lidar_wrt_ground = Eigen::Quaterniond::Identity();
  Eigen::Vector3d normal_lidar = Eigen::Vector3d(0.0, 0.0, 1.0);
  double lidar_height = 0.0;
  int frame_num = 0;
  std::size_t package_count = 0;
  int preprocess_scan_count = 0;
  bool data_accumulation_started = false;
  bool data_accumulation_finished = false;
  double move_start_time_s = 0.0;
};

void drain_packages(PipelineState &pipeline, const FullFrontendRunConfig &run,
                    std::ostream &trace) {
  FrontendMeasureGroup measure;
  while (pipeline.synchronizer.try_sync(measure)) {
    ++pipeline.package_count;
    const double lidar_end_time_s = pipeline.synchronizer.lidar_end_time_s();
    trace << "package " << pipeline.package_count << " "
          << measure.source_scan_index << " " << measure.lidar_beg_time_s << " "
          << lidar_end_time_s << " " << measure.imu.size() << " "
          << measure.lidar.size() << " " << cloud_hash(measure.lidar) << "\n";

    PointCloudXYZI undistorted =
        pipeline.propagator.process(measure, pipeline.state);
    const bool ahrs_initialized =
        pipeline.ahrs.update(measure.imu, pipeline.imu_wrt_ground);

    const auto ground_it =
        pipeline.ground_by_scan.find(
            static_cast<std::uint32_t>(measure.source_scan_index));
    if (ground_it == pipeline.ground_by_scan.end())
      throw std::runtime_error(
          "missing source raw-scan Patchwork++ state");
    const RawGroundState ground = ground_it->second;
    const auto pending_it =
        pipeline.pending_cuts_by_scan.find(
            static_cast<std::uint32_t>(measure.source_scan_index));
    if (pending_it == pipeline.pending_cuts_by_scan.end() ||
        pending_it->second == 0)
      throw std::runtime_error("missing source raw-scan cut count");
    if (--pending_it->second == 0) {
      pipeline.pending_cuts_by_scan.erase(pending_it);
      pipeline.ground_by_scan.erase(ground_it);
    }
    if (ahrs_initialized) {
      pipeline.lidar_wrt_ground = ground.lidar_wrt_ground;
      pipeline.normal_lidar = ground.normal_lidar;
      pipeline.lidar_height = ground.lidar_height;
    }
    trace << "ahrs " << (ahrs_initialized ? 1 : 0) << " "
          << pipeline.imu_wrt_ground.w() << " " << pipeline.imu_wrt_ground.x()
          << " " << pipeline.imu_wrt_ground.y() << " "
          << pipeline.imu_wrt_ground.z() << "\n"
          << "ground_state " << pipeline.lidar_wrt_ground.w() << " "
          << pipeline.lidar_wrt_ground.x() << " "
          << pipeline.lidar_wrt_ground.y() << " "
          << pipeline.lidar_wrt_ground.z() << " "
          << pipeline.normal_lidar.transpose() << " " << pipeline.lidar_height
          << " " << ground.ground_count << " " << ground.nonground_count
          << " " << measure.source_scan_index << "\n";
    write_state(trace, "propagated", pipeline.state);

    LidarGroundEstimate odometry_ground;
    odometry_ground.lidar_ground_rotation = pipeline.lidar_wrt_ground;
    odometry_ground.normal_lidar = pipeline.normal_lidar;
    const LidarOdometryResult odometry_result = pipeline.odometry->process(
        undistorted, odometry_ground, pipeline.state);
    trace << "odometry " << (odometry_result.map_initialized_this_frame ? 1 : 0)
          << " " << (odometry_result.update_performed ? 1 : 0) << " "
          << odometry_result.iterations << " " << odometry_result.rematch_count
          << " " << odometry_result.effective_feature_count << " "
          << odometry_result.deleted_point_count << " "
          << odometry_result.added_point_count << " "
          << odometry_result.map_valid_points << " "
          << odometry_result.residual_mean << " "
          << odometry_result.delta_rotation_deg << " "
          << odometry_result.delta_translation_cm << " "
          << odometry_result.total_distance << " "
          << cloud_hash(odometry_result.downsampled_body) << " "
          << cloud_hash(odometry_result.downsampled_world) << "\n";
    write_state(trace, "updated", pipeline.state);

    if (!odometry_result.update_performed) {
      trace << "frame_skipped map_not_ready\n"
            << "end_package\n";
      continue;
    }

    if (!pipeline.data_accumulation_started &&
        pipeline.state.pos_end.norm() > 0.05) {
      pipeline.data_accumulation_started = true;
      pipeline.move_start_time_s = lidar_end_time_s;
      trace << "motion_start " << pipeline.move_start_time_s << "\n";
    }

    ++pipeline.frame_num;
    trace << "frame " << pipeline.frame_num << " " << lidar_end_time_s << "\n";
    if (pipeline.data_accumulation_started &&
        !pipeline.data_accumulation_finished) {
      pipeline.calibration.push_Lidar_CalibState(
          pipeline.state.rot_end, pipeline.state.pos_end, pipeline.state.bias_g,
          pipeline.state.vel_end, lidar_end_time_s);
      pipeline.calibration.push_Plane_Constraint(
          pipeline.lidar_wrt_ground, pipeline.imu_wrt_ground,
          pipeline.normal_lidar, pipeline.lidar_height);
      trace << "calibration_push " << pipeline.calibration.lidar_state_count()
            << " " << pipeline.calibration.plane_constraint_count() << "\n";

      if (3 * pipeline.frame_num + 2 >=
          static_cast<int>(pipeline.jacobian.rows()))
        throw std::runtime_error(
            "GRIL data-sufficiency Jacobian capacity exceeded");
      pipeline.data_accumulation_finished =
          pipeline.calibration.data_sufficiency_assess(
              pipeline.jacobian, pipeline.frame_num, pipeline.state.bias_g,
              pipeline.config.orig_odom_freq, pipeline.config.cut_frame_num,
              pipeline.lidar_wrt_ground, pipeline.imu_wrt_ground,
              pipeline.lidar_height);
      trace << "data_sufficiency "
            << (pipeline.data_accumulation_finished ? 1 : 0) << "\n";

      if (pipeline.data_accumulation_finished) {
        pipeline.odometry.reset();
        trace << "odometry_quiesced\n";
        pipeline.calibration.dump_batch_trace_v1(
            run.batch_trace_path, pipeline.config.orig_odom_freq,
            pipeline.config.cut_frame_num, pipeline.hard_time.hard_offset_s(),
            pipeline.move_start_time_s);
        trace << "batch_handoff " << pipeline.calibration.all_imu_sample_count()
              << " " << pipeline.calibration.lidar_state_count() << " "
              << pipeline.calibration.plane_constraint_count() << " "
              << pipeline.hard_time.hard_offset_s() << " "
              << pipeline.move_start_time_s << "\n";
        trace.flush();
        run_batch_process(run);
        trace << "batch_complete\nend_package\nEND\n";
        return;
      }
    }
    trace << "end_package\n";
  }
}

} // namespace

HardTimeCompensator::HardTimeCompensator(double configured_time_lag_s)
    : configured_time_lag_s_(configured_time_lag_s) {}

bool HardTimeCompensator::lidar_rolled_back(double raw_timestamp_s) const {
  return raw_timestamp_s < last_lidar_timestamp_s_;
}

bool HardTimeCompensator::observe_lidar(double raw_timestamp_s,
                                        bool imu_queue_nonempty) {
  last_lidar_timestamp_s_ = raw_timestamp_s;
  if (std::abs(last_imu_timestamp_s_ - last_lidar_timestamp_s_) > 1.0 &&
      !hard_offset_locked_ && imu_queue_nonempty) {
    hard_offset_locked_ = true;
    hard_offset_s_ = last_imu_timestamp_s_ - last_lidar_timestamp_s_;
    return true;
  }
  return false;
}

double HardTimeCompensator::compensate_imu(double raw_timestamp_s) const {
  return raw_timestamp_s - hard_offset_s_ - configured_time_lag_s_;
}

bool HardTimeCompensator::imu_rolled_back(
    double compensated_timestamp_s) const {
  return compensated_timestamp_s < last_imu_timestamp_s_;
}

void HardTimeCompensator::observe_imu(double compensated_timestamp_s) {
  last_imu_timestamp_s_ = compensated_timestamp_s;
}

double HardTimeCompensator::hard_offset_s() const { return hard_offset_s_; }

bool HardTimeCompensator::hard_offset_locked() const {
  return hard_offset_locked_;
}

double HardTimeCompensator::last_lidar_timestamp_s() const {
  return last_lidar_timestamp_s_;
}

double HardTimeCompensator::last_imu_timestamp_s() const {
  return last_imu_timestamp_s_;
}

FullFrontendConfig read_full_frontend_config(const std::string &path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("cannot open full frontend config: " + path);
  expect(input, "GRIL_NATIVE_FULL_CONFIG");
  int version = 0;
  input >> version;
  if (version != 1)
    throw std::runtime_error("unsupported full frontend config version");

  FullFrontendConfig config;
  expect(input, "preprocess");
  input >> config.lidar_type >> config.scan_line >> config.blind >>
      config.point_filter_num >> config.feature_extract_enabled >>
      config.cut_frame >> config.cut_frame_num;

  expect(input, "calibration");
  input >> config.orig_odom_freq >> config.mean_acc_norm >>
      config.data_accum_length >> config.x_accumulate >> config.y_accumulate >>
      config.z_accumulate >> config.svd_threshold >> config.imu_sensor_height >>
      config.trans_IL_x >> config.trans_IL_y >> config.trans_IL_z >>
      config.bound_th >> config.set_boundary >> config.verbose >>
      config.gyro_factor >> config.acc_factor >> config.ground_factor;

  expect(input, "mapping");
  double gyr_cov = 0.0;
  double acc_cov = 0.0;
  input >> config.odometry.max_iterations >> config.odometry.cube_side_length >>
      config.odometry.filter_size_surf >> config.odometry.filter_size_map >>
      gyr_cov >> acc_cov >> config.odometry.detection_range >>
      config.odometry.ground_covariance;
  config.gyr_cov = Eigen::Vector3d::Constant(gyr_cov);
  config.acc_cov = Eigen::Vector3d::Constant(acc_cov);

  expect(input, "patchwork");
  input >> config.patchwork.sensor_height >> config.patchwork.num_iter >>
      config.patchwork.num_lpr >> config.patchwork.num_min_pts >>
      config.patchwork.max_flatness_storage >>
      config.patchwork.max_elevation_storage >> config.patchwork.th_seeds >>
      config.patchwork.th_dist >> config.patchwork.th_seeds_v >>
      config.patchwork.th_dist_v >> config.patchwork.max_range >>
      config.patchwork.min_range >> config.patchwork.uprightness_thr >>
      config.patchwork.adaptive_seed_selection_margin >>
      config.patchwork.rnr_ver_angle_thr >>
      config.patchwork.rnr_intensity_thr >> config.patchwork.verbose >>
      config.patchwork.enable_rnr >> config.patchwork.enable_rvpf >>
      config.patchwork.enable_tgr;
  config.patchwork.num_sectors_each_zone = read_vector<int>(input, "sectors");
  config.patchwork.num_rings_each_zone = read_vector<int>(input, "rings");
  config.patchwork.elevation_thresholds =
      read_vector<double>(input, "elevation");
  config.patchwork.flatness_thresholds = read_vector<double>(input, "flatness");

  expect(input, "runtime");
  input >> config.configured_time_lag_s;
  expect(input, "END");
  std::string trailing;
  if (input >> trailing)
    throw std::runtime_error("content after full frontend config END");
  validate_config(config);
  return config;
}

void run_full_frontend(const FullFrontendRunConfig &run) {
  if (run.input_path.empty() || run.config_path.empty() ||
      run.result_path.empty() || run.trace_path.empty() ||
      run.batch_trace_path.empty() || run.batch_executable_path.empty() ||
      run.batch_config_path.empty())
    throw std::runtime_error("full frontend paths must all be specified");
  if (run.forward_gap_policy == ForwardGapPolicy::Reset &&
      (!(run.forward_gap_s > 0.0) || !std::isfinite(run.forward_gap_s)))
    throw std::runtime_error(
        "reset gap policy requires positive --forward-gap-s");

  const FullFrontendConfig config = read_full_frontend_config(run.config_path);
  std::ifstream input(run.input_path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot open native dataset: " + run.input_path);
  std::ofstream trace(run.trace_path);
  if (!trace)
    throw std::runtime_error("cannot create full frontend trace: " +
                             run.trace_path);
  trace << std::setprecision(17);
  trace << "GRIL_FULL_FRONTEND_TRACE 1\n"
        << "gap_policy "
        << (run.forward_gap_policy == ForwardGapPolicy::GoldenEquivalence
                ? "golden_equivalence"
                : "reset")
        << " " << run.forward_gap_s << "\n";

  std::string magic;
  std::getline(input, magic);
  if (magic != "GRIL_NATIVE_DATASET 1")
    throw std::runtime_error("unsupported native dataset version");
  std::uint64_t event_count = 0;
  read_binary(input, event_count, "event count");
  trace << "events " << event_count << "\n";

  std::unique_ptr<PipelineState> pipeline(new PipelineState(config));
  double previous_event_timestamp_s = 0.0;
  bool have_previous_event = false;
  for (std::uint64_t event_index = 0; event_index < event_count;
       ++event_index) {
    std::uint8_t event_type = 0;
    read_binary(input, event_type, "event type");
    std::int64_t timestamp_ns = 0;

    if (event_type == kImuEvent) {
      read_binary(input, timestamp_ns, "IMU timestamp");
      double values[6];
      input.read(reinterpret_cast<char *>(values), sizeof(values));
      if (!input)
        throw std::runtime_error("truncated native dataset at IMU values");
      const double raw_timestamp_s = ros_time_to_seconds(timestamp_ns);
      if (run.forward_gap_policy == ForwardGapPolicy::Reset &&
          have_previous_event &&
          raw_timestamp_s - previous_event_timestamp_s > run.forward_gap_s) {
        trace << "forward_gap_reset " << event_index << " "
              << previous_event_timestamp_s << " " << raw_timestamp_s << "\n";
        pipeline.reset(new PipelineState(config));
      }
      previous_event_timestamp_s = raw_timestamp_s;
      have_previous_event = true;

      const double compensated_timestamp_s =
          pipeline->hard_time.compensate_imu(raw_timestamp_s);
      const bool rollback =
          pipeline->hard_time.imu_rolled_back(compensated_timestamp_s);
      if (rollback) {
        pipeline->synchronizer.clear_imu();
        pipeline->calibration.IMU_buffer_clear();
      }
      pipeline->hard_time.observe_imu(compensated_timestamp_s);
      FrontendImuSample sample;
      sample.timestamp_s = compensated_timestamp_s;
      sample.angular_velocity =
          Eigen::Vector3d(values[0], values[1], values[2]);
      sample.linear_acceleration =
          Eigen::Vector3d(values[3], values[4], values[5]);
      pipeline->synchronizer.push_imu(sample);
      if (!pipeline->data_accumulation_finished) {
        pipeline->calibration.push_ALL_IMU_CalibState(
            sample.angular_velocity, sample.linear_acceleration,
            sample.timestamp_s, pipeline->config.mean_acc_norm);
      }
      trace << "imu_event " << event_index << " " << raw_timestamp_s << " "
            << compensated_timestamp_s << " " << (rollback ? 1 : 0) << " "
            << pipeline->synchronizer.imu_buffer_size() << " "
            << pipeline->calibration.all_imu_sample_count() << "\n";
    } else if (event_type == kLidarEvent) {
      std::uint32_t scan_number = 0;
      std::uint64_t point_count = 0;
      read_binary(input, scan_number, "LiDAR scan number");
      read_binary(input, timestamp_ns, "LiDAR timestamp");
      read_binary(input, point_count, "LiDAR point count");
      std::vector<VelodynePoint> points;
      points.reserve(point_count);
      for (std::uint64_t point_index = 0; point_index < point_count;
           ++point_index) {
        float values[5];
        std::uint16_t ring = 0;
        input.read(reinterpret_cast<char *>(values), sizeof(values));
        read_binary(input, ring, "LiDAR ring");
        if (!input)
          throw std::runtime_error("truncated native dataset at LiDAR point");
        VelodynePoint point;
        point.x = values[0];
        point.y = values[1];
        point.z = values[2];
        point.intensity = values[3];
        point.time_s = values[4];
        point.ring = ring;
        points.push_back(point);
      }

      const double raw_timestamp_s = ros_time_to_seconds(timestamp_ns);
      if (run.forward_gap_policy == ForwardGapPolicy::Reset &&
          have_previous_event &&
          raw_timestamp_s - previous_event_timestamp_s > run.forward_gap_s) {
        trace << "forward_gap_reset " << event_index << " "
              << previous_event_timestamp_s << " " << raw_timestamp_s << "\n";
        pipeline.reset(new PipelineState(config));
      }
      previous_event_timestamp_s = raw_timestamp_s;
      have_previous_event = true;

      const bool rollback =
          pipeline->hard_time.lidar_rolled_back(raw_timestamp_s);
      if (rollback) {
        pipeline->synchronizer.clear_lidar();
        pipeline->ground_by_scan.clear();
        pipeline->pending_cuts_by_scan.clear();
      }
      const bool hard_offset_locked = pipeline->hard_time.observe_lidar(
          raw_timestamp_s, pipeline->synchronizer.imu_buffer_size() > 0);

      const RawGroundState ground = pipeline->segment_ground(points);

      VelodynePreprocessConfig preprocess;
      preprocess.blind = pipeline->config.blind;
      preprocess.point_filter_num = pipeline->config.point_filter_num;
      preprocess.n_scans = pipeline->config.scan_line;
      preprocess.required_frame_num = pipeline->config.cut_frame_num;
      preprocess.scan_count = ++pipeline->preprocess_scan_count;
      const VelodynePreprocessResult result =
          preprocess_velodyne_scan(points, raw_timestamp_s, preprocess);
      if (!result.cut_clouds.empty()) {
        const auto inserted = pipeline->ground_by_scan.emplace(
            scan_number, ground);
        if (!inserted.second)
          throw std::runtime_error("duplicate source LiDAR scan number");
        pipeline->pending_cuts_by_scan.emplace(
            scan_number, result.cut_clouds.size());
      }
      for (std::size_t cut_index = 0; cut_index < result.cut_clouds.size();
           ++cut_index) {
        pipeline->synchronizer.push_lidar(
            result.cut_clouds[cut_index],
            result.cut_timestamps_ms[cut_index] / 1000.0, scan_number);
      }
      trace << "lidar_event " << event_index << " " << scan_number << " "
            << pipeline->preprocess_scan_count << " " << raw_timestamp_s << " "
            << point_count << " "
            << result.surface.size() << " " << result.cut_clouds.size() << " "
            << ground.ground_count << " " << ground.nonground_count << " "
            << (rollback ? 1 : 0) << " " << (hard_offset_locked ? 1 : 0) << " "
            << pipeline->hard_time.hard_offset_s() << " "
            << pipeline->synchronizer.lidar_buffer_size() << "\n";
    } else {
      throw std::runtime_error("unknown native dataset event type");
    }

    drain_packages(*pipeline, run, trace);
    if (pipeline->data_accumulation_finished)
      return;
  }

  if (input.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("content after native dataset events");
  trace << "insufficient_data " << pipeline->frame_num << " "
        << pipeline->calibration.all_imu_sample_count() << " "
        << pipeline->calibration.lidar_state_count() << "\nEND\n";
  throw std::runtime_error(
      "native GRIL exhausted input before data_sufficiency_assess passed");
}
