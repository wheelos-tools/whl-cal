/*
 * Targeted full-frontend semantics tests.
 * Copyright (C) 2026 whl-cal contributors.
 * GPL-2.0-only; see ../LICENSE.
 */

#include <Gril_Calib/FullFrontend.h>
#include <Gril_Calib/Gril_Calib.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

void hard_time() {
  HardTimeCompensator clock(0.02);
  clock.observe_imu(clock.compensate_imu(10.0));
  require(clock.observe_lidar(8.0, true), "hard offset was not locked");
  require(clock.hard_offset_locked(), "hard offset flag is false");
  require(std::abs(clock.hard_offset_s() - 1.98) < 1e-12,
          "hard offset does not use the compensated last IMU stamp");
  require(!clock.observe_lidar(7.0, true),
          "hard offset was locked more than once");
  require(std::abs(clock.compensate_imu(12.0) - 10.0) < 1e-12,
          "IMU compensation order changed");
}

void rollback() {
  HardTimeCompensator clock;
  clock.observe_imu(5.0);
  require(clock.imu_rolled_back(4.0), "IMU rollback not detected");
  require(!clock.imu_rolled_back(5.0), "equal IMU stamp is rollback");

  FrontendSynchronizer synchronizer;
  FrontendImuSample sample;
  sample.timestamp_s = 5.0;
  synchronizer.push_imu(sample);
  synchronizer.clear_imu();
  require(synchronizer.imu_buffer_size() == 0,
          "IMU rollback did not clear the FIFO");

  Gril_Calib calibration;
  calibration.push_ALL_IMU_CalibState(
      Eigen::Vector3d::Zero(), Eigen::Vector3d(0.0, 0.0, 9.81), 5.0, 9.81);
  calibration.IMU_buffer_clear();
  require(calibration.all_imu_sample_count() == 0,
          "IMU rollback did not clear IMU_state_group_ALL");

  clock.observe_lidar(7.0, false);
  require(clock.lidar_rolled_back(6.0), "LiDAR rollback not detected");
  require(!clock.lidar_rolled_back(7.0), "equal LiDAR stamp is rollback");
}

void source_scan_pairing() {
  FrontendSynchronizer synchronizer;
  PointCloudXYZI cloud;
  PointType first;
  first.curvature = 0.0F;
  PointType last;
  last.curvature = 10.0F;
  cloud.push_back(first);
  cloud.push_back(last);
  synchronizer.push_lidar(cloud, 1.0, 42);
  FrontendImuSample before;
  before.timestamp_s = 1.005;
  FrontendImuSample after;
  after.timestamp_s = 1.02;
  synchronizer.push_imu(before);
  synchronizer.push_imu(after);
  FrontendMeasureGroup measure;
  require(synchronizer.try_sync(measure), "package did not synchronize");
  require(measure.source_scan_index == 42, "raw scan pairing was not retained");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("usage: gril_native_full_test MODE");
    const std::string mode(argv[1]);
    if (mode == "hard_time")
      hard_time();
    else if (mode == "rollback")
      rollback();
    else if (mode == "source_scan_pairing")
      source_scan_pairing();
    else
      throw std::runtime_error("unknown test mode");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
