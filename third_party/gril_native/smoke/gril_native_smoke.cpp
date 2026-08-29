/*
 * Native API smoke check, added 2026-08-28.
 * GPL-2.0-only; see ../LICENSE.
 */

#include <Gril_Calib/Gril_Calib.h>

#include <iostream>

int main() {
    Gril_Calib calibration;
    calibration.push_ALL_IMU_CalibState(
        V3D(0.01, -0.02, 0.03), V3D(0.0, 0.0, 9.81), 1.0, 9.81);
    calibration.push_IMU_CalibState(
        V3D(0.01, -0.02, 0.03), V3D(0.0, 0.0, 9.81), 1.0);
    calibration.push_Lidar_CalibState(
        M3D::Identity(), V3D::Zero(), V3D(0.01, -0.02, 0.03),
        V3D::Zero(), 1.0);
    calibration.push_Plane_Constraint(
        QD::Identity(), QD::Identity(), V3D::UnitZ(), 1.0);

    const bool ingested =
        calibration.all_imu_sample_count() == 1 &&
        calibration.imu_state_count() == 1 &&
        calibration.lidar_state_count() == 1 &&
        calibration.plane_constraint_count() == 1;
    if (!ingested) {
        std::cerr << "GRIL native ingestion smoke check failed\n";
        return 1;
    }

    std::cout << "GRIL native API construction and ingestion passed; "
                 "calibration intentionally not run on insufficient data.\n";
    return 0;
}
