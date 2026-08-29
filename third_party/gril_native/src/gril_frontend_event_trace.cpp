/*
 * Replay canonical LiDAR/IMU events through GRIL preprocessing and sync.
 * Copyright (C) 2026 whl-cal contributors.
 * GPL-2.0-only; see ../LICENSE.
 */

#include <Gril_Calib/FrontendCore.h>
#include <Gril_Calib/VelodynePreprocess.h>

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(std::istream &input, const std::string &expected) {
    std::string actual;
    if (!(input >> actual) || actual != expected)
        throw std::runtime_error("expected token: " + expected);
}

void write_cloud(std::ostream &output, const PointCloudXYZI &cloud) {
    output << "input " << cloud.size() << "\n";
    for (const auto &point : cloud.points) {
        output << "point "
               << point.x << " " << point.y << " " << point.z << " "
               << point.intensity << " " << point.curvature << "\n";
    }
}

void drain(
    FrontendSynchronizer &synchronizer,
    std::ostream &output,
    int &package_count,
    int package_limit) {
    FrontendMeasureGroup measure;
    while ((package_limit == 0 || package_count < package_limit) &&
           synchronizer.try_sync(measure)) {
        package_count++;
        output << "package " << package_count << " "
               << measure.lidar_beg_time_s << " "
               << synchronizer.lidar_end_time_s() << "\n";
        output << "imu " << measure.imu.size() << "\n";
        for (const auto &sample : measure.imu)
            output << "imu_sample " << sample.timestamp_s << "\n";
        write_cloud(output, measure.lidar);
        output << "end_package\n";
    }
}

}  // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 3 && argc != 4)
            throw std::runtime_error(
                "usage: gril_native_frontend_event_trace INPUT OUTPUT [--all]");
        const bool process_all =
            argc == 4 && std::string(argv[3]) == "--all";
        if (argc == 4 && !process_all)
            throw std::runtime_error("unsupported frontend event option");
        const int package_limit = process_all ? 0 : 25;
        std::ifstream input(argv[1]);
        std::ofstream output(argv[2]);
        if (!input || !output)
            throw std::runtime_error("could not open frontend event trace");

        expect(input, "GRIL_FRONTEND_EVENT_INPUT");
        int version = 0;
        input >> version;
        if (version != 1)
            throw std::runtime_error("unsupported frontend event version");
        VelodynePreprocessConfig config;
        expect(input, "config");
        input >> config.blind >> config.point_filter_num >> config.n_scans
              >> config.required_frame_num;
        expect(input, "events");
        std::size_t event_count = 0;
        input >> event_count;

        FrontendSynchronizer synchronizer;
        output << std::setprecision(17);
        output << "GRIL_SYNC_TRACE 1\n";
        int package_count = 0;
        for (std::size_t event_index = 0; event_index < event_count; ++event_index) {
            std::string event_type;
            input >> event_type;
            if (event_type == "imu") {
                std::int64_t timestamp_ns = 0;
                input >> timestamp_ns;
                FrontendImuSample sample;
                sample.timestamp_s =
                    static_cast<double>(timestamp_ns) / 1000000000.0;
                synchronizer.push_imu(sample);
            } else if (event_type == "lidar") {
                int scan_count = 0;
                std::int64_t timestamp_ns = 0;
                std::size_t point_count = 0;
                input >> scan_count >> timestamp_ns >> point_count;
                std::vector<VelodynePoint> points;
                points.reserve(point_count);
                for (std::size_t index = 0; index < point_count; ++index) {
                    expect(input, "point");
                    VelodynePoint point;
                    input >> point.x >> point.y >> point.z
                          >> point.intensity >> point.time_s >> point.ring;
                    points.push_back(point);
                }
                config.scan_count = scan_count;
                const double timestamp_s =
                    static_cast<double>(timestamp_ns) / 1000000000.0;
                const auto result =
                    preprocess_velodyne_scan(points, timestamp_s, config);
                for (std::size_t index = 0;
                     index < result.cut_clouds.size();
                     ++index) {
                    synchronizer.push_lidar(
                        result.cut_clouds[index],
                        result.cut_timestamps_ms[index] / 1000.0);
                }
            } else {
                throw std::runtime_error("unknown frontend event");
            }
            drain(synchronizer, output, package_count, package_limit);
        }
        expect(input, "END");
        drain(synchronizer, output, package_count, package_limit);
        if (!process_all && package_count != package_limit)
            throw std::runtime_error("frontend event replay did not produce 25 packages");
        if (process_all && package_count == 0)
            throw std::runtime_error("frontend event replay produced no packages");
        output << "END\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
