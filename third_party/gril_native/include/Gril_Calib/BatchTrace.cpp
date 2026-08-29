/*
 * Versioned GRIL batch trace I/O.
 * Copyright (C) 2026 whl-cal contributors.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License, version 2.
 * See ../../LICENSE.
 */

#include "BatchTrace.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {

void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error("invalid GRIL batch trace: " + message);
    }
}

void require_line_end(std::istringstream &line, const std::string &context) {
    std::string extra;
    require(!(line >> extra), "unexpected field in " + context);
}

double read_finite(std::istringstream &line, const std::string &context) {
    double value = 0.0;
    require(static_cast<bool>(line >> value), "missing number in " + context);
    require(std::isfinite(value), "non-finite number in " + context);
    return value;
}

std::string next_line(std::istream &input, const std::string &context) {
    std::string line;
    require(static_cast<bool>(std::getline(input, line)),
            "unexpected end while reading " + context);
    require(!line.empty(), "empty line while reading " + context);
    return line;
}

void read_key_int(std::istream &input, const char *expected, int &value) {
    std::istringstream line(next_line(input, expected));
    std::string key;
    require(static_cast<bool>(line >> key >> value),
            std::string("invalid ") + expected);
    require(key == expected, std::string("expected ") + expected);
    require_line_end(line, expected);
}

void read_key_double(
    std::istream &input, const char *expected, double &value) {
    std::istringstream line(next_line(input, expected));
    std::string key;
    require(static_cast<bool>(line >> key),
            std::string("invalid ") + expected);
    require(key == expected, std::string("expected ") + expected);
    value = read_finite(line, expected);
    require_line_end(line, expected);
}

std::size_t read_count(std::istream &input, const char *expected) {
    std::istringstream line(next_line(input, expected));
    std::string key;
    unsigned long long count = 0;
    require(static_cast<bool>(line >> key >> count),
            std::string("invalid ") + expected);
    require(key == expected, std::string("expected ") + expected);
    require(count <= std::numeric_limits<std::size_t>::max(),
            std::string("count overflow in ") + expected);
    require_line_end(line, expected);
    return static_cast<std::size_t>(count);
}

CalibState read_state(std::istream &input, const char *record_name) {
    std::istringstream line(next_line(input, record_name));
    std::string key;
    require(static_cast<bool>(line >> key), "missing state record name");
    require(key == record_name, std::string("expected ") + record_name);

    CalibState state;
    state.timeStamp = read_finite(line, record_name);
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            state.rot_end(row, col) = read_finite(line, record_name);
        }
    }
    for (int i = 0; i < 3; i++)
        state.pos_end[i] = read_finite(line, record_name);
    for (int i = 0; i < 3; i++)
        state.ang_vel[i] = read_finite(line, record_name);
    for (int i = 0; i < 3; i++)
        state.linear_vel[i] = read_finite(line, record_name);
    for (int i = 0; i < 3; i++)
        state.ang_acc[i] = read_finite(line, record_name);
    for (int i = 0; i < 3; i++)
        state.linear_acc[i] = read_finite(line, record_name);
    require_line_end(line, record_name);
    return state;
}

GroundConstraint read_ground(std::istream &input) {
    std::istringstream line(next_line(input, "ground"));
    std::string key;
    require(static_cast<bool>(line >> key), "missing ground record name");
    require(key == "ground", "expected ground");

    GroundConstraint constraint;
    const double lidar_w = read_finite(line, "ground");
    const double lidar_x = read_finite(line, "ground");
    const double lidar_y = read_finite(line, "ground");
    const double lidar_z = read_finite(line, "ground");
    constraint.lidar_wrt_ground =
        QD(lidar_w, lidar_x, lidar_y, lidar_z);
    const double imu_w = read_finite(line, "ground");
    const double imu_x = read_finite(line, "ground");
    const double imu_y = read_finite(line, "ground");
    const double imu_z = read_finite(line, "ground");
    constraint.imu_wrt_ground = QD(imu_w, imu_x, imu_y, imu_z);
    for (int i = 0; i < 3; i++)
        constraint.normal_lidar[i] = read_finite(line, "ground");
    constraint.distance_lidar = read_finite(line, "ground");
    require_line_end(line, "ground");
    return constraint;
}

void write_state(
    std::ostream &output, const char *record_name,
    const CalibState &state) {
    output << record_name << " " << state.timeStamp;
    for (int row = 0; row < 3; row++)
        for (int col = 0; col < 3; col++)
            output << " " << state.rot_end(row, col);
    output << " " << state.pos_end.transpose()
           << " " << state.ang_vel.transpose()
           << " " << state.linear_vel.transpose()
           << " " << state.ang_acc.transpose()
           << " " << state.linear_acc.transpose() << "\n";
}

}  // namespace

BatchTrace read_batch_trace(std::istream &input) {
    {
        std::istringstream line(next_line(input, "header"));
        std::string magic;
        int version = 0;
        require(static_cast<bool>(line >> magic >> version),
                "invalid header");
        require(magic == "GRIL_BATCH_TRACE", "wrong magic");
        require(version == 1, "unsupported version");
        require_line_end(line, "header");
    }

    BatchTrace trace;
    read_key_int(input, "orig_odom_freq", trace.orig_odom_freq);
    read_key_int(input, "cut_frame_num", trace.cut_frame_num);
    read_key_double(
        input, "timediff_imu_wrt_lidar",
        trace.timediff_imu_wrt_lidar);
    read_key_double(input, "move_start_time", trace.move_start_time);

    const std::size_t imu_count = read_count(input, "imu_states");
    for (std::size_t i = 0; i < imu_count; i++)
        trace.normalized_imu_states.push_back(read_state(input, "imu"));

    const std::size_t lidar_count = read_count(input, "lidar_states");
    for (std::size_t i = 0; i < lidar_count; i++)
        trace.lidar_states.push_back(read_state(input, "lidar"));

    const std::size_t ground_count =
        read_count(input, "ground_constraints");
    for (std::size_t i = 0; i < ground_count; i++)
        trace.ground_constraints.push_back(read_ground(input));

    require(next_line(input, "END") == "END", "expected END");
    std::string trailing;
    while (std::getline(input, trailing))
        require(trailing.empty(), "content after END");
    require(trace.lidar_states.size() == trace.ground_constraints.size(),
            "LiDAR and ground-constraint counts differ");
    return trace;
}

BatchTrace read_batch_trace_file(const std::string &path) {
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("cannot open GRIL batch trace: " + path);
    return read_batch_trace(input);
}

void write_batch_trace(std::ostream &output, const BatchTrace &trace) {
    output << std::setprecision(17);
    output << "GRIL_BATCH_TRACE 1\n"
           << "orig_odom_freq " << trace.orig_odom_freq << "\n"
           << "cut_frame_num " << trace.cut_frame_num << "\n"
           << "timediff_imu_wrt_lidar "
           << trace.timediff_imu_wrt_lidar << "\n"
           << "move_start_time " << trace.move_start_time << "\n"
           << "imu_states " << trace.normalized_imu_states.size() << "\n";
    for (const auto &state : trace.normalized_imu_states)
        write_state(output, "imu", state);
    output << "lidar_states " << trace.lidar_states.size() << "\n";
    for (const auto &state : trace.lidar_states)
        write_state(output, "lidar", state);
    output << "ground_constraints "
           << trace.ground_constraints.size() << "\n";
    for (const auto &constraint : trace.ground_constraints) {
        output << "ground "
               << constraint.lidar_wrt_ground.w() << " "
               << constraint.lidar_wrt_ground.x() << " "
               << constraint.lidar_wrt_ground.y() << " "
               << constraint.lidar_wrt_ground.z() << " "
               << constraint.imu_wrt_ground.w() << " "
               << constraint.imu_wrt_ground.x() << " "
               << constraint.imu_wrt_ground.y() << " "
               << constraint.imu_wrt_ground.z() << " "
               << constraint.normal_lidar.transpose() << " "
               << constraint.distance_lidar << "\n";
    }
    output << "END\n";
    if (!output)
        throw std::runtime_error("failed to write GRIL batch trace");
}

void write_batch_trace_file(
    const std::string &path, const BatchTrace &trace) {
    std::ofstream output(path);
    if (!output)
        throw std::runtime_error("cannot create GRIL batch trace: " + path);
    write_batch_trace(output, trace);
}

bool validate_batch_trace_for_calibration(
    const BatchTrace &trace, std::string *reason) {
    std::string failure;
    if (trace.orig_odom_freq <= 0)
        failure = "orig_odom_freq must be positive";
    else if (trace.cut_frame_num <= 0)
        failure = "cut_frame_num must be positive";
    else if (trace.lidar_states.size() !=
             trace.ground_constraints.size())
        failure = "each LiDAR state must have one ground constraint";
    else {
        for (std::size_t i = 1;
             i < trace.normalized_imu_states.size(); i++) {
            if (trace.normalized_imu_states[i].timeStamp <=
                trace.normalized_imu_states[i - 1].timeStamp) {
                failure =
                    "normalized IMU timestamps must be strictly increasing";
                break;
            }
        }
        for (std::size_t i = 1;
             failure.empty() && i < trace.lidar_states.size(); i++) {
            if (trace.lidar_states[i].timeStamp <=
                trace.lidar_states[i - 1].timeStamp) {
                failure =
                    "LiDAR timestamps must be strictly increasing";
                break;
            }
        }
        std::size_t first_imu = 0;
        while (failure.empty() &&
               first_imu < trace.normalized_imu_states.size() &&
               trace.normalized_imu_states[first_imu].timeStamp <
                   trace.move_start_time - 3.0)
            first_imu++;
        if (failure.empty() &&
            trace.normalized_imu_states.size() - first_imu < 2)
            failure =
                "fewer than two normalized IMU states remain after "
                "move_start_time trimming";

        std::size_t first_lidar = 0;
        while (failure.empty() &&
               first_lidar < trace.lidar_states.size() &&
               trace.lidar_states[first_lidar].timeStamp <
                   trace.move_start_time - 3.0)
            first_lidar++;
        if (failure.empty() && first_lidar == trace.lidar_states.size())
            failure =
                "no LiDAR states remain after move_start_time trimming";

        std::size_t interpolated_count = 0;
        std::size_t imu_index = first_imu + 1;
        for (std::size_t lidar_index = first_lidar;
             failure.empty() &&
             lidar_index < trace.lidar_states.size();
             lidar_index++) {
            const double lidar_time =
                trace.lidar_states[lidar_index].timeStamp;
            while (imu_index < trace.normalized_imu_states.size() &&
                   trace.normalized_imu_states[imu_index].timeStamp <=
                       lidar_time)
                imu_index++;
            if (imu_index < trace.normalized_imu_states.size() &&
                trace.normalized_imu_states[imu_index - 1].timeStamp <=
                    lidar_time)
                interpolated_count++;
        }
        if (failure.empty() && interpolated_count < 92)
            failure =
                "fewer than 92 interpolated IMU/LiDAR states would remain; "
                "upstream trimming and two filtering passes require 92";
    }
    if (reason)
        *reason = failure;
    return failure.empty();
}
