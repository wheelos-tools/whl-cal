"""Canonical event replay inputs and sync-only trace extraction."""

from __future__ import annotations

from pathlib import Path

from gril.config import load_algorithm_config
from gril.dataset_io import load_dataset


def write_frontend_event_input(
    dataset_path: Path,
    gril_config_path: Path,
    output_path: Path,
    *,
    lidar_scan_count: int = 21,
) -> Path:
    algorithm = load_algorithm_config(gril_config_path)
    preprocess = algorithm["preprocess"]
    calibration = algorithm["calibration"]
    required = {
        "blind": preprocess,
        "point_filter_num": preprocess,
        "scan_line": preprocess,
        "cut_frame_num": calibration,
    }
    missing = [key for key, section in required.items() if key not in section]
    if missing:
        raise ValueError("GRIL frontend config is missing: " + ", ".join(missing))

    dataset = load_dataset(dataset_path)
    lidar = dataset.lidar.normalized()
    imu = dataset.imu.normalized()
    if len(lidar.scan_timestamps_ns) < lidar_scan_count:
        raise ValueError(f"Frontend event replay requires {lidar_scan_count} scans")

    last_scan = lidar_scan_count - 1
    last_start = int(lidar.scan_offsets[last_scan])
    last_end = int(lidar.scan_offsets[last_scan + 1])
    last_point_offset_ns = int(
        round(float(lidar.point_time_s[last_start:last_end].max()) * 1e9)
    )
    end_ns = int(lidar.scan_timestamps_ns[last_scan]) + last_point_offset_ns
    imu_end = int(imu.timestamps_ns.searchsorted(end_ns, side="left")) + 1
    imu_end = min(imu_end, len(imu.timestamps_ns))

    events: list[tuple[int, int, str, int]] = []
    for scan_index in range(lidar_scan_count):
        events.append(
            (
                int(lidar.scan_timestamps_ns[scan_index]),
                1,
                "lidar",
                scan_index,
            )
        )
    for imu_index in range(imu_end):
        events.append((int(imu.timestamps_ns[imu_index]), 0, "imu", imu_index))
    events.sort()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output:
        output.write("GRIL_FRONTEND_EVENT_INPUT 1\n")
        output.write(
            "config "
            f"{float(preprocess['blind']):.17g} "
            f"{int(preprocess['point_filter_num'])} "
            f"{int(preprocess['scan_line'])} "
            f"{int(calibration['cut_frame_num'])}\n"
        )
        output.write(f"events {len(events)}\n")
        for timestamp_ns, _, event_type, index in events:
            if event_type == "imu":
                output.write(f"imu {timestamp_ns}\n")
                continue
            start = int(lidar.scan_offsets[index])
            end = int(lidar.scan_offsets[index + 1])
            output.write(f"lidar {index + 1} {timestamp_ns} {end - start}\n")
            for point_index in range(start, end):
                xyz = lidar.xyz[point_index]
                output.write(
                    "point "
                    f"{float(xyz[0]):.9g} {float(xyz[1]):.9g} "
                    f"{float(xyz[2]):.9g} "
                    f"{float(lidar.intensity[point_index]):.9g} "
                    f"{float(lidar.point_time_s[point_index]):.9g} "
                    f"{int(lidar.ring[point_index])}\n"
                )
        output.write("END\n")
    return output_path


def extract_sync_trace(frontend_trace_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with frontend_trace_path.open() as source, output_path.open("w") as output:
        header = source.readline()
        if header.strip() != "GRIL_FRONTEND_CV_TRACE 1":
            raise ValueError("Unsupported GRIL frontend trace")
        config = source.readline()
        if not config.startswith("config "):
            raise ValueError("Frontend trace is missing config")
        output.write("GRIL_SYNC_TRACE 1\n")

        while True:
            line = source.readline()
            if not line:
                raise ValueError("Frontend trace is missing END")
            if line.strip() == "END":
                output.write("END\n")
                return output_path
            if not line.startswith("package "):
                raise ValueError("Frontend trace package structure is invalid")
            output.write(line)

            imu_header = source.readline()
            output.write(imu_header)
            imu_count = int(imu_header.split()[1])
            for _ in range(imu_count):
                output.write(source.readline())

            if not source.readline().startswith("before "):
                raise ValueError("Frontend trace is missing before state")
            input_header = source.readline()
            if not input_header.startswith("input "):
                raise ValueError("Frontend trace is missing input cloud")
            output.write(input_header)
            input_count = int(input_header.split()[1])
            for _ in range(input_count):
                output.write(source.readline())

            if not source.readline().startswith("after "):
                raise ValueError("Frontend trace is missing after state")
            output_header = source.readline()
            if not output_header.startswith("output "):
                raise ValueError("Frontend trace is missing output cloud")
            output_count = int(output_header.split()[1])
            for _ in range(output_count):
                source.readline()
            if source.readline().strip() != "end_package":
                raise ValueError("Frontend trace package is not terminated")
            output.write("end_package\n")
