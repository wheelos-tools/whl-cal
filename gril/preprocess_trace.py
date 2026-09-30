"""Inputs and comparison for GRIL preprocessing equivalence traces."""

from __future__ import annotations

import subprocess
from pathlib import Path

from gril.config import load_algorithm_config
from gril.dataset_io import load_dataset

TRACE_SCAN_NUMBERS = (1, 20, 21)


def write_preprocess_input(
    dataset_path: Path,
    gril_config_path: Path,
    output_path: Path,
) -> Path:
    dataset = load_dataset(dataset_path)
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
        raise ValueError(
            "GRIL preprocessing trace config is missing: " + ", ".join(missing)
        )

    blind = float(preprocess["blind"])
    point_filter_num = int(preprocess["point_filter_num"])
    n_scans = int(preprocess["scan_line"])
    required_frame_num = int(calibration["cut_frame_num"])
    lidar = dataset.lidar.normalized()
    available = len(lidar.scan_timestamps_ns)
    if available < max(TRACE_SCAN_NUMBERS):
        raise ValueError(
            f"Preprocessing trace requires at least {max(TRACE_SCAN_NUMBERS)} scans"
        )

    with output_path.open("w") as output:
        output.write("GRIL_PREPROCESS_INPUT 1\n")
        output.write(
            f"config {blind:.17g} {point_filter_num} "
            f"{n_scans} {required_frame_num}\n"
        )
        output.write(f"scans {len(TRACE_SCAN_NUMBERS)}\n")
        for scan_number in TRACE_SCAN_NUMBERS:
            index = scan_number - 1
            start = int(lidar.scan_offsets[index])
            end = int(lidar.scan_offsets[index + 1])
            output.write(
                f"scan {scan_number} "
                f"{int(lidar.scan_timestamps_ns[index])} {end - start}\n"
            )
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


def run_native_preprocess_trace(
    executable: Path, input_path: Path, output_path: Path
) -> Path:
    subprocess.run(
        [
            str(executable.resolve()),
            str(input_path.resolve()),
            str(output_path.resolve()),
        ],
        check=True,
    )
    return output_path


def compare_preprocess_traces(reference: Path, candidate: Path) -> None:
    reference_lines = reference.read_text().splitlines()
    candidate_lines = candidate.read_text().splitlines()
    if reference_lines != candidate_lines:
        mismatch = next(
            (
                index
                for index, values in enumerate(
                    zip(reference_lines, candidate_lines), start=1
                )
                if values[0] != values[1]
            ),
            min(len(reference_lines), len(candidate_lines)) + 1,
        )
        raise ValueError(f"Preprocessing traces differ at line {mismatch}")
