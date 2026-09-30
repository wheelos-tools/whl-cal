"""Canonical inputs for ROS-free Patchwork++ equivalence."""

from __future__ import annotations

import subprocess
from itertools import zip_longest
from pathlib import Path

from gril.config import load_algorithm_config
from gril.dataset_io import load_dataset


def write_ground_input(
    dataset_path: Path,
    gril_config_path: Path,
    output_path: Path,
    *,
    scan_count: int = 21,
) -> Path:
    dataset = load_dataset(dataset_path)
    lidar = dataset.lidar.normalized()
    if len(lidar.scan_timestamps_ns) < scan_count:
        raise ValueError(f"Ground trace requires at least {scan_count} scans")
    config = load_algorithm_config(gril_config_path)["patchworkpp"]
    czm = config["czm"]

    scalar_names = (
        "sensor_height",
        "num_iter",
        "num_lpr",
        "num_min_pts",
        "max_flatness_storage",
        "max_elevation_storage",
        "th_seeds",
        "th_dist",
        "th_seeds_v",
        "th_dist_v",
        "max_r",
        "min_r",
        "uprightness_thr",
        "adaptive_seed_selection_margin",
        "RNR_ver_angle_thr",
        "RNR_intensity_thr",
        "enable_RNR",
        "enable_RVPF",
        "enable_TGR",
    )
    missing = [name for name in scalar_names if name not in config]
    if missing:
        raise ValueError("Patchwork++ config is missing: " + ", ".join(missing))

    def sequence(name: str) -> list:
        values = czm[name]
        return list(values)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output:
        output.write("GRIL_GROUND_INPUT 1\n")
        output.write(
            "config "
            + " ".join(
                ("1" if value else "0") if isinstance(value, bool) else str(value)
                for value in (config[name] for name in scalar_names)
            )
            + "\n"
        )
        for label, key in (
            ("sectors", "num_sectors_each_zone"),
            ("rings", "mum_rings_each_zone"),
            ("elevation", "elevation_thresholds"),
            ("flatness", "flatness_thresholds"),
        ):
            values = sequence(key)
            output.write(f"{label} {len(values)} " + " ".join(map(str, values)) + "\n")
        output.write(f"scans {scan_count}\n")
        for scan_index in range(scan_count):
            start = int(lidar.scan_offsets[scan_index])
            end = int(lidar.scan_offsets[scan_index + 1])
            output.write(
                f"scan {scan_index + 1} "
                f"{int(lidar.scan_timestamps_ns[scan_index])} {end - start}\n"
            )
            for point_index in range(start, end):
                xyz = lidar.xyz[point_index]
                output.write(
                    "point "
                    f"{float(xyz[0]):.9g} {float(xyz[1]):.9g} "
                    f"{float(xyz[2]):.9g} "
                    f"{float(lidar.intensity[point_index]):.9g}\n"
                )
        output.write("END\n")
    return output_path


def run_native_ground_trace(
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


def compare_ground_traces(reference: Path, candidate: Path) -> None:
    with reference.open() as reference_stream, candidate.open() as candidate_stream:
        for line_number, (reference_line, candidate_line) in enumerate(
            zip_longest(reference_stream, candidate_stream), start=1
        ):
            if reference_line != candidate_line:
                raise ValueError(f"Ground traces differ at line {line_number}")
