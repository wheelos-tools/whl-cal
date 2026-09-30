#!/usr/bin/env python3
"""Profile GRIL batch-trace angular-rate timing on the actual state timestamps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_states(lines: list[str], section: str) -> np.ndarray:
    header = next(
        (
            index
            for index, line in enumerate(lines)
            if line.startswith(f"{section}_states ")
        ),
        None,
    )
    if header is None:
        raise ValueError(f"Trace has no {section}_states section")
    count = int(lines[header].split()[1])
    rows = []
    for line in lines[header + 1 : header + 1 + count]:
        fields = line.split()
        if not fields or fields[0] != section:
            raise ValueError(f"Malformed {section} state row")
        row = np.asarray(fields[1:], dtype=np.float64)
        if row.size != 25:
            raise ValueError(f"Expected 25 fields in {section} state, got {row.size}")
        rows.append(row)
    states = np.asarray(rows)
    if len(states) != count or not np.isfinite(states).all():
        raise ValueError(f"Incomplete or non-finite {section} state section")
    if np.any(np.diff(states[:, 0]) <= 0.0):
        raise ValueError(f"{section} timestamps must be strictly increasing")
    return states


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    if denominator <= np.finfo(np.float64).eps:
        return float("nan")
    return float(left_centered @ right_centered / denominator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="GRIL_BATCH_TRACE 1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-offset-s", type=float, default=0.1)
    parser.add_argument("--step-s", type=float, default=0.002)
    parser.add_argument("--window-count", type=int, default=4)
    parser.add_argument("--sample-period-s", type=float, default=0.01)
    parser.add_argument("--signal", choices=("norm", "yaw_z"), default="norm")
    parser.add_argument("--smoothing-s", type=float, default=0.0)
    parser.add_argument("--min-window-correlation", type=float, default=0.5)
    args = parser.parse_args()
    if (
        any(
            not np.isfinite(value) or value <= 0.0
            for value in (args.max_offset_s, args.step_s, args.sample_period_s)
        )
        or args.window_count < 1
        or not np.isfinite(args.smoothing_s)
        or args.smoothing_s < 0.0
        or not np.isfinite(args.min_window_correlation)
        or not 0.0 < args.min_window_correlation <= 1.0
    ):
        parser.error("invalid offset, step, sample period, window count, or smoothing")

    lines = args.input.read_text().splitlines()
    if not lines or lines[0].strip() != "GRIL_BATCH_TRACE 1":
        raise ValueError("Input must use the GRIL_BATCH_TRACE 1 format")
    imu_states = read_states(lines, "imu")
    lidar_states = read_states(lines, "lidar")
    imu_time = imu_states[:, 0]
    lidar_time = lidar_states[:, 0]
    if args.signal == "norm":
        imu_omega = np.linalg.norm(imu_states[:, 13:16], axis=1)
        lidar_omega = np.linalg.norm(lidar_states[:, 13:16], axis=1)
    else:
        imu_omega = imu_states[:, 15]
        lidar_omega = lidar_states[:, 15]

    def sampled_pair(
        timestamps: np.ndarray, offset: float
    ) -> tuple[np.ndarray, np.ndarray]:
        lidar = np.interp(timestamps, lidar_time, lidar_omega)
        imu = np.interp(timestamps + offset, imu_time, imu_omega)
        if args.smoothing_s:
            sigma = args.smoothing_s / args.sample_period_s
            lidar = gaussian_filter1d(lidar, sigma)
            imu = gaussian_filter1d(imu, sigma)
        return lidar, imu

    max_offset = args.max_offset_s
    start = max(lidar_time[0], imu_time[0] + max_offset)
    end = min(lidar_time[-1], imu_time[-1] - max_offset)
    if end <= start:
        raise ValueError("The IMU and LiDAR traces have no common timing support")
    edges = np.linspace(start, end, args.window_count + 1)
    offsets = np.arange(-max_offset, max_offset + args.step_s * 0.5, args.step_s)
    if len(offsets) < 3:
        parser.error("offset step must provide at least three search positions")
    curves = []
    profiles = []
    for index in range(args.window_count):
        window_start, window_end = edges[index : index + 2]
        timestamps = np.arange(
            window_start, window_end, args.sample_period_s, dtype=np.float64
        )
        if len(timestamps) < 10:
            raise ValueError(f"Window {index} has fewer than 10 samples")
        lidar_values, imu_values_at_lidar_time = sampled_pair(timestamps, 0.0)
        correlations = []
        for offset in offsets:
            _, imu_values = sampled_pair(timestamps, float(offset))
            correlations.append(pearson(lidar_values, imu_values))
        correlations = np.asarray(correlations)
        if not np.isfinite(correlations).any():
            raise ValueError(f"Window {index} has no finite correlation values")
        peak_index = int(np.nanargmax(correlations))
        spread_key = (
            "angular_rate_norm_std_rad_s"
            if args.signal == "norm"
            else "angular_rate_z_std_rad_s"
        )
        profile = {
            "window": index,
            "start_s": float(window_start),
            "end_s": float(window_end),
            "sample_count": int(len(timestamps)),
            f"imu_{spread_key}": float(np.std(imu_values_at_lidar_time)),
            f"lidar_{spread_key}": float(np.std(lidar_values)),
            "correlation_at_zero_offset": pearson(
                lidar_values, imu_values_at_lidar_time
            ),
            "best_offset_imu_time_plus_offset_equals_lidar_time_s": float(
                offsets[peak_index]
            ),
            "peak_pearson_correlation": float(correlations[peak_index]),
            "peak_at_search_boundary": bool(
                peak_index == 0 or peak_index == len(offsets) - 1
            ),
        }
        curves.append(correlations)
        profiles.append(profile)

    all_timestamps = np.arange(start, end, args.sample_period_s, dtype=np.float64)
    all_lidar_values, all_imu_values = sampled_pair(all_timestamps, 0.0)
    overall_correlations = []
    for offset in offsets:
        _, imu_values = sampled_pair(all_timestamps, float(offset))
        overall_correlations.append(pearson(all_lidar_values, imu_values))
    overall_correlations = np.asarray(overall_correlations)
    if not np.isfinite(overall_correlations).any():
        raise ValueError("Full trace has no finite correlation values")
    overall_peak = int(np.nanargmax(overall_correlations))
    window_offsets = [
        profile["best_offset_imu_time_plus_offset_equals_lidar_time_s"]
        for profile in profiles
    ]
    max_window_spread_s = max(0.01, args.step_s * 2)
    failure_reasons = []
    if any(profile["peak_at_search_boundary"] for profile in profiles):
        failure_reasons.append("window_peak_at_search_boundary")
    if any(
        profile["peak_pearson_correlation"] < args.min_window_correlation
        for profile in profiles
    ):
        failure_reasons.append("weak_window_correlation")
    if np.ptp(window_offsets) > max_window_spread_s:
        failure_reasons.append("window_offset_disagreement")
    report = {
        "schema_version": 1,
        "input": str(args.input),
        "method": (
            "Timestamp-interpolated angular-rate norm Pearson correlation"
            if args.signal == "norm"
            else "Timestamp-interpolated z angular-rate Pearson correlation"
        ),
        "signal": args.signal,
        "smoothing_sigma_s": args.smoothing_s,
        "offset_convention": (
            "compare lidar_norm(t) with imu_norm(t + offset)"
            if args.signal == "norm"
            else "compare lidar_z(t) with imu_z(t + offset)"
        ),
        "search": {
            "range_s": [-max_offset, max_offset],
            "step_s": args.step_s,
            "sample_period_s": args.sample_period_s,
            "window_count": args.window_count,
            "min_window_correlation": args.min_window_correlation,
            "max_window_offset_spread_s": max_window_spread_s,
        },
        "state_support": {
            "imu_state_count": int(len(imu_states)),
            "imu_start_s": float(imu_time[0]),
            "imu_end_s": float(imu_time[-1]),
            "lidar_state_count": int(len(lidar_states)),
            "lidar_start_s": float(lidar_time[0]),
            "lidar_end_s": float(lidar_time[-1]),
            "profile_start_s": float(start),
            "profile_end_s": float(end),
        },
        "global_best": {
            "offset_imu_time_plus_offset_equals_lidar_time_s": float(
                offsets[overall_peak]
            ),
            "peak_pearson_correlation": float(overall_correlations[overall_peak]),
            "correlation_at_zero_offset": pearson(all_lidar_values, all_imu_values),
            "peak_at_search_boundary": bool(
                overall_peak == 0 or overall_peak == len(offsets) - 1
            ),
        },
        "windows": profiles,
        "failure_reasons": failure_reasons,
        "assessment": (
            "inconclusive"
            if failure_reasons
            else "candidate_requires_independent_validation"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = (
        ""
        if args.signal == "norm" and not args.smoothing_s
        else f"_{args.signal}_sigma_{args.smoothing_s:g}s"
    )
    report_path = args.output_dir / f"gril_batch_time_profile{suffix}.yaml"
    report_path.write_text(yaml.safe_dump(report, sort_keys=False))

    figure, axes = plt.subplots(
        args.window_count + 1,
        1,
        figsize=(9, 2.5 * (args.window_count + 1)),
    )
    for axis, profile, curve in zip(axes[:-1], profiles, curves):
        axis.plot(offsets, curve)
        axis.axvline(
            profile["best_offset_imu_time_plus_offset_equals_lidar_time_s"],
            color="tab:red",
            linestyle="--",
        )
        axis.set_title(
            f"Window {profile['window']}: "
            f"r={profile['peak_pearson_correlation']:.3f}, "
            "offset="
            f"{profile['best_offset_imu_time_plus_offset_equals_lidar_time_s']:.3f}s"
        )
        axis.set_ylabel("Pearson r")
        axis.grid(True, alpha=0.3)
    axes[-1].plot(offsets, overall_correlations, color="black")
    axes[-1].axvline(offsets[overall_peak], color="tab:red", linestyle="--")
    axes[-1].set_title(
        f"All windows: r={overall_correlations[overall_peak]:.3f}, "
        f"offset={offsets[overall_peak]:.3f}s"
    )
    axes[-1].set_xlabel(
        "Offset added to IMU timestamp (s); compare imu(t+offset) with lidar(t)"
    )
    axes[-1].set_ylabel("Pearson r")
    axes[-1].grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(args.output_dir / f"gril_batch_time_profile{suffix}.png", dpi=160)
    plt.close(figure)
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
