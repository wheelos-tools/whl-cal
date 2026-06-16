from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import yaml

from camera.intrinsic import CameraCalibrator


def default_intrinsic_config_payload():
    target_type = "chessboard"
    payload = {
        "target_type": "chessboard",
        "require_release_ready": False,
        "camera_index": 0,
        "cameras": [],
        "window_name": "Industrial Calibration Tool",
        "window_width": 1280,
        "window_height": 720,
        "capture": {
            "force_resolution": False,
            "width": None,
            "height": None,
            "fourcc": None,
            "buffersize": 1,
            "warmup_frames": 12,
            "reconnect_bad_frame_burst": 30,
            "initial_ready_timeout_s": 10.0,
            "latest_frame_read_timeout_s": 0.25,
            "reconnect_sleep_s": 0.5,
        },
        "distortion_model": "plumb_bob",
        "optimization": {"resize_factor": 0.5, "detection_interval": 2},
        "undistortion_preview": {"alpha": 1.0, "center_principal_point": False},
        "workflow": {
            "root_dir": "outputs/camera_intrinsic",
            "save_live_accepted_frames": True,
        },
        "auto_capture_settings": {
            "grid_shape": [3, 3],
            "min_total_samples": 9,
            "pose_novelty_area_delta": 0.02,
            "pose_novelty_aspect_delta": 0.12,
            "pose_novelty_center_distance_ratio": 0.08,
            "samples_per_grid": 1,
            "delay_between_captures": 1.0,
            "stability_frames": 5,
            "stability_threshold": 2.0,
            "stability_threshold_ratio": 0.02,
        },
    }

    if target_type == "chessboard":
        payload.update(
            {
                "pattern_size": [11, 8],
                "square_size": 0.025,
            }
        )
    elif target_type == "aprilgrid":
        payload["aprilgrid"] = {
            "dictionary": "DICT_APRILTAG_36h11",
            "grid_cols": 6,
            "grid_rows": 6,
            "tag_size": 0.04,
            "tag_spacing_ratio": 0.3,
            "min_tags_per_frame": 6,
        }
    elif target_type == "charuco":
        payload["charuco"] = {
            "dictionary": "DICT_4X4_100",
            "squares_x": 6,
            "squares_y": 8,
            "square_length": 0.04,
            "marker_length": 0.02,
            "min_corners_per_frame": 12,
        }

    return payload


def build_argument_parser():
    parser = argparse.ArgumentParser(description="WheelOS Industrial Camera Calibrator")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Directory of images for headless calibration",
    )
    parser.add_argument(
        "--pattern-size", default=None, help="Override pattern size as W,H (optional)"
    )
    parser.add_argument(
        "--headless-live-max-seconds",
        type=float,
        default=0,
        help="Max seconds for automatic live headless mode. 0 means no timeout.",
    )
    parser.add_argument(
        "--require-release-ready",
        action="store_true",
        help="Return non-zero when quality gates are not release-ready.",
    )
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="Collect accepted live samples only and skip calibration.",
    )
    parser.add_argument(
        "--session-name",
        default=None,
        help="Stable name for the capture session directory, e.g. round01_front_cam.",
    )
    parser.add_argument(
        "--write-default-config",
        action="store_true",
        help="Write a default config to --config and exit.",
    )
    return parser


def write_default_config(config_path):
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(default_intrinsic_config_payload(), file, sort_keys=False)


def apply_pattern_size_override(calibrator, pattern_size_arg):
    if not pattern_size_arg:
        return
    if calibrator.target_type != "chessboard":
        print("[WARN] --pattern-size is only used for chessboard target_type.")
        return
    try:
        width, height = map(int, pattern_size_arg.split(","))
    except Exception:
        print("[WARN] invalid --pattern-size, ignoring")
        return
    calibrator.pattern_size = (width, height)
    calibrator.target_detector.update_chessboard_pattern(
        calibrator.pattern_size,
        calibrator.square_size,
    )


def dispatch_run(calibrator, args):
    if args.images_dir:
        return calibrator.run_headless(args.images_dir)

    display_available = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if display_available:
        try:
            return calibrator.run()
        except cv2.error as exc:
            print(f"[WARN] GUI mode failed ({exc}). Falling back to headless live mode.")
    else:
        print(
            "[WARN] No DISPLAY/WAYLAND_DISPLAY detected. Falling back to headless live mode."
        )
    return calibrator.run_live_headless(max_seconds=args.headless_live_max_seconds)


def run_cli(args):
    if args.capture_only and args.images_dir:
        print("[ERROR] --capture-only cannot be used together with --images-dir.")
        return 1

    config_path = Path(args.config).expanduser()
    if args.write_default_config:
        write_default_config(config_path)
        print(f"[INFO] Default config written to {config_path}")
        return 0
    if not config_path.exists():
        write_default_config(config_path)
        print(f"[INFO] Default config created at {config_path}")

    calibrator = CameraCalibrator(
        str(config_path),
        session_name=args.session_name,
        capture_only=args.capture_only,
    )
    if args.require_release_ready:
        calibrator.require_release_ready = True
    apply_pattern_size_override(calibrator, args.pattern_size)
    return dispatch_run(calibrator, args)


def main() -> None:
    parser = build_argument_parser()
    raise SystemExit(run_cli(parser.parse_args()))


if __name__ == "__main__":
    main()
