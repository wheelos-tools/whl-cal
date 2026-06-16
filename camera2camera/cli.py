from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from camera2camera.live_capture import run_live_capture
from camera2camera.reference_pipeline import (
    default_reference_config_payload,
    run_reference_calibration_from_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Industrial camera-to-camera calibration."
    )
    parser.add_argument(
        "--config",
        default="camera2camera_config.yaml",
        help="Path to the camera2camera YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--write-default-config",
        action="store_true",
        help="Write a default config to --config and exit.",
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
        help="Return non-zero when live/final quality gates are not release-ready.",
    )
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="Collect accepted live stereo pairs only and skip final calibration.",
    )
    parser.add_argument(
        "--session-name",
        default=None,
        help="Stable name for the live capture session directory.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    if args.write_default_config:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(default_reference_config_payload(), file, sort_keys=False)
        return

    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(default_reference_config_payload(), file, sort_keys=False)
        raise SystemExit(
            "Config "
            f"{config_path} did not exist. A default file was created; "
            "edit it and rerun."
        )

    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    live_capture_enabled = bool(
        ((payload.get("live_capture", {}) or {}).get("enabled", False))
        or args.capture_only
    )
    if live_capture_enabled:
        raise SystemExit(
            run_live_capture(
                payload,
                base_directory=config_path.parent,
                session_name=args.session_name,
                capture_only=args.capture_only,
                require_release_ready=(True if args.require_release_ready else None),
                output_dir_override=args.output_dir,
                headless_live_max_seconds=args.headless_live_max_seconds,
            )
        )

    run_reference_calibration_from_config(
        str(config_path),
        output_dir_override=args.output_dir,
    )


if __name__ == "__main__":
    main()
