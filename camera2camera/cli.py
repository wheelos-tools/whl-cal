from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from camera2camera.reference_pipeline import (
    default_reference_config_payload, run_reference_calibration_from_config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Industrial camera-to-camera checkerboard calibration."
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

    run_reference_calibration_from_config(
        str(config_path),
        output_dir_override=args.output_dir,
    )


if __name__ == "__main__":
    main()
