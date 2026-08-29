"""GRIL configuration identity used by migration comparisons."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ALGORITHM_SECTIONS = (
    "preprocess",
    "calibration",
    "mapping",
    "patchworkpp",
)
LAUNCH_ALGORITHM_KEYS = (
    "max_iteration",
    "cube_side_length",
)


def load_algorithm_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"GRIL config must be a mapping: {path}")
    missing = [section for section in ALGORITHM_SECTIONS if section not in value]
    if missing:
        raise ValueError(
            f"GRIL config is missing algorithm sections: {', '.join(missing)}"
        )
    return {
        "launch": {key: value[key] for key in LAUNCH_ALGORITHM_KEYS if key in value},
        **{section: value[section] for section in ALGORITHM_SECTIONS},
    }


def config_digest(config: dict[str, Any]) -> str:
    canonical = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compare_configs(reference_path: Path, candidate_path: Path) -> dict[str, Any]:
    reference = load_algorithm_config(reference_path)
    candidate = load_algorithm_config(candidate_path)
    reference_digest = config_digest(reference)
    candidate_digest = config_digest(candidate)
    return {
        "verdict": (
            "equivalent" if reference_digest == candidate_digest else "different"
        ),
        "reference_path": str(Path(reference_path).resolve()),
        "candidate_path": str(Path(candidate_path).resolve()),
        "reference_digest": reference_digest,
        "candidate_digest": candidate_digest,
        "compared_sections": ["launch", *ALGORITHM_SECTIONS],
    }
