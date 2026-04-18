#!/usr/bin/env python3

# Copyright 2026 The WheelOS Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Created Date: 2026-02-09
# Author: daohu527

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R


def stamp_ns_to_dict(stamp_ns: int | None) -> dict:
    if stamp_ns is None or stamp_ns < 0:
        return {
            "secs": 0,
            "nsecs": 0,
        }

    return {
        "secs": int(stamp_ns // 1_000_000_000),
        "nsecs": int(stamp_ns % 1_000_000_000),
    }


def stamp_dict_to_ns(stamp: dict | None) -> int | None:
    if not isinstance(stamp, dict):
        return None
    secs = int(stamp.get("secs", 0))
    nsecs = int(stamp.get("nsecs", 0))
    return secs * 1_000_000_000 + nsecs


def matrix_from_transform_dict(transform_dict: dict) -> np.ndarray:
    translation = transform_dict["translation"]
    rotation = transform_dict["rotation"]

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = R.from_quat([
        float(rotation["x"]),
        float(rotation["y"]),
        float(rotation["z"]),
        float(rotation["w"]),
    ]).as_matrix()
    transform[:3, 3] = [
        float(translation["x"]),
        float(translation["y"]),
        float(translation["z"]),
    ]
    return transform


def transform_dict_from_matrix(matrix: np.ndarray) -> dict:
    rotation = R.from_matrix(matrix[:3, :3]).as_quat()
    translation = matrix[:3, 3]
    return {
        "translation": {
            "x": float(translation[0]),
            "y": float(translation[1]),
            "z": float(translation[2]),
        },
        "rotation": {
            "x": float(rotation[0]),
            "y": float(rotation[1]),
            "z": float(rotation[2]),
            "w": float(rotation[3]),
        },
    }


def build_extrinsics_payload(parent_frame: str,
                            child_frame: str,
                            matrix: np.ndarray,
                            stamp_ns: int | None = None,
                            seq: int = 0,
                            metrics: dict | None = None,
                            metadata: dict | None = None) -> dict:
    payload = {
        "header": {
            "stamp": stamp_ns_to_dict(stamp_ns),
            "seq": int(seq),
            "frame_id": parent_frame,
        },
        "transform": transform_dict_from_matrix(matrix),
        "child_frame_id": child_frame,
    }
    if metrics:
        payload["metrics"] = metrics
    if metadata:
        payload["metadata"] = metadata
    return payload


def parse_transform_payload(payload) -> tuple[np.ndarray, str, str, int | None]:
    parent_frame = ""
    child_frame = ""
    stamp_ns = None

    if isinstance(payload, dict):
        header = payload.get("header", {})
        if isinstance(header, dict):
            parent_frame = str(header.get("frame_id", ""))
            stamp_ns = stamp_dict_to_ns(header.get("stamp"))
        child_frame = str(payload.get("child_frame_id", ""))

        if "transform" in payload:
            return matrix_from_transform_dict(payload["transform"]), parent_frame, child_frame, stamp_ns
        if "translation" in payload and "rotation" in payload:
            return matrix_from_transform_dict(payload), parent_frame, child_frame, stamp_ns
        if "extrinsic_matrix" in payload:
            return np.array(payload["extrinsic_matrix"], dtype=float), parent_frame, child_frame, stamp_ns

    return np.array(payload, dtype=float), parent_frame, child_frame, stamp_ns


def load_extrinsics_file(path: str) -> tuple[np.ndarray, str, str, int | None, dict]:
    suffix = Path(path).suffix.lower()
    with open(path, "r", encoding="utf-8") as file:
        if suffix == ".json":
            payload = json.load(file)
        else:
            payload = yaml.safe_load(file)

    matrix, parent_frame, child_frame, stamp_ns = parse_transform_payload(payload)
    return matrix, parent_frame, child_frame, stamp_ns, payload


def save_extrinsics_yaml(path: str,
                         parent_frame: str,
                         child_frame: str,
                         matrix: np.ndarray,
                         stamp_ns: int | None = None,
                         seq: int = 0,
                         metrics: dict | None = None,
                         metadata: dict | None = None) -> dict:
    payload = build_extrinsics_payload(
        parent_frame=parent_frame,
        child_frame=child_frame,
        matrix=matrix,
        stamp_ns=stamp_ns,
        seq=seq,
        metrics=metrics,
        metadata=metadata,
    )
    with open(path, "w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)
    return payload


def extrinsics_filename(parent_frame: str, child_frame: str) -> str:
    return f"{parent_frame}_{child_frame}_extrinsics.yaml"