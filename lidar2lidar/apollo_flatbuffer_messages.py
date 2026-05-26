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
# Created Date: 2026-05-09

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Callable, Sequence

import flatbuffers.number_types as number_types
import numpy as np
from flatbuffers.table import Table

_POINTXYZIT_DTYPE = np.dtype(
    {
        "names": ["x", "y", "z", "intensity", "timestamp"],
        "formats": ["<f4", "<f4", "<f4", "<u4", "<u8"],
        "offsets": [0, 4, 8, 12, 16],
        "itemsize": 24,
    }
)

_POINTCLOUD_TYPE_NAMES = {
    "::apollo::drivers::flat::PointCloud",
    "apollo.drivers.flat.PointCloud",
}


def _decode_string(value: bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8")


class _FlatHeaderTable:
    __slots__ = ("_tab",)

    def Init(self, buf: bytes, pos: int) -> None:
        self._tab = Table(buf, pos)

    def TimestampSec(self) -> float:
        offset = self._tab.Offset(4)
        if offset == 0:
            return 0.0
        value = self._tab.Get(
            number_types.Float64Flags,
            offset + self._tab.Pos,
        )
        return float(value)

    def ModuleName(self) -> str:
        offset = self._tab.Offset(6)
        if offset == 0:
            return ""
        return _decode_string(self._tab.String(offset + self._tab.Pos))

    def SequenceNum(self) -> int:
        offset = self._tab.Offset(8)
        if offset == 0:
            return 0
        value = self._tab.Get(
            number_types.Uint32Flags,
            offset + self._tab.Pos,
        )
        return int(value)

    def FrameId(self) -> str:
        offset = self._tab.Offset(20)
        if offset == 0:
            return ""
        return _decode_string(self._tab.String(offset + self._tab.Pos))


class _FlatPointCloudTable:
    __slots__ = ("_tab",)

    @classmethod
    def GetRootAs(cls, buf: bytes, offset: int = 0) -> "_FlatPointCloudTable":
        root_offset = struct.unpack_from("<I", buf, offset)[0]
        message = cls()
        message.Init(buf, root_offset + offset)
        return message

    def Init(self, buf: bytes, pos: int) -> None:
        self._tab = Table(buf, pos)

    def Header(self) -> _FlatHeaderTable | None:
        offset = self._tab.Offset(4)
        if offset == 0:
            return None
        header = _FlatHeaderTable()
        header.Init(
            self._tab.Bytes,
            self._tab.Indirect(offset + self._tab.Pos),
        )
        return header

    def FrameId(self) -> str:
        offset = self._tab.Offset(6)
        if offset == 0:
            return ""
        return _decode_string(self._tab.String(offset + self._tab.Pos))

    def IsDense(self) -> bool:
        offset = self._tab.Offset(8)
        if offset == 0:
            return False
        value = self._tab.Get(
            number_types.BoolFlags,
            offset + self._tab.Pos,
        )
        return bool(value)

    def MeasurementTime(self) -> float:
        offset = self._tab.Offset(12)
        if offset == 0:
            return 0.0
        value = self._tab.Get(
            number_types.Float64Flags,
            offset + self._tab.Pos,
        )
        return float(value)

    def Width(self) -> int:
        offset = self._tab.Offset(14)
        if offset == 0:
            return 0
        value = self._tab.Get(
            number_types.Uint32Flags,
            offset + self._tab.Pos,
        )
        return int(value)

    def Height(self) -> int:
        offset = self._tab.Offset(16)
        if offset == 0:
            return 0
        value = self._tab.Get(
            number_types.Uint32Flags,
            offset + self._tab.Pos,
        )
        return int(value)

    def PointRecords(self) -> np.ndarray:
        offset = self._tab.Offset(10)
        if offset == 0:
            return np.empty((0,), dtype=_POINTXYZIT_DTYPE)
        vector_start = self._tab.Vector(offset)
        length = self._tab.VectorLen(offset)
        return np.frombuffer(
            self._tab.Bytes,
            dtype=_POINTXYZIT_DTYPE,
            count=length,
            offset=vector_start,
        )


@dataclass(frozen=True)
class FlatHeaderMessage:
    timestamp_sec: float
    module_name: str
    sequence_num: int
    frame_id: str


@dataclass(frozen=True)
class FlatPointXYZIT:
    x: float
    y: float
    z: float
    intensity: int
    timestamp: int


class FlatPointSequence(Sequence[FlatPointXYZIT]):
    def __init__(self, point_records: np.ndarray):
        self._point_records = point_records

    def __len__(self) -> int:
        return int(self._point_records.shape[0])

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        record = self._point_records[index]
        return FlatPointXYZIT(
            x=float(record["x"]),
            y=float(record["y"]),
            z=float(record["z"]),
            intensity=int(record["intensity"]),
            timestamp=int(record["timestamp"]),
        )


class FlatPointCloudMessage:
    def __init__(
        self,
        *,
        header: FlatHeaderMessage,
        frame_id: str,
        is_dense: bool,
        measurement_time: float,
        width: int,
        height: int,
        point_records: np.ndarray,
    ):
        self.header = header
        self.frame_id = frame_id
        self.is_dense = is_dense
        self.measurement_time = measurement_time
        self.width = width
        self.height = height
        self.point = FlatPointSequence(point_records)
        self._point_records = point_records

    def points_xyz_array(self) -> np.ndarray:
        if self._point_records.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        xyz = np.empty((self._point_records.shape[0], 3), dtype=np.float64)
        xyz[:, 0] = self._point_records["x"]
        xyz[:, 1] = self._point_records["y"]
        xyz[:, 2] = self._point_records["z"]
        return xyz

    def point_records_array(self) -> np.ndarray:
        return self._point_records


def decode_flat_pointcloud(payload: bytes) -> FlatPointCloudMessage:
    message = _FlatPointCloudTable.GetRootAs(payload)
    header_table = message.Header()
    module_name = header_table.ModuleName() if header_table is not None else ""
    sequence_num = (
        header_table.SequenceNum() if header_table is not None else 0
    )  # noqa: E501
    frame_id = header_table.FrameId() if header_table is not None else ""
    header = FlatHeaderMessage(
        timestamp_sec=(
            header_table.TimestampSec() if header_table is not None else 0.0
        ),
        module_name=module_name,
        sequence_num=sequence_num,
        frame_id=frame_id,
    )
    frame_id = message.FrameId() or header.frame_id
    return FlatPointCloudMessage(
        header=header,
        frame_id=frame_id,
        is_dense=message.IsDense(),
        measurement_time=message.MeasurementTime(),
        width=message.Width(),
        height=message.Height(),
        point_records=message.PointRecords(),
    )


def get_flat_message_decoder(
    type_name: str,
) -> Callable[[bytes], object] | None:
    if type_name in _POINTCLOUD_TYPE_NAMES:
        return decode_flat_pointcloud
    return None
