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

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


_TYPE_DOUBLE = descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE
_TYPE_FLOAT = descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT
_TYPE_UINT32 = descriptor_pb2.FieldDescriptorProto.TYPE_UINT32
_TYPE_UINT64 = descriptor_pb2.FieldDescriptorProto.TYPE_UINT64
_TYPE_STRING = descriptor_pb2.FieldDescriptorProto.TYPE_STRING
_TYPE_BOOL = descriptor_pb2.FieldDescriptorProto.TYPE_BOOL
_TYPE_MESSAGE = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE

_LABEL_OPTIONAL = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
_LABEL_REPEATED = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED


def _add_field(message: descriptor_pb2.DescriptorProto,
               name: str,
               number: int,
               field_type: int,
               label: int = _LABEL_OPTIONAL,
               type_name: str | None = None) -> None:
    field = message.field.add()
    field.name = name
    field.number = number
    field.label = label
    field.type = field_type
    if type_name is not None:
        field.type_name = type_name


def _build_common_file() -> descriptor_pb2.FileDescriptorProto:
    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "apollo/common/minimal_common.proto"
    file_proto.package = "apollo.common"
    file_proto.syntax = "proto2"

    header = file_proto.message_type.add()
    header.name = "Header"
    _add_field(header, "timestamp_sec", 1, _TYPE_DOUBLE)
    _add_field(header, "module_name", 2, _TYPE_STRING)
    _add_field(header, "sequence_num", 3, _TYPE_UINT32)
    _add_field(header, "frame_id", 9, _TYPE_STRING)

    point3d = file_proto.message_type.add()
    point3d.name = "Point3D"
    _add_field(point3d, "x", 1, _TYPE_DOUBLE)
    _add_field(point3d, "y", 2, _TYPE_DOUBLE)
    _add_field(point3d, "z", 3, _TYPE_DOUBLE)

    point_enu = file_proto.message_type.add()
    point_enu.name = "PointENU"
    _add_field(point_enu, "x", 1, _TYPE_DOUBLE)
    _add_field(point_enu, "y", 2, _TYPE_DOUBLE)
    _add_field(point_enu, "z", 3, _TYPE_DOUBLE)

    quaternion = file_proto.message_type.add()
    quaternion.name = "Quaternion"
    _add_field(quaternion, "qx", 1, _TYPE_DOUBLE)
    _add_field(quaternion, "qy", 2, _TYPE_DOUBLE)
    _add_field(quaternion, "qz", 3, _TYPE_DOUBLE)
    _add_field(quaternion, "qw", 4, _TYPE_DOUBLE)

    return file_proto


def _build_pointcloud_file() -> descriptor_pb2.FileDescriptorProto:
    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "apollo/drivers/minimal_pointcloud.proto"
    file_proto.package = "apollo.drivers"
    file_proto.syntax = "proto2"
    file_proto.dependency.append("apollo/common/minimal_common.proto")

    point = file_proto.message_type.add()
    point.name = "PointXYZIT"
    _add_field(point, "x", 1, _TYPE_FLOAT)
    _add_field(point, "y", 2, _TYPE_FLOAT)
    _add_field(point, "z", 3, _TYPE_FLOAT)
    _add_field(point, "intensity", 4, _TYPE_UINT32)
    _add_field(point, "timestamp", 5, _TYPE_UINT64)

    cloud = file_proto.message_type.add()
    cloud.name = "PointCloud"
    _add_field(cloud, "header", 1, _TYPE_MESSAGE, type_name=".apollo.common.Header")
    _add_field(cloud, "frame_id", 2, _TYPE_STRING)
    _add_field(cloud, "is_dense", 3, _TYPE_BOOL)
    _add_field(cloud, "point", 4, _TYPE_MESSAGE, label=_LABEL_REPEATED, type_name=".apollo.drivers.PointXYZIT")
    _add_field(cloud, "measurement_time", 5, _TYPE_DOUBLE)
    _add_field(cloud, "width", 6, _TYPE_UINT32)
    _add_field(cloud, "height", 7, _TYPE_UINT32)

    return file_proto


def _build_localization_file() -> descriptor_pb2.FileDescriptorProto:
    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "apollo/localization/minimal_localization.proto"
    file_proto.package = "apollo.localization"
    file_proto.syntax = "proto2"
    file_proto.dependency.append("apollo/common/minimal_common.proto")

    pose = file_proto.message_type.add()
    pose.name = "Pose"
    _add_field(pose, "position", 1, _TYPE_MESSAGE, type_name=".apollo.common.PointENU")
    _add_field(pose, "orientation", 2, _TYPE_MESSAGE, type_name=".apollo.common.Quaternion")
    _add_field(pose, "linear_velocity", 3, _TYPE_MESSAGE, type_name=".apollo.common.Point3D")
    _add_field(pose, "linear_acceleration", 4, _TYPE_MESSAGE, type_name=".apollo.common.Point3D")
    _add_field(pose, "angular_velocity", 5, _TYPE_MESSAGE, type_name=".apollo.common.Point3D")
    _add_field(pose, "heading", 6, _TYPE_DOUBLE)

    estimate = file_proto.message_type.add()
    estimate.name = "LocalizationEstimate"
    _add_field(estimate, "header", 1, _TYPE_MESSAGE, type_name=".apollo.common.Header")
    _add_field(estimate, "pose", 2, _TYPE_MESSAGE, type_name=".apollo.localization.Pose")
    _add_field(estimate, "measurement_time", 4, _TYPE_DOUBLE)

    corrected_imu = file_proto.message_type.add()
    corrected_imu.name = "CorrectedImu"
    _add_field(corrected_imu, "header", 1, _TYPE_MESSAGE, type_name=".apollo.common.Header")
    _add_field(corrected_imu, "imu", 3, _TYPE_MESSAGE, type_name=".apollo.localization.Pose")

    return file_proto


def _build_gnss_imu_file() -> descriptor_pb2.FileDescriptorProto:
    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "apollo/drivers/gnss/minimal_imu.proto"
    file_proto.package = "apollo.drivers.gnss"
    file_proto.syntax = "proto2"
    file_proto.dependency.append("apollo/common/minimal_common.proto")

    imu = file_proto.message_type.add()
    imu.name = "Imu"
    _add_field(imu, "header", 1, _TYPE_MESSAGE, type_name=".apollo.common.Header")
    _add_field(imu, "measurement_time", 2, _TYPE_DOUBLE)
    _add_field(imu, "measurement_span", 3, _TYPE_FLOAT)
    _add_field(imu, "linear_acceleration", 4, _TYPE_MESSAGE, type_name=".apollo.common.Point3D")
    _add_field(imu, "angular_velocity", 5, _TYPE_MESSAGE, type_name=".apollo.common.Point3D")

    return file_proto


def _build_transform_file() -> descriptor_pb2.FileDescriptorProto:
    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "apollo/transform/minimal_transform.proto"
    file_proto.package = "apollo.transform"
    file_proto.syntax = "proto2"
    file_proto.dependency.append("apollo/common/minimal_common.proto")

    transform = file_proto.message_type.add()
    transform.name = "Transform"
    _add_field(transform, "translation", 1, _TYPE_MESSAGE, type_name=".apollo.common.Point3D")
    _add_field(transform, "rotation", 2, _TYPE_MESSAGE, type_name=".apollo.common.Quaternion")

    stamped = file_proto.message_type.add()
    stamped.name = "TransformStamped"
    _add_field(stamped, "header", 1, _TYPE_MESSAGE, type_name=".apollo.common.Header")
    _add_field(stamped, "child_frame_id", 2, _TYPE_STRING)
    _add_field(stamped, "transform", 3, _TYPE_MESSAGE, type_name=".apollo.transform.Transform")

    stampeds = file_proto.message_type.add()
    stampeds.name = "TransformStampeds"
    _add_field(stampeds, "header", 1, _TYPE_MESSAGE, type_name=".apollo.common.Header")
    _add_field(stampeds, "transforms", 2, _TYPE_MESSAGE, label=_LABEL_REPEATED, type_name=".apollo.transform.TransformStamped")

    return file_proto


def _build_pool() -> descriptor_pool.DescriptorPool:
    pool = descriptor_pool.DescriptorPool()
    for file_proto in (
        _build_common_file(),
        _build_pointcloud_file(),
        _build_localization_file(),
        _build_gnss_imu_file(),
        _build_transform_file(),
    ):
        pool.AddSerializedFile(file_proto.SerializeToString())
    return pool


_POOL = _build_pool()
_MESSAGE_TYPES = {
    type_name: message_factory.GetMessageClass(_POOL.FindMessageTypeByName(type_name))
    for type_name in (
        "apollo.drivers.PointCloud",
        "apollo.localization.LocalizationEstimate",
        "apollo.localization.CorrectedImu",
        "apollo.drivers.gnss.Imu",
        "apollo.transform.TransformStampeds",
    )
}


def get_message_class(type_name: str):
    return _MESSAGE_TYPES.get(type_name)
