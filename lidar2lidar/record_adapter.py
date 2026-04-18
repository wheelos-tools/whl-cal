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

from typing import Iterable, Iterator

from lidar2lidar.apollo_record_messages import get_message_class

try:
    from pycyber.record import RecordReader
except ImportError:
    RecordReader = None


def ensure_record_available() -> None:
    if RecordReader is None:
        raise RuntimeError(
            "pycyber is not installed. Use `pip install -e .` or install `pycyber`."
        )


def _normalize_topics(topics: Iterable[str] | None) -> set[str] | None:
    if topics is None:
        return None
    return {str(topic) for topic in topics}


def _normalize_type_name(type_name: str | bytes) -> str:
    if isinstance(type_name, bytes):
        return type_name.decode("utf-8")
    return str(type_name)


def decode_message(payload: bytes, type_name: str | bytes):
    normalized_type_name = _normalize_type_name(type_name)
    message_cls = get_message_class(normalized_type_name)
    if message_cls is None:
        raise RuntimeError(f"Unsupported Apollo record message type: {normalized_type_name}")

    message = message_cls()
    message.ParseFromString(payload)
    return message


class Record:
    def __init__(self, file_name: str):
        ensure_record_available()
        self._reader = RecordReader(file_name)

    def __enter__(self) -> "Record":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read_raw_messages(self,
                          topics: Iterable[str] | None = None) -> Iterator[tuple[str, bytes, str, int]]:
        topic_filter = _normalize_topics(topics)
        for bag_message in self._reader.read_messages():
            topic = str(bag_message.topic)
            if topic_filter is not None and topic not in topic_filter:
                continue
            yield (
                topic,
                bytes(bag_message.message),
                _normalize_type_name(bag_message.data_type),
                int(bag_message.timestamp),
            )

    def read_messages(self, topics: Iterable[str] | None = None) -> Iterator[tuple[str, object, int]]:
        for topic, payload, type_name, timestamp_ns in self.read_raw_messages(topics=topics):
            yield topic, decode_message(payload, type_name), timestamp_ns
