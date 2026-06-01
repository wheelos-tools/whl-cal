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

import os
from typing import Iterable, Iterator

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

try:
    from cyber_record.record import Record as RecordReader
except ImportError:
    RecordReader = None

# isort: off
from lidar2lidar import apollo_flatbuffer_messages as flat_messages  # noqa: E402
from lidar2lidar import apollo_record_messages as record_messages  # noqa: E402

# isort: on


def ensure_record_available() -> None:
    if RecordReader is None:
        raise RuntimeError(
            (
                "cyber_record is not installed. "
                "Use `pip install -e .` or install `cyber_record`."
            )
        )


def _normalize_topics(topics: Iterable[str] | None) -> set[str] | None:
    if topics is None:
        return None
    return {str(topic) for topic in topics}


def _normalize_type_name(type_name: str | None) -> str:
    if not type_name:
        return ""
    return str(type_name).replace("::", ".").strip(".")


class Record:
    def __init__(self, file_name: str):
        ensure_record_available()
        self._reader = RecordReader(file_name)

    def __enter__(self) -> "Record":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    @property
    def _py_reader(self):
        return getattr(self._reader, "_reader", None)

    def _iter_messages(
        self, topics: Iterable[str] | None = None
    ) -> Iterator[tuple[str, bytes, str, int]]:
        topic_filter = _normalize_topics(topics)
        reader = self._py_reader
        if reader is None:
            raise RuntimeError("cyber_record reader internals are unavailable.")

        for chunk_body_index in reader._get_chunk_body_indexs(None, None):
            proto_chunk_body = reader.read_chunk_body(chunk_body_index.position)
            if proto_chunk_body is None:
                continue
            reader.chunk.swap(proto_chunk_body)

            while not reader.chunk.end():
                single_message = reader.chunk.next_message()
                topic = str(single_message.channel_name)
                if topic_filter is not None and topic not in topic_filter:
                    continue
                channel_cache = reader.channels.get(topic)
                type_name = ""
                if channel_cache is not None:
                    type_name = str(channel_cache.message_type)
                yield (
                    topic,
                    bytes(single_message.content),
                    type_name,
                    int(single_message.time),
                )

    def _decode_message(self, topic: str, payload: bytes):
        reader = self._py_reader
        if reader is None:
            raise RuntimeError("cyber_record reader internals are unavailable.")

        message_type = reader.message_type_pool.get(topic)
        if message_type is not None:
            message = message_type()
            message.ParseFromString(payload)
            return message

        channel_cache = reader.channels.get(topic)
        type_name = str(channel_cache.message_type) if channel_cache is not None else ""

        custom_message_type = record_messages.get_message_class(
            _normalize_type_name(type_name)
        )
        if custom_message_type is not None:
            message = custom_message_type()
            message.ParseFromString(payload)
            return message

        flat_decoder = flat_messages.get_flat_message_decoder(type_name)
        if flat_decoder is not None:
            return flat_decoder(payload)

        return None

    def read_raw_messages(
        self, topics: Iterable[str] | None = None
    ) -> Iterator[tuple[str, bytes, str, int]]:
        for topic, payload, type_name, timestamp_ns in self._iter_messages(
            topics=topics
        ):
            yield topic, payload, type_name, timestamp_ns

    def read_messages(
        self, topics: Iterable[str] | None = None
    ) -> Iterator[tuple[str, object, int]]:
        for topic, payload, _, timestamp_ns in self._iter_messages(topics=topics):
            yield topic, self._decode_message(topic, payload), timestamp_ns
