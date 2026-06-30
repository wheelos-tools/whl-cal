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

try:
    from lidar2lidar.pycyber_record_adapter import Record as PycyberRecordAdapter
except ImportError:
    PycyberRecordAdapter = None

# isort: off
from lidar2lidar import apollo_flatbuffer_messages as flat_messages  # noqa: E402
from lidar2lidar import apollo_record_messages as record_messages  # noqa: E402

# isort: on


def ensure_record_available() -> None:
    _select_record_backend()


def _select_record_backend() -> str:
    backend = os.environ.get("WHL_CAL_RECORD_BACKEND", "auto").strip().lower()
    if backend not in {"auto", "cyber_record", "pycyber"}:
        raise RuntimeError(
            "Unsupported WHL_CAL_RECORD_BACKEND value: "
            f"{backend!r}. Expected auto, cyber_record, or pycyber."
        )

    if backend == "cyber_record":
        if RecordReader is None:
            raise RuntimeError(
                "cyber_record is not installed. Use `pip install -e .` or install "
                "cyber_record, or switch WHL_CAL_RECORD_BACKEND to pycyber."
            )
        return "cyber_record"

    if backend == "pycyber":
        if PycyberRecordAdapter is None:
            raise RuntimeError(
                "pycyber is not installed. Install pycyber and retry, or switch "
                "WHL_CAL_RECORD_BACKEND to cyber_record."
            )
        return "pycyber"

    if RecordReader is not None:
        return "cyber_record"
    if PycyberRecordAdapter is not None:
        return "pycyber"

    raise RuntimeError(
        "Neither cyber_record nor pycyber is installed. Use `pip install -e .` "
        "or install one of the record backends."
    )


def _normalize_topics(topics: Iterable[str] | None) -> set[str] | None:
    if topics is None:
        return None
    return {str(topic) for topic in topics}


def _normalize_type_name(type_name: str | None) -> str:
    if not type_name:
        return ""
    return str(type_name).replace("::", ".").strip(".")


class _CyberRecordAdapter:
    def __init__(self, file_name: str):
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


class Record:
    def __init__(self, file_name: str):
        ensure_record_available()
        backend = _select_record_backend()
        self._backend_name = backend
        self._impl = (
            _CyberRecordAdapter(file_name)
            if backend == "cyber_record"
            else PycyberRecordAdapter(file_name)
        )

    def __enter__(self) -> "Record":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if hasattr(self._impl, "close"):
            self._impl.close()
        return False

    def __iter__(self):
        return self.read_messages()

    def read_raw_messages(
        self, topics: Iterable[str] | None = None
    ) -> Iterator[tuple[str, bytes, str, int]]:
        yield from self._impl.read_raw_messages(topics=topics)

    def read_messages(
        self, topics: Iterable[str] | None = None
    ) -> Iterator[tuple[str, object, int]]:
        yield from self._impl.read_messages(topics=topics)

    def get_messagenumber(self, channel_name):
        if hasattr(self._impl, "get_messagenumber"):
            return self._impl.get_messagenumber(channel_name)
        return 0

    def get_messagetype(self, channel_name):
        if hasattr(self._impl, "get_messagetype"):
            return self._impl.get_messagetype(channel_name)
        return ""

    def get_protodesc(self, channel_name):
        if hasattr(self._impl, "get_protodesc"):
            return self._impl.get_protodesc(channel_name)
        return b""

    def get_headerstring(self):
        if hasattr(self._impl, "get_headerstring"):
            return self._impl.get_headerstring()
        return ""

    def get_channellist(self):
        if hasattr(self._impl, "get_channellist"):
            return self._impl.get_channellist()
        return []

    def reset(self):
        if hasattr(self._impl, "reset"):
            return self._impl.reset()
        return None
