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

from lidar2lidar.apollo_flatbuffer_messages import get_flat_message_decoder
from lidar2lidar.apollo_record_messages import get_message_class

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

try:
    from cyber_record.record import Record as CyberRecordReader
except (
    Exception
):  # noqa: BLE001 - surface backend availability in ensure_record_available
    CyberRecordReader = None


def ensure_record_available() -> None:
    if CyberRecordReader is None:
        raise RuntimeError("No record backend is available. Install `cyber-record`.")


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
    flat_message_decoder = get_flat_message_decoder(normalized_type_name)
    if flat_message_decoder is not None:
        return flat_message_decoder(payload)
    message_cls = get_message_class(normalized_type_name)
    if message_cls is None:
        raise RuntimeError(
            f"Unsupported Apollo record message type: {normalized_type_name}"
        )

    message = message_cls()
    message.ParseFromString(payload)
    return message


class Record:
    def __init__(self, file_name: str):
        ensure_record_available()
        self._backend = "cyber_record"
        self._reader = CyberRecordReader(file_name)
        self._channel_type_by_topic: dict[str, str] = {}
        self._channel_type_by_topic = self._extract_cyber_channel_types()

    def __enter__(self) -> "Record":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._backend == "cyber_record" and self._reader is not None:
            close_fn = getattr(self._reader, "close", None)
            if callable(close_fn):
                close_fn()
        return False

    def _extract_cyber_channel_types(self) -> dict[str, str]:
        if self._reader is None:
            return {}
        channels = getattr(getattr(self._reader, "_reader", None), "channels", {})
        mapping: dict[str, str] = {}
        if isinstance(channels, dict):
            for topic, channel_cache in channels.items():
                type_name = getattr(channel_cache, "message_type", "")
                if type_name:
                    mapping[str(topic)] = str(type_name)
        return mapping

    def _iter_cyber_raw_entries(
        self,
        topics: Iterable[str] | None = None,
    ) -> Iterator[tuple[str, bytes, str, int]]:
        topic_filter = _normalize_topics(topics)
        inner_reader = getattr(self._reader, "_reader", None)
        if inner_reader is None:
            return

        for chunk_body_index in inner_reader._get_chunk_body_indexs(None, None):
            proto_chunk_body = inner_reader.read_chunk_body(chunk_body_index.position)
            if proto_chunk_body is None:
                continue
            inner_reader.chunk.swap(proto_chunk_body)

            while not inner_reader.chunk.end():
                single_message = inner_reader.chunk.next_message()
                topic_name = str(single_message.channel_name)
                if topic_filter is not None and topic_name not in topic_filter:
                    continue

                yield (
                    topic_name,
                    bytes(single_message.content),
                    _normalize_type_name(
                        self._channel_type_by_topic.get(topic_name, "")
                    ),
                    int(single_message.time),
                )

    def read_raw_messages(
        self, topics: Iterable[str] | None = None
    ) -> Iterator[tuple[str, bytes, str, int]]:
        if self._backend == "cyber_record":
            yield from self._iter_cyber_raw_entries(topics=topics)
            return

    def read_messages(
        self, topics: Iterable[str] | None = None
    ) -> Iterator[tuple[str, object, int]]:
        if self._backend == "cyber_record":
            for topic_name, payload, type_name, timestamp_ns in self.read_raw_messages(
                topics=topics
            ):
                yield topic_name, decode_message(payload, type_name), int(timestamp_ns)
            return
