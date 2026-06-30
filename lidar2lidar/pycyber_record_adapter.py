#!/usr/bin/env python3

from __future__ import annotations

import importlib
import logging
from typing import Iterable, Iterator

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

# isort: off
from lidar2lidar import apollo_flatbuffer_messages as flat_messages  # noqa: E402
from lidar2lidar import apollo_record_messages as record_messages  # noqa: E402

# isort: on

MAX_UINT64 = 18446744073709551615
_logger = logging.getLogger(__name__)

try:
    from pycyber.record import RecordReader  # type: ignore
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError(f"pycyber is required for this backend: {exc}") from exc


def _normalize_topics(topics: Iterable[str] | None) -> set[str] | None:
    if topics is None:
        return None
    return {str(topic) for topic in topics}


def _normalize_type_name(type_name: str | None) -> str:
    if not type_name:
        return ""
    return str(type_name).replace("::", ".").strip(".")


class Record:
    """Read-only pycyber-backed record adapter.

    The adapter mirrors the subset of the cyber_record interface used by this
    repository. It keeps payload decoding best-effort: protobuf descriptors are
    aggregated into a local pool when available, and unsupported payloads are
    returned as raw bytes.
    """

    def __init__(
        self,
        file_name,
        mode: str = "r",
        compression=None,
        chunk_threshold=None,
        allow_unindexed=False,
        options=None,
    ):
        if mode not in ("r", None):
            raise ValueError(f"mode {mode!r} is not supported by pycyber adapter")

        self._file = file_name
        self._reader = RecordReader(str(file_name))
        self._descriptor_pool = None
        self._message_factory = None
        self._message_cache = {}

        try:
            self._build_descriptor_pool()
        except Exception as exc:  # pragma: no cover - best effort fallback
            _logger.debug("building descriptor pool failed: %s", exc)

    def _build_descriptor_pool(self) -> None:
        pool = descriptor_pool.DescriptorPool()
        files_map: dict[str, bytes] = {}

        proto_desc_mod = None
        for candidate in (
            "pycyber.proto.proto_desc_pb2",
            "cyber_record.cyber.proto.proto_desc_pb2",
        ):
            try:
                proto_desc_mod = importlib.import_module(candidate)
                break
            except Exception:
                proto_desc_mod = None

        def _add_file_descriptor(fd_proto) -> None:
            try:
                name = getattr(fd_proto, "name", None) or "<anon>"
            except Exception:
                name = "<anon>"
            try:
                files_map[name] = fd_proto.SerializeToString()
            except Exception:
                pass

        def _add_proto_desc_recursive(proto_desc_obj) -> None:
            if not proto_desc_obj or not getattr(proto_desc_obj, "desc", None):
                return
            for dependency in getattr(proto_desc_obj, "dependencies", ()):
                _add_proto_desc_recursive(dependency)
            try:
                fd = descriptor_pb2.FileDescriptorProto()
                fd.ParseFromString(proto_desc_obj.desc)
                _add_file_descriptor(fd)
            except Exception:
                pass

        try:
            channels = list(self._reader.get_channellist())
        except Exception:
            channels = []

        for channel in channels:
            try:
                pd_bytes = self._reader.get_protodesc(channel)
            except Exception:
                pd_bytes = None
            if not pd_bytes:
                continue

            parsed = False
            if proto_desc_mod is not None:
                try:
                    proto_desc = proto_desc_mod.ProtoDesc()
                    proto_desc.ParseFromString(pd_bytes)
                    _add_proto_desc_recursive(proto_desc)
                    parsed = True
                except Exception:
                    parsed = False
            if parsed:
                continue

            try:
                file_set = descriptor_pb2.FileDescriptorSet()
                file_set.ParseFromString(pd_bytes)
                if file_set.file:
                    for file_proto in file_set.file:
                        _add_file_descriptor(file_proto)
                    parsed = True
            except Exception:
                parsed = False
            if parsed:
                continue

            try:
                file_proto = descriptor_pb2.FileDescriptorProto()
                file_proto.ParseFromString(pd_bytes)
                _add_file_descriptor(file_proto)
            except Exception:
                pass

        remaining = dict(files_map)
        progress = True
        while remaining and progress:
            progress = False
            for name, serialized in list(remaining.items()):
                try:
                    pool.AddSerializedFile(serialized)
                    del remaining[name]
                    progress = True
                except Exception:
                    pass

        self._descriptor_pool = pool
        self._message_factory = message_factory.MessageFactory()

    def _get_message_class(self, msg_type: str):
        if (
            not msg_type
            or self._descriptor_pool is None
            or self._message_factory is None
        ):
            return None
        if msg_type in self._message_cache:
            return self._message_cache[msg_type]

        descriptor = None
        try:
            descriptor = self._descriptor_pool.FindMessageTypeByName(msg_type)
        except Exception:
            try:
                descriptor = self._descriptor_pool.FindMessageTypeByName(
                    msg_type.lstrip(".")
                )
            except Exception:
                descriptor = None

        if descriptor is None:
            self._message_cache[msg_type] = None
            return None

        try:
            msg_cls = self._message_factory.GetPrototype(descriptor)
        except Exception:
            msg_cls = None

        self._message_cache[msg_type] = msg_cls
        return msg_cls

    def _decode_message(self, topic: str, payload: bytes):
        msg_type = ""
        try:
            msg_type = str(self._reader.get_messagetype(topic) or "")
        except Exception:
            msg_type = ""

        msg_cls = self._get_message_class(_normalize_type_name(msg_type))
        if msg_cls is None:
            try:
                pd_bytes = self._reader.get_protodesc(topic)
            except Exception:
                pd_bytes = None
            if pd_bytes:
                try:
                    file_set = descriptor_pb2.FileDescriptorSet()
                    file_set.ParseFromString(pd_bytes)
                    for file_proto in file_set.file:
                        try:
                            self._descriptor_pool.AddSerializedFile(
                                file_proto.SerializeToString()
                            )
                        except Exception:
                            pass
                    msg_cls = self._get_message_class(_normalize_type_name(msg_type))
                except Exception:
                    pass
                if msg_cls is None:
                    try:
                        file_proto = descriptor_pb2.FileDescriptorProto()
                        file_proto.ParseFromString(pd_bytes)
                        self._descriptor_pool.AddSerializedFile(
                            file_proto.SerializeToString()
                        )
                        msg_cls = self._get_message_class(
                            _normalize_type_name(msg_type)
                        )
                    except Exception:
                        pass

        if msg_cls is not None:
            message = msg_cls()
            message.ParseFromString(payload)
            return message

        custom_message_type = record_messages.get_message_class(
            _normalize_type_name(msg_type)
        )
        if custom_message_type is not None:
            message = custom_message_type()
            message.ParseFromString(payload)
            return message

        flat_decoder = flat_messages.get_flat_message_decoder(msg_type)
        if flat_decoder is not None:
            return flat_decoder(payload)

        return payload

    def _iter_messages(
        self, topics: Iterable[str] | None = None
    ) -> Iterator[tuple[str, object, int]]:
        topic_filter = _normalize_topics(topics)
        try:
            bags = self._reader.read_messages(0, MAX_UINT64)
        except TypeError:
            bags = self._reader.read_messages()

        for bag in bags:
            try:
                topic = bag.topic
                payload = bag.message
                timestamp_ns = bag.timestamp
            except Exception:
                try:
                    topic, payload, _data_type, timestamp_ns = bag
                except Exception:
                    continue

            topic = str(topic)
            if topic_filter is not None and topic not in topic_filter:
                continue

            parsed = payload
            if isinstance(payload, (bytes, bytearray, memoryview)):
                parsed = self._decode_message(topic, bytes(payload))

            yield topic, parsed, int(timestamp_ns)

    def __enter__(self) -> "Record":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def __iter__(self):
        return self.read_messages()

    def read_raw_messages(
        self, topics: Iterable[str] | None = None
    ) -> Iterator[tuple[str, bytes, str, int]]:
        for topic, message, timestamp_ns in self._iter_messages(topics=topics):
            if isinstance(message, (bytes, bytearray, memoryview)):
                payload = bytes(message)
            elif hasattr(message, "SerializeToString"):
                payload = message.SerializeToString()
            else:
                payload = b""
            yield topic, payload, self.get_messagetype(topic), timestamp_ns

    def read_messages(
        self, topics: Iterable[str] | None = None
    ) -> Iterator[tuple[str, object, int]]:
        yield from self._iter_messages(topics=topics)

    def get_messagenumber(self, channel_name):
        try:
            return self._reader.get_messagenumber(channel_name)
        except Exception:
            return 0

    def get_messagetype(self, channel_name):
        try:
            return self._reader.get_messagetype(channel_name)
        except Exception:
            return ""

    def get_protodesc(self, channel_name):
        try:
            return self._reader.get_protodesc(channel_name)
        except Exception:
            return b""

    def get_headerstring(self):
        try:
            return self._reader.get_headerstring()
        except Exception:
            return ""

    def get_channellist(self):
        try:
            return self._reader.get_channellist()
        except Exception:
            return []

    def reset(self):
        try:
            return self._reader.reset()
        except Exception:
            return None

    def close(self):
        return None
