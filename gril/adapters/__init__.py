"""Input adapters for canonical GRIL datasets."""

from __future__ import annotations

from types import MappingProxyType

from gril.adapters.base import AdapterConfig, InputAdapter
from gril.adapters.core_record import CoreApolloRecordAdapter
from gril.adapters.rosbag1 import Rosbag1Adapter

ADAPTERS = MappingProxyType(
    {
        "record": CoreApolloRecordAdapter,
        "bag": Rosbag1Adapter,
    }
)


def supported_input_types() -> tuple[str, ...]:
    """Return registered source-adapter names in deterministic order."""
    return tuple(sorted(ADAPTERS))


def create_adapter(input_type: str, config: AdapterConfig) -> InputAdapter:
    """Create the data-ingress adapter for a supported source type."""
    try:
        adapter_type = ADAPTERS[input_type]
    except KeyError as error:
        supported = ", ".join(supported_input_types())
        raise ValueError(
            f"Unsupported GRIL input type {input_type!r}; expected one of: {supported}"
        ) from error
    return adapter_type(config)


__all__ = [
    "ADAPTERS",
    "CoreApolloRecordAdapter",
    "Rosbag1Adapter",
    "create_adapter",
    "supported_input_types",
]
