#!/usr/bin/env python
"""Compatibility wrapper: camera2lidar.learning_based -> lidar2camera.learning_based

The canonical implementation lives in lidar2camera/learning_based.py.
This small wrapper preserves backwards-compatibility for callers that run
`python camera2lidar/learning_based.py`.
"""

from __future__ import annotations

from importlib import import_module


def main() -> None:
    module = import_module("lidar2camera.learning_based")
    # The original module exposes a main() entrypoint; delegate to it.
    if hasattr(module, "main"):
        module.main()
    else:
        raise SystemExit("lidar2camera.learning_based has no main() entrypoint")


if __name__ == "__main__":
    main()
