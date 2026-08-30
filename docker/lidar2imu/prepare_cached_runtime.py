#!/usr/bin/env python3
"""Assemble a minimal GRIL runtime bundle from validated local artifacts."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

import yaml

CORE_RUNTIME_LIBRARIES = {
    "ld-linux-x86-64.so.2",
    "libc.so.6",
    "libdl.so.2",
    "libgcc_s.so.1",
    "libm.so.6",
    "libpthread.so.0",
    "librt.so.1",
    "libstdc++.so.6",
}
LDD_ENTRY = re.compile(r"^\s*(\S+)\s+=>\s+(\S+)\s+\(")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_libraries(binary: Path, library_prefix: Path) -> dict[str, Path]:
    environment = dict(os.environ)
    search_paths = (
        library_prefix / "usr/lib",
        library_prefix / "usr/lib/x86_64-linux-gnu",
    )
    environment["LD_LIBRARY_PATH"] = ":".join(str(path) for path in search_paths)
    output = subprocess.run(
        ["ldd", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout
    libraries = {}
    for line in output.splitlines():
        match = LDD_ENTRY.match(line)
        if match is None:
            continue
        name, resolved = match.groups()
        if name in CORE_RUNTIME_LIBRARIES:
            continue
        path = Path(resolved)
        if not path.is_file():
            raise FileNotFoundError(f"Unresolved runtime library {name}: {resolved}")
        libraries[name] = path
    return libraries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--library-prefix", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docker/lidar2imu/runtime"),
    )
    args = parser.parse_args()

    output_bin = args.output_dir / "bin"
    output_lib = args.output_dir / "lib"
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    output_bin.mkdir(parents=True)
    output_lib.mkdir(parents=True)

    binaries = {
        "gril_native_full_frontend": args.frontend,
        "gril_native_batch": args.batch,
    }
    libraries = {}
    for binary in binaries.values():
        libraries.update(runtime_libraries(binary, args.library_prefix))

    artifacts = {}
    for name, source in binaries.items():
        target = output_bin / name
        shutil.copy2(source, target)
        artifacts[f"bin/{name}"] = file_sha256(target)
    for name, source in sorted(libraries.items()):
        target = output_lib / name
        shutil.copy2(source.resolve(), target)
        artifacts[f"lib/{name}"] = file_sha256(target)

    manifest = {
        "schema_version": 1,
        "source": "validated_local_cache",
        "artifacts": artifacts,
    }
    (args.output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False)
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
