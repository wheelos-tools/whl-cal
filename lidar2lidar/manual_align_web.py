#!/usr/bin/env python3

"""Web UI for manually aligning multi-LiDAR point clouds to a fixed reference."""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from lidar2lidar.manual_align_loader import (
    align_edge_session,
    auto_align_session,
    export_manual_alignment,
    load_manual_align_session,
    probe_manual_align_frames,
)

STATIC_DIR = Path(__file__).resolve().parent / "static" / "manual_align"


class ManualAlignServerState:
    def __init__(
        self,
        *,
        record_path: str,
        conf_dir: str | None,
        target_topic: str | None,
        sync_threshold_ms: float,
        voxel_size: float | None,
        frame_index: int,
        output_dir: str,
        workflow_yaml: str | None,
    ) -> None:
        self.record_path = record_path
        self.conf_dir = conf_dir
        self.target_topic = target_topic
        self.sync_threshold_ms = sync_threshold_ms
        self.voxel_size = voxel_size
        self.frame_index = frame_index
        self.output_dir = output_dir
        self.workflow_yaml = workflow_yaml
        self.lock = threading.Lock()
        self.session = None
        self.last_error: str | None = None

    def get_frames(self) -> dict:
        return probe_manual_align_frames(
            record_path=self.record_path,
            conf_dir=self.conf_dir,
            target_topic=self.target_topic,
            sync_threshold_ms=self.sync_threshold_ms,
            workflow_yaml=self.workflow_yaml,
        )

    def reload(self, *, frame_index: int | None = None) -> dict:
        with self.lock:
            if frame_index is not None:
                self.frame_index = int(frame_index)
            try:
                self.session = load_manual_align_session(
                    record_path=self.record_path,
                    conf_dir=self.conf_dir,
                    target_topic=self.target_topic,
                    sync_threshold_ms=self.sync_threshold_ms,
                    voxel_size=self.voxel_size,
                    frame_index=self.frame_index,
                    workflow_yaml=self.workflow_yaml,
                )
                self.last_error = None
                return self.session.to_api_dict()
            except Exception as exc:
                self.last_error = str(exc)
                raise

    def get_points(self, frame_id: str) -> dict:
        with self.lock:
            if self.session is None:
                raise RuntimeError("Session not loaded.")
            for sensor in self.session.sensors:
                if sensor.frame_id == frame_id:
                    return {
                        "frame_id": sensor.frame_id,
                        "topic": sensor.topic,
                        "point_count": sensor.point_count,
                        "positions": sensor.positions,
                        "color_rgb": sensor.color_rgb,
                        "fixed": sensor.fixed,
                    }
            raise KeyError(f"Unknown frame_id: {frame_id}")

    def export(self, transforms: dict[str, list[list[float]]]) -> dict:
        with self.lock:
            if self.session is None:
                raise RuntimeError("Session not loaded.")
            return export_manual_alignment(
                self.session,
                transforms,
                self.output_dir,
            )

    def auto_align(
        self,
        *,
        relation_id: str | None = None,
        method: int = 2,
        registration_voxel_size: float | None = None,
        seed_transforms: dict[str, list[list[float]]] | None = None,
        max_windows: int | None = None,
        loop_closure: bool | None = None,
    ) -> dict:
        with self.lock:
            if self.session is None:
                raise RuntimeError("Session not loaded.")
            session = self.session
            default_registration_voxel = session.align_settings.get(
                "registration_voxel_size", 0.04
            )
        effective_registration_voxel = (
            float(registration_voxel_size)
            if registration_voxel_size is not None
            else float(default_registration_voxel)
        )
        logging.info("Auto-align started (relation_id=%s, method=%s).", relation_id, method)
        if relation_id:
            result = align_edge_session(
                session,
                relation_id,
                method=int(method),
                registration_voxel_size=effective_registration_voxel,
                seed_transforms=seed_transforms,
                max_windows=max_windows,
                loop_closure=bool(loop_closure) if loop_closure is not None else False,
            )
        else:
            result = auto_align_session(
                session,
                method=int(method),
                registration_voxel_size=effective_registration_voxel,
                seed_transforms=seed_transforms,
                max_windows=max_windows,
                loop_closure=loop_closure,
            )
        logging.info(
            "Auto-align finished: %s/%s edges succeeded.",
            result["summary"]["aligned_count"],
            result["summary"]["edge_count"],
        )
        return result


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length > 0 else b"{}"
    return json.loads(raw.decode("utf-8"))


def make_handler(state: ManualAlignServerState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            logging.info("%s - %s", self.address_string(), format % args)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/api/health":
                _json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "loaded": state.session is not None,
                        "error": state.last_error,
                    },
                )
                return

            if path == "/api/frames":
                try:
                    _json_response(self, 200, state.get_frames())
                except Exception as exc:
                    _json_response(self, 500, {"error": str(exc)})
                return

            if path == "/api/session":
                try:
                    query = parse_qs(parsed.query)
                    frame_index = query.get("frame_index", [None])[0]
                    payload = state.reload(
                        frame_index=int(frame_index)
                        if frame_index is not None
                        else None
                    )
                    _json_response(self, 200, payload)
                except Exception as exc:
                    _json_response(self, 500, {"error": str(exc)})
                return

            if path.startswith("/api/points/"):
                frame_id = path.rsplit("/", 1)[-1]
                try:
                    _json_response(self, 200, state.get_points(frame_id))
                except Exception as exc:
                    _json_response(self, 500, {"error": str(exc)})
                return

            if path == "/" or path == "/index.html":
                self._serve_file(STATIC_DIR / "index.html")
                return

            if path.startswith("/static/"):
                rel = path[len("/static/") :]
                self._serve_file(STATIC_DIR / rel)
                return

            self.send_error(404, "Not Found")

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/export":
                    payload = _read_json_body(self)
                    transforms = payload.get("transforms", {})
                    if not isinstance(transforms, dict):
                        raise ValueError(
                            "transforms must be an object keyed by frame_id."
                        )
                    result = state.export(transforms)
                    _json_response(self, 200, {"ok": True, **result})
                    return

                if parsed.path == "/api/auto-align":
                    payload = _read_json_body(self)
                    with state.lock:
                        default_registration_voxel = None
                        if state.session is not None:
                            default_registration_voxel = state.session.align_settings.get(
                                "registration_voxel_size"
                            )
                    registration_voxel_size = payload.get("registration_voxel_size")
                    if registration_voxel_size is None and default_registration_voxel is not None:
                        registration_voxel_size = default_registration_voxel
                    result = state.auto_align(
                        relation_id=payload.get("relation_id"),
                        method=int(payload.get("method", 2)),
                        registration_voxel_size=(
                            float(registration_voxel_size)
                            if registration_voxel_size is not None
                            else None
                        ),
                        seed_transforms=payload.get("seed_transforms"),
                        max_windows=(
                            int(payload["max_windows"])
                            if payload.get("max_windows") is not None
                            else None
                        ),
                        loop_closure=payload.get("loop_closure"),
                    )
                    _json_response(self, 200, {"ok": True, **result})
                    return

                self.send_error(404, "Not Found")
            except Exception as exc:
                _json_response(self, 500, {"error": str(exc)})

        def _serve_file(self, file_path: Path) -> None:
            if not file_path.exists() or not file_path.is_file():
                self.send_error(404, "Not Found")
                return
            content = file_path.read_bytes()
            mime, _ = mimetypes.guess_type(str(file_path))
            self.send_response(200)
            self.send_header("Content-Type", mime or "application/octet-stream")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache, must-revalidate")
            self.end_headers()
            self.wfile.write(content)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch a web UI to manually align LiDAR point clouds."
    )
    parser.add_argument(
        "--record-path",
        required=True,
        help="Apollo .record file or directory containing record files.",
    )
    parser.add_argument(
        "--conf-dir",
        default=None,
        help="Directory with *_extrinsics.yaml initial guesses.",
    )
    parser.add_argument(
        "--target-topic",
        default="/apollo/sensor/vanjeelidar/left_front/PointCloud2",
        help="Reference LiDAR topic (fixed in the UI).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/lidar2lidar/manual_align",
        help="Directory for exported extrinsics and merged point clouds.",
    )
    parser.add_argument(
        "--sync-threshold-ms",
        type=float,
        default=50.0,
        help="Maximum timestamp difference when picking synchronized frames.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=None,
        help="Voxel downsample size for web preview (meters); default from workflow YAML.",
    )
    parser.add_argument(
        "--workflow-yaml",
        default=None,
        help=(
            "Perimeter workflow YAML for topic/relation selection "
            "(default: conf/workflow_raw4_perimeter_loop.yaml)."
        ),
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Target-frame scan index to load from the record.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", type=int, default=8765, help="Bind port.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    state = ManualAlignServerState(
        record_path=args.record_path,
        conf_dir=args.conf_dir,
        target_topic=args.target_topic,
        sync_threshold_ms=args.sync_threshold_ms,
        voxel_size=args.voxel_size,
        frame_index=args.frame_index,
        output_dir=args.output_dir,
        workflow_yaml=args.workflow_yaml,
    )

    try:
        frames = state.get_frames()
        state.frame_index = int(frames.get("default_frame_index", 0))
        state.reload(frame_index=state.frame_index)
        logging.info(
            "Loaded frame index %d/%d (%d sensors).",
            state.frame_index,
            frames.get("target_frame_count", 0),
            len(state.session.sensors),
        )
    except Exception as exc:
        logging.warning("Initial load failed: %s", exc)

    handler = make_handler(state)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    logging.info("Manual align web UI: http://%s:%d", args.host, args.port)
    logging.info("Export directory: %s", args.output_dir)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
