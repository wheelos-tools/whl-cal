#!/usr/bin/env python3

"""Capture-source utilities for camera intrinsic calibration."""

import os
import threading
import time

import cv2
import numpy as np


def display_size(cfg):
    return (
        int(cfg.get("window_width", 1280)),
        int(cfg.get("window_height", 720)),
    )


def resolve_capture_source(cfg):
    cameras = cfg.get("cameras")
    if isinstance(cameras, list) and cameras:
        selector = cfg.get("camera_selection", cfg.get("camera_index", 0))
        try:
            selected = int(selector)
        except (TypeError, ValueError):
            selected = 0
        selected = max(0, min(len(cameras) - 1, selected))
        entry = cameras[selected]
        if isinstance(entry, dict):
            source = entry.get("uri")
            camera_cfg = dict(entry)
        else:
            source = str(entry)
            camera_cfg = {}
        if not source:
            source = int(cfg.get("camera_index", 0))
        return source, {
            "selected_camera_index": int(selected),
            "camera_config": camera_cfg,
            "source_type": "network" if isinstance(source, str) else "device",
        }

    camera_uri = cfg.get("camera_uri")
    if camera_uri:
        return str(camera_uri), {
            "selected_camera_index": None,
            "camera_config": {},
            "source_type": "network",
        }

    return int(cfg.get("camera_index", 0)), {
        "selected_camera_index": None,
        "camera_config": {},
        "source_type": "device",
    }


def capture_config(cfg, capture_source_meta):
    capture_cfg = cfg.get("capture", {}) or {}
    camera_cfg = capture_source_meta.get("camera_config", {}) or {}
    width = capture_cfg.get("width", cfg.get("capture_width", camera_cfg.get("width")))
    height = capture_cfg.get(
        "height", cfg.get("capture_height", camera_cfg.get("height"))
    )
    fps = capture_cfg.get("fps", camera_cfg.get("fps"))
    buffersize = capture_cfg.get("buffersize", camera_cfg.get("buffersize"))
    return {
        "force_resolution": bool(
            capture_cfg.get(
                "force_resolution",
                cfg.get("force_capture_resolution", False),
            )
        ),
        "width": None if width in (None, "") else int(width),
        "height": None if height in (None, "") else int(height),
        "fourcc": capture_cfg.get("fourcc"),
        "fps": None if fps in (None, "") else float(fps),
        "codec": camera_cfg.get("codec"),
        "buffersize": None if buffersize in (None, "") else int(buffersize),
    }


def _is_rtsp_source(source):
    return isinstance(source, str) and source.lower().startswith("rtsp://")


def open_capture(source):
    backends = []

    if isinstance(source, str):
        if _is_rtsp_source(source) and not os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
            )

        ffmpeg_backend = getattr(cv2, "CAP_FFMPEG", None)
        gstreamer_backend = getattr(cv2, "CAP_GSTREAMER", None)
        if ffmpeg_backend is not None:
            backends.append(("FFMPEG", ffmpeg_backend))
        if gstreamer_backend is not None:
            backends.append(("GSTREAMER", gstreamer_backend))
        backends.append(("ANY", None))
    else:
        v4l2_backend = getattr(cv2, "CAP_V4L2", None)
        if v4l2_backend is not None:
            backends.append(("V4L2", v4l2_backend))
        backends.append(("ANY", None))

    for backend_name, backend in backends:
        cap = cv2.VideoCapture(source) if backend is None else cv2.VideoCapture(source, backend)
        if cap.isOpened():
            print(f"[INFO] Capture source opened via backend={backend_name}")
            return cap, backend_name
        cap.release()

    return None, None


class LatestFrameCapture:
    managed_capture = True

    def __init__(self, source, cfg, capture_source_meta):
        self.source = source
        self.cfg = cfg
        self.capture_source_meta = capture_source_meta or {}
        capture_cfg = cfg.get("capture", {}) or {}
        self.read_timeout_s = float(capture_cfg.get("latest_frame_read_timeout_s", 0.25))
        self.initial_ready_timeout_s = float(capture_cfg.get("initial_ready_timeout_s", 10.0))
        self.warmup_frames = max(0, int(capture_cfg.get("warmup_frames", 12)))
        self.reconnect_bad_frame_burst = max(
            10,
            int(capture_cfg.get("reconnect_bad_frame_burst", 30)),
        )
        self.reconnect_sleep_s = float(capture_cfg.get("reconnect_sleep_s", 0.5))

        self.backend_name = None
        self._cap = None
        self._latest_frame = None
        self._latest_frame_id = 0
        self._last_delivered_frame_id = 0
        self._consecutive_bad_frames = 0
        self._warmup_remaining = self.warmup_frames
        self._open_count = 0
        self._reconnect_count = 0
        self._total_empty_reads = 0
        self._total_invalid_frames = 0
        self._total_valid_frames = 0
        self._total_delivered_frames = 0
        self._warmup_cycles = 0
        self._last_reconnect_reason = None
        self._opened_at_monotonic = None
        self._has_seen_valid_frame_since_open = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._new_frame_event = threading.Event()
        self._ready_event = threading.Event()

        if self._open_stream():
            self._reader_thread = threading.Thread(
                target=self._reader_loop,
                name="camera-intrinsic-latest-frame-reader",
                daemon=True,
            )
            self._reader_thread.start()
        else:
            self._reader_thread = None

    def _open_stream(self):
        cap, backend_name = open_capture(self.source)
        if cap is None:
            return False
        apply_capture_settings(cap, self.cfg, self.capture_source_meta)
        with self._lock:
            self._cap = cap
            self.backend_name = backend_name
            self._consecutive_bad_frames = 0
            self._warmup_remaining = self.warmup_frames
            self._open_count += 1
            self._warmup_cycles += 1
            self._opened_at_monotonic = time.monotonic()
            self._has_seen_valid_frame_since_open = False
        if self.warmup_frames > 0:
            print(
                "[INFO] Warming up network stream:",
                f"skipping first {self.warmup_frames} valid frames after connect/reconnect.",
            )
        return True

    def _close_stream(self):
        with self._lock:
            cap = self._cap
            self._cap = None
        if cap is not None:
            cap.release()

    def _reopen_stream(self, reason):
        print(f"[WARN] Reopening network capture: {reason}")
        with self._lock:
            self._reconnect_count += 1
            self._last_reconnect_reason = str(reason)
        self._ready_event.clear()
        self._new_frame_event.clear()
        self._close_stream()
        time.sleep(max(0.0, self.reconnect_sleep_s))

    def _startup_grace_active(self):
        with self._lock:
            opened_at_monotonic = self._opened_at_monotonic
            has_seen_valid_frame_since_open = self._has_seen_valid_frame_since_open
        if opened_at_monotonic is None or has_seen_valid_frame_since_open:
            return False
        return (time.monotonic() - opened_at_monotonic) < self.initial_ready_timeout_s

    def _reader_loop(self):
        while not self._stop_event.is_set():
            with self._lock:
                cap = self._cap
            if cap is None:
                if not self._open_stream():
                    time.sleep(max(0.1, self.reconnect_sleep_s))
                continue

            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                with self._lock:
                    self._total_empty_reads += 1
                self._consecutive_bad_frames += 1
                if self._startup_grace_active():
                    time.sleep(0.01)
                    continue
                if self._consecutive_bad_frames >= self.reconnect_bad_frame_burst:
                    self._reopen_stream("consecutive empty read failures")
                else:
                    time.sleep(0.01)
                continue

            if is_visually_empty_frame(frame):
                with self._lock:
                    self._total_invalid_frames += 1
                self._consecutive_bad_frames += 1
                if self._startup_grace_active():
                    continue
                if self._consecutive_bad_frames >= self.reconnect_bad_frame_burst:
                    self._reopen_stream("consecutive invalid decoded frames")
                continue

            with self._lock:
                self._total_valid_frames += 1
                self._has_seen_valid_frame_since_open = True
            self._consecutive_bad_frames = 0
            if self._warmup_remaining > 0:
                self._warmup_remaining -= 1
                if self._warmup_remaining == 0:
                    print("[INFO] Network stream warm-up complete; live detection enabled.")
                continue

            with self._lock:
                self._latest_frame = frame
                self._latest_frame_id += 1
            self._ready_event.set()
            self._new_frame_event.set()

    def wait_until_ready(self, timeout=None):
        wait_timeout = self.initial_ready_timeout_s if timeout is None else float(timeout)
        return self._ready_event.wait(timeout=max(0.0, wait_timeout))

    def read(self):
        deadline = time.time() + max(0.0, self.read_timeout_s)
        while not self._stop_event.is_set():
            with self._lock:
                if (
                    self._latest_frame is not None
                    and self._latest_frame_id != self._last_delivered_frame_id
                ):
                    frame = self._latest_frame.copy()
                    self._last_delivered_frame_id = self._latest_frame_id
                    self._total_delivered_frames += 1
                    return True, frame
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            self._new_frame_event.wait(timeout=min(0.05, remaining))
            self._new_frame_event.clear()
        return False, None

    def get(self, prop_id):
        with self._lock:
            cap = self._cap
        if cap is None:
            return 0.0
        return float(cap.get(prop_id))

    def diagnostics(self):
        with self._lock:
            startup_grace_remaining_s = 0.0
            if self._opened_at_monotonic is not None and not self._has_seen_valid_frame_since_open:
                startup_grace_remaining_s = max(
                    0.0,
                    float(self.initial_ready_timeout_s)
                    - (time.monotonic() - self._opened_at_monotonic),
                )
            return {
                "backend_name": self.backend_name,
                "open_count": int(self._open_count),
                "reconnect_count": int(self._reconnect_count),
                "warmup_cycles": int(self._warmup_cycles),
                "warmup_frames": int(self.warmup_frames),
                "warmup_remaining": int(self._warmup_remaining),
                "consecutive_bad_frames": int(self._consecutive_bad_frames),
                "total_empty_reads": int(self._total_empty_reads),
                "total_invalid_frames": int(self._total_invalid_frames),
                "total_valid_frames": int(self._total_valid_frames),
                "total_delivered_frames": int(self._total_delivered_frames),
                "latest_frame_id": int(self._latest_frame_id),
                "last_reconnect_reason": self._last_reconnect_reason,
                "has_seen_valid_frame_since_open": bool(
                    self._has_seen_valid_frame_since_open
                ),
                "startup_grace_remaining_s": float(startup_grace_remaining_s),
                "ready": bool(self._ready_event.is_set()),
            }

    def is_ready(self):
        return bool(self._ready_event.is_set())

    def release(self):
        self._stop_event.set()
        self._new_frame_event.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
        self._close_stream()


def open_managed_capture(source, cfg, capture_source_meta):
    if (capture_source_meta or {}).get("source_type") == "network":
        cap = LatestFrameCapture(source, cfg, capture_source_meta)
        if cap.backend_name is None:
            return None, None
        if not cap.wait_until_ready():
            print(
                "[WARN] Network capture did not deliver a ready frame during startup warm-up; continuing to wait in the main loop."
            )
        return cap, cap.backend_name
    return open_capture(source)


def apply_capture_settings(cap, cfg, capture_source_meta):
    if getattr(cap, "managed_capture", False):
        return
    cap_cfg = capture_config(cfg, capture_source_meta)
    if cap_cfg.get("fps") is not None:
        cap.set(cv2.CAP_PROP_FPS, float(cap_cfg["fps"]))
    requested_buffersize = cap_cfg.get("buffersize")
    if requested_buffersize is None and capture_source_meta.get("source_type") == "network":
        requested_buffersize = 1
    if requested_buffersize is not None and hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        if cap.set(cv2.CAP_PROP_BUFFERSIZE, int(requested_buffersize)):
            print(f"[INFO] Requested capture buffer size: {int(requested_buffersize)}")
    fourcc = cap_cfg.get("fourcc")
    if isinstance(fourcc, str) and len(fourcc) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if (
        cap_cfg["force_resolution"]
        and cap_cfg["width"] is not None
        and cap_cfg["height"] is not None
    ):
        print(
            "[INFO] Requesting capture resolution "
            f"{cap_cfg['width']}x{cap_cfg['height']}."
        )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_cfg["height"])
    else:
        print(
            "[INFO] Capture resolution is not forced. "
            "Using the camera's native/as-is output avoids accidental FOV crop."
        )
    if cap_cfg.get("codec"):
        print(f"[INFO] Configured camera codec hint: {cap_cfg['codec']}")


def build_capture_runtime_info(cfg, capture_source, capture_source_meta, cap, frame):
    cap_cfg = capture_config(cfg, capture_source_meta)
    display_w, display_h = display_size(cfg)
    actual_h, actual_w = frame.shape[:2]
    reported_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    reported_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    render_scale = min(display_w / max(actual_w, 1), display_h / max(actual_h, 1))
    render_w = max(1, int(actual_w * render_scale))
    render_h = max(1, int(actual_h * render_scale))
    pad_x = max(0, (display_w - render_w) // 2)
    pad_y = max(0, (display_h - render_h) // 2)
    warnings = []
    if cap_cfg["force_resolution"]:
        warnings.append(
            "Forced capture resolution can crop some sensors before the 3x3 grid is drawn."
        )
        warnings.append(
            "If the live grid looks clipped, disable capture.force_resolution or switch to a native mode such as 4:3."
        )
    if abs((display_w / max(display_h, 1)) - (actual_w / max(actual_h, 1))) > 0.05:
        warnings.append(
            "Display window aspect differs from capture; the app letterboxes for display, but that is not sensor crop."
        )
    runtime_info = {
        "capture_source": str(capture_source),
        "capture_source_type": capture_source_meta.get("source_type"),
        "selected_camera_index": capture_source_meta.get("selected_camera_index"),
        "requested_capture_resolution": (
            None
            if cap_cfg["width"] is None or cap_cfg["height"] is None
            else {
                "width": int(cap_cfg["width"]),
                "height": int(cap_cfg["height"]),
            }
        ),
        "force_capture_resolution": bool(cap_cfg["force_resolution"]),
        "fourcc": cap_cfg.get("fourcc"),
        "fps": cap_cfg.get("fps"),
        "codec": cap_cfg.get("codec"),
        "requested_buffersize": cap_cfg.get("buffersize"),
        "actual_capture_resolution": {
            "width": int(actual_w),
            "height": int(actual_h),
        },
        "reported_capture_resolution": {
            "width": int(reported_w),
            "height": int(reported_h),
        },
        "display_resolution": {
            "width": int(display_w),
            "height": int(display_h),
        },
        "display_rendering": {
            "render_width": int(render_w),
            "render_height": int(render_h),
            "pad_x": int(pad_x),
            "pad_y": int(pad_y),
            "scale": float(render_scale),
            "aspect_ratio_preserved": True,
            "fills_display_without_padding": bool(
                render_w == display_w and render_h == display_h
            ),
        },
        "warnings": warnings,
    }
    if hasattr(cap, "diagnostics"):
        runtime_info["stream_health"] = cap.diagnostics()
    return runtime_info


def log_capture_runtime_info(runtime_info):
    actual = runtime_info["actual_capture_resolution"]
    display = runtime_info["display_resolution"]
    rendering = runtime_info.get("display_rendering") or {}
    requested = runtime_info.get("requested_capture_resolution")
    print(
        "[INFO] Live capture resolution:",
        f"{actual['width']}x{actual['height']}",
        "| display window:",
        f"{display['width']}x{display['height']}",
    )
    if rendering:
        print(
            "[INFO] Display rendering:",
            f"render={rendering.get('render_width', 0)}x{rendering.get('render_height', 0)}",
            f"pad=({rendering.get('pad_x', 0)},{rendering.get('pad_y', 0)})",
            f"scale={rendering.get('scale', 0.0):.3f}",
        )
    if requested is not None:
        print(
            "[INFO] Requested capture resolution:",
            f"{requested['width']}x{requested['height']}",
        )
    if runtime_info.get("requested_buffersize") is not None:
        print(
            "[INFO] Requested capture buffer size:",
            int(runtime_info["requested_buffersize"]),
        )
    stream_health = runtime_info.get("stream_health") or {}
    if stream_health:
        print(
            "[INFO] Stream health:",
            f"reconnects={stream_health.get('reconnect_count', 0)}",
            f"invalid_frames={stream_health.get('total_invalid_frames', 0)}",
            f"empty_reads={stream_health.get('total_empty_reads', 0)}",
            f"delivered_frames={stream_health.get('total_delivered_frames', 0)}",
        )
    for warning in runtime_info.get("warnings", []):
        print("[WARN]", warning)


def is_visually_empty_frame(frame):
    if frame is None or frame.size == 0:
        return True
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_mean = float(np.mean(gray))
    gray_std = float(np.std(gray))
    if gray_std < 1.0:
        return True
    return gray_mean < 1.0 and gray_std < 1.0
