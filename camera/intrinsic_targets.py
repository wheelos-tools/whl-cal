#!/usr/bin/env python3

"""Calibration target detection utilities for intrinsic calibration."""

from dataclasses import dataclass

import cv2
import numpy as np

try:
    from pupil_apriltags import Detector as PupilAprilTagDetector
except ImportError:  # pragma: no cover - optional dependency
    PupilAprilTagDetector = None


_APRILTAG_DICTIONARY_TO_FAMILY = {
    "DICT_APRILTAG_16h5": "tag16h5",
    "DICT_APRILTAG_25h9": "tag25h9",
    "DICT_APRILTAG_36h10": "tag36h10",
    "DICT_APRILTAG_36h11": "tag36h11",
}


@dataclass
class DetectionResult:
    found: bool
    image_points: np.ndarray = None
    object_points: np.ndarray = None
    marker_corners: list = None
    marker_ids: np.ndarray = None
    feature_ids: np.ndarray = None
    debug_info: dict = None


class CalibrationTargetDetector:
    """Unified detector for chessboard and AprilGrid targets."""

    def __init__(self, cfg, pattern_size=None, square_size=None):
        self.cfg = cfg
        self.target_type = str(
            cfg.get("target_type", cfg.get("pattern_type", "chessboard"))
        ).lower()
        self.pattern_size = None if pattern_size is None else tuple(pattern_size)
        self.square_size = None if square_size is None else float(square_size)

        if self.target_type not in ("chessboard", "aprilgrid", "charuco"):
            raise ValueError(
                "target_type must be one of: 'chessboard', 'aprilgrid', 'charuco'. "
                f"Got: {self.target_type}"
            )

        self.last_result = DetectionResult(found=False)
        self._init_chessboard_reference_if_needed()
        self._init_aprilgrid_if_needed()
        self._init_charuco_if_needed()

    def _normalize_scale_factors(self, configured_scales, default_scales):
        values = configured_scales if configured_scales is not None else default_scales
        if not isinstance(values, (list, tuple)):
            values = [values]

        scales = []
        for raw_value in values:
            try:
                scale = float(raw_value)
            except (TypeError, ValueError):
                continue
            if scale <= 0:
                continue
            if any(abs(scale - existing) < 1e-6 for existing in scales):
                continue
            scales.append(scale)

        if not scales:
            scales = [1.0]
        return tuple(scales)

    def _init_chessboard_reference_if_needed(self):
        if self.target_type != "chessboard":
            self.chessboard_objp = None
            return
        if self.pattern_size is None or self.square_size is None:
            raise ValueError(
                "Chessboard calibration requires pattern_size and square_size "
                "in the config."
            )
        self.chessboard_objp = np.zeros(
            (self.pattern_size[0] * self.pattern_size[1], 3), np.float32
        )
        self.chessboard_objp[:, :2] = np.mgrid[
            0 : self.pattern_size[0], 0 : self.pattern_size[1]
        ].T.reshape(-1, 2)
        self.chessboard_objp *= self.square_size

    def update_chessboard_pattern(self, pattern_size, square_size):
        self.pattern_size = tuple(pattern_size)
        self.square_size = float(square_size)
        self._init_chessboard_reference_if_needed()

    def _init_aprilgrid_if_needed(self):
        self.apr_cfg = self.cfg.get("aprilgrid", {}) or {}
        self.april_dictionary_name = str(
            self.apr_cfg.get("dictionary", "DICT_APRILTAG_36h11")
        )
        self.min_tags_per_frame = int(self.apr_cfg.get("min_tags_per_frame", 6))
        self.april_detection_scale_factors = self._normalize_scale_factors(
            self.apr_cfg.get("detection_scale_factors"),
            [1.0, 2.0, 3.0],
        )

        if self.target_type != "aprilgrid":
            self.aruco_detector = None
            self.april_board = None
            self.pupil_april_detector = None
            self.april_board_marker_lookup = {}
            return

        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "OpenCV aruco module is required for aprilgrid target_type."
            )

        dictionary_id = getattr(cv2.aruco, self.april_dictionary_name, None)
        if dictionary_id is None:
            dictionary_id = getattr(cv2.aruco, "DICT_APRILTAG_36h11")
            self.april_dictionary_name = "DICT_APRILTAG_36h11"

        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        detector_params = cv2.aruco.DetectorParameters()
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

        cols = int(self.apr_cfg.get("grid_cols", self.apr_cfg.get("columns", 6)))
        rows = int(self.apr_cfg.get("grid_rows", self.apr_cfg.get("rows", 6)))
        tag_size = float(self.apr_cfg.get("tag_size", 0.04))
        spacing = self.apr_cfg.get("tag_spacing")
        if spacing is None:
            ratio = float(self.apr_cfg.get("tag_spacing_ratio", 0.3))
            spacing = tag_size * ratio
        self.april_board = cv2.aruco.GridBoard(
            (cols, rows), tag_size, float(spacing), dictionary
        )
        self.april_board_shape = (cols, rows)
        self.april_tag_size = tag_size
        self.april_tag_spacing = float(spacing)
        self.april_board_marker_lookup = {}
        board_ids = np.asarray(self.april_board.getIds(), dtype=np.int32).reshape(-1)
        board_obj_points = list(self.april_board.getObjPoints())
        for marker_id, corners in zip(board_ids.tolist(), board_obj_points):
            self.april_board_marker_lookup[int(marker_id)] = np.asarray(
                corners, dtype=np.float32
            ).reshape(1, 4, 3)
        family = _APRILTAG_DICTIONARY_TO_FAMILY.get(self.april_dictionary_name)
        self.pupil_april_detector = None
        if PupilAprilTagDetector is not None and family is not None:
            self.pupil_april_detector = PupilAprilTagDetector(
                families=family,
                nthreads=max(1, int(self.apr_cfg.get("pupil_nthreads", 1))),
                quad_decimate=float(self.apr_cfg.get("pupil_quad_decimate", 1.0)),
                quad_sigma=float(self.apr_cfg.get("pupil_quad_sigma", 0.0)),
                refine_edges=1,
                decode_sharpening=float(
                    self.apr_cfg.get("pupil_decode_sharpening", 0.25)
                ),
            )

    def _init_charuco_if_needed(self):
        self.charuco_cfg = self.cfg.get("charuco", {}) or {}
        self.charuco_dictionary_name = str(
            self.charuco_cfg.get("dictionary", "DICT_4X4_100")
        )
        self.min_charuco_corners = int(
            self.charuco_cfg.get("min_corners_per_frame", 12)
        )

        if self.target_type != "charuco":
            self.charuco_detector = None
            self.charuco_board = None
            return

        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "OpenCV aruco module is required for charuco target_type."
            )

        dictionary_id = getattr(cv2.aruco, self.charuco_dictionary_name, None)
        if dictionary_id is None:
            dictionary_id = getattr(cv2.aruco, "DICT_4X4_100")
            self.charuco_dictionary_name = "DICT_4X4_100"

        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        squares_x = int(self.charuco_cfg.get("squares_x", 6))
        squares_y = int(self.charuco_cfg.get("squares_y", 8))
        square_length = float(self.charuco_cfg.get("square_length", 0.04))
        marker_length = float(self.charuco_cfg.get("marker_length", 0.02))

        if marker_length >= square_length:
            raise ValueError(
                "charuco.marker_length must be smaller than charuco.square_length"
            )

        self.charuco_board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            square_length,
            marker_length,
            dictionary,
        )
        self.charuco_detector = cv2.aruco.CharucoDetector(
            self.charuco_board,
            cv2.aruco.CharucoParameters(),
            cv2.aruco.DetectorParameters(),
        )
        self.charuco_shape = (squares_x, squares_y)
        self.charuco_square_length = square_length
        self.charuco_marker_length = marker_length

    def target_config(self):
        if self.target_type == "aprilgrid":
            return {
                "type": "aprilgrid",
                "dictionary": self.april_dictionary_name,
                "grid_cols": int(self.april_board_shape[0]),
                "grid_rows": int(self.april_board_shape[1]),
                "tag_size": float(self.april_tag_size),
                "tag_spacing": float(self.april_tag_spacing),
                "min_tags_per_frame": int(self.min_tags_per_frame),
                "min_points_per_frame": int(self.min_tags_per_frame) * 4,
            }
        if self.target_type == "charuco":
            return {
                "type": "charuco",
                "dictionary": self.charuco_dictionary_name,
                "squares_x": int(self.charuco_shape[0]),
                "squares_y": int(self.charuco_shape[1]),
                "square_length": float(self.charuco_square_length),
                "marker_length": float(self.charuco_marker_length),
                "min_corners_per_frame": int(self.min_charuco_corners),
                "min_points_per_frame": int(self.min_charuco_corners),
            }
        return {
            "type": "chessboard",
            "pattern_size": [int(self.pattern_size[0]), int(self.pattern_size[1])],
            "square_size": float(self.square_size),
        }

    def detect(self, gray, frame_counter, optimization_cfg):
        detection_interval = max(
            1,
            int((optimization_cfg or {}).get("detection_interval", 1)),
        )
        if frame_counter % detection_interval != 0:
            return DetectionResult(
                found=False,
                debug_info={
                    "target_type": self.target_type,
                    "skipped": True,
                    "skip_reason": "detection_interval",
                    "detection_interval": int(detection_interval),
                },
            )

        if self.target_type == "aprilgrid":
            result = self._detect_aprilgrid(gray, optimization_cfg)
        elif self.target_type == "charuco":
            result = self._detect_charuco(gray)
        else:
            result = self._detect_chessboard(gray, optimization_cfg)

        self.last_result = result
        return result

    def _detect_chessboard(self, gray, optimization_cfg):
        factor = float(optimization_cfg["resize_factor"])
        small = cv2.resize(
            gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA
        )
        found, small_corners = cv2.findChessboardCorners(
            small,
            self.pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK,
        )
        if not found:
            return DetectionResult(found=False)

        corners = np.asarray(small_corners / factor, dtype=np.float32)
        refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        return DetectionResult(
            found=True,
            image_points=refined,
            object_points=self.chessboard_objp.copy(),
            feature_ids=np.arange(refined.shape[0], dtype=np.int32).reshape(-1, 1),
        )

    def _detect_aprilgrid_markers_pupil(self, gray, normalized_scales):
        if self.pupil_april_detector is None:
            return None
        attempts = []
        best_marker_corners = None
        best_marker_ids = None
        best_marker_count = -1
        best_scale = None
        for effective_scale in normalized_scales:
            if abs(effective_scale - 1.0) < 1e-6:
                scaled_gray = gray
            else:
                interpolation = (
                    cv2.INTER_CUBIC if effective_scale > 1.0 else cv2.INTER_AREA
                )
                scaled_gray = cv2.resize(
                    gray,
                    None,
                    fx=effective_scale,
                    fy=effective_scale,
                    interpolation=interpolation,
                )
            detections = self.pupil_april_detector.detect(
                scaled_gray,
                estimate_tag_pose=False,
            )
            detections = [
                detection
                for detection in detections
                if int(getattr(detection, "tag_id", -1))
                in self.april_board_marker_lookup
            ]
            detections.sort(key=lambda item: int(getattr(item, "tag_id", -1)))
            marker_corners = []
            marker_ids = []
            for detection in detections:
                corners = np.asarray(detection.corners, dtype=np.float32).reshape(
                    1, 4, 2
                )
                if abs(effective_scale - 1.0) >= 1e-6:
                    corners = corners / effective_scale
                marker_corners.append(corners)
                marker_ids.append([int(detection.tag_id)])
            marker_count = int(len(marker_ids))
            attempts.append(
                {
                    "backend": "pupil_apriltags",
                    "scale": float(effective_scale),
                    "detected_marker_count": marker_count,
                    "detected_marker_ids": [
                        int(value[0])
                        for value in marker_ids[: min(20, len(marker_ids))]
                    ],
                    "rejected_candidate_count": 0,
                    "met_min_tags_threshold": bool(
                        marker_count >= self.min_tags_per_frame
                    ),
                }
            )
            if marker_count > best_marker_count or (
                marker_count == best_marker_count
                and best_scale is not None
                and abs(effective_scale - 1.0) < abs(best_scale - 1.0)
            ):
                best_marker_corners = marker_corners
                best_marker_ids = (
                    np.asarray(marker_ids, dtype=np.int32).reshape(-1, 1)
                    if marker_ids
                    else None
                )
                best_marker_count = marker_count
                best_scale = float(effective_scale)
        selected_ids = []
        if best_marker_ids is not None:
            selected_ids = [
                int(value) for value in np.asarray(best_marker_ids).reshape(-1).tolist()
            ]
        return (
            best_marker_corners,
            best_marker_ids,
            {
                "target_type": "aprilgrid",
                "detector_backend": "pupil_apriltags",
                "dictionary": self.april_dictionary_name,
                "candidate_scales": [float(value) for value in normalized_scales],
                "attempts": attempts,
                "min_tags_per_frame": int(self.min_tags_per_frame),
                "detected_marker_count": int(max(best_marker_count, 0)),
                "selected_scale": best_scale,
                "selected_marker_ids": selected_ids,
                "selected_rejected_candidate_count": 0,
                "selected_by": "max_marker_count",
            },
        )

    def _detect_aprilgrid_markers_opencv(self, gray, normalized_scales):
        attempts = []
        best_marker_corners = None
        best_marker_ids = None
        best_marker_count = -1
        best_scale = None
        best_rejected_count = 0

        for effective_scale in normalized_scales:
            if abs(effective_scale - 1.0) < 1e-6:
                scaled_gray = gray
            else:
                interpolation = (
                    cv2.INTER_CUBIC if effective_scale > 1.0 else cv2.INTER_AREA
                )
                scaled_gray = cv2.resize(
                    gray,
                    None,
                    fx=effective_scale,
                    fy=effective_scale,
                    interpolation=interpolation,
                )

            marker_corners, marker_ids, _rejected = self.aruco_detector.detectMarkers(
                scaled_gray
            )
            marker_count = 0 if marker_ids is None else int(len(marker_ids))
            rejected_count = 0 if _rejected is None else int(len(_rejected))
            flattened_ids = []
            if marker_ids is not None:
                flattened_ids = [
                    int(value) for value in np.asarray(marker_ids).reshape(-1).tolist()
                ]
            attempts.append(
                {
                    "backend": "opencv_aruco",
                    "scale": float(effective_scale),
                    "detected_marker_count": int(marker_count),
                    "detected_marker_ids": flattened_ids,
                    "rejected_candidate_count": int(rejected_count),
                    "met_min_tags_threshold": bool(
                        marker_count >= self.min_tags_per_frame
                    ),
                }
            )
            if marker_count <= 0:
                continue

            if abs(effective_scale - 1.0) >= 1e-6:
                marker_corners = [
                    np.asarray(corners, dtype=np.float32) / effective_scale
                    for corners in marker_corners
                ]

            if marker_count > best_marker_count or (
                marker_count == best_marker_count
                and best_scale is not None
                and abs(effective_scale - 1.0) < abs(best_scale - 1.0)
            ):
                best_marker_corners = marker_corners
                best_marker_ids = np.asarray(marker_ids, dtype=np.int32).reshape(-1, 1)
                best_marker_count = int(marker_count)
                best_scale = float(effective_scale)
                best_rejected_count = int(rejected_count)

        selected_ids = []
        if best_marker_ids is not None:
            selected_ids = [
                int(value) for value in np.asarray(best_marker_ids).reshape(-1).tolist()
            ]
        return (
            best_marker_corners,
            best_marker_ids,
            {
                "target_type": "aprilgrid",
                "detector_backend": "opencv_aruco",
                "dictionary": self.april_dictionary_name,
                "candidate_scales": [float(value) for value in normalized_scales],
                "attempts": attempts,
                "min_tags_per_frame": int(self.min_tags_per_frame),
                "detected_marker_count": int(max(best_marker_count, 0)),
                "selected_scale": best_scale,
                "selected_marker_ids": selected_ids,
                "selected_rejected_candidate_count": int(best_rejected_count),
                "selected_by": "max_marker_count",
            },
        )

    def _detect_aprilgrid_markers(self, gray, optimization_cfg):
        base_factor = float((optimization_cfg or {}).get("resize_factor", 1.0))
        candidate_scales = []
        if base_factor > 0 and abs(base_factor - 1.0) >= 1e-6:
            candidate_scales.append(base_factor)
        candidate_scales.extend(self.april_detection_scale_factors)
        normalized_scales = self._normalize_scale_factors(candidate_scales, [1.0])
        pupil_result = self._detect_aprilgrid_markers_pupil(gray, normalized_scales)
        if pupil_result is not None:
            marker_corners, marker_ids, debug_info = pupil_result
            if marker_ids is not None and len(marker_ids) >= self.min_tags_per_frame:
                return marker_corners, marker_ids, debug_info
        return self._detect_aprilgrid_markers_opencv(gray, normalized_scales)

    def _detect_aprilgrid(self, gray, optimization_cfg):
        marker_corners, marker_ids, debug_info = self._detect_aprilgrid_markers(
            gray,
            optimization_cfg,
        )
        if marker_ids is None or len(marker_ids) < self.min_tags_per_frame:
            debug_info.update(
                {
                    "found": False,
                    "failure_stage": "min_tags_per_frame",
                    "matched_point_count": 0,
                }
            )
            return DetectionResult(
                found=False,
                marker_corners=marker_corners,
                marker_ids=marker_ids,
                debug_info=debug_info,
            )

        obj_points, img_points = self.april_board.matchImagePoints(
            marker_corners, marker_ids
        )
        if obj_points is None or img_points is None:
            debug_info.update(
                {
                    "found": False,
                    "failure_stage": "board_match",
                    "matched_point_count": 0,
                }
            )
            return DetectionResult(
                found=False,
                marker_corners=marker_corners,
                marker_ids=marker_ids,
                debug_info=debug_info,
            )

        obj_points = np.asarray(obj_points, dtype=np.float32).reshape(-1, 3)
        img_points = np.asarray(img_points, dtype=np.float32).reshape(-1, 1, 2)
        if obj_points.shape[0] < 8 or img_points.shape[0] < 8:
            debug_info.update(
                {
                    "found": False,
                    "failure_stage": "matched_points",
                    "matched_point_count": int(obj_points.shape[0]),
                }
            )
            return DetectionResult(
                found=False,
                marker_corners=marker_corners,
                marker_ids=marker_ids,
                debug_info=debug_info,
            )

        refined = cv2.cornerSubPix(
            gray,
            img_points,
            (5, 5),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01),
        )

        debug_info.update(
            {
                "found": True,
                "failure_stage": None,
                "matched_marker_count": int(len(marker_ids)),
                "matched_point_count": int(obj_points.shape[0]),
            }
        )

        return DetectionResult(
            found=True,
            image_points=refined,
            object_points=obj_points,
            marker_corners=marker_corners,
            marker_ids=marker_ids,
            feature_ids=marker_ids,
            debug_info=debug_info,
        )

    def _detect_charuco(self, gray):
        charuco_corners, charuco_ids, marker_corners, marker_ids = (
            self.charuco_detector.detectBoard(gray)
        )

        if charuco_ids is None or charuco_corners is None:
            return DetectionResult(found=False)
        if len(charuco_ids) < self.min_charuco_corners:
            return DetectionResult(found=False)

        obj_points, img_points = self.charuco_board.matchImagePoints(
            charuco_corners, charuco_ids
        )
        if obj_points is None or img_points is None:
            return DetectionResult(found=False)

        obj_points = np.asarray(obj_points, dtype=np.float32).reshape(-1, 3)
        img_points = np.asarray(img_points, dtype=np.float32).reshape(-1, 1, 2)
        if obj_points.shape[0] < 8 or img_points.shape[0] < 8:
            return DetectionResult(found=False)

        refined = cv2.cornerSubPix(
            gray,
            img_points,
            (5, 5),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01),
        )

        return DetectionResult(
            found=True,
            image_points=refined,
            object_points=obj_points,
            marker_corners=list(marker_corners) if marker_corners is not None else None,
            marker_ids=marker_ids,
            feature_ids=charuco_ids,
        )
