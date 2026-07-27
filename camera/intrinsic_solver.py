#!/usr/bin/env python3

"""Numerical camera intrinsic solver helpers."""

import cv2
import numpy as np


def normalize_distortion_model(distortion_model):
    aliases = {
        "plumb_bob": "plumb_bob",
        "pinhole": "plumb_bob",
        "opencv": "plumb_bob",
        "radtan": "plumb_bob",
        "fisheye": "fisheye",
        "equidistant": "fisheye",
    }
    key = str(distortion_model or "plumb_bob").strip().lower()
    if key not in aliases:
        raise ValueError(
            "Unsupported distortion_model. Use one of: "
            "plumb_bob/pinhole/opencv/radtan or fisheye/equidistant."
        )
    return aliases[key]


def _as_fisheye_object_points(points):
    return np.asarray(points, dtype=np.float64).reshape(-1, 1, 3)


def _as_fisheye_image_points(points):
    return np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)


def calibrate_camera(objpoints, imgpoints, image_size_wh, distortion_model="plumb_bob"):
    model = normalize_distortion_model(distortion_model)
    image_size = tuple(map(int, image_size_wh))
    if model == "fisheye":
        object_points = [_as_fisheye_object_points(points) for points in objpoints]
        image_points = [_as_fisheye_image_points(points) for points in imgpoints]
        camera_matrix = np.eye(3, dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            | cv2.fisheye.CALIB_CHECK_COND
            | cv2.fisheye.CALIB_FIX_SKEW
        )
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            1e-6,
        )
        return cv2.fisheye.calibrate(
            object_points,
            image_points,
            image_size,
            camera_matrix,
            dist_coeffs,
            flags=flags,
            criteria=criteria,
        )
    return cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)


def project_points(
    object_points,
    rvec,
    tvec,
    camera_matrix,
    dist_coeffs,
    distortion_model="plumb_bob",
):
    model = normalize_distortion_model(distortion_model)
    if model == "fisheye":
        return cv2.fisheye.projectPoints(
            _as_fisheye_object_points(object_points),
            np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            np.asarray(camera_matrix, dtype=np.float64),
            np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1),
        )
    return cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )


def mean_reprojection_error(
    objpoints,
    imgpoints,
    camera_matrix,
    dist_coeffs,
    rvecs,
    tvecs,
    distortion_model="plumb_bob",
):
    """Return the mean per-view 2D reprojection RMS in pixels."""
    total_error = 0.0
    valid_view_count = 0
    for index in range(len(objpoints)):
        if len(objpoints[index]) == 0 or len(imgpoints[index]) == 0:
            continue
        projected_points, _ = project_points(
            objpoints[index],
            rvecs[index],
            tvecs[index],
            camera_matrix,
            dist_coeffs,
            distortion_model=distortion_model,
        )
        if projected_points is None:
            continue
        point_count = len(projected_points)
        if point_count:
            total_error += cv2.norm(
                imgpoints[index], projected_points, cv2.NORM_L2
            ) / np.sqrt(point_count)
            valid_view_count += 1
    return total_error / max(valid_view_count, 1)


def build_undistortion_model(
    camera_matrix,
    dist_coeffs,
    image_size_wh,
    preview_cfg,
    alpha=None,
    distortion_model="plumb_bob",
):
    model = normalize_distortion_model(distortion_model)
    preview_alpha = float(preview_cfg.get("alpha", 1.0) if alpha is None else alpha)
    width, height = map(int, image_size_wh)
    if model == "fisheye":
        center_principal_point = False
        new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            np.asarray(camera_matrix, dtype=np.float64),
            np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1),
            (width, height),
            np.eye(3, dtype=np.float64),
            balance=float(np.clip(preview_alpha, 0.0, 1.0)),
            new_size=(width, height),
        )
        x, y, roi_w, roi_h = 0, 0, width, height
    else:
        center_principal_point = bool(preview_cfg.get("center_principal_point", False))
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            dist_coeffs,
            (width, height),
            preview_alpha,
            (width, height),
            centerPrincipalPoint=center_principal_point,
        )
        x, y, roi_w, roi_h = [int(value) for value in roi]
    preview_info = {
        "distortion_model": model,
        "alpha": float(preview_alpha),
        "center_principal_point": center_principal_point,
        "input_image_size": {"width": width, "height": height},
        "undistorted_image_size": {"width": width, "height": height},
        "preserves_input_resolution": True,
        "optimized_camera_matrix": np.asarray(new_camera_matrix, dtype=float).tolist(),
        "valid_roi": {
            "x": x,
            "y": y,
            "width": roi_w,
            "height": roi_h,
        },
    }
    return new_camera_matrix, preview_info


def undistort_for_preview(
    image,
    camera_matrix,
    dist_coeffs,
    preview_cfg,
    alpha=None,
    distortion_model="plumb_bob",
):
    model = normalize_distortion_model(distortion_model)
    new_camera_matrix, preview_info = build_undistortion_model(
        camera_matrix,
        dist_coeffs,
        (image.shape[1], image.shape[0]),
        preview_cfg,
        alpha=alpha,
        distortion_model=model,
    )
    if model == "fisheye":
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            np.asarray(camera_matrix, dtype=np.float64),
            np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1),
            np.eye(3, dtype=np.float64),
            np.asarray(new_camera_matrix, dtype=np.float64),
            (int(image.shape[1]), int(image.shape[0])),
            cv2.CV_16SC2,
        )
        undistorted = cv2.remap(
            image,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
    else:
        undistorted = cv2.undistort(
            image,
            camera_matrix,
            dist_coeffs,
            None,
            new_camera_matrix,
        )
    return undistorted, preview_info
