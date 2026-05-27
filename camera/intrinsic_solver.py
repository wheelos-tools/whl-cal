#!/usr/bin/env python3

"""Numerical camera intrinsic solver helpers."""

import cv2
import numpy as np


def calibrate_camera(objpoints, imgpoints, image_size_wh):
    return cv2.calibrateCamera(objpoints, imgpoints, tuple(image_size_wh), None, None)


def mean_reprojection_error(objpoints, imgpoints, camera_matrix, dist_coeffs, rvecs, tvecs):
    total_error = 0.0
    for index in range(len(objpoints)):
        projected_points, _ = cv2.projectPoints(
            objpoints[index],
            rvecs[index],
            tvecs[index],
            camera_matrix,
            dist_coeffs,
        )
        total_error += cv2.norm(
            imgpoints[index], projected_points, cv2.NORM_L2
        ) / len(projected_points)
    return total_error / max(len(objpoints), 1)


def build_undistortion_model(camera_matrix, dist_coeffs, image_size_wh, preview_cfg, alpha=None):
    preview_alpha = float(preview_cfg.get("alpha", 1.0) if alpha is None else alpha)
    width, height = map(int, image_size_wh)
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


def undistort_for_preview(image, camera_matrix, dist_coeffs, preview_cfg, alpha=None):
    new_camera_matrix, preview_info = build_undistortion_model(
        camera_matrix,
        dist_coeffs,
        (image.shape[1], image.shape[0]),
        preview_cfg,
        alpha=alpha,
    )
    undistorted = cv2.undistort(
        image,
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix,
    )
    return undistorted, preview_info
