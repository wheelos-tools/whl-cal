"""Lie-group derivatives and tangent conversions in [rotation, translation] order."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_right_jacobian(rotvec: np.ndarray) -> np.ndarray:
    """Map additive rotvec changes to right-local rotation increments."""
    theta2 = float(rotvec @ rotvec)
    K = skew(rotvec)
    if theta2 < 1e-8:
        a = 0.5 - theta2 / 24.0 + theta2**2 / 720.0
        b = 1.0 / 6.0 - theta2 / 120.0 + theta2**2 / 5040.0
    else:
        theta = np.sqrt(theta2)
        a = (1.0 - np.cos(theta)) / theta2
        b = (theta - np.sin(theta)) / (theta2 * theta)
    return np.eye(3) - a * K + b * (K @ K)


def se3_log6(transform: np.ndarray) -> np.ndarray:
    """Return the actual SE(3) logarithm, including rotation/translation coupling."""
    rotvec = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
    left_jacobian = so3_right_jacobian(-rotvec)
    translation = np.linalg.solve(left_jacobian, transform[:3, 3])
    return np.r_[rotvec, translation]


def information_left_to_right(
    information: np.ndarray, transform: np.ndarray
) -> np.ndarray:
    """Convert left-target information to the right-source tangent at transform."""
    rotation = transform[:3, :3]
    adjoint = np.block(
        [
            [rotation, np.zeros((3, 3))],
            [skew(transform[:3, 3]) @ rotation, rotation],
        ]
    )
    converted = adjoint.T @ information @ adjoint
    return 0.5 * (converted + converted.T)
