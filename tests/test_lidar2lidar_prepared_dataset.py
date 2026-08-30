from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from lidar2lidar.prepared_dataset import _message_pose_to_matrix


class PreparedDatasetPoseExtractionTest(unittest.TestCase):
    def test_extracts_apollo_gps_localization_pose(self) -> None:
        msg = SimpleNamespace(
            localization=SimpleNamespace(
                position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
                orientation=SimpleNamespace(qx=0.0, qy=0.0, qz=0.0, qw=1.0),
            )
        )

        transform = _message_pose_to_matrix(msg)

        self.assertIsNotNone(transform)
        np.testing.assert_allclose(transform[:3, 3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(transform[:3, :3], np.eye(3))

    def test_extracts_localization_estimate_pose(self) -> None:
        msg = SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=4.0, y=5.0, z=6.0),
                orientation=SimpleNamespace(qx=0.0, qy=0.0, qz=0.0, qw=1.0),
            )
        )

        transform = _message_pose_to_matrix(msg)

        self.assertIsNotNone(transform)
        np.testing.assert_allclose(transform[:3, 3], [4.0, 5.0, 6.0])


if __name__ == "__main__":
    unittest.main()
