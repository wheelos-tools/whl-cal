import unittest

import numpy as np
import open3d as o3d

from lidar2lidar.fov import AngularSupport, estimate_angular_support, shared_fov_clouds


class AngularSupportTest(unittest.TestCase):
    @staticmethod
    def _points_at_angles(angles_deg):
        radians = np.radians(np.asarray(angles_deg, dtype=float))
        return np.column_stack(
            (np.cos(radians), np.sin(radians), np.zeros_like(radians))
        )

    def test_estimates_wrapped_sector(self):
        points = self._points_at_angles([170.0, 175.0, -179.0, -175.0, -170.0])

        support = estimate_angular_support(
            [points],
            coverage_ratio=1.0,
            margin_deg=0.0,
        )

        self.assertLessEqual(support.span_deg, 20.0)
        self.assertTrue(support.contains(np.array([179.0, -179.0])).all())
        self.assertFalse(support.contains(np.array([0.0])))

    def test_estimates_front_sector(self):
        points = self._points_at_angles(np.linspace(-60.0, 60.0, 25))

        support = estimate_angular_support(
            [points],
            coverage_ratio=1.0,
            margin_deg=0.0,
        )

        self.assertAlmostEqual(support.span_deg, 120.0)
        self.assertTrue(support.contains(np.array([-60.0, 0.0, 60.0])).all())
        self.assertFalse(support.contains(np.array([100.0])))

    def test_rejects_invalid_configuration_and_empty_points(self):
        with self.assertRaises(ValueError):
            estimate_angular_support([], coverage_ratio=0.0)
        with self.assertRaises(ValueError):
            estimate_angular_support([np.empty((0, 3))])

    def test_crops_each_sensor_in_the_other_sensor_frame(self):
        source_points = self._points_at_angles([-90.0, 0.0, 90.0, 180.0])
        target_points = self._points_at_angles([0.0, 90.0, 180.0, 270.0])
        source_cloud = o3d.geometry.PointCloud()
        source_cloud.points = o3d.utility.Vector3dVector(source_points)
        target_cloud = o3d.geometry.PointCloud()
        target_cloud.points = o3d.utility.Vector3dVector(target_points)
        transform = np.eye(4)
        transform[:3, :3] = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        support = AngularSupport(
            start_deg=0.0,
            span_deg=90.0,
            coverage_ratio=1.0,
            point_count=4,
            margin_deg=0.0,
        )

        cropped_source, cropped_target, metrics = shared_fov_clouds(
            source_cloud,
            target_cloud,
            transform,
            support,
            support,
        )

        self.assertEqual(len(cropped_source.points), 2)
        self.assertEqual(len(cropped_target.points), 2)
        self.assertEqual(metrics["source_retained_ratio"], 0.5)
        self.assertEqual(metrics["target_retained_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
