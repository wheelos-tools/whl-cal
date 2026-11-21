from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from lidar2lidar.semantic_registration import (
    LidarFrameData,
    PrimitiveExtractionConfig,
    ReflectivityConfig,
    VisibilityConfig,
    extract_geometric_primitives,
    normalize_reflectivity,
    ray_intersects_aabb_mask,
    register_semantic_l2l,
)


class SemanticRegistrationTests(unittest.TestCase):
    def test_reflectivity_normalization_invariant_to_sensor_gain(self) -> None:
        rng = np.random.default_rng(11)
        n = 400
        pts = np.column_stack(
            [
                rng.uniform(4.0, 28.0, size=n),
                rng.uniform(-10.0, 10.0, size=n),
                rng.uniform(-0.5, 2.5, size=n),
            ]
        )
        r = np.linalg.norm(pts, axis=1)
        base_refl = np.full(n, 20.0)
        base_refl[:35] = 220.0  # Retro-reflective lane markings / sign
        raw_a = base_refl / np.power(r, 1.5)
        raw_b = 3.5 * raw_a  # 3.5x hardware intensity gain difference

        za, _ = normalize_reflectivity(pts, raw_a, config=ReflectivityConfig())
        zb, _ = normalize_reflectivity(pts, raw_b, config=ReflectivityConfig())
        self.assertGreater(float(np.min(za[:35])), 2.0)
        self.assertLess(float(np.max(np.abs(za[:35] - zb[:35]))), 0.25)

    def test_ego_aabb_ray_occlusion_and_corridor_degeneracy(self) -> None:
        # Rays passing through [1..3] x [-1..1] x [-0.5..1.0] vs clear rays
        bmin = np.array([1.0, -1.0, -0.5])
        bmax = np.array([3.0, 1.0, 1.0])
        pts_test = np.array(
            [
                [6.0, 0.0, 0.2],  # blocked by [1..3]
                [0.5, 0.0, 0.2],  # in front of box -> visible
                [2.0, 5.0, 0.2],  # passes at x=2 when y=5 -> misses box!
                [2.0, 0.0, 4.0],  # passes at x=2 when z=4 -> misses box!
            ],
            dtype=np.float64,
        )
        occ = ray_intersects_aabb_mask(pts_test, bmin, bmax)
        np.testing.assert_array_equal(occ, np.array([True, False, False, False]))

        # Corridor-only scene (parallel walls x = +-6, no poles, no markings)
        gy, gz = np.meshgrid(np.linspace(-14.0, 14.0, 90), np.linspace(0.0, 2.5, 14))
        wall_l = np.column_stack([np.full(gy.size, -6.0), gy.ravel(), gz.ravel()])
        wall_r = np.column_stack([np.full(gy.size, 6.0), gy.ravel(), gz.ravel()])
        gx, gy2 = np.meshgrid(np.linspace(-5.5, 5.5, 24), np.linspace(-14.0, 14.0, 48))
        gnd = np.column_stack([gx.ravel(), gy2.ravel(), np.full(gx.size, -0.5)])
        corridor_pts = np.vstack([gnd, wall_l, wall_r])
        res = register_semantic_l2l(
            LidarFrameData(points=corridor_pts),
            LidarFrameData(points=corridor_pts),
            np.eye(4),
        )
        self.assertTrue(res["success"])
        self.assertTrue(res["localizability"]["corridor_degenerate"])
        weak_dir = np.abs(res["localizability"]["weakest_translation_direction_target"])
        # Weakest direction must align with corridor Y axis!
        self.assertGreater(float(weak_dir[1]), 0.90)

    def test_semantic_l2l_recovers_6dof_and_rejects_clutter(
        self,
    ) -> None:
        rng = np.random.default_rng(23)
        # Ground + two orthogonal walls + vertical pole + retro stripe + clutter
        gx, gy = np.meshgrid(np.linspace(-12.0, 12.0, 28), np.linspace(-12.0, 12.0, 28))
        gnd = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, -0.5)])
        gnd_i = np.full(len(gnd), 25.0)
        # Retro-reflective crosswalk dashes on ground along x = 2.0, y in [-3, 3]
        rx = np.linspace(1.7, 2.3, 12)
        ry = np.linspace(-3.0, 3.0, 24)
        rxx, ryy = np.meshgrid(rx, ry)
        retro_pts = np.column_stack([rxx.ravel(), ryy.ravel(), np.full(rxx.size, -0.5)])
        retro_i = np.full(len(retro_pts), 240.0)

        wy, wz = np.meshgrid(np.linspace(-10.0, 10.0, 88), np.linspace(0.0, 2.8, 16))
        wall_x = np.column_stack([np.full(wy.size, -8.5), wy.ravel(), wz.ravel()])
        wx, wz2 = np.meshgrid(np.linspace(-8.0, 10.0, 80), np.linspace(0.0, 2.8, 16))
        wall_y = np.column_stack([wx.ravel(), np.full(wx.size, 9.5), wz2.ravel()])

        pz = np.linspace(0.0, 3.2, 48)
        ang = np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)
        pole1 = np.vstack(
            [
                np.column_stack(
                    [5.0 + 0.08 * np.cos(ang), -4.0 + 0.08 * np.sin(ang), np.full(8, z)]
                )
                for z in pz
            ]
        )
        # High-sphericity foliage clutter
        clutter = rng.uniform([-5.0, -5.0, 0.3], [-3.0, -3.0, 2.0], size=(250, 3))

        pts_a = np.vstack([gnd, retro_pts, wall_x, wall_y, pole1, clutter])
        int_a = np.concatenate(
            [
                gnd_i,
                retro_i,
                np.full(len(wall_x) + len(wall_y) + len(pole1) + len(clutter), 30.0),
            ]
        )

        # True transform mapping B -> A: 175 deg yaw + roll/pitch + [-5.2, 2.8, 0.25]
        T_true = np.eye(4, dtype=np.float64)
        T_true[:3, :3] = Rot.from_euler(
            "xyz", [2.0, -1.8, 175.0], degrees=True
        ).as_matrix()
        T_true[:3, 3] = np.array([-5.2, 2.8, 0.25])
        T_inv = np.linalg.inv(T_true)

        pts_b = (pts_a @ T_inv[:3, :3].T + T_inv[:3, 3]) + rng.normal(
            0.0, 0.004, size=pts_a.shape
        )
        # Simulate different LiDAR gain on B
        int_b = int_a * 2.2

        # Perturb initial guess by 5 deg yaw and 0.65 m translation
        T_pert = np.eye(4, dtype=np.float64)
        T_pert[:3, :3] = Rot.from_euler(
            "xyz", [0.0, 0.0, 5.0], degrees=True
        ).as_matrix()
        T_pert[:3, 3] = np.array([0.45, -0.45, 0.12])
        T_init = T_true @ T_pert

        prim_a = extract_geometric_primitives(
            LidarFrameData(points=pts_a, intensity=int_a),
            prim_cfg=PrimitiveExtractionConfig(),
        )
        self.assertGreater(prim_a.stats["facade"], 20)
        self.assertGreater(prim_a.stats["pole"], 2)
        self.assertGreater(prim_a.stats["intensity"], 15)
        self.assertGreater(prim_a.stats["scatter_rejected_voxels"], 2)

        res = register_semantic_l2l(
            LidarFrameData(points=pts_b, intensity=int_b),
            LidarFrameData(points=pts_a, intensity=int_a),
            T_init,
            vis_cfg=VisibilityConfig(reciprocal_check=True),
        )
        self.assertTrue(res["success"])
        T_est = res["transformation"]
        dR = np.linalg.inv(T_true[:3, :3]) @ T_est[:3, :3]
        rot_err_deg = math.degrees(Rot.from_matrix(dR).magnitude())
        trans_err_m = float(np.linalg.norm(T_true[:3, 3] - T_est[:3, 3]))
        self.assertLess(rot_err_deg, 0.25)
        self.assertLess(trans_err_m, 0.04)
        self.assertFalse(res["localizability"]["corridor_degenerate"])

        # Large seed perturbation (+16 deg yaw, 1.5 m shift, 2 deg tilt) must be
        # recovered cleanly via the 4-DoF BEV FFT coarse fallback!
        T_big_pert = np.eye(4, dtype=np.float64)
        T_big_pert[:3, :3] = Rot.from_euler(
            "xyz", [2.0, -1.8, 16.0], degrees=True
        ).as_matrix()
        T_big_pert[:3, 3] = np.array([1.1, -1.0, 0.25])
        res_coarse = register_semantic_l2l(
            LidarFrameData(points=pts_b, intensity=int_b),
            LidarFrameData(points=pts_a, intensity=int_a),
            T_big_pert @ T_true,
            vis_cfg=VisibilityConfig(reciprocal_check=True),
        )
        self.assertTrue(res_coarse["success"])
        T_est_c = res_coarse["transformation"]
        rot_err_c = math.degrees(
            Rot.from_matrix(np.linalg.inv(T_true[:3, :3]) @ T_est_c[:3, :3]).magnitude()
        )
        trans_err_c = float(np.linalg.norm(T_true[:3, 3] - T_est_c[:3, 3]))
        self.assertLess(rot_err_c, 0.30)
        self.assertLess(trans_err_c, 0.05)


if __name__ == "__main__":
    unittest.main()
