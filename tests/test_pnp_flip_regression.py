#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""回归测试: 单靶 PnP 平面二义性 180° 翻转的先验纠偏 (PnpSolver)
合成 45° 附近斜视标靶 + 像素噪声, 验证传入 expected_z_cam 后必然选中与先验同向的解。
(源自 temp/test_pnp_flip_fix.py, P2 收编为标准单测)"""
import unittest

import cv2
import numpy as np

from src.vision.pnp_solver import PnpSolver


def make_obs(rng, solver, theta_eff_deg, az_deg, dist=420.0, noise_px=0.5, yaw_deg=0.0, obj=None):
    """构造斜视观测: theta_eff = 标靶法向与相机光轴的夹角 (有效视角, 保证非掠射)"""
    th, ph, yaw = np.radians([theta_eff_deg, az_deg, yaw_deg])
    n = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    axis = np.cross([0.0, 0.0, 1.0], n)
    s = float(np.linalg.norm(axis))
    ang = float(np.arccos(np.clip(n[2], -1.0, 1.0)))
    R_align = np.eye(3) if s < 1e-9 else cv2.Rodrigues(axis / s * ang)[0]
    R_t_c = R_align @ cv2.Rodrigues(np.array([0.0, 0.0, yaw]))[0]  # 含靶面内自转
    t_t_c = np.array([8.0, -5.0, dist])   # 标靶中心在相机系
    pts = solver.obj_points if obj is None else obj
    proj, _ = cv2.projectPoints(pts, cv2.Rodrigues(R_t_c)[0], t_t_c,
                                solver.camera_matrix, solver.dist_coeffs)
    c = proj.reshape(4, 2) + rng.normal(0, noise_px, (4, 2))
    return c, R_t_c, t_t_c


class TestPnPPriorDisambiguation(unittest.TestCase):

    """expected_z_cam 先验应在全参数域消除 180° 翻转"""

    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.default_rng(42)
        cls.solver = PnpSolver(
            camera_matrix=np.array([[615.0, 0, 320], [0, 615.0, 240], [0, 0, 1.0]]),
            dist_coeffs=np.zeros((5, 1)),
            marker_size_mm=50.0
        )

    def test_prior_eliminates_flip(self):
        n = 300
        fails_prior = 0
        for dist in (500.0, 800.0, 1200.0):
            for noise in (0.3, 0.6, 1.0):
                for size_err in (1.0, 1.02):
                    s_true = 25.0 * size_err
                    obj_true = np.array(
                        [[-s_true, s_true, 0], [s_true, s_true, 0],
                         [s_true, -s_true, 0], [-s_true, -s_true, 0]], dtype=np.float64)
                    for _ in range(n):
                        tilt = self.rng.uniform(30, 70)  # 有效视角 30-70°, 45° 附近为二义性高发区
                        az = self.rng.uniform(0, 360)
                        yaw = self.rng.uniform(0, 360)
                        c, R_true, _ = make_obs(
                            self.rng, self.solver, tilt, az, dist=dist,
                            noise_px=noise, yaw_deg=yaw, obj=obj_true)
                        z_true_cam = R_true[:, 2]  # 真值法向 (相机系)
                        ok1, r1, _ = self.solver.solve_single_tag_pnp(
                            c, expected_z_cam=z_true_cam)  # 带先验
                        if ok1 and float(cv2.Rodrigues(r1)[0][:, 2] @ z_true_cam) < 0:
                            fails_prior += 1
        self.assertEqual(fails_prior, 0, "expected_z_cam 先验纠偏失效, 出现 180° 翻转!")
        self.assertEqual(fails_prior, 0, "expected_z_cam 先验纠偏失效, 出现 180° 翻转!")


if __name__ == "__main__":
    unittest.main()
