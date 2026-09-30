#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""回归测试: _tag_local_frame 法向方向 — 世界角点重构坐标系应与单靶 PnP 位姿同向
(修复绿/蓝棱柱反向问题的守护测试; 源自 temp/test_prism_normal.py, P2 收编为标准单测)"""
import unittest

import cv2
import numpy as np

from tools.tracker.common import _tag_local_frame
from src.vision.pnp_solver import PnpSolver


class TestTagLocalFrameNormal(unittest.TestCase):

    """重构法向应与 PnP 法向同向, 且平移/顶面中心几何一致"""

    @classmethod
    def setUpClass(cls):
        # 任意外参: 相机看着一块斜置标靶
        cls.K = np.array([[1363.68, 0, 971.19], [0, 1361.19, 566.26], [0, 0, 1.0]])
        dist = np.zeros((5, 1))
        cls.r_true, cls.t_true = np.array([[0.3], [0.0], [0.0]]), np.array([[10.0], [-20.0], [600.0]])
        cls.pnp_solver = PnpSolver(
            tags_map={}, camera_matrix=cls.K, dist_coeffs=dist, marker_size_mm=40.0)

    def test_normal_and_translation_consistency(self):
        proj, _ = cv2.projectPoints(self.pnp_solver.obj_points, self.r_true, self.t_true,
                                    self.K, np.zeros((5, 1)))
        corners_img = proj.reshape(4, 2)

        # 1) 蓝色路径: 单靶 PnP 位姿
        ok, r_pnp, t_pnp = self.pnp_solver.solve_single_tag_pnp(corners_img)
        self.assertTrue(ok, "PnP 失败")

        # 2) 绿色路径: 构造世界角点 (真值位姿) -> _tag_local_frame -> 合成相机系位姿
        T_w_t = np.eye(4)
        R_t, _ = cv2.Rodrigues(self.r_true)
        T_w_t[:3, :3], T_w_t[:3, 3] = R_t.reshape(3, 3), self.t_true.reshape(3)
        s = self.pnp_solver.marker_size_mm / 2.0
        local = np.array([[-s, s, 0, 1], [s, s, 0, 1], [s, -s, 0, 1], [-s, -s, 0, 1]],
                         dtype=np.float64)
        wc = (T_w_t @ local.T).T[:, :3]  # 世界角点 (4,3), 顺序同 get_tag_world_corners

        R_frame, c_w = _tag_local_frame(wc)
        # 法向一致性: 重构 Z 轴 与 PnP 旋转第三列 点积应为 +1 (同向)
        R_pnp_m, _ = cv2.Rodrigues(r_pnp)
        dot = float(R_frame[:, 2] @ R_pnp_m[:, 2])
        self.assertGreater(dot, 0.99, "重构法向与 PnP 法向反向!")

        # 合成相机系位姿 vs 直接 PnP 位姿: 平移应一致 (mm 级)
        t_ct = np.eye(3) @ c_w
        self.assertTrue(np.allclose(t_ct, np.asarray(t_pnp).reshape(3), atol=0.5), "平移不一致")

        # 棱柱顶面中心应沿 +Z 朝外 (远离安装面): 顶面中心世界坐标 = c + 75*法向
        top_center_w = c_w + R_frame[:, 2] * 75.0
        self.assertTrue(np.allclose(
            top_center_w - c_w, R_t.reshape(3, 3) @ np.array([0, 0, 75.0]), atol=1e-6),
            "顶面中心未沿面外 +75mm")


if __name__ == "__main__":
    unittest.main()
