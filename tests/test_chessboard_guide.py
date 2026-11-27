# -*- coding: utf-8 -*-
"""3D 透视引导式棋盘格几何投影与对准评估单元测试"""

import unittest

from src.calibration.solvers.chessboard_guide import (
    default_guide_poses,
    evaluate_alignment,
    project_guide_mesh,
)


class TestChessboardGuide(unittest.TestCase):

    def test_24_poses_projection_within_viewport_bounds(self):
        """验证 24 个标准姿态在 16:9 横屏与 4:3 下均完整落于安全视口内"""
        poses = default_guide_poses()
        self.assertEqual(len(poses), 24)
        for aspect in (16.0 / 9.0, 4.0 / 3.0, 3.0 / 4.0):
            for pose in poses:
                mesh = project_guide_mesh(pose, pattern_size=(9, 6), viewport_aspect=aspect)
                self.assertEqual(len(mesh.outer_quad), 4)
                self.assertEqual(len(mesh.corner_depths), 4)
                self.assertGreater(len(mesh.grid_lines), 0)
                for nx, ny in mesh.outer_quad:
                    self.assertGreaterEqual(nx, 0.02, f"pose {pose.index} x out of bounds: {nx}")
                    self.assertLessEqual(nx, 0.98, f"pose {pose.index} x out of bounds: {nx}")
                    self.assertGreaterEqual(ny, 0.02, f"pose {pose.index} y out of bounds: {ny}")
                    self.assertLessEqual(ny, 0.98, f"pose {pose.index} y out of bounds: {ny}")

    def test_evaluate_alignment_and_invariance(self):
        """验证完全对准、180° 翻转不变性、偏移方向提示、尺度提示及防抖稳定判定"""
        pose = default_guide_poses()[0]
        mesh = project_guide_mesh(pose, pattern_size=(9, 6), viewport_aspect=16.0 / 9.0)
        tgt = mesh.outer_quad

        # 1. 完全对准 + 静止
        fb_ok = evaluate_alignment(tgt, tgt, prev_quad_norm=tgt)
        self.assertTrue(fb_ok.detected)
        self.assertTrue(fb_ok.aligned)
        self.assertTrue(fb_ok.stable)
        self.assertGreater(fb_ok.match_score, 0.99)

        # 2. 180° 反转 + 循环移位不变性
        reversed_shifted = list(reversed(tgt))[2:] + list(reversed(tgt))[:2]
        fb_inv = evaluate_alignment(reversed_shifted, tgt, prev_quad_norm=reversed_shifted)
        self.assertTrue(fb_inv.aligned)
        self.assertTrue(fb_inv.stable)

        # 3. 向左偏移 -> 提示向右移动
        shifted_left = [(x - 0.15, y) for x, y in tgt]
        fb_left = evaluate_alignment(shifted_left, tgt)
        self.assertFalse(fb_left.aligned)
        self.assertIn("向右移动", fb_left.guidance_text)

        # 4. 尺寸过小 -> 提示靠近
        cx = sum(p[0] for p in tgt) / 4.0
        cy = sum(p[1] for p in tgt) / 4.0
        shrunk = [(cx + (x - cx) * 0.5, cy + (y - cy) * 0.5) for x, y in tgt]
        fb_small = evaluate_alignment(shrunk, tgt)
        self.assertFalse(fb_small.aligned)
        self.assertIn("靠近", fb_small.guidance_text)

        # 5. 对准但上一帧位移过大 -> 防抖拦截 (stable=False)
        prev_jitter = [(x + 0.04, y + 0.04) for x, y in tgt]
        fb_shaky = evaluate_alignment(tgt, tgt, prev_quad_norm=prev_jitter)
        self.assertTrue(fb_shaky.aligned)
        self.assertFalse(fb_shaky.stable)
        self.assertIn("晃动", fb_shaky.guidance_text)


if __name__ == "__main__":
    unittest.main()
