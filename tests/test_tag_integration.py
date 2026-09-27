#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TagLocalizer 与 AsparagusAnalyzer 集成测试
- 测试三级标定降级链的正确行为
- 验证 AprilTag 在线定位 → 历史缓存 → 手工标定 → 未标定防撞 的完整降级路径
"""

import unittest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.vision.asparagus_analyzer import AsparagusAnalyzer, AsparagusTarget


class MockTagLocalizer:
    """模拟 TagLocalizer，可控制返回成功/失败"""

    def __init__(self, succeed: bool = True, transform: np.ndarray = None):
        self.succeed = succeed
        self.transform = transform if transform is not None else np.array([
            [1.0, 0.0,  0.0, 100.0],
            [0.0, 1.0,  0.0,  50.0],
            [0.0, 0.0, -1.0, 640.0],
            [0.0, 0.0,  0.0,   1.0]
        ])
        self.call_count = 0

    def localize_camera(self, image):
        self.call_count += 1
        if self.succeed:
            info = {
                "detected_tag_ids": [1, 2, 3],
                "static_tags_count": 3,
                "reprojection_error_px": 0.35,
                "inlier_count": 12,
                "tag0_dynamic_yaw_deg": None
            }
            return True, self.transform, info
        else:
            info = {
                "detected_tag_ids": [],
                "static_tags_count": 0,
                "reprojection_error_px": 0.0,
                "inlier_count": 0,
                "tag0_dynamic_yaw_deg": None
            }
            return False, None, info


def make_test_scene(analyzer):
    """构造包含 1 根芦笋的测试场景 (深色底板 640mm，中央绿色芦笋在 580mm，凸起 60mm)"""
    h, w = 480, 640
    analyzer.update_intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    color = np.full((h, w, 3), 40, dtype=np.uint8)  # 深灰传送带底板
    color[200:230, 240:480] = [30, 180, 40]        # 芦笋黄绿色
    depth = np.full((h, w), 640, dtype=np.uint16)  # 底板 640mm
    depth[200:230, 240:480] = 580                  # 中央放置一根芦笋 (580mm，凸起 60mm)
    return color, depth


class TestTagIntegration(unittest.TestCase):
    """测试 AprilTag 三级标定降级链"""

    def test_tag_online_calibration(self):
        """场景 1：有 TagLocalizer + 标靶可见 → 使用 tag_online 外参"""
        analyzer = AsparagusAnalyzer()
        mock_localizer = MockTagLocalizer(succeed=True)
        analyzer.set_tag_localizer(mock_localizer)

        color, depth = make_test_scene(analyzer)
        targets = analyzer.analyze(color, depth)

        self.assertGreater(len(targets), 0, "应当检测到至少 1 根芦笋")
        top = targets[0]
        self.assertEqual(top.calibration_source, "tag_online")
        self.assertTrue(mock_localizer.call_count >= 1, "TagLocalizer 应当被调用")

        # 验证坐标经过了标定矩阵变换（Z 应翻转）
        # 原始 grip_z ~ 580mm, 经过 [0,0,-1,640] 变换后 robot_z ≈ 640 - 580 = 60mm
        self.assertAlmostEqual(top.robot_z, 640.0 - top.grip_z, delta=5.0)

    def test_tag_cached_fallback(self):
        """场景 2：有 TagLocalizer + 第一帧成功 + 第二帧失败 → 使用 tag_cached"""
        analyzer = AsparagusAnalyzer()
        mock_localizer = MockTagLocalizer(succeed=True)
        analyzer.set_tag_localizer(mock_localizer)

        color, depth = make_test_scene(analyzer)

        # 第一帧：标靶可见
        targets1 = analyzer.analyze(color, depth)
        self.assertEqual(targets1[0].calibration_source, "tag_online")

        # 第二帧：标靶被遮挡
        mock_localizer.succeed = False
        targets2 = analyzer.analyze(color, depth)
        self.assertGreater(len(targets2), 0)
        self.assertEqual(targets2[0].calibration_source, "tag_cached")

        # 验证缓存外参仍然产生正确的坐标变换
        self.assertAlmostEqual(targets2[0].robot_z, 640.0 - targets2[0].grip_z, delta=5.0)

    def test_hand_eye_fallback(self):
        """场景 3：无 TagLocalizer + 有手工矩阵 → 使用 hand_eye"""
        analyzer = AsparagusAnalyzer()
        t_calib = np.array([
            [1.0, 0.0,  0.0, 200.0],
            [0.0, 1.0,  0.0, 100.0],
            [0.0, 0.0, -1.0, 640.0],
            [0.0, 0.0,  0.0,   1.0]
        ])
        analyzer.set_hand_eye_matrix(t_calib)

        color, depth = make_test_scene(analyzer)
        targets = analyzer.analyze(color, depth)

        self.assertGreater(len(targets), 0)
        self.assertEqual(targets[0].calibration_source, "hand_eye")

    def test_uncalibrated_safety(self):
        """场景 4：完全无标定 → 防撞保护模式"""
        analyzer = AsparagusAnalyzer()

        color, depth = make_test_scene(analyzer)
        targets = analyzer.analyze(color, depth)

        self.assertGreater(len(targets), 0)
        top = targets[0]
        self.assertEqual(top.calibration_source, "uncalibrated")

        # 验证安全机制：robot_z 应为相对凸起高度 (~60mm)，绝非相机深度 (~580mm)
        self.assertLess(top.robot_z, 100.0, "未标定时 Z 应为凸起高度 (<100mm)，不能是相机深度")
        self.assertGreater(top.robot_z, 10.0, "凸起高度应大于 10mm")

    def test_tag_online_overrides_hand_eye(self):
        """验证 AprilTag 在线定位优先于手工标定"""
        analyzer = AsparagusAnalyzer()

        # 设置手工标定矩阵 (平移 200, 100)
        t_hand = np.array([
            [1.0, 0.0,  0.0, 200.0],
            [0.0, 1.0,  0.0, 100.0],
            [0.0, 0.0, -1.0, 640.0],
            [0.0, 0.0,  0.0,   1.0]
        ])
        analyzer.set_hand_eye_matrix(t_hand)

        # 同时设置 TagLocalizer (不同的平移 300, 150)
        tag_transform = np.array([
            [1.0, 0.0,  0.0, 300.0],
            [0.0, 1.0,  0.0, 150.0],
            [0.0, 0.0, -1.0, 640.0],
            [0.0, 0.0,  0.0,   1.0]
        ])
        mock_localizer = MockTagLocalizer(succeed=True, transform=tag_transform)
        analyzer.set_tag_localizer(mock_localizer)

        color, depth = make_test_scene(analyzer)
        targets = analyzer.analyze(color, depth)

        self.assertGreater(len(targets), 0)
        top = targets[0]
        # 应使用 tag_online（平移 300），而非 hand_eye（平移 200）
        self.assertEqual(top.calibration_source, "tag_online")

    def test_gcode_reflects_calibration_source(self):
        """验证 G-code 输出正确反映各标定来源"""
        base = AsparagusTarget(
            id=1, center_px=(320.0, 240.0), length_px=200.0, diam_px=20.0,
            yaw_deg=15.0, axis_vector=(1.0, 0.0),
            box_corners=np.zeros((4, 2), dtype=np.int32),
            contour=np.zeros((1, 1, 2), dtype=np.int32),
            length_mm=180.0, diam_mm=12.0,
            grip_x=10.0, grip_y=-5.0, grip_z=580.0,
            z_top=575.0, rel_height_mm=60.0,
            robot_x=310.0, robot_y=145.0, robot_z=60.0, robot_r=15.0,
            is_topmost=True
        )

        from src.control.scara_motion_planner import ScaraMotionPlanner
        planner = ScaraMotionPlanner()

        # uncalibrated
        base.calibration_source = "uncalibrated"
        gcode = planner.generate_pick_gcode(base)
        self.assertIn("UNCALIBRATED", gcode)
        self.assertIn("安全警告", gcode)

        # tag_online
        base.calibration_source = "tag_online"
        gcode = planner.generate_pick_gcode(base)
        self.assertIn("AprilTag 在线", gcode)

        # tag_cached
        base.calibration_source = "tag_cached"
        gcode = planner.generate_pick_gcode(base)
        self.assertIn("历史缓存", gcode)

        # hand_eye
        base.calibration_source = "hand_eye"
        gcode = planner.generate_pick_gcode(base)
        self.assertIn("手工标定矩阵", gcode)


if __name__ == "__main__":
    unittest.main()
