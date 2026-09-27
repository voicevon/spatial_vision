"""
手眼矩阵兜底层级与 SCARA 防撞 G-code 自动化测试套件
====================================================
覆盖核心安全与精度规范：
  1. 未标定工况下相机 500+mm 镜头深度硬拦截与凸起净高防撞测试
  2. config.yaml 手工矩阵 (hand_eye 兜底层级) 的齐次变换与机械臂世界坐标解算测试
注: 原示教式接触手眼标定向导 (hand_eye_calibration.py) 已随技术路线收口移除，
    hand_eye 层级仅消费 config.yaml 中的 T_cam_to_scara 矩阵。
"""

import os
import sys
import unittest
import numpy as np

# 导入工程模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.vision.asparagus_analyzer import AsparagusAnalyzer, AsparagusTarget


class TestHandEyeMatrixAndSafety(unittest.TestCase):

    def test_uncalibrated_gcode_safety_interception(self):
        """
        [生死线防撞测试]
        验证当手眼矩阵未标定时，系统坚决拦截相机 500+mm 镜头深度，
        绝不能在下探指令中出现 Z500+，且必须输出安全警告。
        """
        camera_depth = 536.8  # 相机镜头绝对深度 (若直接使用必撞机毁机)
        rel_height = 28.5     # 芦笋相对工作台的凸起净高度 (安全下探高度)

        target = AsparagusTarget(
            id=1,
            center_px=(640.0, 360.0),
            length_px=450.0,
            diam_px=30.0,
            yaw_deg=15.0,
            axis_vector=(0.965, 0.258),
            box_corners=np.zeros((4, 2), dtype=np.int32),
            contour=np.zeros((10, 1, 2), dtype=np.int32),
            length_mm=210.0,
            diam_mm=16.5,
            grip_x=12.0,
            grip_y=-5.0,
            grip_z=camera_depth,
            z_top=camera_depth - 10.0,
            rel_height_mm=rel_height,
            robot_x=12.0,
            robot_y=-5.0,
            robot_z=rel_height,       # 未标定时强制采用相对台面凸起
            robot_r=15.0,
            is_topmost=True,
            calibration_source="uncalibrated"
        )

        from src.control.scara_motion_planner import ScaraMotionPlanner
        planner = ScaraMotionPlanner(safe_z=80.0)
        gcode = planner.generate_pick_gcode(target, drop_x=220.0, drop_y=0.0, slot_index=0)

        # 1. 验证包含安全未标定警告
        self.assertIn("UNCALIBRATED", gcode)
        self.assertIn("安全警告", gcode)

        # 2. 验证下探指令采用的是相对高度 (28.5mm)，绝不是 536.8mm
        self.assertIn(f"G1 Z{rel_height:.2f}", gcode)
        self.assertNotIn("536.8", gcode)
        self.assertNotIn("Z536", gcode)

    def test_calibrated_robot_transformation(self):
        """测试 hand_eye 兜底层级: 齐次矩阵乘法正确将相机系坐标映射至机器人基座系"""
        from src.control.scara_motion_planner import ScaraMotionPlanner
        analyzer = AsparagusAnalyzer()

        # 构造平移与翻转矩阵 (例如相机沿 X 平移 200mm, Y 平移 100mm, Z 方向为 640 - Z_cam)
        t_calib = np.array([
            [1.0, 0.0,  0.0, 200.0],
            [0.0, 1.0,  0.0, 100.0],
            [0.0, 0.0, -1.0, 640.0],
            [0.0, 0.0,  0.0,   1.0]
        ])
        analyzer.set_hand_eye_matrix(t_calib)
        self.assertTrue(analyzer.is_hand_eye_calibrated)

        # 构造模拟深度图与图像 (1 根芦笋)
        h, w = 480, 640
        analyzer.update_intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        color = np.full((h, w, 3), 40, dtype=np.uint8)
        color[200:230, 240:480] = [30, 180, 40]
        depth = np.full((h, w), 640, dtype=np.uint16)  # 底板 640mm
        depth[200:230, 240:480] = 580                  # 芦笋 580mm

        targets = analyzer.analyze(color, depth)
        self.assertGreater(len(targets), 0)
        top = targets[0]

        self.assertEqual(top.calibration_source, "hand_eye")
        self.assertAlmostEqual(top.robot_z, 640.0 - top.grip_z, delta=5.0)
        planner = ScaraMotionPlanner()
        gcode = planner.generate_pick_gcode(top)
        self.assertIn("手工标定矩阵", gcode)


if __name__ == "__main__":
    unittest.main()
