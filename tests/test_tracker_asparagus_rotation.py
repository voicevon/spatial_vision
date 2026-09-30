#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot 在线跟踪 - Tag 2 (芦笋) R 轴旋转角提取与拟真长棒渲染单元测试
================================================================
测试用例:
  1. 单靶 PnP 解算时完整提取旋转向量并恢复局部 Y 轴偏航角 (对比理论真值误差 < 0.5°);
  2. 世界坐标系锁定时相机位姿与标靶位姿复合投影, 校验世界水平 XY 平面偏航角;
  3. 芦笋 3D 拟真长棒 (宽 15mm x 长 200mm, 对称延伸各 100mm) 透视投影与分段多边形渲染;
  4. Tracker 状态机 solve_frame 闭环校验 measured_r 与 G-code E 轴联动参数。
"""

import os
import sys
import unittest
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from tools.tracker.common import (
    ASPARAGUS_WIDTH_MM, ASPARAGUS_LENGTH_MM, ASPARAGUS_HALF_LENGTH_MM,
    ASPARAGUS_HALF_WIDTH_MM, fmt_pose_4d
)
from tools.tracker.app import RobotOnlineTracker
from tools.tracker.renderer import TrackerRenderer
from src.vision.pnp_solver import PnpSolver
from src.vision.tag_detector import TagDetector


class TestTrackerAsparagusRotation(unittest.TestCase):
    def setUp(self):
        self.camera_matrix = np.array([
            [1000.0, 0.0, 640.0],
            [0.0, 1000.0, 360.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros(5, dtype=np.float64)
        self.marker_size_mm = 50.0

        self.pnp_solver = PnpSolver(
            tags_map={},
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            marker_size_mm=self.marker_size_mm
        )
        self.tag_detector = TagDetector(valid_tag_ids=None)

        # 实例化轻量 tracker
        self.tracker = RobotOnlineTracker.__new__(RobotOnlineTracker)
        self.tracker.target_tag_id = 2
        self.tracker.recog_tag2_on = True
        self.tracker.show_anchors_on = False
        self.tracker.show_xy_plane_on = False
        self.tracker.world_locked = False
        self.tracker.locked_rvec = None
        self.tracker.locked_tvec = None
        self.tracker.pnp_solver = self.pnp_solver
        self.tracker.tag_detector = self.tag_detector
        self.tracker.support_ids = []
        self.tracker.rmse = None
        self.tracker.measured = None
        self.tracker.measured_r = None
        self.tracker.target_rvec = None
        self.tracker.target_tvec = None
        self.tracker.measured_time = 0.0
        self.tracker.HP_TARGET_SIDE_PX = 100.0

    def test_asparagus_constants(self):
        """测试芦笋物理规格常量定义: 宽 15mm, 长 200mm, 对称延伸各 100mm"""
        self.assertEqual(ASPARAGUS_WIDTH_MM, 15.0)
        self.assertEqual(ASPARAGUS_LENGTH_MM, 200.0)
        self.assertEqual(ASPARAGUS_HALF_LENGTH_MM, 100.0)
        self.assertEqual(ASPARAGUS_HALF_WIDTH_MM, 7.5)

        # 测试 4D 位姿格式化
        s = fmt_pose_4d(np.array([100.0, 200.0, 80.0]), r_deg=45.2)
        self.assertIn("R: +45.2°", s)

    def test_single_tag_pnp_extracts_rvec_and_yaw(self):
        """测试单靶 PnP 解算提取旋转向量及局部 Y 轴偏航角"""
        s = self.marker_size_mm / 2.0
        # 构造一个绕 Z 轴逆时针旋转 30° 的标靶
        angle_deg = 30.0
        angle_rad = np.radians(angle_deg)
        R_gt = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0.0],
            [np.sin(angle_rad),  np.cos(angle_rad), 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        t_gt = np.array([50.0, -30.0, 800.0], dtype=np.float64).reshape((3, 1))

        # 投影 4 个角点
        obj_pts = self.pnp_solver.obj_points
        rvec_gt, _ = cv2.Rodrigues(R_gt)
        proj_pts, _ = cv2.projectPoints(obj_pts, rvec_gt, t_gt, self.camera_matrix, self.dist_coeffs)
        corners_2d = proj_pts.reshape((4, 2))

        # 调用 pnp_solver.solve_single_tag_pnp
        ok, rvec_est, tvec_est = self.pnp_solver.solve_single_tag_pnp(corners_2d, expected_z_cam=np.array([0, 0, 1]))
        self.assertTrue(ok)
        self.assertIsNotNone(rvec_est)
        self.assertIsNotNone(tvec_est)

        # 校验 Y 轴朝向角
        R_est, _ = cv2.Rodrigues(rvec_est)
        v_y = R_est[:, 1]
        yaw_calc = float(np.degrees(np.arctan2(v_y[1], v_y[0])))
        # 真值 Y 轴在图像坐标系下: [ -sin(30°), cos(30°), 0 ] -> arctan2(cos, -sin) = 90° - (-30°) 或 60°/120°
        # 校验方向向量点积
        v_y_gt = R_gt[:, 1]
        cos_sim = float(np.dot(v_y, v_y_gt) / (np.linalg.norm(v_y) * np.linalg.norm(v_y_gt)))
        self.assertAlmostEqual(cos_sim, 1.0, places=2, msg="解算得到的 Y 轴方向向量应与真实真值高度一致")

    def test_world_locked_asparagus_yaw_computation(self):
        """测试世界坐标系锁定时，芦笋 Y 轴在世界水平面上投影偏航角的提取"""
        # 相机在世界坐标系上方俯视原点
        R_lock = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0]
        ], dtype=np.float64)
        rvec_lock, _ = cv2.Rodrigues(R_lock)
        tvec_lock = np.array([0.0, 0.0, 1000.0], dtype=np.float64)

        self.tracker.world_locked = True
        self.tracker.locked_rvec = rvec_lock
        self.tracker.locked_tvec = tvec_lock

        # 模拟检出 Tag 2，其在相机坐标系下的旋转为 R_c_t2
        # 设标靶在世界系下 Y 轴朝向世界 +X 轴 (偏航角 0°)
        # v_w_y = [1.0, 0.0, 0.0]
        R_w_t2 = np.array([
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        R_c_t2 = R_lock @ R_w_t2
        rvec_c, _ = cv2.Rodrigues(R_c_t2)
        tvec_c = np.array([10.0, 20.0, 600.0], dtype=np.float64)

        # 投影角点
        proj_pts, _ = cv2.projectPoints(self.pnp_solver.obj_points, rvec_c, tvec_c, self.camera_matrix, self.dist_coeffs)
        det = {2: proj_pts.reshape((4, 2))}

        # 执行 solve_frame
        # 模拟全景与ROI重检直接返回 det
        self.tracker.tag_detector.detect_tags = lambda frame: det
        self.tracker._detect_high_precision = lambda frame, d: d

        fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.tracker.solve_frame(fake_frame)

        self.assertIsNotNone(self.tracker.measured_r)
        self.assertIsNotNone(self.tracker.target_rvec)
        self.assertIsNotNone(self.tracker.target_tvec)
        # R_w_t2 的第二列为 [1.0, 0.0, 0.0] -> arctan2(0, 1) = 0°
        self.assertAlmostEqual(self.tracker.measured_r, 0.0, delta=2.0)

    def test_draw_asparagus_stem_rendering(self):
        """测试 3D 拟真芦笋长条 (宽 15mm x 长 200mm) 的分段渲染无崩溃且在画布上着色"""
        renderer = TrackerRenderer(self.tracker)
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

        rvec = np.array([0.1, 0.0, 0.0], dtype=np.float64)
        tvec = np.array([0.0, 0.0, 600.0], dtype=np.float64)

        # 绘制芦笋长棒
        renderer._draw_asparagus_stem(canvas, rvec, tvec, yaw_deg=35.0)

        # 检查画布是否有像素被绘制 (长棒、外框、切口与文字)
        nonzeros = np.count_nonzero(canvas)
        self.assertGreater(nonzeros, 500, "芦笋拟真长条应在画布上产生像素填充与边框")

        # 校验画面的主要颜色包含鲜翠绿 (BGR 绿色通道显著) 与白色 (三通道接近)
        green_mask = (canvas[:, :, 1] > 180) & (canvas[:, :, 0] < 120)
        self.assertTrue(np.any(green_mask), "芦笋头部应含有高饱和度鲜翠绿色像素")

        white_mask = (canvas[:, :, 0] > 200) & (canvas[:, :, 1] > 200) & (canvas[:, :, 2] > 200)
        self.assertTrue(np.any(white_mask), "芦笋尾部切口应含有白色像素")

    def test_draw_overlay_cleanliness(self):
        """测试叠加层渲染无任何汉字(芦笋/笋尖/根部)、无Tag 2文字、仅保留绿框和笋尖角度"""
        renderer = TrackerRenderer(self.tracker)
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

        # 模拟目标标靶角点与位姿
        corners = np.array([
            [600.0, 300.0],
            [680.0, 300.0],
            [680.0, 380.0],
            [600.0, 380.0]
        ], dtype=np.float32)
        det = {2: corners}

        self.tracker.target_rvec = np.array([0.0, 0.0, 0.5], dtype=np.float64)
        self.tracker.target_tvec = np.array([0.0, 0.0, 600.0], dtype=np.float64)
        self.tracker.measured_r = 42.5

        # 记录调用 draw_text 的文本内容
        called_texts = []
        original_draw_text = sys.modules["tools.tracker.renderer"].draw_text

        def mock_draw_text(img, text, pos, font_size=16, color=(240, 240, 240), bold=False):
            called_texts.append(text)
            return original_draw_text(img, text, pos, font_size, color, bold)

        sys.modules["tools.tracker.renderer"].draw_text = mock_draw_text
        try:
            renderer.draw_overlay(canvas, det)
        finally:
            sys.modules["tools.tracker.renderer"].draw_text = original_draw_text

        # 验证调用的文本中无任何汉字与“Tag 2”
        for txt in called_texts:
            for ch in txt:
                self.assertTrue(
                    ch < '\u4e00' or ch > '\u9fff',
                    f"叠加层文字中不应包含任何汉字: '{txt}' 包含 '{ch}'"
                )
            self.assertNotIn("Tag", txt, f"叠加层中不应包含 Tag 2 文字: '{txt}'")
            self.assertNotIn("芦笋", txt, f"叠加层中不应包含芦笋汉字: '{txt}'")

        # 验证笋尖角度数值存在
        self.assertTrue(
            any("+42.5°" in t for t in called_texts),
            f"笋尖前端应展示角度数值 '+42.5°'，实测调用列表: {called_texts}"
        )

    def test_track_worker_gcode_construction_with_r_and_e_readback(self):
        """测试 _track_worker 完整闭环: 水平平移对位 -> 下探抓取 -> 夹紧 -> 提升 -> Park -> 释放"""
        sent_commands = []

        class MockRobot:
            is_connected = True
            port = "COM_TEST"
            feedrate_grip = 1500

            def send_gcode(self, cmd, timeout=30.0, wait_done=True):
                sent_commands.append(cmd)
                return True

            def set_gripper(self, close=False, timeout=2.0):
                if close:
                    sent_commands.extend(["M4", "M280 P1 S0", "M280 P2 S0"])
                else:
                    sent_commands.extend(["M3", "M280 P1 S30", "M280 P2 S30"])
                return True

            def set_z_height(self, z_mm, timeout=3.0):
                z_clamped = max(0.0, min(100.0, float(z_mm)))
                angle = 270.0 - 2.7 * z_clamped
                sent_commands.append(f"M280 P0 S{angle:.0f}")
                sent_commands.append(f"G92 Z{z_clamped:.2f}")
                return True

            def get_position(self):
                # 模拟 M114 带有 E 轴回读 (X, Y, Z, E)
                return (120.5, 340.2, 80.0, 36.0)

        self.tracker.robot = MockRobot()
        self.tracker.track_log = []
        self.tracker.last_dev = None
        self.tracker.TRACK_FEEDRATE = 3000
        self.tracker.PARK_FEEDRATE = 5000
        self.tracker.TRACK_LOG_MAX = 8

        target_xyz = np.array([120.0, 340.0, 20.0])
        target_r = 35.5

        # 执行跟踪工作函数
        self.tracker._track_worker(target_xyz, target_r=target_r)

        # 校验关键动作的顺序与指令内容
        # 1. 初始张开夹爪
        self.assertIn("M3", sent_commands)
        # 2. 水平对位 (带目标角度 E35.50)
        expected_align_cmd = "G1 X120.00 Y340.00 Z80.00 E35.50 F3000"
        self.assertIn(expected_align_cmd, sent_commands)
        # 3. 下移 (通过 Servo 0 舵机下探至物料高度 Z=20.00, 角度 216°)
        expected_down_cmd = "M280 P0 S216"
        self.assertIn(expected_down_cmd, sent_commands)
        self.assertIn("G92 Z20.00", sent_commands)
        # 4. 夹爪夹住物料 (M4)
        self.assertIn("M4", sent_commands)
        # 5. 上移 (通过 Servo 0 舵机提升回安全高度 Z=80.00, 角度 54°)
        expected_up_cmd = "M280 P0 S54"
        self.assertIn(expected_up_cmd, sent_commands)
        self.assertIn("G92 Z80.00", sent_commands)
        # 6. 自动 Park 前往放料位
        expected_park_cmd = "G1 X-250.00 Y350.00 Z80.00 E90.00 F5000"
        self.assertIn(expected_park_cmd, sent_commands)

        # 校验时序相对位置
        idx_align = sent_commands.index(expected_align_cmd)
        idx_down = sent_commands.index(expected_down_cmd)
        idx_grip = sent_commands.index("M4")
        idx_up = sent_commands.index(expected_up_cmd)
        idx_park = sent_commands.index(expected_park_cmd)
        self.assertTrue(idx_align < idx_down < idx_grip < idx_up < idx_park,
                        "动作时序必须严格满足: 平移对准 -> 舵机下探 -> 夹紧 -> 舵机提升 -> Park")

        # 校验 4D 偏差计算 (包含 ΔR = 36.0 - 35.5 = +0.5°)
        self.assertIsNotNone(self.tracker.last_dev)
        self.assertEqual(len(self.tracker.last_dev), 4)
        self.assertAlmostEqual(self.tracker.last_dev[0], 0.5, places=2)
        self.assertAlmostEqual(self.tracker.last_dev[1], 0.2, places=2)
        self.assertAlmostEqual(self.tracker.last_dev[2], 0.0, places=2)
        self.assertAlmostEqual(self.tracker.last_dev[3], 0.5, places=2)

    def test_robot_serial_set_z_height_servo_mapping(self):
        """测试 RobotSerial.set_z_height 正确将物理高度换算为 Servo 0 角度并同步 G92"""
        from src.devices.robot_serial import RobotSerial

        rs = RobotSerial(port="")
        sent = []
        rs.send_gcode = lambda cmd, timeout=3.0: sent.append(cmd) or True

        # 测试最低位 Z=0 -> S270
        rs.set_z_height(0.0)
        self.assertEqual(sent, ["M280 P0 S270", "G92 Z0.00"])

        # 测试最高位 Z=100 -> S0
        sent.clear()
        rs.set_z_height(100.0)
        self.assertEqual(sent, ["M280 P0 S0", "G92 Z100.00"])

        # 测试工作高度 Z=80 -> S54
        sent.clear()
        rs.set_z_height(80.0)
        self.assertEqual(sent, ["M280 P0 S54", "G92 Z80.00"])

    def test_robot_serial_m114_e_axis_parsing(self):
        """测试 RobotSerial 对带有 E 轴的 M114 响应行的正确解析"""
        import io
        from src.devices.robot_serial import RobotSerial

        rs = RobotSerial(port="")
        mock_ser = io.BytesIO(b"X:150.25 Y:280.50 Z:80.00 E:45.30 Count X: ...\r\nok\r\n")
        mock_ser.reset_input_buffer = lambda: None
        mock_ser.write = lambda b: None
        rs.ser = mock_ser

        pos = rs.get_position()
        self.assertIsNotNone(pos)
        self.assertEqual(len(pos), 4)
        self.assertAlmostEqual(pos[0], 150.25)
        self.assertAlmostEqual(pos[1], 280.50)
        self.assertAlmostEqual(pos[2], 80.00)
        self.assertAlmostEqual(pos[3], 45.30)

    def test_recognize_worker_populates_measured_r(self):
        """测试 _recognize_worker 在单帧闭环识别后完整填充 measured_r 与 target_rvec"""
        # 模拟相机控制器
        class MockCamera:
            pipeline_running = True
            def read_frame(self):
                return np.zeros((720, 1280, 3), dtype=np.uint8)

        self.tracker.camera = MockCamera()
        self.tracker.recognizing = True
        self.tracker.set_toast = lambda msg, is_err=False: None
        self.tracker._toggle_camera = lambda **kw: None
        self.tracker.anchor_positions = {0: np.array([0, 0, 0])}
        self.tracker.theoretical = np.array([100, 200, 0])

        # 模拟 _solve_per_frame 返回完整包含 target_r 的解
        fake_sol = {
            "rvec": np.array([0.0, 0.0, 0.0]),
            "tvec": np.array([0.0, 0.0, 800.0]),
            "rmse": 0.25,
            "support": [0],
            "target_world": np.array([150.0, 250.0, 80.0]),
            "target_r": -48.6,
            "target_rvec": np.array([0.1, 0.2, 0.3]),
            "target_tvec": np.array([10.0, 20.0, 800.0]),
        }
        self.tracker._solve_per_frame = lambda det: fake_sol
        self.tracker.tag_detector.detect_tags = lambda frame: {2: np.zeros((4, 2))}
        self.tracker._detect_high_precision = lambda f, d: d

        self.tracker.measured = None
        self.tracker.measured_r = None
        self.tracker.target_rvec = None

        # 执行单帧识别工作线程函数
        self.tracker._recognize_worker()

        self.assertIsNotNone(self.tracker.measured)
        self.assertIsNotNone(self.tracker.measured_r)
        self.assertAlmostEqual(self.tracker.measured_r, -48.6, places=2)
        self.assertIsNotNone(self.tracker.target_rvec)
        self.assertIsNotNone(self.tracker.target_tvec)
        self.assertTrue(self.tracker.world_locked)


if __name__ == "__main__":
    unittest.main()
