#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_isolate_wheels_production.py
======================================
分离轮 8 通道自动生产工作台 (isolate_wheels_production) 单元测试矩阵：
1. 工作台初始化与中枢绑定
2. 工位场景选择与 ROI 加载 (步骤一)
3. 相机设备选择与分辨率切换、相机启停 (步骤二)
4. 视觉分离自动计数与数字只读展示 (步骤三: 严禁人工篡改)
5. 单次生产模式：人工触发节拍下发 (步骤四)
6. 连续流水线模式：收到下位机 MQTT done 自动递推下发下一拍 (步骤四)
7. 批次产量统计、PPM 效能计算与安全急停
8. 生产大屏 GUI 离线渲染冒烟测试
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

import cv2
import numpy as np

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.isolate_wheels_production.app import IsolateWheelsProductionApp
from tools.isolate_wheels_production.renderer import LOGIC_H, LOGIC_W


class TestIsolateWheelsProduction(unittest.TestCase):
    """分离轮视觉自动化生产工作台单元测试"""

    def setUp(self):
        tmp_settings = os.path.join(tempfile.mkdtemp(prefix="test_prod_"), "gui_settings.json")
        self.app = IsolateWheelsProductionApp(settings_file=tmp_settings)
        # Mock MQTT client
        self.app.controller._client = MagicMock()
        self.app.controller._connected = True
        self.app.controller.devices["F8EC"] = "idle"

    def tearDown(self):
        self.app.cleanup()

    def test_initial_state(self):
        """验证初始状态与工位/相机默认配置"""
        self.assertEqual(len(self.app.detected_counts), 8)
        self.assertEqual(self.app.discharged_counts, [0] * 8)
        self.assertEqual(self.app.total_cycles, 0)
        self.assertFalse(self.app.auto_pipeline)
        self.assertFalse(self.app.paused)
        self.assertEqual(self.app.load_speed, 1.0)
        self.assertTrue(hasattr(self.app, "current_workspace_id"))
        self.assertEqual(self.app.camera_w, 1280)
        self.assertEqual(self.app.camera_h, 720)

    def test_workspace_switch(self):
        """测试步骤一: 工位切换与 ROI 标靶同步"""
        ws_list = self.app.workspace_mgr.list_workspaces()
        if ws_list:
            target_ws = ws_list[0].workspace_id
            self.app.switch_workspace(target_ws)
            self.assertEqual(self.app.current_workspace_id, target_ws)
            self.assertTrue(len(self.app.detector.rois) == 8)

    def test_camera_and_resolution(self):
        """测试步骤二: 相机切换、分辨率切换与启停"""
        # 切换 Mock 相机
        self.app.switch_camera_type("mock")
        self.assertEqual(self.app.camera_type, "mock")
        # 切换分辨率
        self.app.switch_resolution(640, 480)
        self.assertEqual(self.app.camera_w, 640)
        self.assertEqual(self.app.camera_h, 480)
        # 启动相机
        ok = self.app.open_camera()
        self.assertTrue(ok)
        self.assertTrue(self.app.is_camera_running)

        # 触发一次 update 读取帧
        self.app.update()
        self.assertIsNotNone(self.app.current_frame)
        self.app.close_camera()
        self.assertFalse(self.app.is_camera_running)

    def test_vision_detector_auto_counts(self):
        """测试步骤三: 视觉分离检测自动计数 (数字只读展示，严禁人工修改)"""
        # 构造一张带有明显前景的图像 (尺寸 640x480)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        norm_rois = self.app.detector.roi_norm_rects
        if len(norm_rois) >= 8:
            # 在第 1 个轮和第 8 个轮的 ROI 区域绘制模拟白色物料圆形
            nx1, ny1, nw1, nh1 = norm_rois[0]
            cx1 = int((nx1 + nw1 / 2) * 640)
            cy1 = int((ny1 + nh1 / 2) * 480)
            cv2.circle(img, (cx1, cy1), 22, (255, 255, 255), -1)

            nx8, ny8, nw8, nh8 = norm_rois[7]
            cx8 = int((nx8 + nw8 / 2) * 640)
            cy8 = int((ny8 + nh8 / 2) * 480)
            cv2.circle(img, (cx8, cy8), 22, (255, 255, 255), -1)

        counts = self.app.detector.process_frame(img)
        self.assertEqual(len(counts), 8)
        for c in counts:
            self.assertTrue(0 <= c <= 9)
        self.assertGreaterEqual(counts[0], 1)
        self.assertGreaterEqual(counts[7], 1)

    def test_manual_single_beat(self):
        """测试步骤四(A): 单次生产模式 (人工触发节拍下发)"""
        self.app.load_speed = 1.5
        self.app.detected_counts = [1, 2, 0, 0, 1, 0, 0, 3]

        ok = self.app.send_step_beat()
        self.assertTrue(ok)
        self.app.controller._client.publish.assert_called_once()
        topic, payload = self.app.controller._client.publish.call_args[0][:2]
        self.assertIn("flux/loader/F8EC/cmd", topic)
        self.assertIn('"speed": 1.5', payload)
        self.assertIn('"counts": [1, 2, 0, 0, 1, 0, 0, 3]', payload)
        self.assertTrue(self.app.waiting_done)

    def test_auto_pipeline_closed_loop(self):
        """测试步骤四(B): 连续自动生产流水线模式 (收到 MQTT done 时自动下发下一拍)"""
        # 启动连续流水线
        self.app.detected_counts = [2, 0, 1, 0, 0, 0, 0, 1]
        self.app.start_pipeline()
        self.assertTrue(self.app.auto_pipeline)
        self.assertTrue(self.app.waiting_done)
        self.assertEqual(self.app.controller._client.publish.call_count, 1)

        # 下位机执行完成，上位机收到 MQTT done
        # 在此期间视觉更新了最新物料数
        self.app.detected_counts = [0, 1, 1, 1, 0, 0, 0, 2]
        self.app._on_done_event("F8EC", "load")

        # 验证:
        # 1. 累计统计累加了上一拍的数量
        self.assertEqual(self.app.discharged_counts, [2, 0, 1, 0, 0, 0, 0, 1])
        self.assertEqual(self.app.total_cycles, 1)
        # 2. 自动以最新视觉检测数量下发了下一个节拍！publish 调了第 2 次
        self.assertEqual(self.app.controller._client.publish.call_count, 2)
        _, payload2 = self.app.controller._client.publish.call_args[0][:2]
        self.assertIn('"counts": [0, 1, 1, 1, 0, 0, 0, 2]', payload2)

    def test_calculate_ppm_and_reset(self):
        """测试 PPM 生产效能计算与统计重置"""
        self.app.detected_counts = [1] * 8
        self.app.cycle_durations = [2.0]  # 2秒一拍
        # (60 / 2) * 8 = 240 PPM
        self.assertAlmostEqual(self.app.calculate_ppm(), 240.0, places=1)

        self.app.total_cycles = 5
        self.app.discharged_counts = [5] * 8
        self.app.reset_batch_statistics()
        self.assertEqual(self.app.total_cycles, 0)
        self.assertEqual(self.app.discharged_counts, [0] * 8)
        self.assertEqual(len(self.app.cycle_durations), 0)

    def test_safety_stop_on_offline(self):
        """测试设备离线或故障时自动安全停止流水线"""
        self.app.auto_pipeline = True
        self.app.waiting_done = True
        self.app._on_state_event("F8EC", "offline")
        self.assertFalse(self.app.auto_pipeline)
        self.assertFalse(self.app.waiting_done)

    def test_render_smoke_test(self):
        """测试生产工作台工业大屏离线渲染无崩溃"""
        canvas = self.app.render()
        self.assertIsInstance(canvas, np.ndarray)
        self.assertEqual(canvas.shape, (LOGIC_H, LOGIC_W, 3))
        # 画布应当有绘制内容 (非全黑)
        self.assertGreater(int(np.sum(canvas)), 0)


if __name__ == "__main__":
    unittest.main()
