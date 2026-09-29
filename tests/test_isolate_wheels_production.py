#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_isolate_wheels_production.py
======================================
分离轮 8 通道自动生产工作台 (isolate_wheels_production) 单元测试矩阵：
1. 工作台初始化与中枢绑定
2. 单拍生产节拍下发与速度配置
3. 生产配方调整与快捷预设
4. 节拍完成回调、产量统计与 PPM 计算
5. 自动连续循环状态机与安全急停
6. 生产大屏 GUI 离线渲染冒烟测试
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.isolate_wheels_production.app import IsolateWheelsProductionApp
from tools.isolate_wheels_production.renderer import LOGIC_H, LOGIC_W


class TestIsolateWheelsProduction(unittest.TestCase):
    """分离轮生产工作台单元测试"""

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
        """验证初始状态与配方"""
        self.assertEqual(self.app.recipe_counts, [1] * 8)
        self.assertEqual(self.app.discharged_counts, [0] * 8)
        self.assertEqual(self.app.total_cycles, 0)
        self.assertFalse(self.app.auto_running)
        self.assertFalse(self.app.paused)
        self.assertEqual(self.app.load_speed, 1.0)

    def test_send_step_beat(self):
        """测试单拍出料下发"""
        self.app.load_speed = 1.5
        ok = self.app.send_step_beat()
        self.assertTrue(ok)
        self.app.controller._client.publish.assert_called_once()
        topic, payload = self.app.controller._client.publish.call_args[0][:2]
        self.assertIn("flux/loader/F8EC/cmd", topic)
        self.assertIn('"speed": 1.5', payload)
        self.assertIn('"counts": [1, 1, 1, 1, 1, 1, 1, 1]', payload)

    def test_done_event_accumulation(self):
        """测试完成事件触发产量累加与周期计算"""
        self.app.recipe_counts = [2, 0, 1, 0, 0, 0, 0, 3]
        self.app.cycle_start_time = 100.0

        # 模拟收到 done 回调
        self.app._on_done_event("F8EC", "load")

        self.assertEqual(self.app.total_cycles, 1)
        self.assertEqual(self.app.discharged_counts, [2, 0, 1, 0, 0, 0, 0, 3])
        self.assertEqual(len(self.app.cycle_durations), 1)

    def test_calculate_ppm(self):
        """测试 PPM 计算"""
        self.app.recipe_counts = [1] * 8  # 每拍 8 件
        self.app.cycle_durations = [2.0]   # 每拍 2 秒
        # 60 / 2 * 8 = 240 PPM
        self.assertAlmostEqual(self.app.calculate_ppm(), 240.0, places=1)

    def test_recipe_presets_and_reset(self):
        """测试配方预设与重置"""
        self.app.recipe_counts = [3] * 8
        self.app.total_cycles = 10
        self.app.discharged_counts = [30] * 8

        self.app.reset_batch_statistics()
        self.assertEqual(self.app.total_cycles, 0)
        self.assertEqual(self.app.discharged_counts, [0] * 8)

    def test_safety_stop_on_offline(self):
        """测试设备离线或故障时自动安全停止"""
        self.app.auto_running = True
        self.app._on_state_event("F8EC", "offline")
        self.assertFalse(self.app.auto_running)

    def test_render_smoke_test(self):
        """测试生产工作台离线渲染无崩溃"""
        canvas = self.app.render()
        self.assertIsInstance(canvas, np.ndarray)
        self.assertEqual(canvas.shape, (LOGIC_H, LOGIC_W, 3))
        # 画布应当有绘制内容 (非全黑)
        self.assertGreater(int(np.sum(canvas)), 0)


if __name__ == "__main__":
    unittest.main()
