#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用 GUI 视窗管理器单元测试 (tests/test_gui_window_manager.py)
============================================================
覆盖 GuiWindowManager 的命名空间隔离配置持久化、缩放控制、防抖落盘与键鼠事件处理
"""

import os
import json
import tempfile
import unittest
from unittest.mock import patch

from src.ui.gui_window_manager import GuiWindowManager


class TestGuiWindowManager(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_settings_file = os.path.join(self.temp_dir.name, "gui_settings.json")
        self.mgr = GuiWindowManager(
            app_id="test_app",
            base_w=1280,
            base_h=720,
            settings_file=self.test_settings_file
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_initialization(self):
        """测试初始默认参数"""
        self.assertEqual(self.mgr.app_id, "test_app")
        self.assertEqual(self.mgr.scale_pct, 100)
        self.assertAlmostEqual(self.mgr.scale, 1.0)
        self.assertEqual(self.mgr.canvas_w, 1280)
        self.assertEqual(self.mgr.canvas_h, 720)

    def test_zoom_in_out_reset(self):
        """测试比例增减、边界限制与复位"""
        # 1. 放大 +10%
        changed, toast = self.mgr.apply_zoom(+10)
        self.assertTrue(changed)
        self.assertEqual(self.mgr.scale_pct, 110)
        self.assertEqual(self.mgr.canvas_w, int(1280 * 1.1))
        self.assertEqual(self.mgr.canvas_h, int(720 * 1.1))
        self.assertIn("110%", toast)

        # 2. 连续放大至上限 200%
        for _ in range(15):
            self.mgr.apply_zoom(+10)
        self.assertEqual(self.mgr.scale_pct, 200)

        # 越界拦截
        changed, _ = self.mgr.apply_zoom(+10)
        self.assertFalse(changed)
        self.assertEqual(self.mgr.scale_pct, 200)

        # 3. 缩小至 50% 下限
        for _ in range(20):
            self.mgr.apply_zoom(-10)
        self.assertEqual(self.mgr.scale_pct, 50)
        changed, _ = self.mgr.apply_zoom(-10)
        self.assertFalse(changed)

        # 4. 一键复位
        changed, toast = self.mgr.apply_zoom(0, reset=True)
        self.assertTrue(changed)
        self.assertEqual(self.mgr.scale_pct, 100)
        self.assertEqual(self.mgr.canvas_w, 1280)
        self.assertEqual(self.mgr.canvas_h, 720)
        self.assertIn("100%", toast)

    def test_multi_app_settings_isolation(self):
        """测试多个不同 GUI 应用在同一设置文件中的命名空间隔离与独立持久化"""
        app1 = GuiWindowManager(app_id="app_alpha", settings_file=self.test_settings_file)
        app2 = GuiWindowManager(app_id="app_beta", settings_file=self.test_settings_file)

        # app1 设为 130%
        app1.apply_zoom(+30)
        self.assertEqual(app1.scale_pct, 130)

        # app2 设为 80%
        app2.apply_zoom(-20)
        self.assertEqual(app2.scale_pct, 80)

        # 验证 JSON 内容结构包含两个独立的 key
        self.assertTrue(os.path.exists(self.test_settings_file))
        with open(self.test_settings_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("app_alpha", data)
        self.assertIn("app_beta", data)
        self.assertEqual(data["app_alpha"]["scale_pct"], 130)
        self.assertEqual(data["app_beta"]["scale_pct"], 80)

        # 模拟重启两个应用，验证分别准确恢复
        new_app1 = GuiWindowManager(app_id="app_alpha", settings_file=self.test_settings_file)
        new_app2 = GuiWindowManager(app_id="app_beta", settings_file=self.test_settings_file)
        self.assertEqual(new_app1.scale_pct, 130)
        self.assertEqual(new_app2.scale_pct, 80)

    def test_mouse_wheel_handling(self):
        """测试 Ctrl + 鼠标滚轮缩放"""
        self.mgr._force_ctrl_pressed = True

        # 向上滚 (+1) 放大
        handled, toast = self.mgr.handle_mouse_wheel(event=10, flags=1)
        self.assertTrue(handled)
        self.assertEqual(self.mgr.scale_pct, 110)

        # 向下滚 (-1) 缩小
        handled, toast = self.mgr.handle_mouse_wheel(event=10, flags=-1)
        self.assertTrue(handled)
        self.assertEqual(self.mgr.scale_pct, 100)

        # 未按 Ctrl 应该忽略
        self.mgr._force_ctrl_pressed = False
        with patch("sys.platform", "darwin"):  # 屏蔽 win32 物理查询
            handled, _ = self.mgr.handle_mouse_wheel(event=10, flags=1)
            self.assertFalse(handled)

    def test_keyboard_fallback(self):
        """测试笔记本键盘快捷键全兼容"""
        self.mgr._force_ctrl_pressed = True

        # '=' (免 Shift)
        changed, _ = self.mgr.handle_keyboard_fallback(ord('='))
        self.assertTrue(changed)
        self.assertEqual(self.mgr.scale_pct, 110)

        # '+' (带 Shift)
        changed, _ = self.mgr.handle_keyboard_fallback(ord('+'))
        self.assertTrue(changed)
        self.assertEqual(self.mgr.scale_pct, 120)

        # '-' (减号)
        changed, _ = self.mgr.handle_keyboard_fallback(ord('-'))
        self.assertTrue(changed)
        self.assertEqual(self.mgr.scale_pct, 110)

        # '_' (下划线)
        changed, _ = self.mgr.handle_keyboard_fallback(ord('_'))
        self.assertTrue(changed)
        self.assertEqual(self.mgr.scale_pct, 100)

        # '0' (复位)
        self.mgr.apply_zoom(+50)
        changed, _ = self.mgr.handle_keyboard_fallback(ord('0'))
        self.assertTrue(changed)
        self.assertEqual(self.mgr.scale_pct, 100)

    def test_poll_events_composite(self):
        """测试 poll_events 综合轮询逻辑"""
        # 1. 模拟按 ESC 退出
        res = self.mgr.poll_events(raw_key=27)
        self.assertTrue(res.should_quit)

        # 2. 模拟普通键无退出
        res = self.mgr.poll_events(raw_key=-1)
        self.assertFalse(res.should_quit)
        self.assertEqual(res.canvas_w, 1280)
        self.assertEqual(res.canvas_h, 720)


if __name__ == "__main__":
    unittest.main()
