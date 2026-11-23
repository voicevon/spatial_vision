#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单根芦笋测量工作室 GUI 单元测试 (tests/test_asparagus_measure_studio.py)
严格遵循测试隔离铁律：使用临时 settings_file 与临时样本目录，不污染真实偏好。
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from tests.test_single_asparagus_measurer import _make_synthetic_asparagus
import tools.asparagus_measure_studio.data_io as measure_data_io
from tools.asparagus_measure_studio.app import AsparagusMeasureStudioApp


class TestAsparagusMeasureStudio(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sample_dir = os.path.join(self.temp_dir, "samples")
        os.makedirs(self.sample_dir, exist_ok=True)
        self.tmp_settings = os.path.join(self.temp_dir, "gui_settings.json")

        # 写入一张合成测试样本（带对齐深度）
        img = _make_synthetic_asparagus(
            orientation="left_to_right", white_px=60, green_px=280,
            diam_at_20mm_px=26, diam_far_px=14, mm_per_px=0.5,
        )
        cv2.imwrite(os.path.join(self.sample_dir, "sample_01.png"), img)
        np.save(os.path.join(self.sample_dir, "sample_01_depth.npy"),
                np.full((img.shape[0], img.shape[1]), 500.0, dtype=np.float32))

        p = patch.object(measure_data_io, "GUI_SETTINGS_FILE", self.tmp_settings)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))

    def test_offline_init_measure_and_render(self):
        """离线模式初始化、自动载入样本、测量并成功渲染 1280x800 画布"""
        app = AsparagusMeasureStudioApp(sample_dir=self.sample_dir, settings_file=self.tmp_settings)
        self.assertEqual(len(app.samples), 1)
        self.assertIsNotNone(app.measurement)
        self.assertTrue(app.measurement.valid)
        self.assertTrue(app.measurement.has_white_root)

        canvas = app.render()
        self.assertEqual(canvas.shape, (800, 1280, 3))
        self.assertTrue(np.any(canvas > 0))
        app.cleanup()

    def test_orientation_cycle_and_settings_persistence(self):
        """切换朝向与手动 ROI 后持久化落盘，重新启动自动恢复"""
        app = AsparagusMeasureStudioApp(sample_dir=self.sample_dir, settings_file=self.tmp_settings)
        app.cycle_orientation()
        self.assertEqual(app.measurer.config.orientation, "right_to_left")
        app.manual_roi = (100, 80, 420, 200)
        app.cleanup()

        app2 = AsparagusMeasureStudioApp(sample_dir=self.sample_dir, settings_file=self.tmp_settings)
        self.assertEqual(app2.measurer.config.orientation, "right_to_left")
        self.assertEqual(app2.manual_roi, (100, 80, 420, 200))
        app2.cleanup()

    def test_live_camera_mode_with_mock_service(self):
        """通过 Mock CameraService 验证实时模式切换、逐帧测量与安全释放"""
        mock_cam = MagicMock()
        mock_cam.is_running = False
        mock_cam.depth_enabled = True
        mock_cam.stream_desc = "Mock 640x360 RGB-D"

        frame = _make_synthetic_asparagus(
            orientation="left_to_right", white_px=0, green_px=260,
            diam_at_20mm_px=24, diam_far_px=12, mm_per_px=0.5,
        )
        depth = np.full((frame.shape[0], frame.shape[1]), 500.0, dtype=np.float32)

        def _fake_start(*args, **kwargs):
            mock_cam.is_running = True
            return True

        def _fake_stop():
            mock_cam.is_running = False

        mock_cam.start_realsense.side_effect = _fake_start
        mock_cam.stop.side_effect = _fake_stop
        mock_cam.read_rgbd_frame.return_value = (frame, depth)

        app = AsparagusMeasureStudioApp(
            sample_dir=self.sample_dir,
            settings_file=self.tmp_settings,
            camera_service=mock_cam,
        )
        app.toggle_source_mode()
        self.assertEqual(app.source_mode, "live")
        app.on_tick()
        self.assertIsNotNone(app.measurement)
        self.assertTrue(app.measurement.valid)
        self.assertFalse(app.measurement.has_white_root)

        canvas = app.render()
        self.assertEqual(canvas.shape, (800, 1280, 3))

        app.cleanup()
        mock_cam.stop.assert_called()

    def test_realsense_failure_does_not_fallback_to_usb(self):
        """RealSense 工位未连接设备时直接报错提示，不盲目回退 USB 摄像头"""
        mock_cam = MagicMock()
        mock_cam.is_running = False
        mock_cam.start_realsense.side_effect = RuntimeError("未检测到 RealSense 设备")

        app = AsparagusMeasureStudioApp(
            sample_dir=self.sample_dir,
            settings_file=self.tmp_settings,
            camera_service=mock_cam,
        )
        fake_ws = MagicMock(camera_type="realsense", resolution=(1280, 720), camera_serial=None)
        with patch.object(app.workspace_mgr, "get_workspace_by_id", return_value=fake_ws):
            app.toggle_source_mode()

        self.assertEqual(app.source_mode, "offline")
        mock_cam.start_realsense.assert_called_once()
        mock_cam.start_usb.assert_not_called()
        self.assertIn("未检测到 RealSense 设备", app.error_msg)
        app.cleanup()

    def test_usb_workspace_starts_usb_camera_directly(self):
        """USB 工位按工位绑定分辨率与设备索引直接启动 USB 相机"""
        mock_cam = MagicMock()
        mock_cam.is_running = False
        mock_cam.stream_desc = "USB[#1] 1920x1080"

        app = AsparagusMeasureStudioApp(
            sample_dir=self.sample_dir,
            settings_file=self.tmp_settings,
            camera_service=mock_cam,
        )
        fake_ws = MagicMock(camera_type="usb", resolution=(1920, 1080), camera_serial="1")
        with patch.object(app.workspace_mgr, "get_workspace_by_id", return_value=fake_ws):
            app.toggle_source_mode()

        self.assertEqual(app.source_mode, "live")
        mock_cam.start_usb.assert_called_once_with(1920, 1080, fps=30, device_index="1")
        mock_cam.start_realsense.assert_not_called()
        app.cleanup()


if __name__ == "__main__":
    unittest.main()
