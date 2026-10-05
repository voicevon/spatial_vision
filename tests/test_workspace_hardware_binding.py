#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：Workspace 工位相机硬件与分辨率单一真理源 (SSOT) 绑定
============================================================
验证:
1. Workspace 模型默认包含 camera_type, camera_serial, camera_resolution
2. 工位 workspace_meta.yaml 准确持久化与加载硬件规格
3. WorkspaceManager.create_workspace 支持指定相机类型与规格
4. WorkspaceManager.bind_workspace_camera 具备原子写穿更新能力
5. CaptureWizard 载入工位时完全受工位硬件参数驱动
"""

import os
import sys
import tempfile
import unittest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.workspace.workspace_manager import WorkspaceManager, Workspace
from tools.capture.capture_wizard import CaptureWizard
import tools.capture.capture_wizard as wizard_mod


class TestWorkspaceHardwareBinding(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="ws_hw_test_")
        self.mgr = WorkspaceManager(workspaces_dir=self.tmp_dir)

    def test_workspace_model_hardware_defaults_and_meta(self):
        """测试工位模型默认硬件参数与元数据持久化读写"""
        ws = self.mgr.create_workspace(
            alias="测试工位A",
            description="默认硬件工位"
        )
        self.assertEqual(ws.camera_type, "realsense")
        self.assertEqual(ws.camera_serial, "")
        self.assertEqual(ws.resolution, (1920, 1080))
        self.assertEqual(ws.resolution_str, "1920x1080")

        # 检查持久化文件
        meta_file = ws.meta_path
        self.assertTrue(os.path.exists(meta_file))
        with open(meta_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.assertEqual(data["camera_type"], "realsense")
        self.assertEqual(data["camera_serial"], "")
        self.assertEqual(data["camera_resolution"], [1920, 1080])

        # 重新加载
        loaded_ws = Workspace.load(ws.workspace_dir)
        self.assertIsNotNone(loaded_ws)
        self.assertEqual(loaded_ws.camera_type, "realsense")
        self.assertEqual(loaded_ws.resolution_str, "1920x1080")

    def test_create_and_update_workspace_custom_hardware(self):
        """测试指定 USB 相机与 720P 分辨率创建及后续热更新"""
        ws = self.mgr.create_workspace(
            alias="USB工位B",
            description="USB相机工位",
            camera_type="usb",
            camera_serial="1",
            camera_resolution=[1280, 720]
        )
        self.assertEqual(ws.camera_type, "usb")
        self.assertEqual(ws.camera_serial, "1")
        self.assertEqual(ws.resolution_str, "1280x720")

        # 更新硬件为 RealSense 1080P
        ok, msg = self.mgr.bind_workspace_camera(
            workspace_id=ws.workspace_id,
            camera_type="realsense",
            camera_serial="233522070123",
            camera_resolution=[1920, 1080]
        )
        self.assertTrue(ok)
        self.assertIn("已更新", msg)

        # 重新载入验证
        refreshed_ws = self.mgr.get_workspace_by_id(ws.workspace_id, force_refresh=True)
        self.assertEqual(refreshed_ws.camera_type, "realsense")
        self.assertEqual(refreshed_ws.camera_serial, "233522070123")
        self.assertEqual(refreshed_ws.resolution_str, "1920x1080")

    def test_capture_wizard_hardware_ssot_binding(self):
        """测试采图向导完全受当前工位硬件与分辨率驱动"""
        ws_custom = self.mgr.create_workspace(
            alias="特定硬件工位",
            camera_type="usb",
            camera_serial="2",
            camera_resolution=[1280, 720]
        )

        wizard_mod.GUI_SETTINGS_FILE = os.path.join(self.tmp_dir, "gui_settings.json")
        wiz = CaptureWizard(workspace_id=ws_custom.workspace_id, output_dir=None, workspace_mgr=self.mgr)

        # 验证采图向导自动装载了工位的相机类型与分辨率
        self.assertEqual(wiz.camera_type, "usb")
        self.assertEqual(wiz.camera_serial, "2")
        self.assertEqual(wiz.resolution, "1280x720")
        self.assertEqual((wiz.frame_w, wiz.frame_h), (1280, 720))

        # 切换到另一个默认工位 (RealSense 1080P)
        ws_default = self.mgr.create_workspace(alias="默认1080P工位")
        wiz.switch_workspace(ws_default.workspace_id)

        self.assertEqual(wiz.camera_type, "realsense")
        self.assertEqual(wiz.camera_serial, "")
        self.assertEqual(wiz.resolution, "1920x1080")
        self.assertEqual((wiz.frame_w, wiz.frame_h), (1920, 1080))

    def test_hub_hardware_edit_lock_defense(self):
        """测试在已有照片时锁定硬件修改，未锁定空白工位允许修改"""
        from tools.workspace_hub.app import WorkspaceHubApp
        import numpy as np
        import cv2

        ws = self.mgr.create_workspace(alias="锁定测试工位")
        app = WorkspaceHubApp(workspace_mgr=self.mgr)
        app.state.select_tree_workspace(0)
        cur_ws = app.state.get_selected_workspace()
        self.assertIsNotNone(cur_ws)

        # 1. 模拟向工位写入一张测试照片，使其处于锁定状态
        test_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        img_p = os.path.join(cur_ws.intrinsics_raw_images_dir, "test_lock.png")
        cv2.imwrite(img_p, test_img)
        self.assertGreater(cur_ws.get_image_count("intrinsics"), 0)

        # 尝试触发修改硬件，应被防御拦截
        app._action_edit_hardware_config()
        self.assertIn("已锁定硬件配置", app.state.toast_msg)
        self.assertIn("禁止修改", app.state.toast_msg)


if __name__ == "__main__":
    unittest.main()
