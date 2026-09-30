#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
建图工作站下拉选项持久化单元测试 (tests/test_mapping_dropdown_persistence.py)
========================================================================
验证点：
  1. 下拉选项默认持久化与文件格式；
  2. 修改各下拉选项（筛选范围、排序方式、BA理论视图、实测识别视图、XY平面基准）后的持久化与自动恢复；
  3. UI 交互点击（DD_SELECT_* 与 TOGGLE_DRAW_XY_PLANE）自动持久化；
  4. 损坏或非法配置文件的健壮性与安全回退。
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import cv2
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.workspace.workspace_manager import WorkspaceManager
from tools.spatial_mapping_studio.app import SpatialMappingStudioApp, APP_ID


class TestMappingDropdownPersistence(unittest.TestCase):
    def setUp(self):
        WorkspaceManager._current_ws_id = None
        self.test_dir = tempfile.mkdtemp(prefix="test_mapping_persist_")
        self.workspaces_dir = os.path.join(self.test_dir, "workspaces")
        os.makedirs(self.workspaces_dir, exist_ok=True)
        self.config_path = os.path.join(self.test_dir, "config.yaml")
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write("calibration: {}\n")

        self.mgr = WorkspaceManager(workspaces_dir=self.workspaces_dir, config_path=self.config_path)
        self.ws_a = self.mgr.create_workspace(alias="工位A")
        self.ws_b = self.mgr.create_workspace(alias="工位B")
        for ws in (self.ws_a, self.ws_b):
            with open(ws.whitelist_path, "w", encoding="utf-8") as f:
                f.write("tag_default_size_mm: 50.0\nwhitelist: [0, 1, 2]\n")

        self.image_dir = os.path.join(self.test_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)
        for i in range(1, 3):
            img = np.full((720, 1280, 3), 50, dtype=np.uint8)
            cv2.imwrite(os.path.join(self.image_dir, f"view_{i:04d}.png"), img)

        self.map_path = os.path.join(self.test_dir, "test_tags_map.yaml")
        with open(self.map_path, "w", encoding="utf-8") as f:
            yaml.dump({
                "marker_size_mm": 50.0,
                "tags": {
                    0: {"corners": [[-25, 25, 0], [25, 25, 0], [25, -25, 0], [-25, -25, 0]]}
                }
            }, f)

        self.manifest_path = os.path.join(self.test_dir, "test_tag_observations.yaml")
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            yaml.dump({
                "version": "2.0_test",
                "tag_family": "DICT_APRILTAG_16h5",
                "marker_size_mm": 50.0,
                "summary": {"total_images": 2, "total_observations": 2, "total_kept": 2},
                "images": {
                    f"view_{i:04d}.png": {
                        "file_name": f"view_{i:04d}.png",
                        "image_path": os.path.join(self.image_dir, f"view_{i:04d}.png"),
                        "detected_count": 1,
                        "enabled": True,
                        "observations": [{"tag_id": 0, "keep": True, "corners": [[10, 10], [50, 10], [50, 50], [10, 50]]}]
                    }
                    for i in range(1, 3)
                }
            }, f)

        self.settings_file = os.path.join(self.test_dir, "gui_settings.json")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_app(self, workspace_id=None):
        with patch("tools.spatial_mapping_studio.app.WorkspaceManager", return_value=self.mgr):
            return SpatialMappingStudioApp(
                map_path=self.map_path,
                image_dir=self.image_dir,
                marker_size_mm=50.0,
                win_w=1280,
                win_h=720,
                manifest_path=self.manifest_path,
                workspace_id=workspace_id or self.ws_a.workspace_id,
                settings_file=self.settings_file
            )

    def test_save_and_load_dropdown_state(self):
        """测试修改所有下拉选项后，落盘持久化并在新启动时正确恢复"""
        app = self._create_app()
        # 修改各下拉选项
        app.filter_mode = "warning"
        app.sort_mode = "err_desc"
        app.ba_view_mode = "2d"
        app.obs_view_mode = "off"
        app.show_xy_plane_on = True
        app.plane_z = 250.0
        app.save_dropdown_state()

        self.assertTrue(os.path.exists(self.settings_file))
        with open(self.settings_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn(APP_ID, data)
        saved = data[APP_ID]["dropdown_state"]
        self.assertEqual(saved["filter_mode"], "warning")
        self.assertEqual(saved["sort_mode"], "err_desc")
        self.assertEqual(saved["ba_view_mode"], "2d")
        self.assertEqual(saved["obs_view_mode"], "off")
        self.assertTrue(saved["show_xy_plane"])
        self.assertEqual(saved["plane_z"], 250.0)

        # 启动新实例，应自动加载上次保存的状态
        new_app = self._create_app()
        self.assertEqual(new_app.filter_mode, "warning")
        self.assertEqual(new_app.sort_mode, "err_desc")
        self.assertEqual(new_app.ba_view_mode, "2d")
        self.assertEqual(new_app.obs_view_mode, "off")
        self.assertTrue(new_app.show_xy_plane_on)
        self.assertEqual(new_app.plane_z, 250.0)

    def test_ui_events_trigger_persistence(self):
        """测试通过模拟 UI 点击各个下拉选项自动触发持久化"""
        app = self._create_app()

        # 1. 模拟筛选下拉选择
        app._handle_button_click("DD_SELECT_FILTER", ("FILTER_DROPDOWN", "excluded"), 0, 0)
        self.assertEqual(app.filter_mode, "excluded")

        # 2. 模拟排序下拉选择
        app._handle_button_click("DD_SELECT_SORT", ("SORT_DROPDOWN", "tags_desc"), 0, 0)
        self.assertEqual(app.sort_mode, "tags_desc")

        # 3. 模拟 BA 理论显示下拉选择
        app._handle_button_click("DD_SELECT_BA", ("BA_VIEW_DROPDOWN", "off"), 0, 0)
        self.assertEqual(app.ba_view_mode, "off")

        # 4. 模拟实测识别显示下拉选择
        app._handle_button_click("DD_SELECT_OBS", ("OBS_VIEW_DROPDOWN", "2d"), 0, 0)
        self.assertEqual(app.obs_view_mode, "2d")

        # 5. 模拟 Z 轴高度下拉选择
        app._handle_button_click("DD_SELECT_Z", ("PLANE_Z_DROPDOWN", "350"), 0, 0)
        self.assertEqual(app.plane_z, 350.0)
        self.assertTrue(app.show_xy_plane_on)

        # 6. 验证持久化文件中已同步最新状态
        with open(self.settings_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        saved = data[APP_ID]["dropdown_state"]
        self.assertEqual(saved["filter_mode"], "excluded")
        self.assertEqual(saved["sort_mode"], "tags_desc")
        self.assertEqual(saved["ba_view_mode"], "off")
        self.assertEqual(saved["obs_view_mode"], "2d")
        self.assertEqual(saved["plane_z"], 350.0)
        self.assertTrue(saved["show_xy_plane"])

        # 7. 模拟关闭 XY 平面
        app._handle_button_click("DD_SELECT_Z_OFF", ("PLANE_Z_DROPDOWN", "NONE"), 0, 0)
        self.assertFalse(app.show_xy_plane_on)
        with open(self.settings_file, "r", encoding="utf-8") as f:
            saved_off = json.load(f)[APP_ID]["dropdown_state"]
        self.assertFalse(saved_off["show_xy_plane"])

    def test_corrupted_settings_fallback(self):
        """测试损坏配置文件的安全兜底"""
        with open(self.settings_file, "w", encoding="utf-8") as f:
            f.write("{invalid_json: true,,}")

        # 启动时不应抛出异常，使用系统默认值
        app = self._create_app()
        self.assertEqual(app.filter_mode, "all")
        self.assertEqual(app.sort_mode, "name_asc")
        self.assertEqual(app.ba_view_mode, "3d")
        self.assertEqual(app.obs_view_mode, "3d")


if __name__ == "__main__":
    unittest.main()
