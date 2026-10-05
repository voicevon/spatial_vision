#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, shutil, tempfile, unittest
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.workspace.workspace_manager import WorkspaceManager
import tools.capture.capture_wizard as capture_wizard_mod
import tools.spatial_mapping_studio.app as mapping_studio_mod
import tools.asparagus_pose_studio.data_io as asparagus_io_mod
from tools.capture.capture_wizard import CaptureWizard
from tools.spatial_mapping_studio.app import SpatialMappingStudioApp
from tools.asparagus_pose_studio import AsparagusPoseStudioApp

class TestWorkspaceDropdownPersistence(unittest.TestCase):
    def setUp(self):
        # 当前工位为类级运行时状态, 用例间显式重置防串扰
        WorkspaceManager._current_ws_id = None
        self.test_dir = tempfile.mkdtemp(prefix='test_ws_persist_')
        # 隔离 GUI 偏好文件: 绝不触碰真实 config/gui_settings.json
        tmp_settings = os.path.join(self.test_dir, 'gui_settings.json')
        for mod in (capture_wizard_mod, mapping_studio_mod, asparagus_io_mod):
            p = patch.object(mod, 'GUI_SETTINGS_FILE', tmp_settings)
            p.start()
            self.addCleanup(p.stop)
        self.workspaces_dir = os.path.join(self.test_dir, 'workspaces')
        os.makedirs(self.workspaces_dir, exist_ok=True)
        self.config_path = os.path.join(self.test_dir, 'config.yaml')
        with open(self.config_path, 'w', encoding='utf-8') as f:
            f.write('calibration: {}\n')

        self.mgr = WorkspaceManager(workspaces_dir=self.workspaces_dir, config_path=self.config_path)
        self.ws_a = self.mgr.create_workspace(alias='工位A')
        self.ws_b = self.mgr.create_workspace(alias='工位B')
        self.ws_c = self.mgr.create_workspace(alias='工位C')
        for ws in (self.ws_a, self.ws_b, self.ws_c):
            with open(ws.whitelist_path, 'w', encoding='utf-8') as f:
                f.write('tag_default_size_mm: 50.0\nwhitelist: [0, 1, 2]\n')

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch('tools.capture.capture_wizard.CameraService')
    @patch('tools.capture.capture_wizard.WorkspaceManager')
    def test_capture_wizard_persists_workspace(self, mock_ws_mgr_cls, mock_cam_service):
        mock_ws_mgr_cls.return_value = self.mgr
        mock_cam = MagicMock()
        mock_cam.start.return_value = True
        mock_cam_service.return_value = mock_cam

        wizard = CaptureWizard()
        wizard.switch_workspace(self.ws_b.workspace_id)
        self.assertEqual(wizard.current_workspace_id, self.ws_b.workspace_id)
        self.assertEqual(self.mgr.get_current_workspace_id(), self.ws_b.workspace_id)

    @patch('tools.spatial_mapping_studio.app.WorkspaceManager')
    def test_spatial_mapping_studio_persists_workspace(self, mock_ws_mgr_cls):
        mock_ws_mgr_cls.return_value = self.mgr

        app = SpatialMappingStudioApp()
        app.switch_workspace(self.ws_c.workspace_id)
        self.assertEqual(app.current_workspace_id, self.ws_c.workspace_id)
        self.assertEqual(self.mgr.get_current_workspace_id(), self.ws_c.workspace_id)

    @patch('tools.asparagus_pose_studio.app.WorkspaceManager')
    def test_asparagus_pose_studio_persists_workspace(self, mock_ws_mgr_cls):
        mock_ws_mgr_cls.return_value = self.mgr
        app = AsparagusPoseStudioApp()
        app.switch_workspace(self.ws_a.workspace_id)
        self.assertEqual(app.current_workspace_id, self.ws_a.workspace_id)
        self.assertEqual(self.mgr.get_current_workspace_id(), self.ws_a.workspace_id)

    @patch('tools.asparagus_pose_studio.app.WorkspaceManager')
    @patch('tools.capture.capture_wizard.CameraService')
    @patch('tools.capture.capture_wizard.WorkspaceManager')
    @patch('tools.spatial_mapping_studio.app.WorkspaceManager')
    def test_cross_tool_workspace_persistence_handoff(self, mock_studio_ws_cls, mock_cap_ws_cls, mock_cam_service, mock_aspar_ws_cls):
        mock_cap_ws_cls.return_value = self.mgr
        mock_studio_ws_cls.return_value = self.mgr
        mock_aspar_ws_cls.return_value = self.mgr
        mock_cam = MagicMock()
        mock_cam.start.return_value = True
        mock_cam_service.return_value = mock_cam

        # 1. 采集向导中切换至工位 B
        wiz = CaptureWizard()
        wiz.switch_workspace(self.ws_b.workspace_id)
        self.assertEqual(self.mgr.get_current_workspace_id(), self.ws_b.workspace_id)

        # 2. 空间建图工作站无参启动，自动加载最新持久化的工位 B
        studio = SpatialMappingStudioApp()
        self.assertEqual(studio.current_workspace_id, self.ws_b.workspace_id)

        # 3. 空间建图工作站中切换至工位 C
        studio.switch_workspace(self.ws_c.workspace_id)
        self.assertEqual(self.mgr.get_current_workspace_id(), self.ws_c.workspace_id)

        # 4. 芦笋位姿工作室无参启动，自动加载最新持久化的工位 C
        offline = AsparagusPoseStudioApp()
        self.assertEqual(offline.current_workspace_id, self.ws_c.workspace_id)

if __name__ == '__main__':
    unittest.main()
