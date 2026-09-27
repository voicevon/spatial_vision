# -*- coding: utf-8 -*-
"""
Workspace Hub GUI Smart ROI & Production Metadata Integration Test
验证：
1. ROI 物件编辑弹窗可视化载入与修改 Smart ROI 属性 (role, target_intent, binding, min_confidence)
2. 弹窗事件触发 (角色切换, 意图切换, 槽位循环切换)
3. 保存后写穿持久化至 rois.yaml
4. Workspace 概览卡片上的全局生产模式切换与持久化 (workspace.yaml)
5. HubRenderer 渲染流水线无异常无越界
"""

import os
import shutil
import tempfile
import unittest
import numpy as np

from src.calibration.workspace_manager import WorkspaceManager
from src.calibration.roi_manager import RoiSpaceManager, RoiDefinition
from src.calibration.coordinate_manager import CoordinateTreeManager, FrameDefinition
from tools.workspace_hub.hub_state import HubState
from tools.workspace_hub.hub_renderer import HubRenderer
from tools.workspace_hub.hub_hit_tester import HubHitTester


class TestHubSmartRoiGui(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_hub_smart_roi_")
        self.ws_mgr = WorkspaceManager(workspaces_dir=self.test_dir)
        self.ws = self.ws_mgr.create_workspace("TestStation", "测试工业工位")
        
        # 初始化 State 与 Renderer
        self.state = HubState(self.ws_mgr)
        self.renderer = HubRenderer()
        self.hit_tester = HubHitTester(self.renderer)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_smart_roi_modal_open_and_cycle_events(self):
        """测试在 Hub GUI 中新建与编辑 Smart ROI 的完整生命周期"""
        state = self.state
        state.select_workspace_at_index(0)
        
        # 1. 打开新建 ROI 弹窗
        state.open_roi_modal()
        self.assertTrue(state.roi_modal_open)
        self.assertEqual(state.roi_modal_data["role"], "source")
        self.assertEqual(state.roi_modal_data["target_intent"], "pose_pick")
        self.assertEqual(state.roi_modal_data["binding"].get("slot_index"), 0)

        # 2. 模拟点击切换角色为 destination (落料槽)
        state.set_roi_modal_role("destination")
        self.assertEqual(state.roi_modal_data["role"], "destination")
        
        # 3. 模拟点击切换意图为 piece_count
        state.set_roi_modal_intent("piece_count")
        self.assertEqual(state.roi_modal_data["target_intent"], "piece_count")

        # 4. 循环切换落料槽位
        next_slot = state.cycle_roi_modal_slot()
        self.assertEqual(next_slot, 1)
        self.assertEqual(state.roi_modal_data["binding"]["slot_index"], 1)

        # 5. 保存 ROI
        state.roi_modal_data["roi_id"] = "tray_slot_1"
        state.roi_modal_data["name"] = "1号分选槽"
        state.roi_modal_data["size_xyz_mm"] = [100.0, 150.0, 50.0]
        ok, msg = state.save_roi_modal()
        self.assertTrue(ok, f"保存失败: {msg}")
        self.assertFalse(state.roi_modal_open)

        # 6. 从持久化磁盘重新读取验证
        reloaded_mgr = RoiSpaceManager(self.ws.workspace_id, rois_yaml_path=self.ws.rois_path)
        saved_roi = reloaded_mgr.get_roi("tray_slot_1")
        self.assertIsNotNone(saved_roi)
        self.assertEqual(saved_roi.role, "destination")
        self.assertEqual(saved_roi.target_intent, "piece_count")
        self.assertEqual(saved_roi.binding.get("slot_index"), 1)

    def test_workspace_production_mode_toggle(self):
        """测试 Workspace 概览卡片上的全局生产模式切换与保存"""
        state = self.state
        state.select_workspace_at_index(0)
        ws = state.get_selected_workspace()
        self.assertIsNotNone(ws)

        # 初始模式应当为 scara_sorting
        self.assertEqual(ws.production.get("mode"), "scara_sorting")

        # 点击切换模式 -> wheel_inspection
        m1 = state.cycle_workspace_production_mode()
        self.assertEqual(m1, "wheel_inspection")
        self.assertEqual(ws.production.get("mode"), "wheel_inspection")

        # 再次切换 -> none (通用标定观察)
        m2 = state.cycle_workspace_production_mode()
        self.assertEqual(m2, "none")
        self.assertEqual(ws.production.get("mode"), "none")

        # 再次切换 -> 循环回 scara_sorting
        m3 = state.cycle_workspace_production_mode()
        self.assertEqual(m3, "scara_sorting")

        # 验证 workspace.yaml 持久化重载
        reloaded_ws = self.ws_mgr.get_workspace_by_id(ws.workspace_id, force_refresh=True)
        self.assertEqual(reloaded_ws.production.get("mode"), "scara_sorting")

    def test_hub_render_with_smart_roi_modal(self):
        """测试 HubRenderer 在打开 Smart ROI 弹窗和列表卡片时的完整渲染"""
        state = self.state
        state.select_workspace_at_index(0)
        
        # 1. 常规大盘渲染
        state.active_tab = HubState.TAB_REPORT
        canvas1 = self.renderer.render(state)
        self.assertEqual(canvas1.shape, (720, 960, 3))

        # 2. ROI 列表页签渲染 (工位多坐标系与 ROI 集合)
        state.active_tab = HubState.TAB_FRAMES_ROIS
        canvas2 = self.renderer.render(state)
        self.assertEqual(canvas2.shape, (720, 960, 3))

        # 3. 打开 Smart ROI 弹窗渲染
        state.open_roi_modal()
        canvas3 = self.renderer.render(state)
        self.assertEqual(canvas3.shape, (720, 960, 3))
        # 确保没有崩溃抛错且画布内容非全黑
        self.assertGreater(np.count_nonzero(canvas3), 1000)


if __name__ == "__main__":
    unittest.main()
