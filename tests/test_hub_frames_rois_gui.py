# -*- coding: utf-8 -*-
"""
Workspace Hub 坐标系与 3D ROI GUI 交互端到端自动化测试
验证：
1. GeometryState 弹窗生命周期、强 Schema 校验与 YAML 写穿
2. HubRenderer 结构化弹窗双缓冲渲染与视觉质感
3. Hit-test 交互热区命中精度 (双层树微观模式)
4. 快捷键层次化分发
"""

import os
import sys
import shutil
import tempfile
import unittest
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.workspace_hub.hub_state import HubState
from tools.workspace_hub.hub_renderer import (
    HubRenderer,
    GEOM_MODAL_X,
    GEOM_MODAL_Y,
    GEOM_MODAL_SAVE,
    GEOM_MODAL_CANCEL,
    GEOM_MODAL_CLOSE,
    WS_BTN_NEW_FRAME,
    FRAME_ADD_ROI_BTN,
)
from tools.workspace_hub.app import WorkspaceHubApp
from src.calibration.workspace_manager import WorkspaceManager


class TestHubFramesRoisGui(unittest.TestCase):
    def setUp(self):
        self.tmp_root = tempfile.mkdtemp(prefix="test_hub_geom_")
        self.ws_mgr = WorkspaceManager(workspaces_dir=self.tmp_root)
        self.ws = self.ws_mgr.create_workspace("现场视觉测试工位", "自动化测试工位沙盒")
        self.assertIsNotNone(self.ws)

    def tearDown(self):
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def test_hub_frames_rois_end_to_end(self):
        state = HubState(workspace_mgr=self.ws_mgr)
        renderer = HubRenderer()

        # -----------------------------------------------------------------
        # 1. 验证 Hit Test 列表元素探测 (大盘中的“新增坐标系”与微观页签中的“+ 新增 3D ROI”)
        # -----------------------------------------------------------------
        state.set_tab(HubState.TAB_REPORT)
        hit_add_frame = renderer.hit_test(WS_BTN_NEW_FRAME[0] + 10, WS_BTN_NEW_FRAME[1] + 10, state)
        self.assertEqual(hit_add_frame, "btn_add_frame")

        # 切换到微观坐标系下的 3D ROI 页签
        state.select_tree_frame(0, "world")
        state.set_tab(HubState.TAB_FRAME_ROIS)
        self.assertEqual(state.active_tab, HubState.TAB_FRAME_ROIS)

        hit_add_roi = renderer.hit_test(FRAME_ADD_ROI_BTN[0] + 10, FRAME_ADD_ROI_BTN[1] + 10, state)
        self.assertEqual(hit_add_roi, "btn_add_frame_roi")

        # -----------------------------------------------------------------
        # 2. 验证新建机构相对坐标系弹窗 & 写穿
        # -----------------------------------------------------------------
        state.geometry.open_frame_modal()
        self.assertTrue(state.geometry.frame_modal_open)
        self.assertTrue(state.geometry.frame_modal_is_new)

        # 修改表单数据
        state.geometry.frame_modal_data["frame_id"] = "frame_flange"
        state.geometry.frame_modal_data["name"] = "机械臂末端法兰"
        state.geometry.frame_modal_data["type"] = "fixed_transform"
        state.geometry.frame_modal_data["status"] = "manual"
        state.geometry.frame_modal_data["known_dof"] = [True] * 6
        state.geometry.frame_modal_data["translation_xyz_mm"] = [150.0, 30.0, -80.0]
        state.geometry.frame_modal_data["rotation_rpy_deg"] = [0.0, 45.0, 0.0]

        # Hit test 校验弹窗内部控件命中
        hit_close = renderer.hit_test(GEOM_MODAL_CLOSE[0] + 5, GEOM_MODAL_CLOSE[1] + 5, state)
        self.assertEqual(hit_close, "frame_modal_close")

        hit_save = renderer.hit_test(GEOM_MODAL_SAVE[0] + 10, GEOM_MODAL_SAVE[1] + 10, state)
        self.assertEqual(hit_save, "frame_modal_save")

        # 保存弹窗
        ok, msg = state.geometry.save_frame_modal()
        self.assertTrue(ok, f"Save frame failed: {msg}")
        self.assertFalse(state.geometry.frame_modal_open)

        # 验证底层 frames.yaml 真正写穿
        frames = state.geometry.get_coordinate_frames()
        frame_ids = [f.frame_id for f in frames]
        self.assertIn("frame_flange", frame_ids)
        saved_f = state.geometry.coord_mgr.get_frame("frame_flange")
        self.assertEqual(saved_f.translation_xyz_mm, [150.0, 30.0, -80.0])

        # -----------------------------------------------------------------
        # 3. 验证 6DoF 外参位姿与先验约束独立编辑器 (全未知/部分已知/全已知)
        # -----------------------------------------------------------------
        state.geometry.open_frame_modal("frame_flange")
        self.assertTrue(state.geometry.frame_modal_open)

        # (1) 打开 6DoF 模态窗
        state.geometry.open_pose6d_modal(0)
        self.assertTrue(state.geometry.pose6d_modal_open)
        canvas_p6 = renderer.render(state)
        self.assertEqual(canvas_p6.shape, (720, 960, 3))

        # (2) 测试一键设为全未知 (待BA平差反向求解)
        state.geometry.pose6d_set_all_unknown()
        self.assertEqual(sum(1 for b in state.geometry.pose6d_modal_known if b), 0)
        state.geometry.save_pose6d_modal()
        self.assertFalse(state.geometry.pose6d_modal_open)
        self.assertEqual(state.geometry.frame_modal_data["status"], "unknown")

        # 保存并验证底层为 unknown 且外参数值为 None，拓扑严格阻断
        ok_save_unk, _ = state.geometry.save_frame_modal()
        self.assertTrue(ok_save_unk)
        f_unk = state.geometry.coord_mgr.get_frame("frame_flange")
        self.assertEqual(f_unk.status, "unknown")
        self.assertIsNone(f_unk.translation_xyz_mm)
        self.assertIsNone(f_unk.rotation_rpy_deg)
        _, is_res_unk = state.geometry.coord_mgr.get_relative_transform_to_parent("frame_flange")
        self.assertFalse(is_res_unk)

        # (3) 测试水平面运动先验约束 (部分已知 partial 模式: Roll=0°, Pitch=0°)
        state.geometry.open_frame_modal("frame_flange")
        state.geometry.open_pose6d_modal(0)
        state.geometry.pose6d_set_planar_preset()
        self.assertTrue(state.geometry.pose6d_modal_known[3])  # Roll
        self.assertTrue(state.geometry.pose6d_modal_known[4])  # Pitch
        self.assertFalse(state.geometry.pose6d_modal_known[0]) # X 未知
        state.geometry.save_pose6d_modal()
        self.assertEqual(state.geometry.frame_modal_data["status"], "partial")
        ok_save_part, _ = state.geometry.save_frame_modal()
        self.assertTrue(ok_save_part)
        f_part = state.geometry.coord_mgr.get_frame("frame_flange")
        self.assertEqual(f_part.status, "partial")
        _, is_res_part = state.geometry.coord_mgr.get_relative_transform_to_parent("frame_flange")
        self.assertFalse(is_res_part)

        # (4) 测试软键盘输入与一键全已知 (manual 模式)
        state.geometry.open_frame_modal("frame_flange")
        state.geometry.open_pose6d_modal(0)
        state.geometry.pose6d_set_all_known()
        state.geometry.pose6d_select_axis(0)
        state.geometry.pose6d_pad_key("清空")
        for ch in "200.5":
            state.geometry.pose6d_pad_key(ch)
        state.geometry.pose6d_pad_key("确认")
        self.assertAlmostEqual(state.geometry.pose6d_modal_vals[0], 200.5, delta=1e-4)

        state.geometry.save_pose6d_modal()
        self.assertEqual(state.geometry.frame_modal_data["status"], "manual")
        ok_save_man, _ = state.geometry.save_frame_modal()
        self.assertTrue(ok_save_man)
        f_man = state.geometry.coord_mgr.get_frame("frame_flange")
        self.assertEqual(f_man.status, "manual")
        self.assertAlmostEqual(f_man.translation_xyz_mm[0], 200.5, delta=1e-4)
        _, is_res_man = state.geometry.coord_mgr.get_relative_transform_to_parent("frame_flange")
        self.assertTrue(is_res_man)

        # -----------------------------------------------------------------
        # 4. 验证新建 3D ROI 空间物件弹窗 & 强 Schema 校验
        # -----------------------------------------------------------------
        state.geometry.open_roi_modal()
        self.assertTrue(state.geometry.roi_modal_open)

        state.geometry.roi_modal_data["roi_id"] = "roi_gripper_zone"
        state.geometry.roi_modal_data["name"] = "夹爪安全作业区"
        state.geometry.roi_modal_data["frame_id"] = "frame_flange"
        state.geometry.roi_modal_data["category"] = "general"
        state.geometry.roi_modal_data["center_xyz_mm"] = [0.0, 0.0, 100.0]

        # 4.1 强 Schema 约束：尺寸 dx <= 0 应当拦截报错
        state.geometry.roi_modal_data["size_xyz_mm"] = [-10.0, 60.0, 80.0]
        ok_bad, msg_bad = state.geometry.save_roi_modal()
        self.assertFalse(ok_bad)
        self.assertIn("严格大于 0", msg_bad)
        self.assertTrue(state.geometry.roi_modal_open)

        # 4.2 修正为合法正数后保存
        state.geometry.roi_modal_data["size_xyz_mm"] = [120.0, 60.0, 80.0]
        ok_good, msg_good = state.geometry.save_roi_modal()
        self.assertTrue(ok_good, f"Save roi failed: {msg_good}")
        self.assertFalse(state.geometry.roi_modal_open)

        # 验证底层 rois.yaml 真正写穿
        rois = state.geometry.get_roi_spaces()
        roi_ids = [r.roi_id for r in rois]
        self.assertIn("roi_gripper_zone", roi_ids)
        saved_roi = state.geometry.roi_mgr.get_roi("roi_gripper_zone")
        self.assertEqual(saved_roi.size_xyz_mm, [120.0, 60.0, 80.0])
        self.assertEqual(saved_roi.frame_id, "frame_flange")

        # -----------------------------------------------------------------
        # 5. 验证下拉框 (Dropdown Box) 交互与展开浮层渲染
        # -----------------------------------------------------------------
        state.geometry.open_frame_modal("frame_flange")
        parent_trigger_x = GEOM_MODAL_X + 125
        parent_trigger_y = GEOM_MODAL_Y + 56 + 80 + 10
        hit_dd_toggle = renderer.hit_test(parent_trigger_x, parent_trigger_y, state)
        self.assertEqual(hit_dd_toggle, ("dropdown_toggle", "frame_parent"))

        state.geometry.active_dropdown = "frame_parent"
        canvas_dropdown = renderer.render(state)
        self.assertEqual(canvas_dropdown.shape, (720, 960, 3))

        hit_dismiss = renderer.hit_test(10, 10, state)
        self.assertEqual(hit_dismiss, "dropdown_dismiss")

        hit_sel = renderer.hit_test(parent_trigger_x, GEOM_MODAL_Y + 56 + 80 + 28 + 10, state)
        self.assertEqual(hit_sel, ("dropdown_select", "frame_parent", "world"))
        state.geometry.active_dropdown = None

        # -----------------------------------------------------------------
        # 6. 验证唯一 ID 修改与级联重命名 (Cascade Rename)
        # -----------------------------------------------------------------
        state.geometry.frame_modal_data["frame_id"] = "frame_flange_v2"
        ok_rename, msg_rename = state.geometry.save_frame_modal()
        self.assertTrue(ok_rename, f"Rename frame failed: {msg_rename}")
        self.assertIn("frame_flange_v2", [f.frame_id for f in state.geometry.get_coordinate_frames()])
        self.assertNotIn("frame_flange", [f.frame_id for f in state.geometry.get_coordinate_frames()])

        updated_roi = state.geometry.roi_mgr.get_roi("roi_gripper_zone")
        self.assertEqual(updated_roi.frame_id, "frame_flange_v2")

        # ROI 重命名
        state.geometry.open_roi_modal("roi_gripper_zone")
        state.geometry.roi_modal_data["roi_id"] = "roi_gripper_zone_v2"
        ok_roi_rename, msg_roi_rename = state.geometry.save_roi_modal()
        self.assertTrue(ok_roi_rename, f"Rename roi failed: {msg_roi_rename}")
        self.assertIn("roi_gripper_zone_v2", [r.roi_id for r in state.geometry.get_roi_spaces()])
        self.assertNotIn("roi_gripper_zone", [r.roi_id for r in state.geometry.get_roi_spaces()])

        # -----------------------------------------------------------------
        # 7. 删除操作写穿验证
        # -----------------------------------------------------------------
        ok_del_roi, _ = state.geometry.delete_roi("roi_gripper_zone_v2")
        self.assertTrue(ok_del_roi)
        self.assertNotIn("roi_gripper_zone_v2", [r.roi_id for r in state.geometry.get_roi_spaces()])

        # -----------------------------------------------------------------
        # 8. 验证 WorkspaceHubApp 快捷键分发与白名单/锚点更新
        # -----------------------------------------------------------------
        app = WorkspaceHubApp(workspace_mgr=self.ws_mgr)
        app.state.current_ws_id = self.ws.workspace_id

        # 8.1 标注 Tag #05 局部坐标并自动放行
        ok_tag, _ = app.state.whitelist.update_tag_anchor(5, {"xyz_mm": [12.5, -45.0, 100.0], "known": [True, True, True]})
        self.assertTrue(ok_tag)
        wl_data = app.state.whitelist.get_tag_whitelist()
        self.assertIn(5, wl_data.get("allowed_ids", []))
        self.assertEqual(wl_data.get("tag_anchors", {}).get(5), {
            "xyz_mm": [12.5, -45.0, 100.0],
            "known": [True, True, True]
        })

        # 清除 Tag #05 坐标标注
        ok_clr, _ = app.state.whitelist.update_tag_anchor(5, None)
        self.assertTrue(ok_clr)
        wl_data_clr = app.state.whitelist.get_tag_whitelist()
        self.assertNotIn(5, wl_data_clr.get("tag_anchors", {}))

        # 8.2 专用坐标编辑弹窗
        app.state.whitelist.open_anchor_editor(5)
        self.assertTrue(app.state.whitelist.anchor_modal_open)
        self.assertEqual(app.state.whitelist.anchor_axis_sel, 0)
        app.on_key(9)  # Tab
        self.assertEqual(app.state.whitelist.anchor_axis_sel, 1)
        app.on_key(ord('?'))
        self.assertFalse(app.state.whitelist.anchor_modal_known[1])
        app.on_key(27)  # ESC
        self.assertFalse(app.state.whitelist.anchor_modal_open)

        # 8.3 快捷键 on_key 层次化退出测试
        # (1) 大图展开模式下按 ESC
        app.state.gallery.set_view_mode(HubState.VIEW_EXPANDED)
        handled_esc = app.on_key(27)
        self.assertTrue(handled_esc)
        self.assertEqual(app.state.gallery.view_mode, HubState.VIEW_STANDARD)

        # (2) 坐标系弹窗下按 ESC
        app.state.geometry.open_frame_modal()
        self.assertTrue(app.state.geometry.frame_modal_open)
        handled_esc = app.on_key(27)
        self.assertTrue(handled_esc)
        self.assertFalse(app.state.geometry.frame_modal_open)

        # (3) 3D ROI 弹窗下按 Enter 保存
        app.state.geometry.open_roi_modal()
        app.state.geometry.roi_modal_data["roi_id"] = "roi_shortcut_test"
        app.state.geometry.roi_modal_data["name"] = "快捷键测试物件"
        handled_enter = app.on_key(13)
        self.assertTrue(handled_enter)
        self.assertFalse(app.state.geometry.roi_modal_open)
        self.assertIn("roi_shortcut_test", [r.roi_id for r in app.state.geometry.get_roi_spaces()])

        # (4) 标靶物理边长弹窗与保存
        app.state.whitelist.open_marker_size_editor()
        self.assertTrue(app.state.whitelist.marker_size_modal_open)
        app.on_key(ord('c'))
        for ch in "42.125":
            app.on_key(ord(ch))
        self.assertEqual(app.state.whitelist.marker_size_buf, "42.125")
        handled_enter = app.on_key(13)
        self.assertTrue(handled_enter)
        self.assertFalse(app.state.whitelist.marker_size_modal_open)
        self.assertEqual(app.state.whitelist.get_workspace_marker_size(), 42.125)

        # (5) 无任何弹窗激活时按 ESC 返回 False
        handled_esc = app.on_key(27)
        self.assertFalse(handled_esc)


if __name__ == "__main__":
    unittest.main()
