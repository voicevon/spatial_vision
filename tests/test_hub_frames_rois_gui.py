# -*- coding: utf-8 -*-
"""
Workspace Hub 坐标系与 3D ROI GUI 交互端到端自动化测试
验证：
1. HubState 弹窗生命周期、强 Schema 校验与 YAML 写穿
2. HubRenderer 结构化弹窗双缓冲渲染与视觉质感
3. Hit-test 交互热区命中精度
"""

import os
import sys
import shutil
import tempfile
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
    frame_btn_add_rect,
    roi_btn_add_rect,
    frame_row_edit_rect,
    roi_row_edit_rect,
)
from src.calibration.workspace_manager import WorkspaceManager


def test_hub_frames_rois_end_to_end():
    tmp_root = tempfile.mkdtemp(prefix="test_hub_geom_")
    try:
        # 1. 准备沙盒工位
        ws_mgr = WorkspaceManager(workspaces_dir=tmp_root)
        ws = ws_mgr.create_workspace("现场视觉测试工位", "自动化测试工位沙盒")
        assert ws is not None

        state = HubState(workspace_mgr=ws_mgr)
        renderer = HubRenderer()

        # 切换到【坐标系&ROI】页签
        state.set_tab(HubState.TAB_FRAMES_ROIS)
        assert state.active_tab == HubState.TAB_FRAMES_ROIS

        # -----------------------------------------------------------------
        # 2. 验证 Hit Test 列表元素探测
        # -----------------------------------------------------------------
        # 新增坐标系按钮
        bx, by, bw, bh = frame_btn_add_rect()
        hit = renderer.hit_test(bx + 10, by + 10, state)
        assert hit == "btn_add_frame", f"Expected btn_add_frame, got {hit}"

        # 新增 ROI 按钮
        rx, ry, rw, rh = roi_btn_add_rect()
        hit = renderer.hit_test(rx + 10, ry + 10, state)
        assert hit == "btn_add_roi", f"Expected btn_add_roi, got {hit}"

        # -----------------------------------------------------------------
        # 3. 验证新建机构相对坐标系弹窗 & 写穿
        # -----------------------------------------------------------------
        state.open_frame_modal()
        assert state.frame_modal_open is True
        assert state.frame_modal_is_new is True

        # 修改表单数据
        state.frame_modal_data["frame_id"] = "frame_flange"
        state.frame_modal_data["name"] = "机械臂末端法兰"
        state.frame_modal_data["type"] = "fixed_transform"
        state.frame_modal_data["status"] = "manual"
        state.frame_modal_data["known_dof"] = [True] * 6
        state.frame_modal_data["translation_xyz_mm"] = [150.0, 30.0, -80.0]
        state.frame_modal_data["rotation_rpy_deg"] = [0.0, 45.0, 0.0]

        # Hit test 校验弹窗内部控件命中
        hit_close = renderer.hit_test(GEOM_MODAL_CLOSE[0] + 5, GEOM_MODAL_CLOSE[1] + 5, state)
        assert hit_close == "frame_modal_close", f"Got {hit_close}"

        hit_save = renderer.hit_test(GEOM_MODAL_SAVE[0] + 10, GEOM_MODAL_SAVE[1] + 10, state)
        assert hit_save == "frame_modal_save", f"Got {hit_save}"

        # 保存弹窗
        ok, msg = state.save_frame_modal()
        assert ok is True, f"Save frame failed: {msg}"
        assert state.frame_modal_open is False

        # 验证底层 frames.yaml 真正写穿
        frames = state.get_coordinate_frames()
        frame_ids = [f.frame_id for f in frames]
        assert "frame_flange" in frame_ids
        saved_f = state.coord_mgr.get_frame("frame_flange")
        assert saved_f.translation_xyz_mm == [150.0, 30.0, -80.0]

        # -----------------------------------------------------------------
        # 3.5 验证 6DoF 外参位姿与先验约束独立编辑器 (全未知/部分已知/全已知)
        # -----------------------------------------------------------------
        state.open_frame_modal("frame_flange")
        assert state.frame_modal_open is True

        # (1) 打开 6DoF 模态窗
        state.open_pose6d_modal(0)
        assert state.pose6d_modal_open is True
        canvas_p6 = renderer.render(state)
        assert canvas_p6.shape == (720, 960, 3)

        # (2) 测试一键设为全未知 (待BA平差反向求解)
        state.pose6d_set_all_unknown()
        assert sum(1 for b in state.pose6d_modal_known if b) == 0
        state.save_pose6d_modal()
        assert state.pose6d_modal_open is False
        assert state.frame_modal_data["status"] == "unknown"

        # 保存并验证底层为 unknown 且外参数值为 None，拓扑严格阻断
        ok_save_unk, _ = state.save_frame_modal()
        assert ok_save_unk is True
        f_unk = state.coord_mgr.get_frame("frame_flange")
        assert f_unk.status == "unknown"
        assert f_unk.translation_xyz_mm is None
        assert f_unk.rotation_rpy_deg is None
        _, is_res_unk = state.coord_mgr.get_relative_transform_to_parent("frame_flange")
        assert is_res_unk is False

        # (3) 测试水平面运动先验约束 (部分已知 partial 模式: Roll=0°, Pitch=0°)
        state.open_frame_modal("frame_flange")
        state.open_pose6d_modal(0)
        state.pose6d_set_planar_preset()
        assert state.pose6d_modal_known[3] is True  # Roll
        assert state.pose6d_modal_known[4] is True  # Pitch
        assert state.pose6d_modal_known[0] is False # X 未知
        state.save_pose6d_modal()
        assert state.frame_modal_data["status"] == "partial"
        ok_save_part, _ = state.save_frame_modal()
        assert ok_save_part is True
        f_part = state.coord_mgr.get_frame("frame_flange")
        assert f_part.status == "partial"
        _, is_res_part = state.coord_mgr.get_relative_transform_to_parent("frame_flange")
        assert is_res_part is False  # 部分已知在BA平差前仍然保持未解阻断

        # (4) 测试软键盘输入与一键全已知 (manual 模式)
        state.open_frame_modal("frame_flange")
        state.open_pose6d_modal(0)
        state.pose6d_set_all_known()
        # 选中 X 轴并用软键盘输入 200.5
        state.pose6d_select_axis(0)
        state.pose6d_pad_key("清空")
        for ch in "200.5":
            state.pose6d_pad_key(ch)
        state.pose6d_pad_key("确认")
        assert abs(state.pose6d_modal_vals[0] - 200.5) < 1e-4

        state.save_pose6d_modal()
        assert state.frame_modal_data["status"] == "manual"
        ok_save_man, _ = state.save_frame_modal()
        assert ok_save_man is True
        f_man = state.coord_mgr.get_frame("frame_flange")
        assert f_man.status == "manual"
        assert abs(f_man.translation_xyz_mm[0] - 200.5) < 1e-4
        _, is_res_man = state.coord_mgr.get_relative_transform_to_parent("frame_flange")
        assert is_res_man is True  # 全已知正常放行

        # -----------------------------------------------------------------
        # 4. 验证新建 3D ROI 空间物件弹窗 & 强 Schema 校验
        # -----------------------------------------------------------------

        state.open_roi_modal()
        assert state.roi_modal_open is True

        state.roi_modal_data["roi_id"] = "roi_gripper_zone"
        state.roi_modal_data["name"] = "夹爪安全作业区"
        state.roi_modal_data["frame_id"] = "frame_flange"
        state.roi_modal_data["category"] = "general"
        state.roi_modal_data["center_xyz_mm"] = [0.0, 0.0, 100.0]

        # 4.1 强 Schema 约束：尺寸 dx <= 0 应当拦截报错
        state.roi_modal_data["size_xyz_mm"] = [-10.0, 60.0, 80.0]
        ok_bad, msg_bad = state.save_roi_modal()
        assert ok_bad is False
        assert "严格大于 0" in msg_bad
        assert state.roi_modal_open is True  # 弹窗保持打开以供修改

        # 4.2 修正为合法正数后保存
        state.roi_modal_data["size_xyz_mm"] = [120.0, 60.0, 80.0]
        ok_good, msg_good = state.save_roi_modal()
        assert ok_good is True, f"Save roi failed: {msg_good}"
        assert state.roi_modal_open is False

        # 验证底层 rois.yaml 真正写穿
        rois = state.get_roi_spaces()
        roi_ids = [r.roi_id for r in rois]
        assert "roi_gripper_zone" in roi_ids
        saved_roi = state.roi_mgr.get_roi("roi_gripper_zone")
        assert saved_roi.size_xyz_mm == [120.0, 60.0, 80.0]
        assert saved_roi.frame_id == "frame_flange"

        # -----------------------------------------------------------------
        # 5. 验证下拉框 (Dropdown Box) 交互与展开浮层渲染
        # -----------------------------------------------------------------
        state.open_frame_modal("frame_flange")
        # 5.1 点击父坐标系下拉触发条
        parent_trigger_x = GEOM_MODAL_X + 125
        parent_trigger_y = GEOM_MODAL_Y + 56 + 80 + 10
        hit_dd_toggle = renderer.hit_test(parent_trigger_x, parent_trigger_y, state)
        assert hit_dd_toggle == ("dropdown_toggle", "frame_parent"), f"Got {hit_dd_toggle}"

        # 展开下拉菜单
        state.active_dropdown = "frame_parent"
        # 渲染展开下拉浮层画面
        canvas_dropdown = renderer.render(state)
        cv2.imwrite(os.path.join(PROJECT_ROOT, "data", "test_frame_dropdown.png"), canvas_dropdown)

        # 5.2 浮层点击外部应触发 dismiss
        hit_dismiss = renderer.hit_test(10, 10, state)
        assert hit_dismiss == "dropdown_dismiss", f"Got {hit_dismiss}"

        # 5.3 浮层内点击第一项 [world] 触发 select
        hit_sel = renderer.hit_test(parent_trigger_x, GEOM_MODAL_Y + 56 + 80 + 28 + 10, state)
        assert hit_sel == ("dropdown_select", "frame_parent", "world"), f"Got {hit_sel}"
        state.active_dropdown = None

        # -----------------------------------------------------------------
        # 6. 验证唯一 ID 修改与级联重命名 (Cascade Rename)
        # -----------------------------------------------------------------
        # 将 frame_flange 唯一 ID 重命名为 frame_flange_v2
        state.frame_modal_data["frame_id"] = "frame_flange_v2"
        ok_rename, msg_rename = state.save_frame_modal()
        assert ok_rename is True, f"Rename frame failed: {msg_rename}"
        assert "frame_flange_v2" in [f.frame_id for f in state.get_coordinate_frames()]
        assert "frame_flange" not in [f.frame_id for f in state.get_coordinate_frames()]

        # 验证关联的 ROI 的 frame_id 已经级联自动更新为 frame_flange_v2
        updated_roi = state.roi_mgr.get_roi("roi_gripper_zone")
        assert updated_roi.frame_id == "frame_flange_v2", f"Cascade update failed: {updated_roi.frame_id}"

        # 验证 ROI 唯一 ID 重命名
        state.open_roi_modal("roi_gripper_zone")
        state.roi_modal_data["roi_id"] = "roi_gripper_zone_v2"
        ok_roi_rename, msg_roi_rename = state.save_roi_modal()
        assert ok_roi_rename is True, f"Rename roi failed: {msg_roi_rename}"
        assert "roi_gripper_zone_v2" in [r.roi_id for r in state.get_roi_spaces()]
        assert "roi_gripper_zone" not in [r.roi_id for r in state.get_roi_spaces()]

        # -----------------------------------------------------------------
        # 7. 渲染画布截图保存验证
        # -----------------------------------------------------------------
        os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)

        # 7.1 渲染列表看板画面
        canvas_list = renderer.render(state)
        cv2.imwrite(os.path.join(PROJECT_ROOT, "data", "test_tab_frames_rois_list.png"), canvas_list)

        # 7.2 渲染编辑 Frame 弹窗画面
        state.open_frame_modal("frame_flange_v2")
        canvas_frame_modal = renderer.render(state)
        cv2.imwrite(os.path.join(PROJECT_ROOT, "data", "test_frame_modal.png"), canvas_frame_modal)
        state.close_frame_modal()

        # 7.3 渲染编辑 ROI 弹窗画面
        state.open_roi_modal("roi_gripper_zone_v2")
        canvas_roi_modal = renderer.render(state)
        cv2.imwrite(os.path.join(PROJECT_ROOT, "data", "test_roi_modal.png"), canvas_roi_modal)
        state.close_roi_modal()

        # 7.4 渲染标靶位姿与坐标系页面 (frame_pose_tags) 包含 unknown / partial 外参
        state.current_page = "frame_pose_tags"
        canvas_pose_tags = renderer.render(state)
        assert canvas_pose_tags is not None
        cv2.imwrite(os.path.join(PROJECT_ROOT, "data", "test_frame_pose_tags_page.png"), canvas_pose_tags)
        state.current_page = "frames_rois"

        # -----------------------------------------------------------------
        # 8. 删除操作写穿验证
        # -----------------------------------------------------------------
        # 删除 ROI
        ok_del_roi, _ = state.delete_roi("roi_gripper_zone_v2")
        assert ok_del_roi is True
        assert "roi_gripper_zone_v2" not in [r.roi_id for r in state.get_roi_spaces()]

        # -----------------------------------------------------------------
        # 9. 验证 WorkspaceHubApp 快捷键分发与白名单/锚点下沉更新
        # -----------------------------------------------------------------
        from tools.workspace_hub.app import WorkspaceHubApp
        app = WorkspaceHubApp(workspace_mgr=ws_mgr)
        app.state.current_ws_id = ws.workspace_id

        # 9.1 白名单与物理锚点下沉验证
        wl_path = ws_mgr.ensure_tag_whitelist(ws.workspace_id)
        assert os.path.exists(wl_path)

        # 标注 Tag #05 局部坐标并自动放行 (统一标准结构)
        ok_tag, msg_tag = app.state.update_tag_anchor(5, {"xyz_mm": [12.5, -45.0, 100.0], "known": [True, True, True]})
        assert ok_tag is True
        wl_data = app.state.get_whitelist_data()
        assert 5 in wl_data.get("allowed_ids", [])
        assert wl_data.get("tag_anchors", {}).get(5) == {
            "xyz_mm": [12.5, -45.0, 100.0],
            "known": [True, True, True]
        }

        # 清除 Tag #05 坐标标注
        ok_clr, _ = app.state.update_tag_anchor(5, None)
        assert ok_clr is True
        wl_data_clr = app.state.get_whitelist_data()
        assert 5 not in wl_data_clr.get("tag_anchors", {})

        # 9.2 专用坐标编辑弹窗 (无 Windows 输入框) 与 on_key 测试
        app.state.open_anchor_editor(5)
        assert app.state.anchor_modal_open is True
        # 按 Tab 切换轴
        assert app.state.anchor_axis_sel == 0
        app.on_key(9)
        assert app.state.anchor_axis_sel == 1
        # 输入 '?' 设为未知
        app.on_key(ord('?'))
        assert app.state.anchor_modal_known[1] is False
        # 按 ESC 取消弹窗
        app.on_key(27)
        assert app.state.anchor_modal_open is False

        # 9.3 快捷键 on_key 层次化退出测试
        # (1) 大图展开模式下按 ESC
        app.state.set_view_mode(HubState.VIEW_EXPANDED)
        handled_esc = app.on_key(27)
        assert handled_esc is True
        assert app.state.view_mode == HubState.VIEW_STANDARD

        # (2) 坐标系弹窗下按 ESC
        app.state.open_frame_modal()
        assert app.state.frame_modal_open is True
        handled_esc = app.on_key(27)
        assert handled_esc is True
        assert app.state.frame_modal_open is False

        # (3) 3D ROI 弹窗下按 Enter 保存
        app.state.open_roi_modal()
        app.state.roi_modal_data["roi_id"] = "roi_shortcut_test"
        app.state.roi_modal_data["name"] = "快捷键测试物件"
        handled_enter = app.on_key(13)
        assert handled_enter is True
        assert app.state.roi_modal_open is False
        assert "roi_shortcut_test" in [r.roi_id for r in app.state.get_roi_spaces()]

        # (4) 白名单编辑态下按 ESC
        app.state.enter_whitelist_edit()
        assert app.state.whitelist_edit_mode is True
        handled_esc = app.on_key(27)
        assert handled_esc is True
        assert app.state.whitelist_edit_mode is False

        # 9.4 标靶物理边长 (Tag 公共物理属性) 中间卡片与专属弹窗测试
        app.state.set_tab(HubState.TAB_FRAME_POSE_TAGS)
        hit_sz_btn = app.renderer.hit_test(810, 248, app.state)
        assert hit_sz_btn == "btn_edit_marker_size"

        app.state.open_marker_size_editor()
        assert app.state.marker_size_modal_open is True
        # 清空并键盘键入 42.125
        app.on_key(ord('c'))
        for ch in "42.125":
            app.on_key(ord(ch))
        assert app.state.marker_size_buf == "42.125"
        # Enter 提交保存
        handled_enter = app.on_key(13)
        assert handled_enter is True
        assert app.state.marker_size_modal_open is False
        assert app.state.get_workspace_marker_size() == 42.125
        assert app.state.whitelist_edit_mode is False

        # (5) 无任何弹窗激活时按 ESC 返回 False
        handled_esc = app.on_key(27)
        assert handled_esc is False

        print("√ 全部 Workspace Hub 坐标系与 3D ROI GUI 端到端测试均 100% 通过！")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    test_hub_frames_rois_end_to_end()
