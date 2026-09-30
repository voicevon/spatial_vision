# -*- coding: utf-8 -*-
"""
Workspace Hub 两层树导航与坐标系专属 Tag 分段矩阵端到端自动化测试
验证：
1. 左侧两层树导航 (工位一级节点折叠展开，坐标系二级子节点)
2. 右侧上下文联动 (工位级大盘 3 Tab 与 坐标系专属 2 Tab)
3. Tag ID 分段与专属 10-Slot 芯片矩阵放行保存
4. 专属 3D ROI 物件列表与渲染
5. 渲染输出高清验收截图
"""

import os
import sys
import tempfile
import shutil
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.workspace_hub.hub_state import HubState
from tools.workspace_hub.hub_renderer import HubRenderer, frame_tag_chip_rect
from src.workspace.workspace_manager import WorkspaceManager
from src.workspace.coordinate_manager import FrameDefinition
from src.workspace.roi_manager import RoiDefinition


def test_two_level_tree_and_tag_partitioning():
    tmp_root = tempfile.mkdtemp(prefix="test_tree_hub_")
    try:
        # 1. 初始化沙盒环境与工位
        ws_mgr = WorkspaceManager(workspaces_dir=tmp_root)
        ws = ws_mgr.create_workspace("1号总装工作站", "视觉检测主工位")
        assert ws is not None

        state = HubState(workspace_mgr=ws_mgr)
        renderer = HubRenderer()

        # 添加一个自定义机构相对坐标系
        f_conv = FrameDefinition(
            frame_id="frame_conveyor",
            name="传送带主机构",
            parent_frame_id="world",
            type="fixed_transform",
            translation_xyz_mm=[500.0, 0.0, 120.0],
            rotation_rpy_deg=[0.0, 0.0, 90.0]
        )
        state.coord_mgr.add_frame(f_conv)
        state.coord_mgr.save()

        # 添加一个属于传送带的 ROI 和一个属于 world 的 ROI
        r_belt = RoiDefinition(
            roi_id="roi_belt_detect",
            name="同步带工作面",
            category="belt",
            frame_id="frame_conveyor",
            center_xyz_mm=[0.0, 0.0, 10.0],
            size_xyz_mm=[600.0, 200.0, 50.0]
        )
        r_world = RoiDefinition(
            roi_id="roi_station_boundary",
            name="工位安全干涉边界",
            category="general",
            frame_id="world",
            center_xyz_mm=[0.0, 0.0, 500.0],
            size_xyz_mm=[1000.0, 1000.0, 800.0]
        )
        state.roi_mgr.add_roi(r_belt)
        state.roi_mgr.add_roi(r_world)
        state.roi_mgr.save()

        # -------------------------------------------------------------
        # 2. 验证左侧两层树导航布局与展开/折叠
        # -------------------------------------------------------------
        # 初始状态未展开
        tree_items_collapsed = renderer._get_tree_layout(state)
        ws_items = [it for it in tree_items_collapsed if it["type"] == "workspace"]
        frame_items = [it for it in tree_items_collapsed if it["type"] == "frame"]
        assert len(ws_items) == 1
        assert len(frame_items) == 0, "折叠状态下不应渲染坐标系子节点"

        # 展开工位节点
        state.toggle_workspace_expanded(ws.workspace_id)
        assert ws.workspace_id in state.expanded_workspaces

        tree_items_expanded = renderer._get_tree_layout(state)
        frame_items = [it for it in tree_items_expanded if it["type"] == "frame"]
        assert len(frame_items) == 2, f"展开后应渲染 world 和 frame_conveyor，实际获取 {len(frame_items)}"
        assert frame_items[0]["frame_id"] == "world"
        assert frame_items[1]["frame_id"] == "frame_conveyor"

        # -------------------------------------------------------------
        # 3. 验证点击上下文联动 (工位级 vs 坐标系级 Tab 切换)
        # -------------------------------------------------------------
        # 3.1 初始为工位选中
        assert state.selected_tree_item[0] == "workspace"
        tabs_ws = state.get_current_tabs()
        assert len(tabs_ws) == 3
        assert [t[0] for t in tabs_ws] == [HubState.TAB_REPORT, HubState.TAB_CALIB_IMAGES, HubState.TAB_PROD_IMAGES]

        # 渲染工位大盘截图
        canvas_ws = renderer.render(state)
        assert canvas_ws.shape == (720, 960, 3)

        # 3.2 切换选中坐标系子节点: world
        state.select_tree_frame(0, "world")
        assert state.selected_tree_item == ("frame", 0, "world")
        assert state.active_tab == HubState.TAB_FRAME_POSE_TAGS
        tabs_frame = state.get_current_tabs()
        assert len(tabs_frame) == 2
        assert [t[0] for t in tabs_frame] == [HubState.TAB_FRAME_POSE_TAGS, HubState.TAB_FRAME_ROIS]

        # 3.3 切换选中坐标系子节点: frame_conveyor
        state.select_tree_frame(0, "frame_conveyor")
        assert state.selected_tree_item == ("frame", 0, "frame_conveyor")
        cur_f = state.get_selected_frame()
        assert cur_f.frame_id == "frame_conveyor"

        # -------------------------------------------------------------
        # 4. 验证 Tag ID 分段规格与专属放行保存
        # -------------------------------------------------------------
        # world 坐标系区间必须为 0~9
        range_world = state.get_frame_tag_range("world")
        assert range_world == list(range(0, 10))

        # frame_conveyor (第 2 个坐标系，index 1) 区间必须为 10~19
        range_conv = state.get_frame_tag_range("frame_conveyor")
        assert range_conv == list(range(10, 20))

        # 在 frame_conveyor 下勾选放行 Tag #12 与 Tag #15
        state.toggle_frame_tag_allowed("frame_conveyor", 12)
        state.toggle_frame_tag_allowed("frame_conveyor", 15)

        conv_tags = state.get_frame_tags_status("frame_conveyor")
        assert 12 in conv_tags and 15 in conv_tags
        assert len(conv_tags) == 2

        # 在 world 坐标系下勾选放行 Tag #02
        state.toggle_frame_tag_allowed("world", 2)
        world_tags = state.get_frame_tags_status("world")
        assert world_tags == [2]

        # 验证工位底层 whitelist 并集保存
        import yaml
        wl_path = os.path.join(ws.workspace_dir, "tag_whitelist.yaml")
        assert os.path.isfile(wl_path)
        with open(wl_path, "r", encoding="utf-8") as f:
            saved_wl = yaml.safe_load(f)
        assert saved_wl["allowed_ids"] == [2, 12, 15]

        # -------------------------------------------------------------
        # 5. 验证坐标系专属 3D ROI 隔离筛选
        # -------------------------------------------------------------
        conv_rois = state.get_frame_rois("frame_conveyor")
        assert len(conv_rois) == 1
        assert conv_rois[0].roi_id == "roi_belt_detect"

        world_rois = state.get_frame_rois("world")
        assert len(world_rois) == 1
        assert world_rois[0].roi_id == "roi_station_boundary"

        # -------------------------------------------------------------
        # 6. 渲染多页面并生成验收截图
        # -------------------------------------------------------------
        # 截图 A: 工位大盘视图 (展开树 + 工位选中)
        state.select_tree_workspace(0)
        img_ws_dashboard = renderer.render(state).copy()

        # 截图 B: 坐标系视图 - 机构参数与 Tag 分段矩阵 (选中 frame_conveyor)
        state.select_tree_frame(0, "frame_conveyor")
        state.set_tab(HubState.TAB_FRAME_POSE_TAGS)
        img_frame_tags = renderer.render(state).copy()

        # 截图 C: 坐标系视图 - 3D ROI 空间物件
        state.set_tab(HubState.TAB_FRAME_ROIS)
        img_frame_rois = renderer.render(state).copy()

        # 验证 tag_bound 动标绑定类型坐标系的渲染健壮性
        f_tag = FrameDefinition(
            frame_id="frame_tag_slider",
            name="滑块动标坐标系",
            parent_frame_id="world",
            type="tag_bound",
            tag_id=18,
            offset_xyz_mm=[10.0, 20.0, 30.0]
        )
        state.coord_mgr.add_frame(f_tag)
        state.coord_mgr.save()
        state.select_tree_frame("frame_tag_slider")
        state.set_tab(HubState.TAB_FRAME_POSE_TAGS)
        img_tag_bound = renderer.render(state)
        assert img_tag_bound.shape == (720, 960, 3)

        # 保存到成果目录
        out_dir = r"C:\Users\feng\.gemini\antigravity-ide\brain\143366f2-221b-4f9e-8fee-7678b4b64745"
        cv2.imwrite(os.path.join(out_dir, "test_view_ws_dashboard.png"), img_ws_dashboard)
        cv2.imwrite(os.path.join(out_dir, "test_view_frame_tags.png"), img_frame_tags)
        cv2.imwrite(os.path.join(out_dir, "test_view_frame_rois.png"), img_frame_rois)

        print("=== 所有自动化测试断言均 100% 成功通过！截图已保存至成果目录 ===")

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    test_two_level_tree_and_tag_partitioning()
