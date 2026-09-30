import os
import sys
import shutil
import tempfile
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.workspace.workspace_manager import WorkspaceManager
from src.workspace.coordinate_manager import FrameDefinition
from src.workspace.roi_manager import RoiDefinition
from tools.workspace_hub.hub_state import HubState
from tools.workspace_hub.hub_renderer import (
    HubRenderer, FRAME_ROI_PREV_BTN, FRAME_ROI_NEXT_BTN,
    FRAME_ROI_SCROLL_TRACK, frame_roi_edit_btn, frame_roi_del_btn
)
from tools.workspace_hub.hub_hit_tester import HubHitTester


def test_roi_scrolling_and_interaction():
    temp_dir = tempfile.mkdtemp(prefix="test_roi_scroll_")
    try:
        ws_mgr = WorkspaceManager(workspaces_dir=temp_dir)
        ws = ws_mgr.create_workspace("TestStation", "测试工位")
        assert ws is not None, "创建工位失败"

        state = HubState(workspace_mgr=ws_mgr)
        renderer = HubRenderer()
        hit_tester = HubHitTester(renderer)

        # 确保初始化选中
        state.select_tree_workspace(0)
        state.coord_mgr.add_frame(FrameDefinition(
            frame_id="frame_test",
            name="测试机构",
            parent_frame_id="world",
            type="fixed_transform",
            translation_xyz_mm=[100.0, 0.0, 0.0],
            rotation_rpy_deg=[0.0, 0.0, 0.0]
        ))
        state.coord_mgr.save()

        # 添加 12 个 3D ROI 物件到 frame_test 坐标系
        for i in range(12):
            roi = RoiDefinition(
                roi_id=f"roi_item_{i:02d}",
                name=f"物件_{i+1}",
                category="general",
                frame_id="frame_test",
                center_xyz_mm=[100.0 * i, 50.0, 20.0],
                size_xyz_mm=[60.0, 40.0, 30.0],
                rotation_rpy_deg=[0.0, 0.0, float(i * 5)]
            )
            state.roi_mgr.add_roi(roi)
        state.roi_mgr.save()

        # 切换到 frame_test 坐标系的 TAB_FRAME_ROIS
        state.select_tree_frame(0, "frame_test")
        state.set_tab(HubState.TAB_FRAME_ROIS)

        rois = state.get_frame_rois("frame_test")
        assert len(rois) == 12, f"预期 12 个 ROI，实际 {len(rois)}"
        assert state.roi_scroll_offset == 0

        # 1. 验证首页渲染 (未滚动状态)
        canvas = renderer.render(state)
        assert isinstance(canvas, np.ndarray)
        assert canvas.shape == (720, 960, 3)

        # 2. 验证第一页命中测试
        edit_0_hit = hit_tester.hit_test(frame_roi_edit_btn(0)[0] + 5, frame_roi_edit_btn(0)[1] + 5, state)
        assert edit_0_hit == ("frame_roi_edit", "roi_item_00"), f"实际为: {edit_0_hit}"

        # 3. 验证下翻按钮命中
        next_btn_hit = hit_tester.hit_test(FRAME_ROI_NEXT_BTN[0] + 10, FRAME_ROI_NEXT_BTN[1] + 10, state)
        assert next_btn_hit == "btn_roi_next", f"实际为: {next_btn_hit}"

        # 4. 模拟滚动下翻 2 个物件
        state.scroll_roi_list(2)
        assert state.roi_scroll_offset == 2

        # 渲染滚动后界面
        canvas_scrolled = renderer.render(state)
        assert isinstance(canvas_scrolled, np.ndarray)

        # 现在第 0 行卡片显示的是 roi_item_02
        edit_scrolled_hit = hit_tester.hit_test(frame_roi_edit_btn(0)[0] + 5, frame_roi_edit_btn(0)[1] + 5, state)
        assert edit_scrolled_hit == ("frame_roi_edit", "roi_item_02"), f"实际为: {edit_scrolled_hit}"

        # 5. 验证滚动边界保护 (max_offset = 12 - 7 = 5)
        state.scroll_roi_list(100)
        assert state.roi_scroll_offset == 5, f"夹紧失败，预期 5，实际 {state.roi_scroll_offset}"

        # 上翻回到顶部
        state.scroll_roi_list(-100)
        assert state.roi_scroll_offset == 0

        # 6. 验证滚动条轨道点击快速跳转
        track_hit = hit_tester.hit_test(FRAME_ROI_SCROLL_TRACK[0] + 2, FRAME_ROI_SCROLL_TRACK[1] + 200, state)
        assert isinstance(track_hit, tuple) and track_hit[0] == "roi_scrollbar_click"

        state.jump_roi_scroll_by_y(FRAME_ROI_SCROLL_TRACK[1] + 276) # 中间位置
        assert 1 <= state.roi_scroll_offset <= 5

        # 7. 验证删除物件后自动边界保护
        state.delete_roi("roi_item_11")
        assert len(state.get_frame_rois("frame_test")) == 11
        assert state.roi_scroll_offset <= 11 - 7

        print("=== 所有 3D ROI 滚动、布局与命中测试 100% 成功通过 ===")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_roi_scrolling_and_interaction()
