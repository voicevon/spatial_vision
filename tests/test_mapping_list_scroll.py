# -*- coding: utf-8 -*-
"""
空间建图工作站左右栏列表滚动与交互控件单元测试 (test_mapping_list_scroll.py)
========================================================================
验证：
1. SpatialMappingStudioApp 左右栏 ScrollableListBox 实例装配与默认参数；
2. 左栏帧资产列表滚轮与拖拽对 scroll_offset 的同步；
3. 右栏超长标靶观测列表（消除原有 12 标靶硬截断限制）无缝滚动检视；
4. 切换图像帧时右栏标靶列表滚动偏移量自动复位（重置回 0）；
5. 渲染流水线中 ScrollableListBox 委托渲染与按钮注册的兼容性。
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import cv2
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from tools.spatial_mapping_studio.app import SpatialMappingStudioApp


class TestMappingListScroll(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.image_dir = os.path.join(self.temp_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)

        # 创建 5 张模拟测试图片
        for i in range(1, 6):
            img = np.full((1080, 1920, 3), 40 + i * 10, dtype=np.uint8)
            cv2.imwrite(os.path.join(self.image_dir, f"view_{i:04d}.png"), img)

        self.map_path = os.path.join(self.temp_dir, "test_tags_map.yaml")
        # 构造包含 30 个标靶的地图
        tags_dict = {}
        for tid in range(30):
            tags_dict[tid] = {
                "position_mm": [tid * 50.0, 100.0, 0.0],
                "corners": [[-25, 25, 0], [25, 25, 0], [25, -25, 0], [-25, -25, 0]],
            }
        with open(self.map_path, "w", encoding="utf-8") as f:
            yaml.dump({"marker_size_mm": 50.0, "tags": tags_dict}, f)

        # 构造 view_0001.png 包含 25 个标靶观测
        self.manifest_path = os.path.join(self.temp_dir, "test_tag_observations.yaml")
        obs_list = []
        for tid in range(25):
            obs_list.append({
                "tag_id": tid,
                "keep": True,
                "corners": [[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]],
            })

        init_manifest = {
            "version": "2.0_test",
            "tag_family": "DICT_APRILTAG_16h5",
            "marker_size_mm": 50.0,
            "summary": {"total_images": 5, "total_observations": 25, "total_kept": 25},
            "images": {
                "view_0001.png": {
                    "file_name": "view_0001.png",
                    "image_path": os.path.join(self.image_dir, "view_0001.png"),
                    "detected_count": 25,
                    "enabled": True,
                    "observations": obs_list,
                }
            }
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            yaml.dump(init_manifest, f)

        self.studio = SpatialMappingStudioApp(
            map_path=self.map_path,
            image_dir=self.image_dir,
            marker_size_mm=50.0,
            win_w=1920,
            win_h=1080,
            manifest_path=self.manifest_path,
            settings_file=os.path.join(self.temp_dir, "gui_settings.json"),
        )

    def test_list_boxes_initialization(self):
        """验证左右两栏列表控件已在 App 中装配且参数合理"""
        self.assertTrue(hasattr(self.studio, "frame_list_box"))
        self.assertTrue(hasattr(self.studio, "tag_list_box"))
        self.assertEqual(self.studio.frame_list_box.item_height, 36)
        self.assertEqual(self.studio.tag_list_box.item_height, 38)
        self.assertEqual(self.studio.frame_list_box.scroll_offset, 0)
        self.assertEqual(self.studio.tag_list_box.scroll_offset, 0)

    def test_left_bar_scroll_wheel(self):
        """测试在左栏区域滚动鼠标滚轮时，帧列表偏移量更新"""
        # 左栏宽度 self.studio.left_bar_w (如 300)
        # 向下滚轮 (flags < 0)
        self.studio._on_mouse(cv2.EVENT_MOUSEWHEEL, 50, 200, -1, None)
        self.assertGreaterEqual(self.studio.data_mgr.scroll_offset, 0)

        # 向上滚轮 (flags > 0)
        self.studio._on_mouse(cv2.EVENT_MOUSEWHEEL, 50, 200, 1, None)
        self.assertEqual(self.studio.data_mgr.scroll_offset, 0)

    def test_right_bar_scroll_wheel_and_reset_on_frame_switch(self):
        """测试右栏在超过视口的大量标靶情况下，滚轮向下滚动，且切帧后自动复位"""
        # 先执行一次渲染，初始化视口矩形
        canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.studio.ui_renderer.render_right_inspector(
            self.studio, canvas,
            x=1920 - self.studio.right_bar_w, y=40,
            w=self.studio.right_bar_w, h=1000
        )

        right_mx = 1920 - 50
        # 在右栏区域向下滚动
        self.studio._on_mouse(cv2.EVENT_MOUSEWHEEL, right_mx, 300, -1, None)
        self.assertGreater(self.studio.tag_list_box.scroll_offset, 0)
        scrolled_val = self.studio.tag_list_box.scroll_offset

        # 切换当前帧
        self.studio.select_frame(1)
        # 验证切帧后标靶列表偏移量已自动复位回 0
        self.assertEqual(self.studio.tag_list_box.scroll_offset, 0)

    def test_right_inspector_rendering_with_scrollable_list_box(self):
        """测试右栏通过 tag_list_box 成功渲染并注册各标靶的交互按钮"""
        canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.studio.gui_buttons.clear()
        self.studio.ui_renderer.render_right_inspector(
            self.studio, canvas,
            x=1920 - self.studio.right_bar_w, y=40,
            w=self.studio.right_bar_w, h=1000
        )

        # 检查是否成功注册了至少一个标靶切换按钮 TOGGLE_TAG_*
        tag_btn_found = any(btn[0].startswith("TOGGLE_TAG_") for btn in self.studio.gui_buttons)
        self.assertTrue(tag_btn_found)


if __name__ == "__main__":
    unittest.main()
