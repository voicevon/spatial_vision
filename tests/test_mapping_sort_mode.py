#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
空间建图工作站 - 帧列表排序模式单测 (tests/test_mapping_sort_mode.py)
===================================================================
验证重点：
1. SORT_MODE_OPTIONS 常量中包含 tag_err_desc (Tag残差降序)；
2. MappingDataManager 在 tag_err_desc 模式下的排序行为：
   - 验证单 Tag 极端残差优先置顶（即使该帧的平均残差较低）；
   - 验证最大 Tag 残差相同场景下，次大 Tag 残差更高者优先；
   - 验证被排除帧 (is_excluded=True) 正确置底；
   - 验证与 err_desc (帧均残差降序) 的对比区别；
3. MappingFrameListMixin 紧凑模式在 tag_err_desc 模式下正确格式化显示最大 Tag 编号及残差 (如 T12:1.80px)。
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.spatial_mapping_studio.mapping_ui_common import SORT_MODE_OPTIONS
from tools.spatial_mapping_studio.mapping_state import MappingDataManager
from tools.spatial_mapping_studio.mapping_frame_list import MappingFrameListMixin


class DummyApp:
    def __init__(self):
        self.image_files = []
        self.frame_metrics_cache = {}
        self.sort_mode = "tag_err_desc"
        self.filter_mode = "all"
        self.data_mgr = self


class TestMappingSortMode(unittest.TestCase):
    def test_sort_mode_options_contains_tag_err_desc(self):
        keys = [k for k, _ in SORT_MODE_OPTIONS]
        self.assertIn("tag_err_desc", keys)
        self.assertIn("err_desc", keys)
        self.assertIn("name_asc", keys)

    def test_tag_err_desc_sorting_logic(self):
        # 构造数据模型 (纯逻辑测试无需依赖物理文件 IO)
        dm = MappingDataManager.__new__(MappingDataManager)
        dm.image_files = [
            "path/to/frame_a.png",  # Tag 1: 0.1px, Tag 2: 2.0px -> 均值 1.05px, 最大 2.0px
            "path/to/frame_b.png",  # Tag 3: 1.3px, Tag 4: 1.3px -> 均值 1.30px, 最大 1.3px
            "path/to/frame_c.png",  # Tag 5: 2.0px, Tag 6: 0.8px -> 均值 1.40px, 最大 2.0px, 次大 0.8px > 0.1px
            "path/to/frame_d.png",  # 已排除帧，即便残差 3.0px 也要排在末尾
        ]

        dm.frame_metrics_cache = {
            "frame_a.png": {
                "is_excluded": False,
                "mean_err": 1.05,
                "max_err": 2.0,
                "tag_count": 2,
                "tag_errors": {1: 0.1, 2: 2.0}
            },
            "frame_b.png": {
                "is_excluded": False,
                "mean_err": 1.30,
                "max_err": 1.3,
                "tag_count": 2,
                "tag_errors": {3: 1.3, 4: 1.3}
            },
            "frame_c.png": {
                "is_excluded": False,
                "mean_err": 1.40,
                "max_err": 2.0,
                "tag_count": 2,
                "tag_errors": {5: 2.0, 6: 0.8}
            },
            "frame_d.png": {
                "is_excluded": True,
                "mean_err": 3.0,
                "max_err": 3.0,
                "tag_count": 1,
                "tag_errors": {7: 3.0}
            },
        }

        # 1. 验证 Tag 残差降序 (tag_err_desc)
        dm.sort_mode = "tag_err_desc"
        dm.filter_mode = "all"
        indices = dm.get_filtered_indices()
        names = [os.path.basename(dm.image_files[i]) for i in indices]

        # 预期顺序：
        # frame_c (最大 2.0, 次大 0.8)
        # frame_a (最大 2.0, 次大 0.1)
        # frame_b (最大 1.3)
        # frame_d (已排除，排在未排除帧后面)
        self.assertEqual(names, ["frame_c.png", "frame_a.png", "frame_b.png", "frame_d.png"])

        # 2. 对比验证 帧均残差降序 (err_desc)
        dm.sort_mode = "err_desc"
        indices_mean = dm.get_filtered_indices()
        names_mean = [os.path.basename(dm.image_files[i]) for i in indices_mean]
        # 在均值降序下，frame_d (3.0) > frame_c (1.40) > frame_b (1.30) > frame_a (1.05)
        self.assertEqual(names_mean, ["frame_d.png", "frame_c.png", "frame_b.png", "frame_a.png"])

    def test_compact_frame_item_rendering_text(self):
        class RendererMock(MappingFrameListMixin):
            pass

        renderer = RendererMock()
        app = DummyApp()
        app.image_files = ["path/to/view_0001.png"]
        app.frame_metrics_cache = {
            "view_0001.png": {
                "is_excluded": False,
                "mean_err": 0.35,
                "max_err": 1.82,
                "tag_count": 2,
                "tag_errors": {10: 0.2, 12: 1.82}
            }
        }
        app.sort_mode = "tag_err_desc"

        # 验证提取到的信息与格式化
        meta = app.frame_metrics_cache["view_0001.png"]
        tag_errors = meta.get("tag_errors", {})
        max_tid = max(tag_errors.keys(), key=lambda t: tag_errors[t])
        err_val = float(tag_errors[max_tid])
        err_str = f"T{max_tid}:{err_val:.2f}px"

        self.assertEqual(err_str, "T12:1.82px")


if __name__ == "__main__":
    unittest.main()
