# -*- coding: utf-8 -*-
"""
通用工业级可滚动列表组件 ScrollableListBox 单元测试
===================================================
验证：
1. 几何与视口计算 (get_visible_count, clamp_scroll_offset)
2. 滚轮步进与滚动导航 (handle_scroll, scroll_to_index)
3. 鼠标交互事件分发 (条目点击选择、Hover 悬停检测)
4. 滚动条状态机 (滑块拖拽追踪、轨道点击快速跳转、释放复位)
5. 委托渲染模式 (draw_item_callback 参数与调用次数正确性)
6. 空列表优雅降级与边界健壮性
"""

import unittest
import numpy as np

from src.utils.gui_components import ScrollableListBox


class TestScrollableListBox(unittest.TestCase):
    def setUp(self):
        self.list_box = ScrollableListBox(
            item_height=30,
            item_gap=2,
            scrollbar_width=8,
            auto_hide_scrollbar=True,
            render_item_background=True,
            scroll_speed=2,
        )

    def test_visible_count_and_clamp(self):
        # stride = 30 + 2 = 32
        # vh = 320 -> 320 // 32 = 10 visible items
        vis_c = self.list_box.get_visible_count(320)
        self.assertEqual(vis_c, 10)

        # 15 items, visible 10 -> max offset = 5
        self.list_box.scroll_offset = 12
        self.list_box.clamp_scroll_offset(total_items=15, visible_count=10)
        self.assertEqual(self.list_box.scroll_offset, 5)

        # items <= visible -> max offset = 0
        self.list_box.scroll_offset = 3
        self.list_box.clamp_scroll_offset(total_items=8, visible_count=10)
        self.assertEqual(self.list_box.scroll_offset, 0)

    def test_scroll_to_index(self):
        # 20 items, visible 5
        total_items = 20
        vis_c = 5

        # 向上滚动到第 2 项 (当前在 8)
        self.list_box.scroll_offset = 8
        self.list_box.scroll_to_index(index=2, total_items=total_items, visible_count=vis_c)
        self.assertEqual(self.list_box.scroll_offset, 2)

        # 向下滚动到第 12 项 (超出当前可见区间 [2..6])
        self.list_box.scroll_to_index(index=12, total_items=total_items, visible_count=vis_c)
        # 应滚动到让 12 位于底部: 12 - 5 + 1 = 8
        self.assertEqual(self.list_box.scroll_offset, 8)

        # 目标项已经在可见范围内 [8..12] -> offset 保持不变
        self.list_box.scroll_to_index(index=10, total_items=total_items, visible_count=vis_c)
        self.assertEqual(self.list_box.scroll_offset, 8)

    def test_handle_scroll(self):
        # 渲染一次以初始化 _last_visible_count (例如 10 项)
        canvas = np.zeros((320, 200, 3), dtype=np.uint8)
        items = [f"Item {i}" for i in range(25)]
        self.list_box.render(canvas, (10, 0, 180, 320), items)

        self.assertEqual(self.list_box.scroll_offset, 0)

        # 向下滚 1 步 (delta_lines = 1, speed = 2 -> offset + 2)
        changed = self.list_box.handle_scroll(1, total_items=25)
        self.assertTrue(changed)
        self.assertEqual(self.list_box.scroll_offset, 2)

        # 向下滚多步至底部超界
        self.list_box.handle_scroll(20, total_items=25)
        # max_offset = 25 - 10 = 15
        self.assertEqual(self.list_box.scroll_offset, 15)

        # 向上滚
        self.list_box.handle_scroll(-1, total_items=25)
        self.assertEqual(self.list_box.scroll_offset, 13)

    def test_item_click_and_hover(self):
        canvas = np.zeros((320, 200, 3), dtype=np.uint8)
        items = [f"Frame_{i:02d}" for i in range(20)]
        rect = (20, 10, 160, 320)  # y: 10..330, item_h: 30, gap: 2 -> stride: 32

        self.list_box.render(canvas, rect, items)

        # Row 0: y in [10, 40]
        handled, idx = self.list_box.handle_mouse_down(mx=50, my=25, total_items=20)
        self.assertTrue(handled)
        self.assertEqual(idx, 0)
        self.assertEqual(self.list_box.selected_index, 0)

        # Row 2: y in [10 + 2*32, 10 + 2*32 + 30] = [74, 104]
        handled, idx = self.list_box.handle_mouse_down(mx=50, my=80, total_items=20)
        self.assertTrue(handled)
        self.assertEqual(idx, 2)
        self.assertEqual(self.list_box.selected_index, 2)

        # 测试 Hover
        self.list_box.handle_mouse_move(mx=50, my=80, total_items=20)
        self.assertEqual(self.list_box.hover_index, 2)

        # 移出条目区域
        self.list_box.handle_mouse_move(mx=5, my=5, total_items=20)
        self.assertEqual(self.list_box.hover_index, -1)

    def test_scrollbar_thumb_drag_and_track_jump(self):
        canvas = np.zeros((320, 200, 3), dtype=np.uint8)
        items = [f"Tag_{i}" for i in range(50)]
        rect = (10, 10, 180, 320)  # vh=320, vis=10, max_offset=40

        self.list_box.render(canvas, rect, items)
        tx, ty, tw, th = self.list_box._last_thumb_rect
        self.assertGreater(tw, 0)
        self.assertGreater(th, 0)

        # 1. 命中滑块按下 -> 进入拖拽态
        handled, clicked_idx = self.list_box.handle_mouse_down(mx=tx + 2, my=ty + 5, total_items=50)
        self.assertTrue(handled)
        self.assertIsNone(clicked_idx)
        self.assertTrue(self.list_box.is_dragging_thumb)

        # 2. 拖拽鼠标下移 50px
        self.list_box.handle_mouse_move(mx=tx + 2, my=ty + 5 + 50, total_items=50)
        self.assertGreater(self.list_box.scroll_offset, 0)

        # 3. 释放鼠标
        released = self.list_box.handle_mouse_up()
        self.assertTrue(released)
        self.assertFalse(self.list_box.is_dragging_thumb)

        # 4. 点击滚动条轨道中下方 -> 快速跳转
        rx, ry, rw, rh = self.list_box._last_track_rect
        handled, _ = self.list_box.handle_mouse_down(mx=rx + 2, my=ry + rh - 10, total_items=50)
        self.assertTrue(handled)
        # 应接近最大 offset (40)
        self.assertGreater(self.list_box.scroll_offset, 30)

    def test_delegate_render_callback(self):
        canvas = np.zeros((200, 300, 3), dtype=np.uint8)
        items = [{"id": 1, "name": "Alpha"}, {"id": 2, "name": "Beta"}, {"id": 3, "name": "Gamma"}]
        rendered_records = []

        def custom_draw(c, item_rect, item, idx, is_hover, is_selected):
            rendered_records.append((idx, item["name"], item_rect, is_selected))

        self.list_box.selected_index = 1
        self.list_box.render(
            canvas=canvas,
            rect=(10, 10, 280, 180),
            items=items,
            draw_item_callback=custom_draw,
        )

        self.assertEqual(len(rendered_records), 3)
        self.assertEqual(rendered_records[0][1], "Alpha")
        self.assertFalse(rendered_records[0][3])
        self.assertEqual(rendered_records[1][1], "Beta")
        self.assertTrue(rendered_records[1][3])  # selected

    def test_empty_items(self):
        canvas = np.zeros((100, 200, 3), dtype=np.uint8)
        # 空数据列表不报错
        self.list_box.render(canvas, (10, 10, 180, 80), items=[], empty_text="无标靶数据")
        self.assertEqual(len(self.list_box._item_rects), 0)
        self.assertEqual(self.list_box._last_thumb_rect, (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
