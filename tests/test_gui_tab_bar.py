# -*- coding: utf-8 -*-
"""
全局基础 GUI 交互控件库 TabBar 单元测试
====================================
验证：
1. TabItem 数据封装与 TabBar 动态配置
2. 单源几何排版 (自适应文字宽度、固定宽度、间距)
3. 状态切换流转与 on_change 事件回调
4. Hit-test 单源热区碰撞检测与 handle_click 闭环
5. 双模视觉渲染冒烟测试 (STYLE_CAPSULE / STYLE_PILL / STYLE_LINE)
"""

import unittest
import numpy as np

from src.utils.gui_components import TabBar, TabItem


class TestGuiTabBar(unittest.TestCase):
    def test_tab_bar_initialization_and_select(self):
        called_keys = []
        bar = TabBar(
            tabs=[("tab_a", "首页面板"), ("tab_b", "数据中心", "★"), ("tab_c", "系统设置")],
            active_key="tab_a",
            on_change=lambda k: called_keys.append(k),
        )

        self.assertEqual(len(bar.items), 3)
        self.assertEqual(bar.active_key, "tab_a")
        self.assertEqual(bar.items[1].badge, "★")

        # 切换到 tab_b
        changed = bar.select("tab_b")
        self.assertTrue(changed)
        self.assertEqual(bar.active_key, "tab_b")
        self.assertEqual(called_keys, ["tab_b"])

        # 重复切换同一 tab 不触发改变
        changed_again = bar.select("tab_b")
        self.assertFalse(changed_again)
        self.assertEqual(len(called_keys), 1)

    def test_tab_bar_layout_and_hit_test(self):
        bar = TabBar(
            tabs=[
                TabItem(key="t1", label="看板"),
                TabItem(key="t2", label="相册", enabled=False),
                TabItem(key="t3", label="设置"),
            ],
            fixed_width=100,
            tab_height=30,
            spacing=10,
        )

        container = (50, 10, 500, 40)
        layout = bar.compute_layout(container)
        self.assertEqual(len(layout), 3)

        # 验证各矩形位置 (50 起始，宽 100，间距 10)
        k1, r1 = layout[0]
        self.assertEqual(k1, "t1")
        self.assertEqual(r1, (50, 15, 100, 30))

        k2, r2 = layout[1]
        self.assertEqual(k2, "t2")
        self.assertEqual(r2, (160, 15, 100, 30))

        k3, r3 = layout[2]
        self.assertEqual(k3, "t3")
        self.assertEqual(r3, (270, 15, 100, 30))

        # Hit-test 探测
        self.assertEqual(bar.hit_test(60, 20), "t1")
        # t2 是 disabled，命中应返回 None
        self.assertIsNone(bar.hit_test(170, 20))
        self.assertEqual(bar.hit_test(280, 20), "t3")
        self.assertIsNone(bar.hit_test(10, 10))

        # handle_click
        hit = bar.handle_click(280, 20)
        self.assertEqual(hit, "t3")
        self.assertEqual(bar.active_key, "t3")

    def test_tab_bar_render_smoke(self):
        canvas = np.zeros((200, 600, 3), dtype=np.uint8)
        bar = TabBar(
            tabs=[("report", "体检报告"), ("calib", "标定相册"), ("prod", "生产相册", "★")],
            style=TabBar.STYLE_CAPSULE,
        )

        # 渲染胶囊风格
        res_capsule = bar.render(canvas, (20, 20, 560, 50), mouse_pos=(30, 30))
        self.assertEqual(len(res_capsule), 3)

        # 渲染药丸风格
        bar.style = TabBar.STYLE_PILL
        res_pill = bar.render(canvas, (20, 80, 560, 50))
        self.assertEqual(len(res_pill), 3)

        # 渲染极简下划线风格
        bar.style = TabBar.STYLE_LINE
        res_line = bar.render(canvas, (20, 140, 560, 50))
        self.assertEqual(len(res_line), 3)


if __name__ == "__main__":
    unittest.main()
