"""
GUI 控制中心自动化单元测试 (tests/test_gui_launcher.py)
=====================================================
验证 Dashboard 控制中心的卡片目录、状态采集、碰撞测试与渲染稳定性
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.gui_launcher import GuiLauncherApp, build_tools_catalog, ToolCardMeta


class TestGuiLauncher(unittest.TestCase):

    def setUp(self):
        # 使用隔离虚拟配置文件，确保测试不受用户本机历史记忆影响，也不污染用户真实偏好
        self.app = GuiLauncherApp(settings_file="__test_isolated_dummy_settings__.json")

    def test_tools_catalog_integrity(self):
        """测试工具目录数据结构完整性与快捷键不重复"""
        catalog = build_tools_catalog()
        self.assertEqual(len(catalog), 13)

        seen_keys = set()
        seen_shortcuts = set()
        valid_categories = {
            "A — Workspace",
            "B — 标定建图与生产验证",
            "D — 硬件调试与系统运维"
        }
        for tool in catalog:
            self.assertIsInstance(tool, ToolCardMeta)
            self.assertTrue(len(tool.title) > 0)
            self.assertTrue(len(tool.summary) > 0)
            self.assertTrue(len(tool.details) > 0)
            self.assertTrue(len(tool.inputs) > 0)
            self.assertTrue(len(tool.outputs) > 0)
            self.assertIn(tool.category, valid_categories)

            # 键值唯一性
            self.assertNotIn(tool.key_id, seen_keys, f"重复的 key_id: {tool.key_id}")
            seen_keys.add(tool.key_id)

            # 快捷键唯一性
            self.assertNotIn(tool.shortcut, seen_shortcuts, f"重复的 shortcut: {tool.shortcut}")
            seen_shortcuts.add(tool.shortcut)

    def test_app_initialization_and_status(self):
        """测试 Launcher 应用初始化与系统状态探针"""
        self.assertEqual(self.app.win_mgr.canvas_w, 1280)
        self.assertEqual(self.app.win_mgr.canvas_h, 1000)
        self.assertTrue(self.app._running)
        self.assertIsNotNone(self.app.system_status)

    def test_canvas_rendering(self):
        """测试 1280x720 双缓冲画布离线渲染稳定性"""
        canvas = self.app._render_canvas()
        self.assertIsInstance(canvas, np.ndarray)
        self.assertEqual(canvas.shape, (1000, 1280, 3))

    def test_hit_test_cards(self):
        """测试鼠标卡片网格碰撞检测 (基于三分组布局)"""
        # 第一张卡片 idx=0 (A组工位，全宽卡片)
        hit_0 = self.app._hit_test_cards(100, 100)
        self.assertEqual(hit_0, 0)
        hit_0_right = self.app._hit_test_cards(350, 100)
        self.assertEqual(hit_0_right, 0)  # 全宽跨两列

        # 第二张卡片 idx=1 (B组左列, AprilTag)
        hit_1 = self.app._hit_test_cards(100, 250)
        self.assertEqual(hit_1, 1)

        # 第三张卡片 idx=2 (B组右列, 采图向导)
        hit_2 = self.app._hit_test_cards(350, 250)
        self.assertEqual(hit_2, 2)

        # 越界区域 (如右侧说明栏 x=900) 应该返回 -1
        hit_none = self.app._hit_test_cards(900, 300)
        self.assertEqual(hit_none, -1)

    def test_keyboard_navigation(self):
        """测试键盘方向键导航与选择变更 (三分组布局)"""
        self.app.selected_tool_idx = 0
        # 从 row=0 (idx=0) 按下键 (40) -> row=1, col=0 (idx=1)
        self.app._handle_keyboard(40)
        self.assertEqual(self.app.selected_tool_idx, 1)

        # 从 row=1, col=0 按右键 (39) -> row=1, col=1 (idx=2)
        self.app._handle_keyboard(39)
        self.assertEqual(self.app.selected_tool_idx, 2)

        # 从 row=1, col=1 按下键 (40) -> row=2, col=1 (idx=4)
        self.app._handle_keyboard(40)
        self.assertEqual(self.app.selected_tool_idx, 4)

        # 从 row=2, col=1 按左键 (37) -> row=2, col=0 (idx=3)
        self.app._handle_keyboard(37)
        self.assertEqual(self.app.selected_tool_idx, 3)

        # 从 row=2, col=0 按上键 (38) -> row=1, col=0 (idx=1)
        self.app._handle_keyboard(38)
        self.assertEqual(self.app.selected_tool_idx, 1)

        # 从 row=1, col=0 按上键 (38) -> row=0 (idx=0)
        self.app._handle_keyboard(38)
        self.assertEqual(self.app.selected_tool_idx, 0)

        # 测试 D 组边界 (row=7: idx 12; 按下键不再越界)
        self.app.selected_tool_idx = 12
        self.app._handle_keyboard(40)
        self.assertEqual(self.app.selected_tool_idx, 12)

    def test_toast_message(self):
        """测试动态 Toast 提示设置"""
        self.app.set_toast("测试提示消息", duration=2.0)
        self.assertEqual(self.app.toast_msg, "测试提示消息")

    def test_dynamic_resize_canvas_rendering(self):
        """测试用户拖拽缩放窗口大小时，物理分辨率自适应与矢量重绘无锯齿"""
        # 模拟窗口拉大到 1600x900
        self.app.win_mgr.canvas_w = 1600
        self.app.win_mgr.canvas_h = 900
        canvas = self.app._render_canvas()
        self.assertEqual(canvas.shape, (900, 1600, 3))

        # 模拟全高清 1920x1080
        self.app.win_mgr.canvas_w = 1920
        self.app.win_mgr.canvas_h = 1080
        canvas_fhd = self.app._render_canvas()
        self.assertEqual(canvas_fhd.shape, (1080, 1920, 3))

    def test_zoom_shortcuts_laptop_compatibility(self):
        """全面测试笔记本电脑各种键位、Ctrl组合与逆向坐标映射"""
        self.app.win_mgr._force_ctrl_pressed = True

        # 1. 放大 (Zoom In): 兼容未按Shift '=' 与按Shift '+'
        self.app.win_mgr.scale_pct = 100
        self.app._handle_keyboard(ord('='))
        self.assertEqual(self.app.win_mgr.scale_pct, 110)

        self.app._handle_keyboard(ord('+'))
        self.assertEqual(self.app.win_mgr.scale_pct, 120)

        self.app._handle_keyboard(187)  # VK_OEM_PLUS
        self.assertEqual(self.app.win_mgr.scale_pct, 130)

        # 2. 缩小 (Zoom Out): 兼容未按Shift '-' 与按Shift '_' (下划线)
        self.app._handle_keyboard(ord('-'))
        self.assertEqual(self.app.win_mgr.scale_pct, 120)

        self.app._handle_keyboard(ord('_'))
        self.assertEqual(self.app.win_mgr.scale_pct, 110)

        self.app._handle_keyboard(189)  # VK_OEM_MINUS
        self.assertEqual(self.app.win_mgr.scale_pct, 100)

        self.app._handle_keyboard(31)   # Ctrl+- 控制字符
        self.assertEqual(self.app.win_mgr.scale_pct, 90)

        # 3. 复位 (Reset): '0' 或 VK_0
        self.app._handle_keyboard(ord('0'))
        self.assertEqual(self.app.win_mgr.scale_pct, 100)

        # 4. 边界保护: 50% ~ 200%
        for _ in range(15):
            self.app._handle_keyboard(ord('-'))
        self.assertEqual(self.app.win_mgr.scale_pct, 50)

        for _ in range(25):
            self.app._handle_keyboard(ord('+'))
        self.assertEqual(self.app.win_mgr.scale_pct, 200)

        # 5. 鼠标与当前物理视口坐标 1:1 原生对齐验证 (真矢量无畸变模式)
        self.app.win_mgr.scale_pct = 150
        self.app.win_mgr.canvas_w, self.app.win_mgr.canvas_h = 1920, 1080
        self.app._on_mouse(0, 960, 540, 0, None)
        self.assertEqual((self.app.mouse_x, self.app.mouse_y), (960, 540))

        # 6. Ctrl + 鼠标滚轮向上/向下平滑缩放
        self.app.win_mgr.scale_pct = 100
        self.app._on_mouse(10, 500, 300, 1, None)   # 向上滚
        self.assertEqual(self.app.win_mgr.scale_pct, 110)
        self.app._on_mouse(10, 500, 300, -1, None)  # 向下滚
        self.assertEqual(self.app.win_mgr.scale_pct, 100)

    def test_settings_persistence(self):
        """测试用户缩放比例与窗口尺寸持久化保存与二次启动自动恢复 (隔离环境运行，杜绝污染真实用户偏好)"""
        import os
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmpdir:
            test_cfg = os.path.join(tmpdir, "test_gui_settings.json")
            with patch("tools.gui_launcher.GUI_SETTINGS_FILE", test_cfg):
                # 1. 模拟缩放到 120%
                test_app = GuiLauncherApp()
                test_app.win_mgr.scale_pct = 100
                test_app._apply_zoom(+20)
                self.assertEqual(test_app.win_mgr.scale_pct, 120)
                self.assertTrue(os.path.exists(test_cfg))

                # 2. 模拟新启动一个实例，验证自动记忆恢复
                new_app = GuiLauncherApp()
                self.assertEqual(new_app.win_mgr.scale_pct, 120)
                self.assertEqual(new_app.win_mgr.canvas_w, int(1280 * 1.2))
                self.assertEqual(new_app.win_mgr.canvas_h, int(1000 * 1.2))

                # 3. 模拟拖动拉伸窗口改变分辨率，验证自动落盘
                new_app.win_mgr.canvas_w = 1600
                new_app.win_mgr.canvas_h = 900
                new_app._save_settings()

                # 4. 再次启动新实例验证窗口尺寸保持 1600x900
                third_app = GuiLauncherApp()
                self.assertEqual(third_app.win_mgr.scale_pct, 120)
                self.assertEqual(third_app.win_mgr.canvas_w, 1600)
                self.assertEqual(third_app.win_mgr.canvas_h, 900)

    def test_text_wrapping_utility(self):
        """测试文本根据最大像素宽度自适应折行计算"""
        from tools.gui_launcher import wrap_text_by_width, get_cached_font
        long_chinese_text = "【标定收尾验证】完全离线工具：文件照片输入，解算顶层芦笋空间位姿并预览 SCARA 抓取 G-code。"
        lines = wrap_text_by_width(long_chinese_text, font_size=14, max_width=200)
        self.assertTrue(len(lines) > 1)
        font = get_cached_font(14)
        for line in lines:
            bbox = font.getbbox(line)
            w = bbox[2] - bbox[0]
            self.assertLessEqual(w, 200)

    def test_asparagus_card_position_and_shortcuts(self):
        """测试芦笋离线验证卡片位于 B 组 (idx=3)，快捷键为 4，B 组生产卡片与 D 组硬件调试卡片快捷键对齐"""
        catalog = build_tools_catalog()
        ids = [t.key_id for t in catalog]
        self.assertEqual(ids[3], "asparagus_offline")
        self.assertEqual(catalog[3].shortcut, "4")
        self.assertEqual(catalog[3].category, "B — 标定建图与生产验证")
        self.assertTrue(catalog[3].is_gui)

        # 核心生产双工位均归属于 B 大类
        self.assertEqual(catalog[4].key_id, "scara_production")
        self.assertEqual(catalog[4].shortcut, "5")
        self.assertEqual(catalog[4].category, "B — 标定建图与生产验证")

        self.assertEqual(catalog[5].key_id, "isolate_wheels_production")
        self.assertEqual(catalog[5].shortcut, "0")
        self.assertEqual(catalog[5].category, "B — 标定建图与生产验证")

        # D 组硬件调试
        self.assertEqual(catalog[6].key_id, "d435_live")
        self.assertEqual(catalog[6].shortcut, "6")
        self.assertEqual(catalog[7].key_id, "scara_debug")
        self.assertEqual(catalog[7].shortcut, "7")
        self.assertEqual(catalog[8].key_id, "isolate_wheels_debug")
        self.assertEqual(catalog[8].shortcut, "8")

    def test_grid_layout_two_plus_two_plus_one(self):
        """测试 B/D 两组网格布局几何正确 (A 1张全宽, B 2x2+1, D 2+2+2+1)"""
        rects = [self.app._get_card_rect(i) for i in range(13)]
        # B 组: idx 1/2 第一行, 3/4 第二行, 5 第三行左列
        self.assertEqual(rects[1][1], rects[2][1])
        self.assertEqual(rects[3][1], rects[4][1])
        self.assertGreater(rects[3][1], rects[1][1])
        self.assertGreater(rects[5][1], rects[3][1])
        self.assertEqual(rects[5][0], rects[1][0])

        # D 组 (从 idx 6 开始): 6/7 第一行, 8/9 第二行
        self.assertEqual(rects[6][1], rects[7][1])
        self.assertEqual(rects[8][1], rects[9][1])
        self.assertGreater(rects[8][1], rects[6][1])

        # D 组整体低于 B 组
        self.assertGreater(rects[6][1], rects[5][1])
        # 全部卡片在 1000px 基准画布内
        for r in rects:
            self.assertLess(r[1] + r[3], 1000)

    def test_default_overview_panel_rendering(self):
        """测试无卡片选中/悬停时，右侧默认渲染系统环境与硬件健康总览面板"""
        self.app.selected_tool_idx = -1
        self.app.hover_tool_idx = -1
        canvas = self.app._render_canvas()
        self.assertEqual(canvas.shape, (1000, 1280, 3))
        # 验证画布非全黑
        self.assertTrue(np.any(canvas > 0))

    def test_inspector_panel_rendering_with_selection(self):
        """测试选中卡片时，右侧渲染对应工具的 Inspector 详尽指南与自动折行"""
        self.app.selected_tool_idx = 2  # tag_wizard
        self.app.hover_tool_idx = -1
        canvas = self.app._render_canvas()
        self.assertEqual(canvas.shape, (1000, 1280, 3))
        self.assertTrue(np.any(canvas > 0))

    def test_suspended_modal_rendering_and_darkening(self):
        """测试子工具启动后主窗口全屏压暗与挂起模态居中展示"""
        # 正常渲染基准画布
        self.app.is_subtool_running = False
        normal_canvas = self.app._render_canvas()
        normal_mean = float(np.mean(normal_canvas))

        # 开启挂起态
        self.app.is_subtool_running = True
        self.app.running_tool_meta = self.app.tools[0]
        suspended_canvas = self.app._render_canvas()

        self.assertEqual(suspended_canvas.shape, (1000, 1280, 3))
        # 验证暗化蒙版生效：背景区域平均亮度应显著降低（约 20%~30% 水平）
        suspended_mean = float(np.mean(suspended_canvas))
        self.assertLess(suspended_mean, normal_mean * 0.5)

    def test_suspended_input_blocking(self):
        """测试挂起期间对鼠标悬停、点击以及键盘快捷键的 100% 绝对拦截"""
        self.app.is_subtool_running = True
        self.app.running_tool_meta = self.app.tools[0]
        self.app.mouse_x, self.app.mouse_y = 100, 100
        self.app.selected_tool_idx = 0

        # 模拟鼠标移动到 (500, 500)
        self.app._on_mouse(0, 500, 500, 0, None)
        # 坐标与悬停应被直接 return，不得更新
        self.assertEqual((self.app.mouse_x, self.app.mouse_y), (100, 100))

        # 模拟键盘按键（例如数字键 2，ESC 键）
        handled = self.app._handle_keyboard(ord('2'))
        self.assertFalse(handled)
        self.assertEqual(self.app.selected_tool_idx, 0)  # 未被切换


if __name__ == "__main__":
    unittest.main()

