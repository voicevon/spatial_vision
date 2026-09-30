#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""嵌入式终端面板单元测试 (tests/test_terminal_panel.py)
=====================================================
验证 TerminalPanel 的 ANSI 颜色解析、\\r 进度条语义、环形缓冲、
像素折行与子进程流式读取全生命周期 (隔离运行, 不依赖真实诊断工具)
"""

import sys
import time
import unittest

import numpy as np

from src.ui.terminal_panel import TerminalPanel, DEFAULT_FG


class TestTerminalPanel(unittest.TestCase):

    def test_ansi_color_parse(self):
        """SGR 前景色解析: 颜色码生效, 非颜色转义剔除, reset 恢复默认色, 逐段携带位置色"""
        frags = TerminalPanel._parse_ansi("\x1b[32mOK\x1b[0m plain \x1b[2K\x1b[1A tail")
        self.assertEqual(frags[0], ("OK", (90, 215, 90)))          # 绿色
        self.assertEqual(frags[1], (" plain ", DEFAULT_FG))         # reset 后默认色
        self.assertEqual(frags[2], (" tail", DEFAULT_FG))           # 光标码剔除但不合并段落

    def test_glyph_translation(self):
        """字形规约 (commit 层生效): ✔✗⚠ → √×! , pip emoji 剔除 (msyh 缺失字形防护)"""
        panel = TerminalPanel(max_lines=10)
        panel._commit_line("✔ 安装成功 ✗ 失败 ⚠ 警告 ✨ 完成")
        text = "".join(t for t, _ in panel._lines[0])
        self.assertIn("√ 安装成功", text)
        self.assertIn("× 失败", text)
        self.assertIn("! 警告", text)
        self.assertNotIn("✨", text)

    def test_ring_buffer_max_lines(self):
        """环形缓冲: 超过 max_lines 后旧行淘汰, total_lines 计数不丢"""
        panel = TerminalPanel(max_lines=5)
        for i in range(10):
            panel._commit_line(f"line {i}")
        self.assertEqual(len(panel._lines), 5)
        self.assertEqual(panel.total_lines, 10)
        # _lines[-1] 是片段列表, 取末行文本验证
        last_text = "".join(t for t, _ in panel._lines[-1])
        self.assertEqual(last_text, "line 9")

    def test_pixel_wrap(self):
        """像素折行: 长行按面板宽度拆分, 每行不超过宽度上限"""
        panel = TerminalPanel(max_lines=50)
        panel._body_w = 120
        long_text = "芦笋抓取" * 100        # 400 个 CJK 字符
        panel._commit_line(long_text)
        self.assertGreater(len(panel._rows), 1)
        # 每个可视行的总像素宽 ≤ body_w (允许单字符溢出容差)
        from src.ui.terminal_panel import get_cached_font
        font = get_cached_font(panel.font_size)
        for row in panel._rows:
            w = sum(font.getlength(t) for t, _ in row)
            self.assertLessEqual(w, 120 + 30)

    def test_subprocess_lifecycle_cr_and_colors(self):
        """子进程全生命周期: 流式读取 / \\r 进度条覆盖 / ANSI 颜色行 / 退出码与状态行"""
        panel = TerminalPanel(max_lines=100)
        child = (
            "import sys,time;"
            "sys.stdout.write('10%'); sys.stdout.flush(); time.sleep(0.15);"
            "sys.stdout.write('\\r 50%'); sys.stdout.flush(); time.sleep(0.15);"
            "sys.stdout.write('\\r\\x1b[32m 100% OK\\x1b[0m\\n');"
            "sys.stdout.write('\\r\\n');"
            "print('done')"
        )
        panel.start([sys.executable, "-c", child])
        deadline = time.time() + 8
        while panel.is_running() and time.time() < deadline:
            time.sleep(0.05)
        self.assertFalse(panel.is_running())
        self.assertEqual(panel._exit_code, 0)
        texts = ["".join(t for t, _ in row) for row in panel._rows]
        joined = "\n".join(texts)
        # \r 覆盖语义: "10%" 与 "50%" 被覆盖, 只留最终 " 100% OK"
        self.assertIn(" 100% OK", joined)
        self.assertNotIn("10%\n", "\n".join(texts))
        self.assertNotRegex(joined, r"^10%")
        # \r\n 不产生空行; 普通行正常提交
        self.assertIn("done", texts)
        # 系统行: 命令回显与退出状态
        self.assertIn("[终端] $", joined)
        self.assertIn("exit=0", joined)
        self.assertIn("已完成", panel.status_text)

    def test_draw_smoke(self):
        """渲染冒烟: 在空白画布上 draw 不抛异常且底部状态行可获取"""
        panel = TerminalPanel()
        panel._commit_line("\x1b[31m错误示例\x1b[0m")
        panel._commit_line("普通中文行")
        canvas = np.zeros((300, 500, 3), dtype=np.uint8)
        panel.draw(canvas, (10, 10, 490, 280))
        self.assertIsInstance(panel.status_text, str)

    def test_scroll_and_follow(self):
        """滚动语义: 上翻回看, 触底自动恢复跟随"""
        panel = TerminalPanel(max_lines=100)
        for i in range(80):
            panel._commit_line(f"row {i}")
        panel._body_w = 400
        # draw 一次以同步 _body_w → _rows 折行
        canvas = np.zeros((400, 600, 3), dtype=np.uint8)
        panel.draw(canvas, (10, 10, 590, 380))
        visible = max(1, (380 - 10 - 8) // panel.row_h)
        self.assertLess(visible, len(panel._rows))
        panel.scroll(10, visible)                 # 上翻
        self.assertIsNotNone(panel._scroll_pos)
        start = panel._scroll_pos
        panel.scroll(-10 ** 6, visible)           # 一路下滚 → 触底恢复跟随
        self.assertIsNone(panel._scroll_pos)
        self.assertLess(start, len(panel._rows))


if __name__ == "__main__":
    unittest.main()
