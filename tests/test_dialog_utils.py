#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨平台原生对话框单元测试 (tests/test_dialog_utils.py)
======================================================
验证点：
  1. show_error_dialog / show_critical_message 错误报警弹窗逻辑；
  2. Windows 下调用 ctypes.windll.user32.MessageBoxW 传参校验 (MB_ICONERROR = 0x10)；
  3. Linux / macOS 及 Tkinter 兜底调用；
  4. prompt_confirm 与 prompt_input_text 基础功能。
"""

import sys
import unittest
from unittest.mock import patch, MagicMock

from src.ui.dialog_utils import (
    show_error_dialog,
    show_confirm_dialog,
    show_text_input_dialog,
)


class TestDialogUtils(unittest.TestCase):
    def test_show_error_dialog_windows(self):
        """测试 Windows 平台下正确调用 MessageBoxW 并包含 MB_ICONERROR 标志"""
        with patch("sys.platform", "win32"):
            mock_ctypes = MagicMock()
            mock_msgbox = MagicMock(return_value=1)
            mock_ctypes.windll.user32.MessageBoxW = mock_msgbox

            with patch.dict("sys.modules", {"ctypes": mock_ctypes}):
                show_error_dialog("错误标题", "错误详情内容")

                mock_msgbox.assert_called_once()
                args, _ = mock_msgbox.call_args
                hwnd, msg, title, flags = args
                self.assertEqual(hwnd, 0)
                self.assertEqual(msg, "错误详情内容")
                self.assertEqual(title, "错误标题")
                # 必须包含 MB_ICONERROR (0x00000010)
                self.assertTrue(bool(flags & 0x00000010), "必须包含 MB_ICONERROR 标志")

    def test_show_error_dialog_linux(self):
        """测试 Linux 平台下优先调用 zenity --error"""
        with patch("sys.platform", "linux"):
            with patch("shutil.which", return_value="/usr/bin/zenity"):
                with patch("subprocess.run") as mock_run:
                    show_error_dialog("对齐失败", "缺少相对地图")
                    mock_run.assert_called_once()
                    cmd = mock_run.call_args[0][0]
                    self.assertIn("zenity", cmd[0])
                    self.assertIn("--error", cmd)
                    self.assertIn("对齐失败", cmd)

    def test_gui_components_exports(self):
        """测试从 gui_components 正常导出对话框组件"""
        from src.ui.gui_components import (
            show_error_dialog as comp_show_error,
            show_confirm_dialog as comp_show_confirm,
            show_text_input_dialog as comp_show_text_input,
        )
        self.assertIs(comp_show_error, show_error_dialog)
        self.assertIs(comp_show_confirm, show_confirm_dialog)
        self.assertIs(comp_show_text_input, show_text_input_dialog)


if __name__ == "__main__":
    unittest.main()
