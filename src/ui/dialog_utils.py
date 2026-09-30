# -*- coding: utf-8 -*-
"""
跨平台原生对话框适配工具 (Dialog Utils)
提供系统原生、轻量级、兼顾中文输入法与跨平台（Windows / Linux / macOS）的对话框接口。

特性：
1. 确认框 (prompt_confirm)：
   - Windows 下优先使用 ctypes.windll.user32.MessageBoxW 原生内核调用（微秒级响应、零外部依赖、零幽灵窗口）
   - Linux 下优先使用 zenity / kdialog 系统原生弹窗
   - 无图形环境自动降级为控制台标准确认
2. 文本输入框 (prompt_input_text)：
   - Linux 下优先使用 zenity --entry（原生支持系统 Fcitx/IBus 中文输入法）
   - Windows / 跨平台图形环境下统一受控托管中文输入，业务代码完全解耦
   - 异常或无图形环境自动降级为控制台 input()
"""

import os
import sys
import shutil
import subprocess
from typing import Optional
from src.utils.logger import get_logger

log = get_logger(__name__)


def prompt_confirm(title: str, message: str) -> bool:
    """
    弹出跨平台原生确认对话框 (Yes / No)。

    :param title: 弹窗标题
    :param message: 提示信息
    :return: 用户点击“是”返回 True，点击“否”或关闭返回 False
    """
    # 1. Windows 原生内核 API: 微秒级原生对话框，完全不加载任何 GUI 引擎
    if sys.platform == "win32":
        try:
            import ctypes
            IDYES = 6
            MB_YESNO = 0x00000004
            MB_ICONQUESTION = 0x00000020
            MB_TOPMOST = 0x00040000
            res = ctypes.windll.user32.MessageBoxW(0, message, title, MB_YESNO | MB_ICONQUESTION | MB_TOPMOST)
            return res == IDYES
        except Exception as e:
            log.warning(f"Windows 原生 MessageBox 调用异常，尝试备用方案: {e}")

    # 2. Linux 环境优先检测系统原生对话框工具 (zenity / kdialog)
    if sys.platform.startswith("linux"):
        if shutil.which("zenity"):
            try:
                cmd = ["zenity", "--question", "--title", title, "--text", message]
                res = subprocess.run(cmd, capture_output=True)
                return res.returncode == 0
            except Exception as e:
                log.warning(f"zenity 调用异常: {e}")
        elif shutil.which("kdialog"):
            try:
                cmd = ["kdialog", "--title", title, "--yesno", message]
                res = subprocess.run(cmd, capture_output=True)
                return res.returncode == 0
            except Exception as e:
                log.warning(f"kdialog 调用异常: {e}")

    # 3. 跨平台图形环境兜底 (受控封装)
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        ans = messagebox.askyesno(title, message, parent=root)
        root.destroy()
        return bool(ans)
    except Exception:
        pass

    # 4. 无图形界面或全降级兜底: 控制台标准输入
    try:
        print(f"\n[确认] {title}: {message} (y/n): ", end="", flush=True)
        choice = input().strip().lower()
        return choice in ("y", "yes", "true", "1")
    except Exception:
        return True


def prompt_input_text(title: str, prompt_text: str, initial: str = "", default: str = "") -> str:
    """
    弹出跨平台原生单行文本输入对话框，完美支持中文拼音/五笔输入法。

    :param title: 弹窗标题
    :param prompt_text: 提示文字
    :param initial: 初始文本
    :param default: 初始文本 (别名兼容)
    :return: 用户输入的字符串（自动去除首尾空白）；取消或关闭返回空字符串 ""
    """
    init_val = default if default else initial

    # 1. Linux 环境下优先调用系统原生 zenity（原生挂载系统 Fcitx/IBus 中文输入法）
    if sys.platform.startswith("linux") and shutil.which("zenity"):
        try:
            cmd = ["zenity", "--entry", "--title", title, "--text", prompt_text]
            if init_val:
                cmd.extend(["--entry-text", init_val])
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                return res.stdout.strip()
            return ""
        except Exception as e:
            log.warning(f"zenity entry 调用异常: {e}")

    # 2. 图形界面环境统一样式输入框（支持完整中文 IME 输入上屏）
    try:
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        val = simpledialog.askstring(title, prompt_text, initialvalue=init_val, parent=root)
        root.destroy()
        return val.strip() if val else ""
    except Exception as e:
        log.warning(f"图形输入框调用异常: {e}")

    # 3. 命令行终端兜底
    print(f"\n{title}: {prompt_text}")
    try:
        val = input("请输入: ").strip()
        return val
    except Exception:
        return ""


def show_error_dialog(title: str, message: str) -> None:
    """
    弹出跨平台原生错误报警对话框 (带红底白叉 ❌ 图标，Windows Critical Error 风格)。
    具备系统级置顶、报警音效和阻塞式强制确认交互。

    :param title: 弹窗标题
    :param message: 报警详情与排查指导
    """
    log.error(f"[CRITICAL_DIALOG] {title}: {message}")

    # 1. Windows 原生内核 API: 微秒级原生 Critical Error 红色大叉框 (MB_ICONERROR)
    if sys.platform == "win32":
        try:
            import ctypes
            MB_OK = 0x00000000
            MB_ICONERROR = 0x00000010      # 红色圆圈白叉 (Critical Error)
            MB_TOPMOST = 0x00040000        # 系统级置顶，防任何全屏/大窗口遮挡
            MB_SETFOREGROUND = 0x00010000  # 强制获取焦点
            ctypes.windll.user32.MessageBoxW(0, str(message), str(title), MB_OK | MB_ICONERROR | MB_TOPMOST | MB_SETFOREGROUND)
            return
        except Exception as e:
            log.warning(f"Windows 原生 MessageBoxW 错误框调用异常，尝试备用方案: {e}")

    # 2. Linux 环境原生对话框 (zenity / kdialog)
    if sys.platform.startswith("linux"):
        if shutil.which("zenity"):
            try:
                cmd = ["zenity", "--error", "--title", title, "--text", message]
                subprocess.run(cmd, capture_output=True)
                return
            except Exception as e:
                log.warning(f"zenity error 调用异常: {e}")
        elif shutil.which("kdialog"):
            try:
                cmd = ["kdialog", "--title", title, "--error", message]
                subprocess.run(cmd, capture_output=True)
                return
            except Exception as e:
                log.warning(f"kdialog error 调用异常: {e}")

    # 3. macOS 原生弹窗 (osascript)
    if sys.platform == "darwin":
        try:
            script = f'display alert "{title}" message "{message}" as critical'
            subprocess.run(["osascript", "-e", script], capture_output=True)
            return
        except Exception as e:
            log.warning(f"osascript alert 调用异常: {e}")

    # 4. 跨平台图形环境兜底 (受控 Tkinter 弹窗)
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        messagebox.showerror(title, message, parent=root)
        root.destroy()
        return
    except Exception:
        pass

    # 5. 无图形界面兜底
    print(f"\n[CRITICAL ERROR] ========================================")
    print(f"[*] 标题: {title}")
    print(f"[*] 内容:\n{message}")
    print(f"========================================================\n")


# 别名兼容
show_critical_message = show_error_dialog
prompt_error = show_error_dialog

