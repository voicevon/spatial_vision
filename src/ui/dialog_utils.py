# -*- coding: utf-8 -*-
"""
跨平台原生对话框适配工具 (Dialog Utils)
提供系统原生、轻量级、兼顾中文输入法与跨平台（Windows / Linux / macOS）的对话框接口。

特性：
1. 确认框 (show_confirm_dialog)：
   - Windows 下优先使用 ctypes.windll.user32.MessageBoxW 原生内核调用（微秒级响应、零外部依赖、零幽灵窗口）
   - Linux 下优先使用 zenity / kdialog 系统原生弹窗
   - 无图形环境自动降级为控制台标准确认
2. 文本输入框 (show_text_input_dialog)：
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


def show_confirm_dialog(title: str, message: str) -> bool:
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


def show_text_input_dialog(title: str, prompt_text: str, initial: str = "") -> str:
    """
    弹出跨平台原生单行文本输入对话框，完美支持中文拼音/五笔输入法。

    :param title: 弹窗标题
    :param prompt_text: 提示文字
    :param initial: 初始文本
    :return: 用户输入的字符串（自动去除首尾空白）；取消或关闭返回空字符串 ""
    """
    init_val = initial

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


def show_info_dialog(title: str, message: str) -> None:
    """
    弹出跨平台原生信息提示对话框 (带蓝色信息 ℹ️ 图标)。
    具备系统级置顶与确认交互。

    :param title: 弹窗标题
    :param message: 提示详情
    """
    log.info(f"[INFO_DIALOG] {title}: {message}")

    # 1. Windows 原生内核 API (MB_ICONINFORMATION)
    if sys.platform == "win32":
        try:
            import ctypes
            MB_OK = 0x00000000
            MB_ICONINFORMATION = 0x00000040  # 蓝色圆圈叹号 (Information)
            MB_TOPMOST = 0x00040000        # 系统级置顶
            MB_SETFOREGROUND = 0x00010000  # 强制获取焦点
            ctypes.windll.user32.MessageBoxW(0, str(message), str(title), MB_OK | MB_ICONINFORMATION | MB_TOPMOST | MB_SETFOREGROUND)
            return
        except Exception as e:
            log.warning(f"Windows 原生 MessageBoxW 信息框调用异常: {e}")

    # 2. Linux 环境原生对话框
    if sys.platform.startswith("linux"):
        if shutil.which("zenity"):
            try:
                subprocess.run(["zenity", "--info", "--title", title, "--text", message], capture_output=True)
                return
            except Exception:
                pass
        elif shutil.which("kdialog"):
            try:
                subprocess.run(["kdialog", "--title", title, "--msgbox", message], capture_output=True)
                return
            except Exception:
                pass

    # 3. macOS 原生弹窗
    if sys.platform == "darwin":
        try:
            script = f'display alert "{title}" message "{message}"'
            subprocess.run(["osascript", "-e", script], capture_output=True)
            return
        except Exception:
            pass

    # 4. 跨平台 Tkinter 弹窗兜底
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        messagebox.showinfo(title, message, parent=root)
        root.destroy()
        return
    except Exception:
        pass

    # 5. 终端标准输出兜底
    print(f"\n[INFO DIALOG] ==========================================")
    print(f"[*] 标题: {title}")
    print(f"[*] 内容:\n{message}")
    print(f"========================================================\n")


def prompt_hardware_config(
    title: str = "工位相机硬件与分辨率配置",
    initial_camera_type: str = "realsense",
    initial_serial: str = "",
    initial_resolution: str = "1920x1080",
    available_devices: Optional[dict] = None
) -> Optional[dict]:
    """
    弹出结构化硬件配置窗口，配置工位绑定的相机类型、序列号及标定分辨率单一真理源。
    返回 dict {"camera_type": str, "camera_serial": str, "camera_resolution": [w, h]}，取消返回 None
    """
    try:
        import tkinter as tk
        from tkinter import ttk

        result = {"cancelled": True, "data": None}

        root = tk.Tk()
        root.title(title)
        root.geometry("420x310")
        root.resizable(False, False)
        root.attributes("-topmost", True)

        # 居中显示
        root.update_idletasks()
        w = root.winfo_width()
        h = root.winfo_height()
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")

        frame = ttk.Frame(root, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)

        # 1. 相机类型
        ttk.Label(frame, text="相机设备类型:", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 4))
        cam_type_var = tk.StringVar(value=initial_camera_type.lower() if initial_camera_type in ("realsense", "usb") else "realsense")
        type_frame = ttk.Frame(frame)
        type_frame.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 10))
        r1 = ttk.Radiobutton(type_frame, text="RealSense D435 / D400", variable=cam_type_var, value="realsense")
        r2 = ttk.Radiobutton(type_frame, text="USB 普通/工业摄像头", variable=cam_type_var, value="usb")
        r1.pack(side=tk.LEFT, padx=(0, 16))
        r2.pack(side=tk.LEFT)

        # 2. 相机硬件标识 (序列号或设备索引)
        ttk.Label(frame, text="硬件序列号 / 设备号 (可留空自动识别):", font=("Segoe UI", 10, "bold")).grid(row=2, column=0, sticky="w", pady=(0, 4))
        serial_var = tk.StringVar(value=str(initial_serial or ""))
        serial_combo = ttk.Combobox(frame, textvariable=serial_var, width=38)

        def update_serial_options(*args):
            ctype = cam_type_var.get()
            opts = []
            if available_devices:
                if ctype == "realsense":
                    for d in available_devices.get("realsense", []):
                        if d.get("serial"):
                            opts.append(d["serial"])
                else:
                    for d in available_devices.get("usb", []):
                        opts.append(str(d.get("index", 0)))
            serial_combo["values"] = opts
            cur_s = serial_var.get()
            if not cur_s and opts:
                serial_var.set(opts[0])

        cam_type_var.trace_add("write", update_serial_options)
        update_serial_options()
        if initial_serial and initial_serial not in serial_combo["values"]:
            vals = list(serial_combo["values"])
            vals.insert(0, initial_serial)
            serial_combo["values"] = vals
            serial_var.set(initial_serial)

        serial_combo.grid(row=3, column=0, columnspan=2, sticky="we", pady=(0, 10))

        # 3. 图像物理分辨率规格 (单一真理源)
        ttk.Label(frame, text="标定与作业物理分辨率 (单一真理源):", font=("Segoe UI", 10, "bold")).grid(row=4, column=0, sticky="w", pady=(0, 4))
        res_var = tk.StringVar(value=initial_resolution or "1920x1080")
        res_combo = ttk.Combobox(frame, textvariable=res_var, width=38, state="readonly")
        res_combo["values"] = ("1920x1080", "1280x720", "640x480")
        res_combo.grid(row=5, column=0, columnspan=2, sticky="we", pady=(0, 16))

        # 底部操作栏
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=6, column=0, columnspan=2, sticky="e", pady=(8, 0))

        def on_confirm():
            ctype = cam_type_var.get()
            sn = serial_var.get().strip()
            res_str = res_combo.get().strip() or "1920x1080"
            try:
                rw, rh = [int(x) for x in res_str.split("x")]
            except Exception:
                rw, rh = 1920, 1080
            result["cancelled"] = False
            result["data"] = {
                "camera_type": ctype,
                "camera_serial": sn,
                "camera_resolution": [rw, rh],
            }
            root.destroy()

        def on_cancel():
            root.destroy()

        cancel_btn = ttk.Button(btn_frame, text="取消", command=on_cancel)
        cancel_btn.pack(side=tk.RIGHT, padx=(8, 0))
        confirm_btn = ttk.Button(btn_frame, text="确认保存", command=on_confirm)
        confirm_btn.pack(side=tk.RIGHT)

        root.bind("<Return>", lambda e: on_confirm())
        root.bind("<Escape>", lambda e: on_cancel())

        root.mainloop()

        return None if result["cancelled"] else result["data"]

    except Exception as e:
        log.warning(f"[prompt_hardware_config] 弹窗异常: {e}")
        return None


def prompt_create_workspace_dialog(
    default_alias: str = "Workspace_1",
    available_devices: Optional[dict] = None
) -> Optional[dict]:
    """
    弹出新建工位综合配置窗口：
    输入工位名称、选择相机硬件类型、指定序列号、选择分辨率规格。
    返回 dict {"alias": str, "camera_type": str, "camera_serial": str, "camera_resolution": [w, h]}，取消返回 None
    """
    try:
        import tkinter as tk
        from tkinter import ttk

        result = {"cancelled": True, "data": None}

        root = tk.Tk()
        root.title("新建工位 (Workspace)")
        root.geometry("440x360")
        root.resizable(False, False)
        root.attributes("-topmost", True)

        # 居中显示
        root.update_idletasks()
        w = root.winfo_width()
        h = root.winfo_height()
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")

        frame = ttk.Frame(root, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)

        # 1. 工位别名
        ttk.Label(frame, text="工位别名 (支持中文，如: 2号机架高位):", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 4))
        alias_var = tk.StringVar(value=default_alias)
        alias_entry = ttk.Entry(frame, textvariable=alias_var, width=42)
        alias_entry.grid(row=1, column=0, columnspan=2, sticky="we", pady=(0, 10))
        alias_entry.focus_set()
        alias_entry.select_range(0, tk.END)

        # 2. 相机类型
        ttk.Label(frame, text="相机硬件类型:", font=("Segoe UI", 10, "bold")).grid(row=2, column=0, sticky="w", pady=(0, 4))
        cam_type_var = tk.StringVar(value="realsense")
        type_frame = ttk.Frame(frame)
        type_frame.grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 10))
        r1 = ttk.Radiobutton(type_frame, text="RealSense D435 / D400", variable=cam_type_var, value="realsense")
        r2 = ttk.Radiobutton(type_frame, text="USB 普通/工业摄像头", variable=cam_type_var, value="usb")
        r1.pack(side=tk.LEFT, padx=(0, 16))
        r2.pack(side=tk.LEFT)

        # 3. 相机硬件标识 (序列号或设备索引)
        ttk.Label(frame, text="硬件序列号 / 设备号 (可留空自动识别):", font=("Segoe UI", 10, "bold")).grid(row=4, column=0, sticky="w", pady=(0, 4))
        serial_var = tk.StringVar(value="")
        serial_combo = ttk.Combobox(frame, textvariable=serial_var, width=40)
        serial_opts = []
        if available_devices:
            for d in available_devices.get("realsense", []):
                if d.get("serial"):
                    serial_opts.append(d["serial"])
            for d in available_devices.get("usb", []):
                serial_opts.append(str(d.get("index", 0)))
        serial_combo["values"] = serial_opts
        serial_combo.grid(row=5, column=0, columnspan=2, sticky="we", pady=(0, 10))

        # 4. 图像物理分辨率规格 (单一真理源)
        ttk.Label(frame, text="标定与作业物理分辨率 (单一真理源):", font=("Segoe UI", 10, "bold")).grid(row=6, column=0, sticky="w", pady=(0, 4))
        res_var = tk.StringVar(value="1920x1080")
        res_combo = ttk.Combobox(frame, textvariable=res_var, width=40, state="readonly")
        res_combo["values"] = ("1920x1080", "1280x720", "640x480")
        res_combo.grid(row=7, column=0, columnspan=2, sticky="we", pady=(0, 16))

        # 底部操作栏
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=8, column=0, columnspan=2, sticky="e", pady=(8, 0))

        def on_confirm():
            name = alias_var.get().strip() or default_alias
            ctype = cam_type_var.get()
            sn = serial_var.get().strip()
            res_str = res_combo.get().strip() or "1920x1080"
            try:
                rw, rh = [int(x) for x in res_str.split("x")]
            except Exception:
                rw, rh = 1920, 1080
            result["cancelled"] = False
            result["data"] = {
                "alias": name,
                "camera_type": ctype,
                "camera_serial": sn,
                "camera_resolution": [rw, rh],
            }
            root.destroy()

        def on_cancel():
            root.destroy()

        cancel_btn = ttk.Button(btn_frame, text="取消", command=on_cancel)
        cancel_btn.pack(side=tk.RIGHT, padx=(8, 0))
        confirm_btn = ttk.Button(btn_frame, text="创建工位", command=on_confirm)
        confirm_btn.pack(side=tk.RIGHT)

        root.bind("<Return>", lambda e: on_confirm())
        root.bind("<Escape>", lambda e: on_cancel())

        root.mainloop()

        return None if result["cancelled"] else result["data"]

    except Exception as e:
        log.warning(f"[prompt_create_workspace_dialog] 弹窗异常: {e}")
        return None


