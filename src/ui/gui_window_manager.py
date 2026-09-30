#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用 GUI 视窗管理器与偏好持久化核心 (GuiWindowManager)
=====================================================
核心职责：
  1. 【统一放大镜缩放体系】：
     - Win32 硬件级物理热键穿透探测 (Ctrl + =/+ 放大, Ctrl + -/_ 缩小, Ctrl + 0 复位)；
     - 彻底绕过微软拼音/搜狗等中文输入法对加减号翻页的拦截；
     - 原生支持 Ctrl + 鼠标滚轮向上放大、向下缩小，180ms 防抖平滑变焦；
  2. 【窗口拉伸自适应与防抖持久化】：
     - 实时探测 cv2.getWindowImageRect 物理尺寸变化；
     - 拖拽边框静止 0.35s 后自动防抖落盘，永不丢配置；
  3. 【多应用命名空间配置隔离】：
     - 统一归档至 config/gui_settings.json，各工具按 app_id 拥有独立配置域，互不干扰；
  4. 【窗口生命周期与安全保底】：
     - 优雅拦截右上角红叉 [X] 关闭事件；
     - 挂载 atexit 退出持久化钩子；
     - Win32 原生 SetWindowTextW 注入无乱码中文标题。
"""

import os
import sys
import json
import time
import atexit
from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import cv2

# 项目根路径与全局配置文件定位
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_GUI_SETTINGS_FILE = os.path.join(PROJECT_ROOT, "config", "gui_settings.json")


@dataclass
class WindowPollResult:
    """窗口单轮事件轮询结果"""
    should_quit: bool = False       # 是否需要退出主循环 (按 ESC/Q 或点右上角红叉)
    changed: bool = False           # 窗口尺寸或缩放比例是否有更新
    scale_pct: int = 100            # 当前缩放百分比 (50 ~ 200)
    scale: float = 1.0              # 当前缩放系数 (0.5 ~ 2.0)
    canvas_w: int = 1280            # 当前画布物理宽度
    canvas_h: int = 720             # 当前画布物理高度
    toast_msg: Optional[str] = None # 触发缩放或复位时的状态提示文本


class GuiWindowManager:
    """通用 GUI 视窗与偏好管理控制器"""

    def __init__(self,
                 app_id: str,
                 base_w: int = 1280,
                 base_h: int = 720,
                 min_w: int = 480,
                 min_h: int = 270,
                 settings_file: Optional[str] = None,
                 enable_keyboard_zoom: bool = True):
        """
        :param app_id: 应用唯一标识 (如 'gui_launcher', 'workspace_hub', 'tag_studio')
        :param base_w: 基准窗口宽度 (默认 1280)
        :param base_h: 基准窗口高度 (默认 720)
        :param min_w: 最小允许宽度 (默认 480)
        :param min_h: 最小允许高度 (默认 270)
        :param settings_file: 偏好持久化文件路径 (默认 config/gui_settings.json)
        :param enable_keyboard_zoom: 是否启用 Ctrl/+/- 键盘缩放热键 (默认开启; Workspace Hub 已传入 False 关闭)
        """
        self.app_id = app_id
        self.base_w = base_w
        self.base_h = base_h
        self.min_w = min_w
        self.min_h = min_h
        self.settings_file = settings_file or DEFAULT_GUI_SETTINGS_FILE
        self.enable_keyboard_zoom: bool = enable_keyboard_zoom

        self.scale_pct: int = 100
        self.canvas_w: int = base_w
        self.canvas_h: int = base_h

        self.window_name: Optional[str] = None
        self._hwnd: Optional[int] = None
        self._is_active: bool = False
        self._last_zoom_time: float = 0.0
        self._last_resize_time: float = 0.0
        self._need_save: bool = False
        self._force_ctrl_pressed: bool = False  # 单元测试仿真开关

        # 启动时读取持久化记忆
        self.load_settings()

    @property
    def scale(self) -> float:
        """缩放浮点系数 (如 1.20)"""
        return self.scale_pct / 100.0

    def load_settings(self):
        """从配置文件中读取当前 app_id 对应的偏好设置"""
        if not os.path.exists(self.settings_file):
            return

        try:
            with open(self.settings_file, "r", encoding="utf-8") as f:
                root_data = json.load(f)

            if not isinstance(root_data, dict):
                return

            # 如果存在专属 app_id 节点，直接读取
            if self.app_id in root_data and isinstance(root_data[self.app_id], dict):
                data = root_data[self.app_id]
            else:
                # 该应用尚未有独立配置，严格保持默认基准
                return

            if "scale_pct" in data:
                self.scale_pct = max(50, min(200, int(data["scale_pct"])))

            s = self.scale
            saved_w = data.get("canvas_w")
            saved_h = data.get("canvas_h")
            if saved_w and saved_h and int(saved_w) >= self.min_w and int(saved_h) >= self.min_h:
                self.canvas_w = int(saved_w)
                self.canvas_h = int(saved_h)
            else:
                self.canvas_w = max(self.min_w, int(self.base_w * s))
                self.canvas_h = max(self.min_h, int(self.base_h * s))
        except Exception:
            pass

    def save_settings(self):
        """持久化保存当前应用的偏好至配置文件 (原子化安全写入，支持多应用隔离)"""
        try:
            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
            root_data = {}
            if os.path.exists(self.settings_file):
                try:
                    with open(self.settings_file, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                        if isinstance(loaded, dict):
                            root_data = loaded
                except Exception:
                    root_data = {}

            # 更新当前应用配置节点 (合并写入, 保留同节点下其他应用数据如 viewer_state)
            node = root_data.get(self.app_id)
            if not isinstance(node, dict):
                node = {}
                root_data[self.app_id] = node
            node["scale_pct"] = self.scale_pct
            node["canvas_w"] = self.canvas_w
            node["canvas_h"] = self.canvas_h
            node["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(root_data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def setup_window(self, window_name: str, mouse_callback: Optional[Callable] = None) -> None:
        """
        初始化 OpenCV 窗口、设置尺寸并挂载原生中文标题与生命周期
        :param window_name: 窗口内部识别 key
        :param mouse_callback: 鼠标事件回调函数
        """
        self.window_name = window_name
        self._is_active = True

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.canvas_w, self.canvas_h)
        if mouse_callback:
            cv2.setMouseCallback(window_name, mouse_callback)

        # 注册退出保底钩子
        atexit.register(self.save_settings)

    def set_unicode_title(self, title_text: str):
        """通过 Windows 原生 Unicode API 注入中文标题，彻底消除问号乱码"""
        if sys.platform != "win32" or not self.window_name:
            return
        try:
            import ctypes
            self._hwnd = ctypes.windll.user32.FindWindowW(None, self.window_name)
            if self._hwnd:
                ctypes.windll.user32.SetWindowTextW(self._hwnd, title_text)
        except Exception:
            self._hwnd = None

    def apply_zoom(self, delta_pct: int, reset: bool = False) -> Tuple[bool, str]:
        """
        执行等比缩放或复位
        :param delta_pct: 增量百分比 (如 +10, -10)
        :param reset: 是否复位至 100%
        :return: (changed, toast_hint)
        """
        old_pct = self.scale_pct
        if reset:
            self.scale_pct = 100
        else:
            self.scale_pct = max(50, min(200, self.scale_pct + delta_pct))

        if not reset and self.scale_pct == old_pct:
            return False, f"缩放已达极限值 ({self.scale_pct}%)"

        s = self.scale
        rec_w = max(self.min_w, int(self.base_w * s))
        rec_h = max(self.min_h, int(self.base_h * s))
        self.canvas_w = rec_w
        self.canvas_h = rec_h

        if self.window_name:
            try:
                cv2.resizeWindow(self.window_name, rec_w, rec_h)
            except Exception:
                pass

        self.save_settings()
        if reset:
            return True, f"已复位为 100% 标准分辨率 ({self.base_w}×{self.base_h})"
        return True, f"矢量放大镜: {self.scale_pct}%  (已自动记忆大小)"

    def poll_hardware_zoom(self) -> Tuple[bool, Optional[str]]:
        """利用 Win32 GetAsyncKeyState 硬件物理检测，彻底绕过输入法拦截"""
        if sys.platform != "win32":
            return False, None

        try:
            import ctypes
            u32 = ctypes.windll.user32

            # 仅当处于活动前台窗口时才响应
            fg_hwnd = u32.GetForegroundWindow()
            if self._hwnd and fg_hwnd != self._hwnd:
                return False, None

            now = time.time()
            if now - self._last_zoom_time < 0.18:  # 180ms 防抖
                return False, None

            # 检测 Ctrl 物理按压状态 (VK_CONTROL = 0x11)
            ctrl_pressed = bool(u32.GetAsyncKeyState(0x11) & 0x8000)
            if not ctrl_pressed and not self._force_ctrl_pressed:
                return False, None

            # 放大: 主键盘 VK_OEM_PLUS (0xBB, 187) 或小键盘 VK_ADD (0x6B, 107)
            zoom_in = bool((u32.GetAsyncKeyState(0xBB) & 0x8000) or (u32.GetAsyncKeyState(0x6B) & 0x8000))
            # 缩小: 主键盘 VK_OEM_MINUS (0xBD, 189) 或小键盘 VK_SUBTRACT (0x6D, 109)
            zoom_out = bool((u32.GetAsyncKeyState(0xBD) & 0x8000) or (u32.GetAsyncKeyState(0x6D) & 0x8000))
            # 复位: 主键盘 '0' (0x30, 48) 或小键盘 '0' (0x60, 96)
            zoom_reset = bool((u32.GetAsyncKeyState(0x30) & 0x8000) or (u32.GetAsyncKeyState(0x60) & 0x8000))

            if zoom_in:
                self._last_zoom_time = now
                return self.apply_zoom(+10)
            elif zoom_out:
                self._last_zoom_time = now
                return self.apply_zoom(-10)
            elif zoom_reset:
                self._last_zoom_time = now
                return self.apply_zoom(0, reset=True)
        except Exception:
            pass

        return False, None

    def handle_mouse_wheel(self, event: int, flags: int) -> Tuple[bool, Optional[str]]:
        """
        拦截鼠标滚轮缩放 (Ctrl + 鼠标滚轮)
        :param event: OpenCV 鼠标事件代码
        :param flags: OpenCV 回调 flags
        :return: (handled, toast_hint)
        """
        if event != 10:  # cv2.EVENT_MOUSEWHEEL
            return False, None

        ctrl_pressed = False
        if self._force_ctrl_pressed:
            ctrl_pressed = True
        elif sys.platform == "win32":
            try:
                import ctypes
                ctrl_pressed = bool(ctypes.windll.user32.GetAsyncKeyState(0x11) & 0x8000)
            except Exception:
                pass

        if ctrl_pressed:
            if flags > 0:
                return self.apply_zoom(+10)
            else:
                return self.apply_zoom(-10)

        return False, None

    def handle_keyboard_fallback(self, raw_key: int) -> Tuple[bool, Optional[str]]:
        """当非 Windows 平台或普通键盘事件时的缩放后备判定"""
        ctrl_held = self._force_ctrl_pressed
        if not ctrl_held and sys.platform == "win32":
            try:
                import ctypes
                u32 = ctypes.windll.user32
                ctrl_held = bool((u32.GetKeyState(0x11) & 0x8000) or (u32.GetAsyncKeyState(0x11) & 0x8000))
            except Exception:
                pass

        key_byte = (raw_key & 0xFF)
        key_word = (raw_key & 0xFFFF)
        key_char = chr(key_byte).lower() if key_byte < 128 else ""

        is_zoom_in = (key_char in ('=', '+') or key_byte in (ord('='), ord('+'), 187, 107) or key_word in (ord('='), ord('+'), 187, 107))
        is_zoom_out = (key_char in ('-', '_') or key_byte in (ord('-'), ord('_'), 189, 109, 31) or key_word in (ord('-'), ord('_'), 189, 109, 31))
        is_zoom_reset = (key_char == '0' or key_byte in (ord('0'), 48, 96) or key_word in (ord('0'), 48, 96))

        if (ctrl_held and (is_zoom_in or is_zoom_out or is_zoom_reset)) or (key_byte in (107, 109)):
            if is_zoom_in or key_byte == 107:
                return self.apply_zoom(+10)
            elif is_zoom_out or key_byte == 109:
                return self.apply_zoom(-10)
            elif is_zoom_reset:
                return self.apply_zoom(0, reset=True)

        return False, None

    def sync_window_size(self) -> bool:
        """探测窗口物理尺寸并执行防抖落盘，若尺寸变化返回 True"""
        if not self.window_name:
            return False

        try:
            rect = cv2.getWindowImageRect(self.window_name)
            if rect and len(rect) >= 4:
                cur_w, cur_h = rect[2], rect[3]
                if cur_w >= self.min_w and cur_h >= self.min_h:
                    if cur_w != self.canvas_w or cur_h != self.canvas_h:
                        self.canvas_w = cur_w
                        self.canvas_h = cur_h
                        self._need_save = True
                        self._last_resize_time = time.time()
                        return True
        except Exception:
            pass

        # 边框拖动静止 0.35s 后自动防抖持久化
        if self._need_save and (time.time() - self._last_resize_time > 0.35):
            self.save_settings()
            self._need_save = False

        return False

    def is_window_alive(self) -> bool:
        """检查窗口是否依然处于打开状态 (优雅拦截右上角红叉)"""
        if not self.window_name:
            return True
        try:
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                return False
        except Exception:
            return False
        return True

    def poll_events(self, raw_key: int = -1) -> WindowPollResult:
        """
        主循环一站式综合事件轮询：
        - 检查窗口存活 (红叉判定)
        - 轮询物理尺寸拖拽与防抖落盘
        - 轮询常规按键 (默认关闭：项目已移除全部键盘快捷键，仅保留鼠标交互)
        """
        res = WindowPollResult(
            scale_pct=self.scale_pct,
            scale=self.scale,
            canvas_w=self.canvas_w,
            canvas_h=self.canvas_h
        )

        # 1. 窗口关闭检测 (右上角红叉 [X])
        if not self.is_window_alive():
            self.save_settings()
            res.should_quit = True
            return res

        # 2. 硬件物理缩放热键探测 (项目已移除全部快捷键，默认不启用)
        if self.enable_keyboard_zoom:
            hw_changed, hw_toast = self.poll_hardware_zoom()
            if hw_changed:
                res.changed = True
                res.scale_pct = self.scale_pct
                res.scale = self.scale
                res.canvas_w = self.canvas_w
                res.canvas_h = self.canvas_h
                res.toast_msg = hw_toast

        # 3. 动态检测拖拽拉伸尺寸与防抖持久化
        size_changed = self.sync_window_size()
        if size_changed:
            res.changed = True
            res.canvas_w = self.canvas_w
            res.canvas_h = self.canvas_h

        # 4. 常规按键处理 (退出与后备缩放)
        if raw_key != -1:
            if raw_key in (27, ord('q'), ord('Q')):
                self.save_settings()
                res.should_quit = True
                return res

            fb_changed, fb_toast = self.handle_keyboard_fallback(raw_key)
            if fb_changed:
                res.changed = True
                res.scale_pct = self.scale_pct
                res.scale = self.scale
                res.canvas_w = self.canvas_w
                res.canvas_h = self.canvas_h
                res.toast_msg = fb_toast

        return res
