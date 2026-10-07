# -*- coding: utf-8 -*-
"""
纯 OpenCV 矢量 GUI 轻量化应用基类 (BaseCvApp)
===============================================================================
提供面向桌面调试视窗的通用基础设施：
1. 窗口与缩放管理: 基于 GuiWindowManager 实现全鼠标化/等比自适应缩放呈现
2. 坐标空间映射: 物理像素坐标到逻辑画布坐标的等比逆映射 (支持居中黑边补偿)
3. 鼠标交互与命中: 维护统一的 (mouse_x, mouse_y)，提供 pt_in() 命中测试与 on_click 钩子
4. 基础微控件: 现代化带 Hover/禁用的扁平按钮 draw_btn()
5. 浮动消息反馈: Toast 提示与淡出机制
6. 标准生命周期主循环: setup() -> 循环[on_tick() -> render() -> present] -> cleanup()
"""

import time
from typing import Optional, Tuple
import cv2
import numpy as np

from src.ui.gui_window_manager import GuiWindowManager
from src.utils.text_rendering import draw_text


class BaseCvApp:
    """纯 OpenCV 交互 GUI 轻量应用基类"""

    # 默认现代深色主题配色 (BGR)
    COLOR_BG = (16, 18, 22)
    COLOR_PANEL = (24, 28, 36)
    COLOR_BORDER = (48, 56, 70)
    COLOR_ACCENT = (0, 200, 240)
    COLOR_GREEN = (0, 220, 140)
    COLOR_AMBER = (0, 180, 255)
    COLOR_TEXT = (235, 240, 248)
    COLOR_SUB = (160, 172, 188)
    COLOR_MUTED = (110, 120, 136)

    def __init__(
        self,
        app_id: str,
        base_w: int,
        base_h: int,
        window_name: str,
        window_title: str,
        settings_file: Optional[str] = None,
        min_w: int = 640,
        min_h: int = 480,
        enable_keyboard_zoom: bool = False,
        bg_color: Optional[Tuple[int, int, int]] = None,
        responsive: bool = False,
    ):
        self.app_id = app_id
        self.base_w = base_w
        self.base_h = base_h
        self.window_name = window_name
        self.window_title = window_title
        self.bg_color = bg_color or self.COLOR_BG
        self.responsive = responsive

        self.win_mgr = GuiWindowManager(
            app_id=app_id,
            base_w=base_w,
            base_h=base_h,
            min_w=min_w,
            min_h=min_h,
            settings_file=settings_file,
            enable_keyboard_zoom=enable_keyboard_zoom,
        )

        self._running = True
        self.mouse_x = -1
        self.mouse_y = -1
        self._toast = ""
        self._toast_until = 0.0

    # ==================== 生命周期钩子 (子类可选择性覆盖) ====================
    def setup(self):
        """应用初始化钩子 (在主循环开始前执行, 如连接网络/打开硬件)"""
        pass

    def on_tick(self):
        """每帧渲染前调用 (用于心跳检测、定时器检查等)"""
        pass

    def render(self) -> np.ndarray:
        """核心渲染接口: 返回尺寸为 (base_h, base_w, 3) 的画布 (子类必须实现)"""
        raise NotImplementedError("Subclasses must implement render()")

    def on_click(self, x: int, y: int):
        """鼠标左键单击简易响应钩子 (x, y 为已逆变换的逻辑画布坐标)"""
        pass

    def on_double_click(self, x: int, y: int):
        """鼠标左键双击响应钩子 (默认走 on_click)"""
        self.on_click(x, y)

    def on_mouse_down(self, x: int, y: int, button: str):
        """鼠标按下钩子 (button 为 'left' / 'right' / 'middle')"""
        if button == "left":
            self.on_click(x, y)

    def on_mouse_up(self, x: int, y: int, button: str):
        """鼠标抬起钩子 (button 为 'left' / 'right' / 'middle')"""
        pass

    def on_mouse_move(self, x: int, y: int):
        """鼠标移动响应钩子"""
        pass

    def on_mouse_wheel(self, delta: int, flags: int):
        """鼠标滚轮响应钩子 (非 Ctrl 缩放状态下的常规滚轮滚动, 用于相册/列表滚动)"""
        pass

    def on_key(self, key: int) -> bool:
        """键盘按键响应钩子, 返回 True 表示事件已被消费, False 则允许走默认逻辑 (如 ESC 退出)"""
        return False

    def stop(self):
        """退出/停止应用事件循环"""
        self._running = False

    def cleanup(self):
        """应用退出前清理钩子 (在销毁窗口后执行, 如断开网络/释放设备)"""
        pass

    # ==================== 实用交互与判定工具 ====================
    @staticmethod
    def pt_in(x: int, y: int, rect: Tuple[int, int, int, int]) -> bool:
        """矩形命中判定 (x, y, w, h)"""
        rx, ry, rw, rh = rect
        return rx <= x <= rx + rw and ry <= y <= ry + rh

    def set_toast(self, msg: str, duration: float = 4.0):
        """设置底部浮动 Toast 消息"""
        self._toast = msg
        self._toast_until = time.time() + duration

    def draw_btn(
        self,
        canvas: np.ndarray,
        rect: Tuple[int, int, int, int],
        label: str,
        mpos: Optional[Tuple[int, int]] = None,
        theme_color: Optional[Tuple[int, int, int]] = None,
        enabled: bool = True,
        bold: bool = False,
        font_size: int = 12,
        active: bool = False,
    ):
        """通用现代化扁平矩形按钮绘制 (支持 Hover 高亮、禁用置灰、选中激活态)"""
        x, y, w, h = rect
        mx, my = mpos if mpos is not None else (self.mouse_x, self.mouse_y)
        hov = enabled and self.pt_in(mx, my, rect)

        if not enabled:
            bg = (20, 24, 30)
            border = (40, 46, 56)
            text_col = self.COLOR_MUTED
        elif active:
            bg = (30, 52, 48)
            border = self.COLOR_GREEN
            text_col = (255, 255, 255)
        elif hov:
            bg = (36, 44, 56)
            border = theme_color or self.COLOR_ACCENT
            text_col = (255, 255, 255)
        else:
            bg = (26, 32, 42)
            border = theme_color or self.COLOR_BORDER
            text_col = (230, 238, 248)

        cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), border, 2 if (hov or active or theme_color) else 1)

        est_w = 8 * len(label) if label.isascii() else 14 * len(label)
        draw_text(
            canvas,
            label,
            (x + max(4, (w - est_w) // 2), y + (h - 14) // 2),
            font_size=font_size,
            color=text_col,
            bold=bold or hov or active,
        )

    def draw_toast(self, canvas: np.ndarray, pos: Optional[Tuple[int, int]] = None):
        """绘制浮动 Toast 提示"""
        if self._toast and time.time() < self._toast_until:
            tx, ty = pos if pos is not None else (24, self.base_h - 14)
            draw_text(canvas, self._toast[:100], (tx, ty), font_size=12, color=self.COLOR_GREEN, bold=True)

    # ==================== 视窗缩放与呈现 ====================
    def present_scaled(self, raw: np.ndarray) -> np.ndarray:
        """按窗口当前物理分辨率呈现 (响应式直接返回, 固定画布严格等比缩放并补黑边)"""
        cw, ch = self.win_mgr.canvas_w, self.win_mgr.canvas_h
        if (cw == self.base_w and ch == self.base_h) or (raw.shape[1] == cw and raw.shape[0] == ch):
            return raw

        present = np.full((ch, cw, 3), self.bg_color, dtype=np.uint8)
        scale = min(cw / float(self.base_w), ch / float(self.base_h))
        target_w = int(round(self.base_w * scale))
        target_h = int(round(self.base_h * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        scaled = cv2.resize(raw, (target_w, target_h), interpolation=interp)
        pad_x = (cw - target_w) // 2
        present[0:target_h, pad_x:pad_x + target_w] = scaled
        return present

    def _on_mouse_event(self, event, x, y, flags, param):
        """鼠标事件统一处理: 缩放拦截、坐标逆变换、分发点击与移动"""
        # 0. 优先处理 Ctrl + 滚轮缩放
        if event == 10:  # cv2.EVENT_MOUSEWHEEL
            handled, toast = self.win_mgr.handle_mouse_wheel(event, flags)
            if handled and toast:
                self.set_toast(toast)
                return

        # 1. 物理坐标 -> 逻辑画布坐标逆变换
        cw, ch = self.win_mgr.canvas_w, self.win_mgr.canvas_h
        if self.responsive:
            lx = max(0, min(cw - 1, x))
            ly = max(0, min(ch - 1, y))
        elif cw != self.base_w or ch != self.base_h:
            scale = min(cw / float(self.base_w), ch / float(self.base_h))
            pad_x = (cw - int(round(self.base_w * scale))) // 2
            lx = max(0, min(self.base_w - 1, int((x - pad_x) / max(1e-6, scale))))
            ly = max(0, min(self.base_h - 1, int(y / max(1e-6, scale))))
        else:
            lx = max(0, min(self.base_w - 1, x))
            ly = max(0, min(self.base_h - 1, y))

        self.mouse_x, self.mouse_y = lx, ly

        if event == cv2.EVENT_MOUSEMOVE:
            self.on_mouse_move(lx, ly)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.on_mouse_down(lx, ly, "left")
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            self.on_double_click(lx, ly)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.on_mouse_down(lx, ly, "right")
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.on_mouse_down(lx, ly, "middle")
        elif event == cv2.EVENT_LBUTTONUP:
            self.on_mouse_up(lx, ly, "left")
        elif event == cv2.EVENT_RBUTTONUP:
            self.on_mouse_up(lx, ly, "right")
        elif event == cv2.EVENT_MBUTTONUP:
            self.on_mouse_up(lx, ly, "middle")
        elif event == 10:  # cv2.EVENT_MOUSEWHEEL (普通无 Ctrl 滚轮)
            # OpenCV 中 delta 蕴含在 flags 高位中，正负代表方向
            delta = 1 if flags > 0 else -1
            self.on_mouse_wheel(delta, flags)

    # ==================== 主运行循环 ====================
    def create_window(self):
        """创建/重置 OpenCV 窗口并绑定事件回调"""
        self.win_mgr.setup_window(self.window_name, self._on_mouse_event)
        self.win_mgr.set_unicode_title(self.window_title)
        try:
            cv2.resizeWindow(self.window_name, self.win_mgr.canvas_w, self.win_mgr.canvas_h)
        except Exception:
            pass

    def run(self, wait_ms: int = 30):
        """启动应用标准事件循环"""
        self.create_window()
        self.setup()

        while self._running:
            poll_res = self.win_mgr.poll_events()
            if poll_res.should_quit:
                break
            if poll_res.toast_msg:
                self.set_toast(poll_res.toast_msg)

            self.on_tick()

            raw = self.render()
            present = self.present_scaled(raw)
            cv2.imshow(self.window_name, present)

            key = cv2.waitKeyEx(wait_ms)
            if key != -1:
                consumed = self.on_key(key)
                if not consumed and key in (27,):  # ESC 键默认退出
                    break

        # 退出清理
        self.cleanup()
        cv2.destroyAllWindows()
