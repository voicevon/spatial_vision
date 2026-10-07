#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应屏幕与视口分层渲染引擎 (ViewportManager)
=============================================================================
核心设计哲学：
  彻底解决 OpenCV 工作台在多分辨率屏幕与不同缩放比例（1080P/2K/4K/笔记本125%/150%）下
  因超屏导致底部控制栏被任务栏遮挡、以及简单缩放导致按钮与文字缩水模糊的两难矛盾。

架构机制：
  1. 【分层独立渲染】：
     - 顶部 HUD 栏与底部操作栏：直接以目标窗口的 1:1 物理像素绘制，尺寸恒定、文字锐利、绝不缩水；
     - 中间视口 (Viewport)：原始高分辨率图像（如 1920x1080）按保持宽高比 (Aspect Ratio) 自动等比缩放并居中放置；
  2. 【双向坐标精准映射】：
     - 窗口坐标 <-> 原始图像坐标无缝双向转换，确保点击画面中标靶时 100% 像素级精确；
  3. 【安全工作区动态检测】：
     - Windows 环境下自动查询任务栏排除后的净可用工作区，智能约束默认窗口尺寸，保证底部按钮 100% 常驻屏幕内。
=============================================================================
"""

import sys
from typing import Tuple, Optional
import numpy as np
import cv2

from src.ui.text_rendering import measure_text, put_text

# Windows API 可用性检测
if sys.platform == "win32":
    try:
        import ctypes
        from ctypes import wintypes
        HAVE_WIN32_API = True
    except Exception:
        HAVE_WIN32_API = False
else:
    HAVE_WIN32_API = False


def get_safe_screen_size(preferred_w: int = 1280, preferred_h: int = 720) -> Tuple[int, int]:
    """
    智能获取当前显示器安全可用窗口分辨率（避开任务栏与标题栏）
    :param preferred_w: 首选宽度 (默认 1280)
    :param preferred_h: 首选高度 (默认 720)
    :return: (safe_w, safe_h)
    """
    if HAVE_WIN32_API:
        try:
            user32 = ctypes.windll.user32
            # 兼容高 DPI 感知 (优先现代 Per-Monitor V2 感知，杜绝鼠标坐标与画面缩放位移)
            try:
                shcore = ctypes.windll.shcore
                shcore.SetProcessDpiAwareness(2)
            except Exception:
                try:
                    user32.SetProcessDPIAware()
                except Exception:
                    pass

            rect = wintypes.RECT()
            # SPI_GETWORKAREA = 48
            if user32.SystemParametersInfoW(48, 0, ctypes.byref(rect), 0):
                work_w = rect.right - rect.left
                work_h = rect.bottom - rect.top

                # 为窗口标题栏和系统边框留出 40px 的安全余量
                max_w = max(640, work_w - 30)
                max_h = max(480, work_h - 45)

                # 优先采用建议尺寸，但如果屏幕工作区较小，自动缩小并保持 16:9
                target_w = min(preferred_w, max_w)
                target_h = min(preferred_h, max_h)

                # 维持接近 16:9 的视觉比例
                if target_w / target_h > 16.0 / 9.0:
                    target_w = int(target_h * 16.0 / 9.0)
                else:
                    target_h = int(target_w * 9.0 / 16.0)

                return max(640, target_w), max(480, target_h)
        except Exception:
            pass

    # 兜底默认采用超稳定的 720P 安全分辨率
    return preferred_w, preferred_h


class ViewportManager:
    """
    视口变换与布局管理器
    """

    def __init__(self, win_w: int = 1280, win_h: int = 720, top_bar_h: int = 50, bottom_bar_h: int = 0):
        self.win_w = win_w
        self.win_h = win_h
        self.top_bar_h = top_bar_h
        self.bottom_bar_h = bottom_bar_h

        # 基础视口变换参数 (自适应窗口大小与全貌适配)
        self.scale = 1.0
        self.pad_x = 0
        self.pad_y = 0
        self.view_w = self.win_w
        self.view_h = max(100, self.win_h - self.top_bar_h - self.bottom_bar_h)
        self.img_w = 1920
        self.img_h = 1080
        self.fitted_w = 0
        self.fitted_h = 0
        self.calculate_transform(self.img_w, self.img_h)

        # 用户交互缩放与平移 (以鼠标指针为锚点无级滚轮缩放与中键平移漫游)
        self.user_zoom = 1.0        # 用户缩放倍率，范围 [1.0, 8.0]
        self.pan_x = 0.0            # 视口平移像素位移 (窗口物理像素)
        self.pan_y = 0.0
        self.is_panning = False     # 是否正在按住鼠标中键拖拽平移
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.pan_orig_x = 0.0
        self.pan_orig_y = 0.0

    def update_window_size(self, win_w: int, win_h: int):
        """窗口尺寸改变时更新布局参数"""
        self.win_w = max(480, int(win_w))
        self.win_h = max(320, int(win_h))
        self.view_w = self.win_w
        self.view_h = max(100, self.win_h - self.top_bar_h - self.bottom_bar_h)
        if self.img_w > 0 and self.img_h > 0:
            self.calculate_transform(self.img_w, self.img_h)

    def sync_window_size(self, window_name: str) -> bool:
        """
        实时感知并同步 OpenCV 窗口的真实物理尺寸（支持鼠标拖拽 Resize 与窗口最大化 Maximize）
        :param window_name: OpenCV 窗口名称
        :return: bool, 若窗口物理尺寸发生改变并成功更新布局则返回 True，否则返回 False
        """
        try:
            # 窗口未创建或已关闭时跳过
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                return False

            rect = cv2.getWindowImageRect(window_name)
            if rect and len(rect) == 4:
                _, _, cur_w, cur_h = rect
                # 过滤异常尺寸（如最小化时可能为负数或 0）
                if cur_w >= 480 and cur_h >= 320:
                    if cur_w != self.win_w or cur_h != self.win_h:
                        self.update_window_size(cur_w, cur_h)
                        return True
        except Exception:
            pass
        return False

    def calculate_transform(self, img_w: int, img_h: int):
        """
        计算原始图像等比例缩放放入视口的变换矩阵
        """
        self.img_w = img_w
        self.img_h = img_h
        self.view_w = self.win_w
        self.view_h = max(100, self.win_h - self.top_bar_h - self.bottom_bar_h)

        # 等比缩放系数
        scale_x = self.view_w / float(self.img_w)
        scale_y = self.view_h / float(self.img_h)
        self.scale = min(scale_x, scale_y)

        self.fitted_w = int(round(self.img_w * self.scale))
        self.fitted_h = int(round(self.img_h * self.scale))

        # 居中偏移量 (视口内局部坐标)
        self.pad_x = (self.view_w - self.fitted_w) // 2
        self.pad_y = (self.view_h - self.fitted_h) // 2

    def _constrain_pan(self):
        """约束平移边界，防止图像完全脱出可视视口"""
        if self.user_zoom <= 1.001:
            self.pan_x = 0.0
            self.pan_y = 0.0
            return

        total_scale = self.scale * self.user_zoom
        scaled_w = self.img_w * total_scale
        scaled_h = self.img_h * total_scale

        # 允许留出 60px 的视觉余量，防止拖丢
        margin = 60
        min_pan_x = margin - scaled_w - self.pad_x
        max_pan_x = (self.view_w - margin) - self.pad_x
        self.pan_x = max(min_pan_x, min(max_pan_x, self.pan_x))

        min_pan_y = margin - scaled_h - self.pad_y
        max_pan_y = (self.view_h - margin) - self.pad_y
        self.pan_y = max(min_pan_y, min(max_pan_y, self.pan_y))

    def handle_mouse_wheel(self, win_x: int, win_y: int, flags: int) -> bool:
        """
        处理鼠标滚轮事件：以鼠标当前 (win_x, win_y) 所在的原图像素为中心进行无级放大/缩小
        :param win_x: 鼠标在 OpenCV 窗口中的 X 坐标
        :param win_y: 鼠标在 OpenCV 窗口中的 Y 坐标
        :param flags: OpenCV 回调 flags，> 0 表示向上滚 (放大)，< 0 表示向下滚 (缩小)
        :return: bool，视口缩放或位移是否发生实质改变
        """
        vy1 = self.top_bar_h
        vy2 = self.win_h - self.bottom_bar_h
        if win_y < vy1 or win_y >= vy2 or win_x < 0 or win_x >= self.win_w:
            return False

        # 向上滚轮放大，向下滚轮缩小
        factor = 1.25 if flags > 0 else 0.80
        old_zoom = self.user_zoom
        new_zoom = max(1.0, min(8.0, old_zoom * factor))

        # 若缩小到贴近 1.0x，直接复位到完美居中
        if new_zoom <= 1.02:
            if old_zoom == 1.0 and self.pan_x == 0.0 and self.pan_y == 0.0:
                return False
            self.user_zoom = 1.0
            self.pan_x = 0.0
            self.pan_y = 0.0
            return True

        if abs(new_zoom - old_zoom) < 1e-4:
            return False

        # 以当前鼠标所在原图像素点为锚点：
        total_scale_old = self.scale * old_zoom
        total_scale_new = self.scale * new_zoom

        content_x1_old = self.pad_x + self.pan_x
        content_y1_old = vy1 + self.pad_y + self.pan_y

        rel_x = win_x - content_x1_old
        rel_y = win_y - content_y1_old

        img_x = rel_x / max(1e-6, total_scale_old)
        img_y = rel_y / max(1e-6, total_scale_old)

        # 缩放后，保持原图该像素点仍位于鼠标下方
        content_x1_new = win_x - img_x * total_scale_new
        content_y1_new = win_y - img_y * total_scale_new

        self.pan_x = content_x1_new - self.pad_x
        self.pan_y = content_y1_new - (vy1 + self.pad_y)
        self.user_zoom = new_zoom

        self._constrain_pan()
        return True

    def start_pan(self, win_x: int, win_y: int):
        """开始中键拖拽平移漫游"""
        if self.user_zoom > 1.001:
            self.is_panning = True
            self.pan_start_x = win_x
            self.pan_start_y = win_y
            self.pan_orig_x = self.pan_x
            self.pan_orig_y = self.pan_y

    def update_pan(self, win_x: int, win_y: int) -> bool:
        """更新拖拽位移"""
        if not self.is_panning:
            return False
        dx = win_x - self.pan_start_x
        dy = win_y - self.pan_start_y
        old_x, old_y = self.pan_x, self.pan_y
        self.pan_x = self.pan_orig_x + dx
        self.pan_y = self.pan_orig_y + dy
        self._constrain_pan()
        return (self.pan_x != old_x or self.pan_y != old_y)

    def end_pan(self):
        """结束拖拽平移"""
        self.is_panning = False

    def reset_zoom(self) -> bool:
        """一键复位缩放与平移回 1:1 全貌适配状态"""
        if self.user_zoom != 1.0 or self.pan_x != 0.0 or self.pan_y != 0.0:
            self.user_zoom = 1.0
            self.pan_x = 0.0
            self.pan_y = 0.0
            self.is_panning = False
            return True
        return False

    def win_to_img_coords(self, win_x: int, win_y: int) -> Tuple[Optional[int], Optional[int]]:
        """
        将窗口点击坐标转换为原始图像像素坐标 (自适应当前缩放倍率与平移偏移)
        若点击在视口外（如顶部栏、底部栏、左右黑边），返回 (None, None)
        """
        vy1 = self.top_bar_h
        vy2 = self.win_h - self.bottom_bar_h
        if win_y < vy1 or win_y >= vy2 or win_x < 0 or win_x >= self.win_w:
            return None, None

        total_scale = self.scale * self.user_zoom
        content_x1 = self.pad_x + self.pan_x
        content_y1 = vy1 + self.pad_y + self.pan_y
        content_x2 = content_x1 + self.img_w * total_scale
        content_y2 = content_y1 + self.img_h * total_scale

        if content_x1 <= win_x < content_x2 and content_y1 <= win_y < content_y2:
            img_x = int(round((win_x - content_x1) / total_scale))
            img_y = int(round((win_y - content_y1) / total_scale))
            img_x = max(0, min(self.img_w - 1, img_x))
            img_y = max(0, min(self.img_h - 1, img_y))
            return img_x, img_y
        return None, None

    def img_to_win_coords(self, img_x: float, img_y: float) -> Tuple[int, int]:
        """
        将原始图像像素坐标转换为当前窗口视口像素坐标
        """
        total_scale = self.scale * self.user_zoom
        content_x1 = self.pad_x + self.pan_x
        content_y1 = self.top_bar_h + self.pad_y + self.pan_y
        win_x = int(round(content_x1 + img_x * total_scale))
        win_y = int(round(content_y1 + img_y * total_scale))
        return win_x, win_y

    def is_in_top_bar(self, win_x: int, win_y: int) -> bool:
        """是否点击在顶部栏"""
        return 0 <= win_y < self.top_bar_h and 0 <= win_x < self.win_w

    def is_in_bottom_bar(self, win_x: int, win_y: int) -> bool:
        """是否点击在底部栏"""
        return (self.win_h - self.bottom_bar_h) <= win_y < self.win_h and 0 <= win_x < self.win_w

    def render_viewport(self, canvas: np.ndarray, content_img: np.ndarray, bg_color: Tuple[int, int, int] = (16, 17, 22)) -> np.ndarray:
        """
        将原始高分图像贴入目标窗口画布的视口区域 (支持全自动滚轮无级缩放与平移漫游)
        :param canvas: 尺寸为 (win_h, win_w, 3) 的窗口底板画布
        :param content_img: 原始尺寸的高清画面 (如 1920x1080)
        :param bg_color: 视口黑边背景填充色 (默认极简深色 #101116)
        """
        can_h, can_w = canvas.shape[:2]
        if can_h != self.win_h or can_w != self.win_w:
            self.update_window_size(can_w, can_h)

        h_c, w_c = content_img.shape[:2]
        if w_c != self.img_w or h_c != self.img_h or self.fitted_w == 0:
            self.calculate_transform(w_c, h_c)

        # 视口整体底色清理
        vy1 = self.top_bar_h
        vy2 = self.win_h - self.bottom_bar_h
        canvas[vy1:vy2, 0:self.win_w] = bg_color

        total_scale = self.scale * self.user_zoom
        content_x1 = self.pad_x + self.pan_x
        content_y1 = vy1 + self.pad_y + self.pan_y
        content_x2 = content_x1 + w_c * total_scale
        content_y2 = content_y1 + h_c * total_scale

        if self.user_zoom <= 1.001 and self.pan_x == 0.0 and self.pan_y == 0.0:
            # 基础无缩放全景适配模式：直接快速 resize 全图并居中放置
            if self.scale != 1.0:
                interp = cv2.INTER_AREA if self.scale < 1.0 else cv2.INTER_LINEAR
                resized_img = cv2.resize(content_img, (self.fitted_w, self.fitted_h), interpolation=interp)
            else:
                resized_img = content_img

            tx1 = self.pad_x
            tx2 = self.pad_x + self.fitted_w
            ty1 = vy1 + self.pad_y
            ty2 = vy1 + self.pad_y + self.fitted_h
            canvas[ty1:ty2, tx1:tx2] = resized_img
            cv2.rectangle(canvas, (tx1 - 1, ty1 - 1), (tx2, ty2), (45, 48, 58), 1)

        else:
            # 动态可见 ROI 局部裁剪模式 (零卡顿、极速高效，支持超大倍率缩放)
            dst_x1 = max(0, int(np.floor(content_x1)))
            dst_x2 = min(self.win_w, int(np.ceil(content_x2)))
            dst_y1 = max(vy1, int(np.floor(content_y1)))
            dst_y2 = min(vy2, int(np.ceil(content_y2)))

            if dst_x2 > dst_x1 and dst_y2 > dst_y1:
                # 反向推算在原图 content_img 上的可见采样区域
                src_x1 = max(0.0, (dst_x1 - content_x1) / total_scale)
                src_x2 = min(float(w_c), (dst_x2 - content_x1) / total_scale)
                src_y1 = max(0.0, (dst_y1 - content_y1) / total_scale)
                src_y2 = min(float(h_c), (dst_y2 - content_y1) / total_scale)

                ix1, iy1 = int(np.floor(src_x1)), int(np.floor(src_y1))
                ix2, iy2 = int(np.ceil(src_x2)), int(np.ceil(src_y2))
                ix1, iy1 = max(0, min(w_c - 1, ix1)), max(0, min(h_c - 1, iy1))
                ix2, iy2 = max(ix1 + 1, min(w_c, ix2)), max(iy1 + 1, min(h_c, iy2))

                crop = content_img[iy1:iy2, ix1:ix2]
                dst_w = dst_x2 - dst_x1
                dst_h = dst_y2 - dst_y1

                if crop.size > 0 and dst_w > 0 and dst_h > 0:
                    interp = cv2.INTER_LINEAR if total_scale >= 1.0 else cv2.INTER_AREA
                    resized_crop = cv2.resize(crop, (dst_w, dst_h), interpolation=interp)
                    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = resized_crop

            # 右上角浮层显示缩放倍率胶囊徽章
            zoom_pill_w = 215
            zoom_pill_h = 26
            zx1 = self.win_w - zoom_pill_w - 12
            zy1 = vy1 + 10
            zx2 = zx1 + zoom_pill_w
            zy2 = zy1 + zoom_pill_h

            pill_overlay = canvas.copy()
            cv2.rectangle(pill_overlay, (zx1, zy1), (zx2, zy2), (20, 24, 32), -1)
            cv2.addWeighted(pill_overlay, 0.85, canvas, 0.15, 0, canvas)
            cv2.rectangle(canvas, (zx1, zy1), (zx2, zy2), (0, 215, 255), 1)

            pill_txt = f"[{self.user_zoom:.1f}x 放大 | 中键拖拽 | 0复位]"
            put_text(canvas, pill_txt, (zx1 + 8, zy1 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        return canvas


def draw_styled_button(
    canvas: np.ndarray,
    rect: Tuple[int, int, int, int],
    label: str,
    mouse_pos: Tuple[int, int] = (-1, -1),
    btn_type: str = "normal",  # normal, primary, success, danger, warning, purple, info
    is_active: bool = False,
    is_enabled: bool = True,
    font_scale: float = 0.42,
) -> bool:
    """
    绘制符合统一工业设计规范的现代化按钮（纯 ASCII/标准中文，严禁 Emoji，杜绝乱码）
    :param canvas: 目标图像
    :param rect: (x1, y1, x2, y2)
    :param label: 纯文本标签 (严禁包含 Emoji)
    :param mouse_pos: 当前鼠标像素坐标
    :param btn_type: 按钮语义类型 (normal/primary/success/danger/warning/purple/info)
    :param is_active: 是否处于激活/按下状态
    :param is_enabled: 是否可用
    :param font_scale: 字号缩放
    :return: is_hovered
    """
    x1, y1, x2, y2 = rect
    mx, my = mouse_pos
    is_hover = is_enabled and (x1 <= mx <= x2 and y1 <= my <= y2)

    PALETTE = {
        "normal":  {"bg": (38, 40, 48),  "border": (70, 75, 90),   "text": (220, 225, 230)},
        "primary": {"bg": (70, 48, 22),  "border": (220, 140, 0),  "text": (255, 255, 255)}, # 科技蓝
        "info":    {"bg": (65, 50, 25),  "border": (230, 160, 40),  "text": (255, 255, 255)}, # 靛蓝/青蓝
        "success": {"bg": (25, 65, 35),  "border": (0, 215, 90),   "text": (255, 255, 255)}, # 成功绿
        "danger":  {"bg": (35, 30, 75),  "border": (70, 60, 210),  "text": (255, 255, 255)}, # 警示红
        "warning": {"bg": (25, 60, 90),  "border": (0, 180, 240),  "text": (255, 255, 255)}, # 琥珀黄
        "purple":  {"bg": (65, 30, 65),  "border": (200, 80, 220), "text": (255, 255, 255)}, # 审核紫
    }
    theme = PALETTE.get(btn_type, PALETTE["normal"])
    bg_col = list(theme["bg"])
    border_col = list(theme["border"])
    text_col = list(theme["text"])

    if not is_enabled:
        bg_col = (28, 28, 32)
        border_col = (45, 45, 52)
        text_col = (105, 105, 115)
    elif is_active:
        bg_col = [min(255, c + 40) for c in bg_col]
        border_col = (255, 255, 255)
    elif is_hover:
        bg_col = [min(255, c + 25) for c in bg_col]
        border_col = [min(255, c + 50) for c in border_col]

    cv2.rectangle(canvas, (x1, y1), (x2, y2), tuple(bg_col), -1)
    border_thickness = 2 if (is_hover or is_active) else 1
    cv2.rectangle(canvas, (x1, y1), (x2, y2), tuple(border_col), border_thickness)

    # 文本居中对齐
    (lw, lh), _ = measure_text(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    tx = x1 + max(4, (x2 - x1 - lw) // 2)
    ty = y1 + (y2 - y1 + lh) // 2
    put_text(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, tuple(text_col), 1, cv2.LINE_AA)
    return is_hover


def draw_segmented_toggle(
    canvas: np.ndarray,
    rect: Tuple[int, int, int, int],
    options: list,
    active_idx: int = 0,
    mouse_pos: Tuple[int, int] = (-1, -1),
    shortcut: str = "",
    active_color: Tuple[int, int, int] = (150, 95, 20),
    active_key: Optional[str] = None,
    **kwargs
) -> list:
    """
    绘制现代工业级分段胶囊乒乓开关 (Segmented Control，状态一目了然，绝不混淆)
    :param canvas: 目标画布
    :param rect: (x1, y1, x2, y2)
    :param options: [ (key, "选项显示文字"), ... ]
    :param active_idx: 当前激活的选项索引 (0-based)
    :param mouse_pos: (mx, my)
    :param shortcut: 快捷键提示 (如 "Tab")
    :param active_color: 激活状态块的高亮底色 (BGR)
    :param active_key: 当前激活选项的 key (若传入则覆盖 active_idx)
    :return: 各选项点击区域 [(key, (sx1, sy1, sx2, sy2))]
    """
    if active_key is not None:
        for i, opt in enumerate(options):
            if isinstance(opt, (list, tuple)) and str(opt[0]) == str(active_key):
                active_idx = i
                break
    x1, y1, x2, y2 = rect
    mx, my = mouse_pos
    total_w = x2 - x1
    num_opts = max(1, len(options))
    seg_w = total_w // num_opts

    # 绘制外层胶囊边框与深色底板
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (28, 30, 36), -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (65, 70, 82), 1)

    result_sub_rects = []

    for i, (opt_key, opt_title) in enumerate(options):
        sx1 = x1 + i * seg_w
        sx2 = sx1 + seg_w if (i < num_opts - 1) else x2
        is_active = (i == active_idx)
        is_hover = (sx1 <= mx <= sx2 and y1 <= my <= y2)

        if is_active:
            # 激活态：高亮纯色底板 + 纯白文字 + 细白边框
            cv2.rectangle(canvas, (sx1 + 1, y1 + 1), (sx2 - 1, y2 - 1), active_color, -1)
            cv2.rectangle(canvas, (sx1 + 1, y1 + 1), (sx2 - 1, y2 - 1), (255, 255, 255), 1)
            bullet = "[*]"
            txt_color = (255, 255, 255)
            font_thick = 2
        else:
            # 未激活态：暗底，悬停时微亮提示可点击
            seg_bg = (44, 48, 58) if is_hover else (28, 30, 36)
            cv2.rectangle(canvas, (sx1 + 1, y1 + 1), (sx2 - 1, y2 - 1), seg_bg, -1)
            bullet = "[ ]"
            txt_color = (170, 175, 185) if not is_hover else (225, 230, 235)
            font_thick = 1

        disp_txt = f"{bullet} {opt_title}"
        if i == num_opts - 1 and shortcut:
            disp_txt += f" ({shortcut})"

        (tw, th), _ = measure_text(disp_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.40, font_thick)
        tx = sx1 + max(4, (sx2 - sx1 - tw) // 2)
        ty = y1 + (y2 - y1 + th) // 2
        put_text(canvas, disp_txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, txt_color, font_thick, cv2.LINE_AA)

        # 中间分隔线
        if i > 0:
            cv2.line(canvas, (sx1, y1 + 3), (sx1, y2 - 3), (55, 60, 72), 1)

        result_sub_rects.append((opt_key, (sx1, y1, sx2, y2)))

    return result_sub_rects


def draw_dropdown_box(
    canvas: np.ndarray,
    rect: Tuple[int, int, int, int],
    options: list,  # [ (key, "显示文字"), ... ]
    active_key: str,
    is_open: bool,
    mouse_pos: Tuple[int, int] = (-1, -1),
    shortcut: str = "F",
    active_color: Tuple[int, int, int] = (150, 95, 20)
) -> Tuple[list, Optional[Tuple[int, int, int, int]]]:
    """
    绘制现代工业级下拉选择框 (Drop-down ComboBox / Popover Menu)
    :param canvas: 目标画布
    :param rect: (x1, y1, x2, y2) 主选择框区域
    :param options: [ (key, "选项显示文字"), ... ]
    :param active_key: 当前激活选项的 key
    :param is_open: 下拉菜单是否展开
    :param mouse_pos: (mx, my) 当前鼠标像素坐标
    :param shortcut: 快捷键提示 (如 "F")
    :param active_color: 展开或激活高亮色 (BGR)
    :return: (item_rects, menu_bounding_rect)
             item_rects: [ (key, (ix1, iy1, ix2, iy2)), ... ]
             menu_bounding_rect: 菜单整体覆盖外接矩形 (mx1, my1, mx2, my2) 用于点击外部检测
    """
    x1, y1, x2, y2 = rect
    mx, my = mouse_pos
    is_hover_main = (x1 <= mx <= x2 and y1 <= my <= y2)

    # 找到当前激活选项的显示文字
    current_label = "请选择"
    for k, lbl in options:
        if k == active_key:
            current_label = lbl
            break

    # 1. 绘制主选择框
    bg_main = (45, 48, 58) if (is_hover_main or is_open) else (28, 30, 36)
    border_color = (255, 255, 255) if is_open else ((180, 190, 210) if is_hover_main else (65, 70, 82))
    cv2.rectangle(canvas, (x1, y1), (x2, y2), bg_main, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, 2 if (is_open or is_hover_main) else 1)

    # 主框文本与箭头
    arrow_char = "^" if is_open else "v"
    display_text = f"{current_label}"
    if shortcut:
        display_text += f" ({shortcut})"
    display_text += f"  {arrow_char}"

    (tw, th), _ = measure_text(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    tx = x1 + max(6, (x2 - x1 - tw) // 2)
    ty = y1 + (y2 - y1 + th) // 2
    txt_color = (255, 255, 255) if (is_open or is_hover_main) else (210, 215, 225)
    put_text(canvas, display_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, txt_color, 1, cv2.LINE_AA)

    item_rects = [("DROPDOWN_TOGGLE", (x1, y1, x2, y2))]
    menu_bounding_rect = None

    # 2. 若处于展开状态，绘制下拉悬浮菜单 (Popover)
    if is_open and options:
        item_h = 32
        menu_w = max(x2 - x1, 185)
        menu_h = len(options) * item_h + 8
        mx1 = x1
        my1 = y2 + 3
        mx2 = mx1 + menu_w
        my2 = my1 + menu_h
        menu_bounding_rect = (mx1, my1, mx2, my2)

        # 菜单半透明深色阴影与底板
        overlay = canvas.copy()
        cv2.rectangle(overlay, (mx1, my1), (mx2, my2), (18, 20, 25), -1)
        cv2.addWeighted(overlay, 0.96, canvas, 0.04, 0, canvas)
        cv2.rectangle(canvas, (mx1, my1), (mx2, my2), (90, 95, 115), 1)

        # 绘制每一个菜单项
        for idx, (opt_key, opt_lbl) in enumerate(options):
            iy1 = my1 + 4 + idx * item_h
            iy2 = iy1 + item_h
            ix1 = mx1 + 4
            ix2 = mx2 - 4
            is_active = (opt_key == active_key)
            is_hover_item = (ix1 <= mx <= ix2 and iy1 <= my <= iy2)

            if is_active:
                cv2.rectangle(canvas, (ix1, iy1), (ix2, iy2), active_color, -1)
                cv2.rectangle(canvas, (ix1, iy1), (ix2, iy2), (255, 255, 255), 1)
                bullet = "[*]"
                col = (255, 255, 255)
                thick = 2
            elif is_hover_item:
                cv2.rectangle(canvas, (ix1, iy1), (ix2, iy2), (55, 60, 75), -1)
                bullet = "[ ]"
                col = (255, 255, 255)
                thick = 1
            else:
                bullet = "[ ]"
                col = (180, 185, 195)
                thick = 1

            item_text = f" {bullet} {opt_lbl}"
            put_text(canvas, item_text, (ix1 + 8, iy1 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, thick, cv2.LINE_AA)

            item_rects.append((f"SELECT_{opt_key}", (ix1, iy1, ix2, iy2)))

            # 分割线
            if idx < len(options) - 1:
                cv2.line(canvas, (ix1 + 6, iy2), (ix2 - 6, iy2), (40, 44, 55), 1)

    return item_rects, menu_bounding_rect

