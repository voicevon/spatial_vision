#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局基础 GUI 交互控件库 (gui_components)
================================================================================
跨应用通用的标准视觉交互控件：
1. draw_dropdown_button: 统一现代微质感下拉菜单头部按钮 (支持 Hover 微光、展开态高亮、强调色)
2. render_dropdown_popup: 统一置顶悬浮下拉选项浮层 (半透明磨砂遮罩、当前选中态发光、自动计算弹窗坐标)
3. draw_dashboard_button: 统一工业风碳灰卡片按钮 (支持语义侧条、运行态金色发光、禁用态)
4. render_floating_tooltip: 统一高对比度科技悬浮气泡浮层 (智能边界贴靠避让与翻转、语义前缀高亮、磨砂半透明融合、圆角质感)
5. draw_rounded_rectangle: 统一抗锯齿圆角矩形绘制函数 (支持实体圆角填充与平滑发光描边)

所有控件均深度绑定 src.ui.gui_theme.GuiTheme 单源调色板。
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

from src.ui.gui_theme import GuiTheme
from src.ui.text_rendering import draw_text, get_cached_font, measure_text, put_text
from src.ui.dialog_utils import (
    show_error_dialog,
    show_critical_message,
    prompt_confirm,
    prompt_input_text,
)


def draw_dropdown_button(
    canvas: np.ndarray,
    rect: Tuple[int, int, int, int],
    label: str,
    is_open: bool,
    mouse_pos: Tuple[int, int] = (-1, -1),
    prefix: str = "",
    theme_color: Optional[Tuple[int, int, int]] = None,
    font_size: int = 13,
) -> bool:
    """绘制统一的现代微质感下拉菜单头部按钮

    Args:
        canvas: 目标画布 (BGR)
        rect: (x1, y1, x2, y2) 坐标范围
        label: 按钮文本
        is_open: 当前是否处于展开状态
        mouse_pos: 当前鼠标坐标 (mx, my)，用于计算 hover 状态
        prefix: 前缀文本 (例如 "场景: ")
        theme_color: 自定义高亮色 (BGR)，默认使用 GuiTheme.BORDER_SEL
        font_size: 字体大小

    Returns:
        bool: 当前鼠标是否悬停在该按钮上
    """
    x1, y1, x2, y2 = rect
    mx, my = mouse_pos
    is_hover = (x1 <= mx <= x2 and y1 <= my <= y2)

    active_col = theme_color if theme_color is not None else GuiTheme.BORDER_SEL

    if is_open:
        bg_col = GuiTheme.CARD_SEL
        border_col = active_col
        text_col = GuiTheme.WHITE
        arrow = "▲"
    elif is_hover:
        bg_col = GuiTheme.CARD_HOVER
        border_col = active_col
        text_col = GuiTheme.BTN_TEXT_HOVER
        arrow = "▼"
    else:
        bg_col = GuiTheme.CARD_BG
        border_col = GuiTheme.BORDER
        text_col = GuiTheme.BTN_TEXT
        arrow = "▼"

    cv2.rectangle(canvas, (x1, y1), (x2, y2), bg_col, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 1)

    display_txt = f"{prefix}{label} {arrow}" if prefix else f"{label} {arrow}"

    try:
        font = get_cached_font(font_size, bold=(is_open or is_hover))
        bbox = font.getbbox(display_txt)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = x1 + max(6, (x2 - x1 - tw) // 2 - bbox[0])
        ty = y1 + (y2 - y1 - th) // 2 - bbox[1]
        draw_text(canvas, display_txt, (tx, ty), font_size=font_size, color=text_col, bold=(is_open or is_hover))
    except Exception:
        (tw, th), _ = measure_text(display_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        tx = x1 + max(6, (x2 - x1 - tw) // 2)
        ty = y1 + (y2 - y1 + th) // 2
        put_text(canvas, display_txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, text_col, 1, cv2.LINE_AA)

    return is_hover


def render_dropdown_popup(
    canvas: np.ndarray,
    anchor_rect: Tuple[int, int, int, int],
    options: Sequence[Tuple[str, str]],
    active_key: str,
    btn_prefix: str = "DD_",
    item_h: int = 30,
    min_width: int = 210,
    max_visible: int = 15,
    mouse_pos: Tuple[int, int] = (-1, -1),
    font_size: int = 12,
) -> List[Tuple[str, Tuple[int, int, int, int], str]]:
    """在画布上渲染置顶悬浮下拉选项列表浮层

    Args:
        canvas: 目标画布 (BGR)
        anchor_rect: 触发按钮的矩形 (x1, y1, x2, y2)，用于决定弹窗位置
        options: 选项序列 [(key, display_label), ...]
        active_key: 当前激活的选项 key
        btn_prefix: 注册的按钮 ID 前缀，生成的按钮 ID 为 f"{btn_prefix}{idx}"
        item_h: 每项高度 (px)
        min_width: 浮层最小宽度 (px)
        max_visible: 最多展示行数
        mouse_pos: 鼠标当前坐标 (mx, my) 用于 hover 高亮
        font_size: 字体大小

    Returns:
        List[Tuple[str, Tuple[int, int, int, int], str]]: 生成的交互按钮注册列表 [(btn_id, item_rect, key), ...]
    """
    if not options:
        return []

    rx1, ry1, rx2, ry2 = anchor_rect
    pop_w = max(rx2 - rx1, min_width)
    pop_x1 = rx1
    pop_y1 = ry2 + 2

    ch, cw = canvas.shape[:2]
    if pop_x1 + pop_w > cw - 8:
        pop_x1 = max(8, cw - pop_w - 8)

    disp_opts = options[:max_visible]
    pop_x2 = pop_x1 + pop_w
    pop_y2 = min(ch - 8, pop_y1 + len(disp_opts) * item_h + 6)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (pop_x1, pop_y1), (pop_x2, pop_y2), (24, 28, 36), -1)
    cv2.addWeighted(overlay, 0.96, canvas, 0.04, 0, canvas)
    cv2.rectangle(canvas, (pop_x1, pop_y1), (pop_x2, pop_y2), GuiTheme.BORDER_SEL, 1)

    registered_buttons = []
    mx, my = mouse_pos
    for i, (key, label) in enumerate(disp_opts):
        iy1 = pop_y1 + 3 + i * item_h
        iy2 = iy1 + item_h
        if iy2 > pop_y2 - 2:
            break

        is_active = (key == active_key)
        item_rect = (pop_x1 + 2, iy1, pop_x2 - 2, iy2)
        is_hover = (item_rect[0] <= mx <= item_rect[2] and item_rect[1] <= my <= item_rect[3])

        if is_active:
            cv2.rectangle(canvas, (pop_x1 + 2, iy1), (pop_x2 - 2, iy2), GuiTheme.CARD_SEL, -1)
            text_color = GuiTheme.ACCENT
        elif is_hover:
            cv2.rectangle(canvas, (pop_x1 + 2, iy1), (pop_x2 - 2, iy2), GuiTheme.CARD_HOVER, -1)
            text_color = GuiTheme.BTN_TEXT_HOVER
        else:
            text_color = GuiTheme.WHITE

        draw_text(
            canvas,
            label,
            (pop_x1 + 10, iy1 + (item_h - 16) // 2 - 2),
            font_size=font_size,
            color=text_color,
            bold=is_active or is_hover,
        )
        btn_id = f"{btn_prefix}{i}"
        registered_buttons.append((btn_id, item_rect, key))

    return registered_buttons


def draw_dashboard_button(
    canvas: np.ndarray,
    rect: Tuple[int, int, int, int],
    label: str,
    mouse_pos: Tuple[int, int] = (-1, -1),
    accent: Optional[Tuple[int, int, int]] = None,
    is_running: bool = False,
    font_size: int = 12,
    disabled: bool = False,
) -> bool:
    """绘制与 Dashboard 同源的工业风碳灰卡片式按钮:
    深色底 + 沉稳边框, 悬停冷青微光, 左缘语义色条, 运行中金色高亮, 禁用态置灰

    Returns:
        bool: 当前鼠标是否悬停在按钮上
    """
    x1, y1, x2, y2 = rect
    mx, my = mouse_pos
    is_hover = (x1 <= mx <= x2 and y1 <= my <= y2) and not disabled

    if disabled:
        bg_col = GuiTheme.BTN_DISABLED_BG
        border_col = GuiTheme.BTN_DISABLED_BORDER
        text_col = GuiTheme.TEXT_DISABLED
        border_th = 1
    elif is_running:
        bg_col = GuiTheme.CARD_HOVER
        border_col = GuiTheme.GOLD
        text_col = (250, 225, 140)
        border_th = 2
    elif is_hover:
        bg_col = GuiTheme.CARD_HOVER
        border_col = GuiTheme.BORDER_HOVER
        text_col = GuiTheme.WHITE
        border_th = 2
    else:
        bg_col = GuiTheme.CARD_BG
        border_col = GuiTheme.BORDER
        text_col = GuiTheme.BTN_TEXT
        border_th = 1

    cv2.rectangle(canvas, (x1, y1), (x2, y2), bg_col, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, border_th)

    if accent is not None and not disabled:
        cv2.rectangle(canvas, (x1 + 1, y1 + 1), (x1 + 4, y2 - 1), accent, -1)

    bbox = get_cached_font(font_size, bold=True).getbbox(label)
    tw, t_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = x1 + max(4, ((x2 - x1) - tw) // 2 - bbox[0])
    ty = y1 + ((y2 - y1) - t_h) // 2 - bbox[1]
    draw_text(canvas, label, (tx, ty), font_size=font_size, color=text_col, bold=True)

    return is_hover


_CACHED_LOGO: Optional[np.ndarray] = None
_CACHED_LOGO_SIZE: int = -1


def get_cached_logo(size: int = 32) -> Optional[np.ndarray]:
    """读取并缓存标准 LOGO 图像 (assets/logo.png)"""
    global _CACHED_LOGO, _CACHED_LOGO_SIZE
    if _CACHED_LOGO is not None and _CACHED_LOGO_SIZE == size:
        return _CACHED_LOGO
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    logo_path = os.path.join(base_dir, "assets", "logo.png")
    if os.path.exists(logo_path):
        img = cv2.imread(logo_path, cv2.IMREAD_COLOR)
        if img is not None:
            _CACHED_LOGO = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            _CACHED_LOGO_SIZE = size
            return _CACHED_LOGO
    return None


def draw_app_header(
    canvas: np.ndarray,
    x: int = 12,
    y: int = 6,
    sub_title: str = "",
    icon_size: int = 32,
) -> int:
    """在 GUI 顶部统一绘制科技感品牌 LOGO、主标题与子系统模块名称

    Args:
        canvas: 目标画布 (BGR)
        x: 左侧起始 X
        y: 顶部起始 Y
        sub_title: 子模块名 (例如 "AprilTag 管理器", "标定场景中心", "空间位姿追踪器")
        icon_size: 图标尺寸 (默认 32x32)

    Returns:
        int: 标题组件右侧边缘的 X 坐标 (方便后续横向排布工具栏按钮)
    """
    logo = get_cached_logo(icon_size)
    curr_x = x
    if logo is not None:
        h, w = logo.shape[:2]
        ch, cw = canvas.shape[:2]
        if y + h <= ch and curr_x + w <= cw:
            canvas[y:y + h, curr_x:curr_x + w] = logo
            cv2.rectangle(canvas, (curr_x, y), (curr_x + w, y + h), (0, 216, 180), 1)
        curr_x += w + 10
    else:
        cv2.circle(canvas, (curr_x + icon_size // 2, y + icon_size // 2), icon_size // 2 - 2, (0, 216, 180), 2)
        cv2.circle(canvas, (curr_x + icon_size // 2, y + icon_size // 2), 3, (0, 255, 255), -1)
        curr_x += icon_size + 10

    # 绘制主标题 "FluxVision 3D"
    draw_text(canvas, "FluxVision 3D", (curr_x, y - 2), font_size=15, color=(0, 240, 220), bold=True)

    # 绘制副标题 (芦笋上料自动化 | <sub_title>)
    sub_text = f"芦笋上料自动化 | {sub_title}" if sub_title else "芦笋上料自动化"
    draw_text(canvas, sub_text, (curr_x, y + 17), font_size=11, color=(140, 160, 180))

    try:
        font1 = get_cached_font(15, bold=True)
        w1 = font1.getbbox("FluxVision 3D")[2]
        font2 = get_cached_font(11, bold=False)
        w2 = font2.getbbox(sub_text)[2]
        text_w = max(w1, w2)
    except Exception:
        text_w = 160

    return curr_x + text_w + 18


def draw_rounded_rectangle(
    canvas: np.ndarray,
    rect: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    radius: int = 8,
    thickness: int = 1,
    fill: bool = False,
) -> None:
    """在画布上绘制抗锯齿圆角矩形 (支持实体填充或平滑发光描边)

    Args:
        canvas: 目标画布图像 (BGR)
        rect: (x, y, w, h) 矩形范围
        color: 绘制颜色 (B, G, R)
        radius: 圆角半径 (像素，默认 8)
        thickness: 描边线宽 (fill=True 时忽略)
        fill: 是否填充内部
    """
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return
    r = max(0, min(radius, w // 2, h // 2))

    if r <= 0:
        if fill:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)
        else:
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
        return

    if fill:
        # 十字中心填充
        cv2.rectangle(canvas, (x + r, y), (x + w - r, y + h), color, -1)
        cv2.rectangle(canvas, (x, y + r), (x + w, y + h - r), color, -1)
        # 四角实心抗锯齿圆
        cv2.circle(canvas, (x + r, y + r), r, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x + w - r, y + r), r, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x + r, y + h - r), r, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x + w - r, y + h - r), r, color, -1, lineType=cv2.LINE_AA)
    else:
        # 四条直边
        cv2.line(canvas, (x + r, y), (x + w - r, y), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(canvas, (x + r, y + h), (x + w - r, y + h), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(canvas, (x, y + r), (x, y + h - r), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(canvas, (x + w, y + r), (x + w, y + h - r), color, thickness, lineType=cv2.LINE_AA)
        # 四角抗锯齿圆弧
        cv2.ellipse(canvas, (x + r, y + r), (r, r), 180, 0, 90, color, thickness, lineType=cv2.LINE_AA)
        cv2.ellipse(canvas, (x + w - r, y + r), (r, r), 270, 0, 90, color, thickness, lineType=cv2.LINE_AA)
        cv2.ellipse(canvas, (x + w - r, y + h - r), (r, r), 0, 0, 90, color, thickness, lineType=cv2.LINE_AA)
        cv2.ellipse(canvas, (x + r, y + h - r), (r, r), 90, 0, 90, color, thickness, lineType=cv2.LINE_AA)



def render_floating_tooltip(
    canvas: np.ndarray,
    title: str,
    lines: Sequence[Any],
    anchor_pos: Tuple[int, int],
    max_width: int = 560,
    theme_color: Optional[Tuple[int, int, int]] = None,
    font_size: int = 11,
    line_height: int = 20,
    corner_radius: int = 8,
) -> Tuple[int, int, int, int]:
    """绘制高对比度、科技质感的悬浮气泡框 (自动贴靠锚点并计算视口边界避让、翻转与圆角质感)

    Args:
        canvas: 目标画布 (BGR)
        title: 气泡标题文本
        lines: 提示正文列表 (支持纯文本 str，或结构化元组 (tag, text) / (tag, color, text))
        anchor_pos: (ax, ay) 锚点坐标 (通常为当前鼠标或控件基准点)
        max_width: 气泡框最大宽度 (像素，默认 560)
        theme_color: 强调色/发光边框色 (BGR)，默认使用 (0, 240, 200) 科技青
        font_size: 正文字体大小 (默认 11)
        line_height: 行高步进 (默认 20)
        corner_radius: 气泡圆角半径 (默认 8)

    Returns:
        Tuple[int, int, int, int]: 实际渲染的气泡矩形 (tx, ty, tw, th)
    """
    ch, cw = canvas.shape[:2]
    ax, ay = anchor_pos

    # 动态测量最宽行以自适应气泡宽度，避免空白或文字溢出
    measured_max_w = 0
    for item in lines:
        if isinstance(item, (tuple, list)):
            if len(item) == 3:
                tag, _, text = item
                text_for_w = f"[{tag}]  {text}"
            elif len(item) >= 2:
                tag, text = item[0], item[1]
                text_for_w = f"[{tag}]  {text}"
            else:
                text_for_w = str(item[0]) if item else ""
        else:
            text_for_w = str(item)
        if text_for_w:
            try:
                (tw_line, _), _ = measure_text(text_for_w, font_size=font_size)
                if tw_line > measured_max_w:
                    measured_max_w = tw_line
            except Exception:
                measured_max_w = max(measured_max_w, len(text_for_w) * 11)

    try:
        (title_w, _), _ = measure_text(f"★ {title}", font_size=13, bold=True)
        measured_max_w = max(measured_max_w, title_w)
    except Exception:
        measured_max_w = max(measured_max_w, len(title) * 13)

    content_w = measured_max_w + 34
    tw = max(280, min(max_width, content_w))
    tw = min(tw, cw - 32)
    th = 38 + len(lines) * line_height + 12

    # 横向边界避让
    tx = max(16, min(cw - tw - 16, ax + 15))

    # 纵向计算与底部溢出自动向上翻转
    ty = ay + 15
    if ty + th > ch - 16:
        ty = max(16, ay - th - 10)
    ty = max(16, min(ch - th - 16, ty))

    accent_col = theme_color if theme_color is not None else (0, 240, 200)

    # 区域半透明暗色磨砂背景融合 (带有平滑圆角 Mask 遮罩)
    sub = canvas[ty:ty + th, tx:tx + tw]
    bg = np.full_like(sub, (16, 20, 28))
    blended = cv2.addWeighted(bg, 0.94, sub, 0.06, 0)
    card_mask = np.zeros((th, tw), dtype=np.uint8)
    draw_rounded_rectangle(card_mask, (0, 0, tw, th), 255, radius=corner_radius, fill=True)
    np.copyto(sub, blended, where=(card_mask[:, :, None] > 0))
    canvas[ty:ty + th, tx:tx + tw] = sub

    # 标题栏底色 (带顶部两处圆角)
    title_h = 30
    title_sub = canvas[ty:ty + title_h, tx:tx + tw]
    title_bg = np.full_like(title_sub, (24, 34, 44))
    t_mask = np.zeros((title_h, tw), dtype=np.uint8)
    r = min(corner_radius, title_h, tw // 2)
    cv2.rectangle(t_mask, (0, r), (tw, title_h), 255, -1)
    cv2.rectangle(t_mask, (r, 0), (tw - r, r), 255, -1)
    cv2.circle(t_mask, (r, r), r, 255, -1, lineType=cv2.LINE_AA)
    cv2.circle(t_mask, (tw - r, r), r, 255, -1, lineType=cv2.LINE_AA)
    np.copyto(title_sub, title_bg, where=(t_mask[:, :, None] > 0))
    canvas[ty:ty + title_h, tx:tx + tw] = title_sub

    # 标题栏底部分割线
    cv2.line(canvas, (tx + 1, ty + title_h), (tx + tw - 1, ty + title_h), (0, 200, 160), 1)

    # 外层发光圆角细边框与内衬双边
    draw_rounded_rectangle(canvas, (tx, ty, tw, th), accent_col, radius=corner_radius, thickness=1)
    inner_r = max(2, corner_radius - 1)
    draw_rounded_rectangle(canvas, (tx + 1, ty + 1, tw - 2, th - 2), (30, 60, 55), radius=inner_r, thickness=1)

    # 标题文字
    draw_text(canvas, f"★ {title}", (tx + 12, ty + 6), font_size=13, color=(0, 255, 220), bold=True)

    # 正文内容逐行语义着色
    cur_y = ty + 38
    for item in lines:
        if isinstance(item, (tuple, list)):
            if len(item) == 3:
                tag, val_col, text = item
            elif len(item) >= 2:
                tag, text = item[0], item[1]
                val_col = (220, 235, 245)
            else:
                tag, val_col, text = "", (220, 235, 245), str(item[0]) if item else ""

            tag_label = f"[{tag}] " if tag else ""
            if tag_label:
                (lw, _), _ = measure_text(tag_label, font_size=font_size, bold=True)
                draw_text(canvas, tag_label, (tx + 14, cur_y), font_size=font_size, color=(0, 240, 210), bold=True)
                draw_text(canvas, str(text), (tx + 14 + lw + 4, cur_y), font_size=font_size, color=val_col, bold=False)
            else:
                draw_text(canvas, str(text), (tx + 14, cur_y), font_size=font_size, color=val_col, bold=False)
            cur_y += line_height
            continue

        line = str(item)
        if not line:
            cur_y += 6
            continue
        col = (220, 235, 245)
        bold = False
        if line.startswith("【") or line.startswith("["):
            col = (0, 240, 210)
            bold = True
        elif line.startswith("•") or line.startswith("-"):
            col = (180, 210, 230)
        elif "★" in line or "注意" in line:
            col = (255, 205, 80)
        elif "!" in line or "警告" in line or "错误" in line:
            col = (80, 90, 255)
        draw_text(canvas, line, (tx + 14, cur_y), font_size=font_size, color=col, bold=bold)
        cur_y += line_height

    return (tx, ty, tw, th)


@dataclass
class TabItem:
    """标准 Tab 页签项定义"""
    key: str
    label: str
    badge: Optional[str] = None
    badge_color: Optional[Tuple[int, int, int]] = None
    enabled: bool = True
    custom_width: Optional[int] = None


class TabBar:
    """跨应用通用的标准视觉 Tab 页签交互组件 (TabBar)
    ==================================================
    统一管理多窗口的 Tab 状态流转、单源排版几何计算、视觉渲染与 Hit-test 碰撞检测。
    """
    STYLE_CAPSULE = "capsule"   # 现代工业风发光胶囊卡片 (Workspace Hub 风格)
    STYLE_PILL    = "pill"      # 圆角紧凑药丸卡片 (Tag Manager 风格)
    STYLE_LINE    = "line"      # 极简下划线风格

    def __init__(
        self,
        tabs: Sequence[Union[TabItem, Tuple[str, str], Tuple[str, str, str]]] = (),
        active_key: Optional[str] = None,
        style: str = STYLE_CAPSULE,
        tab_height: int = 34,
        spacing: int = 10,
        fixed_width: Optional[int] = None,
        font_size: int = 13,
        on_change: Optional[Callable[[str], None]] = None,
    ):
        self.style = style
        self.tab_height = tab_height
        self.spacing = spacing
        self.fixed_width = fixed_width
        self.font_size = font_size
        self.on_change = on_change

        self.items: List[TabItem] = []
        self._active_key: Optional[str] = None
        self._last_layout: List[Tuple[str, Tuple[int, int, int, int]]] = []
        self._last_container_rect: Optional[Tuple[int, int, int, int]] = None

        self.set_tabs(tabs, active_key=active_key)

    @property
    def active_key(self) -> Optional[str]:
        return self._active_key

    @active_key.setter
    def active_key(self, key: Optional[str]):
        self.select(key)

    def set_tabs(
        self,
        tabs: Sequence[Union[TabItem, Tuple[str, str], Tuple[str, str, str]]],
        active_key: Optional[str] = None,
    ):
        """动态配置或更新页签项集合"""
        new_items: List[TabItem] = []
        for t in tabs:
            if isinstance(t, TabItem):
                new_items.append(t)
            elif isinstance(t, (tuple, list)):
                if len(t) == 2:
                    new_items.append(TabItem(key=str(t[0]), label=str(t[1])))
                elif len(t) >= 3:
                    new_items.append(TabItem(key=str(t[0]), label=str(t[1]), badge=str(t[2])))
                else:
                    new_items.append(TabItem(key=str(t[0]), label=str(t[0])))

        self.items = new_items

        # 校验或初始化 active_key
        if active_key is not None:
            self._active_key = active_key
        elif self.items and (self._active_key is None or self._active_key not in [it.key for it in self.items]):
            self._active_key = self.items[0].key

    def select(self, key: Optional[str]) -> bool:
        """切换选中 Tab，若发生改变则触发 on_change 并返回 True"""
        if key == self._active_key:
            return False
        # 验证 key 是否有效且启用
        target_item = next((it for it in self.items if it.key == key), None)
        if target_item and not target_item.enabled:
            return False

        self._active_key = key
        if self.on_change and key is not None:
            try:
                self.on_change(key)
            except Exception:
                pass
        return True

    def compute_layout(
        self,
        container_rect: Tuple[int, int, int, int],
    ) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """单源真理几何排版计算：根据容器矩形计算每个 Tab 的绝对像素边界 (x, y, w, h)"""
        self._last_container_rect = container_rect
        cx, cy, cw, ch = container_rect
        layout: List[Tuple[str, Tuple[int, int, int, int]]] = []
        cur_x = cx
        th = min(ch, self.tab_height)
        ty = cy + (ch - th) // 2

        for item in self.items:
            # 计算 Tab 宽度
            if self.fixed_width is not None:
                tw = self.fixed_width
            elif item.custom_width is not None:
                tw = item.custom_width
            else:
                # 动态测量文字宽度
                approx_w = sum(13 if ord(c) > 127 else 8 for c in item.label)
                if item.badge:
                    approx_w += sum(12 if ord(c) > 127 else 7 for c in item.badge) + 14
                tw = max(80, approx_w + 26)

            rect = (cur_x, ty, tw, th)
            layout.append((item.key, rect))
            cur_x += tw + self.spacing

        self._last_layout = layout
        return layout

    def render(
        self,
        canvas: np.ndarray,
        container_rect: Tuple[int, int, int, int],
        mouse_pos: Tuple[int, int] = (-1, -1),
    ) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """渲染绘制 TabBar 交互组件"""
        layout = self.compute_layout(container_rect)
        mx, my = mouse_pos

        for item in self.items:
            # 查找对应几何位置
            rect_entry = next((r for k, r in layout if k == item.key), None)
            if not rect_entry:
                continue
            tx, ty, tw, th = rect_entry
            is_active = (item.key == self._active_key)
            is_hover = (item.enabled and tx <= mx <= tx + tw and ty <= my <= ty + th)

            if not item.enabled:
                bg_col = (18, 20, 24)
                border_col = (35, 40, 48)
                text_col = (90, 100, 115)
            elif is_active:
                bg_col = (28, 44, 40) if self.style == self.STYLE_CAPSULE else GuiTheme.CARD_SEL
                border_col = (0, 255, 180) if self.style == self.STYLE_CAPSULE else GuiTheme.ACCENT
                text_col = (0, 255, 200) if self.style == self.STYLE_CAPSULE else GuiTheme.WHITE
            elif is_hover:
                bg_col = (34, 40, 52)
                border_col = (0, 200, 240) if self.style == self.STYLE_CAPSULE else GuiTheme.BORDER_SEL
                text_col = (0, 220, 255) if self.style == self.STYLE_CAPSULE else GuiTheme.BTN_TEXT_HOVER
            else:
                bg_col = (22, 27, 35)
                border_col = (45, 55, 72) if self.style == self.STYLE_CAPSULE else GuiTheme.BORDER
                text_col = (160, 175, 195) if self.style == self.STYLE_CAPSULE else GuiTheme.BTN_TEXT

            # 绘制背景与边框
            if self.style == self.STYLE_CAPSULE:
                cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), bg_col, -1)
                border_th = 2 if is_active else 1
                cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), border_col, border_th)
                if is_active:
                    # 激活态底部发光指示条
                    cv2.rectangle(canvas, (tx + 8, ty + th - 3), (tx + tw - 8, ty + th - 1), (0, 255, 180), -1)
            elif self.style == self.STYLE_PILL:
                draw_rounded_rectangle(canvas, (tx, ty, tw, th), border_col, radius=4, thickness=2 if is_active else 1, fill=True)
                cv2.rectangle(canvas, (tx + 1, ty + 1), (tx + tw - 1, ty + th - 1), bg_col, -1)
            elif self.style == self.STYLE_LINE:
                if is_active:
                    cv2.rectangle(canvas, (tx + 4, ty + th - 3), (tx + tw - 4, ty + th), border_col, -1)
                elif is_hover:
                    cv2.rectangle(canvas, (tx + 4, ty + th - 2), (tx + tw - 4, ty + th), (80, 120, 160), -1)

            # 绘制文字与角标
            approx_w = sum(13 if ord(c) > 127 else 8 for c in item.label)
            if item.badge:
                approx_w += sum(12 if ord(c) > 127 else 7 for c in item.badge) + 14

            text_x = tx + max(6, (tw - approx_w) // 2)
            text_y = ty + (th - 16) // 2
            draw_text(canvas, item.label, (text_x, text_y), font_size=self.font_size, color=text_col, bold=is_active)

            # 绘制角标 Badge (若有)
            if item.badge:
                bw = sum(12 if ord(c) > 127 else 7 for c in item.badge)
                bx = text_x + approx_w - bw
                b_color = item.badge_color or ((0, 230, 255) if is_active else (180, 200, 220))
                draw_text(canvas, item.badge, (bx, text_y), font_size=max(10, self.font_size - 2), color=b_color, bold=True)

        return layout

    def hit_test(self, x: int, y: int) -> Optional[str]:
        """单源热区碰撞检测：根据最近一次布局，返回命中的 tab key，未命中返回 None"""
        for key, (tx, ty, tw, th) in self._last_layout:
            if tx <= x <= tx + tw and ty <= y <= ty + th:
                item = next((it for it in self.items if it.key == key), None)
                if item and item.enabled:
                    return key
        return None

    def handle_click(self, x: int, y: int) -> Optional[str]:
        """点击快捷处理：检测命中、自动切换内部 active_key，并返回选中的 key"""
        hit_key = self.hit_test(x, y)
        if hit_key:
            self.select(hit_key)
            return hit_key
        return None


class ScrollableListBox:
    """
    通用工业级可滚动列表组件 (ScrollableListBox)
    =============================================
    采用委托渲染模式 (Delegate Pattern):
    - 容器层 (ScrollableListBox):
      1. 虚拟化视口计算 (可见行数、总项数自适应、安全几何边界)
      2. 平滑滚动偏移控制 (scroll_offset、滚轮自适应步进、范围夹紧)
      3. 工业级自适应滚动条 (轨道 Track、自适应高度滑块 Thumb、滑块拖拽与轨道跳跃)
      4. 鼠标热区与状态机 (悬停项 hover_index、选中项 selected_index、点击选择分发)
      5. 统一标准化的条目高亮底衬与外边框绘制
    - 委托绘制层 (draw_item_callback):
      接收 (canvas, rect, item_data, index, is_hover, is_selected)
      由业务方自由渲染高个性化内容 (如状态灯、多列彩色数字、进度条、复杂文字排版等)
    """

    def __init__(
        self,
        item_height: int = 36,
        item_gap: int = 2,
        scrollbar_width: int = 6,
        auto_hide_scrollbar: bool = True,
        render_item_background: bool = True,
        scroll_speed: int = 2,
    ):
        self.item_height = item_height
        self.item_gap = item_gap
        self.scrollbar_width = scrollbar_width
        self.auto_hide_scrollbar = auto_hide_scrollbar
        self.render_item_background = render_item_background
        self.scroll_speed = scroll_speed

        # 运行时状态
        self.scroll_offset: int = 0
        self.selected_index: int = -1
        self.hover_index: int = -1
        self.is_dragging_thumb: bool = False
        self._drag_start_y: int = 0
        self._drag_start_offset: int = 0

        # 最近一次渲染几何缓存
        self._last_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._last_track_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._last_thumb_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._last_visible_count: int = 0
        self._last_total_items: int = 0
        self._item_rects: List[Tuple[int, Tuple[int, int, int, int]]] = []

    def get_visible_count(self, viewport_h: int) -> int:
        """根据当前视口高度计算最多可容纳的完整条目数"""
        stride = self.item_height + self.item_gap
        if stride <= 0:
            return 1
        return max(1, viewport_h // stride)

    def clamp_scroll_offset(self, total_items: int, visible_count: int):
        """确保滚动偏移量位于 [0, max_offset] 有效区间内"""
        max_offset = max(0, total_items - visible_count)
        self.scroll_offset = max(0, min(self.scroll_offset, max_offset))

    def scroll_to_index(self, index: int, total_items: int, visible_count: Optional[int] = None):
        """确保指定 index 条目滚动至可见区域内"""
        if total_items <= 0:
            self.scroll_offset = 0
            return
        vis_c = visible_count or self._last_visible_count or 1
        index = max(0, min(index, total_items - 1))
        if index < self.scroll_offset:
            self.scroll_offset = index
        elif index >= self.scroll_offset + vis_c:
            self.scroll_offset = index - vis_c + 1
        self.clamp_scroll_offset(total_items, vis_c)

    def handle_scroll(self, delta_lines: int, total_items: int) -> bool:
        """鼠标滚轮处理 (delta_lines: 正数向下滚，负数向上滚)"""
        vis_c = self._last_visible_count or 1
        old_offset = self.scroll_offset
        self.scroll_offset += delta_lines * self.scroll_speed
        self.clamp_scroll_offset(total_items, vis_c)
        return self.scroll_offset != old_offset

    def handle_mouse_down(self, mx: int, my: int, total_items: int) -> Tuple[bool, Optional[int]]:
        """
        鼠标按下事件分发:
        Returns:
            (handled: bool, clicked_index: Optional[int])
        """
        # 1. 检查是否点击滚动条滑块 (开始拖动)
        if self._last_thumb_rect[2] > 0:
            tx, ty, tw, th = self._last_thumb_rect
            if tx <= mx <= tx + tw and ty <= my <= ty + th:
                self.is_dragging_thumb = True
                self._drag_start_y = my
                self._drag_start_offset = self.scroll_offset
                return True, None

            # 2. 检查是否点击滚动条轨道 (轨道快速跳转)
            rx, ry, rw, rh = self._last_track_rect
            if rx <= mx <= rx + rw and ry <= my <= ry + rh:
                vis_c = self._last_visible_count or 1
                max_offset = max(0, total_items - vis_c)
                if max_offset > 0:
                    ratio = max(0.0, min(1.0, (my - ry) / float(rh)))
                    self.scroll_offset = int(round(ratio * max_offset))
                    self.clamp_scroll_offset(total_items, vis_c)
                return True, None

        # 3. 检查是否点击具体数据条目
        for idx, (ix, iy, iw, ih) in self._item_rects:
            if ix <= mx <= ix + iw and iy <= my <= iy + ih:
                self.selected_index = idx
                return True, idx

        return False, None

    def handle_mouse_move(self, mx: int, my: int, total_items: int) -> bool:
        """鼠标移动事件: 支持滑块实时拖动与 Hover 悬停状态更新"""
        # 1. 正在拖拽滑块中
        if self.is_dragging_thumb:
            rx, ry, rw, rh = self._last_track_rect
            vis_c = self._last_visible_count or 1
            max_offset = max(0, total_items - vis_c)
            if rh > 0 and max_offset > 0:
                dy = my - self._drag_start_y
                step_per_px = max_offset / float(max(1, rh - self._last_thumb_rect[3]))
                new_offset = int(round(self._drag_start_offset + dy * step_per_px))
                self.scroll_offset = max(0, min(new_offset, max_offset))
            return True

        # 2. 悬停检测
        old_hover = self.hover_index
        self.hover_index = -1
        for idx, (ix, iy, iw, ih) in self._item_rects:
            if ix <= mx <= ix + iw and iy <= my <= iy + ih:
                self.hover_index = idx
                break
        return self.hover_index != old_hover

    def handle_mouse_up(self) -> bool:
        """释放滑块拖拽"""
        if self.is_dragging_thumb:
            self.is_dragging_thumb = False
            return True
        return False

    def render(
        self,
        canvas: np.ndarray,
        rect: Tuple[int, int, int, int],
        items: List[Any],
        draw_item_callback: Optional[Callable[[np.ndarray, Tuple[int, int, int, int], Any, int, bool, bool], None]] = None,
        mouse_pos: Tuple[int, int] = (-1, -1),
        empty_text: str = "暂无数据项",
    ):
        """
        执行列表与滚动条统一渲染:
        Args:
            canvas: 绘制目标画布 (BGR)
            rect: 视口区域 (vx, vy, vw, vh)
            items: 数据项列表
            draw_item_callback: 委托条目绘制函数 (canvas, item_rect, item_data, index, is_hover, is_selected)
            mouse_pos: 当前鼠标坐标 (mx, my)
            empty_text: 列表为空时的居中文本
        """
        vx, vy, vw, vh = rect
        self._last_rect = rect
        total_items = len(items)
        self._last_total_items = total_items

        visible_count = self.get_visible_count(vh)
        self._last_visible_count = visible_count
        self.clamp_scroll_offset(total_items, visible_count)

        has_scrollbar = (total_items > visible_count)
        sb_w = self.scrollbar_width if (has_scrollbar or not self.auto_hide_scrollbar) else 0
        content_w = vw - (sb_w + 4 if sb_w > 0 else 0)

        # 实时检测当前鼠标位置的 hover_index (如果未被拖动锁定)
        mx, my = mouse_pos
        if not self.is_dragging_thumb and (vx <= mx <= vx + vw and vy <= my <= vy + vh):
            self.handle_mouse_move(mx, my, total_items)
        elif not self.is_dragging_thumb:
            self.hover_index = -1

        # 1. 空状态兜底
        if total_items == 0:
            self._item_rects.clear()
            self._last_track_rect = (0, 0, 0, 0)
            self._last_thumb_rect = (0, 0, 0, 0)
            draw_text(canvas, empty_text, (vx + vw // 2 - 40, vy + vh // 2 - 8), font_size=12, color=GuiTheme.TEXT_MUTED)
            return

        # 2. 逐行绘制可视条目
        self._item_rects.clear()
        stride = self.item_height + self.item_gap

        for row_i in range(visible_count):
            item_idx = self.scroll_offset + row_i
            if item_idx >= total_items:
                break

            iy = vy + row_i * stride
            item_rect = (vx, iy, content_w, self.item_height)
            self._item_rects.append((item_idx, item_rect))

            is_selected = (item_idx == self.selected_index)
            is_hover = (item_idx == self.hover_index)

            # 绘制统一的底层背景与高亮框 (业务方亦可在其 draw_item_callback 中覆盖)
            if self.render_item_background:
                if is_selected:
                    bg_col = (48, 42, 28)
                    border_col = (0, 220, 255)
                    border_th = 2
                elif is_hover:
                    bg_col = (34, 38, 48)
                    border_col = (65, 75, 95)
                    border_th = 1
                else:
                    bg_col = (25, 27, 34) if item_idx % 2 == 0 else (20, 22, 28)
                    border_col = (38, 40, 50)
                    border_th = 1

                cv2.rectangle(canvas, (vx, iy), (vx + content_w, iy + self.item_height), bg_col, -1)
                cv2.rectangle(canvas, (vx, iy), (vx + content_w, iy + self.item_height), border_col, border_th)

            # 调用业务委托绘制函数
            if draw_item_callback is not None:
                draw_item_callback(canvas, item_rect, items[item_idx], item_idx, is_hover, is_selected)
            else:
                # 默认纯文本回退绘制
                txt = str(items[item_idx])
                text_col = (255, 255, 255) if is_selected else (200, 210, 225)
                draw_text(canvas, txt, (vx + 10, iy + (self.item_height - 14) // 2), font_size=12, color=text_col, bold=is_selected)

        # 3. 绘制微质感滚动条 (仅当需要滚动或显式配置时)
        if sb_w > 0 and has_scrollbar:
            track_x = vx + vw - sb_w - 2
            track_y = vy + 2
            track_h = vh - 4
            self._last_track_rect = (track_x, track_y, sb_w, track_h)

            # 轨道底衬
            cv2.rectangle(canvas, (track_x, track_y), (track_x + sb_w, track_y + track_h), (20, 22, 28), -1)
            cv2.rectangle(canvas, (track_x, track_y), (track_x + sb_w, track_y + track_h), (35, 40, 50), 1)

            # 自适应滑块高度 (最小安全高度 20px)
            thumb_ratio = max(0.08, min(1.0, float(visible_count) / float(total_items)))
            thumb_h = max(20, int(round(track_h * thumb_ratio)))

            # 滑块 Y 轴位置计算
            max_offset = max(1, total_items - visible_count)
            scroll_ratio = max(0.0, min(1.0, float(self.scroll_offset) / float(max_offset)))
            thumb_y = track_y + int(round(scroll_ratio * (track_h - thumb_h)))
            self._last_thumb_rect = (track_x, thumb_y, sb_w, thumb_h)

            # 滑块颜色：拖拽中金色发光，鼠标悬停时高亮科技蓝，常态为典雅暗银色
            is_thumb_hover = (track_x <= mx <= track_x + sb_w and thumb_y <= my <= thumb_y + thumb_h)
            if self.is_dragging_thumb:
                thumb_col = (0, 240, 255)
            elif is_thumb_hover:
                thumb_col = (0, 200, 230)
            else:
                thumb_col = (75, 85, 105)

            cv2.rectangle(canvas, (track_x, thumb_y), (track_x + sb_w, thumb_y + thumb_h), thumb_col, -1)
        else:
            self._last_track_rect = (0, 0, 0, 0)
            self._last_thumb_rect = (0, 0, 0, 0)





