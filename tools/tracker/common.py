#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robot 在线跟踪 - 共享视觉样式常量与绘制工具 (主控制器与渲染器共用)"""

import numpy as np

from src.ui.gui_theme import GuiTheme

# ============================ 视觉样式常量 (BGR, 统一取自 GuiTheme 主题单源) ============================
COLOR_BG = GuiTheme.BG           # 工具栏 / 占位背景
COLOR_CARD_BG = GuiTheme.CARD_BG  # 按钮常态底色
COLOR_CARD_SEL = GuiTheme.CARD_SEL  # 乒乓开关激活底色
COLOR_BORDER = GuiTheme.BORDER   # 常态描边
COLOR_BORDER_HOVER = GuiTheme.BORDER_HOVER  # 悬停描边 (主题统一)
COLOR_BORDER_SEL = GuiTheme.BORDER_SEL  # 激活描边
COLOR_BTN_HOVER = GuiTheme.BTN_HOVER  # 悬停底色 (主题统一)
COLOR_BTN_TEXT_HOVER = GuiTheme.BTN_TEXT_HOVER  # 按钮悬停文字 (主题统一)
COLOR_TEXT_DISABLED = GuiTheme.TEXT_DISABLED  # 按钮禁用文字 (主题统一)
COLOR_ACCENT = GuiTheme.ACCENT   # 主题强调色
COLOR_TEXT_SUB = GuiTheme.TEXT_SUB
COL_GREEN = (110, 220, 90)      # 实测 / 目标标靶高亮 (数据可视化色, 本地保留)
COL_YELLOW = (90, 200, 245)     # 理论值 (数据可视化色, 本地保留)
COL_CYAN = (235, 205, 80)       # 偏差 / 支撑标靶 (数据可视化色, 本地保留)
COL_BLUE = (255, 170, 0)        # 单帧实测棱柱 (蓝, 与已知Tag叠加的实测色一致)
COL_RED = (85, 85, 245)         # 错误提示
COL_WHITE = GuiTheme.WHITE
COL_GRAY = GuiTheme.GRAY
COL_PANEL_BG = (26, 26, 30)     # 信息面板底色 (数据可视化色, 本地保留)
COL_PANEL_EDGE = (95, 95, 105)  # 信息面板描边 (数据可视化色, 本地保留)

TOOLBAR_H = 78  # 顶部工具栏高度 (双排: 第一排相机/串口连接, 第二排世界系与跟踪)


def list_serial_ports():
    """枚举系统可用串口列表 (COMx), pyserial 缺失时返回空列表"""
    try:
        import serial.tools.list_ports
        return [p.device for p in serial.tools.list_ports.comports()]
    except Exception:
        return []


def fmt_point(p, signed=False):
    """三维/四维坐标一行式格式化 (一位小数, 右对齐, 无 X/Y/Z 前缀, 支持可选 R 轴)"""
    if p is None:
        return "    --      --      --"
    if len(p) >= 4:
        r_str = f"  R:{p[3]:+6.1f}°" if signed else f"  R:{p[3]:6.1f}°"
        base = f"{p[0]:+8.1f} {p[1]:+8.1f} {p[2]:+8.1f}" if signed else f"{p[0]:8.1f} {p[1]:8.1f} {p[2]:8.1f}"
        return base + r_str
    if signed:
        return f"{p[0]:+8.1f} {p[1]:+8.1f} {p[2]:+8.1f}"
    return f"{p[0]:8.1f} {p[1]:8.1f} {p[2]:8.1f}"


def _tag_local_frame(wc):
    """由标靶世界角点构造局部坐标系 (X右/Y上/Z=面法向朝镜头, 右手系) 与中心点。
    角点顺序: 左上, 右上, 右下, 左下; Z = X×Y 朝外 (与单靶 PnP 位姿同向)。"""
    x = wc[1] - wc[0]
    x = x / (np.linalg.norm(x) + 1e-9)
    y0 = wc[0] - wc[3]
    y0 = y0 / (np.linalg.norm(y0) + 1e-9)
    z = np.cross(x, y0)
    z = z / (np.linalg.norm(z) + 1e-9)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-9)
    return np.column_stack([x, y, z]), wc.mean(axis=0)


PRISM_HW_MM = 15.0        # 棱柱截面半宽 mm (截面 30x30, 与 Offline Studio 一致)
PRISM_HEIGHT_MM = 75.0    # 棱柱生长高度 mm (与 Offline Studio 一致)

# ============================ 芦笋几何物理尺寸 (宽15mm x 长200mm, 以Tag中心对称延伸各100mm) ============================
ASPARAGUS_WIDTH_MM = 15.0         # 芦笋物理宽度 (mm)
ASPARAGUS_LENGTH_MM = 200.0       # 芦笋物理总长 (mm)
ASPARAGUS_HALF_LENGTH_MM = 100.0  # 芦笋半长 (以 Tag 为中心对称延伸各 100mm)
ASPARAGUS_HALF_WIDTH_MM = 7.5     # 芦笋半宽 (mm)


def fmt_pose_4d(p, r_deg=None):
    """四维位姿格式化 (X Y Z R)"""
    if p is None:
        return "    --      --      --      --"
    base = f"{p[0]:7.1f} {p[1]:7.1f} {p[2]:7.1f}"
    if r_deg is not None:
        return f"{base}  R:{r_deg:+6.1f}°"
    return f"{base}  R:   -- "

