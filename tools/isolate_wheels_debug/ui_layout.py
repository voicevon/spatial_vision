#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
isolate_wheels_debug 界面几何与布局常量 (ui_layout)
====================================================
定义 1280 × 760 统一画布下的所有控件坐标、颜色主题及动态 Rect 计算纯函数。
"""

from typing import Tuple

# ==================== 画布与颜色主题 (单源标准 1280 × 760) ====================
LOGIC_W, LOGIC_H = 1280, 760

COLOR_BG = (14, 16, 20)
COLOR_PANEL = (20, 24, 30)
COLOR_CARD_SUB = (20, 23, 30)
COLOR_BORDER = (38, 46, 58)
COLOR_BORDER_HL = (0, 190, 235)
COLOR_TEXT = (220, 230, 242)
COLOR_SUB = (130, 142, 160)
COLOR_MUTED = (75, 85, 100)
COLOR_ACCENT = (0, 190, 235)
COLOR_GREEN = (0, 210, 130)
COLOR_AMBER = (0, 175, 255)

STATE_COLORS = {
    "idle": (0, 210, 130),
    "running": (0, 175, 255),
    "offline": (80, 80, 255),
}
STATE_TEXTS = {
    "idle": "IDLE 空闲 (可受理命令)",
    "running": "RUNNING 节拍执行中 (等待完成)",
    "offline": "OFFLINE 掉线 (遗嘱)",
}

# 顶栏按钮
BTN_CONNECT = (1030, 14, 70, 30)
BTN_DISCONNECT = (1108, 14, 70, 30)
BTN_QUIT = (1186, 14, 66, 30)

# 在线设备芯片选择
DEV_CHIP_X0, DEV_CHIP_Y, DEV_CHIP_W, DEV_CHIP_H, DEV_CHIP_STEP = 540, 14, 88, 30, 96
DEV_CHIP_MAX = 4

# 8 通道机架卡片整体尺寸
RACK_PANEL = (24, 58, 1232, 342)
TAB_MULTI = (988, 68, 126, 26)
TAB_SINGLE = (1120, 68, 122, 26)

# 8 个立式通道列 (物理实物排布: 屏幕从左到右显示 1号轮 -> 8号轮)
COL_W = 140
COL_GAP = 12
COL_X0 = 38
COL_Y0 = 100
COL_H = 224

# 机架底部操作工具栏
BTN_CLEAR_COUNTS = (38, 344, 128, 38)
BTN_RESET_ANGLES = (174, 344, 128, 38)
BTN_SEND_LOAD = (310, 344, 210, 38)
BOX_LOAD_SPEED = (530, 344, 130, 38)

BTN_DIR_FWD = (854, 344, 68, 38)
BTN_DIR_REV = (928, 344, 68, 38)
BTN_SEND_MOTOR = (1004, 344, 238, 38)

# 下半区双栏
STATUS_CARD = (24, 406, 594, 316)
LOG_CARD = (634, 406, 622, 316)
BTN_CLEAR_LOG = (1182, 414, 64, 24)
LOG_HEADER_H = 34
LOG_LINE_H = 24

# 角度下拉快速预设列表
MULTI_ANGLE_OPTS = (0.0, 22.5, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0,
                    -22.5, -45.0, -90.0, -180.0, -360.0)
POPUP_ROW_H = 24

# 生产节拍速度倍率预设列表
LOAD_SPEED_OPTS = (0.1, 0.2, 0.5, 1.0, 1.5, 2.0)
SPEED_POPUP_ROW_H = 26


# ==================== 矩形几何计算纯函数 ====================
def device_chip_rect(idx: int) -> Tuple[int, int, int, int]:
    return (DEV_CHIP_X0 + idx * DEV_CHIP_STEP, DEV_CHIP_Y, DEV_CHIP_W, DEV_CHIP_H)


def col_rect(col: int) -> Tuple[int, int, int, int]:
    return (COL_X0 + col * (COL_W + COL_GAP), COL_Y0, COL_W, COL_H)


def col_header_rect(col: int) -> Tuple[int, int, int, int]:
    x, y, w, _ = col_rect(col)
    return (x, y, w, 30)


def col_count_minus(col: int) -> Tuple[int, int, int, int]:
    x, y, w, _ = col_rect(col)
    return (x + 10, y + 84, 56, 26)


def col_count_plus(col: int) -> Tuple[int, int, int, int]:
    x, y, w, _ = col_rect(col)
    return (x + w - 10 - 56, y + 84, 56, 26)


def col_angle_box(col: int) -> Tuple[int, int, int, int]:
    x, y, w, _ = col_rect(col)
    return (x + 10, y + 146, w - 20, 30)


def col_angle_minus(col: int) -> Tuple[int, int, int, int]:
    x, y, w, _ = col_rect(col)
    return (x + 10, y + 182, 56, 26)


def col_angle_plus(col: int) -> Tuple[int, int, int, int]:
    x, y, w, _ = col_rect(col)
    return (x + w - 10 - 56, y + 182, 56, 26)


def angle_popup_rect(popup_col: int) -> Tuple[int, int, int, int]:
    if popup_col < 0:
        return (0, 0, 0, 0)
    bx, by, bw, _ = col_angle_box(popup_col)
    h = len(MULTI_ANGLE_OPTS) * POPUP_ROW_H + 8
    top = max(10, by - h - 4)
    return (bx - 8, top, bw + 16, by - 4 - top)


def speed_popup_rect() -> Tuple[int, int, int, int]:
    sx, sy, sw, _ = BOX_LOAD_SPEED
    h = len(LOAD_SPEED_OPTS) * SPEED_POPUP_ROW_H + 8
    return (sx, sy - h - 4, sw, h)
