#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多视角采图向导 - 渲染器:
顶部单排工具栏:
[工位: XXX ▼] [用途: 标定/生产 ▼] [相机类型 ▼] [分辨率 ▼] [开启/关闭] ... [退出 X]
下拉浮层 / 画布合成 / Toast。
只读 CaptureWizard 的状态并绘制, 不修改业务状态;
按钮命中表 (buttons) 每帧由 draw_toolbar 重建, 供主控制器鼠标分发使用。
"""

import time
import cv2
import numpy as np

from src.ui.gui_theme import GuiTheme
from src.ui.gui_components import (
    draw_dropdown_button,
    render_dropdown_popup,
)
from src.utils.text_rendering import draw_text

# 视觉样式常量 (BGR, 统一取自 GuiTheme 主题单源)
COLOR_BG = GuiTheme.BG                    # 工具栏 / 占位背景
COLOR_CARD_BG = GuiTheme.CARD_BG          # 按钮常态底色
COLOR_CARD_SEL = GuiTheme.CARD_SEL        # 乒乓开关激活底色
COLOR_BORDER = GuiTheme.BORDER            # 常态描边
COLOR_BORDER_HOVER = GuiTheme.BORDER_HOVER  # 悬停描边
COLOR_BORDER_SEL = GuiTheme.BORDER_SEL    # 激活描边
COLOR_BTN_HOVER = GuiTheme.BTN_HOVER      # 悬停底色
COLOR_BTN_TEXT_HOVER = GuiTheme.BTN_TEXT_HOVER  # 按钮悬停文字
COLOR_TEXT_SUB = GuiTheme.TEXT_SUB        # 副文字
COLOR_ACCENT = GuiTheme.ACCENT            # 主题强调色
COL_WHITE = GuiTheme.WHITE
COL_YELLOW = (90, 200, 245)               # 提示文字

TOOLBAR_H = 44  # 顶部工具栏高度


class CaptureRenderer:
    """采图向导渲染器: 输入主控制器状态, 输出画布与按钮命中表"""

    def __init__(self, wizard):
        self.wiz = wizard               # 主控制器状态引用 (只读)
        self.buttons = []               # [(btn_id, (x1,y1,x2,y2), payload), ...] 每帧重建
        self.mouse_pos = (-1, -1)
        self._workspace_rect = None
        self._purpose_rect = None
        self._hw_spec_rect = None

    # ------------------------------ 鼠标辅助 ------------------------------
    def on_mouse_move(self, x, y):
        self.mouse_pos = (x, y)

    def hit_test(self, x, y):
        """命中检测: 返回 (btn_id, payload) 或 None"""
        for btn_id, rect, payload in self.buttons:
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                return btn_id, payload
        return None

    def _is_hover(self, rect):
        x1, y1, x2, y2 = rect
        mx, my = self.mouse_pos
        return x1 <= mx <= x2 and y1 <= my <= y2

    def _hover_text(self, hovered, size, bold=False):
        if not hovered:
            return size, bold
        bh = GuiTheme.BTN_BEHAVIOR
        return int(size * bh["HOVER_SCALE"]), (bold or bh["HOVER_BOLD"])

    # ------------------------------ 下拉控件 ------------------------------
    def _draw_dropdown_button(self, canvas, rect, label, is_open):
        draw_dropdown_button(canvas, rect, label, is_open, self.mouse_pos, font_size=14)

    def _render_dropdown_popup(self, canvas, rect, options, active_key, btn_prefix):
        btns = render_dropdown_popup(canvas, rect, options, active_key, btn_prefix=btn_prefix, item_h=30)
        self.buttons.extend(btns)

    # ------------------------------ 工具栏 ------------------------------
    def draw_toolbar(self, canvas):
        """顶部单排工具栏:
        [工位 ▼] [用途 ▼] [相机类型 ▼] [分辨率 ▼] [开启/关闭] ... [退出 X]
        """
        wiz = self.wiz
        tw = canvas.shape[1]
        self.buttons = []
        cv2.rectangle(canvas, (0, 0), (tw, TOOLBAR_H), COLOR_BG, -1)
        cv2.line(canvas, (0, TOOLBAR_H - 1), (tw, TOOLBAR_H - 1), COLOR_BORDER, 1)

        y1, y2 = 6, 38
        gap = 6
        group_gap = 60  # 正常间隔 6px 的 10 倍大间隔，四大功能组视觉明确隔离

        # ==================== 第 1 大组：工作空间 与 采集用途 ====================
        # 1.1 工作空间选择下拉 (文案由"工位"更新为"工作空间")
        ws_x1 = 8
        ws_w = 195
        ws_x2 = ws_x1 + ws_w
        ws_label = getattr(wiz, "current_workspace_name", "默认")
        self._draw_dropdown_button(canvas, (ws_x1, y1, ws_x2, y2), f"工作空间: {ws_label}",
                                   is_open=(wiz.active_dropdown == "WS_DROPDOWN"))
        self.buttons.append(("TOGGLE_WS_DD", (ws_x1, y1, ws_x2, y2), "WS_DROPDOWN"))
        self._workspace_rect = (ws_x1, y1, ws_x2, y2)

        # 1.2 用途选择下拉 (内参 / 外参建图 / 生产采样)
        pur_x1 = ws_x2 + gap
        pur_w = 140
        pur_x2 = pur_x1 + pur_w
        pur_label = getattr(wiz, "current_purpose_label", "外参建图")
        self._draw_dropdown_button(canvas, (pur_x1, y1, pur_x2, y2), f"用途: {pur_label}",
                                   is_open=(wiz.active_dropdown == "PURPOSE_DROPDOWN"))
        self.buttons.append(("TOGGLE_PURPOSE_DD", (pur_x1, y1, pur_x2, y2), "PURPOSE_DROPDOWN"))
        self._purpose_rect = (pur_x1, y1, pur_x2, y2)

        # 组 1 与 组 2 间细微分割线
        sep1_x = pur_x2 + group_gap // 2
        cv2.line(canvas, (sep1_x, y1 + 4), (sep1_x, y2 - 4), (50, 56, 68), 1)

        # ==================== 第 2 大组：工位硬件与规格只读卡片 + 取流开关 ====================
        # 2.1 工位硬件设备与分辨率只读锁定卡片 (完全由 Workspace SSOT 驱动，禁止采图端动态更改)
        cam_type_str = "RealSense D435" if getattr(wiz, "camera_type", "realsense") == "realsense" else "USB 摄像头"
        cam_sn = getattr(wiz, "camera_serial", "") or ""
        cam_label = f"{cam_type_str} ({cam_sn})" if cam_sn else cam_type_str
        hw_spec_label = f"设备: {cam_label} · {wiz.resolution} [锁定]"

        hw_x1 = pur_x2 + group_gap
        hw_w = 265
        hw_x2 = hw_x1 + hw_w
        cv2.rectangle(canvas, (hw_x1, y1), (hw_x2, y2), (22, 28, 38), -1)
        cv2.rectangle(canvas, (hw_x1, y1), (hw_x2, y2), (45, 60, 80), 1)
        draw_text(canvas, hw_spec_label, (hw_x1 + 12, y1 + (y2 - y1 - 14) // 2), 12, (150, 195, 240), bold=True)
        self.buttons.append(("LOCKED_HW_CLICK", (hw_x1, y1, hw_x2, y2), None))
        self._hw_spec_rect = (hw_x1, y1, hw_x2, y2)

        # 2.2 开启/关闭取流乒乓按钮
        sw_x1 = hw_x2 + gap
        sw_w = 75
        sw_x2 = sw_x1 + sw_w
        sw_hover = self._is_hover((sw_x1, y1, sw_x2, y2))
        if wiz.pipeline_running:
            sw_bg, sw_border, sw_txt, sw_label = (55, 45, 30), (255, 160, 40), (255, 200, 80), "关闭"
        else:
            sw_bg, sw_border, sw_txt, sw_label = COLOR_CARD_BG, COLOR_BORDER, COLOR_TEXT_SUB, "开启"
        if sw_hover:
            sw_bg, sw_border = COLOR_BTN_HOVER, COLOR_BORDER_HOVER
        cv2.rectangle(canvas, (sw_x1, y1), (sw_x2, y2), sw_bg, -1)
        cv2.rectangle(canvas, (sw_x1, y1), (sw_x2, y2), sw_border, 1)
        sw_size, _ = self._hover_text(sw_hover, 15, True)
        draw_text(canvas, sw_label, (sw_x1 + (sw_w - 32) // 2, y1 + (y2 - y1 - 16) // 2 - 1), sw_size, sw_txt, True)
        self.buttons.append(("TOGGLE_CAMERA", (sw_x1, y1, sw_x2, y2), None))

        # 组 2 与 组 3 间细微分割线
        sep2_x = sw_x2 + group_gap // 2
        cv2.line(canvas, (sep2_x, y1 + 4), (sep2_x, y2 - 4), (50, 56, 68), 1)

        # ==================== 第 3 大组：拍照按钮 (位于中间偏右) ====================
        cap_x1 = sw_x2 + group_gap
        cap_w = 115
        cap_x2 = cap_x1 + cap_w
        cap_hover = self._is_hover((cap_x1, y1, cap_x2, y2))

        if wiz.pipeline_running:
            cap_bg = COLOR_BTN_HOVER if cap_hover else (24, 38, 34)
            cap_border = (0, 255, 180) if cap_hover else (0, 200, 140)
            cap_txt = (0, 255, 200) if cap_hover else (235, 250, 245)
            cap_label = "拍照 (空格)"
        else:
            cap_bg = COLOR_CARD_BG
            cap_border = COLOR_BORDER
            cap_txt = (110, 115, 125)
            cap_label = "拍照"

        cv2.rectangle(canvas, (cap_x1, y1), (cap_x2, y2), cap_bg, -1)
        cv2.rectangle(canvas, (cap_x1, y1), (cap_x2, y2), cap_border, 1)
        cap_size, _ = self._hover_text(cap_hover, 14, True)
        draw_text(canvas, cap_label, (cap_x1 + 14, y1 + (y2 - y1 - 16) // 2 - 1), cap_size, cap_txt, True)
        self.buttons.append(("CAPTURE", (cap_x1, y1, cap_x2, y2), None))

        # ==================== 第 4 大组：退出按钮 (最右侧) ====================
        exit_x1, exit_x2 = tw - 90, tw - 8
        q_hover = self._is_hover((exit_x1, y1, exit_x2, y2))
        cv2.rectangle(canvas, (exit_x1, y1), (exit_x2, y2),
                      COLOR_BTN_HOVER if q_hover else COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (exit_x1, y1), (exit_x2, y2),
                      COLOR_BORDER_HOVER if q_hover else COLOR_BORDER, 1)
        q_size, _ = self._hover_text(q_hover, 14, True)
        draw_text(canvas, "退出 X", (exit_x1 + 14, y1 + (y2 - y1 - 16) // 2 - 1), q_size,
                  COLOR_BTN_TEXT_HOVER if q_hover else (190, 190, 200), True)
        self.buttons.append(("QUIT", (exit_x1, y1, exit_x2, y2), None))

        # 展开的下拉浮层 (置顶最后绘制)
        if wiz.active_dropdown == "WS_DROPDOWN" and getattr(self, "_workspace_rect", None):
            self._render_dropdown_popup(canvas, self._workspace_rect,
                                        wiz.workspace_options, wiz.current_workspace_id, "DD_WS_")
        elif wiz.active_dropdown == "PURPOSE_DROPDOWN" and getattr(self, "_purpose_rect", None):
            self._render_dropdown_popup(canvas, self._purpose_rect,
                                        wiz.purpose_options, wiz.purpose, "DD_PURPOSE_")

    # ------------------------------ 画布合成 ------------------------------
    def make_canvas(self):
        cw = self.wiz.win_mgr.canvas_w
        ch = self.wiz.win_mgr.canvas_h
        return np.full((ch, cw, 3), COLOR_BG, dtype=np.uint8)

    def compose_canvas(self, frame):
        cw = self.wiz.win_mgr.canvas_w
        ch = self.wiz.win_mgr.canvas_h
        canvas = np.full((ch, cw, 3), COLOR_BG, dtype=np.uint8)

        area_h = ch - TOOLBAR_H
        fh, fw = frame.shape[:2]
        scale = min(cw / fw, area_h / fh)
        nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        frame2 = cv2.resize(frame, (nw, nh), interpolation=interp)
        x0, y0 = (cw - nw) // 2, TOOLBAR_H + (area_h - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = frame2
        return canvas

    def draw_toast(self, canvas):
        wiz = self.wiz
        h, w = canvas.shape[:2]
        if time.time() - wiz.status_toast_time < 2.5 and wiz.status_toast:
            (tw, _), _ = self._measure(wiz.status_toast)
            toast_x = (w - tw) // 2
            cv2.rectangle(canvas, (toast_x - 12, h - 60), (toast_x + tw + 12, h - 25),
                          (0, 120, 0), -1)
            draw_text(canvas, wiz.status_toast, (toast_x, h - 52), 16, COL_WHITE)

    @staticmethod
    def _measure(text):
        from src.utils.text_rendering import measure_text
        return measure_text(text, font_size=16, bold=False)
