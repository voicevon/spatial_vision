#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flux_vision_3d | isolate_wheels_production - 生产大屏视图渲染器
=============================================================
专为工业生产大屏定制的高对比度、信息密集型现代化 UI 渲染模块：
- 8 轮通道横向拓扑流水线（物料从左向右：1 -> 2 -> ... -> 8 -> 物理出料口）
- 生产模式与效能指标监控（PPM、节拍耗时、累计出料、实时拓扑负载）
- 参数与操作面板：速度倍率调节、单步/连续自动生产开关、配方预设
"""

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.gui_theme import GuiTheme
from src.utils.text_rendering import draw_text, measure_text

# 逻辑视窗分辨率
LOGIC_W = 1280
LOGIC_H = 830

# 配色系统
COLOR_BG = (20, 22, 26)
COLOR_PANEL = (28, 32, 38)
COLOR_PANEL_ALT = (34, 38, 46)
COLOR_BORDER = (45, 52, 62)
COLOR_BORDER_HL = (65, 80, 100)
COLOR_TEXT = (240, 240, 245)
COLOR_SUB = (160, 168, 180)
COLOR_MUTED = (90, 100, 115)
COLOR_ACCENT = (230, 160, 50)      # 琥珀金
COLOR_GREEN = (60, 185, 120)       # 运行与在线
COLOR_BLUE = (80, 150, 240)        # 科技蓝
COLOR_AMBER = (240, 160, 40)       # 警告/动作中
COLOR_RED = (225, 75, 75)          # 故障/离线
COLOR_PURPLE = (175, 110, 240)     # 连续自动模式紫


def draw_centered_text(
    canvas: np.ndarray,
    text: str,
    cx: int,
    cy: int,
    font_size: int = 14,
    color: Tuple[int, int, int] = COLOR_TEXT,
    bold: bool = False,
):
    """在 (cx, cy) 中心对齐绘制文本"""
    if not text:
        return
    (tw, th), _ = measure_text(text, font_size=font_size, bold=bold)
    draw_text(canvas, text, (cx - tw // 2, cy - th // 2), font_size=font_size, color=color, bold=bold)


class ProductionRenderer:
    """分离轮 8 通道自动化生产工作台渲染器"""

    @staticmethod
    def pt_in(x: int, y: int, rect: Tuple[int, int, int, int]) -> bool:
        rx, ry, rw, rh = rect
        return rx <= x < rx + rw and ry <= y < ry + rh

    @classmethod
    def draw_btn(
        cls,
        canvas: np.ndarray,
        rect: Tuple[int, int, int, int],
        text: str,
        mpos: Tuple[int, int],
        theme_color: Optional[Tuple[int, int, int]] = None,
        enabled: bool = True,
        bold: bool = False,
        active: bool = False,
    ):
        rx, ry, rw, rh = rect
        hover = enabled and cls.pt_in(mpos[0], mpos[1], rect)

        if not enabled:
            bg = (24, 26, 30)
            border = (40, 44, 52)
            fg = COLOR_MUTED
        elif active:
            bg = (50, 75, 110) if theme_color is None else (
                min(255, int(theme_color[0] * 0.4 + 20)),
                min(255, int(theme_color[1] * 0.4 + 20)),
                min(255, int(theme_color[2] * 0.4 + 20)),
            )
            border = theme_color or COLOR_ACCENT
            fg = (255, 255, 255)
        elif hover:
            bg = (42, 48, 58)
            border = theme_color or COLOR_ACCENT
            fg = (255, 255, 255)
        else:
            bg = COLOR_PANEL
            border = COLOR_BORDER
            fg = COLOR_TEXT

        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), bg, -1)
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), border, 1)

        draw_centered_text(
            canvas,
            text,
            rx + rw // 2,
            ry + rh // 2,
            font_size=12,
            color=fg,
            bold=bold or active,
        )

    def render(self, app, canvas: np.ndarray, mpos: Tuple[int, int]):
        """执行完整工作台画面绘制"""
        canvas[:] = COLOR_BG

        self._draw_header(app, canvas, mpos)
        self._draw_pipeline(app, canvas, mpos)
        self._draw_control_panel(app, canvas, mpos)
        self._draw_metrics_panel(app, canvas)
        self._draw_log_panel(app, canvas)

        if app.popup_speed:
            self._draw_speed_popup(app, canvas, mpos)

    # ---------------- 顶部状态栏 ----------------
    def _draw_header(self, app, canvas: np.ndarray, mpos: Tuple[int, int]):
        cv2.rectangle(canvas, (0, 0), (LOGIC_W, 56), COLOR_PANEL, -1)
        cv2.line(canvas, (0, 56), (LOGIC_W, 56), COLOR_BORDER, 1)

        # 标题与版本
        draw_text(canvas, "flux_vision_3d | 分离轮 8 通道自动生产工作台", (18, 12), font_size=18, color=COLOR_TEXT, bold=True)
        draw_text(canvas, "FLUX ISOLATE WHEELS AUTO PRODUCTION SUITE v1.3", (18, 36), font_size=11, color=COLOR_SUB)

        # 生产模式状态 Badge
        mode_text = "● 自动循环生产中" if app.auto_running else ("❚❚ 生产暂停" if app.paused else "○ 手动就绪")
        mode_color = COLOR_GREEN if app.auto_running else (COLOR_AMBER if app.paused else COLOR_BLUE)
        badge_w = 140
        cv2.rectangle(canvas, (540, 14), (540 + badge_w, 42), (25, 30, 38), -1)
        cv2.rectangle(canvas, (540, 14), (540 + badge_w, 42), mode_color, 1)
        draw_centered_text(canvas, mode_text, 540 + badge_w // 2, 28, font_size=12, color=mode_color, bold=True)

        # MQTT 设备连接状态
        dev_state = app.controller.get_device_state(app.selected_devid)
        online = app.controller.is_connected
        st_color = COLOR_GREEN if (online and dev_state == "idle") else (COLOR_AMBER if dev_state == "running" else COLOR_RED)
        st_text = f"Broker: {'在线' if online else '离线'} | 设备 [{app.selected_devid}]: {dev_state or '未知'}"
        draw_text(canvas, st_text, (710, 20), font_size=13, color=st_color, bold=True)

        # 速度倍率下拉按钮
        speed_rect = (1120, 14, 140, 28)
        spd_str = f"速度倍率: {app.load_speed:.1f}x ▼"
        self.draw_btn(canvas, speed_rect, spd_str, mpos, theme_color=COLOR_ACCENT, bold=True)

    # ---------------- 核心生产流水线拓扑图 ----------------
    def _draw_pipeline(self, app, canvas: np.ndarray, mpos: Tuple[int, int]):
        py = 70
        ph = 250
        cv2.rectangle(canvas, (18, py), (LOGIC_W - 18, py + ph), COLOR_PANEL, -1)
        cv2.rectangle(canvas, (18, py), (LOGIC_W - 18, py + ph), COLOR_BORDER, 1)

        draw_text(canvas, "8 托架通道实时物流拓扑 (物料从左向右流向，8 号轮右侧出料)", (32, py + 12), font_size=13, color=COLOR_SUB, bold=True)

        # 8 个托架卡片
        card_w = 118
        card_h = 180
        gap = 14
        start_x = 32
        card_y = py + 42

        for i in range(8):
            cx = start_x + i * (card_w + gap)
            is_active = (app.recipe_counts[i] > 0)
            b_color = COLOR_BORDER_HL if is_active else COLOR_BORDER
            fill_color = (36, 42, 52) if is_active else COLOR_PANEL_ALT

            # 卡片背景
            cv2.rectangle(canvas, (cx, card_y), (cx + card_w, card_y + card_h), fill_color, -1)
            cv2.rectangle(canvas, (cx, card_y), (cx + card_w, card_y + card_h), b_color, 1)

            # 顶部轮号
            draw_centered_text(canvas, f"轮 #{i + 1}", cx + card_w // 2, card_y + 20, font_size=14, color=COLOR_ACCENT, bold=True)

            # 本轮目标出料数
            cnt = app.recipe_counts[i]
            draw_centered_text(canvas, f"{cnt}", cx + card_w // 2, card_y + 60, font_size=28, color=COLOR_TEXT if cnt > 0 else COLOR_MUTED, bold=True)
            draw_centered_text(canvas, "配方落料数", cx + card_w // 2, card_y + 90, font_size=11, color=COLOR_SUB)

            # 步进加减按钮
            btn_minus = (cx + 10, card_y + 106, 44, 26)
            btn_plus = (cx + card_w - 54, card_y + 106, 44, 26)
            self.draw_btn(canvas, btn_minus, "-1", mpos, theme_color=COLOR_SUB, enabled=(cnt > 0))
            self.draw_btn(canvas, btn_plus, "+1", mpos, theme_color=COLOR_ACCENT)

            # 累计产量统计
            discharged = app.discharged_counts[i]
            draw_centered_text(canvas, f"已出料: {discharged} 件", cx + card_w // 2, card_y + 154, font_size=11, color=COLOR_GREEN if discharged > 0 else COLOR_MUTED)

            # 流向箭头 (1->2, 2->3 ... 8->出口)
            if i < 7:
                arrow_x = cx + card_w + 2
                arrow_y = card_y + card_h // 2
                cv2.arrowedLine(canvas, (arrow_x, arrow_y), (arrow_x + gap - 4, arrow_y), COLOR_BORDER_HL, 1, tipLength=0.4)

        # 8 号轮右侧物理出料口指示
        out_x = start_x + 8 * (card_w + gap)
        cv2.rectangle(canvas, (out_x, card_y), (out_x + 110, card_y + card_h), (25, 45, 35), -1)
        cv2.rectangle(canvas, (out_x, card_y), (out_x + 110, card_y + card_h), COLOR_GREEN, 1)
        draw_centered_text(canvas, "物理出料口", out_x + 55, card_y + 30, font_size=14, color=COLOR_GREEN, bold=True)
        draw_centered_text(canvas, "8号右侧出口", out_x + 55, card_y + 55, font_size=11, color=COLOR_SUB)
        total_out = sum(app.discharged_counts)
        draw_centered_text(canvas, f"{total_out}", out_x + 55, card_y + 100, font_size=28, color=COLOR_GREEN, bold=True)
        draw_centered_text(canvas, "总出料 (件)", out_x + 55, card_y + 140, font_size=11, color=COLOR_SUB)

    # ---------------- 生产操作面板 ----------------
    def _draw_control_panel(self, app, canvas: np.ndarray, mpos: Tuple[int, int]):
        py = 335
        pw = 740
        ph = 210
        cv2.rectangle(canvas, (18, py), (18 + pw, py + ph), COLOR_PANEL, -1)
        cv2.rectangle(canvas, (18, py), (18 + pw, py + ph), COLOR_BORDER, 1)

        draw_text(canvas, "生产运行调度控制", (32, py + 14), font_size=14, color=COLOR_TEXT, bold=True)

        # 核心按钮：启动连续自动生产 / 暂停生产
        btn_run_rect = (32, py + 45, 210, 44)
        if app.auto_running:
            self.draw_btn(canvas, btn_run_rect, "❚❚ 暂停自动循环 (SPACE)", mpos, theme_color=COLOR_AMBER, bold=True, active=True)
        else:
            self.draw_btn(canvas, btn_run_rect, "▶ 启动连续生产 (SPACE)", mpos, theme_color=COLOR_GREEN, bold=True)

        # 单拍出料生产
        btn_step_rect = (255, py + 45, 180, 44)
        self.draw_btn(canvas, btn_step_rect, "► 执行单拍出料 (S)", mpos, theme_color=COLOR_BLUE, bold=True, enabled=not app.auto_running)

        # 紧急停止 / 立即打断
        btn_stop_rect = (450, py + 45, 140, 44)
        self.draw_btn(canvas, btn_stop_rect, "⏹ 停止生产 (ESC)", mpos, theme_color=COLOR_RED, bold=True)

        # 清除批次计数
        btn_clear_rect = (605, py + 45, 135, 44)
        self.draw_btn(canvas, btn_clear_rect, "↺ 重置批次统计", mpos, theme_color=COLOR_MUTED)

        # 配方快捷设置按钮组
        draw_text(canvas, "配方快捷预设:", (32, py + 115), font_size=12, color=COLOR_SUB)
        preset_all1 = (120, py + 105, 110, 32)
        preset_all2 = (240, py + 105, 110, 32)
        preset_fill0 = (360, py + 105, 110, 32)
        self.draw_btn(canvas, preset_all1, "全部设为 1", mpos, theme_color=COLOR_BORDER_HL)
        self.draw_btn(canvas, preset_all2, "全部设为 2", mpos, theme_color=COLOR_BORDER_HL)
        self.draw_btn(canvas, preset_fill0, "清空配方 (0)", mpos, theme_color=COLOR_BORDER_HL)

        # 节拍间隔调节
        draw_text(canvas, "自动循环间隔:", (32, py + 160), font_size=12, color=COLOR_SUB)
        delays = [0.5, 1.0, 2.0, 3.0]
        for idx, d in enumerate(delays):
            btn_d = (120 + idx * 75, py + 152, 65, 30)
            is_cur = abs(app.cycle_delay - d) < 0.05
            self.draw_btn(canvas, btn_d, f"{d}s", mpos, theme_color=COLOR_ACCENT, active=is_cur)

    # ---------------- 生产指标大屏 ----------------
    def _draw_metrics_panel(self, app, canvas: np.ndarray):
        px = 772
        py = 335
        pw = LOGIC_W - px - 18
        ph = 210
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), COLOR_PANEL, -1)
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), COLOR_BORDER, 1)

        draw_text(canvas, "实时生产效能统计 (OEE)", (px + 16, py + 14), font_size=14, color=COLOR_TEXT, bold=True)

        # 四大指标卡片网格
        grid = [
            ("总完成节拍 (Cycles)", f"{app.total_cycles} 拍", COLOR_ACCENT),
            ("累计出料总量 (Pcs)", f"{sum(app.discharged_counts)} 件", COLOR_GREEN),
            ("平均节拍耗时 (CT)", f"{app.avg_cycle_time:.2f} s", COLOR_BLUE),
            ("瞬时生产速率 (PPM)", f"{app.calculate_ppm():.1f} 件/分", COLOR_PURPLE),
        ]

        gw = (pw - 48) // 2
        gh = 70
        for idx, (label, val, col) in enumerate(grid):
            gx = px + 16 + (idx % 2) * (gw + 16)
            gy = py + 45 + (idx // 2) * (gh + 12)
            cv2.rectangle(canvas, (gx, gy), (gx + gw, gy + gh), COLOR_PANEL_ALT, -1)
            cv2.rectangle(canvas, (gx, gy), (gx + gw, gy + gh), COLOR_BORDER, 1)
            draw_text(canvas, label, (gx + 12, gy + 10), font_size=11, color=COLOR_SUB)
            draw_text(canvas, val, (gx + 12, gy + 32), font_size=20, color=col, bold=True)

    # ---------------- 生产日志面板 ----------------
    def _draw_log_panel(self, app, canvas: np.ndarray):
        py = 560
        ph = LOGIC_H - py - 18
        cv2.rectangle(canvas, (18, py), (LOGIC_W - 18, py + ph), COLOR_PANEL, -1)
        cv2.rectangle(canvas, (18, py), (LOGIC_W - 18, py + ph), COLOR_BORDER, 1)

        draw_text(canvas, "生产流水与遥测日志", (32, py + 12), font_size=13, color=COLOR_SUB, bold=True)

        # 日志内容绘制 (显示最近 9 条)
        with app.controller._lock:
            lines = list(app.controller.log_lines)[-9:]
        for idx, line in enumerate(lines):
            ly = py + 38 + idx * 22
            col = COLOR_TEXT
            if "done" in line.lower() or "节拍完成" in line:
                col = COLOR_GREEN
            elif "fail" in line.lower() or "error" in line.lower() or "离线" in line:
                col = COLOR_RED
            elif "下发" in line:
                col = COLOR_ACCENT
            draw_text(canvas, line, (32, ly), font_size=12, color=col)

    # ---------------- 速度选择下拉弹窗 ----------------
    def _draw_speed_popup(self, app, canvas: np.ndarray, mpos: Tuple[int, int]):
        speeds = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
        pop_w = 140
        pop_h = len(speeds) * 28 + 8
        pop_x = 1120
        pop_y = 44

        cv2.rectangle(canvas, (pop_x, pop_y), (pop_x + pop_w, pop_y + pop_h), (24, 28, 36), -1)
        cv2.rectangle(canvas, (pop_x, pop_y), (pop_x + pop_w, pop_y + pop_h), COLOR_ACCENT, 1)

        for idx, spd in enumerate(speeds):
            iy = pop_y + 4 + idx * 28
            rect = (pop_x + 4, iy, pop_w - 8, 24)
            is_cur = abs(app.load_speed - spd) < 0.01
            self.draw_btn(canvas, rect, f"{spd:.1f}x 倍速", mpos, theme_color=COLOR_ACCENT, active=is_cur)
