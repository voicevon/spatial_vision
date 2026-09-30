#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flux_vision_3d | isolate_wheels_production - 生产大屏视图渲染器
=============================================================
专为工业生产大屏定制的双流联动渲染模块：
1. 顶部控制栏：工位选择、相机源与分辨率配置、相机启停、速度调节、生产模式 Badge；
2. 左侧主区：实时相机画面与 8 通道视觉分离 ROI 动态检测叠加层；
3. 右侧主区：8 轮横向物流拓扑看板（视觉实时自动计数展示、物理出料口、累计统计、PPM效能指标）；
4. 底部主区：生产流水线调度控制面板与实时事件遥测日志；
5. 下拉弹窗：工位选择、相机设备、分辨率、速度倍率。
"""

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.ui.gui_theme import GuiTheme
from src.ui.text_rendering import draw_text, measure_text

# 逻辑视窗分辨率
LOGIC_W = 1280
LOGIC_H = 850

# 配色系统
COLOR_BG = (18, 20, 24)
COLOR_PANEL = (25, 29, 36)
COLOR_PANEL_ALT = (32, 36, 45)
COLOR_BORDER = (45, 52, 64)
COLOR_BORDER_HL = (65, 85, 110)
COLOR_TEXT = (240, 240, 245)
COLOR_SUB = (155, 165, 180)
COLOR_MUTED = (85, 95, 110)
COLOR_ACCENT = (230, 160, 50)      # 琥珀金
COLOR_GREEN = (50, 205, 110)       # 运行与在线
COLOR_BLUE = (70, 150, 240)        # 科技蓝
COLOR_AMBER = (240, 155, 40)       # 警告/动作中
COLOR_RED = (230, 70, 70)          # 故障/离线
COLOR_PURPLE = (175, 110, 240)     # 连续流水线紫


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
    """分离轮 8 通道自动化视觉生产工作台渲染器"""

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
            bg = (22, 24, 28)
            border = (38, 42, 50)
            fg = COLOR_MUTED
        elif active:
            bg = (45, 68, 98) if theme_color is None else (
                min(255, int(theme_color[0] * 0.35 + 20)),
                min(255, int(theme_color[1] * 0.35 + 20)),
                min(255, int(theme_color[2] * 0.35 + 20)),
            )
            border = theme_color or COLOR_ACCENT
            fg = (255, 255, 255)
        elif hover:
            bg = (38, 44, 54)
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

        self._draw_header_controls(app, canvas, mpos)
        self._draw_video_view(app, canvas, mpos)
        self._draw_wheels_dashboard(app, canvas, mpos)
        self._draw_control_and_logs(app, canvas, mpos)

        # 浮动弹窗层
        if app.popup_ws:
            self._draw_ws_popup(app, canvas, mpos)
        elif app.popup_cam:
            self._draw_cam_popup(app, canvas, mpos)
        elif app.popup_res:
            self._draw_res_popup(app, canvas, mpos)
        elif app.popup_speed:
            self._draw_speed_popup(app, canvas, mpos)

    # ---------------- 顶部控制栏 ----------------
    def _draw_header_controls(self, app, canvas: np.ndarray, mpos: Tuple[int, int]):
        cv2.rectangle(canvas, (0, 0), (LOGIC_W, 60), COLOR_PANEL, -1)
        cv2.line(canvas, (0, 60), (LOGIC_W, 60), COLOR_BORDER, 1)

        # 标题与副标
        draw_text(canvas, "flux_vision_3d | 分离轮 8 通道视觉生产工作台", (18, 12), font_size=16, color=COLOR_TEXT, bold=True)
        dev_st = app.controller.get_device_state(app.controller.selected_devid)
        dev_tag = f"设备 [{app.controller.selected_devid}]: {dev_st or '未知'}"
        draw_text(canvas, f"AUTO ISOLATE WHEELS VISION PIPELINE | {dev_tag}", (18, 36), font_size=11, color=COLOR_SUB)

        # 步骤 1：工位选择下拉按钮
        ws_name = app.current_workspace_name
        if len(ws_name) > 12:
            ws_name = ws_name[:11] + "…"
        ws_rect = (430, 14, 160, 32)
        self.draw_btn(canvas, ws_rect, f"工位: {ws_name} ▼", mpos, theme_color=COLOR_BLUE, bold=True)

        # 步骤 2：相机设备与分辨率选择
        cam_desc = app.current_camera_desc
        cam_rect = (600, 14, 130, 32)
        self.draw_btn(canvas, cam_rect, f"相机: {cam_desc} ▼", mpos, theme_color=COLOR_BLUE)

        res_desc = f"{app.camera_w}x{app.camera_h}"
        res_rect = (740, 14, 110, 32)
        self.draw_btn(canvas, res_rect, f"{res_desc} ▼", mpos, theme_color=COLOR_BLUE)

        # 相机启停按钮
        cam_btn_rect = (860, 14, 90, 32)
        if app.is_camera_running:
            self.draw_btn(canvas, cam_btn_rect, "关闭相机", mpos, theme_color=COLOR_RED, bold=True, active=True)
        else:
            self.draw_btn(canvas, cam_btn_rect, "打开相机", mpos, theme_color=COLOR_GREEN, bold=True)

        # 速度倍率下拉按钮
        speed_rect = (960, 14, 100, 32)
        self.draw_btn(canvas, speed_rect, f"速度: {app.load_speed:.1f}x ▼", mpos, theme_color=COLOR_ACCENT, bold=True)

        # 生产模式 Badge
        mode_rect = (1070, 14, 192, 32)
        if app.auto_pipeline:
            mode_text = "● 自动流水线运行中"
            mode_col = COLOR_PURPLE
        elif app.paused:
            mode_text = "❚❚ 生产暂停"
            mode_col = COLOR_AMBER
        else:
            mode_text = "○ 单次手动模式"
            mode_col = COLOR_BLUE

        cv2.rectangle(canvas, (mode_rect[0], mode_rect[1]), (mode_rect[0] + mode_rect[2], mode_rect[1] + mode_rect[3]), (22, 26, 34), -1)
        cv2.rectangle(canvas, (mode_rect[0], mode_rect[1]), (mode_rect[0] + mode_rect[2], mode_rect[1] + mode_rect[3]), mode_col, 1)
        draw_centered_text(canvas, mode_text, mode_rect[0] + mode_rect[2] // 2, mode_rect[1] + mode_rect[3] // 2, font_size=12, color=mode_col, bold=True)

    # ---------------- 左侧实时相机画面与 ROI 叠加层 ----------------
    def _draw_video_view(self, app, canvas: np.ndarray, mpos: Tuple[int, int]):
        vx, vy, vw, vh = 18, 70, 610, 480
        cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), COLOR_PANEL, -1)
        cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), COLOR_BORDER, 1)

        # 顶栏标题
        draw_text(canvas, "实时相机取流与 8 通道视觉分离检测 (自动计数)", (vx + 14, vy + 12), font_size=13, color=COLOR_TEXT, bold=True)
        fps_text = f"{app.camera_fps:.1f} FPS" if app.is_camera_running else "未连接"
        draw_text(canvas, fps_text, (vx + vw - 75, vy + 12), font_size=11, color=COLOR_GREEN if app.is_camera_running else COLOR_MUTED, bold=True)

        # 视频渲染区域
        disp_x, disp_y, disp_w, disp_h = vx + 10, vy + 38, vw - 20, vh - 48
        
        if app.current_frame is not None and app.is_camera_running:
            # 缩放至显示区域
            frame_resized = cv2.resize(app.current_frame, (disp_w, disp_h))
            # 叠加 8 个 ROI 及实时检测框
            app.detector.draw_overlay(frame_resized)
            canvas[disp_y:disp_y + disp_h, disp_x:disp_x + disp_w] = frame_resized
        else:
            # 相机未开启或黑屏占位
            cv2.rectangle(canvas, (disp_x, disp_y), (disp_x + disp_w, disp_y + disp_h), (14, 16, 20), -1)
            cv2.rectangle(canvas, (disp_x, disp_y), (disp_x + disp_w, disp_y + disp_h), COLOR_BORDER, 1)
            draw_centered_text(canvas, "相机未启动，请点击顶部 [打开相机] 开始视觉分离", disp_x + disp_w // 2, disp_y + disp_h // 2 - 12, font_size=13, color=COLOR_SUB)
            draw_centered_text(canvas, f"当前选定源: {app.current_camera_desc} ({app.camera_w}x{app.camera_h})", disp_x + disp_w // 2, disp_y + disp_h // 2 + 16, font_size=11, color=COLOR_MUTED)

    # ---------------- 右侧分离轮生产拓扑与效能看板 ----------------
    def _draw_wheels_dashboard(self, app, canvas: np.ndarray, mpos: Tuple[int, int]):
        dx, dy, dw, dh = 640, 70, LOGIC_W - 640 - 18, 480
        cv2.rectangle(canvas, (dx, dy), (dx + dw, dy + dh), COLOR_PANEL, -1)
        cv2.rectangle(canvas, (dx, dy), (dx + dw, dy + dh), COLOR_BORDER, 1)

        draw_text(canvas, "8 托架通道实时物流拓扑 (物料从左向右传输，8 号轮右侧出料)", (dx + 14, dy + 12), font_size=13, color=COLOR_TEXT, bold=True)
        draw_text(canvas, "视觉自动检出数量（无需人工干预）", (dx + 14, dy + 32), font_size=11, color=COLOR_SUB)

        # 8 个轮托架卡片横向排布
        grid_y = dy + 55
        card_w = 60
        card_h = 180
        gap = 8
        start_x = dx + 14

        for i in range(8):
            cx = start_x + i * (card_w + gap)
            cnt = app.detected_counts[i]
            is_active = (cnt > 0)
            
            fill_col = (34, 44, 56) if is_active else COLOR_PANEL_ALT
            b_col = COLOR_ACCENT if is_active else COLOR_BORDER

            cv2.rectangle(canvas, (cx, grid_y), (cx + card_w, grid_y + card_h), fill_col, -1)
            cv2.rectangle(canvas, (cx, grid_y), (cx + card_w, grid_y + card_h), b_col, 1)

            # 轮号
            draw_centered_text(canvas, f"#{i + 1}", cx + card_w // 2, grid_y + 16, font_size=13, color=COLOR_ACCENT, bold=True)

            # 视觉数出的待分离数量
            num_col = COLOR_GREEN if is_active else COLOR_MUTED
            draw_centered_text(canvas, f"{cnt}", cx + card_w // 2, grid_y + 65, font_size=32, color=num_col, bold=True)
            draw_centered_text(canvas, "待分离", cx + card_w // 2, grid_y + 102, font_size=10, color=COLOR_SUB)

            # 累计出料
            discharged = app.discharged_counts[i]
            draw_centered_text(canvas, f"{discharged}", cx + card_w // 2, grid_y + 140, font_size=14, color=COLOR_TEXT, bold=True)
            draw_centered_text(canvas, "已出料", cx + card_w // 2, grid_y + 160, font_size=10, color=COLOR_MUTED)

            # 箭头流向指示 (1->2, ..., 8->出口)
            if i < 7:
                ax = cx + card_w + 1
                ay = grid_y + card_h // 2
                cv2.arrowedLine(canvas, (ax, ay), (ax + gap - 2, ay), COLOR_BORDER_HL, 1, tipLength=0.5)

        # 8 号轮右侧物理出料口指示
        out_x = start_x + 8 * (card_w + gap) + 4
        out_w = dw - (out_x - dx) - 14
        out_rect = (out_x, grid_y, out_w, card_h)
        cv2.rectangle(canvas, (out_x, grid_y), (out_x + out_w, grid_y + card_h), (20, 38, 28), -1)
        cv2.rectangle(canvas, (out_x, grid_y), (out_x + out_w, grid_y + card_h), COLOR_GREEN, 1)

        draw_centered_text(canvas, "物理出口", out_x + out_w // 2, grid_y + 24, font_size=13, color=COLOR_GREEN, bold=True)
        draw_centered_text(canvas, "8号右侧出料", out_x + out_w // 2, grid_y + 46, font_size=10, color=COLOR_SUB)
        total_out = sum(app.discharged_counts)
        draw_centered_text(canvas, f"{total_out}", out_x + out_w // 2, grid_y + 95, font_size=28, color=COLOR_GREEN, bold=True)
        draw_centered_text(canvas, "累计总出料 (件)", out_x + out_w // 2, grid_y + 140, font_size=10, color=COLOR_SUB)

        # 下方：生产效能指标 (OEE 概览)
        stat_y = grid_y + card_h + 20
        draw_text(canvas, "生产效能实时指标 (OEE 指标监视)", (dx + 14, stat_y), font_size=13, color=COLOR_TEXT, bold=True)

        cards = [
            ("总完成节拍 (Cycles)", f"{app.total_cycles} 拍", COLOR_ACCENT),
            ("实时生产速率 (PPM)", f"{app.calculate_ppm():.1f} 件/分", COLOR_PURPLE),
            ("平均节拍耗时 (CT)", f"{app.avg_cycle_time:.2f} s", COLOR_BLUE),
            ("下位机通信响应", f"{app.controller.last_done_cmd or 'idle'}", COLOR_GREEN),
        ]
        stat_w = (dw - 28 - 24) // 2
        stat_h = 75
        for idx, (lbl, val, col) in enumerate(cards):
            gx = dx + 14 + (idx % 2) * (stat_w + 24)
            gy = stat_y + 24 + (idx // 2) * (stat_h + 12)
            cv2.rectangle(canvas, (gx, gy), (gx + stat_w, gy + stat_h), COLOR_PANEL_ALT, -1)
            cv2.rectangle(canvas, (gx, gy), (gx + stat_w, gy + stat_h), COLOR_BORDER, 1)
            draw_text(canvas, lbl, (gx + 12, gy + 10), font_size=11, color=COLOR_SUB)
            draw_text(canvas, val, (gx + 12, gy + 34), font_size=20, color=col, bold=True)

    # ---------------- 底部控制条与日志面板 ----------------
    def _draw_control_and_logs(self, app, canvas: np.ndarray, mpos: Tuple[int, int]):
        by = 560
        bh = LOGIC_H - by - 16
        cv2.rectangle(canvas, (18, by), (LOGIC_W - 18, by + bh), COLOR_PANEL, -1)
        cv2.rectangle(canvas, (18, by), (LOGIC_W - 18, by + bh), COLOR_BORDER, 1)

        # 左侧生产控制按键组 (宽 520px)
        draw_text(canvas, "生产调度调度控制 (步骤四: 单次模式 / 连续自动流水线)", (32, by + 12), font_size=13, color=COLOR_TEXT, bold=True)

        # 启动连续流水线 / 暂停流水线
        btn_pipeline = (32, by + 40, 220, 46)
        if app.auto_pipeline:
            self.draw_btn(canvas, btn_pipeline, "❚❚ 暂停连续流水线 (SPACE)", mpos, theme_color=COLOR_AMBER, bold=True, active=True)
        else:
            self.draw_btn(canvas, btn_pipeline, "▶ 启动连续生产流水线 (SPACE)", mpos, theme_color=COLOR_PURPLE, bold=True)

        # 单拍生产出料
        btn_single = (264, by + 40, 180, 46)
        self.draw_btn(canvas, btn_single, "► 执行单次生产节拍 (S)", mpos, theme_color=COLOR_BLUE, bold=True, enabled=not app.auto_pipeline)

        # 紧急停止
        btn_stop = (454, by + 40, 120, 46)
        self.draw_btn(canvas, btn_stop, "⏹ 停止 (ESC)", mpos, theme_color=COLOR_RED, bold=True)

        # 重置批次统计
        btn_reset = (584, by + 40, 130, 46)
        self.draw_btn(canvas, btn_reset, "↺ 重置批次统计 (C)", mpos, theme_color=COLOR_MUTED)

        # 流水线模式运行机制提示胶囊
        tip_box = (32, by + 100, 682, 38)
        cv2.rectangle(canvas, (tip_box[0], tip_box[1]), (tip_box[0] + tip_box[2], tip_box[1] + tip_box[3]), (22, 26, 32), -1)
        cv2.rectangle(canvas, (tip_box[0], tip_box[1]), (tip_box[0] + tip_box[2], tip_box[1] + tip_box[3]), COLOR_BORDER, 1)
        if app.auto_pipeline:
            tip_msg = "【流水线闭环运行中】当收到下位机 MQTT done 信号时，自动以视觉最新检出数量下发下一个节拍"
            tip_col = COLOR_PURPLE
        else:
            tip_msg = "【单次生产模式】由人工点击按钮或按 [S] 触发当前节拍下发；也可按 [SPACE] 切入连续流水线"
            tip_col = COLOR_SUB
        draw_text(canvas, tip_msg, (tip_box[0] + 12, tip_box[1] + 11), font_size=11, color=tip_col, bold=app.auto_pipeline)

        # 右侧日志面板 (从 X: 730 开始)
        lx = 730
        draw_text(canvas, "生产遥测流水与事件日志", (lx + 10, by + 12), font_size=13, color=COLOR_SUB, bold=True)
        cv2.line(canvas, (lx, by + 8), (lx, by + bh - 8), COLOR_BORDER, 1)

        with app.controller._lock:
            lines = list(app.controller.log_lines)[-8:]
        for idx, line in enumerate(lines):
            ly = by + 36 + idx * 24
            col = COLOR_TEXT
            if "done" in line.lower() or "节拍完成" in line:
                col = COLOR_GREEN
            elif "fail" in line.lower() or "error" in line.lower() or "离线" in line:
                col = COLOR_RED
            elif "下发" in line:
                col = COLOR_ACCENT
            draw_text(canvas, line, (lx + 12, ly), font_size=11, color=col)

    # ---------------- 弹窗浮层：工位选择下拉 ----------------
    def _draw_ws_popup(self, app, canvas: np.ndarray, mpos: Tuple[int, int]):
        ws_list = app.workspace_mgr.list_workspaces()
        pop_w = 200
        pop_h = max(40, len(ws_list) * 28 + 8)
        pop_x = 430
        pop_y = 48

        cv2.rectangle(canvas, (pop_x, pop_y), (pop_x + pop_w, pop_y + pop_h), (24, 28, 36), -1)
        cv2.rectangle(canvas, (pop_x, pop_y), (pop_x + pop_w, pop_y + pop_h), COLOR_BLUE, 1)

        for idx, ws in enumerate(ws_list):
            iy = pop_y + 4 + idx * 28
            rect = (pop_x + 4, iy, pop_w - 8, 24)
            is_cur = (ws.workspace_id == app.current_workspace_id)
            self.draw_btn(canvas, rect, ws.name, mpos, theme_color=COLOR_BLUE, active=is_cur)

    # ---------------- 弹窗浮层：相机源选择下拉 ----------------
    def _draw_cam_popup(self, app, canvas: np.ndarray, mpos: Tuple[int, int]):
        cams = app.camera_types
        pop_w = 140
        pop_h = len(cams) * 28 + 8
        pop_x = 600
        pop_y = 48

        cv2.rectangle(canvas, (pop_x, pop_y), (pop_x + pop_w, pop_y + pop_h), (24, 28, 36), -1)
        cv2.rectangle(canvas, (pop_x, pop_y), (pop_x + pop_w, pop_y + pop_h), COLOR_BLUE, 1)

        for idx, (cid, cname) in enumerate(cams):
            iy = pop_y + 4 + idx * 28
            rect = (pop_x + 4, iy, pop_w - 8, 24)
            is_cur = (cid == app.camera_type)
            self.draw_btn(canvas, rect, cname, mpos, theme_color=COLOR_BLUE, active=is_cur)

    # ---------------- 弹窗浮层：分辨率选择下拉 ----------------
    def _draw_res_popup(self, app, canvas: np.ndarray, mpos: Tuple[int, int]):
        resolutions = app.resolutions
        pop_w = 120
        pop_h = len(resolutions) * 28 + 8
        pop_x = 740
        pop_y = 48

        cv2.rectangle(canvas, (pop_x, pop_y), (pop_x + pop_w, pop_y + pop_h), (24, 28, 36), -1)
        cv2.rectangle(canvas, (pop_x, pop_y), (pop_x + pop_w, pop_y + pop_h), COLOR_BLUE, 1)

        for idx, (rw, rh) in enumerate(resolutions):
            iy = pop_y + 4 + idx * 28
            rect = (pop_x + 4, iy, pop_w - 8, 24)
            is_cur = (rw == app.camera_w and rh == app.camera_h)
            self.draw_btn(canvas, rect, f"{rw}x{rh}", mpos, theme_color=COLOR_BLUE, active=is_cur)

    # ---------------- 弹窗浮层：速度选择下拉 ----------------
    def _draw_speed_popup(self, app, canvas: np.ndarray, mpos: Tuple[int, int]):
        speeds = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
        pop_w = 110
        pop_h = len(speeds) * 28 + 8
        pop_x = 960
        pop_y = 48

        cv2.rectangle(canvas, (pop_x, pop_y), (pop_x + pop_w, pop_y + pop_h), (24, 28, 36), -1)
        cv2.rectangle(canvas, (pop_x, pop_y), (pop_x + pop_w, pop_y + pop_h), COLOR_ACCENT, 1)

        for idx, spd in enumerate(speeds):
            iy = pop_y + 4 + idx * 28
            rect = (pop_x + 4, iy, pop_w - 8, 24)
            is_cur = abs(app.load_speed - spd) < 0.01
            self.draw_btn(canvas, rect, f"{spd:.1f}x 倍速", mpos, theme_color=COLOR_ACCENT, active=is_cur)
