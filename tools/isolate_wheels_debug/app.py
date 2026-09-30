#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Isolator WHEELS 调试视窗 (flux_isolate_wheels 分离轮 ESP32 MQTT 调试 GUI)
========================================================================
协议依据: flux_isolate_wheels/doc/通讯协议.md (v1.3, 2026-09-28)
架构重构:
  - 接入 src.control.isolate_wheels_controller.IsolateWheelsController 统一控制中枢
  - 视图逻辑委托给 tools.isolate_wheels_debug.renderer.WheelDebugRenderer
  - 布局与几何常数提取至 tools.isolate_wheels_debug.ui_layout
"""

import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.control.isolate_wheels_controller import (
    IsolateWheelsController,
    BROKER_HOST, BROKER_PORT, TOPIC_PREFIX, DEFAULT_DEVID,
    LOAD_SPEED_OPTS,
)
from src.ui.base_cv_app import BaseCvApp
from src.utils.logger import get_logger
from tools.isolate_wheels_debug.renderer import WheelDebugRenderer
from tools.isolate_wheels_debug.ui_layout import (
    LOGIC_W, LOGIC_H,
    COLOR_BG, COLOR_PANEL, COLOR_BORDER, COLOR_BORDER_HL,
    COLOR_TEXT, COLOR_SUB, COLOR_MUTED, COLOR_ACCENT, COLOR_GREEN, COLOR_AMBER,
    STATE_COLORS, STATE_TEXTS,
    BTN_CONNECT, BTN_DISCONNECT, BTN_QUIT,
    DEV_CHIP_X0, DEV_CHIP_Y, DEV_CHIP_W, DEV_CHIP_H, DEV_CHIP_STEP, DEV_CHIP_MAX,
    RACK_PANEL, TAB_MULTI, TAB_SINGLE,
    COL_W, COL_GAP, COL_X0, COL_Y0, COL_H,
    BTN_CLEAR_COUNTS, BTN_RESET_ANGLES, BTN_SEND_LOAD, BOX_LOAD_SPEED,
    BTN_DIR_FWD, BTN_DIR_REV, BTN_SEND_MOTOR,
    STATUS_CARD, LOG_CARD, BTN_CLEAR_LOG, LOG_HEADER_H, LOG_LINE_H,
    MULTI_ANGLE_OPTS, POPUP_ROW_H, SPEED_POPUP_ROW_H,
    device_chip_rect, col_rect, col_header_rect, col_count_minus, col_count_plus,
    col_angle_box, col_angle_minus, col_angle_plus,
    angle_popup_rect, speed_popup_rect,
)

log = get_logger("flux_vision.tools.isolate_wheels_debug.app")


class IsolateWheelsDebuggerApp(BaseCvApp):
    """分离轮 MQTT 调试主应用 (控制器 + 视图桥接)"""

    # 导出颜色与布局常量以保持对测试和外部的 100% 兼容
    COLOR_BG = COLOR_BG
    COLOR_PANEL = COLOR_PANEL
    COLOR_BORDER = COLOR_BORDER
    COLOR_BORDER_HL = COLOR_BORDER_HL
    COLOR_TEXT = COLOR_TEXT
    COLOR_SUB = COLOR_SUB
    COLOR_MUTED = COLOR_MUTED
    COLOR_ACCENT = COLOR_ACCENT
    COLOR_GREEN = COLOR_GREEN
    COLOR_AMBER = COLOR_AMBER

    def __init__(self, settings_file: Optional[str] = None):
        super().__init__(
            app_id="isolate_wheels_debug",
            base_w=LOGIC_W,
            base_h=LOGIC_H,
            window_name="flux_vision_3d | isolate_wheels",
            window_title="flux_vision_3d | Isolator WHEELS 8 通道调试工作台",
            settings_file=settings_file,
            enable_keyboard_zoom=False,
        )

        # 核心控制中枢
        self.controller = IsolateWheelsController()
        self.renderer = WheelDebugRenderer()

        # UI 交互状态
        self.log_scroll: Optional[int] = None
        self.counts: List[int] = [0] * 8
        self.motor_mode = "multi"
        self.motor_sel = 1
        self.motor_dir = 1
        self.motor_angle = 90.0
        self.multi_angles: List[float] = [0.0] * 8
        self.popup_col = -1
        self.load_speed: float = 1.0
        self.popup_speed: bool = False

    # ==================== 生命周期钩子 ====================
    def setup(self):
        self.connect_broker()

    def cleanup(self):
        self.disconnect_broker()

    def connect_broker(self):
        self.controller.connect()
        self.set_toast(f"正在连接 Broker {BROKER_HOST}:{BROKER_PORT} ...")

    def disconnect_broker(self):
        self.controller.disconnect()
        self.set_toast("已断开与 Broker 的连接。")

    # ==================== 核心命令发布分发 ====================
    def send_load(self) -> bool:
        ok, msg, payload = self.controller.send_load(self.counts, speed=self.load_speed)
        if ok:
            spd_tag = f" ({self.load_speed:.1f}x)"
            log.info(f"[WHEELS] {time.strftime('%H:%M:%S')} 已下发 -> load{spd_tag}: {payload}")
            self.set_toast(f"节拍已下发至 {self.controller.selected_devid}: {payload}")
        else:
            self.set_toast(msg)
        return ok

    def send_motor(self) -> bool:
        ok, msg, payload = self.controller.send_motor(self.motor_sel, self.motor_dir, self.motor_angle)
        if ok:
            log.info(f"[WHEELS] {time.strftime('%H:%M:%S')} 已下发 -> motor #{self.motor_sel}: {payload}")
            self.set_toast(f"单电机调试命令已下发至 {self.controller.selected_devid}: {payload}")
        else:
            self.set_toast(msg)
        return ok

    def send_multi(self) -> bool:
        ok, msg, payload = self.controller.send_multi(self.multi_angles)
        if ok:
            log.info(f"[WHEELS] {time.strftime('%H:%M:%S')} 已下发 -> multi 8轴: {payload}")
            self.set_toast(f"多电机调试命令已下发至 {self.controller.selected_devid}: {payload}")
        else:
            self.set_toast(msg)
        return ok

    def send_motor_cmd(self):
        if self.motor_mode == "single":
            self.send_motor()
        else:
            self.send_multi()

    # ==================== 事件调度分发 ====================
    def on_click(self, x: int, y: int):
        # 0. 弹层拦截
        if self.popup_speed:
            spx, spy, spw, sph = speed_popup_rect()
            if WheelDebugRenderer.pt_in(x, y, (spx, spy, spw, sph)):
                row = (y - spy - 4) // SPEED_POPUP_ROW_H
                if 0 <= row < len(LOAD_SPEED_OPTS):
                    self.load_speed = float(LOAD_SPEED_OPTS[row])
                    self.set_toast(f"节拍速度倍率已切换为 {self.load_speed:.1f}x")
            self.popup_speed = False
            return

        if self.popup_col >= 0:
            px, py, pw, ph = angle_popup_rect(self.popup_col)
            if WheelDebugRenderer.pt_in(x, y, (px, py, pw, ph)):
                row = (y - py - 4) // POPUP_ROW_H
                if 0 <= row < len(MULTI_ANGLE_OPTS):
                    chosen = float(MULTI_ANGLE_OPTS[row])
                    idx = self.popup_col
                    if self.motor_mode == "multi":
                        self.multi_angles[idx] = chosen
                    else:
                        self.motor_angle = abs(chosen) if chosen != 0 else 90.0
                self.popup_col = -1
                return
            self.popup_col = -1
            return

        # 1. 顶栏按钮
        if WheelDebugRenderer.pt_in(x, y, BTN_CONNECT):
            self.connect_broker()
            return
        if WheelDebugRenderer.pt_in(x, y, BTN_DISCONNECT):
            self.disconnect_broker()
            return
        if WheelDebugRenderer.pt_in(x, y, BTN_QUIT):
            self._running = False
            return

        # 2. 在线设备切换
        if DEV_CHIP_Y <= y <= DEV_CHIP_Y + DEV_CHIP_H:
            dev_ids = list(self.controller.devices.keys()) or [self.controller.selected_devid]
            for i in range(min(len(dev_ids), DEV_CHIP_MAX)):
                if WheelDebugRenderer.pt_in(x, y, device_chip_rect(i)):
                    self.controller.selected_devid = dev_ids[i]
                    return

        # 3. 模式切换 Tab
        if WheelDebugRenderer.pt_in(x, y, TAB_MULTI):
            self.motor_mode = "multi"
            return
        if WheelDebugRenderer.pt_in(x, y, TAB_SINGLE):
            self.motor_mode = "single"
            return

        # 4. 8 通道交互
        for col in range(8):
            idx = col
            if WheelDebugRenderer.pt_in(x, y, col_header_rect(col)):
                self.motor_sel = idx + 1
                return
            if WheelDebugRenderer.pt_in(x, y, col_count_minus(col)):
                self.counts[idx] = max(0, self.counts[idx] - 1)
                return
            if WheelDebugRenderer.pt_in(x, y, col_count_plus(col)):
                self.counts[idx] = min(9, self.counts[idx] + 1)
                return
            if WheelDebugRenderer.pt_in(x, y, col_angle_box(col)):
                self.popup_col = col
                self.popup_speed = False
                if self.motor_mode == "single":
                    self.motor_sel = idx + 1
                return
            if WheelDebugRenderer.pt_in(x, y, col_angle_minus(col)):
                if self.motor_mode == "multi":
                    self.multi_angles[idx] = max(-360.0, round(self.multi_angles[idx] - 22.5, 1))
                else:
                    self.motor_sel = idx + 1
                    self.motor_angle = max(22.5, round(self.motor_angle - 22.5, 1))
                return
            if WheelDebugRenderer.pt_in(x, y, col_angle_plus(col)):
                if self.motor_mode == "multi":
                    self.multi_angles[idx] = min(360.0, round(self.multi_angles[idx] + 22.5, 1))
                else:
                    self.motor_sel = idx + 1
                    self.motor_angle = min(360.0, round(self.motor_angle + 22.5, 1))
                return

        # 5. 机架底部操作条
        if WheelDebugRenderer.pt_in(x, y, BTN_CLEAR_COUNTS):
            self.counts = [0] * 8
            self.set_toast("8 托架数量已清零。")
            return
        if WheelDebugRenderer.pt_in(x, y, BTN_RESET_ANGLES):
            self.multi_angles = [0.0] * 8
            self.motor_angle = 90.0
            self.set_toast("所有电机角度已归零/重置。")
            return
        if WheelDebugRenderer.pt_in(x, y, BTN_SEND_LOAD):
            self.send_load()
            return
        if WheelDebugRenderer.pt_in(x, y, BOX_LOAD_SPEED):
            self.popup_speed = not self.popup_speed
            self.popup_col = -1
            return
        if self.motor_mode == "single":
            if WheelDebugRenderer.pt_in(x, y, BTN_DIR_FWD):
                self.motor_dir = 1
                return
            if WheelDebugRenderer.pt_in(x, y, BTN_DIR_REV):
                self.motor_dir = 0
                return
        if WheelDebugRenderer.pt_in(x, y, BTN_SEND_MOTOR):
            self.send_motor_cmd()
            return

        # 6. 日志清空按钮
        if WheelDebugRenderer.pt_in(x, y, BTN_CLEAR_LOG):
            with self.controller._lock:
                self.controller.log_lines.clear()
            self.log_scroll = None
            self.set_toast("设备实时日志已清空。")
            return

    def on_mouse_wheel(self, delta: int, flags: int):
        lx, ly = self.mouse_x, self.mouse_y
        x, y, w, h = LOG_CARD
        if not (x <= lx <= x + w and y <= ly <= y + h):
            return
        with self.controller._lock:
            total = len(self.controller.log_lines)
        max_lines = (h - 58 - 12) // LOG_LINE_H
        if total <= max_lines:
            self.log_scroll = None
            return
        rows_delta = -3 if delta > 0 else 3
        base = (total - max_lines) if self.log_scroll is None else self.log_scroll
        pos = max(0, min(total - max_lines, base + rows_delta))
        self.log_scroll = None if pos >= total - max_lines else pos

    # ==================== 渲染系统 ====================
    def render(self) -> np.ndarray:
        canvas = np.full((LOGIC_H, LOGIC_W, 3), COLOR_BG, dtype=np.uint8)
        mpos = (self.mouse_x, self.mouse_y)

        # 顶栏
        broker_desc = f"MQTT {BROKER_HOST}:{BROKER_PORT} (协议 v1.3)"
        with self.controller._lock:
            dev_ids = sorted(self.controller.devices.keys())
        self.renderer.draw_topbar(canvas, mpos, broker_desc, self.controller._connected,
                                  self.controller.selected_devid, dev_ids, self.controller.devices)

        # 8 通道机架
        st = self.controller.get_device_state(self.controller.selected_devid)
        can_send = self.controller._connected and st == "idle"
        self.renderer.draw_rack_panel(
            canvas, mpos, self.motor_mode, self.motor_sel, self.counts,
            self.multi_angles, self.motor_angle, self.load_speed, self.popup_speed,
            can_send, self.motor_dir
        )

        # 状态面板
        self.renderer.draw_status_panel(
            canvas, self.controller.selected_devid, st, self.controller.done_count,
            self.controller.last_done_cmd, self.controller.last_done_time,
            self.controller.last_publish_msg, self.controller.last_cmd_json
        )

        # 日志面板
        with self.controller._lock:
            lines = list(self.controller.log_lines)
        self.renderer.draw_log_panel(canvas, mpos, TOPIC_PREFIX, self.log_scroll, lines)

        # 弹层
        if self.popup_col >= 0:
            cur = self.multi_angles[self.popup_col] if self.motor_mode == "multi" else self.motor_angle
            self.renderer.draw_angle_popup(canvas, mpos, self.popup_col, cur)

        if self.popup_speed:
            self.renderer.draw_speed_popup(canvas, mpos, self.load_speed)

        self.draw_toast(canvas)
        return canvas


if __name__ == "__main__":
    app = IsolateWheelsDebuggerApp()
    app.run()
