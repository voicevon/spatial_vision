#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
isolate_wheels_debug 视图渲染器 (WheelDebugRenderer)
=====================================================
纯粹负责 1280 × 760 逻辑画布上的所有 OpenCV 矢量图元、控件外框、状态指示灯与富文本绘制。
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional

from src.utils.text_rendering import draw_text
from tools.isolate_wheels_debug.ui_layout import (
    LOGIC_W, LOGIC_H,
    COLOR_BG, COLOR_PANEL, COLOR_BORDER, COLOR_BORDER_HL,
    COLOR_TEXT, COLOR_SUB, COLOR_MUTED, COLOR_ACCENT, COLOR_GREEN, COLOR_AMBER,
    STATE_COLORS, STATE_TEXTS,
    BTN_CONNECT, BTN_DISCONNECT, BTN_QUIT,
    DEV_CHIP_MAX, device_chip_rect,
    RACK_PANEL, TAB_MULTI, TAB_SINGLE,
    BTN_CLEAR_COUNTS, BTN_RESET_ANGLES, BTN_SEND_LOAD, BOX_LOAD_SPEED,
    BTN_DIR_FWD, BTN_DIR_REV, BTN_SEND_MOTOR,
    STATUS_CARD, LOG_CARD, BTN_CLEAR_LOG, LOG_LINE_H,
    MULTI_ANGLE_OPTS, POPUP_ROW_H,
    LOAD_SPEED_OPTS, SPEED_POPUP_ROW_H,
    col_rect, col_header_rect, col_count_minus, col_count_plus,
    col_angle_box, col_angle_minus, col_angle_plus,
    angle_popup_rect, speed_popup_rect,
)


class WheelDebugRenderer:
    """分离轮调试台纯渲染器"""

    @staticmethod
    def pt_in(x: int, y: int, rect: Tuple[int, int, int, int]) -> bool:
        rx, ry, rw, rh = rect
        return rx <= x <= rx + rw and ry <= y <= ry + rh

    def draw_btn(
        self,
        canvas: np.ndarray,
        rect: Tuple[int, int, int, int],
        text: str,
        mpos: Tuple[int, int],
        theme_color: Optional[Tuple[int, int, int]] = None,
        enabled: bool = True,
        bold: bool = False,
    ):
        """通用交互按钮绘制"""
        x, y, w, h = rect
        hov = enabled and self.pt_in(mpos[0], mpos[1], rect)

        if not enabled:
            bg, border, col = (22, 26, 32), (32, 38, 48), COLOR_MUTED
        elif theme_color:
            bg = (int(theme_color[0] * 0.35), int(theme_color[1] * 0.35), int(theme_color[2] * 0.35)) if hov else \
                 (int(theme_color[0] * 0.2), int(theme_color[1] * 0.2), int(theme_color[2] * 0.2))
            border = theme_color if hov else (int(theme_color[0] * 0.6), int(theme_color[1] * 0.6), int(theme_color[2] * 0.6))
            col = COLOR_TEXT
        else:
            bg = (34, 42, 54) if hov else (24, 28, 36)
            border = COLOR_ACCENT if hov else COLOR_BORDER
            col = COLOR_TEXT if hov else COLOR_SUB

        cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), border, 1)

        font_size = 11
        est_w = int(font_size * 0.9 * len(text))
        tx = x + max(4, (w - est_w) // 2)
        ty = y + max(4, (h - 16) // 2)
        draw_text(canvas, text, (tx, ty), font_size=font_size, color=col, bold=bold or (enabled and hov))

    def draw_topbar(
        self,
        canvas: np.ndarray,
        mpos: Tuple[int, int],
        broker_desc: str,
        connected: bool,
        selected_devid: str,
        dev_ids: List[str],
        devices: dict,
    ):
        """顶栏: 标题、Broker状态指示、在线设备、系统操作按钮"""
        draw_text(canvas, "Isolator WHEELS 调试", (24, 18), font_size=17, color=COLOR_TEXT, bold=True)
        draw_text(canvas, broker_desc, (260, 22), font_size=12, color=COLOR_SUB)

        draw_text(canvas, "设备:", (490, 22), font_size=12, color=COLOR_SUB)
        display_devs = dev_ids if dev_ids else [selected_devid]
        for i, devid in enumerate(display_devs[:DEV_CHIP_MAX]):
            rx, ry, rw, rh = device_chip_rect(i)
            sel = (devid == selected_devid)
            st = devices.get(devid, "")
            hov = self.pt_in(mpos[0], mpos[1], (rx, ry, rw, rh))
            bg = (26, 48, 56) if sel else ((34, 42, 54) if hov else COLOR_PANEL)
            border = COLOR_ACCENT if sel else COLOR_BORDER
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), bg, -1)
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), border, 2 if sel else 1)
            st_col = STATE_COLORS.get(st, COLOR_MUTED)
            cv2.circle(canvas, (rx + 12, ry + rh // 2), 4, st_col, -1)
            draw_text(canvas, devid, (rx + 22, ry + 7), font_size=12,
                      color=COLOR_TEXT if sel else COLOR_SUB, bold=sel)

        dot_col = COLOR_GREEN if connected else (80, 80, 255)
        dot_text = "MQTT 在线" if connected else "MQTT 离线"
        cv2.circle(canvas, (948, 29), 5, dot_col, -1)
        draw_text(canvas, dot_text, (960, 22), font_size=12, color=dot_col, bold=True)

        self.draw_btn(canvas, BTN_CONNECT, "连接", mpos, theme_color=(0, 160, 200), enabled=not connected)
        self.draw_btn(canvas, BTN_DISCONNECT, "断开", mpos, theme_color=(140, 90, 60), enabled=connected)
        self.draw_btn(canvas, BTN_QUIT, "退出", mpos, theme_color=(120, 60, 60))

    def draw_rack_panel(
        self,
        canvas: np.ndarray,
        mpos: Tuple[int, int],
        motor_mode: str,
        motor_sel: int,
        counts: List[int],
        multi_angles: List[float],
        motor_angle: float,
        load_speed: float,
        popup_speed: bool,
        can_send: bool,
        motor_dir: int,
    ):
        """8 通道机架卡片与通道条"""
        x, y, w, h = RACK_PANEL
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_PANEL, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_BORDER, 1)

        draw_text(canvas, "8 通道集成控制台 (物理实物排布: 左 1 号轮 -> 右 8 号轮 -> 出口)",
                  (x + 16, y + 12), font_size=13, color=COLOR_TEXT, bold=True)

        # Tab 模式
        for tab_rect, mode_name, label in ((TAB_MULTI, "multi", "多电机 (multi)"),
                                           (TAB_SINGLE, "single", "单电机 (motor)")):
            sel = (motor_mode == mode_name)
            tx, ty, tw, th = tab_rect
            hov = self.pt_in(mpos[0], mpos[1], tab_rect)
            bg = (24, 46, 54) if sel else ((32, 38, 48) if hov else (20, 24, 30))
            border = COLOR_ACCENT if sel else COLOR_BORDER
            cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), bg, -1)
            cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), border, 2 if sel else 1)
            est = 11 * len(label)
            draw_text(canvas, label, (tx + (tw - est) // 2, ty + 6), font_size=11,
                      color=COLOR_TEXT if sel else COLOR_SUB, bold=sel)

        # 8 通道列
        for col in range(8):
            idx = col
            cx, cy, cw, ch = col_rect(col)
            is_single_sel = (motor_mode == "single" and motor_sel == idx + 1)

            bg_col = (28, 34, 44) if is_single_sel else (20, 24, 31)
            border_col = COLOR_ACCENT if is_single_sel else COLOR_BORDER
            cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), bg_col, -1)
            cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), border_col, 2 if is_single_sel else 1)

            # Header
            hx, hy, hw, hh = col_header_rect(col)
            h_bg = (20, 48, 56) if is_single_sel else (26, 32, 42)
            cv2.rectangle(canvas, (hx, hy), (hx + hw, hy + hh), h_bg, -1)
            cv2.line(canvas, (hx, hy + hh), (hx + hw, hy + hh), border_col, 1)
            title = f"{idx + 1} 号轮" + (" (选)" if is_single_sel else "")
            est_t = 13 * len(title)
            draw_text(canvas, title, (hx + max(4, (hw - est_t) // 2), hy + 7), font_size=12,
                      color=COLOR_ACCENT if is_single_sel else COLOR_TEXT, bold=True)

            # 数量
            draw_text(canvas, "托架数量", (cx + 12, cy + 36), font_size=10, color=COLOR_MUTED)
            val = counts[idx]
            vx, vy, vw, vh = (cx + 10, cy + 50, cw - 20, 28)
            if val == 0:
                v_bg, v_border, v_color = (16, 20, 26), COLOR_BORDER, COLOR_MUTED
            elif val == 1:
                v_bg, v_border, v_color = (20, 44, 32), (0, 180, 100), COLOR_GREEN
            else:
                v_bg, v_border, v_color = (44, 36, 20), (0, 150, 220), COLOR_AMBER
            cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), v_bg, -1)
            cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), v_border, 1)
            draw_text(canvas, str(val), (vx + vw // 2 - 5, vy + 6), font_size=14, color=v_color, bold=True)

            self.draw_btn(canvas, col_count_minus(col), "-", mpos, enabled=val > 0)
            self.draw_btn(canvas, col_count_plus(col), "+", mpos, enabled=val < 9)

            cv2.line(canvas, (cx + 8, cy + 118), (cx + cw - 8, cy + 118), (36, 42, 54), 1)

            # 角度
            draw_text(canvas, "电机角度", (cx + 12, cy + 126), font_size=10, color=COLOR_MUTED)
            cur_ang = multi_angles[idx] if motor_mode == "multi" else (motor_angle if is_single_sel else 0.0)
            ax, ay, aw, ah = col_angle_box(col)
            ang_hov = self.pt_in(mpos[0], mpos[1], (ax, ay, aw, ah))
            ang_bg = (30, 42, 50) if ang_hov else (18, 22, 28)
            cv2.rectangle(canvas, (ax, ay), (ax + aw, ay + ah), ang_bg, -1)
            cv2.rectangle(canvas, (ax, ay), (ax + aw, ay + ah), COLOR_ACCENT if ang_hov else COLOR_BORDER, 1)

            r_ang = round(float(cur_ang), 1)
            ang_str = f"{int(r_ang) if r_ang == int(r_ang) else r_ang}°"
            ang_col = COLOR_ACCENT if cur_ang != 0 else COLOR_MUTED
            draw_text(canvas, ang_str, (ax + 16, ay + 7), font_size=12, color=ang_col, bold=cur_ang != 0)
            draw_text(canvas, "v", (ax + aw - 16, ay + 8), font_size=10, color=COLOR_SUB)

            self.draw_btn(canvas, col_angle_minus(col), "-", mpos)
            self.draw_btn(canvas, col_angle_plus(col), "+", mpos)

        # 底部工具栏
        self.draw_btn(canvas, BTN_CLEAR_COUNTS, "全部清零 (数量)", mpos)
        self.draw_btn(canvas, BTN_RESET_ANGLES, "全部归零 (角度)", mpos)
        self.draw_btn(canvas, BTN_SEND_LOAD, "发送 load 节拍", mpos,
                      theme_color=(0, 180, 120) if can_send else None, enabled=can_send, bold=True)

        # 速度倍率下拉框
        sx, sy, sw, sh = BOX_LOAD_SPEED
        spd_hov = self.pt_in(mpos[0], mpos[1], BOX_LOAD_SPEED)
        spd_bg = (32, 46, 56) if spd_hov or popup_speed else (18, 24, 32)
        spd_border = COLOR_ACCENT if spd_hov or popup_speed else COLOR_BORDER
        cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), spd_bg, -1)
        cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), spd_border, 2 if popup_speed else 1)
        spd_str = f"速度: {load_speed:.1f}x" if load_speed != 1.0 else "速度: 1.0 (标速)"
        draw_text(canvas, spd_str, (sx + 10, sy + 11), font_size=11, color=COLOR_TEXT, bold=True)
        draw_text(canvas, "v", (sx + sw - 18, sy + 12), font_size=10,
                  color=COLOR_ACCENT if popup_speed else COLOR_SUB)

        if motor_mode == "single":
            for b_rect, label, d in ((BTN_DIR_FWD, "正转", 1), (BTN_DIR_REV, "反转", 0)):
                sel_dir = (motor_dir == d)
                self.draw_btn(canvas, b_rect, label, mpos,
                              theme_color=COLOR_GREEN if sel_dir else None, bold=sel_dir)
            motor_btn_text = f"发送 motor (#{motor_sel}轮)"
        else:
            motor_btn_text = "发送 multi (8轴多电机)"

        self.draw_btn(canvas, BTN_SEND_MOTOR, motor_btn_text, mpos,
                      theme_color=(0, 160, 220) if can_send else None, enabled=can_send, bold=True)

    def draw_status_panel(
        self,
        canvas: np.ndarray,
        devid: str,
        state: str,
        done_count: int,
        last_done_cmd: str,
        last_done_time: str,
        last_publish_msg: str,
        last_cmd_json: str,
    ):
        """状态与命令监控面板"""
        x, y, w, h = STATUS_CARD
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_PANEL, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_BORDER, 1)
        cv2.rectangle(canvas, (x, y), (x + 4, y + h), COLOR_ACCENT, -1)

        draw_text(canvas, f"设备监控与通讯 [{devid}]", (x + 16, y + 12), font_size=13, color=COLOR_TEXT, bold=True)

        st_col = STATE_COLORS.get(state, COLOR_MUTED)
        st_text = STATE_TEXTS.get(state, "未上线 (等待保留消息)")
        cv2.circle(canvas, (x + 24, y + 48), 6, st_col, -1)
        draw_text(canvas, st_text, (x + 38, y + 40), font_size=17, color=st_col, bold=True)

        stat_line = f"节拍完成: {done_count} 次  |  最近 done: {last_done_cmd or '-'} @ {last_done_time or '--:--:--'}"
        draw_text(canvas, stat_line, (x + 16, y + 74), font_size=12, color=COLOR_SUB)
        draw_text(canvas, f"最近下发: {last_publish_msg or '(尚未下发)'}", (x + 16, y + 96), font_size=11, color=COLOR_MUTED)

        # JSON 预览
        jx, jy, jw, jh = (x + 16, y + 122, w - 32, 178)
        cv2.rectangle(canvas, (jx, jy), (jx + jw, jy + jh), (14, 17, 22), -1)
        cv2.rectangle(canvas, (jx, jy), (jx + jw, jy + jh), (30, 36, 46), 1)
        draw_text(canvas, "命令 JSON 预览 (最近下发):", (jx + 10, jy + 8), font_size=10, color=COLOR_MUTED)
        preview = last_cmd_json or '(尚未下发任何命令)'
        for row_i, line in enumerate(preview.split('\n')[:6]):
            draw_text(canvas, line[:68], (jx + 10, jy + 28 + row_i * 20), font_size=11,
                      color=COLOR_TEXT if last_cmd_json else COLOR_MUTED)

    def draw_log_panel(
        self,
        canvas: np.ndarray,
        mpos: Tuple[int, int],
        topic_prefix: str,
        log_scroll: Optional[int],
        lines: List[str],
    ):
        """实时日志视窗"""
        x, y, w, h = LOG_CARD
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_PANEL, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_BORDER, 1)

        following = log_scroll is None
        title = f"设备实时日志 ({topic_prefix}/+/log)" + ("" if following else "  ↑ 回看模式 (滚至底部恢复跟随)")
        draw_text(canvas, title, (x + 16, y + 14), font_size=13,
                  color=COLOR_TEXT if following else COLOR_AMBER, bold=True)
        self.draw_btn(canvas, BTN_CLEAR_LOG, "清空日志", mpos)

        vx, vy, vw, vh = (x + 16, y + 46, w - 32, h - 58)
        cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), (14, 16, 20), -1)
        cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), (34, 40, 52), 1)

        max_lines = (vh - 12) // LOG_LINE_H
        total = len(lines)
        if following or total <= max_lines:
            visible = lines[-max_lines:]
        else:
            start = max(0, min(log_scroll, total - max_lines))
            visible = lines[start:start + max_lines]

        for i, ln in enumerate(visible):
            ly = vy + 8 + i * LOG_LINE_H
            col = COLOR_AMBER if "SYS" in ln else ((100, 100, 255) if "err" in ln.lower() else (195, 205, 218))
            draw_text(canvas, ln[:78], (vx + 10, ly), font_size=11, color=col)

        if total > max_lines:
            sb_x = vx + vw - 6
            sb_h = vh - 8
            thumb_h = max(16, int(sb_h * max_lines / total))
            scroll_top = 0 if following else max(0, min(log_scroll, total - max_lines))
            thumb_y = vy + 4 + (sb_h - thumb_h if following else int((scroll_top / max(1, total - max_lines)) * (sb_h - thumb_h)))
            cv2.rectangle(canvas, (sb_x, vy + 4), (sb_x + 4, vy + 4 + sb_h), (24, 28, 36), -1)
            thumb_col = COLOR_ACCENT if not following else (60, 70, 85)
            cv2.rectangle(canvas, (sb_x, thumb_y), (sb_x + 4, thumb_y + thumb_h), thumb_col, -1)

    def draw_angle_popup(
        self,
        canvas: np.ndarray,
        mpos: Tuple[int, int],
        popup_col: int,
        cur_angle: float,
    ):
        """角度选择弹窗"""
        px, py, pw, ph = angle_popup_rect(popup_col)
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (28, 34, 44), -1)
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), COLOR_BORDER_HL, 2)

        for i, opt in enumerate(MULTI_ANGLE_OPTS):
            oy = py + 4 + i * POPUP_ROW_H
            hov = (px <= mpos[0] <= px + pw and oy <= mpos[1] < oy + POPUP_ROW_H)
            if hov:
                cv2.rectangle(canvas, (px + 2, oy), (px + pw - 2, oy + POPUP_ROW_H), (42, 52, 66), -1)
            r = round(float(opt), 1)
            opt_str = f"{int(r) if r == int(r) else r}°"
            label = "0° 不动作" if opt == 0 else opt_str
            is_cur = (cur_angle == opt)
            draw_text(canvas, label, (px + 10, oy + 5), font_size=11,
                      color=COLOR_GREEN if is_cur else (COLOR_TEXT if hov else COLOR_SUB), bold=is_cur)

    def draw_speed_popup(
        self,
        canvas: np.ndarray,
        mpos: Tuple[int, int],
        cur_speed: float,
    ):
        """速度倍率选择弹窗"""
        px, py, pw, ph = speed_popup_rect()
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (28, 34, 44), -1)
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), COLOR_BORDER_HL, 2)

        for i, opt in enumerate(LOAD_SPEED_OPTS):
            oy = py + 4 + i * SPEED_POPUP_ROW_H
            hov = (px <= mpos[0] <= px + pw and oy <= mpos[1] < oy + SPEED_POPUP_ROW_H)
            if hov:
                cv2.rectangle(canvas, (px + 2, oy), (px + pw - 2, oy + SPEED_POPUP_ROW_H), (42, 52, 66), -1)
            label = f"{opt:.1f}x (标准速度)" if opt == 1.0 else f"{opt:.1f}x"
            is_cur = (abs(cur_speed - opt) < 1e-3)
            draw_text(canvas, label, (px + 10, oy + 6), font_size=11,
                      color=COLOR_GREEN if is_cur else (COLOR_TEXT if hov else COLOR_SUB), bold=is_cur)
