# -*- coding: utf-8 -*-
"""
SCARA 机械臂调试终端 — 纯渲染层 (ScaraDebugRenderer)
====================================================
按 flux_vision_3d 深色工业风在 1280x800 逻辑画布上绘制三列布局：
  左列: 串口连接 / 实时状态 / 限位诊断
  中列: 步长档位 / 笛卡尔与关节点动 / Z 轴快捷 / 夹爪舵机
  右列: 原点与维护 / 直达坐标 / 工位跳转 / 搬运宏 / G-code 透传
底部: 滚动日志区
无任何业务逻辑，按钮命中区域由 buttons() 统一给出。
"""

from typing import List, Tuple

import cv2
import numpy as np

from src.ui.gui_theme import GuiTheme
from src.utils.text_rendering import draw_text

# 逻辑画布尺寸
LOGIC_W = 1280
LOGIC_H = 800

# 调色板: 统一取自 GuiTheme 主题单源
COL_BG = GuiTheme.BG
COL_PANEL = GuiTheme.CARD_BG
COL_PANEL_HOVER = GuiTheme.CARD_HOVER
COL_BORDER = GuiTheme.BORDER
COL_BORDER_HOVER = GuiTheme.BORDER_HOVER
COL_ACCENT = GuiTheme.ACCENT
COL_TEXT = GuiTheme.TEXT
COL_SUB = GuiTheme.TEXT_SUB
COL_MUTED = GuiTheme.TEXT_MUTED
COL_BTN = GuiTheme.BTN
COL_BTN_BORDER = GuiTheme.BTN_BORDER
COL_BTN_HOVER = GuiTheme.BTN_HOVER
COL_BTN_DISABLED_BG = GuiTheme.BTN_DISABLED_BG
COL_BTN_DISABLED_BORDER = GuiTheme.BTN_DISABLED_BORDER
COL_BTN_TEXT = GuiTheme.BTN_TEXT
COL_BTN_TEXT_HOVER = GuiTheme.BTN_TEXT_HOVER
COL_TEXT_DISABLED = GuiTheme.TEXT_DISABLED
COL_OK = GuiTheme.OK
COL_WARN = GuiTheme.WARN
COL_ERR = GuiTheme.ERR
COL_GOLD = GuiTheme.GOLD
COL_WHITE = GuiTheme.WHITE

Btn = Tuple[str, int, int, int, int]  # (btn_id, x, y, w, h)


class ScaraDebugRenderer:
    """SCARA 调试终端纯渲染器：布局绘制与按钮命中区域定义"""

    def __init__(self):
        self.mouse_x = -1
        self.mouse_y = -1
        self._buttons: List[Btn] = []

    # ------------------------------------------------------------------
    # 命中区域
    # ------------------------------------------------------------------
    @property
    def buttons(self) -> List[Btn]:
        return self._buttons

    def hit_test(self, x: int, y: int) -> str:
        """返回命中的按钮 id，未命中返回空串 (后登记的浮层按钮优先)"""
        for bid, bx, by, bw, bh in reversed(self._buttons):
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return bid
        return ""

    # ------------------------------------------------------------------
    # 主渲染入口
    # ------------------------------------------------------------------
    def render(self, app) -> np.ndarray:
        canvas = np.full((LOGIC_H, LOGIC_W, 3), COL_BG, dtype=np.uint8)
        self._buttons = []

        self._render_top_bar(canvas, app)
        self._render_left_column(canvas, app)
        self._render_mid_column(canvas, app)
        self._render_right_column(canvas, app)
        self._render_log_area(canvas, app)
        self._render_footer(canvas, app)
        self._render_dropdowns(canvas, app)
        return canvas

    # ------------------------------------------------------------------
    # 基础绘制元件
    # ------------------------------------------------------------------
    def _panel(self, canvas, x, y, w, h, title: str, accent=COL_ACCENT) -> int:
        """绘制区块面板与标题，返回内容起始 y"""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COL_PANEL, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COL_BORDER, 1)
        cv2.rectangle(canvas, (x, y), (x + 4, y + h), accent, -1)
        draw_text(canvas, title, (x + 12, y + 7), font_size=13, color=(192, 206, 222), bold=True)
        return y + 32

    def _button(self, canvas, bid: str, label: str, x, y, w, h,
                enabled: bool = True, accent=None, hoverable: bool = True) -> None:
        """绘制单个按钮并登记命中区
        样式全部取自 GuiTheme 单源 (BTN_* / BORDER_HOVER / BTN_BEHAVIOR):
          - hover 一律背景提亮 (BTN_HOVER) + 统一悬停描边 (BORDER_HOVER) + 悬停文字 (BTN_TEXT_HOVER)
          - accent 仅用于文字语义着色 (绿=确认/橙=警示/红=退出), 不影响边框
          - 禁用态用 BTN_DISABLED_* (保持按钮外观), hover 仍有背景提亮反馈
        """
        self._buttons.append((bid, x, y, w, h))
        hovered = hoverable and (x <= self.mouse_x <= x + w and y <= self.mouse_y <= y + h)
        if not enabled:
            bg, border = COL_BTN_DISABLED_BG, COL_BTN_DISABLED_BORDER
        else:
            bg = COL_BTN_HOVER if hovered else COL_BTN
            border = COL_BORDER_HOVER if hovered else COL_BTN_BORDER
        cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), border, 1)
        if not enabled:
            tcol, bold = COL_TEXT_DISABLED, False
        elif hovered:
            tcol, bold = COL_BTN_TEXT_HOVER, GuiTheme.BTN_BEHAVIOR["HOVER_BOLD"]
        elif accent:
            tcol, bold = accent, False
        else:
            tcol, bold = COL_BTN_TEXT, False
        # 文本居中 (字号倍数同样取自主题 BTN_BEHAVIOR)
        font_size = int(13 * GuiTheme.BTN_BEHAVIOR["HOVER_SCALE"]) if hovered else 13
        draw_text(canvas, label, (x + 8, y + (h - 16) // 2 + 2), font_size=font_size, color=tcol,
                  bold=bold)

    # ------------------------------------------------------------------
    # 顶栏: 标题 / 串口下拉框 + 连接切换 / M84 / G28 / G92 / 退出
    # ------------------------------------------------------------------
    def _render_top_bar(self, canvas, app):
        cv2.rectangle(canvas, (0, 0), (LOGIC_W, 46), (17, 20, 26), -1)
        cv2.line(canvas, (0, 46), (LOGIC_W, 46), COL_BORDER, 1)
        cv2.circle(canvas, (24, 23), 5, COL_ACCENT, -1)
        draw_text(canvas, "SCARA 调试", (40, 8),
                  font_size=16, color=COL_ACCENT, bold=True)
        draw_text(canvas, "MKS Base V1.6 / Marlin 2.0", (40, 27), font_size=11, color=COL_SUB)

        conn = app.robot.is_connected()

        # 串口下拉框 (选端口 / 刷新端口)
        dd_label = f"串口: {app.current_port} ✓" if conn else \
            f"串口: {app.selected_port or '未连接'} ▼"
        self._button(canvas, "dd_open:serial", dd_label[:26], 496, 8, 190, 30,
                     accent=COL_OK if conn else COL_WARN)

        # 连接/断开切换按钮三态: 连接 -> 正在连接... -> 已连接 (再按断开)
        if app.connecting:
            self._button(canvas, "conn_toggle", "正在连接...", 694, 8, 92, 30, accent=COL_WARN)
        else:
            self._button(canvas, "conn_toggle", "已连接" if conn else "连接", 694, 8, 92, 30,
                         accent=COL_OK)

        # M84 释放电机
        self._button(canvas, "m84", "[M84 释放]", 794, 8, 100, 30,
                     enabled=conn, accent=COL_WARN)

        # G28 回零 / G92 设零
        self._button(canvas, "home", "[G28 回零]", 902, 8, 100, 30,
                     enabled=conn, accent=COL_ACCENT)
        self._button(canvas, "g92", "[G92 机械零点]", 1010, 8, 92, 30, enabled=conn)

        # 退出按钮
        self._button(canvas, "quit", "[X] 退出", 1110, 8, 158, 30, accent=COL_ERR)

    # ------------------------------------------------------------------
    # 左列: 实时状态 (含自动刷新) / 限位诊断
    # ------------------------------------------------------------------
    def _render_left_column(self, canvas, app):
        x, w = 12, 396
        # 1. 实时状态
        cy = self._panel(canvas, x, 58, w, 230, "实时状态 (M114)", accent=COL_GOLD)
        p, a = app.robot.current_pose, app.robot.current_angles
        # 按照用户要求: X, Y, Z, R 单行展示，一位小数，逗号空格隔开，角度用小圈 °
        coord_str = f"X: {p.x:.1f}, Y: {p.y:.1f}, Z: {p.z:.1f}, R: {p.r:.1f}°"
        draw_text(canvas, coord_str, (x + 16, cy + 4), font_size=15, color=COL_TEXT, bold=True)
        draw_text(canvas, f"大臂 θ: {a.theta:.1f}°,  小臂 ψ: {a.psi:.1f}°", (x + 16, cy + 34), font_size=14, color=COL_SUB)
        # 自动刷新 checkbox
        cbx, cby = x + 16, cy + 72
        checked = app.auto_refresh
        cb_hover = (cbx <= self.mouse_x <= cbx + 200 and cby <= self.mouse_y <= cby + 26)
        cv2.rectangle(canvas, (cbx, cby), (cbx + 22, cby + 22),
                      COL_PANEL_HOVER if cb_hover else COL_BTN, -1)
        cv2.rectangle(canvas, (cbx, cby), (cbx + 22, cby + 22),
                      COL_ACCENT if checked else COL_BTN_BORDER, 2 if checked else 1)
        if checked:
            cv2.line(canvas, (cbx + 5, cby + 12), (cbx + 9, cby + 16), COL_ACCENT, 2)
            cv2.line(canvas, (cbx + 9, cby + 16), (cbx + 17, cby + 5), COL_ACCENT, 2)
        draw_text(canvas, "自动刷新坐标 (0.3s)", (cbx + 32, cby + 3),
                  font_size=13, color=COL_TEXT if checked else COL_SUB, bold=checked)
        self._buttons.append(("auto_refresh", cbx, cby, 200, 26))
        draw_text(canvas, f"步长: {app.jog.step_info}", (cbx, cby + 34),
                  font_size=12, color=COL_MUTED)
        self._button(canvas, "refresh_pos", "[手动刷新 M114]", cbx, cby + 58, 200, 28)

        # 2. 限位诊断
        cy = self._panel(canvas, x, 300, w, 250, "限位与传感器 (M119)", accent=COL_WARN)
        self._button(canvas, "m119", "[限位诊断 M119]", x + 10, cy, 180, 28)
        lines = app.limit_lines[-7:] if app.limit_lines else ["(点击上方按钮执行诊断)"]
        for i, ln in enumerate(lines):
            col = COL_OK if "open" in ln.lower() else (COL_ERR if "triggered" in ln.lower() else COL_SUB)
            draw_text(canvas, ln[:46], (x + 14, cy + 36 + i * 26), font_size=12, color=col)

    # ------------------------------------------------------------------
    # 中列: 步长 / 点动 / Z 快捷 / 夹爪
    # ------------------------------------------------------------------
    def _render_mid_column(self, canvas, app):
        x, w = 420, 238  # 中列宽度缩至原 60% (396 * 0.6 ≈ 238)
        conn = app.robot.is_connected()

        # 1. 步长档位 (标题与下拉框同一行: 左标题右下拉, 面板压扁为一行)
        py, ph = 58, 48
        cv2.rectangle(canvas, (x, py), (x + w, py + ph), COL_PANEL, -1)
        cv2.rectangle(canvas, (x, py), (x + w, py + ph), COL_BORDER, 1)
        cv2.rectangle(canvas, (x, py), (x + 4, py + ph), COL_ACCENT, -1)
        draw_text(canvas, "步长档位", (x + 12, py + 15), font_size=13,
                  color=(192, 206, 222), bold=True)
        step_label = f"{app.jog.step_linear_mm:g}mm/{app.jog.step_rot_deg:g}°"
        self._button(canvas, "dd_open:step", f"{step_label} {'▲' if app.dd_step_open else '▼'}",
                     x + 92, py + 9, w - 102, 30)

        # 2. 笛卡尔点动十字盘 (按用户手绘布局: 4 列 3 行)
        #    第一行: R- | Y+ | R+ | Z+    第二行: X- | (空) | X+ | 指定Z    第三行: (空) | Y- | (空) | Z-
        cy = self._panel(canvas, x, 118, w, 164, "笛卡尔点动十字盘 (Cartesian Jog)")
        bw, bh = 46, 38
        z_bw = 66                              # Z 列按钮略宽 (容纳"指定Z")
        c1 = x + 10                            # 第 1 列 (R- / X-)
        c2 = x + 60                            # 第 2 列 (Y±)
        c3 = x + 110                           # 第 3 列 (R+ / X+)
        c4 = x + 160                           # 第 4 列 (Z 列: Z+/指定Z/Z-)
        r1, r2, r3 = cy, cy + 42, cy + 84
        # 第一行: R- | Y+ | R+ | Z+
        self._button(canvas, "jog:e", "R-", c1, r1, bw, bh, enabled=conn)
        self._button(canvas, "jog:w", "Y+", c2, r1, bw, bh, enabled=conn, accent=COL_ACCENT)
        self._button(canvas, "jog:q", "R+", c3, r1, bw, bh, enabled=conn)
        self._button(canvas, "jog:u", "Z+", c4, r1, z_bw, bh, enabled=conn)
        # 第二行: X- | (空) | X+ | 指定 Z
        self._button(canvas, "jog:a", "X-", c1, r2, bw, bh, enabled=conn)
        # 中央留白: 坐标信息已移至左列坐标面板, 避免重复
        self._button(canvas, "jog:d", "X+", c3, r2, bw, bh, enabled=conn, accent=COL_ACCENT)
        z_label = "指定Z ▲" if app.dd_z_open else "指定Z ▼"
        self._button(canvas, "dd_open:z", z_label, c4, r2, z_bw, bh, enabled=conn)
        # 第三行: (空) | Y- | (空) | Z-
        self._button(canvas, "jog:s", "Y-", c2, r3, bw, bh, enabled=conn, accent=COL_ACCENT)
        self._button(canvas, "jog:j", "Z-", c4, r3, z_bw, bh, enabled=conn)

        # 3. 关节角点动 (大臂左侧上下排列 / 小臂右侧上下排列)
        cy = self._panel(canvas, x, 294, w, 114, "关节角独立点动 (Joint Jog)")
        # 左列: 大臂+ (上) / 大臂- (下)
        self._button(canvas, "jog:o", "大臂θ+", x + 10, cy, 106, 36, enabled=conn)
        self._button(canvas, "jog:l", "大臂θ-", x + 10, cy + 42, 106, 36, enabled=conn)
        # 右列: 小臂+ (上) / 小臂- (下)
        self._button(canvas, "jog:i", "小臂ψ+", x + 122, cy, 106, 36, enabled=conn)
        self._button(canvas, "jog:k", "小臂ψ-", x + 122, cy + 42, 106, 36, enabled=conn)

        # 4. 夹爪舵机 (按用户手绘布局: 2 行 3 列 — 第一行 全开/夹1开/夹2开, 第二行 全闭/夹1闭/夹2闭)
        cy = self._panel(canvas, x, 420, w, 108, "夹爪舵机 (Gripper Servo)")
        btn_w = 68  # 每行 3 个按钮, 宽度 (238-20-14)/3 = 68
        gap_x = 7
        col2 = x + 10 + btn_w + gap_x   # 第 2 列
        col3 = x + 10 + (btn_w + gap_x) * 2  # 第 3 列
        # 第一行: 全开 | 夹1开 | 夹2开
        self._button(canvas, "grip_open", "全开", x + 10, cy, btn_w, 30,
                     enabled=conn, accent=COL_OK)
        self._button(canvas, "grip1_open", "夹1开", col2, cy, btn_w, 30, enabled=conn)
        self._button(canvas, "grip2_open", "夹2开", col3, cy, btn_w, 30, enabled=conn)
        # 第二行: 全闭 | 夹1闭 | 夹2闭
        gy2 = cy + 36
        self._button(canvas, "grip_close", "全闭", x + 10, gy2, btn_w, 30,
                     enabled=conn, accent=COL_WARN)
        self._button(canvas, "grip1_close", "夹1闭", col2, gy2, btn_w, 30, enabled=conn)
        self._button(canvas, "grip2_close", "夹2闭", col3, gy2, btn_w, 30, enabled=conn)

    # ------------------------------------------------------------------
    # 右列: 直达 / 工位 / 宏 / 透传
    # ------------------------------------------------------------------
    def _render_right_column(self, canvas, app):
        x, w = 828, 440
        conn = app.robot.is_connected()

        # 1. 直达坐标 (G28/G92 已上移顶栏)
        cy = self._panel(canvas, x, 58, w, 66, "直达目标坐标 (G1 X Y Z R F)")
        self._button(canvas, "goto", "输入目标坐标 (如 X100 Y300 Z50 R0)...",
                     x + 10, cy, w - 20, 30, enabled=conn, accent=COL_GOLD)

        # 2. 工位跳转 (预设为硬编码默认工位, 只读跳转, 无删/存按钮)
        presets = app.presets.list_presets()
        n_show = min(len(presets), 5)
        cy = self._panel(canvas, x, 140, w, 44 + n_show * 34, "预设工位跳转 (Workstations)")
        keys = list(presets.keys())
        for i, name in enumerate(keys[:5]):
            pose = presets[name]
            label = f"{name[:12]} ({pose.x:.0f},{pose.y:.0f},{pose.z:.0f},{pose.r:.0f})"
            self._button(canvas, f"preset:{i}", label, x + 10, cy + i * 34, w - 20, 30, enabled=conn)

        # 3. 搬运宏 (预设面板高度已减 32: 76→44, 宏面板同步上移)
        macro_top = 140 + 44 + n_show * 34 + 40
        cy = self._panel(canvas, x, macro_top, w, 100, "芦笋搬运节拍宏 (Pick & Place)")
        self._button(canvas, "macro_minus", "-", x + 10, cy, 36, 32, enabled=conn)
        draw_text(canvas, f"x{app.macro_cycles}", (x + 56, cy + 6), font_size=18,
                  color=COL_GOLD, bold=True)
        self._button(canvas, "macro_plus", "+", x + 108, cy, 36, 32, enabled=conn)
        self._button(canvas, "macro_run", f"[执行 {app.macro_cycles} 次搬运循环]",
                     x + 156, cy, 272, 32, enabled=conn, accent=COL_WARN)

        # 4. G-code 透传
        cy = self._panel(canvas, x, macro_top + 112, w, 66, "原生 G-code 透传 (Raw Terminal)")
        self._button(canvas, "gcode_input", "输入 G-code 指令透传至 Marlin (如 M119 / G28)...",
                     x + 10, cy, w - 20, 30, enabled=conn)

    # ------------------------------------------------------------------
    # 下拉框浮层 (最后绘制覆盖基础层; 后登记按钮使命中优先)
    # ------------------------------------------------------------------
    def _render_dropdowns(self, canvas, app):
        if app.dd_serial_open:
            px, pw = 496, 190
            ports = app.port_list
            h = 28 + 4 + len(ports) * 24 + 8
            cv2.rectangle(canvas, (px, 44), (px + pw, 44 + h), (24, 30, 38), -1)
            cv2.rectangle(canvas, (px, 44), (px + pw, 44 + h), COL_ACCENT, 1)
            cy = 50
            conn = app.robot.is_connected()
            self._button(canvas, "dd_serial:__refresh__", "↻ 刷新端口", px + 4, cy, pw - 8, 26)
            cy += 32
            cv2.line(canvas, (px + 4, cy - 4), (px + pw - 4, cy - 4), COL_BORDER, 1)
            if not ports:
                draw_text(canvas, "(无串口设备)", (px + 12, cy), font_size=12, color=COL_MUTED)
            for p in ports[:9]:
                mark = "✓ " if (conn and p == app.current_port) else "  "
                self._button(canvas, f"dd_serial:{p}", f"{mark}{p}", px + 4, cy, pw - 8, 22)
                cy += 24

        if app.dd_step_open:
            # 浮层覆盖步长档位下拉按钮 (第 1 个面板右侧按钮: x+92=512, y=67~97), 宽 136
            px, pw = 512, 136
            py, ph = 99, 3 * 32 + 8
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (24, 30, 38), -1)
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), COL_ACCENT, 1)
            cy = py + 4
            for key, lin, rot in (("1", 1, 1), ("2", 10, 5), ("3", 50, 15)):
                active = (app.jog.step_linear_mm == float(lin))
                mark = "✓ " if active else "  "
                self._button(canvas, f"dd_step:{key}", f"{mark}[{key}] {lin}mm/{rot}°",
                             px + 4, cy, pw - 8, 28, accent=COL_ACCENT if active else None)
                cy += 32

        if app.dd_z_open:
            # 浮层从笛卡尔十字盘 c4 列 r2 的"指定Z"按钮下方展开
            # 按钮位置: c4=x+160=580, r2=cy+42=192, 宽 66 高 38 (按钮底部 = 230)
            px, pw = 580, 66  # 与按钮同宽
            n = 11
            ph = n * 24 + 8
            py = 232  # 从按钮底部 (230) 下方开始向下展开
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (24, 30, 38), -1)
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), COL_ACCENT, 1)
            cur_z = round(app.robot.current_pose.z / 10.0) * 10
            cy = py + 4
            # 按照用户要求: 排列顺序反过来, 最上面是 100, 最下面是 0
            for zv in range(100, -1, -10):
                mark = "✓ " if zv == cur_z else "  "
                self._button(canvas, f"dd_z:{zv}", f"{mark}{zv} mm", px + 4, cy, pw - 8, 22)
                cy += 24

    # ------------------------------------------------------------------
    # 日志区与底栏
    # ------------------------------------------------------------------
    def _render_log_area(self, canvas, app):
        # 扩展日志显示区域，从 y=560 开始，高 214px，可完整展示 10 行通信流水
        x, y, w, h = 12, 560, 1256, 214
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (16, 19, 24), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COL_BORDER, 1)
        draw_text(canvas, "通信日志 (G-code 实时流水)", (x + 10, y + 4), font_size=11, color=COL_MUTED, bold=True)
        lines = list(app.log_lines)[-10:]
        for i, ln in enumerate(lines):
            col = COL_SUB
            if ln.startswith("<") or "ok" in ln.lower() or "成功" in ln or "完成" in ln:
                col = (120, 200, 160)
            elif "ERR" in ln or "失败" in ln or "警告" in ln:
                col = COL_ERR
            elif ln.startswith(">>") or ln.startswith(">"):
                col = (150, 210, 255)
            elif "[宏]" in ln:
                col = COL_GOLD
            draw_text(canvas, ln[:140], (x + 12, y + 22 + i * 19), font_size=12, color=col)

    def _render_footer(self, canvas, app):
        y = LOGIC_H - 22
        cv2.line(canvas, (0, y - 2), (LOGIC_W, y - 2), COL_BORDER, 1)
        draw_text(canvas,
                  "键盘: W/S/A/D/X/Y 点动 | Q/E 旋转 | O/L 大臂 | I/K 小臂 | 1/2/3 步长 | 空格 刷新坐标 | ESC 退出",
                  (12, y + 2), font_size=12, color=COL_MUTED)
        draw_text(canvas, f"日志 {len(app.log_lines)} 条",
                  (LOGIC_W - 140, y + 2), font_size=12, color=COL_MUTED)
