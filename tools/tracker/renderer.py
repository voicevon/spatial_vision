#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robot 在线跟踪 - 渲染器:
工具栏 / 下拉浮层 / Studio 同款棱柱叠加 / XY 平面网格 / 信息面板 / Toast。
只读主控制器 RobotOnlineTracker 的状态并绘制, 不修改业务状态;
按钮命中表 (buttons) 每帧由 draw_toolbar 重建, 供主控制器鼠标分发使用。
"""

import os
import time

import cv2
import numpy as np

from tools.tracker.common import (
    COLOR_ACCENT, COLOR_BG, COLOR_BORDER, COLOR_BORDER_HOVER, COLOR_BORDER_SEL,
    COLOR_BTN_HOVER, COLOR_BTN_TEXT_HOVER, COLOR_CARD_BG, COLOR_CARD_SEL,
    COLOR_TEXT_DISABLED, COLOR_TEXT_SUB, COL_BLUE,
    COL_CYAN, COL_GRAY, COL_GREEN, COL_PANEL_BG, COL_PANEL_EDGE, COL_RED,
    COL_WHITE, COL_YELLOW, TOOLBAR_H,
    PRISM_HW_MM, PRISM_HEIGHT_MM, _tag_local_frame, fmt_point,
    ASPARAGUS_WIDTH_MM, ASPARAGUS_LENGTH_MM, ASPARAGUS_HALF_LENGTH_MM, ASPARAGUS_HALF_WIDTH_MM, fmt_pose_4d)
from src.utils.text_rendering import draw_text, measure_text
from src.utils.gui_theme import GuiTheme
from src.calibration.prism_renderer import draw_prism, COLORS_THEORY, COLORS_OBSERVED


class TrackerRenderer:
    """Robot 在线跟踪渲染器: 输入主控制器状态, 输出画布与按钮命中表"""

    def __init__(self, tracker):
        self.tr = tracker               # 主控制器状态引用 (只读)
        self.buttons = []               # [(btn_id, (x1,y1,x2,y2), payload), ...] 每帧重建
        self.mouse_pos = (-1, -1)
        self._workspace_rect = None
        self._camera_type_rect = None
        self._resolution_rect = None
        self._plane_z_rect = None
        self._port_rect = None
        self._track_group_rect = None

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
        """鼠标是否悬停在 rect 上 (与 _draw_dropdown_button 的 hover 判定一致)"""
        x1, y1, x2, y2 = rect
        mx, my = self.mouse_pos
        return x1 <= mx <= x2 and y1 <= my <= y2

    def _hover_text(self, hovered, size, bold=False):
        """hover 文字行为参数单源 (GuiTheme.BTN_BEHAVIOR): hover 时按主题加粗/放大"""
        if not hovered:
            return size, bold
        bh = GuiTheme.BTN_BEHAVIOR
        return int(size * bh["HOVER_SCALE"]), (bold or bh["HOVER_BOLD"])

    def _draw_checkbox(self, canvas, rect, label, checked, label_color=None, bold=False):
        """勾选框控件: 左侧方框 (√=勾选) + 右侧标签, 整体悬停高亮 (青边框+底色提亮)"""
        x1, y1, x2, y2 = rect
        is_hover = self._is_hover(rect)
        cv2.rectangle(canvas, (x1, y1), (x2, y2),
                      COLOR_BTN_HOVER if is_hover else COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2),
                      COLOR_BORDER_HOVER if is_hover else COLOR_BORDER, 1)
        bs = 16                                   # 方框边长
        bx1, by1 = x1 + 10, y1 + (y2 - y1 - bs) // 2
        bx2, by2 = bx1 + bs, by1 + bs
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2),
                      COLOR_ACCENT if checked else COLOR_BORDER, 1)
        if checked:
            draw_text(canvas, "√", (bx1 + 1, by1 - 2), 15, COLOR_ACCENT, True)
        col = label_color or (COL_WHITE if checked else COLOR_TEXT_SUB)
        cb_size, cb_bold = self._hover_text(is_hover, 14, bold)
        draw_text(canvas, label, (bx2 + 8, y1 + (y2 - y1 - 16) // 2 - 1), cb_size, col, cb_bold)

    # ------------------------------ 工具栏 ------------------------------
    def _draw_dropdown_button(self, canvas, rect, label, is_open):
        """扁平化下拉按钮 (借鉴 d435_viewer / studio_renderer)"""
        x1, y1, x2, y2 = rect
        mx, my = self.mouse_pos
        is_hover = (x1 <= mx <= x2 and y1 <= my <= y2)
        if is_open:
            bg_col, border_col, text_col = COLOR_BTN_HOVER, COLOR_BORDER_SEL, COL_WHITE
            arrow = "▲"
        elif is_hover:
            bg_col, border_col, text_col = COLOR_BTN_HOVER, COLOR_BORDER_HOVER, COL_WHITE
            arrow = "▼"
        else:
            bg_col, border_col, text_col = COLOR_CARD_BG, COLOR_BORDER, COLOR_TEXT_SUB
            arrow = "▼"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg_col, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 1)
        dd_size, dd_bold = self._hover_text(is_hover, 14)
        draw_text(canvas, f"{label} {arrow}", (x1 + 8, y1 + (y2 - y1) // 2 - 8),
                  dd_size, text_col, dd_bold)

    def _render_dropdown_popup(self, canvas, rect, options, active_key, btn_prefix):
        """置顶悬浮下拉列表浮层 (借鉴 d435_viewer)"""
        rx1, ry1, rx2, ry2 = rect
        item_h = 30
        pop_w = max(rx2 - rx1, 210)
        pop_x1, pop_y1 = rx1, ry2 + 2
        pop_x2, pop_y2 = pop_x1 + pop_w, pop_y1 + len(options) * item_h + 6
        overlay = canvas.copy()
        cv2.rectangle(overlay, (pop_x1, pop_y1), (pop_x2, pop_y2), (30, 34, 42), -1)
        cv2.addWeighted(overlay, 0.96, canvas, 0.04, 0, canvas)
        cv2.rectangle(canvas, (pop_x1, pop_y1), (pop_x2, pop_y2), COLOR_BORDER_SEL, 1)

        for i, (key, label) in enumerate(options):
            iy1 = pop_y1 + 3 + i * item_h
            iy2 = iy1 + item_h
            is_active = (key == active_key)
            if is_active:
                cv2.rectangle(canvas, (pop_x1 + 2, iy1), (pop_x2 - 2, iy2), COLOR_CARD_SEL, -1)
            draw_text(canvas, label, (pop_x1 + 10, iy1 + (item_h - 16) // 2 - 2), 15,
                      COLOR_ACCENT if is_active else COL_WHITE, bold=is_active)
            self.buttons.append((f"{btn_prefix}{i}", (pop_x1 + 2, iy1, pop_x2 - 2, iy2), key))

    def draw_toolbar(self, canvas):
        """顶部双排工具栏 (组间以空白分隔):
        第一排: 相机类型 ▼ | 分辨率 ▼ | 开启/关闭 ‖ 串口 ▼ | 连接机械臂 | M84+G92 | Park ... 退出 X
        第二排: [一次性建立世界坐标系] [确定世界坐标系] | [XY平面 ▼] [√显示已知Tag] | [目标 ▼] [识别目标·单次] [√连续识别] ‖ [跟踪目标·单次] [√连续跟踪]
                (第一组: 世界坐标系+显示选项+识别, 居左 | 第二组: 跟踪, 居右; 组内以细分隔线分小组, 组间大空白)
        """
        tr = self.tr
        tw = canvas.shape[1]
        self.buttons = []
        cv2.rectangle(canvas, (0, 0), (tw, TOOLBAR_H), COLOR_BG, -1)
        cv2.line(canvas, (0, TOOLBAR_H - 1), (tw, TOOLBAR_H - 1), COLOR_BORDER, 1)

        y1, y2 = 6, 38       # 第一排按钮
        u1, u2 = 44, 76      # 第二排按钮
        gap = 6
        group_gap = 220      # 组间空白分隔 (RealSense 组 | 机械臂组)

        # ============ 第一排 · 工作空间组 ============

        # 0. 工作空间下拉 (最左上角): 选择工位 → 加载其世界坐标地图 (★=已发布生产工位)
        ws_x1, ws_x2 = 8, 8 + 150
        self._draw_dropdown_button(canvas, (ws_x1, y1, ws_x2, y2), tr.workspace_label,
                                   is_open=(tr.active_dropdown == "WORKSPACE_DROPDOWN"))
        self.buttons.append(("TOGGLE_WS_DD", (ws_x1, y1, ws_x2, y2), "WORKSPACE_DROPDOWN"))
        self._workspace_rect = (ws_x1, y1, ws_x2, y2)

        # ============ 第一排 · RealSense 组 ============

        # 1. 相机类型下拉
        cam_x1 = ws_x2 + gap
        cam_x2 = cam_x1 + 150
        cam_label = dict(tr.camera.camera_options).get(tr.camera.camera_type, tr.camera.camera_type)
        self._draw_dropdown_button(canvas, (cam_x1, y1, cam_x2, y2), cam_label,
                                   is_open=(tr.active_dropdown == "CAMERA_TYPE_DROPDOWN"))
        self.buttons.append(("TOGGLE_CAM_DD", (cam_x1, y1, cam_x2, y2), "CAMERA_TYPE_DROPDOWN"))
        self._camera_type_rect = (cam_x1, y1, cam_x2, y2)

        # 2. 分辨率下拉
        res_x1 = cam_x2 + gap
        res_x2 = res_x1 + 110
        self._draw_dropdown_button(canvas, (res_x1, y1, res_x2, y2), tr.camera.resolution,
                                   is_open=(tr.active_dropdown == "RES_DROPDOWN"))
        self.buttons.append(("TOGGLE_RES_DD", (res_x1, y1, res_x2, y2), "RES_DROPDOWN"))
        self._resolution_rect = (res_x1, y1, res_x2, y2)

        # 3. 开启/关闭乒乓按钮 (悬停高亮)
        sw_x1 = res_x2 + gap
        sw_x2 = sw_x1 + 70
        sw_hover = self._is_hover((sw_x1, y1, sw_x2, y2))
        if tr.camera.pipeline_running:
            sw_bg, sw_border, sw_txt, sw_label = (55, 45, 30), (255, 160, 40), (255, 200, 80), "关闭"
        else:
            sw_bg, sw_border, sw_txt, sw_label = COLOR_CARD_BG, COLOR_BORDER, COLOR_TEXT_SUB, "开启"
        if sw_hover:   # 悬停高亮不覆盖开/关语义色 (仅底色提亮 + 主题悬停描边)
            sw_bg, sw_border = COLOR_BTN_HOVER, COLOR_BORDER_HOVER
        cv2.rectangle(canvas, (sw_x1, y1), (sw_x2, y2), sw_bg, -1)
        cv2.rectangle(canvas, (sw_x1, y1), (sw_x2, y2), sw_border, 1)
        sw_size, _ = self._hover_text(sw_hover, 15, True)
        draw_text(canvas, sw_label, (sw_x1 + 22, y1 + (y2 - y1 - 16) // 2 - 1), sw_size, sw_txt, True)
        self.buttons.append(("TOGGLE_CAMERA", (sw_x1, y1, sw_x2, y2), None))

        # ============ 第一排 · 机械臂组 (与 RealSense 组以空白分隔) ============

        # 4. 机械臂串口下拉: 枚举系统串口, 选择后经 [连接机械臂] 拨号
        pt_x1 = sw_x2 + group_gap
        pt_x2 = pt_x1 + 100
        self._draw_dropdown_button(canvas, (pt_x1, y1, pt_x2, y2), tr.robot.port or "串口",
                                   is_open=(tr.active_dropdown == "PORT_DROPDOWN"))
        self.buttons.append(("TOGGLE_PORT_DD", (pt_x1, y1, pt_x2, y2), "PORT_DROPDOWN"))
        self._port_rect = (pt_x1, y1, pt_x2, y2)

        # 5. 机械臂连接按钮 (三态: 连接机械臂 → 正在连接... → 断开机械臂; 悬停高亮)
        rb_x1 = pt_x2 + gap
        rb_x2 = rb_x1 + 110
        connected = tr.robot.is_connected
        rb_hover = self._is_hover((rb_x1, y1, rb_x2, y2))
        if tr.robot_connecting:
            rb_bg, rb_border, rb_txt, rb_label, rb_bold = \
                COLOR_CARD_BG, (255, 160, 40), (255, 200, 80), "正在连接...", True
        elif connected:
            rb_bg, rb_border, rb_txt, rb_label, rb_bold = \
                COLOR_CARD_SEL, COLOR_BORDER_HOVER if rb_hover else COLOR_BORDER_SEL, \
                COLOR_ACCENT, "断开机械臂", True
        else:
            rb_bg, rb_border, rb_txt, rb_label, rb_bold = \
                COLOR_BTN_HOVER if rb_hover else COLOR_CARD_BG, \
                COLOR_BORDER_HOVER if rb_hover else COLOR_BORDER, \
                COLOR_BTN_TEXT_HOVER if rb_hover else COLOR_TEXT_SUB, "连接机械臂", False
        cv2.rectangle(canvas, (rb_x1, y1), (rb_x2, y2), rb_bg, -1)
        cv2.rectangle(canvas, (rb_x1, y1), (rb_x2, y2), rb_border, 1)
        rb_size, rb_bold = self._hover_text(rb_hover, 14, rb_bold)
        draw_text(canvas, rb_label, (rb_x1 + 20, y1 + (y2 - y1 - 16) // 2 - 1), rb_size, rb_txt, rb_bold)
        self.buttons.append(("TOGGLE_ROBOT", (rb_x1, y1, rb_x2, y2), None))

        # 6. M84+G92 合并按钮: 先发 M84 释放电机, 再发 G92 设当前零点 (未连接置灰; 悬停高亮)
        gx1 = rb_x2 + gap
        gx2 = gx1 + 110
        g_hover = self._is_hover((gx1, y1, gx2, y2))
        g_bg = COLOR_CARD_SEL if connected else COLOR_CARD_BG
        g_border = COLOR_BORDER_SEL if connected else COLOR_BORDER
        g_txt = COLOR_ACCENT if connected else COLOR_TEXT_DISABLED
        if g_hover:
            g_bg = COLOR_BTN_HOVER
            g_border = COLOR_BORDER_HOVER
            g_txt = COLOR_BTN_TEXT_HOVER if not connected else g_txt
        cv2.rectangle(canvas, (gx1, y1), (gx2, y2), g_bg, -1)
        cv2.rectangle(canvas, (gx1, y1), (gx2, y2), g_border, 1)
        g_size, g_bold = self._hover_text(g_hover, 14, connected)
        draw_text(canvas, "M84+G92", (gx1 + 22, y1 + (y2 - y1 - 16) // 2 - 1), g_size,
                  g_txt, g_bold)
        self.buttons.append(("ROBOT_M84_G92", (gx1, y1, gx2, y2), None))

        # 7. Park 按钮: 机械臂回到放料位 (X-250 Y350 Z80, R=90°→E 轴)
        pk_x1 = gx2 + gap
        pk_x2 = pk_x1 + 80
        pk_hover = self._is_hover((pk_x1, y1, pk_x2, y2))
        pk_bg = COLOR_CARD_SEL if connected else COLOR_CARD_BG
        pk_border = COLOR_BORDER_SEL if connected else COLOR_BORDER
        pk_txt = COLOR_ACCENT if connected else COLOR_TEXT_DISABLED
        if pk_hover:
            pk_bg = COLOR_BTN_HOVER
            pk_border = COLOR_BORDER_HOVER
            pk_txt = COLOR_BTN_TEXT_HOVER if not connected else pk_txt
        cv2.rectangle(canvas, (pk_x1, y1), (pk_x2, y2), pk_bg, -1)
        cv2.rectangle(canvas, (pk_x1, y1), (pk_x2, y2), pk_border, 1)
        pk_size, pk_bold = self._hover_text(pk_hover, 14, connected)
        draw_text(canvas, "Park", (pk_x1 + 18, y1 + (y2 - y1 - 16) // 2 - 1), pk_size,
                  pk_txt, pk_bold)
        self.buttons.append(("ROBOT_PARK", (pk_x1, y1, pk_x2, y2), None))

        # 8. 退出按钮 (第一排最右, 悬停高亮)
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

        # ============ 第二排: 第一组 (世界坐标系/显示选项/识别, 居左) + 第二组 (跟踪, 居右) ============
        # 组内以细分隔线划分小组, 两大组之间以大空白分隔 (与第一排风格一致)
        s_gap, g_gap, m_gap = 6, 22, 52   # 组内 / 小组间 / 大组间距

        # ---- 第一组 · 小组1: 世界坐标系标定 ----

        # 8. [一次性建立世界坐标系] 一键单帧闭环按钮:
        #    开相机→拍一张→关相机→识别Tag→定世界系→蓝/绿棱柱
        wc_x1 = 8
        wc_x2 = wc_x1 + 182
        wc_hover = self._is_hover((wc_x1, u1, wc_x2, u2))
        if tr.recognizing:
            wc_label = tr.recog_stage or "识别中..."
            wc_bg, wc_border, wc_txt = COLOR_CARD_BG, (255, 160, 40), (255, 200, 80)
        elif tr.static_frame is not None:
            wc_label = "一次性建立世界坐标系"
            wc_bg = COLOR_BTN_HOVER if wc_hover else COLOR_CARD_SEL
            wc_border, wc_txt = COLOR_BORDER_HOVER if wc_hover else COLOR_BORDER_SEL, COLOR_ACCENT
        else:
            wc_label = "一次性建立世界坐标系"
            wc_bg = COLOR_BTN_HOVER if wc_hover else COLOR_CARD_BG
            wc_border = COLOR_BORDER_HOVER if wc_hover else COLOR_BORDER
            wc_txt = COLOR_BTN_TEXT_HOVER if wc_hover else COLOR_TEXT_SUB
        cv2.rectangle(canvas, (wc_x1, u1), (wc_x2, u2), wc_bg, -1)
        cv2.rectangle(canvas, (wc_x1, u1), (wc_x2, u2), wc_border, 1)
        wc_size, _ = self._hover_text(wc_hover, 14, True)
        draw_text(canvas, wc_label,
                  (wc_x1 + (8 if tr.recognizing else 10), u1 + (u2 - u1 - 16) // 2 - 1),
                  wc_size, wc_txt, True)
        self.buttons.append(("TRIGGER_RECOG", (wc_x1, u1, wc_x2, u2), None))

        # 9. [确定世界坐标系] 一键流程按钮 (FR-12.5, 悬停高亮)
        lk_x1 = wc_x2 + s_gap
        lk_x2 = lk_x1 + 130
        lk_hover = self._is_hover((lk_x1, u1, lk_x2, u2))
        if tr.sampling:
            lk_label, lk_border, lk_txt = (tr.sample_stage or "采样中..."), (255, 160, 40), (255, 200, 80)
        elif tr.world_locked:
            lk_label = "解除锁定"
            lk_border = COLOR_BORDER_HOVER if lk_hover else COLOR_BORDER_SEL
            lk_txt = COLOR_ACCENT
        else:
            lk_label = "确定世界坐标系"
            lk_border = COLOR_BORDER_HOVER if lk_hover else COLOR_BORDER
            lk_txt = COLOR_BTN_TEXT_HOVER if lk_hover else COLOR_TEXT_SUB
        cv2.rectangle(canvas, (lk_x1, u1), (lk_x2, u2), COLOR_CARD_SEL if tr.world_locked else COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (lk_x1, u1), (lk_x2, u2), lk_border, 1)
        lk_size, _ = self._hover_text(lk_hover, 14, True)
        draw_text(canvas, lk_label, (lk_x1 + 8, u1 + (u2 - u1 - 16) // 2 - 1), lk_size, lk_txt, True)
        self.buttons.append(("TOGGLE_LOCK", (lk_x1, u1, lk_x2, u2), None))

        # ---- 第一组 · 小组2: 显示选项 (细分隔线 + XY平面/显示已知Tag) ----
        sep1 = lk_x2 + (g_gap - s_gap) // 2
        cv2.line(canvas, (sep1, u1 + 2), (sep1, u2 - 2), COLOR_BORDER, 1)

        # 10. [XY平面▼] 下拉框: 最高项"不绘制", 其余为绘制高度
        pl_x1 = sep1 + (g_gap - s_gap) // 2 + s_gap
        pl_x2 = pl_x1 + 116
        pl_label = f"Z={tr.plane_z}" if tr.show_xy_plane_on else "XY平面"
        self._draw_dropdown_button(canvas, (pl_x1, u1, pl_x2, u2), pl_label,
                                   is_open=(tr.active_dropdown == "PLANE_DROPDOWN"))
        self.buttons.append(("TOGGLE_PLANE_DD", (pl_x1, u1, pl_x2, u2), "PLANE_DROPDOWN"))
        self._plane_z_rect = (pl_x1, u1, pl_x2, u2)

        # 11. [√显示已知Tag] 勾选框 (绿=BA理论位置 / 蓝=当帧实测位置)
        an_x1 = pl_x2 + s_gap
        an_x2 = an_x1 + 118
        self._draw_checkbox(canvas, (an_x1, u1, an_x2, u2), "显示已知Tag", tr.show_anchors_on)
        self.buttons.append(("TOGGLE_ANCHORS", (an_x1, u1, an_x2, u2), None))

        # ---- 第一组 · 小组3 (最右侧): 目标选择 + 识别 (单次 + 连续) ----
        sep2 = an_x2 + (g_gap - s_gap) // 2
        cv2.line(canvas, (sep2, u1 + 2), (sep2, u2 - 2), COLOR_BORDER, 1)

        # 12. [目标▼] 下拉框: 跟踪目标类型 (Tag 2号标靶 / 顶层芦笋)
        tg_x1 = sep2 + (g_gap - s_gap) // 2 + s_gap
        tg_x2 = tg_x1 + 138
        tg_label = dict(tr.target_options).get(tr.target_kind, tr.target_kind)
        self._draw_dropdown_button(canvas, (tg_x1, u1, tg_x2, u2), f"目标: {tg_label}",
                                   is_open=(tr.active_dropdown == "TARGET_DROPDOWN"))
        self.buttons.append(("TOGGLE_TARGET_DD", (tg_x1, u1, tg_x2, u2), "TARGET_DROPDOWN"))
        self._target_rect = (tg_x1, u1, tg_x2, u2)

        # 13. [识别目标·单次] 一键解算一帧目标世界坐标 (实时流中, 结果 Toast 显示)
        ro_x1 = tg_x2 + s_gap
        ro_x2 = ro_x1 + 110
        ro_hover = self._is_hover((ro_x1, u1, ro_x2, u2))
        cv2.rectangle(canvas, (ro_x1, u1), (ro_x2, u2),
                      COLOR_BTN_HOVER if ro_hover else COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (ro_x1, u1), (ro_x2, u2),
                      COLOR_BORDER_HOVER if ro_hover else COLOR_BORDER, 1)
        ro_size, _ = self._hover_text(ro_hover, 14, True)
        draw_text(canvas, "识别目标·单次", (ro_x1 + 8, u1 + (u2 - u1 - 16) // 2 - 1), ro_size,
                  COLOR_BTN_TEXT_HOVER if ro_hover else COLOR_TEXT_SUB, True)
        self.buttons.append(("TRIGGER_RECOG_TARGET", (ro_x1, u1, ro_x2, u2), None))

        # 14. [√连续识别] 勾选框 (勾选=逐帧解算目标世界坐标, FR-12.6)
        rc_x1 = ro_x2 + s_gap
        rc_x2 = rc_x1 + 104
        self._draw_checkbox(canvas, (rc_x1, u1, rc_x2, u2), "连续识别", tr.recog_tag2_on)
        self.buttons.append(("TOGGLE_RECOG", (rc_x1, u1, rc_x2, u2), None))

        # ============ 第二组: 跟踪目标 (大空白分隔, 内分单次/连续两个小组) ============

        # 15. [跟踪目标·单次] 单次到位: 执行一次"抬起→平移→下探"+ M114 偏差回读
        t1_x1 = rc_x2 + m_gap
        t1_x2 = t1_x1 + 110
        t1_hover = self._is_hover((t1_x1, u1, t1_x2, u2))
        busy = tr.tracking
        if busy:
            t1_bg, t1_border, t1_txt, t1_label = \
                COLOR_CARD_BG, (255, 160, 40), (255, 200, 80), "跟踪中..."
        else:
            t1_bg = COLOR_BTN_HOVER if t1_hover else COLOR_CARD_BG
            t1_border = COLOR_BORDER_HOVER if t1_hover else COLOR_BORDER
            t1_txt = COLOR_BTN_TEXT_HOVER if t1_hover else COLOR_TEXT_SUB
            t1_label = "跟踪目标·单次"
        cv2.rectangle(canvas, (t1_x1, u1), (t1_x2, u2), t1_bg, -1)
        cv2.rectangle(canvas, (t1_x1, u1), (t1_x2, u2), t1_border, 1)
        t1_size, _ = self._hover_text(t1_hover, 14, True)
        draw_text(canvas, t1_label, (t1_x1 + 8, u1 + (u2 - u1 - 16) // 2 - 1), t1_size, t1_txt, True)
        self.buttons.append(("TRIGGER_TRACK_ONCE", (t1_x1, u1, t1_x2, u2), None))

        # 16. [√连续跟踪] 勾选框 (勾选=末端自动跟随目标最新位置)
        t2_x1 = t1_x2 + s_gap
        t2_x2 = t2_x1 + 104
        self._draw_checkbox(canvas, (t2_x1, u1, t2_x2, u2), "连续跟踪", tr.track_armed)
        self.buttons.append(("TOGGLE_TRACK", (t2_x1, u1, t2_x2, u2), None))
        self._track_group_rect = (t1_x1, u1, t2_x2, u2)  # 消息面板锚点 (按钮组正下方)

        # 展开的下拉浮层
        if tr.active_dropdown == "WORKSPACE_DROPDOWN" and self._workspace_rect:
            ws_opts = tr.workspace_options or [("", "(无可用工作空间)")]
            self._render_dropdown_popup(canvas, self._workspace_rect, ws_opts,
                                        tr.workspace_id or None, "DD_WS_")
        elif tr.active_dropdown == "CAMERA_TYPE_DROPDOWN" and self._camera_type_rect:
            self._render_dropdown_popup(canvas, self._camera_type_rect,
                                        tr.camera.camera_options, tr.camera.camera_type, "DD_CAM_")
        elif tr.active_dropdown == "RES_DROPDOWN" and self._resolution_rect:
            self._render_dropdown_popup(canvas, self._resolution_rect,
                                        tr.camera.resolution_options, tr.camera.resolution, "DD_RES_")
        elif tr.active_dropdown == "PLANE_DROPDOWN" and self._plane_z_rect:
            self._render_dropdown_popup(canvas, self._plane_z_rect,
                                        tr.plane_options,
                                        tr.plane_z if tr.show_xy_plane_on else None, "DD_PLANE_")
        elif tr.active_dropdown == "TARGET_DROPDOWN" and self._target_rect:
            self._render_dropdown_popup(canvas, self._target_rect,
                                        tr.target_options, tr.target_kind, "DD_TARGET_")
        elif tr.active_dropdown == "PORT_DROPDOWN" and self._port_rect:
            port_opts = [(p, p) for p in tr.port_options] or [("", "(无可用串口, 请检查 USB)")]
            self._render_dropdown_popup(canvas, self._port_rect, port_opts,
                                        tr.robot.port if tr.robot.port in tr.port_options else None,
                                        "DD_PORT_")

        # 机械臂消息面板 (跟踪/M84/G92 按钮组正下方, 调试用)
        self._draw_track_log(canvas)

    def _draw_track_log(self, canvas):
        """机械臂消息面板: 跟踪/M84/G92 按钮组正下方, 逐条显示 指令 G-code→回读/偏差;
        半透明底板随消息条数增长, 最多 TRACK_LOG_MAX 条 (调试用)"""
        tr = self.tr
        if not tr.track_log or self._track_group_rect is None:
            return
        w_c = canvas.shape[1]
        panel_w = 352
        x2 = min(self._track_group_rect[2], w_c - 8)
        x1 = max(8, x2 - panel_w)
        y1 = TOOLBAR_H + 4
        line_h = 18
        y2 = y1 + 24 + len(tr.track_log) * line_h + 6
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COL_PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COL_PANEL_EDGE, 1)
        draw_text(canvas, "机械臂消息", (x1 + 8, y1 + 4), 14, COLOR_ACCENT, True)
        kind_col = {"info": COL_WHITE, "cmd": COL_CYAN, "ok": COL_GREEN, "err": COL_RED}
        y = y1 + 26
        for t_s, msg, kind in tr.track_log:
            draw_text(canvas, f"{t_s}  {msg}", (x1 + 8, y), 13,
                      kind_col.get(kind, COL_GRAY))
            y += line_h

    # ------------------------------ 棱柱与叠加层 ------------------------------
    def _draw_studio_prism(self, canvas, rvec, tvec, is_theory, is_target=False):
        """统一 PrismRenderer Studio 同款四棱柱: 截面 30x30mm x 生长高度 75mm,
        半透明填充 + 棱线描边 + 顶面中心点; is_theory=True 翡翠绿(BA理论) / False 科技天蓝(实测);
        目标 Tag 额外绘制底面中心点与中心生长轴 (至固定高顶面中心)。
        """
        K, dist = self.tr.pnp_solver.camera_matrix, self.tr.pnp_solver.dist_coeffs
        proj = draw_prism(canvas, K, dist, rvec, tvec,
                          half_w=PRISM_HW_MM, height=PRISM_HEIGHT_MM,
                          colors=COLORS_THEORY if is_theory else COLORS_OBSERVED,
                          alpha=0.35)
        if is_target:
            dot_c = COLORS_THEORY["dot"] if is_theory else COLORS_OBSERVED["dot"]
            pb = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), rvec, tvec, K, dist)[0]
            pb = tuple(pb.reshape(2).astype(int))
            cv2.circle(canvas, pb, 4, dot_c, -1, cv2.LINE_AA)     # 底面中心点
            cv2.line(canvas, pb, tuple(proj["top_center"]), dot_c, 1, cv2.LINE_AA)

    def _draw_asparagus_stem(self, canvas, rvec, tvec, yaw_deg=None):
        """沿 Tag 局部 Y 轴绘制 3D 拟真芦笋长棒 (宽 15mm x 长 200mm, 以Tag为中心对称延伸各 100mm):
        - +Y 轴正向为芦笋头部/笋尖 (0 ~ +100mm): 鲜绿色 (Fresh Green) + 指向微箭头
        - 中后过渡段 (-70 ~ 0mm): 由翠绿平滑过渡至乳白色
        - -Y 轴负向为芦笋尾部/根部 (-100 ~ -70mm): 纯白切口 (Root White)
        """
        if rvec is None or tvec is None:
            return
        tr = self.tr
        K = tr.pnp_solver.camera_matrix
        dist = tr.pnp_solver.dist_coeffs
        h_c, w_c = canvas.shape[:2]

        half_w = ASPARAGUS_HALF_WIDTH_MM    # 7.5 mm
        half_l = ASPARAGUS_HALF_LENGTH_MM   # 100.0 mm

        # 分段渲染: 沿 Y 轴切分为 20 个微切片，逐段透视投影并进行平滑色彩插值
        num_slices = 20
        y_coords = np.linspace(-half_l, half_l, num_slices + 1)

        COL_TIP_GREEN = (70, 225, 90)     # 鲜翠绿 (BGR)
        COL_ROOT_WHITE = (245, 245, 245)  # 根部白 (BGR)
        COL_STEM_EDGE = (40, 160, 60)     # 外轮廓边框色 (BGR)

        poly_slices = []
        for i in range(num_slices):
            y1, y2 = y_coords[i], y_coords[i + 1]
            y_mid = (y1 + y2) * 0.5

            # 计算该段的插值颜色
            if y_mid >= 0.0:
                # 头部 (0 ~ +100mm): 翠绿色
                cur_color = COL_TIP_GREEN
            elif y_mid <= -70.0:
                # 根部末端 (-100 ~ -70mm): 纯白色
                cur_color = COL_ROOT_WHITE
            else:
                # 过渡段 (-70 ~ 0mm): 绿 -> 白 线性平滑渐变
                t = float(-y_mid / 70.0)  # 0.0(绿) -> 1.0(白)
                cur_color = (
                    int(round((1.0 - t) * COL_TIP_GREEN[0] + t * COL_ROOT_WHITE[0])),
                    int(round((1.0 - t) * COL_TIP_GREEN[1] + t * COL_ROOT_WHITE[1])),
                    int(round((1.0 - t) * COL_TIP_GREEN[2] + t * COL_ROOT_WHITE[2])),
                )

            # 该微段的 4 个局部 3D 点
            pts_3d = np.array([
                [-half_w, y1, 0.0],
                [ half_w, y1, 0.0],
                [ half_w, y2, 0.0],
                [-half_w, y2, 0.0],
            ], dtype=np.float64)

            proj, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist)
            proj_2d = proj.reshape(-1, 2)
            if np.all(np.isfinite(proj_2d)):
                p_poly = proj_2d.astype(np.int32)
                poly_slices.append((p_poly, cur_color))

        # 1. 绘制各切片多边形填充
        for p_poly, col in poly_slices:
            cv2.fillPoly(canvas, [p_poly], col, cv2.LINE_AA)

        # 2. 整体外轮廓加粗边框
        outer_3d = np.array([
            [-half_w, -half_l, 0.0],
            [ half_w, -half_l, 0.0],
            [ half_w,  half_l, 0.0],
            [-half_w,  half_l, 0.0],
        ], dtype=np.float64)
        outer_proj, _ = cv2.projectPoints(outer_3d, rvec, tvec, K, dist)
        outer_2d = outer_proj.reshape(-1, 2).astype(np.int32)
        cv2.polylines(canvas, [outer_2d], True, COL_STEM_EDGE, 2, cv2.LINE_AA)

        # 3. 尾部根部切口横截面加粗白色标识 (纯白切口)
        cv2.line(canvas, tuple(outer_2d[0]), tuple(outer_2d[1]), (255, 255, 255), 3, cv2.LINE_AA)

        # 4. 头部尖端三角形箭头 (+Y 方向延伸 15mm 绿色箭头，指向笋尖朝向)
        tip_3d = np.array([
            [-half_w * 1.2, half_l, 0.0],
            [ half_w * 1.2, half_l, 0.0],
            [ 0.0,          half_l + 18.0, 0.0],
        ], dtype=np.float64)
        tip_proj, _ = cv2.projectPoints(tip_3d, rvec, tvec, K, dist)
        tip_2d = tip_proj.reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(canvas, [tip_2d], COL_TIP_GREEN, cv2.LINE_AA)
        cv2.polylines(canvas, [tip_2d], True, (30, 150, 40), 2, cv2.LINE_AA)

        # 5. 笋尖前端角度数值显示 (纯数值无汉字, 居中投影在尖端前方, 防遮挡胶囊底衬)
        if yaw_deg is not None:
            # 笋尖箭头顶点在 Y = half_l + 18mm, 角度文字锚定在尖端前方 10mm (Y = half_l + 28mm)
            text_anchor_3d = np.array([[0.0, half_l + 28.0, 0.0]], dtype=np.float64)
            anchor_proj, _ = cv2.projectPoints(text_anchor_3d, rvec, tvec, K, dist)
            proj_pt = anchor_proj.reshape(-1, 2)[0]
            if np.all(np.isfinite(proj_pt)):
                tx, ty = int(round(proj_pt[0])), int(round(proj_pt[1]))
                ang_str = f"{yaw_deg:+.1f}°"
                font_size = 15
                (tw, th), _ = measure_text(ang_str, font_size=font_size, bold=True)

                pad_x, pad_y = 6, 3
                x1 = tx - tw // 2 - pad_x
                y1 = ty - th // 2 - pad_y
                x2 = tx + tw // 2 + pad_x
                y2 = ty + th // 2 + pad_y + 1

                # 画面边界防溢出保护
                ch, cw = canvas.shape[:2]
                if x1 < 4:
                    x2 += 4 - x1
                    x1 = 4
                if x2 > cw - 4:
                    x1 -= x2 - (cw - 4)
                    x2 = cw - 4
                if y1 < 4:
                    y2 += 4 - y1
                    y1 = 4
                if y2 > ch - 4:
                    y1 -= y2 - (ch - 4)
                    y2 = ch - 4

                # 防遮挡胶囊底板 (深黑灰背景 + 鲜绿描边, 确保在任何背景/长条上均清晰可见)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (20, 24, 28), -1)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (70, 210, 100), 1, cv2.LINE_AA)

                # 绘制高亮抗锯齿角度数值 (RGB 鲜绿色)
                draw_text(canvas, ang_str, (x1 + pad_x, y1 + pad_y), font_size=font_size,
                          color=(90, 240, 120), bold=True)

    def draw_overlay(self, canvas, det):
        """叠加层: 芦笋 3D 拟真绿头白尾长条 (宽15mm x 长200mm) + 目标 Tag 绿色高亮框 + 实测棱柱"""
        tr = self.tr
        corners = det.get(tr.target_tag_id)
        if corners is None:
            return
        pts = corners.reshape((-1, 2)).astype(np.int32)

        # 优先使用主流程解算得出的位姿，或当场解算
        rvec_b, tvec_b = tr.target_rvec, tr.target_tvec
        if rvec_b is None or tvec_b is None:
            z_exp = None
            if tr.world_locked and tr.locked_rvec is not None:
                R_lock, _ = cv2.Rodrigues(tr.locked_rvec)
                z_exp = R_lock @ np.array([0.0, 0.0, 1.0])
            ok_b, rvec_b, tvec_b = tr.pnp_solver.solve_single_tag_pnp(corners, expected_z_cam=z_exp)
            if ok_b:
                tvec_b = np.asarray(tvec_b).reshape(3, 1)

        # 1. 优先绘制芦笋 3D 拟真彩色长棒 (宽 15mm x 长 200mm, 尖端前端展示防遮挡角度数据)
        if rvec_b is not None and tvec_b is not None:
            self._draw_asparagus_stem(canvas, rvec_b, tvec_b, yaw_deg=tr.measured_r)

        # 2. 完整保留原有的 Tag 2 绿色识别四边形角点框 (只留绿框, 去掉中心文字及汉字)
        cv2.polylines(canvas, [pts], True, COL_GREEN, 3, cv2.LINE_AA)

        # 3. 若开启棱柱显示，叠加 Studio 同款蓝色实测棱柱
        if (tr.recog_tag2_on or tr.show_anchors_on) and rvec_b is not None and tvec_b is not None:
            self._draw_studio_prism(canvas, rvec_b, tvec_b, False, is_target=True)

    def draw_anchor_overlay(self, canvas, det):
        """已知标靶棱柱叠加 (与 Offline Studio 同款棱柱参数, 数据源=相机实时帧):
        绿色棱柱=BA 理论位姿棱柱; 蓝色棱柱=当帧实测单靶 PnP 位姿棱柱。
        位姿: 锁定后用锁定位姿, 未锁定用当帧锚定 PnP。
        """
        tr = self.tr
        if not tr.show_anchors_on:
            return
        # 位姿来源 (世界->相机): 优先锁定位姿, 未锁定用当帧锚定 PnP
        rvec, tvec = tr.get_world_pose(det)
        R, _ = cv2.Rodrigues(rvec) if rvec is not None else (None, None)
        t_flat = np.asarray(tvec).reshape(3) if tvec is not None else None

        def _world_tag_pose(wc):
            """世界角点 -> 标靶相机系位姿 (rvec, tvec)"""
            R_t, c_w = _tag_local_frame(wc)
            return cv2.Rodrigues(R @ R_t)[0], R @ c_w + t_flat

        for tid in tr.anchor_positions:
            wc = tr.pnp_solver.get_tag_world_corners(tid)
            if wc is None or R is None:
                continue
            rvec_t, tvec_t = _world_tag_pose(wc)
            if tvec_t[2] <= 1e-6:
                continue
            # 绿色理论棱柱 (Offline Studio 同款: 30x30 截面 x 75mm 生长高)
            self._draw_studio_prism(canvas, rvec_t, tvec_t, True)
            if tid in det:
                # 蓝色实测棱柱 (单靶 PnP 位姿); 传入地图理论法向, 消除 IPPE 平面二义性 180° 翻转
                R_exp, _ = cv2.Rodrigues(rvec_t)
                ok_b, rvec_b, tvec_b = tr.pnp_solver.solve_single_tag_pnp(
                    det[tid], expected_z_cam=R_exp[:, 2])
                if ok_b:
                    self._draw_studio_prism(canvas, rvec_b, np.asarray(tvec_b).reshape(3, 1), False)
                    c = det[tid].reshape(4, 2).mean(axis=0).astype(int)
                    draw_text(canvas, str(tid), (int(c[0]) + 8, int(c[1]) - 22), 14, COL_BLUE, True)
            else:
                # 未入镜的理论 Tag: 在标靶中心标注编号
                pc = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), rvec_t, tvec_t,
                                       tr.pnp_solver.camera_matrix, tr.pnp_solver.dist_coeffs)[0]
                pc = pc.reshape(2).astype(int)
                draw_text(canvas, str(tid), (int(pc[0]) + 8, int(pc[1]) - 22), 14, COL_GREEN, True)

    def draw_recognition_overlay(self, canvas):
        """单帧识别结果叠加 (静态照片, 与 Offline Studio 同款棱柱参数):
        绿色棱柱 = 地图白名单全部 Tag 的 BA 理论位姿棱柱 (需已确定世界坐标系);
        蓝色棱柱 = 当帧识别 Tag 的单靶 PnP 位姿棱柱; 目标 Tag 附带底面/顶面中心点。
        """
        tr = self.tr
        R = t_flat = None
        if tr.world_locked and tr.locked_rvec is not None:
            R, _ = cv2.Rodrigues(tr.locked_rvec)
            t_flat = np.asarray(tr.locked_tvec, dtype=np.float64).reshape(3)
        K = tr.pnp_solver.camera_matrix

        def _world_tag_pose(wc):
            """世界角点 -> 标靶相机系位姿 (rvec, tvec)"""
            R_t, c_w = _tag_local_frame(wc)
            return cv2.Rodrigues(R @ R_t)[0], R @ c_w + t_flat

        # 1. 绿色理论棱柱: 地图白名单全部 Tag (含目标 Tag, 若在地图中)
        map_ids = sorted(set(tr.anchor_positions) |
                         ({tr.target_tag_id} if tr.theoretical is not None else set()))
        for tid in map_ids:
            wc = tr.pnp_solver.get_tag_world_corners(tid)
            if wc is None or R is None:
                continue
            rvec_t, tvec_t = _world_tag_pose(wc)
            if tvec_t[2] <= 1e-6:
                continue
            self._draw_studio_prism(canvas, rvec_t, tvec_t, True,
                                    is_target=(tid == tr.target_tag_id))
            if tid not in (tr.static_det or {}):
                # 未入镜的理论 Tag: 在标靶中心标注编号
                pc = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), rvec_t, tvec_t, K,
                                       tr.pnp_solver.dist_coeffs)[0].reshape(2).astype(int)
                draw_text(canvas, str(tid), (int(pc[0]) + 8, int(pc[1]) - 22), 14, COL_GREEN, True)

        # 2. 蓝色实测棱柱: 当帧识别到的全部 Tag (单靶 PnP)
        # 法向先验: 地图内 Tag 用理论位姿法向, 地图外 Tag (如动态目标) 退用"朝向天空"先验
        sky_z = R @ np.array([0.0, 0.0, 1.0]) if R is not None else None
        for tid, corners in (tr.static_det or {}).items():
            z_exp = None
            if R is not None:
                wc = tr.pnp_solver.get_tag_world_corners(tid)
                if wc is not None:
                    R_t, _ = _tag_local_frame(wc)
                    z_exp = R @ R_t[:, 2]
                else:
                    z_exp = sky_z
            ok_b, rvec_b, tvec_b = tr.pnp_solver.solve_single_tag_pnp(corners, expected_z_cam=z_exp)
            if ok_b:
                self._draw_studio_prism(canvas, rvec_b, np.asarray(tvec_b).reshape(3, 1), False,
                                        is_target=(tid == tr.target_tag_id))
                c = corners.reshape(4, 2).mean(axis=0).astype(int)
                draw_text(canvas, str(tid), (int(c[0]) + 8, int(c[1]) - 22), 14, COL_BLUE, True)
            else:
                pts = corners.reshape((-1, 2)).astype(np.int32)
                cv2.polylines(canvas, [pts], True, COL_BLUE, 2, cv2.LINE_AA)

    def draw_xy_plane_overlay(self, canvas, det):
        """世界 XY 平面透视网格叠加 (高度由下拉框选择, 默认 Z=0 地面):
        两组互相垂直的平行线网格 + 三轴加粗高亮 (X红/Y绿/Z蓝) + 原点标记, 直观透视世界系。
        位姿: 优先锁定位姿, 未锁定用当帧锚定 PnP。
        """
        tr = self.tr
        if not tr.show_xy_plane_on:
            return
        rvec, tvec = tr.get_world_pose(det)
        if rvec is None:
            return
        R, _ = cv2.Rodrigues(rvec)
        t_flat = np.asarray(tvec, dtype=np.float64).reshape(3)
        K = tr.pnp_solver.camera_matrix
        h_c, w_c = canvas.shape[:2]
        ext, step, z0 = tr.PLANE_EXTENT_MM, tr.PLANE_STEP_MM, float(tr.plane_z)

        def _project(p_w):
            p_cam = R @ np.asarray(p_w, dtype=np.float64).reshape(3) + t_flat
            if p_cam[2] <= 1e-6:
                return None
            uv = K @ p_cam
            u, v = int(uv[0] / uv[2]), int(uv[1] / uv[2])
            return (u, v) if (0 <= u < w_c and 0 <= v < h_c) else None

        def _seg(p0, p1, color, thick):
            """长线段沿线采样投影连线 (自动处理出画与近裁剪)"""
            prev = None
            for k in range(25):
                s = k / 24.0
                p = (p0[0] + (p1[0] - p0[0]) * s,
                     p0[1] + (p1[1] - p0[1]) * s,
                     p0[2] + (p1[2] - p0[2]) * s)
                uv = _project(p)
                if uv is not None and prev is not None:
                    cv2.line(canvas, prev, uv, color, thick, cv2.LINE_AA)
                prev = uv

        # 平行线网格 (绘制高度 z0): 平行于 X 轴 (Y=-ext..ext) 与平行于 Y 轴 (X=-ext..ext) 两组
        for i in range(-ext, ext + 1, step):
            _seg((-ext, i, z0), (ext, i, z0), COL_GRAY, 1)
            _seg((i, -ext, z0), (i, ext, z0), COL_GRAY, 1)
        # 坐标轴加粗高亮: X 红 / Y 绿 (随平面高度) / Z 蓝 (0→600mm), 明显可见
        _seg((-ext, 0, z0), (ext, 0, z0), (60, 60, 245), 3)
        _seg((0, -ext, z0), (0, ext, z0), COL_GREEN, 3)
        _seg((0, 0, 0), (0, 0, tr.PLANE_Z_MM), COL_BLUE, 4)
        # Tag 等高辅助红线: 平面高度与某锚定标靶中心 Z 重合且该靶不在原点时,
        # 平移一条红色 X 轴穿过该标靶 (如 Z=196 平面过 Tag 1); Tag 0 在原点, 主 X 轴已穿过
        for tid, c_t in tr.anchor_positions.items():
            if abs(float(c_t[2]) - z0) < 2.0 and (abs(float(c_t[0])) > 1.0 or abs(float(c_t[1])) > 1.0):
                _seg((c_t[0] - ext, c_t[1], z0), (c_t[0] + ext, c_t[1], z0), (60, 60, 245), 2)
                uv = _project((c_t[0] + ext, c_t[1], z0))
                if uv is not None:
                    draw_text(canvas, f"X (Tag {tid})", (uv[0] + 6, uv[1] - 8), 13, (60, 60, 245), True)
        # Z 轴高度刻度 (每 100mm) + 顶端箭头: 俯视相机下高度轴指向镜头呈放射状,
        # 刻度数值让"向上生长"方向一目了然, 消除透视歧义
        for hz in range(100, tr.PLANE_Z_MM, 100):
            tp = _project((0, 0, hz))
            if tp is not None:
                cv2.line(canvas, (tp[0] - 5, tp[1]), (tp[0] + 5, tp[1]), COL_BLUE, 2, cv2.LINE_AA)
                draw_text(canvas, str(hz), (tp[0] + 8, tp[1] - 6), 12, COL_BLUE, True)
        p_top = _project((0, 0, tr.PLANE_Z_MM))
        p_base = _project((0, 0, 0))
        if p_top is not None and p_base is not None:
            d = np.array(p_top, dtype=np.float64) - np.array(p_base, dtype=np.float64)
            n = float(np.linalg.norm(d))
            if n > 24:
                d /= n
                perp = np.array([-d[1], d[0]])
                tip = np.array(p_top, dtype=np.float64)
                wing = 14.0 * d
                arrow = np.array([tip, tip - wing + 6.0 * perp, tip - wing - 6.0 * perp],
                                 dtype=np.int32)
                cv2.fillPoly(canvas, [arrow], COL_BLUE)
        for label, p, col in (("X", (ext + 70, 0, z0), (60, 60, 245)),
                              ("Y", (0, ext + 70, z0), COL_GREEN),
                              ("Z", (0, 0, tr.PLANE_Z_MM + 70), COL_BLUE),
                              ("0", (0, 0, 0), COL_WHITE)):
            uv = _project(p)
            if uv is not None:
                draw_text(canvas, label, (uv[0] + 6, uv[1] - 8), 15, col, True)

    # ------------------------------ 信息面板与 Toast ------------------------------
    def draw_info_panel(self, canvas, y_off=0):
        """左上信息面板 (实测 / 理论 / 偏差 / 世界系状态 / 机械臂状态)"""
        tr = self.tr
        x1, y1 = 14, y_off + 14
        x2, y2 = 512, y_off + 296
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), COL_PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0, canvas)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COL_PANEL_EDGE, 1)

        y = y1 + 26
        draw_text(canvas, f"Robot 在线跟踪 | 目标 Tag {tr.target_tag_id}",
                  (x1 + 14, y), 19, COL_WHITE, True)
        y += 28
        draw_text(canvas, f"世界系地图: {os.path.basename(tr.map_path)}",
                  (x1 + 14, y), 15, COL_GRAY)
        y += 24
        if tr.support_ids and tr.rmse is not None:
            draw_text(canvas, f"支撑: {len(tr.support_ids)} 靶 {tr.support_ids}"
                              f" | RMSE {tr.rmse:.2f}px", (x1 + 14, y), 15, COL_GRAY)
        else:
            draw_text(canvas, "支撑: 无已知标靶入镜, 世界位姿失效",
                      (x1 + 14, y), 15, COL_RED)
        y += 26
        r_txt = f"  R: {tr.measured_r:+6.1f}°" if getattr(tr, "measured_r", None) is not None else "  R:    -- "
        draw_text(canvas, f"实测 {fmt_point(tr.measured)}{r_txt}",
                  (x1 + 14, y), 17, COL_GREEN if tr.measured is not None else COL_GRAY, True)
        y += 26
        theo_txt = f"理论 {fmt_point(tr.theoretical)}" if tr.theoretical is not None \
            else "理论      (地图中无该 Tag)"
        draw_text(canvas, theo_txt, (x1 + 14, y), 17, COL_YELLOW)
        y += 26
        if tr.measured is not None and tr.theoretical is not None:
            dev = tr.measured - tr.theoretical
            draw_text(canvas, f"偏差 {fmt_point(dev, signed=True)}",
                      (x1 + 14, y), 17, COL_CYAN)
        else:
            draw_text(canvas, "偏差          --", (x1 + 14, y), 17, COL_GRAY)
        y += 26
        if tr.sampling:
            draw_text(canvas, f"世界坐标系: {tr.sample_stage}", (x1 + 14, y), 16, COL_YELLOW, True)
        elif tr.world_locked:
            draw_text(canvas, f"世界坐标系: 已锁定 ({tr.lock_info})", (x1 + 14, y), 16, COLOR_ACCENT)
        else:
            draw_text(canvas, "世界坐标系: 未确定 [确定世界坐标系]", (x1 + 14, y), 16, COL_GRAY)
        y += 26
        if tr.robot.is_connected:
            pos_txt = f"末端 {fmt_point(tr.robot_pos)}" if tr.robot_pos is not None \
                else "末端 --"
            draw_text(canvas, f"机械臂: {tr.robot.port} | {pos_txt}",
                      (x1 + 14, y), 15, COL_GREEN)
        else:
            draw_text(canvas, "机械臂: 未连接 [C] 连接", (x1 + 14, y), 15, COL_GRAY)

    def draw_toast(self, canvas):
        """右下角浮动通知"""
        tr = self.tr
        h, w = canvas.shape[:2]
        if time.time() - tr.toast_time < 6.0:
            col = COL_RED if tr.toast_err else COL_WHITE
            tw = int(len(tr.toast) * 16 * 1.05) + 16
            tx = max(w - tw - 20, 20)
            ty = h - 40
            overlay = canvas.copy()
            cv2.rectangle(overlay, (tx - 10, ty - 8), (w - 10, ty + 24), COL_PANEL_BG, -1)
            cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
            draw_text(canvas, tr.toast, (tx, ty), 16, col)

    def make_canvas(self):
        """按当前窗口物理尺寸生成底板画布 (imshow 严格 1:1, 鼠标坐标零偏移)"""
        cw = self.tr.win_mgr.canvas_w
        ch = self.tr.win_mgr.canvas_h
        return np.full((ch, cw, 3), COLOR_BG, dtype=np.uint8)

    def compose_canvas(self, frame):
        """窗口尺寸底板 + 顶部工具栏区 + 视频帧等比缩放居中 (真矢量模式, 无整画布信箱缩放)"""
        cw = self.tr.win_mgr.canvas_w
        ch = self.tr.win_mgr.canvas_h
        canvas = np.full((ch, cw, 3), COLOR_BG, dtype=np.uint8)

        # 视频帧等比 contain 缩放到工具栏下方区域并居中 (不裁剪, 保证 Tag 不出画)
        area_h = ch - TOOLBAR_H
        fh, fw = frame.shape[:2]
        scale = min(cw / fw, area_h / fh)
        nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        frame2 = cv2.resize(frame, (nw, nh), interpolation=interp)
        x0, y0 = (cw - nw) // 2, TOOLBAR_H + (area_h - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = frame2
        return canvas
