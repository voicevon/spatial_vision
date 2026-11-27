#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单根芦笋测量工作室 (Single Asparagus Measure Studio) - 纯只读界面渲染器
======================================================================
严格遵守只读渲染规范：仅读取控制器状态，合成画布并返回每帧交互命中表。
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.ui.gui_theme import GuiTheme
from src.utils.text_rendering import draw_text, measure_text
from src.vision.single_asparagus_measurer import (
    ORIENTATION_OPTIONS,
    SCALE_SOURCE_LABELS,
    SingleAsparagusMeasurer,
)


SLIDER_SPECS: List[Dict[str, Any]] = [
    {"key": "diameter_offset_mm", "label": "直径偏移(mm)", "min": 5.0, "max": 60.0, "step": 1.0, "fmt": "{:.0f}"},
    {"key": "green_h_low", "label": "绿茎 H 下限", "min": 10, "max": 60, "step": 1, "fmt": "{:.0f}"},
    {"key": "green_h_high", "label": "绿茎 H 上限", "min": 65, "max": 120, "step": 1, "fmt": "{:.0f}"},
    {"key": "white_s_max", "label": "白根 S 上限", "min": 20, "max": 130, "step": 1, "fmt": "{:.0f}"},
    {"key": "white_v_min", "label": "白根 V 下限", "min": 80, "max": 220, "step": 1, "fmt": "{:.0f}"},
    {"key": "fallback_mm_per_px", "label": "手动比例(mm/px)", "min": 0.05, "max": 1.00, "step": 0.01, "fmt": "{:.2f}"},
]


class AsparagusMeasureStudioRenderer:
    """单根芦笋测量工作室纯只读 UI 渲染器"""

    @staticmethod
    def compute_metrics(canvas_w: int) -> Dict[str, Any]:
        s = max(0.75, min(1.5, canvas_w / 1280.0))
        return {
            "s": s,
            "L": int(14 * s),
            "R": int(14 * s),
            "top_h": int(88 * s),
            "bot_h": int(34 * s),
            "left_w": int(220 * s),
            "right_w": int(320 * s),
            "gap": int(10 * s),
            "row_h": int(32 * s),
            "fs_title": max(14, int(17 * s)),
            "fs_md": max(11, int(13 * s)),
            "fs_sm": max(10, int(12 * s)),
            "fs_xs": max(9, int(11 * s)),
            "fs_big": max(20, int(26 * s)),
        }

    @staticmethod
    def compute_panels(W: int, H: int, m: Dict[str, Any]):
        y1 = m["top_h"]
        y2 = H - m["bot_h"]
        left_p = (m["L"], y1, m["L"] + m["left_w"], y2)
        right_p = (W - m["R"] - m["right_w"], y1, W - m["R"], y2)
        img_p = (left_p[2] + m["gap"], y1, right_p[0] - m["gap"], y2)
        return left_p, img_p, right_p

    @staticmethod
    def _draw_btn(
        canvas: np.ndarray,
        rect: Tuple[int, int, int, int],
        label: str,
        mouse_pos: Tuple[int, int],
        fs: int,
        active: bool = False,
        accent: Optional[Tuple[int, int, int]] = None,
        danger: bool = False,
    ):
        x1, y1, x2, y2 = rect
        mx, my = mouse_pos
        hover = (x1 <= mx <= x2 and y1 <= my <= y2)
        if danger:
            bg = (52, 24, 26) if hover else (34, 20, 22)
            bd = (210, 70, 70) if hover else (110, 42, 46)
            fg = (235, 185, 185)
        elif active:
            col = accent or GuiTheme.ACCENT
            bg = (28, 48, 60)
            bd = col
            fg = GuiTheme.TEXT
        else:
            bg = GuiTheme.CARD_HOVER if hover else GuiTheme.CARD_BG
            bd = GuiTheme.BORDER_HOVER if hover else GuiTheme.BORDER
            fg = GuiTheme.TEXT if hover else GuiTheme.TEXT_SUB

        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bd, 2 if (active or hover) else 1)
        (tw, th), _ = measure_text(label, font_size=fs)
        tx = x1 + max(4, (x2 - x1 - tw) // 2)
        ty = y1 + max(2, (y2 - y1 - th) // 2 - 1)
        draw_text(canvas, label, (tx, ty), fs, fg, bold=active)

    @classmethod
    def render_scene(cls, canvas: np.ndarray, app: Any):
        """渲染完整界面并返回 (buttons, sample_rows, dd_items, slider_bars)"""
        buttons: List[Tuple[str, Tuple[int, int, int, int], Any]] = []
        sample_rows: List[Tuple[int, Tuple[int, int, int, int]]] = []
        dd_items: List[Tuple[str, Tuple[int, int, int, int], Any]] = []
        slider_bars: List[Tuple[Tuple[int, int, int, int], Dict[str, Any]]] = []

        H, W = canvas.shape[:2]
        m = cls.compute_metrics(W)
        s = m["s"]
        left_p, img_p, right_p = cls.compute_panels(W, H, m)

        # 1. 顶部双排工具栏底板
        cv2.rectangle(canvas, (0, 0), (W, m["top_h"]), (19, 22, 28), -1)
        cv2.line(canvas, (0, m["top_h"] - 1), (W, m["top_h"] - 1), GuiTheme.BORDER, 1)

        # 第一排: 标题 + 工位下拉 + 数据源切换 + ROI 来源 + 退出
        title_str = "单根芦笋测量工作室"
        r1_y1, r1_h = int(10 * s), int(30 * s)
        r1_y2 = r1_y1 + r1_h
        draw_text(canvas, title_str, (m["L"], r1_y1 + int(4 * s)), m["fs_title"], GuiTheme.TEXT, bold=True)
        (tw_title, _), _ = measure_text(title_str, font_size=m["fs_title"])

        cur_x = m["L"] + tw_title + int(18 * s)

        # 工位下拉按钮
        ws_w = int(185 * s)
        ws_rect = (cur_x, r1_y1, cur_x + ws_w, r1_y2)
        ws_label = f"工位: {app.current_workspace_name[:10]} ▾"
        cls._draw_btn(canvas, ws_rect, ws_label, app.mouse_pos, m["fs_sm"], active=(app.active_dropdown == "workspace"))
        buttons.append(("TOGGLE_WS_DD", ws_rect, None))
        cur_x += ws_w + int(10 * s)

        # 数据源切换按钮: 离线样本 / 实时相机
        mode_w = int(130 * s)
        mode_rect = (cur_x, r1_y1, cur_x + mode_w, r1_y2)
        mode_label = "模式: 实时相机" if app.source_mode == "live" else "模式: 离线样本"
        cls._draw_btn(
            canvas, mode_rect, mode_label, app.mouse_pos, m["fs_sm"],
            active=(app.source_mode == "live"), accent=GuiTheme.OK,
        )
        buttons.append(("TOGGLE_SOURCE_MODE", mode_rect, None))
        cur_x += mode_w + int(10 * s)

        # 工位 ROI 下拉
        roi_w = int(175 * s)
        roi_rect = (cur_x, r1_y1, cur_x + roi_w, r1_y2)
        roi_label = f"ROI: {app.current_roi_label[:11]} ▾"
        cls._draw_btn(canvas, roi_rect, roi_label, app.mouse_pos, m["fs_sm"], active=(app.active_dropdown == "roi"))
        buttons.append(("TOGGLE_ROI_DD", roi_rect, None))
        cur_x += roi_w + int(10 * s)

        # 手动画框 ROI 按钮
        draw_w = int(125 * s)
        draw_rect = (cur_x, r1_y1, cur_x + draw_w, r1_y2)
        draw_label = "正在画框(ESC取消)" if app.is_drawing_roi else "手动画框 ROI [D]"
        cls._draw_btn(
            canvas, draw_rect, draw_label, app.mouse_pos, m["fs_sm"],
            active=app.is_drawing_roi, accent=GuiTheme.GOLD,
        )
        buttons.append(("TOGGLE_DRAW_ROI", draw_rect, None))
        cur_x += draw_w + int(10 * s)

        # 清除手动 ROI 按钮
        if app.manual_roi is not None:
            clr_w = int(84 * s)
            clr_rect = (cur_x, r1_y1, cur_x + clr_w, r1_y2)
            cls._draw_btn(canvas, clr_rect, "清除手框", app.mouse_pos, m["fs_sm"])
            buttons.append(("CLEAR_MANUAL_ROI", clr_rect, None))

        # 右侧退出按钮
        exit_w = int(72 * s)
        exit_rect = (W - m["R"] - exit_w, r1_y1, W - m["R"], r1_y2)
        cls._draw_btn(canvas, exit_rect, "退出", app.mouse_pos, m["fs_sm"], danger=True)
        buttons.append(("EXIT", exit_rect, None))

        # 第二排: 步骤药丸 + 朝向切换 + 刷新/抓拍
        r2_y1, r2_h = int(48 * s), int(28 * s)
        r2_y2 = r2_y1 + r2_h
        px = m["L"]
        for step_key, step_name in SingleAsparagusMeasurer.STEP_DEFS:
            bw = int(112 * s)
            rect = (px, r2_y1, px + bw, r2_y2)
            cls._draw_btn(
                canvas, rect, step_name, app.mouse_pos, m["fs_sm"],
                active=(app.active_step_key == step_key), accent=GuiTheme.ACCENT,
            )
            buttons.append(("SELECT_STEP", rect, step_key))
            px += bw + int(8 * s)

        # 朝向切换按钮
        ori_label = dict(ORIENTATION_OPTIONS).get(app.measurer.config.orientation, "左根 → 右尖")
        ori_w = int(136 * s)
        ori_x = px + int(8 * s)
        ori_rect = (ori_x, r2_y1, ori_x + ori_w, r2_y2)
        cls._draw_btn(canvas, ori_rect, f"朝向: {ori_label}", app.mouse_pos, m["fs_sm"], active=True, accent=GuiTheme.ACCENT)
        buttons.append(("CYCLE_ORIENTATION", ori_rect, None))

        # 4 标靶单应性拉正开关
        rect_x = ori_rect[2] + int(8 * s)
        rect_w = int(118 * s)
        rect_btn = (rect_x, r2_y1, rect_x + rect_w, r2_y2)
        rect_locked = bool(app.use_planar_rectify and app.rectifier is not None and app.rectifier.is_locked)
        rect_accent = GuiTheme.OK if rect_locked else GuiTheme.GOLD
        cls._draw_btn(
            canvas, rect_btn, "4标靶拉正 [P]", app.mouse_pos, m["fs_sm"],
            active=bool(app.use_planar_rectify), accent=rect_accent,
        )
        buttons.append(("TOGGLE_RECTIFY", rect_btn, None))

        # 批量评测按钮（离线模式）
        next_x = rect_btn[2] + int(8 * s)
        if app.source_mode == "offline":
            eval_w = int(96 * s)
            eval_btn = (next_x, r2_y1, next_x + eval_w, r2_y2)
            cls._draw_btn(canvas, eval_btn, "批量评测 [B]", app.mouse_pos, m["fs_sm"], active=(app.batch_eval is not None), accent=GuiTheme.ACCENT)
            buttons.append(("RUN_BATCH_EVAL", eval_btn, None))
            next_x = eval_btn[2] + int(8 * s)

        # 导出产线 YAML 按钮
        exp_w = int(110 * s)
        exp_btn = (next_x, r2_y1, next_x + exp_w, r2_y2)
        cls._draw_btn(canvas, exp_btn, "导出YAML [S]", app.mouse_pos, m["fs_sm"])
        buttons.append(("EXPORT_STATION_YAML", exp_btn, None))

        # 右侧操作按钮 (实时抓拍 / 离线刷新 + 复位视口)
        act_w = int(96 * s)
        reset_rect = (W - m["R"] - act_w, r2_y1, W - m["R"], r2_y2)
        cls._draw_btn(canvas, reset_rect, "复位视口", app.mouse_pos, m["fs_sm"])
        buttons.append(("RESET_VIEW", reset_rect, None))

        snap_rect = (reset_rect[0] - int(8 * s) - act_w, r2_y1, reset_rect[0] - int(8 * s), r2_y2)
        if app.source_mode == "live":
            cls._draw_btn(canvas, snap_rect, "抓拍保存", app.mouse_pos, m["fs_sm"], active=True, accent=GuiTheme.OK)
            buttons.append(("CAPTURE_LIVE", snap_rect, None))
        else:
            cls._draw_btn(canvas, snap_rect, "刷新样本", app.mouse_pos, m["fs_sm"])
            buttons.append(("RESCAN", snap_rect, None))

        # 2. 三栏主区域
        cls._draw_left_panel(canvas, m, left_p, app, sample_rows)
        cls._draw_center_viewport(canvas, m, img_p, app)
        cls._draw_right_panel(canvas, m, right_p, app, slider_bars)

        # 3. 底部状态栏
        cls._draw_footer(canvas, W, H, m, app)

        # 4. 顶层悬浮下拉菜单
        if app.active_dropdown == "workspace":
            cls._draw_dropdown(canvas, ws_rect, app.workspace_options, app.current_workspace_id,
                               "SELECT_WS", m, app.mouse_pos, dd_items)
        elif app.active_dropdown == "roi":
            cls._draw_dropdown(canvas, roi_rect, app.roi_options, app.selected_roi_id,
                               "SELECT_ROI", m, app.mouse_pos, dd_items)

        return buttons, sample_rows, dd_items, slider_bars

    @classmethod
    def _draw_left_panel(cls, canvas, m, rect, app, sample_rows):
        x1, y1, x2, y2 = rect
        s = m["s"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GuiTheme.CARD_BG, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GuiTheme.BORDER, 1)

        head_h = int(32 * s)
        cv2.rectangle(canvas, (x1, y1), (x2, y1 + head_h), (24, 30, 40), -1)
        cv2.line(canvas, (x1, y1 + head_h), (x2, y1 + head_h), GuiTheme.BORDER, 1)

        if app.source_mode == "live":
            draw_text(canvas, "实时相机状态", (x1 + int(10 * s), y1 + int(8 * s)), m["fs_md"], GuiTheme.OK, bold=True)
            cy = y1 + head_h + int(14 * s)
            lines = [
                f"后端: {app.camera_service.stream_desc}",
                f"状态: {'取流中' if app.camera_service.is_running else '未连接'}",
                f"深度流: {'已启用' if app.camera_service.depth_enabled else '无 (2D模式)'}",
                f"ROI模式: {'手动2D框' if app.manual_roi else '工位投影/全图'}",
                "",
                "快捷键指引:",
                "[M] 切换离线/实时",
                "[D] 手动画框 ROI",
                "[O] 切换根→尖方向",
                "[SPACE] 抓拍样本",
            ]
            for line in lines:
                if cy > y2 - int(18 * s):
                    break
                col = GuiTheme.TEXT if line.startswith("后端") or line.startswith("状态") else GuiTheme.TEXT_SUB
                draw_text(canvas, line, (x1 + int(10 * s), cy), m["fs_xs"], col)
                cy += int(20 * s)
            return

        # 离线模式: 样本列表
        title = f"工位样本 ({len(app.samples)})"
        draw_text(canvas, title, (x1 + int(10 * s), y1 + int(8 * s)), m["fs_md"], GuiTheme.TEXT, bold=True)

        if not app.samples:
            draw_text(canvas, "当前工位无样本", (x1 + int(12 * s), y1 + head_h + int(20 * s)), m["fs_sm"], GuiTheme.TEXT_MUTED)
            draw_text(canvas, "可切换 [实时相机]", (x1 + int(12 * s), y1 + head_h + int(44 * s)), m["fs_xs"], GuiTheme.ACCENT)
            return

        row_h = m["row_h"]
        gap = int(4 * s)
        cy = y1 + head_h + int(6 * s)
        mx, my = app.mouse_pos
        for idx in range(app.scroll_off, len(app.samples)):
            if cy + row_h > y2 - int(6 * s):
                break
            smp = app.samples[idx]
            r = (x1 + int(6 * s), cy, x2 - int(6 * s), cy + row_h)
            is_sel = (idx == app.sel_idx)
            is_hov = (r[0] <= mx <= r[2] and r[1] <= my <= r[3])
            bg = GuiTheme.CARD_SEL if is_sel else (GuiTheme.CARD_HOVER if is_hov else GuiTheme.CARD_BG)
            bd = GuiTheme.BORDER_SEL if is_sel else GuiTheme.BORDER
            cv2.rectangle(canvas, (r[0], r[1]), (r[2], r[3]), bg, -1)
            cv2.rectangle(canvas, (r[0], r[1]), (r[2], r[3]), bd, 1)

            tag = "[3D]" if smp["has_depth"] else "[2D]"
            name_disp = smp["name"]
            if len(name_disp) > 17:
                name_disp = name_disp[:15] + ".."
            draw_text(canvas, f"{tag} {name_disp}", (r[0] + int(6 * s), r[1] + int(7 * s)),
                      m["fs_xs"], GuiTheme.TEXT if is_sel else GuiTheme.TEXT_SUB, bold=is_sel)
            sample_rows.append((idx, r))
            cy += row_h + gap

    @classmethod
    def _draw_center_viewport(cls, canvas, m, rect, app):
        x1, y1, x2, y2 = rect
        s = m["s"]
        vw, vh = x2 - x1, y2 - y1
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (12, 14, 18), -1)

        vis_img = app.get_active_display_image()
        if vis_img is not None and vis_img.size > 0 and vw > 10 and vh > 10:
            fh, fw = vis_img.shape[:2]
            ix1, iy1, ix2, iy2, _ = app.viewport.compute_image_rect((x1, y1, vw, vh), fw, fh)
            # 裁剪绘制到视口内部
            cx1, cy1 = max(x1, ix1), max(y1, iy1)
            cx2, cy2 = min(x2, ix2), min(y2, iy2)
            if cx2 > cx1 and cy2 > cy1 and ix2 > ix1 and iy2 > iy1:
                sx1 = max(0, int(np.floor((cx1 - ix1) * fw / (ix2 - ix1))))
                sy1 = max(0, int(np.floor((cy1 - iy1) * fh / (iy2 - iy1))))
                sx2 = min(fw, int(np.ceil((cx2 - ix1) * fw / (ix2 - ix1))))
                sy2 = min(fh, int(np.ceil((cy2 - iy1) * fh / (iy2 - iy1))))
                if sx2 > sx1 and sy2 > sy1:
                    crop = vis_img[sy1:sy2, sx1:sx2]
                    resized = cv2.resize(crop, (cx2 - cx1, cy2 - cy1), interpolation=cv2.INTER_LINEAR)
                    canvas[cy1:cy2, cx1:cx2] = resized
        else:
            msg = app.error_msg or "暂无画面 — 请选择工位样本或切换至实时相机模式"
            draw_text(canvas, msg, (x1 + int(24 * s), y1 + vh // 2 - int(10 * s)), m["fs_md"], GuiTheme.TEXT_MUTED)

        # 手动画框 ROI 实时拖拽预览
        if app.is_drawing_roi and app.drag_roi_start is not None and app.drag_roi_curr is not None:
            dx1 = max(x1, min(x2, min(app.drag_roi_start[0], app.drag_roi_curr[0])))
            dy1 = max(y1, min(y2, min(app.drag_roi_start[1], app.drag_roi_curr[1])))
            dx2 = max(x1, min(x2, max(app.drag_roi_start[0], app.drag_roi_curr[0])))
            dy2 = max(y1, min(y2, max(app.drag_roi_start[1], app.drag_roi_curr[1])))
            cv2.rectangle(canvas, (dx1, dy1), (dx2, dy2), GuiTheme.GOLD, 2)

        # 顶部图例胶囊
        legend = "绿线:有效长度  洋红点:白绿分界/根切口  橙线:偏移20mm直径"
        cv2.rectangle(canvas, (x1 + int(8 * s), y1 + int(8 * s)),
                      (min(x2 - int(8 * s), x1 + int(430 * s)), y1 + int(32 * s)), (18, 22, 28), -1)
        draw_text(canvas, legend, (x1 + int(14 * s), y1 + int(13 * s)), m["fs_xs"], GuiTheme.TEXT_SUB)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), GuiTheme.GOLD if app.is_drawing_roi else GuiTheme.BORDER,
                      2 if app.is_drawing_roi else 1)

    @classmethod
    def _draw_right_panel(cls, canvas, m, rect, app, slider_bars):
        x1, y1, x2, y2 = rect
        s = m["s"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GuiTheme.CARD_BG, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GuiTheme.BORDER, 1)

        pad = int(12 * s)
        cy = y1 + int(10 * s)
        draw_text(canvas, "单根测量结果", (x1 + pad, cy), m["fs_md"], GuiTheme.ACCENT, bold=True)
        cy += int(26 * s)

        res = app.measurement
        # 两张核心指标大卡: 有效长度 & 20mm 处直径
        card_h = int(68 * s)
        card_w = (x2 - x1 - pad * 3) // 2
        len_rect = (x1 + pad, cy, x1 + pad + card_w, cy + card_h)
        dia_rect = (len_rect[2] + pad, cy, x2 - pad, cy + card_h)

        for r, label, val_str, col in (
            (len_rect, "绿色有效长度", f"{res.effective_length_mm:.1f} mm" if (res and res.valid) else "-- mm", GuiTheme.OK),
            (dia_rect, f"+{app.measurer.config.diameter_offset_mm:.0f}mm 处直径",
             f"{res.diameter_mm:.1f} mm" if (res and res.valid) else "-- mm", GuiTheme.GOLD),
        ):
            cv2.rectangle(canvas, (r[0], r[1]), (r[2], r[3]), (18, 22, 28), -1)
            cv2.rectangle(canvas, (r[0], r[1]), (r[2], r[3]), col if (res and res.valid) else GuiTheme.BORDER, 1)
            draw_text(canvas, label, (r[0] + int(8 * s), r[1] + int(8 * s)), m["fs_xs"], GuiTheme.TEXT_SUB)
            draw_text(canvas, val_str, (r[0] + int(8 * s), r[1] + int(30 * s)), m["fs_big"] - 4, col, bold=True)

        cy += card_h + int(12 * s)

        # 状态信息区
        if res and res.valid:
            white_txt = "检测到白根 (已剔除白根段)" if res.has_white_root else "全绿无白根 (按根切口起算)"
            scale_txt = (
                "机架4标靶单应性" if res.scale_source == "planar_homography"
                else SCALE_SOURCE_LABELS.get(res.scale_source, res.scale_source)
            )
            info_lines = [
                (f"白根判定: {white_txt}", GuiTheme.TEXT),
                (f"尺度来源: {scale_txt} ({res.mm_per_px:.3f} mm/px)", GuiTheme.TEXT_SUB),
                (f"处理耗时: {res.elapsed_ms:.1f} ms", GuiTheme.TEXT_MUTED),
            ]
            if app.grade_decision is not None:
                gd = app.grade_decision
                g_col = GuiTheme.OK if gd.matched else GuiTheme.GOLD
                info_lines.insert(0, (f"预测分级: Grade {gd.grade} ({gd.name})", g_col))
            gt_pair = app.current_sample_gt
            if gt_pair is not None:
                dl = res.effective_length_mm - gt_pair[0]
                dd = res.diameter_mm - gt_pair[1]
                info_lines.append((f"真值误差: ΔL={dl:+.1f}mm  ΔD={dd:+.1f}mm", GuiTheme.ACCENT))
            if res.warning:
                info_lines.append((f"提示: {res.warning}", GuiTheme.GOLD))
        else:
            reason = res.reason if res else "等待图像输入"
            info_lines = [(f"状态: {reason}", GuiTheme.TEXT_MUTED)]

        if app.use_planar_rectify and app.rectifier is not None:
            st = app.rectifier.status
            r_txt = f"4标靶拉正: 锁定 (err={st.reproj_err_px:.2f}px)" if st.locked else f"4标靶拉正: {st.reason}"
            info_lines.append((r_txt, GuiTheme.OK if st.locked else GuiTheme.GOLD))

        if app.batch_eval is not None:
            be = app.batch_eval
            if be.gt_matched_count > 0:
                be_txt = f"批量({be.valid_count}/{be.total}): L-MAE={be.len_mae_mm:.1f} D-MAE={be.diam_mae_mm:.1f}mm"
            else:
                be_txt = f"批量评测: 有效 {be.valid_count}/{be.total} (无GT表)"
            info_lines.append((be_txt, GuiTheme.ACCENT))

        for text, col in info_lines:
            draw_text(canvas, text, (x1 + pad, cy), m["fs_xs"], col)
            cy += int(20 * s)

        cy += int(8 * s)
        cv2.line(canvas, (x1 + pad, cy), (x2 - pad, cy), GuiTheme.BORDER, 1)
        cy += int(12 * s)

        # 参数微调滑条区
        draw_text(canvas, "算法阈值与参数微调", (x1 + pad, cy), m["fs_md"], GuiTheme.ACCENT, bold=True)
        cy += int(24 * s)

        cfg = app.measurer.config
        track_h = int(16 * s)
        for spec in SLIDER_SPECS:
            if cy + int(36 * s) > y2 - int(10 * s):
                break
            val = float(getattr(cfg, spec["key"]))
            val_txt = spec["fmt"].format(val)
            draw_text(canvas, f"{spec['label']}: {val_txt}", (x1 + pad, cy), m["fs_xs"], GuiTheme.TEXT_SUB)
            cy += int(16 * s)

            bar = (x1 + pad, cy, x2 - pad, cy + track_h)
            cv2.rectangle(canvas, (bar[0], bar[1] + track_h // 2 - 2),
                          (bar[2], bar[1] + track_h // 2 + 2), (42, 50, 62), -1)
            ratio = (val - spec["min"]) / max(1e-6, float(spec["max"] - spec["min"]))
            ratio = max(0.0, min(1.0, ratio))
            hx = int(bar[0] + ratio * (bar[2] - bar[0]))
            cv2.rectangle(canvas, (bar[0], bar[1] + track_h // 2 - 2),
                          (hx, bar[1] + track_h // 2 + 2), GuiTheme.ACCENT, -1)
            cv2.circle(canvas, (hx, bar[1] + track_h // 2), max(5, int(6 * s)), GuiTheme.TEXT, -1)
            slider_bars.append((bar, spec))
            cy += track_h + int(10 * s)

    @classmethod
    def _draw_footer(cls, canvas, W, H, m, app):
        s = m["s"]
        y1 = H - m["bot_h"]
        cv2.rectangle(canvas, (0, y1), (W, H), (14, 17, 22), -1)
        cv2.line(canvas, (0, y1), (W, y1), GuiTheme.BORDER, 1)
        msg = app.toast_text or "提示: [P] 4标靶拉正 | [B] 批量评测 | [S] 导出产线YAML | [D] 画框ROI | [M] 离线/实时 | [ESC] 退出"
        draw_text(canvas, msg, (m["L"], y1 + int(8 * s)), m["fs_xs"], GuiTheme.TEXT_SUB)

    @classmethod
    def _draw_dropdown(cls, canvas, anchor_rect, options, current_val, action_id, m, mouse_pos, dd_items):
        if not options:
            return
        s = m["s"]
        x1, _, x2, y1 = anchor_rect
        item_h = int(28 * s)
        dd_w = max(x2 - x1, int(210 * s))
        total_h = item_h * len(options)
        cv2.rectangle(canvas, (x1, y1 + 2), (x1 + dd_w, y1 + 2 + total_h), (22, 26, 34), -1)
        cv2.rectangle(canvas, (x1, y1 + 2), (x1 + dd_w, y1 + 2 + total_h), GuiTheme.ACCENT, 1)

        mx, my = mouse_pos
        cy = y1 + 2
        for opt_key, opt_label in options:
            r = (x1, cy, x1 + dd_w, cy + item_h)
            is_sel = (opt_key == current_val)
            is_hov = (r[0] <= mx <= r[2] and r[1] <= my <= r[3])
            if is_sel or is_hov:
                cv2.rectangle(canvas, (r[0] + 1, r[1] + 1), (r[2] - 1, r[3] - 1),
                              GuiTheme.CARD_SEL if is_sel else GuiTheme.CARD_HOVER, -1)
            draw_text(canvas, opt_label[:26], (r[0] + int(8 * s), r[1] + int(6 * s)),
                      m["fs_xs"], GuiTheme.TEXT if (is_sel or is_hov) else GuiTheme.TEXT_SUB, bold=is_sel)
            dd_items.append((action_id, r, opt_key))
            cy += item_h
