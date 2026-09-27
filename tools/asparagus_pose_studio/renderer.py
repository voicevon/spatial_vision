#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
芦笋位姿工作室 (Asparagus Pose Studio) - 界面渲染模块
=====================================================
负责真矢量自适应画布重绘、三栏布局面板、视口切片映射、目标卡片与置顶菜单渲染。
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.gui_theme import GuiTheme
from src.utils.gui_components import draw_dropdown_button, render_dropdown_popup, render_floating_tooltip
from src.utils.text_rendering import draw_text, measure_text
from src.vision.pipelines.base_pipeline import PipelineStep
from tools.asparagus_pose_studio.data_io import BASE_W, CALIB_LABELS

# 步骤 0: 原始图 (虚拟步骤, 所有算法路线通用, 视口直接显示原始输入图像)
ORIG_VIEW_STEP = PipelineStep(
    "stage0_original", "0.原始图",
    "原始输入彩色图像 (未经算法处理)"
)

# 步骤 1B: 边缘提取 (虚拟步骤, Canny 边缘预览, 与 1A HSV 分割并列的步骤 1 特征提取分支)
EDGE_VIEW_STEP = PipelineStep(
    "stage1b_edge", "1B.边缘提取",
    "Canny 边缘提取预览 (灰度梯度阈值化轮廓, 与 1A 同级并列分支)"
)

# 步骤 2B: 腐蚀与膨胀 (虚拟步骤, B 系分支清理视图, 数据源 1B Canny 边缘图)
MORPH_VIEW_STEP = PipelineStep(
    "stage2b_morph", "2B.腐蚀与膨胀",
    "B 系分支: 对 1B Canny 边缘图做闭运算清理, 供步骤 3 融合汇合"
)


class AsparagusPoseStudioRenderer:
    """芦笋位姿工作室专属 UI 渲染器"""

    @staticmethod
    def compute_metrics(canvas_w: int) -> Dict[str, Any]:
        """计算物理画布与逻辑基准尺寸的动态缩放度量"""
        s = canvas_w / BASE_W
        return {
            "s": s,
            "L": int(12 * s),            # 全局左边距
            "list_w": int(140 * s),      # 左侧样本列表宽
            "right_w": int(236 * s),     # 右侧结果面板宽
            "header_h": int(118 * s),    # 三排工具栏高度 (第一排全局操作, 第二三排算法步骤视图)
            "bottom_h": int(46 * s),
            "row_h": int(32 * s),        # 样本行高
            "btn_h": int(28 * s),        # 按钮基准高度
            "fs_title": max(14, int(20 * s)),
            "fs_sub": max(10, int(12 * s)),
            "fs_body": max(11, int(13 * s)),
            "fs_small": max(9, int(11 * s)),
            "fs_gcode": max(9, int(11 * s)),
        }

    @staticmethod
    def compute_panels(canvas_w: int, canvas_h: int, m: Dict[str, Any]):
        """计算三栏面板物理矩形: (样本列表, 图像视口, 右侧结果)"""
        top = m["header_h"]
        bottom = canvas_h - m["bottom_h"]
        x0 = m["L"]
        x1 = x0 + m["list_w"]
        x4 = canvas_w - m["L"]
        x3 = x4 - m["right_w"]
        return (
            (x0, top, x1, bottom),
            (x1 + int(10 * m["s"]), top, x3 - int(10 * m["s"]), bottom),
            (x3, top, x4, bottom)
        )

    @staticmethod
    def draw_panel_bg(canvas: np.ndarray, rect: Tuple[int, int, int, int], title: str, m: Dict[str, Any]) -> int:
        """绘制带标题栏的标准卡片面板背景，返回内容区域起始 Y 坐标"""
        x1, y1, x2, y2 = rect
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GuiTheme.CARD_BG, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GuiTheme.BORDER, 1)
        header_h = int(26 * m["s"])
        cv2.rectangle(canvas, (x1, y1), (x2, y1 + header_h), (26, 30, 38), -1)
        draw_text(canvas, title, (x1 + int(8 * m["s"]), y1 + int(6 * m["s"])), m["fs_body"],
                  GuiTheme.ACCENT, bold=True)
        return y1 + int(30 * m["s"])

    @staticmethod
    def draw_button(
        canvas: np.ndarray,
        rect: Tuple[int, int, int, int],
        label: str,
        mouse_pos: Tuple[int, int],
        m: Dict[str, Any],
        enabled: bool = True,
        active: bool = False
    ) -> bool:
        """绘制标准功能按钮并检测悬停高亮"""
        x1, y1, x2, y2 = rect
        mx, my = mouse_pos
        hover = enabled and (x1 <= mx <= x2 and y1 <= my <= y2)

        if not enabled:
            bg, border, col = GuiTheme.BTN_DISABLED_BG, GuiTheme.BTN_DISABLED_BORDER, GuiTheme.TEXT_DISABLED
        elif active:
            bg, border, col = GuiTheme.CARD_SEL, GuiTheme.BORDER_SEL, GuiTheme.WHITE
        elif hover:
            if "退出" in label:
                bg, border, col = (45, 38, 75), (80, 80, 220), (230, 230, 255)
            else:
                bg, border, col = GuiTheme.BTN_HOVER, GuiTheme.BORDER_HOVER, GuiTheme.BTN_TEXT_HOVER
        else:
            if "退出" in label:
                bg, border, col = GuiTheme.BTN, (60, 60, 110), GuiTheme.BTN_TEXT
            else:
                bg, border, col = GuiTheme.BTN, GuiTheme.BTN_BORDER, GuiTheme.BTN_TEXT

        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 2 if hover else 1)
        (tw, th), _ = measure_text(label, font_size=m["fs_sub"])
        draw_text(
            canvas, label,
            (x1 + ((x2 - x1) - tw) // 2, y1 + ((y2 - y1) - th) // 2),
            m["fs_sub"], col,
            bold=(hover and GuiTheme.BTN_BEHAVIOR["HOVER_BOLD"]) or active
        )
        return enabled

    @staticmethod
    def draw_view_pill(
        canvas: np.ndarray,
        rect: Tuple[int, int, int, int],
        label: str,
        is_active: bool,
        mouse_pos: Tuple[int, int],
        m: Dict[str, Any]
    ):
        """绘制算法中间步骤药丸按钮 (互斥激活、发光小圆点指示)"""
        x1, y1, x2, y2 = rect
        mx, my = mouse_pos
        hover = (x1 <= mx <= x2 and y1 <= my <= y2)

        if is_active:
            bg = (32, 68, 48)            # 翡翠绿底色
            border = (0, 235, 120)        # 高光绿边框
            text_col = (255, 255, 255)
        elif hover:
            bg = (34, 38, 46)
            border = (110, 130, 155)
            text_col = (235, 240, 245)
        else:
            bg = (24, 28, 34)
            border = (52, 58, 68)
            text_col = (165, 175, 185)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 2 if is_active else 1)

        (tw, th), _ = measure_text(label, font_size=m["fs_sub"])
        tx = x1 + ((x2 - x1) - tw) // 2
        ty = y1 + ((y2 - y1) - th) // 2
        draw_text(canvas, label, (tx, ty), m["fs_sub"], text_col, bold=is_active)

    @classmethod
    def _draw_step_tooltip(
        cls,
        canvas: np.ndarray,
        m: Dict[str, Any],
        step_obj: Any,
        pill_rect: Tuple[int, int, int, int],
        is_hovered: bool,
        img_panel_rect: Tuple[int, int, int, int]
    ):
        """在步骤药丸附近渲染多行专业知识卡片 (统一调用公共 render_floating_tooltip)"""
        # 整理行内容: (标签, 颜色, 内容)
        content_rows = [
            ("功能定位", (240, 245, 250), step_obj.description)
        ]
        if getattr(step_obj, "details", ""):
            content_rows.append(("算法原理", (120, 215, 255), step_obj.details))
        if getattr(step_obj, "parameters", ""):
            content_rows.append(("核心参数", (255, 210, 70), step_obj.parameters))
        if getattr(step_obj, "pros_cons", ""):
            content_rows.append(("优缺边界", (180, 245, 160), step_obj.pros_cons))

        pill_cx = (pill_rect[0] + pill_rect[2]) // 2
        anchor_pos = (pill_cx - 15, pill_rect[3])
        theme_col = (0, 235, 140) if is_hovered else (70, 160, 230)
        line_h = int(21 * m["s"])

        render_floating_tooltip(
            canvas=canvas,
            title=f"步骤详解 · {step_obj.name}",
            lines=content_rows,
            anchor_pos=anchor_pos,
            theme_color=theme_col,
            font_size=m["fs_small"],
            line_height=line_h,
        )


    @classmethod
    def render_scene(
        cls,
        canvas: np.ndarray,
        app_state: Any
    ) -> Tuple[List[Any], List[Any], List[Any], List[Any], List[Any]]:
        """
        完整渲染场景画布并返回可交互区域映射:
        返回 (buttons, sample_rows, result_rows, dd_items, slider_bars)
        """
        buttons = []
        sample_rows = []
        result_rows = []
        dd_items = []
        slider_bars = []

        m = cls.compute_metrics(canvas.shape[1])
        W, H = canvas.shape[1], canvas.shape[0]

        # 1. 标题文字 (第一排最左侧 Logo)
        title_str = "芦笋位姿工作室"
        draw_text(canvas, title_str,
                  (m["L"], int(15 * m["s"])), m["fs_title"], GuiTheme.TEXT, bold=True)
        (tw_title, _), _ = measure_text(title_str, font_size=m["fs_title"])

        list_p, img_p, right_p = cls.compute_panels(W, H, m)

        # 2. 渲染三栏内容
        cls._draw_sample_list(canvas, m, list_p, app_state, sample_rows)
        cls._draw_image_area(canvas, m, img_p, app_state, slider_bars)
        cls._draw_result_panel(canvas, m, right_p, app_state, result_rows)

        # 3. 第一排工具栏排布：
        #    左侧紧随 Logo: 地图下拉按钮 -> 算法下拉按钮
        #    最右侧: 退出 -> 导出 G-code
        r1_y1 = int(12 * m["s"])
        r1_y2 = r1_y1 + m["btn_h"]

        left_x = m["L"] + tw_title + int(20 * m["s"])

        # 地图下拉按钮 (紧靠 Logo 右侧)
        sc_w = int(170 * m["s"])
        workspace_rect = (left_x, r1_y1, left_x + sc_w, r1_y2)
        app_state._workspace_rect = workspace_rect
        is_sc_open = (app_state.active_dropdown == "WORKSPACE_DROPDOWN")
        draw_dropdown_button(
            canvas, workspace_rect, f"地图: {app_state.current_workspace_name}",
            is_sc_open, app_state.mouse_pos, font_size=m["fs_sub"]
        )
        buttons.append((workspace_rect, ("toggle_dd", "WORKSPACE_DROPDOWN")))
        left_x += sc_w + int(10 * m["s"])

        # 算法下拉按钮 (紧随地图右侧)
        pipe_w = int(210 * m["s"])
        pipeline_rect = (left_x, r1_y1, left_x + pipe_w, r1_y2)
        app_state._pipeline_rect = pipeline_rect
        is_pipe_open = (app_state.active_dropdown == "PIPELINE_DROPDOWN")
        curr_pipe_name = app_state.current_pipeline_name
        short_pipe = curr_pipe_name.split(":")[0] if ":" in curr_pipe_name else curr_pipe_name
        draw_dropdown_button(
            canvas, pipeline_rect, f"算法: {short_pipe}",
            is_pipe_open, app_state.mouse_pos, font_size=m["fs_sub"]
        )
        buttons.append((pipeline_rect, ("toggle_dd", "PIPELINE_DROPDOWN")))

        # 第一排最右侧：退出 [X]
        bx = W - m["L"]

        # 退出按钮
        btn_exit_w = int(96 * m["s"])
        bx -= btn_exit_w
        exit_rect = (bx, r1_y1, bx + btn_exit_w, r1_y2)
        cls.draw_button(canvas, exit_rect, "退出 [X]", app_state.mouse_pos, m, enabled=True)
        buttons.append((exit_rect, ("btn", "退出 [X]")))

        # 4. 视口上方算法阶段步骤药丸视图 (分支流布局) 与悬停提示框:
        #    数据流语义: 0.原始图 → 分叉 {A 系上排 / B 系下排} → 汇合 → 主干编号步骤
        #    步骤名 "NA."/"NB." 前缀决定所属分支排, 纯 "N." 步骤位于主干; 主干药丸位于 1.5 排高度垂直居中
        row_a_y1 = int(48 * m["s"])
        row_b_y1 = row_a_y1 + m["btn_h"] + int(4 * m["s"])
        row_mid_y1 = row_a_y1 + (row_b_y1 - row_a_y1) // 2
        steps = app_state.pipeline.get_steps() if app_state.pipeline else []

        # 在图像视口左侧对齐排布步骤视图
        pill_x = img_p[0]
        (lbl_w, lbl_h), _ = measure_text("步骤视图:", font_size=m["fs_sub"])
        draw_text(canvas, "步骤视图:", (pill_x, row_mid_y1 + (m["btn_h"] - lbl_h) // 2),
                  m["fs_sub"], GuiTheme.TEXT_MUTED)
        pill_x += lbl_w + int(10 * m["s"])

        hovered_step = None
        hovered_rect = None
        mx, my = app_state.mouse_pos

        def _measure_pill_w(name):
            (tw, _), _ = measure_text(name, font_size=m["fs_sub"])
            return max(int(68 * m["s"]), tw + int(18 * m["s"]))

        def _emit_pill(s_obj, px, py1, pw=None):
            """绘制单个步骤药丸并登记悬停命中区, 返回药丸宽度"""
            nonlocal hovered_step, hovered_rect
            pill_w = pw if pw is not None else _measure_pill_w(s_obj.name)
            pill_rect = (px, py1, px + pill_w, py1 + m["btn_h"])
            if pill_rect[0] <= mx <= pill_rect[2] and pill_rect[1] <= my <= pill_rect[3]:
                hovered_step = s_obj
                hovered_rect = pill_rect
            cls.draw_view_pill(canvas, pill_rect, s_obj.name,
                               is_active=(app_state.active_step_key == s_obj.key),
                               mouse_pos=app_state.mouse_pos, m=m)
            buttons.append((pill_rect, ("set_step", s_obj.key)))
            return pill_w

        # 步骤分流 (按步骤名编号后的字母后缀): "NA."→上排 A 系, "NB."→下排 B 系, 其余→主干
        upper_steps, lower_steps, main_steps = [], [], []
        for s_obj in steps:
            mt = re.match(r"^(\d+)([AB])\.", s_obj.name)
            if mt is not None and mt.group(2) == "A":
                upper_steps.append(s_obj)
            elif mt is not None and mt.group(2) == "B":
                lower_steps.append(s_obj)
            else:
                main_steps.append(s_obj)
        pipeline = app_state.pipeline
        if pipeline is not None and hasattr(pipeline, "_stage1b_edges"):
            lower_steps[0:0] = [EDGE_VIEW_STEP, MORPH_VIEW_STEP]   # B 系虚拟步骤: 仅 B 分支流水线 (1B 边缘提取 + 2B 腐蚀与膨胀)

        def _emit_row(objs, px, py1):
            """水平排布一排药丸, 返回排尾 x 坐标 (已含步距)"""
            for s_obj in objs:
                px += _emit_pill(s_obj, px, py1) + int(8 * m["s"])
            return px

        # 主干起点: 0.原始图 (1.5 排高度)
        px = _emit_row([ORIG_VIEW_STEP], pill_x, row_mid_y1)

        # 分叉块: A 系步骤走上排, B 系步骤 (含 1B 虚拟边缘提取) 走下排
        branch_x = px
        upper_end = _emit_row(upper_steps, branch_x, row_a_y1)
        lower_end = _emit_row(lower_steps, branch_x, row_b_y1)
        px = max(upper_end, lower_end)

        # 汇合主干: 主干编号步骤 (1.5 排高度)
        px = _emit_row(main_steps, px, row_mid_y1)

        # 仅鼠标悬停在步骤药丸上时显示提示气泡框，鼠标离开即隐藏
        if hovered_step and hovered_rect and hovered_step.description:
            cls._draw_step_tooltip(
                canvas, m, hovered_step, hovered_rect,
                is_hovered=True,
                img_panel_rect=img_p
            )

        # 5. 底部状态栏
        yb = H - m["bottom_h"] + int(8 * m["s"])
        calib = CALIB_LABELS.get(
            app_state.targets[0].calibration_source if app_state.targets else "uncalibrated", "-"
        )
        n3d = sum(1 for smp in app_state.samples if smp.get("depth"))
        status = (
            f"工位【{app_state.current_workspace_name}】 · "
            f"样本 {len(app_state.samples)} 个 (3D成对 {n3d} / 2D {len(app_state.samples) - n3d}) · "
            f"标定: {calib}"
        )
        draw_text(canvas, status, (m["L"], yb), m["fs_sub"], GuiTheme.TEXT_SUB)
        draw_text(
            canvas,
            "[↑↓] 样本  ·  [空格] 重新解算  ·  右键拖拽  ·  滚轮无级缩放  ·  双击复位",
            (W - int(460 * m["s"]), yb), m["fs_sub"], GuiTheme.TEXT_MUTED
        )

        # 8. 悬浮 Toast
        if app_state._toast_msg and time.time() < app_state._toast_until:
            (tw, th), _ = measure_text(app_state._toast_msg, font_size=m["fs_body"])
            tx, ty = (W - tw) // 2, H - int(76 * m["s"])
            pad = int(10 * m["s"])
            cv2.rectangle(canvas, (tx - pad, ty - int(6 * m["s"])),
                          (tx + tw + pad, ty + th + int(8 * m["s"])), (40, 34, 26), -1)
            cv2.rectangle(canvas, (tx - pad, ty - int(6 * m["s"])),
                          (tx + tw + pad, ty + th + int(8 * m["s"])), GuiTheme.WARN, 1)
            draw_text(canvas, app_state._toast_msg, (tx, ty), m["fs_body"], GuiTheme.WARN, bold=True)

        # 9. 置顶弹出下拉框
        if app_state.active_dropdown == "WORKSPACE_DROPDOWN" and app_state._workspace_rect:
            item_h = int(28 * m["s"])
            btns = render_dropdown_popup(
                canvas, anchor_rect=app_state._workspace_rect,
                options=app_state.workspace_options,
                active_key=app_state.current_workspace_id,
                btn_prefix="DD_MAP_", item_h=item_h, min_width=int(260 * m["s"])
            )
            dd_items = [(r, k) for _, r, k in btns]
        elif app_state.active_dropdown == "PIPELINE_DROPDOWN" and getattr(app_state, "_pipeline_rect", None):
            item_h = int(28 * m["s"])
            btns = render_dropdown_popup(
                canvas, anchor_rect=app_state._pipeline_rect,
                options=app_state.pipeline_options,
                active_key=app_state.pipeline_key,
                btn_prefix="DD_PIPE_", item_h=item_h, min_width=int(260 * m["s"])
            )
            dd_items = [(r, k) for _, r, k in btns]

        return buttons, sample_rows, result_rows, dd_items, slider_bars

    @classmethod
    def _draw_sample_list(cls, canvas, m, rect, app_state, sample_rows):
        y = cls.draw_panel_bg(canvas, rect, f"样本列表 ({len(app_state.samples)})", m)
        if not app_state.samples:
            draw_text(canvas, "当前工位无生产样本", (rect[0] + int(10 * m["s"]), y + int(10 * m["s"])),
                      m["fs_body"], GuiTheme.TEXT_MUTED)
            draw_text(canvas, "请在采集向导中拍摄生产样本,",
                      (rect[0] + int(10 * m["s"]), y + int(32 * m["s"])), m["fs_small"], GuiTheme.TEXT_MUTED)
            draw_text(canvas, "或通过上方地图下拉框切换工位",
                      (rect[0] + int(10 * m["s"]), y + int(50 * m["s"])), m["fs_small"], GuiTheme.TEXT_MUTED)
            return

        x1, _, x2, y2 = rect
        row_h = m["row_h"]
        visible = max(1, (y2 - y - int(6 * m["s"])) // (row_h + int(4 * m["s"])))
        app_state.scroll_off = max(0, min(app_state.scroll_off, len(app_state.samples) - visible))

        for row_i in range(visible):
            idx = app_state.scroll_off + row_i
            if idx >= len(app_state.samples):
                break
            smp = app_state.samples[idx]
            ry1 = y + row_i * (row_h + int(4 * m["s"]))
            ry2 = ry1 + row_h
            is_sel = (idx == app_state.sel_idx)

            if is_sel:
                cv2.rectangle(canvas, (x1 + 2, ry1), (x2 - 2, ry2), GuiTheme.CARD_SEL, -1)
                cv2.rectangle(canvas, (x1 + 2, ry1), (x2 - 2, ry2), GuiTheme.BORDER_SEL, 1)
            elif x1 < app_state.mouse_pos[0] < x2 and ry1 <= app_state.mouse_pos[1] <= ry2:
                cv2.rectangle(canvas, (x1 + 2, ry1), (x2 - 2, ry2), GuiTheme.CARD_HOVER, -1)

            raw_name = smp["name"]
            short_name = raw_name.replace(".png", "").replace(".jpg", "").replace("view_", "")
            if len(short_name) > 10:
                short_name = short_name[-10:]

            draw_text(canvas, short_name, (x1 + int(8 * m["s"]), ry1 + int(6 * m["s"])),
                      m["fs_small"], GuiTheme.WHITE if is_sel else GuiTheme.TEXT_SUB, bold=is_sel)

            tag = "3D" if smp.get("depth") else "2D"
            tag_col = GuiTheme.OK if smp.get("depth") else GuiTheme.TEXT_MUTED
            draw_text(canvas, tag, (x2 - int(24 * m["s"]), ry1 + int(6 * m["s"])),
                      m["fs_small"], tag_col, bold=is_sel)
            sample_rows.append(((x1 + 2, ry1, x2 - 2, ry2), idx))

    @classmethod
    def _draw_image_area(cls, canvas, m, rect, app_state, slider_bars=None):
        x1, y1, x2, y2 = rect
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (10, 12, 16), -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), GuiTheme.BORDER, 1)

        if app_state.vis_img is None:
            msg = app_state.error if app_state.error else ("请在左侧选择样本" if app_state.samples else "当前工位无样本")
            draw_text(canvas, msg, (x1 + int(16 * m["s"]), y1 + int(16 * m["s"])),
                      m["fs_body"], GuiTheme.ERR if app_state.error else GuiTheme.TEXT_MUTED)
            return

        vx, vy, vw, vh = x1 + 2, y1 + 2, x2 - x1 - 4, y2 - y1 - 4
        ih, iw = app_state.vis_img.shape[:2]
        rois = app_state.viewport.compute_viewport_render_rois((vx, vy, vw, vh), iw, ih)

        if rois:
            (src_x1, src_y1, src_x2, src_y2), (dst_x1, dst_y1, dst_x2, dst_y2) = rois
            src_crop = app_state.vis_img[src_y1:src_y2, src_x1:src_x2]
            dw, dh = dst_x2 - dst_x1, dst_y2 - dst_y1
            if dw > 0 and dh > 0 and src_crop.size > 0:
                interp = cv2.INTER_LINEAR if app_state.viewport.zoom_level > 1.0 else cv2.INTER_AREA
                disp = cv2.resize(src_crop, (dw, dh), interpolation=interp)
                canvas[dst_y1:dst_y2, dst_x1:dst_x2] = disp

        # 视口底部模式指示
        if app_state.targets:
            mode_txt = (
                f"{'3D 完整链路' if app_state.mode == '3d' else '2D 预览'} — "
                f"提取前 {len(app_state.targets)} 位目标 "
                f"(已高亮目标 #{app_state.targets[app_state.sel_target].id})"
            )
            mode_col = GuiTheme.OK if app_state.mode == "3d" else GuiTheme.WARN
        else:
            mode_txt = (
                f"未检出符合规格目标 (当前算法: {app_state.current_pipeline_name})"
                if (app_state.samples and app_state.sel_idx >= 0)
                else "请在左侧列表选择样本照片"
            )
            mode_col = GuiTheme.WARN if (app_state.samples and app_state.sel_idx >= 0) else GuiTheme.GOLD
        draw_text(canvas, mode_txt, (x1 + int(8 * m["s"]), y2 - int(20 * m["s"])),
                  m["fs_small"], mode_col, bold=True)

        # 视口右上角缩放比例悬浮指示
        if abs(app_state.viewport.zoom_level - 1.0) > 0.01 or app_state.viewport.is_panning:
            zoom_badge = f"缩放: {app_state.viewport.zoom_level:.1f}x [右键拖拽/双击复位]"
            (zw, zh), _ = measure_text(zoom_badge, font_size=m["fs_small"])
            cv2.rectangle(canvas, (x2 - zw - int(16 * m["s"]), y1 + int(8 * m["s"])),
                          (x2 - int(6 * m["s"]), y1 + zh + int(14 * m["s"])), (20, 24, 30), -1)
            cv2.rectangle(canvas, (x2 - zw - int(16 * m["s"]), y1 + int(8 * m["s"])),
                          (x2 - int(6 * m["s"]), y1 + zh + int(14 * m["s"])), GuiTheme.BORDER, 1)
            draw_text(canvas, zoom_badge, (x2 - zw - int(11 * m["s"]), y1 + int(11 * m["s"])),
                      m["fs_small"], GuiTheme.ACCENT)

        # 步骤调参滑条 (仅当当前流水线为激活步骤声明了 STEP_SLIDERS 时渲染)
        spec_map = getattr(app_state.pipeline, "STEP_SLIDERS", {}) if app_state.pipeline else {}
        specs = spec_map.get(app_state.active_step_key, [])
        if specs and slider_bars is not None:
            cls._draw_step_sliders(canvas, m, rect, app_state, specs, slider_bars)

    @classmethod
    def _draw_step_sliders(cls, canvas, m, rect, app_state, specs, slider_bars):
        """在视口底部渲染调参滑条 (spec 含 attr_low/attr_high 为双滑块区间, 含 attr 为单滑块参数)"""
        x1, _, x2, y2 = rect
        row_h = int(26 * m["s"])
        strip_h = len(specs) * row_h + int(6 * m["s"])
        strip_y2 = y2 - int(24 * m["s"])     # 预留视口底部模式指示行
        strip_y1 = strip_y2 - strip_h

        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1 + 2, strip_y1), (x2 - 2, strip_y2), (14, 18, 24), -1)
        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
        cv2.line(canvas, (x1 + 2, strip_y1), (x2 - 2, strip_y1), GuiTheme.BORDER, 1)

        for i, spec in enumerate(specs):
            ry1 = strip_y1 + int(4 * m["s"]) + i * row_h
            ry2 = ry1 + row_h
            track_y = (ry1 + ry2) // 2
            tx1 = x1 + int(70 * m["s"])      # 左侧标签区
            tx2 = x2 - int(64 * m["s"])      # 右侧数值区
            is_single = "attr" in spec    # 单滑块 (单一参数) 或双滑块 (下限/上限区间)
            lo = int(getattr(app_state.pipeline, spec["attr"] if is_single else spec["attr_low"]))
            hi = lo if is_single else int(getattr(app_state.pipeline, spec["attr_high"]))
            vmin, vmax = spec["vmin"], spec["vmax"]

            def to_px(val: float) -> int:
                span = max(1, vmax - vmin)
                return int(tx1 + (tx2 - tx1) * (val - vmin) / span)

            draw_text(canvas, spec["label"], (x1 + int(10 * m["s"]), ry1 + (row_h - int(13 * m["s"])) // 2),
                      m["fs_small"], GuiTheme.TEXT_SUB, bold=True)

            cv2.line(canvas, (tx1, track_y), (tx2, track_y), (58, 66, 78), 2)
            cv2.line(canvas, (to_px(lo), track_y), (to_px(hi), track_y), (80, 240, 120), 3)

            handle_r = max(4, int(5 * m["s"]) + 1)
            for val in (lo, hi):
                hx = to_px(val)
                cv2.circle(canvas, (hx, track_y), handle_r + 1, (0, 0, 0), -1)
                cv2.circle(canvas, (hx, track_y), handle_r, (80, 240, 120), -1)
                cv2.circle(canvas, (hx, track_y), handle_r + 1, (235, 245, 255), 1)

            draw_text(canvas, f"{lo}" if is_single else f"{lo},{hi}",
                      (x2 - int(58 * m["s"]), ry1 + (row_h - int(13 * m["s"])) // 2),
                      m["fs_small"], GuiTheme.OK, bold=True)

            slider_bars.append(((tx1, ry1, tx2, ry2), dict(spec)))

    @classmethod
    def _draw_result_panel(cls, canvas, m, rect, app_state, result_rows):
        perf_tag = f" · {app_state.pipeline_result.elapsed_ms:.0f}ms" if app_state.pipeline_result else ""
        y = cls.draw_panel_bg(canvas, rect, f"识别结果 (前3位){perf_tag}", m)
        x1, _, x2, y2 = rect

        if not app_state.targets:
            msg = app_state.error if app_state.error else (
                "点击【识别定位】开始分析" if (app_state.samples and app_state.sel_idx >= 0) else "未检出目标"
            )
            draw_text(canvas, msg, (x1 + int(10 * m["s"]), y + int(14 * m["s"])),
                      m["fs_body"], GuiTheme.ERR if app_state.error else GuiTheme.TEXT_MUTED)
            cls._draw_pose_detail_box(canvas, m, rect, app_state)
            return

        card_h = int(60 * m["s"])
        for list_idx, t in enumerate(app_state.targets):
            ry1 = y + list_idx * (card_h + int(8 * m["s"]))
            if ry1 + card_h > y2 - int(190 * m["s"]):
                break
            ry2 = ry1 + card_h
            is_sel = (list_idx == app_state.sel_target)
            is_top = t.is_topmost

            bg_col = (28, 38, 32) if is_sel else (20, 24, 30)
            border_col = (0, 255, 120) if is_sel else ((240, 180, 40) if is_top else (55, 65, 80))
            cv2.rectangle(canvas, (x1 + 4, ry1), (x2 - 4, ry2), bg_col, -1)
            cv2.rectangle(canvas, (x1 + 4, ry1), (x2 - 4, ry2), border_col, 2 if is_sel else 1)

            badge = "#1最优" if is_top else f"#{t.id}候选"
            badge_col = (0, 255, 120) if is_sel else ((240, 180, 40) if is_top else GuiTheme.TEXT_SUB)
            draw_text(canvas, f"{badge} D:{t.diam_mm} L:{int(t.length_mm)}mm",
                      (x1 + int(8 * m["s"]), ry1 + int(5 * m["s"])), m["fs_small"], badge_col, bold=True)

            h_str = f"+{t.rel_height_mm}mm" if t.rel_height_mm > 0 else (f"Z:{int(t.grip_z)}" if t.grip_z > 0 else "--")
            draw_text(canvas, f"方向:{t.yaw_deg}° 凸起:{h_str}",
                      (x1 + int(8 * m["s"]), ry1 + int(23 * m["s"])), m["fs_small"],
                      GuiTheme.WHITE if is_sel else GuiTheme.TEXT_SUB)

            draw_text(canvas, f"S:({int(t.robot_x)},{int(t.robot_y)},{int(t.robot_z)}) R:{int(t.robot_r)}°",
                      (x1 + int(8 * m["s"]), ry1 + int(41 * m["s"])), m["fs_small"],
                      GuiTheme.ACCENT if is_sel else GuiTheme.TEXT_MUTED)

            result_rows.append(((x1 + 4, ry1, x2 - 4, ry2), list_idx))

        cls._draw_pose_detail_box(canvas, m, rect, app_state)

    @classmethod
    def _draw_pose_detail_box(cls, canvas, m, rect, app_state):
        x1, _, x2, y2 = rect
        gh = int(185 * m["s"])
        gy1 = y2 - gh - int(6 * m["s"])
        cv2.rectangle(canvas, (x1 + int(4 * m["s"]), gy1), (x2 - int(4 * m["s"]), y2 - int(4 * m["s"])),
                      (16, 19, 24), -1)
        cv2.rectangle(canvas, (x1 + int(4 * m["s"]), gy1), (x2 - int(4 * m["s"]), y2 - int(4 * m["s"])),
                      GuiTheme.BORDER, 1)

        if app_state.targets and 0 <= app_state.sel_target < len(app_state.targets):
            t = app_state.targets[app_state.sel_target]
            title = f"抓取位姿详情 (目标 #{t.id} · {t.grade}级品)"
            draw_text(canvas, title, (x1 + int(12 * m["s"]), gy1 + int(8 * m["s"])),
                      m["fs_small"], GuiTheme.ACCENT, bold=True)
            
            yy = gy1 + int(32 * m["s"])
            line_h = int(18 * m["s"])
            
            # 世界空间坐标与夹爪姿态
            p_text = f"拾取中心: X={t.robot_x:.1f}  Y={t.robot_y:.1f}  Z={t.robot_z:.1f} mm"
            draw_text(canvas, p_text, (x1 + int(12 * m["s"]), yy), m["fs_body"], GuiTheme.TEXT)
            yy += line_h

            r_text = f"抓取姿态: Yaw={t.robot_r:.1f}° (主轴走向对齐)"
            draw_text(canvas, r_text, (x1 + int(12 * m["s"]), yy), m["fs_body"], (255, 200, 80))
            yy += line_h

            # 品质与几何特征
            q_text = f"几何品质: 直径={t.diam_mm:.1f}mm 长度={int(t.length_mm)}mm"
            draw_text(canvas, q_text, (x1 + int(12 * m["s"]), yy), m["fs_small"], GuiTheme.TEXT_SUB)
            yy += line_h

            # 标定基准源
            cal_map = {
                "tag_online": "AprilTag 实时在线解算基准",
                "tag_cached": "AprilTag 历史锁存基准",
                "hand_eye": "手工标定外参矩阵基准",
                "uncalibrated": "传送带物理几何相对基准",
            }
            src_desc = cal_map.get(t.calibration_source, t.calibration_source)
            s_text = f"坐标基准: {src_desc}"
            draw_text(canvas, s_text, (x1 + int(12 * m["s"]), yy), m["fs_small"], (100, 180, 255))
        else:
            note = "2D 预览无深度" if app_state.mode == "2d" else "未检出目标或无选中物料"
            draw_text(canvas, note, (x1 + int(12 * m["s"]), gy1 + int(14 * m["s"])),
                      m["fs_small"], GuiTheme.TEXT_MUTED)
