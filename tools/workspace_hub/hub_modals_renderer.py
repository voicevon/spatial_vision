# -*- coding: utf-8 -*-
"""
Workspace Hub 模态弹窗专用渲染模块
========================================
负责全系统所有独立模态弹窗与浮层的视觉绘制：
1. 模态脚手架底板 (全屏半透明遮罩 + 卡片底板 + 标题栏 + 关闭/保存/取消操作底栏)
2. 机构相对坐标系结构化表单弹窗 (Fixed Transform / AprilTag 动标绑定)
3. 3D ROI 空间物件结构化表单弹窗 (类别/坐标系下拉选择 + 中心/尺寸/旋转 3 轴有向长方体)
4. AprilTag 世界坐标物理锚点矩阵弹窗 (15 键软键盘 + 逐轴输入/已知状态指示)
5. 工位沙盒与生产机制业务架构说明弹窗 (自包含沙盒架构阐释卡片)
6. 下拉选择框浮层 (委托公共 render_dropdown_popup) 与数值输入卡片
"""

from typing import Any, Tuple, List, Optional
import cv2
import numpy as np

from src.ui.text_rendering import draw_text
from src.ui.gui_components import (
    GuiTheme,
    draw_dropdown_button,
    render_dropdown_popup,
    draw_rounded_rectangle
)
from tools.workspace_hub.hub_state import HubState


class HubModalsRenderer:
    """Workspace Hub 模态弹窗与浮层渲染器"""

    def __init__(self, parent_renderer: Any):
        self.r = parent_renderer

    # ==================== 模态通用脚手架 ====================

    def render_modal_scaffold(
        self,
        canvas: np.ndarray,
        state: HubState,
        title: str,
        border_col: Tuple[int, int, int] = (0, 240, 220),
        box_w: Optional[int] = None,
        box_h: Optional[int] = None,
        show_action_buttons: bool = True
    ) -> Tuple[int, int, int, int, Tuple[int, int]]:
        """
        统一绘制模态弹窗通用脚手架底板：
        全屏半透明暗色遮罩 + 主卡片底板 + 顶部标题栏 + [X] 关闭按钮 + 底部取消/保存操作栏
        返回: (mx, my, mw, mh, mpos)
        """
        from tools.workspace_hub.hub_renderer import (
            GEOM_MODAL_X, GEOM_MODAL_Y, GEOM_MODAL_W, GEOM_MODAL_H,
            GEOM_MODAL_CLOSE, GEOM_MODAL_SAVE, GEOM_MODAL_CANCEL
        )
        mw = box_w if box_w is not None else GEOM_MODAL_W
        mh = box_h if box_h is not None else GEOM_MODAL_H
        mx = (self.r.canvas_w - mw) // 2 if box_w is not None else GEOM_MODAL_X
        my = (self.r.canvas_h - mh) // 2 if box_h is not None else GEOM_MODAL_Y
        mpos = (state.mouse_x, state.mouse_y)

        # 1. 全屏半透明黑色遮罩
        mask = canvas.copy()
        cv2.rectangle(mask, (0, 0), (self.r.canvas_w, self.r.canvas_h), (0, 0, 0), -1)
        cv2.addWeighted(mask, 0.72, canvas, 0.28, 0, canvas)

        # 2. 弹窗主底板与边框 (双层科技线框)
        cv2.rectangle(canvas, (mx, my), (mx + mw, my + mh), (20, 25, 34), -1)
        cv2.rectangle(canvas, (mx, my), (mx + mw, my + mh), border_col, 2)
        cv2.rectangle(canvas, (mx + 3, my + 3), (mx + mw - 3, my + mh - 3), (45, 55, 75), 1)

        # 3. 顶部标题栏
        cv2.rectangle(canvas, (mx, my), (mx + mw, my + 44), (16, 20, 28), -1)
        cv2.line(canvas, (mx, my + 44), (mx + mw, my + 44), border_col, 1)
        cv2.circle(canvas, (mx + 18, my + 22), 5, border_col, -1)
        draw_text(canvas, title, (mx + 32, my + 13), font_size=15, color=GuiTheme.WHITE, bold=True)

        # 4. 右上角 [X] 关闭按钮
        self.r._draw_button(canvas, GEOM_MODAL_CLOSE, "X", mpos, theme_color=border_col)

        # 5. 底部操作按钮栏
        if show_action_buttons:
            cv2.line(canvas, (mx, my + mh - 50), (mx + mw, my + mh - 50), (45, 55, 70), 1)
            self.r._draw_button(canvas, GEOM_MODAL_CANCEL, "取消", mpos)
            self.r._draw_button(canvas, GEOM_MODAL_SAVE, "保存并生效", mpos, theme_color=border_col)

        return mx, my, mw, mh, mpos

    # ==================== 表单组件绘制辅助 ====================

    def draw_text_input(
        self,
        canvas: np.ndarray,
        rect: Tuple[int, int, int, int],
        text: str,
        mouse_pos: Tuple[int, int],
        text_col: Optional[Tuple[int, int, int]] = None
    ) -> bool:

        """统一绘制高可编辑感知的输入框 (深暗内凹底色 + 科技发光边框 + 铅笔修改图标 ✎)"""
        bx, by, bw, bh = rect
        mx, my = mouse_pos
        is_hover = (bx <= mx <= bx + bw and by <= my <= by + bh)

        bg_col = (14, 18, 25) if not is_hover else (22, 32, 44)
        border_col = (0, 255, 180) if is_hover else (45, 65, 75)
        if text_col is None:
            text_col = (0, 255, 220) if is_hover else (220, 235, 235)

        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), bg_col, -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), border_col, 2 if is_hover else 1)

        disp = text if text else "点击输入..."
        if len(disp) > 22:
            disp = disp[:20] + ".."
        draw_text(canvas, disp, (bx + 8, by + (bh - 16) // 2), font_size=12,
                  color=text_col, bold=is_hover)

        # 右侧绘制微型矢量铅笔图标 (45° 科技暗绿/亮青，平滑抗锯齿)
        px = bx + bw - 14
        py = by + bh // 2
        pen_col = (0, 255, 180) if is_hover else (80, 110, 120)
        cv2.line(canvas, (px - 4, py + 3), (px + 3, py - 4), pen_col, 2, cv2.LINE_AA)
        cv2.circle(canvas, (px - 5, py + 4), 1, pen_col, -1, cv2.LINE_AA)
        return is_hover

    def draw_vector3_input_row(
        self,
        canvas: np.ndarray,
        row_label: str,
        label_pos: Tuple[int, int],
        rects: List[Tuple[int, int, int, int]],
        axis_names: List[str],
        values: List[float],
        mouse_pos: Tuple[int, int]
    ):
        """统一绘制 3 轴向量输入行 (Label + 3 个数值输入卡片)"""
        draw_text(canvas, row_label, label_pos, font_size=13, color=GuiTheme.WHITE)
        for rect, axis_name, val in zip(rects, axis_names, values):
            self.draw_text_input(canvas, rect, f"{axis_name}: {val:.1f}", mouse_pos)

    def draw_custom_vector3_row(
        self,
        canvas: np.ndarray,
        row_label: str,
        label_pos: Tuple[int, int],
        rects: List[Tuple[int, int, int, int]],
        axis_names: List[str],
        values_str: List[str],
        mouse_pos: Tuple[int, int]
    ):
        """统一绘制 3 轴向量显示行 (支持未知状态自定义文字与色彩)"""
        draw_text(canvas, row_label, label_pos, font_size=13, color=GuiTheme.WHITE)
        for rect, axis_name, val_str in zip(rects, axis_names, values_str):
            col = (0, 210, 240) if "?" in val_str else None
            self.draw_text_input(canvas, rect, f"{axis_name}: {val_str}", mouse_pos, text_col=col)


    def draw_dropdown_trigger(
        self,
        canvas: np.ndarray,
        rect: Tuple[int, int, int, int],
        text: str,
        mouse_pos: Tuple[int, int],
        is_open: bool = False
    ) -> bool:
        """统一绘制现代下拉选择框触发条 (委托给公共 draw_dropdown_button 控件)"""
        bx, by, bw, bh = rect
        rect_pts = (bx, by, bx + bw, by + bh)
        return draw_dropdown_button(canvas, rect_pts, text, is_open, mouse_pos=mouse_pos, font_size=12)

    def get_dropdown_data(self, dd_type: str, state: HubState):
        """统一获取下拉框的布局矩形、当前选中值以及候选枚举列表"""
        from tools.workspace_hub.hub_renderer import GEOM_MODAL_X, GEOM_MODAL_Y
        mx_box, my_box = GEOM_MODAL_X, GEOM_MODAL_Y
        form_y = my_box + 56
        if dd_type == "frame_type":
            rect = (mx_box + 115, form_y + 40, 360, 28)
            cur_val = state.geometry.frame_modal_data.get("type", "fixed_transform")
            options = [
                ("fixed_transform", "固定刚体外参 (平移 + 旋转)"),
                ("tag_bound", "AprilTag 动标绑定 (动态跟踪)")
            ]
            return rect, cur_val, options

        if dd_type == "frame_parent":
            rect = (mx_box + 115, form_y + 80, 360, 28)
            cur_val = state.geometry.frame_modal_data.get("parent_frame_id", "world")
            cur_fid = state.geometry.frame_modal_data.get("frame_id")
            frames = state.geometry.get_coordinate_frames()
            options = [("world", "world (世界基准绝对原点)")]
            for f in frames:
                if f.frame_id != cur_fid and f.frame_id != "world":
                    options.append((f.frame_id, f"{f.frame_id} ({f.name})"))
            return rect, cur_val, options

        roi_form_y = my_box + 46
        if dd_type == "roi_category":
            rect = (mx_box + 110, roi_form_y + 36, 220, 28)
            cur_val = state.geometry.roi_modal_data.get("category", "belt")
            options = [
                ("belt", "同步带工作面 (belt)"),
                ("wheel", "驱动轮干涉区 (wheel)"),
                ("tray", "料盘工装区 (tray)"),
                ("general", "通用机构部件 (general)")
            ]
            return rect, cur_val, options

        if dd_type == "roi_frame":
            rect = (mx_box + 425, roi_form_y + 36, 225, 28)
            cur_val = state.geometry.roi_modal_data.get("frame_id", "world")
            frames = state.geometry.get_coordinate_frames()
            options = [("world", "world (世界基准绝对原点)")]
            for f in frames:
                if f.frame_id != "world":
                    options.append((f.frame_id, f"{f.frame_id} ({f.name})"))
            return rect, cur_val, options

        return None

    def render_active_dropdown(self, canvas: np.ndarray, state: HubState):
        """在弹窗顶层高亮绘制当前展开的下拉菜单浮层 (委托给公共 render_dropdown_popup 控件)"""
        dd_type = state.geometry.active_dropdown
        if not dd_type:
            return
        dd_data = self.get_dropdown_data(dd_type, state)
        if not dd_data:
            return
        (tx, ty, tw, th), cur_val, options = dd_data
        anchor_rect = (tx, ty, tx + tw, ty + th)
        render_dropdown_popup(canvas, anchor_rect, options, cur_val, item_h=30)

    # ==================== 5 大独立弹窗渲染 ====================

    def render_frame_modal(self, canvas: np.ndarray, state: HubState):
        """渲染机构相对坐标系结构化表单弹窗"""
        d = state.geometry.frame_modal_data
        title_prefix = "新建机构相对坐标系" if state.geometry.frame_modal_is_new else f"编辑坐标系: 【{d.get('name', '')}】"
        mx, my, mw, mh, mpos = self.render_modal_scaffold(canvas, state, title_prefix, border_col=(0, 240, 220))

        # 表单字段排布
        form_y = my + 56

        # 1. 坐标系名称与唯一标识行
        draw_text(canvas, "坐标系名称:", (mx + 24, form_y + 4), font_size=13, color=self.r.COLOR_GRAY)
        self.draw_text_input(canvas, (mx + 115, form_y, 220, 28), str(d.get("name", "")), mpos)

        draw_text(canvas, "唯一 ID:", (mx + 360, form_y + 4), font_size=13, color=self.r.COLOR_GRAY)
        self.draw_text_input(canvas, (mx + 430, form_y, 220, 28), str(d.get("frame_id", "")), mpos)

        # 2. 坐标系类型下拉框
        type_y = form_y + 40
        draw_text(canvas, "坐标系类型:", (mx + 24, type_y + 4), font_size=13, color=self.r.COLOR_GRAY)
        cur_type = d.get("type", "fixed_transform")
        type_label = "固定刚体外参 (平移 + 旋转)" if cur_type == "fixed_transform" else "AprilTag 动标绑定 (动态跟踪)"
        self.draw_dropdown_trigger(canvas, (mx + 115, type_y, 360, 28), type_label, mpos,
                                   is_open=(state.geometry.active_dropdown == "frame_type"))

        # 3. 挂载父坐标系下拉框
        parent_y = form_y + 80
        draw_text(canvas, "父坐标系:", (mx + 24, parent_y + 4), font_size=13, color=self.r.COLOR_GRAY)
        cur_parent = d.get("parent_frame_id", "world")
        frames = state.geometry.get_coordinate_frames()
        p_name = "世界基准绝对原点" if cur_parent == "world" else ""
        for f in frames:
            if f.frame_id == cur_parent:
                p_name = f.name
                break
        parent_label = f"[{cur_parent}] {p_name}" if p_name else f"[{cur_parent}]"
        self.draw_dropdown_trigger(canvas, (mx + 115, parent_y, 360, 28), parent_label, mpos,
                                   is_open=(state.geometry.active_dropdown == "frame_parent"))

        # 4. 几何参数根据类型切换
        param_y = form_y + 128
        cv2.rectangle(canvas, (mx + 20, param_y), (mx + mw - 20, param_y + 155), (28, 34, 46), -1)
        cv2.rectangle(canvas, (mx + 20, param_y), (mx + mw - 20, param_y + 155), (45, 55, 75), 1)

        if cur_type == "fixed_transform":
            from tools.workspace_hub.hub_renderer import (
                FRAME_PARAM_BTN_UNKNOWN, FRAME_PARAM_BTN_EDIT6D
            )
            draw_text(canvas, "固定外参变换矩阵 (相对于父级坐标系):", (mx + 32, param_y + 12), font_size=13, color=(0, 240, 220), bold=True)

            st = d.get("status", "unknown")
            prior_t = d.get("prior_translation_xyz_mm") or [None, None, None]
            prior_r = d.get("prior_rotation_rpy_deg") or [None, None, None]
            n_known = sum(1 for v in prior_t + prior_r if v is not None)
            is_unknown = (st == "unknown" or n_known == 0)

            # 右侧操作按钮
            unk_btn_col = (180, 100, 30) if is_unknown else (90, 60, 35)
            self.r._draw_button(canvas, FRAME_PARAM_BTN_UNKNOWN, "设为全未知", mpos, theme_color=unk_btn_col)
            self.r._draw_button(canvas, FRAME_PARAM_BTN_EDIT6D, "6DoF先验编辑", mpos, theme_color=(0, 200, 180))

            t = d.get("translation_xyz_mm", [0, 0, 0])
            r = d.get("rotation_rpy_deg", [0, 0, 0])

            # 状态提示条
            if is_unknown:
                st_text = "⚠ 外参待定 (未知) · 等待 BA 平差后通过绑定的标靶自动反推"
                st_col = (0, 210, 255)
            elif st == "calibrated":
                st_text = "● 外参已由视觉标定求解 (6/6 轴已解算)"
                st_col = (0, 230, 140)
            elif n_known == 6:
                st_text = "● 外参已全部人工指定 (6/6 轴已知先验)"
                st_col = (0, 230, 140)
            else:
                st_text = f"◐ 部分已知先验约束 ({n_known}/6 轴已知，其余待BA求解)"
                st_col = (255, 180, 50)
            draw_text(canvas, st_text, (mx + 32, param_y + 30), font_size=11, color=st_col)

            # 平移 X, Y, Z
            t_rects = [(mx + 125, param_y + 48, 130, 28), (mx + 265, param_y + 48, 130, 28), (mx + 405, param_y + 48, 130, 28)]
            t_disp = []
            for i in range(3):
                if st == "calibrated" and t is not None:
                    t_disp.append(f"{t[i]:.1f}")
                elif prior_t[i] is not None:
                    t_disp.append(f"{prior_t[i]:.1f}")
                else:
                    t_disp.append("? (待解)")
            self.draw_custom_vector3_row(canvas, "平移 (mm):", (mx + 32, param_y + 54), t_rects, ["X", "Y", "Z"], t_disp, mpos)

            # 旋转 Roll, Pitch, Yaw
            r_rects = [(mx + 125, param_y + 94, 130, 28), (mx + 265, param_y + 94, 130, 28), (mx + 405, param_y + 94, 130, 28)]
            r_disp = []
            for i in range(3):
                if st == "calibrated" and r is not None:
                    r_disp.append(f"{r[i]:.1f}")
                elif prior_r[i] is not None:
                    r_disp.append(f"{prior_r[i]:.1f}")
                else:
                    r_disp.append("? (待解)")
            self.draw_custom_vector3_row(canvas, "旋转 (°):", (mx + 32, param_y + 100), r_rects, ["Roll", "Pitch", "Yaw"], r_disp, mpos)
        else:

            draw_text(canvas, "AprilTag 动标绑定配置:", (mx + 32, param_y + 12), font_size=13, color=(0, 210, 255), bold=True)
            tid = d.get("tag_id", 0)
            off = d.get("offset_xyz_mm", [0, 0, 0])

            draw_text(canvas, "绑定的 AprilTag:", (mx + 32, param_y + 48), font_size=13, color=GuiTheme.WHITE)
            self.draw_text_input(canvas, (mx + 165, param_y + 42, 160, 28), f"Tag ID: #{tid}", mpos)
            draw_text(canvas, "(点击修改绑定的标靶编号)", (mx + 335, param_y + 48), font_size=11, color=self.r.COLOR_GRAY)

            off_rects = [(mx + 185, param_y + 92, 120, 28), (mx + 315, param_y + 92, 120, 28), (mx + 445, param_y + 92, 120, 28)]
            self.draw_vector3_input_row(canvas, "局部偏移 XYZ (mm):", (mx + 32, param_y + 98), off_rects, ["dx", "dy", "dz"], off, mpos)


        # 5. 顶层渲染活跃下拉浮层
        self.render_active_dropdown(canvas, state)

    def render_roi_modal(self, canvas: np.ndarray, state: HubState):
        """渲染 3D ROI 空间物件结构化表单弹窗 (已深度融入 Smart ROI 生产意图与工艺角色)"""
        from tools.workspace_hub.hub_renderer import (
            ROI_ROLE_BTN_SOURCE, ROI_ROLE_BTN_DEST, ROI_ROLE_BTN_KEEPOUT, ROI_ROLE_BTN_GENERAL,
            ROI_INTENT_BTN_PICK, ROI_INTENT_BTN_COUNT, ROI_INTENT_BTN_OCC, ROI_INTENT_BTN_GEN,
            ROI_BIND_SLOT_BTN, ROI_BIND_CAP_BTN, ROI_BIND_CONF_BTN
        )
        d = state.geometry.roi_modal_data
        title_prefix = "新建 3D Smart ROI 空间物件" if state.geometry.roi_modal_is_new else f"编辑 Smart ROI: 【{d.get('name', '')}】"
        mx, my, mw, mh, mpos = self.render_modal_scaffold(canvas, state, title_prefix, border_col=(0, 255, 180))

        # 表单字段排布
        form_y = my + 46

        # 1. 名称与唯一 ID (第一行并排)
        draw_text(canvas, "物件名称:", (mx + 24, form_y + 4), font_size=13, color=self.r.COLOR_GRAY)
        self.draw_text_input(canvas, (mx + 110, form_y, 220, 28), str(d.get("name", "")), mpos)

        draw_text(canvas, "唯一 ID:", (mx + 355, form_y + 4), font_size=13, color=self.r.COLOR_GRAY)
        self.draw_text_input(canvas, (mx + 425, form_y, 225, 28), str(d.get("roi_id", "")), mpos)

        # 2. 部件类别与所属坐标系下拉框 (第二行并排)
        cat_y = form_y + 36
        draw_text(canvas, "部件类别:", (mx + 24, cat_y + 4), font_size=13, color=self.r.COLOR_GRAY)
        cur_cat = d.get("category", "belt")
        cat_map = {
            "belt": "同步带工作面 (belt)",
            "wheel": "驱动轮干涉区 (wheel)",
            "tray": "料盘工装区 (tray)",
            "general": "通用机构部件 (general)"
        }
        cat_label = cat_map.get(cur_cat, f"{cur_cat}")
        self.draw_dropdown_trigger(canvas, (mx + 110, cat_y, 220, 28), cat_label, mpos,
                                   is_open=(state.geometry.active_dropdown == "roi_category"))

        draw_text(canvas, "挂载系:", (mx + 355, cat_y + 4), font_size=13, color=self.r.COLOR_GRAY)
        cur_frame = d.get("frame_id", "world")
        frames = state.geometry.get_coordinate_frames()
        f_name = "世界基准原点" if cur_frame == "world" else ""
        for f in frames:
            if f.frame_id == cur_frame:
                f_name = f.name
                break
        frame_label = f"[{cur_frame}] {f_name}" if f_name else f"[{cur_frame}]"
        self.draw_dropdown_trigger(canvas, (mx + 425, cat_y, 225, 28), frame_label, mpos,
                                   is_open=(state.geometry.active_dropdown == "roi_frame"))

        # 3. ★ Smart ROI 工业生产语义与工艺属性面板 (第三行大卡片)
        smart_y = form_y + 72
        smart_h = 138
        cv2.rectangle(canvas, (mx + 20, smart_y), (mx + mw - 20, smart_y + smart_h), (25, 31, 42), -1)
        cv2.rectangle(canvas, (mx + 20, smart_y), (mx + mw - 20, smart_y + smart_h), (48, 62, 85), 1)

        draw_text(canvas, "★ Smart ROI 工业生产语义与工艺属性 (产线感知与动作解耦):", (mx + 32, smart_y + 9), font_size=13, color=(0, 240, 220), bold=True)

        # 3.1 工艺角色 (Role)
        draw_text(canvas, "工艺角色:", (mx + 32, smart_y + 36), font_size=12, color=self.r.COLOR_GRAY)
        cur_role = d.get("role", "source")
        role_specs = [
            (ROI_ROLE_BTN_SOURCE, "source", "★ 进料源", (0, 255, 140)),
            (ROI_ROLE_BTN_DEST, "destination", "▼ 落料槽", (0, 220, 255)),
            (ROI_ROLE_BTN_KEEPOUT, "keepout", "⛔ 禁行区", (255, 80, 80)),
            (ROI_ROLE_BTN_GENERAL, "general", "◌ 通用区", (180, 190, 205)),
        ]
        for btn_rect, r_val, r_text, r_col in role_specs:
            is_active = (cur_role == r_val)
            theme = r_col if is_active else (60, 72, 90)
            self.r._draw_button(canvas, btn_rect, r_text, mpos, theme_color=theme, enabled=True)

        # 3.2 动作意图 (Target Intent)
        draw_text(canvas, "动作意图:", (mx + 32, smart_y + 70), font_size=12, color=self.r.COLOR_GRAY)
        cur_intent = d.get("target_intent", "pose_pick")
        intent_specs = [
            (ROI_INTENT_BTN_PICK, "pose_pick", "🎯 位姿抓取", (0, 255, 180)),
            (ROI_INTENT_BTN_COUNT, "piece_count", "🔢 根数统计", (255, 200, 60)),
            (ROI_INTENT_BTN_OCC, "occupancy", "📦 在席检测", (220, 140, 255)),
            (ROI_INTENT_BTN_GEN, "general", "⚙ 通用意图", (180, 190, 205)),
        ]
        for btn_rect, i_val, i_text, i_col in intent_specs:
            is_active = (cur_intent == i_val)
            theme = i_col if is_active else (60, 72, 90)
            self.r._draw_button(canvas, btn_rect, i_text, mpos, theme_color=theme, enabled=True)

        # 3.3 设备绑定与容量门限 (Binding)
        draw_text(canvas, "槽位绑定:", (mx + 32, smart_y + 104), font_size=12, color=self.r.COLOR_GRAY)
        binding = d.get("binding", {}) or {}
        slot_idx = int(binding.get("slot_index", 0))
        cap_max = int(binding.get("capacity_max", 20))
        min_conf = float(d.get("min_confidence", 0.3) or 0.3)

        slot_label = f"槽位: #{slot_idx} (切换)"
        cap_label = f"容量: {cap_max} 根"
        conf_label = f"门限: {min_conf:.2f}"

        slot_theme = (0, 220, 255) if cur_role == "destination" else (100, 115, 135)
        self.r._draw_button(canvas, ROI_BIND_SLOT_BTN, slot_label, mpos, theme_color=slot_theme)
        self.r._draw_button(canvas, ROI_BIND_CAP_BTN, cap_label, mpos, theme_color=(255, 180, 50))
        self.r._draw_button(canvas, ROI_BIND_CONF_BTN, conf_label, mpos, theme_color=(120, 190, 240))
        draw_text(canvas, "★ 生产路由调度直接读取此配置", (mx + 515, smart_y + 108), font_size=11, color=(140, 160, 185))

        # 4. 几何长方体参数区 (局部中心 + 尺寸 + 姿态)
        geom_y = smart_y + 148
        geom_h = 150
        cv2.rectangle(canvas, (mx + 20, geom_y), (mx + mw - 20, geom_y + geom_h), (28, 34, 46), -1)
        cv2.rectangle(canvas, (mx + 20, geom_y), (mx + mw - 20, geom_y + geom_h), (45, 55, 75), 1)

        draw_text(canvas, "3D 有向长方体空间定义 (在所属局部坐标系下):", (mx + 32, geom_y + 10), font_size=13, color=(0, 240, 220), bold=True)

        c = d.get("center_xyz_mm", [0, 0, 0])
        s = d.get("size_xyz_mm", [50, 50, 50])
        r = d.get("rotation_rpy_deg", [0, 0, 0])

        # 局部中心
        c_rects = [(mx + 155, geom_y + 34, 115, 26), (mx + 280, geom_y + 34, 115, 26), (mx + 405, geom_y + 34, 115, 26)]
        self.draw_vector3_input_row(canvas, "局部中心 (mm):", (mx + 32, geom_y + 39), c_rects, ["X", "Y", "Z"], c, mpos)

        # 尺寸长宽高 (强 Schema 约束)
        s_rects = [(mx + 155, geom_y + 70, 115, 26), (mx + 280, geom_y + 70, 115, 26), (mx + 405, geom_y + 70, 115, 26)]
        self.draw_vector3_input_row(canvas, "空间尺寸 (mm):", (mx + 32, geom_y + 75), s_rects, ["长 dx", "宽 dy", "高 dz"], s, mpos)
        draw_text(canvas, "★ 约束: 必须 > 0", (mx + 530, geom_y + 75), font_size=11,
                  color=(0, 255, 180) if all(x > 0 for x in s) else (0, 100, 255))

        # 微调姿态
        r_rects = [(mx + 155, geom_y + 106, 115, 26), (mx + 280, geom_y + 106, 115, 26), (mx + 405, geom_y + 106, 115, 26)]
        self.draw_vector3_input_row(canvas, "局部旋转 (°):", (mx + 32, geom_y + 111), r_rects, ["R", "P", "Y"], r, mpos)

        # 5. 顶层渲染活跃下拉浮层
        self.render_active_dropdown(canvas, state)

    def render_anchor_modal(self, canvas: np.ndarray, state: HubState):
        """渲染 AprilTag 物理锚点坐标编辑弹窗"""
        from tools.workspace_hub.hub_renderer import (
            WL_ANCHOR_X, WL_ANCHOR_Y, WL_ANCHOR_W, WL_ANCHOR_H,
            WL_ANCHOR_SAVE, WL_ANCHOR_CANCEL, WL_ANCHOR_DELETE,
            anchor_row_rect, anchor_clear_rect, anchor_padkey_rect, point_in_rect
        )
        MX, MY, MW, MH = WL_ANCHOR_X, WL_ANCHOR_Y, WL_ANCHOR_W, WL_ANCHOR_H
        mpos = (state.mouse_x, state.mouse_y)

        # 半透明黑色遮罩
        mask = canvas.copy()
        cv2.rectangle(mask, (0, 0), (self.r.canvas_w, self.r.canvas_h), (0, 0, 0), -1)
        cv2.addWeighted(mask, 0.6, canvas, 0.4, 0, canvas)

        # 弹窗底板
        cv2.rectangle(canvas, (MX, MY), (MX + MW, MY + MH), (20, 24, 32), -1)
        cv2.rectangle(canvas, (MX, MY), (MX + MW, MY + MH), (0, 200, 240), 2)
        cv2.rectangle(canvas, (MX + 3, MY + 3), (MX + MW - 3, MY + MH - 3), (40, 50, 66), 1)

        # 标题栏
        tag_id = state.whitelist.anchor_modal_tag
        draw_text(canvas, f"Tag #{tag_id:02d} 物理锚点坐标 (支持部分已知)", (MX + 18, MY + 12),
                  font_size=15, color=GuiTheme.WHITE, bold=True)
        n_known = sum(1 for b in state.whitelist.anchor_modal_known if b)
        if n_known == 3:
            status_desc, status_col = "完整锚点 (三轴已知)", (0, 230, 150)
        elif n_known >= 1:
            status_desc, status_col = "部分已知 (BA 自动平差求解未知轴)", (0, 220, 255)
        else:
            status_desc, status_col = "全未知 (保存即清除此锚点)", (150, 160, 175)
        draw_text(canvas, f"已知 {n_known}/3 轴: {status_desc}", (MX + 18, MY + 34),
                  font_size=12, color=status_col)

        # 三轴行: 轴名 + 值 + 已知/待BA求解 + [设为未知]
        for axis in range(3):
            rx, ry, rw, rh = anchor_row_rect(axis)
            hovered = point_in_rect(mpos[0], mpos[1], (rx, ry, rw, rh))
            is_sel = (state.whitelist.anchor_axis_sel == axis)
            is_known = bool(state.whitelist.anchor_modal_known[axis])
            row_bg = (30, 38, 52) if is_sel else ((36, 42, 54) if hovered else (26, 30, 40))
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), row_bg, -1)
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh),
                          (0, 200, 240) if is_sel else (60, 70, 88), 1)
            draw_text(canvas, "XYZ"[axis], (rx + 12, ry + 9), font_size=14,
                      color=(0, 255, 200) if is_known else (150, 160, 175), bold=True)
            if is_sel and state.whitelist.anchor_axis_buf:
                val_text, val_col = state.whitelist.anchor_axis_buf + "_", GuiTheme.WHITE
            elif is_known:
                val_text, val_col = f"{state.whitelist.anchor_modal_xyz[axis]:.1f}", GuiTheme.WHITE
            else:
                val_text, val_col = "? (未知)", (0, 220, 255)
            draw_text(canvas, val_text, (rx + 42, ry + 8), font_size=15, color=val_col, bold=True)
            know_text = "已知" if is_known else "待BA求解"
            draw_text(canvas, know_text, (rx + 180, ry + 10), font_size=12,
                       color=(0, 220, 140) if is_known else (0, 200, 230))
            # 行内 [设为未知] 按钮
            cx, cy, cw, ch = anchor_clear_rect(axis)
            chov = point_in_rect(mpos[0], mpos[1], (cx, cy, cw, ch))
            cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), (70, 46, 36) if chov else (58, 38, 30), -1)
            cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), (150, 90, 60), 1)
            draw_text(canvas, "设为未知", (cx + 8, cy + 7), font_size=11, color=(240, 180, 150))

        # 15 键键盘: 1~9 / . / 0 / -+/ 清空 / 退格 / 确认
        key_labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "0", "-/+", "清空", "退格", "确认"]
        for idx, label in enumerate(key_labels):
            kx, ky, kw, kh = anchor_padkey_rect(idx)
            hov = point_in_rect(mpos[0], mpos[1], (kx, ky, kw, kh))
            if label == "确认":
                bg = (26, 88, 60) if hov else (22, 70, 48)
                border, col = (0, 230, 150), (120, 255, 200)
            elif label in ("退格", "清空"):
                bg = (70, 46, 36) if hov else (58, 38, 30)
                border, col = (150, 90, 60), (230, 170, 140)
            else:
                bg = (40, 48, 64) if hov else (32, 38, 50)
                border, col = (90, 105, 135), (220, 228, 240)
            cv2.rectangle(canvas, (kx, ky), (kx + kw, ky + kh), bg, -1)
            cv2.rectangle(canvas, (kx, ky), (kx + kw, ky + kh), border, 1)
            est_w = 8 * len(label) if label.isascii() else 14 * len(label)
            draw_text(canvas, label, (kx + (kw - est_w) // 2, ky + 11), font_size=13, color=col, bold=True)

        # 底部: 保存 / 取消 / 清除锚点
        self.r._draw_button(canvas, WL_ANCHOR_SAVE, "保存", mpos)
        self.r._draw_button(canvas, WL_ANCHOR_CANCEL, "取消", mpos)
        self.r._draw_button(canvas, WL_ANCHOR_DELETE, "清除锚点", mpos, theme_color=(180, 60, 60))

    def render_pose6d_modal(self, canvas: np.ndarray, state: HubState):
        """渲染 6DoF 外参位姿与先验约束独立模态编辑器"""
        from tools.workspace_hub.hub_renderer import (
            P6_MODAL_W, P6_MODAL_H, P6_MODAL_X, P6_MODAL_Y,
            P6_BTN_UNKNOWN_ALL, P6_BTN_KNOWN_ALL, P6_BTN_PLANAR,
            P6_PAD_LABELS, pose6d_row_rect, pose6d_clear_rect, pose6d_padkey_rect,
            P6_BTN_SAVE, P6_BTN_CANCEL, point_in_rect
        )
        MX, MY, MW, MH = P6_MODAL_X, P6_MODAL_Y, P6_MODAL_W, P6_MODAL_H
        mpos = (state.mouse_x, state.mouse_y)

        # 1. 半透明黑色遮罩
        mask = canvas.copy()
        cv2.rectangle(mask, (0, 0), (self.r.canvas_w, self.r.canvas_h), (0, 0, 0), -1)
        cv2.addWeighted(mask, 0.65, canvas, 0.35, 0, canvas)

        # 2. 弹窗底板与科技边框
        cv2.rectangle(canvas, (MX, MY), (MX + MW, MY + MH), (20, 25, 34), -1)
        cv2.rectangle(canvas, (MX, MY), (MX + MW, MY + MH), (0, 230, 200), 2)
        cv2.rectangle(canvas, (MX + 3, MY + 3), (MX + MW - 3, MY + MH - 3), (40, 52, 68), 1)

        # 3. 标题与状态提示
        fid = state.geometry.frame_modal_data.get("name") or state.geometry.frame_modal_data.get("frame_id", "坐标系")
        draw_text(canvas, f"编辑 6DoF 外参位姿与约束: 【{fid}】", (MX + 20, MY + 14),
                  font_size=15, color=GuiTheme.WHITE, bold=True)

        n_known = sum(1 for b in state.geometry.pose6d_modal_known if b)
        if n_known == 0:
            st_desc, st_col = "全未知模式 (外参全由视觉标靶经 BA 平差自动反推)", (0, 210, 255)
        elif n_known == 6:
            st_desc, st_col = "全已知模式 (6 自由度全部人工确知硬约束)", (0, 230, 140)
        else:
            st_desc, st_col = f"部分已知模式 ({n_known}/6 轴已知约束，其余由算法最优化解算)", (255, 180, 50)
        draw_text(canvas, f"当前状态: {st_desc}", (MX + 20, MY + 38), font_size=12, color=st_col)

        # 4. 快捷预设按钮组
        self.r._draw_button(canvas, P6_BTN_UNKNOWN_ALL, "设为全未知 (BA反推)", mpos, theme_color=(180, 100, 30))
        self.r._draw_button(canvas, P6_BTN_KNOWN_ALL, "设为全已知", mpos, theme_color=(30, 120, 90))
        self.r._draw_button(canvas, P6_BTN_PLANAR, "水平面约束 (Roll=0°, Pitch=0°)", mpos, theme_color=(40, 90, 140))

        # 5. 分栏标题
        draw_text(canvas, "平移维度 (Translation mm):", (MX + 24, MY + 98), font_size=12, color=(160, 180, 205), bold=True)
        draw_text(canvas, "旋转维度 (Rotation deg):", (MX + 316, MY + 98), font_size=12, color=(160, 180, 205), bold=True)

        # 6. 6 轴位姿行
        axis_names = [
            "X (前向)", "Y (横向)", "Z (垂向)",
            "Roll (翻滚)", "Pitch (俯仰)", "Yaw (偏航)"
        ]
        axis_units = ["mm", "mm", "mm", "°", "°", "°"]

        for axis in range(6):
            rx, ry, rw, rh = pose6d_row_rect(axis)
            is_sel = (state.geometry.pose6d_modal_axis_sel == axis)
            is_known = bool(state.geometry.pose6d_modal_known[axis])
            hov = point_in_rect(mpos[0], mpos[1], (rx, ry, rw, rh))

            row_bg = (34, 44, 58) if is_sel else ((32, 38, 48) if hov else (24, 28, 36))
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), row_bg, -1)
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (0, 230, 200) if is_sel else (55, 68, 85), 1)

            # 轴名称
            draw_text(canvas, axis_names[axis], (rx + 10, ry + 11), font_size=12,
                      color=(0, 255, 200) if is_known else (150, 165, 185), bold=True)

            # 数值或状态
            if is_sel and state.geometry.pose6d_modal_axis_buf:
                val_text, val_col = state.geometry.pose6d_modal_axis_buf + "_", GuiTheme.WHITE
            elif is_known:
                val_text, val_col = f"{state.geometry.pose6d_modal_vals[axis]:.1f} {axis_units[axis]}", GuiTheme.WHITE
            else:
                val_text, val_col = "? (待BA求解)", (0, 210, 255)
            draw_text(canvas, val_text, (rx + 95, ry + 11), font_size=13, color=val_col, bold=True)

            # 行内按钮 [设未知] / [设已知]
            cx, cy, cw, ch = pose6d_clear_rect(axis)
            chov = point_in_rect(mpos[0], mpos[1], (cx, cy, cw, ch))
            if is_known:
                btn_bg = (70, 46, 36) if chov else (58, 38, 30)
                btn_border = (160, 90, 60)
                btn_text = "设未知"
                btn_col = (240, 180, 150)
            else:
                btn_bg = (26, 68, 50) if chov else (22, 54, 40)
                btn_border = (0, 200, 140)
                btn_text = "设已知"
                btn_col = (140, 240, 190)

            cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), btn_bg, -1)
            cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), btn_border, 1)
            draw_text(canvas, btn_text, (cx + 8, cy + 6), font_size=11, color=btn_col)

        # 7. 15 键软键盘
        for idx, label in enumerate(P6_PAD_LABELS):
            kx, ky, kw, kh = pose6d_padkey_rect(idx)
            hov = point_in_rect(mpos[0], mpos[1], (kx, ky, kw, kh))
            if label == "确认":
                bg = (26, 92, 64) if hov else (22, 74, 52)
                border, col = (0, 230, 150), (140, 255, 210)
            elif label in ("退格", "清空"):
                bg = (72, 48, 38) if hov else (60, 40, 32)
                border, col = (160, 95, 65), (240, 175, 145)
            else:
                bg = (42, 52, 68) if hov else (32, 40, 52)
                border, col = (95, 110, 140), (220, 230, 245)
            cv2.rectangle(canvas, (kx, ky), (kx + kw, ky + kh), bg, -1)
            cv2.rectangle(canvas, (kx, ky), (kx + kw, ky + kh), border, 1)
            est_w = 8 * len(label) if label.isascii() else 14 * len(label)
            draw_text(canvas, label, (kx + (kw - est_w) // 2, ky + 10), font_size=13, color=col, bold=True)

        # 8. 底部操作按钮
        self.r._draw_button(canvas, P6_BTN_SAVE, "确认并应用", mpos, theme_color=(0, 210, 160))
        self.r._draw_button(canvas, P6_BTN_CANCEL, "取消", mpos)

    def render_help_modal(self, canvas: np.ndarray, state: HubState):
        """渲染工位沙盒与生产机制业务架构说明弹窗"""

        from tools.workspace_hub.hub_renderer import HELP_MODAL_W, HELP_MODAL_H
        modal_w, modal_h = HELP_MODAL_W, HELP_MODAL_H
        mx = (self.r.canvas_w - modal_w) // 2
        my = (self.r.canvas_h - modal_h) // 2
        mpos = (state.mouse_x, state.mouse_y)

        # 半透明黑色遮罩
        mask = canvas.copy()
        cv2.rectangle(mask, (0, 0), (self.r.canvas_w, self.r.canvas_h), (0, 0, 0), -1)
        cv2.addWeighted(mask, 0.72, canvas, 0.28, 0, canvas)

        cv2.rectangle(canvas, (mx, my), (mx + modal_w, my + modal_h), (20, 24, 32), -1)
        cv2.rectangle(canvas, (mx, my), (mx + modal_w, my + modal_h), (0, 220, 160), 2)
        cv2.rectangle(canvas, (mx + 4, my + 4), (mx + modal_w - 4, my + modal_h - 4), (40, 50, 66), 1)

        # 标题栏
        cv2.rectangle(canvas, (mx, my), (mx + modal_w, my + 54), (16, 20, 28), -1)
        cv2.line(canvas, (mx, my + 54), (mx + modal_w, my + 54), self.r.COLOR_BORDER, 1)

        cv2.circle(canvas, (mx + 24, my + 27), 6, self.r.COLOR_GOLD, -1)
        draw_text(canvas, "★ 业务架构解析: Workspace 工位沙盒与生产体系", (mx + 38, my + 15),
                  font_size=17, color=GuiTheme.WHITE, bold=True)

        # 右上角 [X] 关闭按钮
        self.r._draw_button(canvas, (mx + modal_w - 116, my + 11, 100, 32), "[X] 关闭 [H]", mpos)

        # 4 条架构阐释卡片
        intro_text = "在工业机器视觉与机械臂抓取工程中，各工位实行完全自包含的【物理沙盒】机制："
        draw_text(canvas, intro_text, (mx + 30, my + 68), font_size=14, color=(0, 240, 220))

        sections = [
            ("1. 独立工位安全沙盒 (Sandbox Isolation)",
             "每个工位（如“1号机台”、“现场工位A”）均为独立物理沙盒，拥有专属标定照片集、生产照片集与平差结果，互不干扰。",
             (0, 255, 180)),

            ("2. 工位专属生产地图 (Per-Workspace Production Map)",
             "系统无全局唯一地图。每个工位均自包含经过严格平差的高精度几何地图 (tags_map.yaml)，作为该工位专属的空间几何基准。",
             (0, 220, 255)),

            ("3. 地图原子持久化与安全备份 (Safe Atomic Persistence)",
             "平差优化完成后，直接原子持久化写入当前工位沙盒内，并自动保留带时间戳的 .bak 历史备份，杜绝跨工位数据污染与误操作。",
             self.r.COLOR_GOLD),

            ("4. 生产作业按需指定工位 (Production Anchored to Workspace)",
             "实际流水线作业时，生产服务直接对接目标工位，读取本工位专属的几何标定矩阵与白名单，实现按工位精准受控作业！",
             (160, 255, 120))
        ]

        sy = my + 98
        for title, desc, col in sections:
            cv2.rectangle(canvas, (mx + 28, sy), (mx + modal_w - 28, sy + 74), (25, 30, 40), -1)
            cv2.rectangle(canvas, (mx + 28, sy), (mx + modal_w - 28, sy + 74), (44, 52, 68), 1)
            cv2.rectangle(canvas, (mx + 28, sy), (mx + 32, sy + 74), col, -1)

            draw_text(canvas, title, (mx + 42, sy + 8), font_size=14, color=col, bold=True)
            d1 = desc[:48]
            d2 = desc[48:96]
            draw_text(canvas, d1, (mx + 42, sy + 30), font_size=12, color=GuiTheme.WHITE)
            if d2:
                draw_text(canvas, d2, (mx + 42, sy + 48), font_size=12, color=self.r.COLOR_GRAY)
            sy += 82

        footer_y = my + modal_h - 36
        draw_text(canvas, "快捷提示: 鼠标点击右上角 [X]、点击遮罩或直接按键盘 [ESC / H] 即可秒级关闭！",
                  (mx + 32, footer_y), font_size=13, color=self.r.COLOR_GRAY)

    # ==================== 标靶物理边长专属模态弹窗 ====================

    def render_marker_size_modal(self, canvas: np.ndarray, state: HubState):
        """绘制标靶物理边长专属核准与编辑模态弹窗"""
        from tools.workspace_hub.hub_renderer import (
            MS_MODAL_X, MS_MODAL_Y, MS_MODAL_W, MS_MODAL_H,
            MS_BTN_SAVE, MS_BTN_CANCEL, MS_PAD_LABELS, ms_padkey_rect
        )

        mx, my, mw, mh = MS_MODAL_X, MS_MODAL_Y, MS_MODAL_W, MS_MODAL_H
        mpos = (state.mouse_x, state.mouse_y)

        # 1. 半透明黑色遮罩
        mask = canvas.copy()
        cv2.rectangle(mask, (0, 0), (self.r.canvas_w, self.r.canvas_h), (0, 0, 0), -1)
        cv2.addWeighted(mask, 0.72, canvas, 0.28, 0, canvas)

        # 2. 弹窗底板与双层边框
        cv2.rectangle(canvas, (mx, my), (mx + mw, my + mh), (20, 24, 32), -1)
        cv2.rectangle(canvas, (mx, my), (mx + mw, my + mh), (0, 220, 160), 2)
        cv2.rectangle(canvas, (mx + 3, my + 3), (mx + mw - 3, my + mh - 3), (40, 52, 70), 1)

        # 3. 标题栏
        cv2.rectangle(canvas, (mx, my), (mx + mw, my + 44), (16, 20, 28), -1)
        cv2.line(canvas, (mx, my + 44), (mx + mw, my + 44), self.r.COLOR_BORDER, 1)
        cv2.circle(canvas, (mx + 20, my + 22), 5, self.r.COLOR_GOLD, -1)
        draw_text(canvas, "★ 核准 / 设置标靶物理边长 (Tag Marker Size)", (mx + 34, my + 13),
                  font_size=15, color=GuiTheme.WHITE, bold=True)

        # 4. 尺度基准严谨性说明
        intro_txt = "标靶物理边长是空间反投影尺度的唯一基准 (Scale Datum)，请输入真实名义边长 (mm)："
        draw_text(canvas, intro_txt, (mx + 24, my + 54), font_size=11, color=(160, 180, 200))

        # 5. 数值输入框 (大字体呈现 + 闪烁光标)
        bx, by, bw, bh = mx + 24, my + 76, mw - 48, 44
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (12, 16, 22), -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (0, 240, 220), 2)

        buf_str = state.whitelist.marker_size_buf or ""
        disp_txt = f"{buf_str}_ mm" if buf_str else "请输入边长 (如 35.5) mm"
        val_col = (0, 255, 220) if buf_str else (100, 115, 130)
        draw_text(canvas, disp_txt, (bx + 14, by + 11), font_size=18, color=val_col, bold=True)

        # 6. 16 键数字与快捷尺寸软键盘
        for idx, lbl in enumerate(MS_PAD_LABELS):
            kx, ky, kw, kh = ms_padkey_rect(idx)
            is_hov = (kx <= mpos[0] <= kx + kw and ky <= mpos[1] <= ky + kh)

            # 预设尺寸按键使用特殊底色高亮
            if lbl in ("35.5", "50.0", "40.0"):
                key_bg = (32, 48, 42) if is_hov else (22, 34, 30)
                key_border = (0, 255, 180) if is_hov else (0, 180, 130)
                key_text_col = (0, 255, 200)
            elif lbl in ("退格", "清空"):
                key_bg = (40, 32, 32) if is_hov else (28, 22, 24)
                key_border = (255, 100, 100) if is_hov else (120, 50, 50)
                key_text_col = (255, 160, 160)
            else:
                key_bg = (30, 36, 48) if is_hov else (22, 26, 34)
                key_border = (0, 200, 240) if is_hov else (45, 55, 70)
                key_text_col = GuiTheme.WHITE

            cv2.rectangle(canvas, (kx, ky), (kx + kw, ky + kh), key_bg, -1)
            cv2.rectangle(canvas, (kx, ky), (kx + kw, ky + kh), key_border, 1)

            btn_label = f"[{lbl}]" if lbl in ("35.5", "50.0", "40.0") else lbl
            tx = kx + max(6, (kw - len(btn_label) * 9) // 2)
            draw_text(canvas, btn_label, (tx, ky + 9), font_size=13, color=key_text_col, bold=is_hov)

        # 7. 底部操作栏: [取消 (ESC)] 与 [保存并核准 (Enter)]
        self.r._draw_button(canvas, MS_BTN_CANCEL, "取消 (ESC)", mpos)
        self.r._draw_button(canvas, MS_BTN_SAVE, "保存并核准边长 (Enter)", mpos, theme_color=(0, 220, 160))
