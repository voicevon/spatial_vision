# -*- coding: utf-8 -*-
"""
Workspace Hub 坐标系与 3D ROI 空间页面渲染器 (FramesRoisPageRenderer)
===================================================================
负责：
1. 工位多坐标系拓扑树与 Smart ROI 空间卡片 (Coordinate Frames & ROI Spaces 概览页)
2. 机构位姿与 Tag 专属分段放行矩阵页 (位姿参数/Tag 绝对尺度名义边长/10-Slot Tag 分配)
3. 坐标系专属 3D ROI 空间物件列表页 (3D OBB 物件卡片、Smart ROI 语义徽章、滚动翻页条)
"""

import os
import cv2
import numpy as np
from typing import Any

from src.utils.gui_components import draw_text, put_text
from tools.workspace_hub.hub_state import HubState


class FramesRoisPageRenderer:
    """坐标系与 3D ROI 空间页面渲染器"""

    def __init__(self, parent_renderer: Any):
        self.r = parent_renderer

    def render_frames_rois(self, canvas: np.ndarray, state: HubState, sc: Any):
        """页签: 坐标系与 3D ROI 空间管理页 (x: 340~960, y: 50~670)"""
        from tools.workspace_hub.hub_renderer import (
            frame_btn_add_rect, frame_row_edit_rect, frame_row_del_rect,
            roi_btn_add_rect, roi_row_edit_rect, roi_row_del_rect
        )

        box_x, box_y, box_w, box_h = 340, 50, 620, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (18, 22, 28), -1)
        mpos = (state.mouse_x, state.mouse_y)

        # 栏目标题与右上角操作按钮
        draw_text(canvas, "工位多坐标系拓扑与 3D ROI 空间", (box_x + 16, box_y + 14), font_size=16, color=self.r.COLOR_WHITE, bold=True)
        self.r._draw_button(canvas, (760, box_y + 8, 86, 30), "刷新", mpos)
        self.r._draw_button(canvas, (854, box_y + 8, 86, 30), "打开目录", mpos)

        if not sc:
            draw_text(canvas, "请在左侧选择或新建工位", (box_x + 180, box_y + 280), font_size=18, color=self.r.COLOR_GRAY)
            return

        card_x = box_x + 12
        card_w = box_w - 24  # 596px

        # -------------------------------------------------------------------------
        # 1. 机构多坐标系拓扑树卡片 (Coordinate Frame Tree)
        # -------------------------------------------------------------------------
        c1_y = box_y + 46
        c1_h = 260
        cv2.rectangle(canvas, (card_x, c1_y), (card_x + card_w, c1_y + c1_h), (24, 28, 38), -1)
        cv2.rectangle(canvas, (card_x, c1_y), (card_x + card_w, c1_y + c1_h), (42, 50, 68), 1)

        frames = state.geometry.get_coordinate_frames()
        draw_text(canvas, f"机构多坐标系拓扑树 (Coordinate Frames · 共 {len(frames)} 个)",
                  (card_x + 16, c1_y + 11), font_size=14, color=(0, 240, 220), bold=True)
        self.r._draw_button(canvas, frame_btn_add_rect(), "+ 新增坐标系", mpos)
        cv2.line(canvas, (card_x + 14, c1_y + 34), (card_x + card_w - 14, c1_y + 34), self.r.COLOR_BORDER, 1)

        coord_mgr = state.geometry.coord_mgr
        row_y = c1_y + 40
        row_h = 66
        gap_y = 6

        # 显示坐标系列表 (最多前 3 个)
        for i, frame in enumerate(frames[:3]):
            fy = row_y + i * (row_h + gap_y)
            cv2.rectangle(canvas, (card_x + 10, fy), (card_x + card_w - 10, fy + row_h), (30, 36, 48), -1)
            cv2.rectangle(canvas, (card_x + 10, fy), (card_x + card_w - 10, fy + row_h), (50, 60, 80), 1)

            # 类型指示色条
            if frame.type == "world":
                bar_col = (0, 255, 180)
                type_name = "绝对世界系"
            elif frame.type == "tag_bound":
                bar_col = (0, 210, 255)
                type_name = f"动标Tag #{frame.tag_id}"
            else:
                bar_col = (255, 180, 50)
                type_name = "固定外参"
            cv2.rectangle(canvas, (card_x + 10, fy), (card_x + 14, fy + row_h), bar_col, -1)

            # 第一行：名称 + 标识 + 类型徽章 + 状态
            draw_text(canvas, f"{frame.name} ({frame.frame_id})", (card_x + 22, fy + 8), font_size=13, color=self.r.COLOR_WHITE, bold=True)
            cv2.rectangle(canvas, (card_x + 240, fy + 6), (card_x + 335, fy + 26), (20, 26, 36), -1)
            cv2.rectangle(canvas, (card_x + 240, fy + 6), (card_x + 335, fy + 26), bar_col, 1)
            draw_text(canvas, type_name, (card_x + 246, fy + 8), font_size=11, color=bar_col, bold=True)

            is_res = True
            if coord_mgr:
                _, is_res = coord_mgr.get_frame_to_world(frame.frame_id)
            if frame.type == "fixed_transform":
                status_text = "● 已标定" if is_res else ("⚠ 待标定" if getattr(frame, "status", "") == "unknown" else "⚠ 未求解")
                status_col = (0, 230, 140) if is_res else (0, 180, 255)
            elif frame.type == "tag_bound":
                status_text = "● 已解出" if is_res else "⚠ 动标未解"
                status_col = (0, 230, 140) if is_res else (0, 180, 255)
            else:
                status_text = "● 原点基准"
                status_col = (0, 230, 140)
            draw_text(canvas, status_text, (card_x + 350, fy + 8), font_size=11, color=status_col, bold=True)

            # 右侧操作按钮
            self.r._draw_button(canvas, frame_row_edit_rect(i), "编辑", mpos)
            if frame.frame_id != "world":
                self.r._draw_button(canvas, frame_row_del_rect(i), "删除", mpos, theme_color=(180, 60, 60))

            # 第二行：父坐标系与几何参数详情
            parent_info = f"父级: {frame.parent_frame_id or '根节点'}"
            if frame.type == "world":
                param_info = "原点基准 [0, 0, 0] mm | 无旋转"
            elif frame.type == "tag_bound":
                param_info = f"绑定 Tag {frame.tag_id} | 偏移 XYZ: {frame.offset_xyz_mm} mm"
            else:
                if frame.translation_xyz_mm is not None and frame.rotation_rpy_deg is not None:
                    t = [round(float(x), 1) for x in frame.translation_xyz_mm]
                    r = [round(float(x), 1) for x in frame.rotation_rpy_deg]
                    rmse_str = f" (RMSE: {frame.calibration_metrics.get('rmse_mm', 0):.2f}mm)" if getattr(frame, "calibration_metrics", None) and "rmse_mm" in frame.calibration_metrics else ""
                    param_info = f"平移: {t} mm | 旋转: {r}°{rmse_str}"
                else:
                    spec_desc = ""
                    if getattr(frame, "calibration_spec", None):
                        spec_method = frame.calibration_spec.get("method", "")
                        spec_desc = f" [{spec_method}]"
                    param_info = f"外参待解{spec_desc} (BA后自动标定)"

            draw_text(canvas, f"{parent_info}  |  {param_info}", (card_x + 22, fy + 38), font_size=11, color=(160, 175, 195))

        # -------------------------------------------------------------------------
        # 2. 3D ROI 空间物件集合卡片 (ROI Space Collection)
        # -------------------------------------------------------------------------
        c2_y = c1_y + c1_h + 10
        c2_h = 285
        cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (24, 28, 38), -1)
        cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (42, 50, 68), 1)

        rois = state.geometry.get_roi_spaces()
        draw_text(canvas, f"3D ROI 空间物件集合 (3D OBB Objects · 共 {len(rois)} 个)",
                  (card_x + 16, c2_y + 11), font_size=14, color=(0, 255, 180), bold=True)
        self.r._draw_button(canvas, roi_btn_add_rect(), "+ 新增 ROI", mpos)
        cv2.line(canvas, (card_x + 14, c2_y + 34), (card_x + card_w - 14, c2_y + 34), self.r.COLOR_BORDER, 1)

        roi_row_y = c2_y + 40
        roi_row_h = 70
        roi_gap_y = 6

        if not rois:
            draw_text(canvas, "当前工位尚未定义任何 3D ROI 物件 (可点击右上角 [+ 新增 ROI] 结构化配置)",
                      (card_x + 36, c2_y + 90), font_size=13, color=self.r.COLOR_GRAY)
        else:
            roi_mgr = state.geometry.roi_mgr

            for i, roi in enumerate(rois[:3]):
                ry = roi_row_y + i * (roi_row_h + roi_gap_y)
                cv2.rectangle(canvas, (card_x + 10, ry), (card_x + card_w - 10, ry + roi_row_h), (30, 36, 48), -1)
                cv2.rectangle(canvas, (card_x + 10, ry), (card_x + card_w - 10, ry + roi_row_h), (50, 60, 80), 1)

                # ROI 颜色标识
                rgb = roi.visual_color_rgb or [0, 255, 128]
                bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
                cv2.rectangle(canvas, (card_x + 10, ry), (card_x + 14, ry + roi_row_h), bgr, -1)

                # 第一行：名称 + 标识 + 类别徽章 + 挂载坐标系
                draw_text(canvas, f"{roi.name} ({roi.roi_id})", (card_x + 22, ry + 8), font_size=13, color=self.r.COLOR_WHITE, bold=True)
                # 类别徽章
                cv2.rectangle(canvas, (card_x + 240, ry + 6), (card_x + 315, ry + 26), (20, 26, 36), -1)
                cv2.rectangle(canvas, (card_x + 240, ry + 6), (card_x + 315, ry + 26), bgr, 1)
                cat_map = {"belt": "同步带", "wheel": "驱动轮", "tray": "料盘", "general": "通用"}
                cat_label = f"[{cat_map.get(roi.category, roi.category)}]"
                # Smart ROI 角色指示
                r_role = getattr(roi, "role", "source") or "source"
                slot_info = f"#{roi.binding.get('slot_index', 0)}" if getattr(roi, "binding", None) and "slot_index" in roi.binding else ""
                role_labels = {
                    "source": ("★进料源", (0, 255, 140)),
                    "destination": (f"▼落料{slot_info}", (0, 220, 255)),
                    "keepout": ("⛔禁行区", (255, 80, 80)),
                    "general": ("◌通用", (180, 180, 180)),
                }
                role_txt, role_col = role_labels.get(r_role, ("◌通用", (180, 180, 180)))
                draw_text(canvas, f"所属: {roi.frame_id}  |  {role_txt}", (card_x + 325, ry + 8), font_size=11, color=role_col, bold=True)

                # 右侧操作按钮
                self.r._draw_button(canvas, roi_row_edit_rect(i), "编辑", mpos)
                self.r._draw_button(canvas, roi_row_del_rect(i), "删除", mpos, theme_color=(180, 60, 60))

                # 第二行：局部几何尺寸
                c_str = [round(float(x), 1) for x in roi.center_xyz_mm]
                s_str = [round(float(x), 1) for x in roi.size_xyz_mm]
                local_info = f"局部中心: {c_str} | 尺寸(长宽高): {s_str[0]}x{s_str[1]}x{s_str[2]} mm"

                # 第三行：世界坐标系求解
                world_info = "世界 OBB: 未就绪"
                if roi_mgr and coord_mgr:
                    obb = roi_mgr.get_roi_world_obb(roi.roi_id, coord_mgr)
                    if obb and obb.get("is_resolved"):
                        cw = [round(float(x), 1) for x in obb["center_world"]]
                        world_info = f"世界中心: {cw} mm | 8顶点OBB已就绪"
                    elif obb:
                        world_info = "世界 OBB: 依附坐标系动标未就绪"

                draw_text(canvas, local_info, (card_x + 22, ry + 32), font_size=11, color=(150, 165, 185))
                draw_text(canvas, world_info, (card_x + 22, ry + 50), font_size=11, color=(0, 230, 160) if "已就绪" in world_info else self.r.COLOR_GOLD)

        # -------------------------------------------------------------------------
        # 3. 底部快捷提示
        # -------------------------------------------------------------------------
        hint_y = c2_y + c2_h + 4
        draw_text(canvas, "提示: 点击右上角 [+ 新增] 或各行 [编辑] 可直接在 GUI 中配置，修改后即时保存生效",
                  (box_x + 18, hint_y), font_size=12, color=self.r.COLOR_GRAY)

    def render_frame_pose_tags(self, canvas: np.ndarray, state: HubState, ws: Any):
        """坐标系专属页签 1: 机构参数与 Tag 分段管理 (x: 340~960, y: 50~670)"""
        from tools.workspace_hub.hub_renderer import (
            FRAME_EDIT_POSE_BTN, FRAME_TAG_EDIT_SIZE_BTN, frame_tag_chip_rect
        )

        box_x, box_y, box_w, box_h = 340, 50, self.r.canvas_w - 340, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (18, 22, 28), -1)
        mpos = (state.mouse_x, state.mouse_y)

        cur_frame = state.geometry.get_selected_frame()
        if not cur_frame:
            draw_text(canvas, "未选择任何机构坐标系", (box_x + 180, box_y + 280), font_size=18, color=self.r.COLOR_GRAY)
            return

        # 1. 顶部编辑按钮
        self.r._draw_button(canvas, FRAME_EDIT_POSE_BTN, "编辑", mpos)

        card_x = box_x + 16
        card_w = box_w - 32

        # 2. 上部卡片: 机构位姿与空间拓扑 (紧凑精致)
        c1_y = box_y + 40
        c1_h = 134
        cv2.rectangle(canvas, (card_x, c1_y), (card_x + card_w, c1_y + c1_h), (24, 28, 38), -1)
        cv2.rectangle(canvas, (card_x, c1_y), (card_x + card_w, c1_y + c1_h), (42, 52, 70), 1)

        is_world_datum = (cur_frame.type == "world" or cur_frame.frame_id == "world" or not cur_frame.parent_frame_id)
        if is_world_datum:
            type_desc = "工位绝对世界基准 (world)"
        elif cur_frame.type == "fixed_transform":
            type_desc = "固定刚体外参 (fixed_transform)"
        else:
            type_desc = "AprilTag 动标绑定 (tag_bound)"

        draw_text(canvas, f"坐标系标识: {cur_frame.frame_id}", (card_x + 16, c1_y + 12), font_size=13, color=(210, 225, 240))
        draw_text(canvas, f"父坐标系: {cur_frame.parent_frame_id}", (card_x + 220, c1_y + 12), font_size=13, color=(210, 225, 240))
        draw_text(canvas, f"类型: {type_desc}", (card_x + 16, c1_y + 36), font_size=12, color=(0, 220, 200))

        # 动标定义与说明悬停帮助徽章
        help_btn_x, help_btn_y, help_btn_w, help_btn_h = card_x + 310, c1_y + 33, 126, 22
        is_hover_help = (help_btn_x <= mpos[0] <= help_btn_x + help_btn_w and help_btn_y <= mpos[1] <= help_btn_y + help_btn_h)
        help_bg = (30, 48, 48) if is_hover_help else (20, 28, 36)
        help_border = (0, 255, 200) if is_hover_help else (40, 75, 75)
        cv2.rectangle(canvas, (help_btn_x, help_btn_y), (help_btn_x + help_btn_w, help_btn_y + help_btn_h), help_bg, -1)
        cv2.rectangle(canvas, (help_btn_x, help_btn_y), (help_btn_x + help_btn_w, help_btn_y + help_btn_h), help_border, 1)
        draw_text(canvas, "(?) 动标定义说明", (help_btn_x + 8, help_btn_y + 4), font_size=11,
                  color=(0, 255, 220) if is_hover_help else (140, 185, 195), bold=is_hover_help)

        if is_world_datum:
            w_box_y = c1_y + 60
            cv2.rectangle(canvas, (card_x + 16, w_box_y), (card_x + card_w - 16, w_box_y + 62), (18, 22, 32), -1)
            cv2.rectangle(canvas, (card_x + 16, w_box_y), (card_x + card_w - 16, w_box_y + 62), (38, 48, 65), 1)
            draw_text(canvas, "● 工位全局绝对空间基准 (World Datum / Origin)", (card_x + 24, w_box_y + 9), font_size=12, color=(0, 255, 200), bold=True)
            draw_text(canvas, "位姿特性: 恒为单位阵 Identity 4x4 (空间测量全局绝对基准原点，无需外参)", (card_x + 24, w_box_y + 33), font_size=12, color=(160, 180, 200))

        elif cur_frame.type == "fixed_transform":
            if cur_frame.translation_xyz_mm is None or getattr(cur_frame, "rotation_rpy_deg", None) is None:
                unk_box_y = c1_y + 60
                cv2.rectangle(canvas, (card_x + 16, unk_box_y), (card_x + card_w - 16, unk_box_y + 62), (18, 22, 32), -1)
                cv2.rectangle(canvas, (card_x + 16, unk_box_y), (card_x + card_w - 16, unk_box_y + 62), (38, 48, 65), 1)
                draw_text(canvas, "● 外参位姿约束状态: 未知 / 待解 (Unknown / To be Calibrated)", (card_x + 24, unk_box_y + 9), font_size=12, color=(140, 160, 255), bold=True)
                draw_text(canvas, "位姿特性: 6自由度外参均处于待解状态，将在全局建图或视觉求解时进行标定", (card_x + 24, unk_box_y + 33), font_size=11, color=(160, 175, 195))
            else:
                k_dof = getattr(cur_frame, "known_dof", None) or [True] * 6
                tx, ty, tz = cur_frame.translation_xyz_mm
                rx, ry, rz = cur_frame.rotation_rpy_deg

                tx_str = f"X:{tx:+.1f}" if k_dof[0] else "X:? (待解)"
                ty_str = f"Y:{ty:+.1f}" if k_dof[1] else "Y:? (待解)"
                tz_str = f"Z:{tz:+.1f}" if k_dof[2] else "Z:? (待解)"

                rx_str = f"Rx:{rx:+.1f}°" if k_dof[3] else "Rx:? (待解)"
                ry_str = f"Ry:{ry:+.1f}°" if k_dof[4] else "Ry:? (待解)"
                rz_str = f"Rz:{rz:+.1f}°" if k_dof[5] else "Rz:? (待解)"

                t_box_y = c1_y + 60
                cv2.rectangle(canvas, (card_x + 16, t_box_y), (card_x + card_w - 16, t_box_y + 30), (18, 22, 32), -1)
                cv2.rectangle(canvas, (card_x + 16, t_box_y), (card_x + card_w - 16, t_box_y + 30), (38, 48, 65), 1)
                draw_text(canvas, "平移 T [mm]:", (card_x + 24, t_box_y + 7), font_size=11, color=(160, 180, 200))
                draw_text(canvas, f"{tx_str}  {ty_str}  {tz_str}", (card_x + 120, t_box_y + 7), font_size=12, color=(0, 255, 220), bold=True)

                r_box_y = c1_y + 94
                cv2.rectangle(canvas, (card_x + 16, r_box_y), (card_x + card_w - 16, r_box_y + 30), (18, 22, 32), -1)
                cv2.rectangle(canvas, (card_x + 16, r_box_y), (card_x + card_w - 16, r_box_y + 30), (38, 48, 65), 1)
                draw_text(canvas, "旋转 R [deg]:", (card_x + 24, r_box_y + 7), font_size=11, color=(160, 180, 200))
                draw_text(canvas, f"{rx_str}  {ry_str}  {rz_str}", (card_x + 120, r_box_y + 7), font_size=12, color=(255, 200, 60), bold=True)

        else:
            tag_box_y = c1_y + 60
            cv2.rectangle(canvas, (card_x + 16, tag_box_y), (card_x + card_w - 16, tag_box_y + 44), (18, 22, 32), -1)
            cv2.rectangle(canvas, (card_x + 16, tag_box_y), (card_x + card_w - 16, tag_box_y + 44), (38, 48, 65), 1)
            off = getattr(cur_frame, "offset_xyz_mm", [0.0, 0.0, 0.0])
            draw_text(canvas, f"标称安装偏移 offset: [{off[0]:.1f}, {off[1]:.1f}, {off[2]:.1f}] mm", (card_x + 24, tag_box_y + 14), font_size=12, color=(200, 215, 230))

        # 3. 中间卡片: Tag 工位公共物理属性 (空间绝对尺度基准)
        c_mid_y = c1_y + c1_h + 8
        c_mid_h = 58
        cv2.rectangle(canvas, (card_x, c_mid_y), (card_x + card_w, c_mid_y + c_mid_h), (24, 28, 38), -1)
        cv2.rectangle(canvas, (card_x, c_mid_y), (card_x + card_w, c_mid_y + c_mid_h), (48, 60, 80), 1)

        cv2.circle(canvas, (card_x + 18, c_mid_y + 17), 4, (0, 240, 220), -1)
        draw_text(canvas, "Tag 工位公共物理属性 (三维空间绝对尺度基准 Scale Datum)",
                  (card_x + 28, c_mid_y + 9), font_size=12, color=(0, 240, 220), bold=True)

        ms_val = state.whitelist.get_workspace_marker_size()
        if ms_val is not None and ms_val > 0:
            ms_str = f"标靶名义边长: {ms_val:.3f} mm  (已显式核准 √)"
            ms_col = (0, 255, 180)
        else:
            ms_str = "标靶名义边长: 未录入 ⚠️ (建图/跟踪拒绝运行，请点击核准)"
            ms_col = (0, 160, 255)
        draw_text(canvas, ms_str, (card_x + 28, c_mid_y + 32), font_size=13, color=ms_col, bold=True)

        # 边长编辑/核准按钮
        self.r._draw_button(canvas, FRAME_TAG_EDIT_SIZE_BTN, "核准 / 编辑边长", mpos, theme_color=(0, 220, 160))

        # 4. 下部卡片: 10-Slot Tag 专属分配矩阵与局部真值
        c2_y = c_mid_y + c_mid_h + 8
        c2_h = box_h - (c2_y - box_y) - 6
        cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (24, 28, 38), -1)
        cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (42, 52, 70), 1)

        tag_range = state.geometry.get_frame_tag_range(cur_frame.frame_id)
        start_id, end_id = tag_range[0], tag_range[-1]
        draw_text(canvas, f"Tag 专属分段放行矩阵 [分配区间 ID: {start_id:02d} ~ {end_id:02d}]",
                  (card_x + 14, c2_y + 10), font_size=14, color=(0, 240, 220), bold=True)
        cv2.line(canvas, (card_x + 10, c2_y + 44), (card_x + card_w - 10, c2_y + 44), (36, 45, 60), 1)

        allowed_set = set(state.geometry.get_frame_tags_status(cur_frame.frame_id))
        anchors = state.whitelist.get_anchor_map()

        for slot_idx in range(10):
            tag_id = tag_range[slot_idx]
            cx, cy, cw, ch = frame_tag_chip_rect(slot_idx)
            is_allowed = (tag_id in allowed_set)
            is_hover = (cx <= mpos[0] <= cx + cw and cy <= mpos[1] <= cy + ch)

            if is_allowed:
                chip_bg = (24, 40, 36) if is_hover else (18, 32, 28)
                chip_border = (0, 255, 180) if is_hover else (0, 200, 140)
            else:
                chip_bg = (30, 34, 42) if is_hover else (20, 24, 32)
                chip_border = (0, 200, 240) if is_hover else (45, 55, 70)

            cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), chip_bg, -1)
            cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), chip_border, 2 if (is_hover or is_allowed) else 1)

            tag_label = f"#{tag_id:02d}"
            draw_text(canvas, tag_label, (cx + 8, cy + 6), font_size=13,
                      color=(0, 255, 200) if is_allowed else (160, 175, 195), bold=True)

            status_str = "已放行 √" if is_allowed else "未放行"
            status_col = (0, 255, 160) if is_allowed else (120, 135, 150)
            draw_text(canvas, status_str, (cx + cw - 52, cy + 6), font_size=11, color=status_col, bold=is_allowed)

            cv2.line(canvas, (cx + 6, cy + 30), (cx + cw - 6, cy + 30), (35, 45, 58), 1)

            anchor_entry = anchors.get(tag_id) or anchors.get(str(tag_id))
            if isinstance(anchor_entry, dict) and "xyz_mm" in anchor_entry:
                xyz = anchor_entry.get("xyz_mm", [0.0, 0.0, 0.0])
                known = anchor_entry.get("known", [True, True, True])
                sx = f"{xyz[0]:.0f}" if (len(known) > 0 and known[0]) else "?"
                sy = f"{xyz[1]:.0f}" if (len(known) > 1 and known[1]) else "?"
                sz = f"{xyz[2]:.0f}" if (len(known) > 2 and known[2]) else "?"
                if any(known):
                    pos_str = f"P:({sx},{sy},{sz})"
                    pos_col = (255, 210, 80) if all(known) else (0, 220, 255)
                else:
                    pos_str = "P: 未标注"
                    pos_col = (110, 125, 140)
            else:
                pos_str = "P: 未标注"
                pos_col = (110, 125, 140)
            draw_text(canvas, pos_str, (cx + 6, cy + 40), font_size=10, color=pos_col)

            px, py = cx + cw - 14, cy + 48
            pen_col = (0, 255, 180) if is_hover else (70, 95, 110)
            cv2.line(canvas, (px - 3, py + 2), (px + 3, py - 4), pen_col, 1, cv2.LINE_AA)

    def render_frame_rois(self, canvas: np.ndarray, state: HubState, ws: Any):
        """坐标系专属页签 2: 3D ROI 空间物件 (x: 340~960, y: 50~670)"""
        from tools.workspace_hub.hub_renderer import (
            FRAME_ROI_PREV_BTN, FRAME_ROI_NEXT_BTN, FRAME_ADD_ROI_BTN,
            FRAME_ROI_SCROLL_TRACK, frame_roi_row_rect, frame_roi_edit_btn, frame_roi_del_btn
        )

        box_x, box_y, box_w, box_h = 340, 50, self.r.canvas_w - 340, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (18, 22, 28), -1)
        mpos = (state.mouse_x, state.mouse_y)

        cur_frame = state.geometry.get_selected_frame()
        if not cur_frame:
            draw_text(canvas, "未选择任何机构坐标系", (box_x + 180, box_y + 280), font_size=18, color=self.r.COLOR_GRAY)
            return

        rois = state.geometry.get_frame_rois(cur_frame.frame_id)
        total_rois = len(rois)
        max_vis = HubState.ROI_VISIBLE_COUNT
        max_offset = max(0, total_rois - max_vis)
        offset = max(0, min(getattr(state.geometry, "roi_scroll_offset", 0), max_offset))
        state.geometry.roi_scroll_offset = offset

        # 1. 顶部标题栏、翻页控制与新建 ROI 按钮
        if total_rois > max_vis:
            disp_end = min(offset + max_vis, total_rois)
            header_text = f"3D ROI 空间物件 ({cur_frame.frame_id} · 显示 {offset + 1}~{disp_end} / 共 {total_rois} 个)"
            draw_text(canvas, header_text, (box_x + 16, box_y + 14), font_size=14, color=self.r.COLOR_WHITE, bold=True)
            self.r._draw_button(canvas, FRAME_ROI_PREV_BTN, "▲ 上翻", mpos, enabled=(offset > 0))
            self.r._draw_button(canvas, FRAME_ROI_NEXT_BTN, "▼ 下翻", mpos, enabled=(offset < max_offset))
        else:
            header_text = f"3D ROI 空间物件 (坐标系: {cur_frame.frame_id} · 共 {total_rois} 个)"
            draw_text(canvas, header_text, (box_x + 16, box_y + 14), font_size=15, color=self.r.COLOR_WHITE, bold=True)

        self.r._draw_button(canvas, FRAME_ADD_ROI_BTN, "+ 新建本坐标系 ROI", mpos)

        # 2. ROI 列表卡片
        if not rois:
            empty_y = box_y + 180
            cv2.rectangle(canvas, (box_x + 40, empty_y), (box_x + box_w - 40, empty_y + 120), (24, 28, 38), -1)
            cv2.rectangle(canvas, (box_x + 40, empty_y), (box_x + box_w - 40, empty_y + 120), (42, 52, 70), 1)
            draw_text(canvas, f"当前坐标系 [{cur_frame.frame_id}] 下暂无 3D ROI 空间物件",
                      (box_x + 110, empty_y + 36), font_size=16, color=(0, 220, 255), bold=True)
            draw_text(canvas, "点击右上角 [+ 新建本坐标系 ROI] 即可为该机构添加检测工作面包围盒",
                      (box_x + 85, empty_y + 70), font_size=13, color=(140, 160, 180))
            return

        visible_rois = rois[offset : offset + max_vis]
        for i, r in enumerate(visible_rois):
            rx, ry, rw, rh = frame_roi_row_rect(i)
            is_hover = (rx <= mpos[0] <= rx + rw and ry <= mpos[1] <= ry + rh)
            card_bg = (26, 32, 44) if is_hover else (22, 26, 36)
            card_border = (0, 220, 240) if is_hover else (38, 48, 64)

            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), card_bg, -1)
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), card_border, 2 if is_hover else 1)

            cat_map = {"belt": "同步带", "wheel": "驱动轮", "tray": "料盘", "general": "通用"}
            cat_badge = cat_map.get(r.category, r.category)
            cv2.rectangle(canvas, (rx + 10, ry + 10), (rx + 78, ry + 34), (32, 45, 58), -1)
            cv2.rectangle(canvas, (rx + 10, ry + 10), (rx + 78, ry + 34), (0, 200, 240), 1)
            draw_text(canvas, cat_badge, (rx + 18, ry + 14), font_size=11, color=(0, 240, 220), bold=True)

            title = f"{r.name} ({r.roi_id})"
            draw_text(canvas, title, (rx + 86, ry + 13), font_size=14, color=self.r.COLOR_WHITE, bold=True)

            # Smart ROI 语义徽章
            r_role = getattr(r, "role", "source") or "source"
            r_intent = getattr(r, "target_intent", "pose_pick") or "pose_pick"
            slot_info = f"#{r.binding.get('slot_index', 0)}" if getattr(r, "binding", None) and "slot_index" in r.binding else ""
            role_labels = {
                "source": ("★进料源", (0, 255, 140)),
                "destination": (f"▼落料{slot_info}", (0, 220, 255)),
                "keepout": ("⛔禁行区", (255, 80, 80)),
                "general": ("◌通用", (180, 180, 180)),
            }
            role_txt, role_col = role_labels.get(r_role, ("◌通用", (180, 180, 180)))
            intent_labels = {
                "pose_pick": "位姿抓取",
                "piece_count": "根数统计",
                "occupancy": "在席检测",
                "general": "通用意图",
            }
            intent_txt = intent_labels.get(r_intent, r_intent)
            badge_str = f"[{role_txt} · {intent_txt}]"
            draw_text(canvas, badge_str, (rx + 310, ry + 14), font_size=12, color=role_col, bold=True)

            cx, cy, cz = r.center_xyz_mm
            sx, sy, sz = r.size_xyz_mm
            rx_d, ry_d, rz_d = getattr(r, "rotation_rpy_deg", [0.0, 0.0, 0.0])
            geom_str = f"中心: ({cx:.0f},{cy:.0f},{cz:.0f})  尺寸: ({sx:.0f}×{sy:.0f}×{sz:.0f})  旋转: ({rx_d:.0f}°,{ry_d:.0f}°,{rz_d:.0f}°)"
            put_text(canvas, geom_str, (rx + 14, ry + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 180, 200), 1, cv2.LINE_AA)

            self.r._draw_button(canvas, frame_roi_edit_btn(i), "编辑", mpos)
            self.r._draw_button(canvas, frame_roi_del_btn(i), "删除", mpos, theme_color=(180, 60, 60))

        # 3. 极简现代扁平滚动条 (当物件超过单屏容量时自动呈现)
        if total_rois > max_vis:
            tx, ty, tw, th = FRAME_ROI_SCROLL_TRACK
            is_hover_track = (tx - 4 <= mpos[0] <= tx + tw + 4 and ty <= mpos[1] <= ty + th)
            cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (22, 26, 34), -1)
            cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (36, 44, 56), 1)

            thumb_h = max(32, int(th * (float(max_vis) / float(total_rois))))
            thumb_travel = th - thumb_h
            thumb_y = ty + int(thumb_travel * (float(offset) / float(max_offset))) if max_offset > 0 else ty
            is_hover_thumb = (tx - 4 <= mpos[0] <= tx + tw + 4 and thumb_y <= mpos[1] <= thumb_y + thumb_h)

            thumb_color = (0, 240, 220) if is_hover_thumb else ((0, 190, 210) if is_hover_track else (65, 80, 100))
            cv2.rectangle(canvas, (tx + 1, thumb_y), (tx + tw - 1, thumb_y + thumb_h), thumb_color, -1)
