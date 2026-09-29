#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
空间建图工作站 - 事件交互 Mixin (mapping_events.py)
===================================================
从主控拆分出的事件处理职责模块，由 SpatialMappingStudioApp 以 Mixin 方式继承：
  - _on_mouse: OpenCV 鼠标事件命中测试 (滚轮缩放/拖拽平移/双击复位/按钮分发/标靶打叉)
  - _handle_button_click: GUI 按钮 ID 统一分发
无 __init__、无新增实例属性，全部通过宿主 self 与主控制器协作。
"""

import os
from typing import Any

import cv2

from src.calibration.manifest_repository import ManifestRepository
from tools.spatial_mapping_studio.mapping_ui_common import (
    FILTER_MODE_OPTIONS,
    SORT_MODE_OPTIONS,
    BA_VIEW_OPTIONS,
    OBS_VIEW_OPTIONS
)


class MappingEventMixin:

    """鼠标事件交互与 GUI 按钮点击分发职责 (无状态，依赖宿主 self 属性)"""

    def _on_mouse(self, event, mx, my, flags, param):
        self.mouse_pos = (mx, my)

        top_h = self.viewport.top_bar_h if self.viewport else 44
        bot_h = self.viewport.bottom_bar_h if self.viewport else 52
        content_y1 = top_h
        content_y2 = self.win_h - bot_h

        left_x1, left_x2 = 0, self.left_bar_w
        mid_x1, mid_x2 = self.left_bar_w, self.win_w - self.right_bar_w

        right_x1, right_x2 = self.win_w - self.right_bar_w, self.win_w

        # 1. 鼠标滚轮事件 (精准区分：左侧帧列表滚动 vs 右侧标靶列表滚动 vs 中间视口以鼠标为中心缩放)
        if event == cv2.EVENT_MOUSEWHEEL:
            # 滚轮判定方向: flags > 0 为向上滚, flags < 0 为向下滚
            wheel_up = (flags > 0)

            # A. 鼠标光标位于左栏：上下滚动帧资产列表
            if left_x1 <= mx < left_x2:
                filtered_indices = self._get_filtered_indices()
                self.frame_list_box.handle_scroll(-1 if wheel_up else 1, len(filtered_indices))
                self.scroll_offset = self.frame_list_box.scroll_offset
                return

            # B. 鼠标光标位于右栏：上下滚动标靶残差清单
            elif right_x1 <= mx <= right_x2:
                cur_file = self.image_files[self.current_img_idx] if (self.image_files and 0 <= self.current_img_idx < len(self.image_files)) else None
                if cur_file:
                    bname = os.path.basename(cur_file)
                    meta = self.frame_metrics_cache.get(bname, {})
                    obs_cnt = len(meta.get("observations", []))
                    self.tag_list_box.handle_scroll(-1 if wheel_up else 1, obs_cnt)
                return

            # C. 鼠标光标位于中间画布视口：执行以光标为中心的精准缩放 (Zoom In/Out)
            elif mid_x1 <= mx < mid_x2 and content_y1 <= my < content_y2:
                self.viewport.zoom_at(mx, my, wheel_up, (mid_x1, content_y1, mid_x2 - mid_x1, content_y2 - content_y1))
                return

        # 2. 拖拽与移动事件 (支持滚动条滑块拖拽与鼠标右键/中键画布平移)
        if event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
            if mid_x1 <= mx < mid_x2 and content_y1 <= my < content_y2:
                self.viewport.start_pan(mx, my)
                return
        elif event == cv2.EVENT_MOUSEMOVE:
            # 优先处理滚动条滑块拖拽跟随
            if self.frame_list_box.is_dragging_thumb:
                filtered_indices = self._get_filtered_indices()
                self.frame_list_box.handle_mouse_move(mx, my, len(filtered_indices))
                self.scroll_offset = self.frame_list_box.scroll_offset
                return
            if self.tag_list_box.is_dragging_thumb:
                cur_file = self.image_files[self.current_img_idx] if (self.image_files and 0 <= self.current_img_idx < len(self.image_files)) else None
                bname = os.path.basename(cur_file) if cur_file else ""
                obs_cnt = len(self.frame_metrics_cache.get(bname, {}).get("observations", []))
                self.tag_list_box.handle_mouse_move(mx, my, obs_cnt)
                return

            if self.viewport.update_pan(mx, my):
                return
        elif event in (cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP, cv2.EVENT_LBUTTONUP):
            self.frame_list_box.handle_mouse_up()
            self.tag_list_box.handle_mouse_up()
            if self.viewport.is_panning:
                self.viewport.end_pan()
                return

        # 3. 双击事件 (双击左键或右键一键重置缩放)
        if event in (cv2.EVENT_LBUTTONDBLCLK, cv2.EVENT_RBUTTONDBLCLK):
            if mid_x1 <= mx < mid_x2 and content_y1 <= my < content_y2:
                self.reset_viewport_zoom()
                return

        # 4. 鼠标左键点击事件 (GUI 按钮分发，优先命中置顶下拉层与滚动条滑块)
        if event == cv2.EVENT_LBUTTONDOWN:
            # A. 优先检测左栏滚动条滑块与轨道
            if left_x1 <= mx < left_x2:
                filtered_indices = self._get_filtered_indices()
                handled, clicked_idx = self.frame_list_box.handle_mouse_down(mx, my, len(filtered_indices))
                if handled and clicked_idx is None:
                    # 命中了滑块或轨道跳转
                    self.scroll_offset = self.frame_list_box.scroll_offset
                    return

            # B. 优先检测右栏滚动条滑块与轨道
            elif right_x1 <= mx <= right_x2:
                cur_file = self.image_files[self.current_img_idx] if (self.image_files and 0 <= self.current_img_idx < len(self.image_files)) else None
                bname = os.path.basename(cur_file) if cur_file else ""
                obs_cnt = len(self.frame_metrics_cache.get(bname, {}).get("observations", []))
                handled, clicked_idx = self.tag_list_box.handle_mouse_down(mx, my, obs_cnt)
                if handled and clicked_idx is None:
                    # 命中了滑块或轨道跳转
                    return

            clicked_any = False
            for btn_id, (bx1, by1, bx2, by2), extra in reversed(self.gui_buttons):
                if bx1 <= mx <= bx2 and by1 <= my <= by2:
                    self._handle_button_click(btn_id, extra, mx, my)
                    clicked_any = True
                    return

            # 若未点击任何已注册按钮，且当前有下拉菜单展开，则自动收起 (Click-outside)
            if not clicked_any and self.active_dropdown:
                self.active_dropdown = None
                return

            # 5. 检查是否直接点击在中间视口图片的标靶区域上 (画布直接打叉剔除 / 恢复审核模式)
            if mid_x1 <= mx < mid_x2 and content_y1 <= my < content_y2:
                if self.image_files and 0 <= self.current_img_idx < len(self.image_files):
                    cur_file = self.image_files[self.current_img_idx]
                    bname = os.path.basename(cur_file)
                    meta = self.frame_metrics_cache.get(bname, {})
                    obs_list = meta.get("observations", [])

                    if obs_list:
                        bgr = cv2.imread(cur_file)
                        if bgr is not None:
                            frame_h, frame_w = bgr.shape[:2]
                            hit_tid = self.viewport.hit_test_tag(
                                mx, my, obs_list,
                                (mid_x1, content_y1, mid_x2 - mid_x1, content_y2 - content_y1),
                                frame_w, frame_h
                            )
                            if hit_tid is not None:
                                self.toggle_tag_exclusion_in_current_frame(hit_tid)
                                return

    def _handle_button_click(self, btn_id: str, extra: Any, mx: int, my: int):
        if btn_id == "EXIT":
            self.is_running = False
        elif btn_id == "RUN_BA":
            self.start_async_bundle_adjustment()
        elif btn_id == "ALIGN_WORLD_DATUM":
            self.align_current_workspace_world_datum()
        elif btn_id == "EXPORT_REPORT":
            self.export_verification_report()
        elif btn_id == "TOGGLE_WORKSPACE_DROPDOWN":
            self.active_dropdown = None if self.active_dropdown == "WORKSPACE_DROPDOWN" else "WORKSPACE_DROPDOWN"
        elif btn_id == "TOGGLE_BA_VIEW_DROPDOWN":
            self.active_dropdown = None if self.active_dropdown == "BA_VIEW_DROPDOWN" else "BA_VIEW_DROPDOWN"
        elif btn_id == "TOGGLE_OBS_VIEW_DROPDOWN":
            self.active_dropdown = None if self.active_dropdown == "OBS_VIEW_DROPDOWN" else "OBS_VIEW_DROPDOWN"
        elif btn_id == "TOGGLE_VIEW_DROPDOWN":
            self.active_dropdown = None if self.active_dropdown == "BA_VIEW_DROPDOWN" else "BA_VIEW_DROPDOWN"
        elif btn_id == "TOGGLE_FILTER_DROPDOWN":
            self.active_dropdown = None if self.active_dropdown == "FILTER_DROPDOWN" else "FILTER_DROPDOWN"
        elif btn_id == "TOGGLE_SORT_DROPDOWN":
            self.active_dropdown = None if self.active_dropdown == "SORT_DROPDOWN" else "SORT_DROPDOWN"
        elif btn_id == "TOGGLE_DRAW_XY_PLANE":
            self.show_xy_plane_on = not self.show_xy_plane_on
            self.set_toast(f"XY 平面网格{'已开启' if self.show_xy_plane_on else '已关闭'} ({self.get_current_plane_z_label()})")
        elif btn_id == "TOGGLE_PLANE_Z_DROPDOWN":
            self.active_dropdown = None if self.active_dropdown == "PLANE_Z_DROPDOWN" else "PLANE_Z_DROPDOWN"
        elif btn_id.startswith("DD_SELECT_"):
            dd_name, selected_val = extra
            if dd_name == "WORKSPACE_DROPDOWN":
                self.switch_workspace(selected_val)
            elif dd_name == "PLANE_Z_DROPDOWN":
                if selected_val == "NONE":
                    self.show_xy_plane_on = False
                    self.set_toast("XY 平面绘制已关闭")
                else:
                    try:
                        self.plane_z = float(selected_val)
                    except ValueError:
                        self.plane_z = 0.0
                    self.show_xy_plane_on = True
                    self.set_toast(f"XY 平面已平移至 {self.get_current_plane_z_label()}")
            elif dd_name == "BA_VIEW_DROPDOWN":
                self.ba_view_mode = selected_val
                lbl = dict(BA_VIEW_OPTIONS).get(selected_val, selected_val)
                self.set_toast(f"BA 理论显示已切换为: {lbl}")
            elif dd_name == "OBS_VIEW_DROPDOWN":
                self.obs_view_mode = selected_val
                lbl = dict(OBS_VIEW_OPTIONS).get(selected_val, selected_val)
                self.set_toast(f"实测识别显示已切换为: {lbl}")
            elif dd_name == "FILTER_DROPDOWN":
                self.filter_mode = selected_val
                self.scroll_offset = 0
                lbl = dict(FILTER_MODE_OPTIONS).get(selected_val, selected_val)
                self.set_toast(f"筛选模式已切换为: {lbl}")
            elif dd_name == "SORT_DROPDOWN":
                self.sort_mode = selected_val
                self.scroll_offset = 0
                lbl = dict(SORT_MODE_OPTIONS).get(selected_val, selected_val)
                self.set_toast(f"排序方式已切换为: {lbl}")
            self.active_dropdown = None
        elif btn_id.startswith("SELECT_FRAME_"):
            orig_idx = int(extra)
            self.current_img_idx = orig_idx
            self.set_toast(f"已选中帧: {os.path.basename(self.image_files[orig_idx])}")
            self.active_dropdown = None
        elif btn_id == "TOGGLE_FRAME_STATUS":
            self.toggle_current_frame_exclusion()
        elif btn_id.startswith("TOGGLE_TAG_"):
            tid = int(extra)
            self.toggle_tag_exclusion_in_current_frame(tid)

        elif btn_id == "SUPER_EXTRACT_FRAME":
            self.set_toast("正在执行工序 3 工业级超精重提取 (多尺度CLAHE+2x超分+0.01px亚像素精修)...")
            bname, cnt = self.super_extract_current_frame()
            if bname:
                self.set_toast(f"帧 {bname} 超精重提取完成并已原子持久化: 检出 {cnt} 个标靶")
        elif btn_id == "SUPER_EXTRACT_ALL":
            self.start_async_super_extract_all()
        elif btn_id == "DIAGNOSE_FRAME":
            self.toggle_frame_diagnostics()
        elif btn_id == "LAUNCH_TRACKER":
            self.launch_robot_online_tracker()
        elif btn_id == "RUN_AUTO_PRUNE_BA":
            self.start_auto_prune_ba()
        elif btn_id == "TOGGLE_MATRIX_VIEW":
            self.toggle_matrix_view_mode()
        elif btn_id == "DRAW_ROI_OBJECT":
            self.toggle_draw_roi_mode()
        elif btn_id == "ACCEPT_PRUNE":
            self.accept_prune_results()
        elif btn_id == "UNDO_PRUNE":
            self.undo_prune_results()
        elif btn_id == "STOP_PRUNE":
            self.ba_runner.request_stop_pruning()
        elif btn_id == "DISMISS_TOAST":
            self.dismiss_toast()
        elif btn_id == "COPY_TOAST":
            self.copy_toast()
        elif btn_id == "CLOSE_ALIGN_REPORT":
            self.alignment_report = None
        elif btn_id == "TOGGLE_ALIGN_REPORT_SORT":
            self.alignment_report_sort = "err_desc" if getattr(self, "alignment_report_sort", "id") == "id" else "id"
        elif btn_id == "COPY_ALIGN_REPORT":
            self.copy_alignment_report()

