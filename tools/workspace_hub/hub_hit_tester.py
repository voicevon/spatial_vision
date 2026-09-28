# -*- coding: utf-8 -*-
"""
Workspace Hub 交互碰撞与命中测试引擎 (HubHitTester)
======================================================
负责界面所有可交互热区、卡片、按钮、树节点与模态控件的几何判定：
1. 顶部 Header 页签胶囊与退出按钮热区
2. 左侧两层树结构 (工位卡片折叠/选中、坐标系子节点选中) 动态几何排版与命中测试
3. 右侧各页签内操作按钮 (大盘看板、相册翻页/删帧、坐标系与 ROI 新增/编辑/删除)
4. 结构化表单弹窗内部输入框、下拉框展开与枚举选项命中探测
5. 动标说明胶囊 Tooltip 悬浮感应
"""

from typing import Any, Tuple, List, Optional
from tools.workspace_hub.hub_state import HubState


class HubHitTester:
    """Workspace Hub 碰撞与命中探测引擎"""

    def __init__(self, parent_renderer: Any):
        self.r = parent_renderer

    def get_tabs_layout(self, state: HubState) -> List[Tuple[str, str, Tuple[int, int, int, int]]]:
        """计算顶部 Tab 胶囊的动态布局矩形 (委托给统一 TabBar 组件)"""
        from tools.workspace_hub.hub_renderer import HEADER_TAB_X0, HEADER_TAB_Y0, HEADER_TAB_H
        layout = state.tab_bar.compute_layout((HEADER_TAB_X0, HEADER_TAB_Y0, 510, HEADER_TAB_H))
        label_map = {it.key: it.label for it in state.tab_bar.items}
        return [(k, label_map.get(k, k), r) for k, r in layout]

    def get_tree_layout(self, state: HubState) -> List[dict]:
        """计算左侧两层树结构各项的几何矩形与数据标识，供渲染与点击测试统一使用"""
        items = []
        cur_y = 58
        max_y = 604
        for ws_idx, ws in enumerate(state.workspaces):
            if cur_y + 44 > max_y:
                break
            ws_id = ws.workspace_id
            is_expanded = (ws_id in state.expanded_workspaces)
            is_ws_selected = (
                state.selected_tree_item[0] == "workspace"
                and state.selected_workspace_idx == ws_idx
            )

            ws_rect = (10, cur_y, 320, 44)
            arrow_rect = (10, cur_y, 30, 44)
            body_rect = (40, cur_y, 290, 44)

            items.append({
                "type": "workspace",
                "ws_idx": ws_idx,
                "workspace": ws,
                "is_expanded": is_expanded,
                "is_selected": is_ws_selected,
                "ws_rect": ws_rect,
                "arrow_rect": arrow_rect,
                "body_rect": body_rect,
            })
            cur_y += 48

            # 若工位展开，渲染其坐标系子节点
            if is_expanded:
                mgr = state.geometry.get_workspace_coord_mgr(ws)
                frames = mgr.list_frames() if mgr else []

                for f in frames:
                    if cur_y + 30 > max_y:
                        break
                    is_frame_selected = (
                        state.selected_tree_item[0] == "frame"
                        and state.selected_workspace_idx == ws_idx
                        and state.selected_tree_item[2] == f.frame_id
                    )
                    frame_rect = (34, cur_y, 296, 30)
                    items.append({
                        "type": "frame",
                        "ws_idx": ws_idx,
                        "workspace": ws,
                        "frame": f,
                        "frame_id": f.frame_id,
                        "is_selected": is_frame_selected,
                        "rect": frame_rect,
                    })
                    cur_y += 34
        return items

    def hit_test(self, mx: int, my: int, state: HubState) -> Any:
        """根据逻辑坐标探测当前命中交互元素"""
        old_x, old_y = state.mouse_x, state.mouse_y
        try:
            state.mouse_x, state.mouse_y = mx, my
            return self.get_interactive_hover_key(state)
        finally:
            state.mouse_x, state.mouse_y = old_x, old_y

    def get_interactive_hover_key(self, state: HubState) -> Any:
        """获取当前鼠标悬停的交互元素标识 (若鼠标未落在任何可交互组件上返回 None)"""
        from tools.workspace_hub.hub_renderer import (
            GEOM_MODAL_X, GEOM_MODAL_Y, GEOM_MODAL_W, GEOM_MODAL_H,
            GEOM_MODAL_CLOSE, GEOM_MODAL_SAVE, GEOM_MODAL_CANCEL,
            HELP_MODAL_W, HELP_MODAL_H,
            BTN_EXIT_X0, BTN_EXIT_Y0, BTN_EXIT_W, BTN_EXIT_H,
            WS_BTN_RENAME, WS_BTN_OPEN_DIR, WS_BTN_TOGGLE_PROD_MODE, WS_BTN_EDIT_DESC,
            WS_BTN_SYNC_DATA, WS_BTN_CLONE, WS_BTN_DELETE, WS_BTN_NEW_FRAME,
            FRAME_EDIT_POSE_BTN, FRAME_ADD_ROI_BTN, FRAME_TAG_EDIT_SIZE_BTN,
            FRAME_ROI_PREV_BTN, FRAME_ROI_NEXT_BTN, FRAME_ROI_SCROLL_TRACK,
            frame_tag_chip_rect, frame_roi_edit_btn, frame_roi_del_btn,
            frame_btn_add_rect, roi_btn_add_rect,
            frame_row_edit_rect, frame_row_del_rect,
            roi_row_edit_rect, roi_row_del_rect,
            grid_hit_test, point_in_rect
        )

        mx, my = state.mouse_x, state.mouse_y
        if mx < 0 or my < 0:
            return None

        # 0. 坐标系与 3D ROI 结构化弹窗模式 (最高交互层)
        if state.geometry.frame_modal_open:
            mx_box, my_box = GEOM_MODAL_X, GEOM_MODAL_Y
            mw, mh = GEOM_MODAL_W, GEOM_MODAL_H

            # 0.a 活跃下拉框浮层检测 (浮层拥有最高层级交互优先级)
            if state.geometry.active_dropdown in ("frame_type", "frame_parent"):
                dd_data = self.r._get_dropdown_data(state.geometry.active_dropdown, state)
                if dd_data:
                    (tx, ty, tw, th), cur_val, options = dd_data
                    drop_x = tx
                    drop_y = ty + th + 2
                    item_h = 30
                    drop_h = len(options) * item_h
                    if drop_x <= mx <= drop_x + tw and drop_y <= my <= drop_y + drop_h:
                        opt_idx = min(len(options) - 1, max(0, (my - drop_y) // item_h))
                        return ("dropdown_select", state.geometry.active_dropdown, options[opt_idx][0])
                    if point_in_rect(mx, my, (tx, ty, tw, th)):
                        return ("dropdown_toggle", state.geometry.active_dropdown)
                    return "dropdown_dismiss"

            if point_in_rect(mx, my, GEOM_MODAL_CLOSE):
                return "frame_modal_close"
            if point_in_rect(mx, my, GEOM_MODAL_SAVE):
                return "frame_modal_save"
            if point_in_rect(mx, my, GEOM_MODAL_CANCEL):
                return "frame_modal_cancel"
            form_y = my_box + 56
            if point_in_rect(mx, my, (mx_box + 115, form_y, 220, 28)):
                return "frame_field_name"
            if point_in_rect(mx, my, (mx_box + 430, form_y, 220, 28)):
                return "frame_field_id"

            # 下拉框触发条
            type_y = form_y + 40
            if point_in_rect(mx, my, (mx_box + 115, type_y, 360, 28)):
                return ("dropdown_toggle", "frame_type")

            parent_y = form_y + 80
            if point_in_rect(mx, my, (mx_box + 115, parent_y, 360, 28)):
                return ("dropdown_toggle", "frame_parent")

            param_y = form_y + 128
            d = state.geometry.frame_modal_data
            cur_type = d.get("type", "fixed_transform")
            if cur_type == "fixed_transform":
                from tools.workspace_hub.hub_renderer import (
                    FRAME_PARAM_BTN_UNKNOWN, FRAME_PARAM_BTN_EDIT6D
                )
                if point_in_rect(mx, my, FRAME_PARAM_BTN_UNKNOWN):
                    return "frame_set_unknown"
                if point_in_rect(mx, my, FRAME_PARAM_BTN_EDIT6D):
                    return "frame_open_pose6d"
                if point_in_rect(mx, my, (mx_box + 125, param_y + 48, 130, 28)):
                    return ("frame_field_num", "translation", 0)
                if point_in_rect(mx, my, (mx_box + 265, param_y + 48, 130, 28)):
                    return ("frame_field_num", "translation", 1)
                if point_in_rect(mx, my, (mx_box + 405, param_y + 48, 130, 28)):
                    return ("frame_field_num", "translation", 2)
                if point_in_rect(mx, my, (mx_box + 125, param_y + 94, 130, 28)):
                    return ("frame_field_num", "rotation", 0)
                if point_in_rect(mx, my, (mx_box + 265, param_y + 94, 130, 28)):
                    return ("frame_field_num", "rotation", 1)
                if point_in_rect(mx, my, (mx_box + 405, param_y + 94, 130, 28)):
                    return ("frame_field_num", "rotation", 2)

            else:
                if point_in_rect(mx, my, (mx_box + 165, param_y + 42, 160, 28)):
                    return ("frame_field_num", "tag_id", 0)
                if point_in_rect(mx, my, (mx_box + 185, param_y + 92, 120, 28)):
                    return ("frame_field_num", "offset", 0)
                if point_in_rect(mx, my, (mx_box + 315, param_y + 92, 120, 28)):
                    return ("frame_field_num", "offset", 1)
                if point_in_rect(mx, my, (mx_box + 445, param_y + 92, 120, 28)):
                    return ("frame_field_num", "offset", 2)
            if mx < mx_box or mx > mx_box + mw or my < my_box or my > my_box + mh:
                return "frame_modal_mask"
            return "frame_modal_body"

        if state.geometry.roi_modal_open:
            from tools.workspace_hub.hub_renderer import (
                ROI_ROLE_BTN_SOURCE, ROI_ROLE_BTN_DEST, ROI_ROLE_BTN_KEEPOUT, ROI_ROLE_BTN_GENERAL,
                ROI_INTENT_BTN_PICK, ROI_INTENT_BTN_COUNT, ROI_INTENT_BTN_OCC, ROI_INTENT_BTN_GEN,
                ROI_BIND_SLOT_BTN, ROI_BIND_CAP_BTN, ROI_BIND_CONF_BTN
            )
            mx_box, my_box = GEOM_MODAL_X, GEOM_MODAL_Y
            mw, mh = GEOM_MODAL_W, GEOM_MODAL_H

            # 0.b 活跃下拉框浮层检测
            if state.geometry.active_dropdown in ("roi_category", "roi_frame"):
                dd_data = self.r._get_dropdown_data(state.geometry.active_dropdown, state)
                if dd_data:
                    (tx, ty, tw, th), cur_val, options = dd_data
                    drop_x = tx
                    drop_y = ty + th + 2
                    item_h = 30
                    drop_h = len(options) * item_h
                    if drop_x <= mx <= drop_x + tw and drop_y <= my <= drop_y + drop_h:
                        opt_idx = min(len(options) - 1, max(0, (my - drop_y) // item_h))
                        return ("dropdown_select", state.geometry.active_dropdown, options[opt_idx][0])
                    if point_in_rect(mx, my, (tx, ty, tw, th)):
                        return ("dropdown_toggle", state.geometry.active_dropdown)
                    return "dropdown_dismiss"


            if point_in_rect(mx, my, GEOM_MODAL_CLOSE):
                return "roi_modal_close"
            if point_in_rect(mx, my, GEOM_MODAL_SAVE):
                return "roi_modal_save"
            if point_in_rect(mx, my, GEOM_MODAL_CANCEL):
                return "roi_modal_cancel"
            
            form_y = my_box + 46
            # 1. 名称与 ID
            if point_in_rect(mx, my, (mx_box + 110, form_y, 220, 28)):
                return "roi_field_name"
            if point_in_rect(mx, my, (mx_box + 425, form_y, 225, 28)):
                return "roi_field_id"

            # 2. 下拉框触发条 (并排)
            cat_y = form_y + 36
            if point_in_rect(mx, my, (mx_box + 110, cat_y, 220, 28)):
                return ("dropdown_toggle", "roi_category")
            if point_in_rect(mx, my, (mx_box + 425, cat_y, 225, 28)):
                return ("dropdown_toggle", "roi_frame")

            # 3. Smart ROI 工业生产语义与工艺角色交互
            # 3.1 工艺角色 (Role)
            if point_in_rect(mx, my, ROI_ROLE_BTN_SOURCE):
                return ("roi_set_role", "source")
            if point_in_rect(mx, my, ROI_ROLE_BTN_DEST):
                return ("roi_set_role", "destination")
            if point_in_rect(mx, my, ROI_ROLE_BTN_KEEPOUT):
                return ("roi_set_role", "keepout")
            if point_in_rect(mx, my, ROI_ROLE_BTN_GENERAL):
                return ("roi_set_role", "general")

            # 3.2 动作意图 (Target Intent)
            if point_in_rect(mx, my, ROI_INTENT_BTN_PICK):
                return ("roi_set_intent", "pose_pick")
            if point_in_rect(mx, my, ROI_INTENT_BTN_COUNT):
                return ("roi_set_intent", "piece_count")
            if point_in_rect(mx, my, ROI_INTENT_BTN_OCC):
                return ("roi_set_intent", "occupancy")
            if point_in_rect(mx, my, ROI_INTENT_BTN_GEN):
                return ("roi_set_intent", "general")

            # 3.3 设备绑定与容量门限 (Binding)
            if point_in_rect(mx, my, ROI_BIND_SLOT_BTN):
                return "roi_field_slot"
            if point_in_rect(mx, my, ROI_BIND_CAP_BTN):
                return "roi_field_capacity"
            if point_in_rect(mx, my, ROI_BIND_CONF_BTN):
                return "roi_field_conf"

            # 4. 3D 有向长方体空间几何
            smart_y = form_y + 72
            geom_y = smart_y + 148
            if point_in_rect(mx, my, (mx_box + 155, geom_y + 34, 115, 26)):
                return ("roi_field_num", "center", 0)
            if point_in_rect(mx, my, (mx_box + 280, geom_y + 34, 115, 26)):
                return ("roi_field_num", "center", 1)
            if point_in_rect(mx, my, (mx_box + 405, geom_y + 34, 115, 26)):
                return ("roi_field_num", "center", 2)
            if point_in_rect(mx, my, (mx_box + 155, geom_y + 70, 115, 26)):
                return ("roi_field_num", "size", 0)
            if point_in_rect(mx, my, (mx_box + 280, geom_y + 70, 115, 26)):
                return ("roi_field_num", "size", 1)
            if point_in_rect(mx, my, (mx_box + 405, geom_y + 70, 115, 26)):
                return ("roi_field_num", "size", 2)
            if point_in_rect(mx, my, (mx_box + 155, geom_y + 106, 115, 26)):
                return ("roi_field_num", "rotation", 0)
            if point_in_rect(mx, my, (mx_box + 280, geom_y + 106, 115, 26)):
                return ("roi_field_num", "rotation", 1)
            if point_in_rect(mx, my, (mx_box + 405, geom_y + 106, 115, 26)):
                return ("roi_field_num", "rotation", 2)

            if mx < mx_box or mx > mx_box + mw or my < my_box or my > my_box + mh:
                return "roi_modal_mask"
            return "roi_modal_body"

        # 0.1 生产机制业务说明弹窗模式
        if state.is_help_modal_open:
            modal_w, modal_h = HELP_MODAL_W, HELP_MODAL_H
            mx_box = (self.r.canvas_w - modal_w) // 2
            my_box = (self.r.canvas_h - modal_h) // 2
            bx1 = mx_box + modal_w - 116
            by1 = my_box + 11
            bx2 = bx1 + 100
            by2 = by1 + 32
            # 关闭按钮 (带 6px 容差热区)
            if (bx1 - 6) <= mx <= (bx2 + 6) and (by1 - 6) <= my <= (by2 + 6):
                return "help_close"
            # 外部半透明遮罩
            if mx < mx_box or mx > mx_box + modal_w or my < my_box or my > my_box + modal_h:
                return "help_mask"
            return "help_modal_body"

        # 1. 常规看板模式
        # 顶部 Header 交互 (右侧动态区页签 Tab + [退出] 按钮)
        if 0 <= my <= 50:
            if not state.tab_bar._last_layout:
                from tools.workspace_hub.hub_renderer import HEADER_TAB_X0, HEADER_TAB_Y0, HEADER_TAB_H
                state.tab_bar.compute_layout((HEADER_TAB_X0, HEADER_TAB_Y0, 510, HEADER_TAB_H))
            hit_tab = state.tab_bar.hit_test(mx, my)
            if hit_tab:
                return ("hdr_tab_key", hit_tab)
            # [退出] 按钮
            if BTN_EXIT_X0 <= mx <= BTN_EXIT_X0 + BTN_EXIT_W and BTN_EXIT_Y0 <= my <= BTN_EXIT_Y0 + BTN_EXIT_H:
                return "btn_exit"

        # 左侧面板按钮与两层树交互
        if 0 <= mx <= 340:
            div_y1 = 604
            btn1_y = div_y1 + 10
            if 10 <= mx <= 330 and btn1_y <= my <= btn1_y + 40:
                return "btn_new_workspace"

            # 遍历两层树节点
            tree_items = self.get_tree_layout(state)
            for item in tree_items:
                if item["type"] == "workspace":
                    if point_in_rect(mx, my, item["arrow_rect"]):
                        return ("tree_ws_toggle", item["ws_idx"], item["workspace"].workspace_id)
                    if point_in_rect(mx, my, item["body_rect"]):
                        return ("tree_ws_select", item["ws_idx"])
                elif item["type"] == "frame":
                    if point_in_rect(mx, my, item["rect"]):
                        return ("tree_frame_select", item["ws_idx"], item["frame_id"])

        # 右侧动态区页签内容按钮 (x: 340~960)
        if state.gallery.view_mode == HubState.VIEW_EXPANDED:
            if 58 <= my <= 92:
                if 680 <= mx <= 740:
                    return "exp_prev"
                if 746 <= mx <= 806:
                    return "exp_next"
                if 812 <= mx <= 880:
                    return "album_delete"
                if 886 <= mx <= 950:
                    return "exp_restore"

        # 工位大盘看板 (TAB_REPORT) 卡片内嵌按钮
        if state.active_tab == HubState.TAB_REPORT and state.gallery.view_mode == HubState.VIEW_STANDARD:
            if point_in_rect(mx, my, WS_BTN_RENAME):
                return "ws_rename"
            if point_in_rect(mx, my, WS_BTN_OPEN_DIR):
                return "ws_open_dir"
            if point_in_rect(mx, my, WS_BTN_TOGGLE_PROD_MODE):
                return "ws_toggle_prod_mode"
            if point_in_rect(mx, my, WS_BTN_EDIT_DESC):
                return "ws_edit_desc"
            if point_in_rect(mx, my, WS_BTN_SYNC_DATA):
                return "ws_sync_data"
            if point_in_rect(mx, my, WS_BTN_CLONE):
                return "ws_clone"
            if point_in_rect(mx, my, WS_BTN_DELETE):
                return "ws_delete"
            if point_in_rect(mx, my, WS_BTN_NEW_FRAME):
                return "btn_add_frame"

        # 坐标系专属页签 1: 机构参数与 Tag 分段 (TAB_FRAME_POSE_TAGS)
        if state.active_tab == HubState.TAB_FRAME_POSE_TAGS and state.gallery.view_mode == HubState.VIEW_STANDARD:
            if point_in_rect(mx, my, FRAME_EDIT_POSE_BTN):
                return "btn_edit_frame_pose"
            if point_in_rect(mx, my, FRAME_TAG_EDIT_SIZE_BTN):
                return "btn_edit_marker_size"
            if self.r._should_show_tag_bound_tooltip(state, (mx, my)):
                return "tag_bound_help"
            cur_frame = state.geometry.get_selected_frame()
            if cur_frame:
                tag_range = state.geometry.get_frame_tag_range(cur_frame.frame_id)
                for slot_idx in range(10):
                    chip_rect = frame_tag_chip_rect(slot_idx)
                    if point_in_rect(mx, my, chip_rect):
                        tag_id = tag_range[slot_idx]
                        cx, cy, cw, ch = chip_rect
                        if my >= cy + ch - 24:
                            return ("frame_tag_edit_xyz", tag_id)
                        return ("frame_tag_toggle", tag_id)

        # 坐标系专属页签 2: 3D ROI 空间物件 (TAB_FRAME_ROIS)
        if state.active_tab == HubState.TAB_FRAME_ROIS and state.gallery.view_mode == HubState.VIEW_STANDARD:
            if point_in_rect(mx, my, FRAME_ADD_ROI_BTN):
                return "btn_add_frame_roi"
            cur_frame = state.geometry.get_selected_frame()
            if cur_frame:
                rois = state.geometry.get_frame_rois(cur_frame.frame_id)
                total_rois = len(rois)
                max_vis = HubState.ROI_VISIBLE_COUNT
                max_offset = max(0, total_rois - max_vis)
                offset = max(0, min(state.geometry.roi_scroll_offset, max_offset))

                if total_rois > max_vis:
                    if point_in_rect(mx, my, FRAME_ROI_PREV_BTN):
                        return "btn_roi_prev"
                    if point_in_rect(mx, my, FRAME_ROI_NEXT_BTN):
                        return "btn_roi_next"
                    if point_in_rect(mx, my, FRAME_ROI_SCROLL_TRACK):
                        return ("roi_scrollbar_click", my)

                visible_rois = rois[offset : offset + max_vis]
                for i, r in enumerate(visible_rois):
                    if point_in_rect(mx, my, frame_roi_edit_btn(i)):
                        return ("frame_roi_edit", r.roi_id)
                    if point_in_rect(mx, my, frame_roi_del_btn(i)):
                        return ("frame_roi_delete", r.roi_id)

        # 图片卡片网格墙卡片 Hover (标定相册与生产相册通用)
        if state.active_tab in (HubState.TAB_CALIB_IMAGES, HubState.TAB_PROD_IMAGES):
            cell_idx = grid_hit_test(mx, my)
            if cell_idx is not None:
                offset = state.gallery.prod_grid_offset if state.active_tab == HubState.TAB_PROD_IMAGES else state.gallery.image_grid_offset
                return ("grid_item", offset + cell_idx)

        return None

