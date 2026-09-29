"""
空间建图工作站 - 左栏帧列表渲染 Mixin (MappingFrameListMixin)
================================================================================
承载 MappingRenderer 的左栏绘制分区：
1. render_left_frame_list: 高信息密度垂直紧凑帧列表 与 逐帧多轮残差演进矩阵宽表
仅包含纯绘制方法, 不持有任何状态; 通过 self 依赖宿主 MappingRenderer 的其他方法,
由 MRO 解析跨分区调用。
"""

import os
from typing import Any
import cv2
import numpy as np

from src.utils.text_rendering import measure_text, put_text
from src.utils.viewport_manager import draw_styled_button
from src.utils.gui_components import draw_dropdown_button
from tools.spatial_mapping_studio.mapping_ui_common import FILTER_MODE_OPTIONS, SORT_MODE_OPTIONS


class MappingFrameListMixin:
    """左栏帧列表渲染 Mixin (由宿主类 MappingRenderer 组合)"""

    def render_left_frame_list(self, app: Any, canvas: np.ndarray, x: int, y: int, w: int, h: int):
        """左栏：高信息密度垂直紧凑帧列表 或 逐帧多轮残差演进矩阵宽表大视图"""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (22, 24, 30), -1)
        cv2.line(canvas, (x + w, y), (x + w, y + h), (50, 54, 66), 1)

        is_matrix = getattr(app, "matrix_view_mode", False)
        headers = getattr(app.data_mgr, "convergence_headers", [])
        matrix = getattr(app.data_mgr, "frame_convergence_matrix", {})

        header_h = 36
        dd_y1 = y + 6
        dd_y2 = y + header_h

        # 1. 顶部操作栏自适应排版：切换按钮 + 筛选下拉框 + 排序下拉框
        btn_w = 98 if is_matrix else 86
        btn_x1 = x + w - btn_w - 8
        btn_x2 = x + w - 8

        avail_w = max(120, btn_x1 - (x + 8) - 8)
        dd1_w = avail_w // 2
        dd2_w = avail_w - dd1_w - 6

        dd1_x1 = x + 8
        dd1_x2 = dd1_x1 + dd1_w
        dd2_x1 = dd1_x2 + 6
        dd2_x2 = dd2_x1 + dd2_w

        # 筛选范围下拉框
        cur_filter_label = dict(FILTER_MODE_OPTIONS).get(app.filter_mode, "全部帧")
        is_f_open = (app.active_dropdown == "FILTER_DROPDOWN")
        draw_dropdown_button(canvas, (dd1_x1, dd_y1, dd1_x2, dd_y2), cur_filter_label,
                             is_open=is_f_open, mouse_pos=app.mouse_pos)
        app.dropdown_boxes["FILTER_DROPDOWN"] = {
            "rect": (dd1_x1, dd_y1, dd1_x2, dd_y2),
            "options": FILTER_MODE_OPTIONS,
            "active_key": app.filter_mode
        }
        app.gui_buttons.append(("TOGGLE_FILTER_DROPDOWN", (dd1_x1, dd_y1, dd1_x2, dd_y2), "FILTER_DROPDOWN"))

        # 排序方式下拉框
        cur_sort_label = dict(SORT_MODE_OPTIONS).get(app.sort_mode, "文件名升序")
        short_sort = cur_sort_label.split(" ")[0] if "(" in cur_sort_label else cur_sort_label
        is_s_open = (app.active_dropdown == "SORT_DROPDOWN")
        draw_dropdown_button(canvas, (dd2_x1, dd_y1, dd2_x2, dd_y2), short_sort,
                             is_open=is_s_open, mouse_pos=app.mouse_pos)
        app.dropdown_boxes["SORT_DROPDOWN"] = {
            "rect": (dd2_x1, dd_y1, dd2_x2, dd_y2),
            "options": SORT_MODE_OPTIONS,
            "active_key": app.sort_mode
        }
        app.gui_buttons.append(("TOGGLE_SORT_DROPDOWN", (dd2_x1, dd_y1, dd2_x2, dd_y2), "SORT_DROPDOWN"))

        # 矩阵模式切换按钮 (一键展开/收起 10 轮残差对比大表)
        btn_txt = "⊟ 紧凑 (X)" if is_matrix else "⊞ 矩阵 (X)"
        btn_type = "primary" if is_matrix else "secondary"
        draw_styled_button(canvas, (btn_x1, dd_y1, btn_x2, dd_y2), btn_txt,
                           mouse_pos=app.mouse_pos, btn_type=btn_type)
        app.gui_buttons.append(("TOGGLE_MATRIX_VIEW", (btn_x1, dd_y1, btn_x2, dd_y2), "TOGGLE_MATRIX_VIEW"))

        filtered_indices = app._get_filtered_indices()
        if not filtered_indices:
            put_text(canvas, "当前筛选条件下无图像", (x + 60, y + header_h + 45),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 120, 120), 1, cv2.LINE_AA)
            return

        # 2. 列表内容区布局与渲染 (统一接入 ScrollableListBox 工业级组件)
        app.frame_list_box.scroll_offset = app.scroll_offset
        try:
            cur_sel_idx = filtered_indices.index(app.current_img_idx)
        except ValueError:
            cur_sel_idx = -1
        app.frame_list_box.selected_index = cur_sel_idx

        if is_matrix:
            # ==================== 【模式 A: 逐帧多轮残差演进矩阵宽表】 ====================
            th_h = 24
            th_y1 = y + header_h + 6
            th_y2 = th_y1 + th_h
            list_y = th_y2 + 4

            # 绘制矩阵表头背景
            cv2.rectangle(canvas, (x + 6, th_y1), (x + w - 6, th_y2), (32, 36, 46), -1)
            cv2.rectangle(canvas, (x + 6, th_y1), (x + w - 6, th_y2), (52, 58, 72), 1)

            c_name_w = 110
            c_tag_w = 34
            c_round_w = 54
            c_drop_w = 68

            # 表头固定列
            put_text(canvas, "图像帧", (x + 14, th_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 205, 215), 1, cv2.LINE_AA)
            put_text(canvas, "Tag", (x + 14 + c_name_w, th_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 205, 215), 1, cv2.LINE_AA)

            cur_col_x = x + 14 + c_name_w + c_tag_w
            active_headers = headers if headers else ["基准(R0)"]
            for col_idx, h_name in enumerate(active_headers):
                is_latest_col = (col_idx == len(active_headers) - 1)
                th_color = (0, 240, 255) if is_latest_col else (180, 185, 195)
                if is_latest_col and len(active_headers) > 1:
                    cv2.rectangle(canvas, (cur_col_x - 2, th_y1 + 2), (cur_col_x + c_round_w - 4, th_y2 - 2), (25, 60, 75), -1)
                put_text(canvas, h_name, (cur_col_x + 6, th_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, th_color, 1 if not is_latest_col else 2, cv2.LINE_AA)
                cur_col_x += c_round_w

            put_text(canvas, "累计降幅", (cur_col_x + 4, th_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 240, 120), 1, cv2.LINE_AA)

            def _draw_matrix_row(cvs, rect, orig_img_idx, idx, is_hover, is_selected):
                rx1, ry1, rw, rh = rect
                rx2, ry2 = rx1 + rw, ry1 + rh
                p = app.image_files[orig_img_idx]
                bname = os.path.basename(p)
                meta = app.frame_metrics_cache.get(bname, {})

                btn_id = f"SELECT_FRAME_{orig_img_idx}"
                app.gui_buttons.append((btn_id, (rx1, ry1, rx2, ry2), orig_img_idx))

                dot_y = ry1 + rh // 2
                is_excl = meta.get("is_excluded", False)
                dot_c = (0, 0, 240) if is_excl else ((0, 180, 255) if meta.get("mean_err", 0.0) > 0.5 else (0, 230, 80))
                cv2.circle(cvs, (rx1 + 8, dot_y), 3, dot_c, -1)

                name_stem = bname.replace(".png", "").replace(".jpg", "")
                short_name = name_stem if len(name_stem) <= 10 else name_stem[:9] + "…"
                name_col = (255, 255, 255) if is_selected else (200, 205, 215)
                put_text(cvs, short_name, (rx1 + 16, dot_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, name_col, 1, cv2.LINE_AA)

                tag_cnt = meta.get("tag_count", 0)
                put_text(cvs, f"{tag_cnt}T", (rx1 + 8 + c_name_w, dot_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (140, 150, 165), 1, cv2.LINE_AA)

                row_vals = matrix.get(bname, [])
                r_col_x = rx1 + 8 + c_name_w + c_tag_w
                for c_idx in range(len(active_headers)):
                    is_latest_c = (c_idx == len(active_headers) - 1)
                    val = row_vals[c_idx] if c_idx < len(row_vals) else None
                    if is_latest_c and len(active_headers) > 1 and not is_selected:
                        cv2.rectangle(cvs, (r_col_x - 2, ry1 + 2), (r_col_x + c_round_w - 6, ry2 - 2), (20, 42, 54), -1)

                    if val is None or is_excl:
                        v_txt = "EXCL" if is_excl else "--"
                        v_col = (90, 95, 110) if not is_excl else (0, 0, 200)
                        put_text(cvs, v_txt, (r_col_x + 4, dot_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, v_col, 1, cv2.LINE_AA)
                    else:
                        v_txt = f"{val:.1f}" if val >= 100.0 else f"{val:.2f}"
                        v_col = (0, 180, 255) if val > 1.0 else ((0, 220, 255) if val > 0.5 else (0, 240, 100))
                        put_text(cvs, v_txt, (r_col_x + 4, dot_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, v_col, 1, cv2.LINE_AA)
                    r_col_x += c_round_w

                first_v = row_vals[0] if (row_vals and row_vals[0] is not None) else None
                last_v = next((rv for rv in reversed(row_vals) if rv is not None), None)
                if first_v is not None and last_v is not None and first_v > 0.001:
                    drop_val = first_v - last_v
                    drop_pct = (drop_val / first_v) * 100.0
                    if drop_pct >= 5.0:
                        pct_txt = f"↓{drop_pct:.0f}%" if drop_pct >= 10.0 else f"↓{drop_pct:.1f}%"
                        pct_col = (0, 240, 255)
                    elif drop_pct <= -5.0:
                        pct_txt = f"↑{abs(drop_pct):.0f}%"
                        pct_col = (0, 100, 255)
                    else:
                        pct_txt = "0%"
                        pct_col = (150, 160, 175)
                else:
                    pct_txt = "--"
                    pct_col = (110, 115, 125)
                put_text(cvs, pct_txt, (r_col_x + 4, dot_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, pct_col, 1, cv2.LINE_AA)

            app.frame_list_box.item_height = 30
            app.frame_list_box.render(
                canvas=canvas,
                rect=(x + 6, list_y, w - 12, h - (list_y - y) - 10),
                items=filtered_indices,
                draw_item_callback=_draw_matrix_row,
                mouse_pos=app.mouse_pos,
                empty_text="暂无匹配帧",
            )
            app.scroll_offset = app.frame_list_box.scroll_offset

        else:
            # ==================== 【模式 B: 高信息密度垂直紧凑帧列表】 ====================
            list_y = y + header_h + 8
            box_x = x + 6
            box_w = w - 12
            box_h = h - header_h - 16

            def _draw_compact_frame(cvs, rect, orig_img_idx, idx, is_hover, is_selected):
                rx1, ry1, rw, rh = rect
                rx2, ry2 = rx1 + rw, ry1 + rh
                p = app.image_files[orig_img_idx]
                bname = os.path.basename(p)
                meta = app.frame_metrics_cache.get(bname, {})

                btn_id = f"SELECT_FRAME_{orig_img_idx}"
                app.gui_buttons.append((btn_id, (rx1, ry1, rx2, ry2), orig_img_idx))

                dot_y = ry1 + rh // 2
                is_excl = meta.get("is_excluded", False)
                dot_c = (0, 0, 240) if is_excl else ((0, 180, 255) if meta.get("mean_err", 0.0) > 0.5 else (0, 230, 80))
                cv2.circle(cvs, (rx1 + 12, dot_y), 4, dot_c, -1)

                txt_col = (255, 255, 255) if is_selected else (200, 200, 200)
                short_name = bname if len(bname) <= 15 else bname[:12] + "..."
                put_text(cvs, short_name, (rx1 + 22, dot_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, txt_col, 1, cv2.LINE_AA)

                tag_cnt = meta.get("tag_count", 0)
                put_text(cvs, f"{tag_cnt}T", (rx1 + 162, dot_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 160, 175), 1, cv2.LINE_AA)

                if is_excl:
                    put_text(cvs, "EXCL", (rx1 + 212, dot_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 240), 1, cv2.LINE_AA)
                else:
                    err_val = meta.get('mean_err', 0.0)
                    err_str = f"{err_val:.2f}px"
                    err_col = (0, 200, 255) if err_val > 0.5 else (0, 240, 100)
                    put_text(cvs, err_str, (rx1 + 212, dot_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, err_col, 1, cv2.LINE_AA)

            app.frame_list_box.item_height = 36
            app.frame_list_box.render(
                canvas=canvas,
                rect=(box_x, list_y, box_w, box_h),
                items=filtered_indices,
                draw_item_callback=_draw_compact_frame,
                mouse_pos=app.mouse_pos,
                empty_text="暂无匹配帧",
            )
            app.scroll_offset = app.frame_list_box.scroll_offset
