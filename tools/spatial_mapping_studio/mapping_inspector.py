"""
AprilTag 离线标定工作站 - 右栏检视面板与剪枝浮层卡片渲染 Mixin (MappingInspectorMixin)
================================================================================
承载 MappingRenderer 的右栏与剪枝浮层绘制分区：
1. render_right_inspector: 精简属性面板 / 标靶残差清单 / 单帧病因切片诊断双模面板
2. render_prune_ba_card: 迭代残差剪枝平差运行中多轮收敛监控卡片
3. render_prune_settlement_card: 智能剪枝平差结算单对比卡片
仅包含纯绘制方法, 不持有任何状态; 通过 self 依赖宿主 MappingRenderer 的其他方法,
由 MRO 解析跨分区调用。
"""

import os
from typing import Any, Dict, List
import cv2
import numpy as np

from src.ui.text_rendering import measure_text, put_text
from src.ui.viewport_manager import draw_styled_button


class MappingInspectorMixin:
    """右栏检视面板与剪枝浮层卡片渲染 Mixin (由宿主类 MappingRenderer 组合)"""

    def render_right_inspector(self, app: Any, canvas: np.ndarray, x: int, y: int, w: int, h: int):
        """右栏：精简属性与标靶残差清单/病因切片诊断双模面板 (瘦身宽度: 180px)"""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (22, 24, 30), -1)
        cv2.line(canvas, (x, y), (x, y + h), (50, 54, 66), 1)

        if not app.data_mgr.image_files:
            return

        cur_file = app.data_mgr.image_files[app.data_mgr.current_img_idx]
        bname = os.path.basename(cur_file)
        meta = app.data_mgr.frame_metrics_cache.get(bname, {})

        # 1. 顶部当前帧摘要卡片
        put_text(canvas, bname, (x + 8, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

        # 状态切换按钮 (保留 / 剔除)
        is_excl = meta.get("is_excluded", False)
        b_type = "danger" if is_excl else "success"
        b_label = "恢复此帧 (T)" if is_excl else "剔除此帧 (T)"
        draw_styled_button(canvas, (x + 8, y + 30, x + w - 8, y + 54), b_label,
                           mouse_pos=app.mouse_pos, btn_type=b_type)
        app.gui_buttons.append(("TOGGLE_FRAME_STATUS", (x + 8, y + 30, x + w - 8, y + 54), bname))

        # 底部动作区域基准 Y (预留 3 个紧凑按钮: 超精提取、病因诊断/常规面板、Robot 跟踪)
        diag_y = y + h - 88

        # 2. 中间区域：根据 show_frame_diagnostics 模式切换
        is_diag_mode = getattr(app, "show_frame_diagnostics", False)
        if is_diag_mode:
            # 渲染【单帧深度切片病因诊断】面板
            diag_top_y = y + 62
            cv2.line(canvas, (x + 8, diag_top_y), (x + w - 8, diag_top_y), (45, 48, 58), 1)
            put_text(canvas, "单帧病因切片诊断", (x + 8, diag_top_y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 215, 255), 1, cv2.LINE_AA)

            diag = getattr(app, "current_diagnostics", {})
            cy = diag_top_y + 36

            # 指标1：清晰度 Laplace
            lap = diag.get("sharpness", diag.get("laplacian_var", 0.0))
            lap_g = diag.get("sharpness_grade", "")
            put_text(canvas, f"清晰度: {lap:.1f} {lap_g}", (x + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 205, 215), 1, cv2.LINE_AA)
            cy += 20

            # 指标2：对比度 RMS
            c_rms = diag.get("contrast", diag.get("contrast_rms", 0.0))
            c_g = diag.get("contrast_grade", "")
            put_text(canvas, f"对比度: {c_rms:.1f} {c_g}", (x + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 205, 215), 1, cv2.LINE_AA)
            cy += 20

            # 指标3：平均亮度 Mean
            m_lum = diag.get("brightness", diag.get("mean_intensity", 0.0))
            b_g = diag.get("brightness_grade", "")
            put_text(canvas, f"亮度: {m_lum:.1f} {b_g}", (x + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 205, 215), 1, cv2.LINE_AA)
            cy += 24

            # 拒检四边形候选
            rej_c = diag.get("rejected_quads_count", diag.get("false_rejections_count", 0))
            put_text(canvas, f"畸变/超小候选: {rej_c}个", (x + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 175, 185), 1, cv2.LINE_AA)
            cy += 20

            # 理论漏检标靶
            missing = diag.get("missing_projected_tags", []) or diag.get("missing_theoretical_tags", [])
            if missing:
                m_tids = ",".join([f"#{m['tag_id']}" for m in missing])
                put_text(canvas, f"理论漏检: {len(missing)}个", (x + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 140, 255), 1, cv2.LINE_AA)
                cy += 18
                put_text(canvas, f"目标: {m_tids}", (x + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 180, 255), 1, cv2.LINE_AA)
            else:
                put_text(canvas, "理论漏检: 无 (全捕获)", (x + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 220, 100), 1, cv2.LINE_AA)

        else:
            # 渲染常规【标靶与残差清单】面板 (按照残差降序排序: 离差最大排在最上方)
            list_y = y + 62
            cv2.line(canvas, (x + 8, list_y), (x + w - 8, list_y), (45, 48, 58), 1)
            obs_list = meta.get("observations", [])
            tag_errors = meta.get("tag_errors", {})
            # FR-9.6 世界系坐标 (平差锚定后每枚标靶的 XYZ)
            tags_meta = (app.data_mgr.tags_map_data or {}).get("tags", {})
            put_text(canvas, f"标靶残差+世界XYZ ({len(obs_list)}) 降序↓", (x + 8, list_y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 220, 255), 1, cv2.LINE_AA)

            # 按残差降序排序：离差最大的坏标靶置顶优先显示
            sorted_obs_list = sorted(
                obs_list,
                key=lambda o: (tag_errors.get(o.get("tag_id", -1), -1.0), -o.get("tag_id", 0)),
                reverse=True
            )

            box_x = x + 6
            box_y = list_y + 24
            box_w = w - 12
            box_h = max(40, diag_y - box_y - 6)

            def _draw_tag_item(cvs, rect, obs, idx, is_hover, is_selected):
                rx1, ry1, rw, rh = rect
                rx2, ry2 = rx1 + rw, ry1 + rh
                tid = obs["tag_id"]
                keep = obs.get("keep", True)
                err_val = tag_errors.get(tid, 0.0)

                app.gui_buttons.append((f"TOGGLE_TAG_{tid}", (rx1, ry1, rx2, ry2), tid))

                dot_c = (0, 220, 80) if keep else (0, 0, 220)
                cv2.circle(cvs, (rx1 + 10, ry1 + 11), 3, dot_c, -1)

                t_col = (230, 230, 230) if keep else (120, 120, 120)
                put_text(cvs, f"#{tid}", (rx1 + 18, ry1 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.46, t_col, 1, cv2.LINE_AA)

                err_str = f"{err_val:.2f}px" if keep else "EXCL"
                if not keep:
                    err_c = (120, 120, 120)
                elif err_val > 1.0:
                    err_c = (0, 100, 255)  # 离差严重 (>1px) 鲜艳红色
                elif err_val > 0.5:
                    err_c = (0, 200, 255)  # 离差偏大 (>0.5px) 醒目金黄
                else:
                    err_c = (0, 230, 80)   # 优良 (<=0.5px) 荧光绿
                (ew, _), _ = measure_text(err_str, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                put_text(cvs, err_str, (rx2 - ew - 6, ry1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, err_c, 1, cv2.LINE_AA)

                # 第二行: FR-9.6 世界系坐标 XYZ (mm)
                rec = tags_meta.get(tid) or {}
                pos_mm = rec.get("position_mm")
                if pos_mm and len(pos_mm) >= 3:
                    xyz_str = f"X{pos_mm[0]:.0f} Y{pos_mm[1]:.0f} Z{pos_mm[2]:.0f} mm"
                    xyz_c = (140, 190, 220)
                else:
                    xyz_str = "XYZ: --"
                    xyz_c = (110, 115, 125)
                put_text(cvs, xyz_str, (rx1 + 18, ry1 + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.36, xyz_c, 1, cv2.LINE_AA)

            app.tag_list_box.render(
                canvas=canvas,
                rect=(box_x, box_y, box_w, box_h),
                items=sorted_obs_list,
                draw_item_callback=_draw_tag_item,
                mouse_pos=app.mouse_pos,
                empty_text="暂无标靶观测",
            )

        # 3. 底部 3 个紧凑快捷动作
        cv2.line(canvas, (x + 8, diag_y), (x + w - 8, diag_y), (45, 48, 58), 1)

        # 按钮 1: 超精提取 (E)
        draw_styled_button(canvas, (x + 8, diag_y + 4, x + w - 8, diag_y + 28), "超精提取 (E)",
                           mouse_pos=app.mouse_pos, btn_type="primary")
        app.gui_buttons.append(("SUPER_EXTRACT_FRAME", (x + 8, diag_y + 4, x + w - 8, diag_y + 28), bname))

        # 按钮 2: 病因诊断 (D) / 常规面板 (D)
        d_lbl = "常规面板 (D)" if is_diag_mode else "病因诊断 (D)"
        d_typ = "normal" if is_diag_mode else "warning"
        draw_styled_button(canvas, (x + 8, diag_y + 32, x + w - 8, diag_y + 56), d_lbl,
                           mouse_pos=app.mouse_pos, btn_type=d_typ)
        app.gui_buttons.append(("DIAGNOSE_FRAME", (x + 8, diag_y + 32, x + w - 8, diag_y + 56), bname))

        # 按钮 3: Robot 在线跟踪
        draw_styled_button(canvas, (x + 8, diag_y + 60, x + w - 8, diag_y + 84), "Robot 跟踪",
                           mouse_pos=app.mouse_pos, btn_type="success")
        app.gui_buttons.append(("LAUNCH_TRACKER", (x + 8, diag_y + 60, x + w - 8, diag_y + 84), "LAUNCH_TRACKER"))

    def render_prune_ba_card(self, app: Any, canvas: np.ndarray, w: int, h: int):
        """居中展示工序 5-Auto: 迭代残差剪枝平差运行中多轮收敛监控卡片 (集成实时多轮报告列表)"""
        card_w, card_h = 760, 360
        cx1, cy1 = (w - card_w) // 2, (h - card_h) // 2
        overlay = canvas.copy()
        cv2.rectangle(overlay, (cx1, cy1), (cx1 + card_w, cy1 + card_h), (18, 20, 26), -1)
        cv2.addWeighted(overlay, 0.94, canvas, 0.06, 0, canvas)
        cv2.rectangle(canvas, (cx1, cy1), (cx1 + card_w, cy1 + card_h), (255, 140, 0), 2)

        # 1. 标题与急停按钮
        r = getattr(app.ba_runner, "prune_round", 1)
        max_r = getattr(app.ba_runner, "max_prune_rounds", 10)
        title_txt = f"工序 5-Auto: 迭代残差剪枝平差监控 (第 {r}/{max_r} 轮)..."
        put_text(canvas, title_txt, (cx1 + 20, cy1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

        # 急停按钮 (右上角)
        btn_w, btn_h = 105, 24
        btn_x1 = cx1 + card_w - btn_w - 18
        btn_y1 = cy1 + 12
        btn_x2 = btn_x1 + btn_w
        btn_y2 = btn_y1 + btn_h
        draw_styled_button(canvas, (btn_x1, btn_y1, btn_x2, btn_y2), "急停 (Space)",
                           mouse_pos=app.mouse_pos, btn_type="danger")
        app.gui_buttons.append(("STOP_PRUNE", (btn_x1, btn_y1, btn_x2, btn_y2), "STOP_PRUNE"))

        # 2. 进度条与百分比
        pct = max(0.0, min(1.0, app.ba_runner.ba_progress))
        pct_int = int(round(pct * 100))
        pct_str = f"{pct_int}%"
        (pw, _), _ = measure_text(pct_str, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 2)
        put_text(canvas, pct_str, (btn_x1 - pw - 14, cy1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 180, 0), 2, cv2.LINE_AA)

        bar_x1 = cx1 + 20
        bar_y1 = cy1 + 42
        bar_x2 = cx1 + card_w - 20
        bar_y2 = bar_y1 + 10
        bar_w = bar_x2 - bar_x1

        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (30, 34, 44), -1)
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (65, 72, 88), 1)

        fill_w = int(bar_w * pct)
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), (255, 140, 0), -1)
            cv2.line(canvas, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y1), (255, 210, 140), 1)

        # 3. 统计指标胶囊栏 (淘汰数、基准RMSE、当前RMSE、累计改善)
        cap_y1 = cy1 + 58
        cap_y2 = cap_y1 + 28
        cv2.rectangle(canvas, (bar_x1, cap_y1), (bar_x2, cap_y2), (25, 28, 36), -1)
        cv2.rectangle(canvas, (bar_x1, cap_y1), (bar_x2, cap_y2), (48, 54, 68), 1)

        history: List[Dict[str, Any]] = getattr(app.ba_runner, "prune_history", [])
        total_pruned = sum(len(item.get("pruned", [])) for item in history)
        init_rmse = getattr(app.ba_runner, "initial_rmse", app.data_mgr.global_rmse)
        curr_rmse = app.data_mgr.global_rmse
        cum_delta = max(0.0, init_rmse - curr_rmse) if init_rmse > 0 else 0.0
        cum_pct = (cum_delta / init_rmse * 100.0) if init_rmse > 0 else 0.0

        txt_p = f"累计淘汰: {total_pruned} 个"
        txt_i = f"初始基准: {init_rmse:.2f}px"
        txt_c = f"当前残差: {curr_rmse:.2f}px"
        txt_d = f"累计改善: ↓{cum_delta:.2f}px ({cum_pct:.1f}%)"

        put_text(canvas, txt_p, (bar_x1 + 14, cap_y1 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)
        put_text(canvas, txt_i, (bar_x1 + 160, cap_y1 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 185, 195), 1, cv2.LINE_AA)
        put_text(canvas, txt_c, (bar_x1 + 330, cap_y1 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 210, 100), 1, cv2.LINE_AA)
        put_text(canvas, txt_d, (bar_x1 + 500, cap_y1 + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 240, 120), 1, cv2.LINE_AA)

        # 4. 当前运行主阶段与动态细节
        stg_txt = app.ba_runner.ba_stage_text or "智能迭代剪枝平差管线推进中..."
        sub_txt = app.ba_runner.ba_sub_text or "正在执行全场景 BA 平差与共视安全守门..."
        put_text(canvas, stg_txt, (cx1 + 20, cy1 + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)
        put_text(canvas, sub_txt, (cx1 + 20, cy1 + 122), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 190, 80), 1, cv2.LINE_AA)

        # 5. 【核心实时报告表格】 (Live Settlement Table)
        tbl_x1 = bar_x1
        tbl_y1 = cy1 + 134
        tbl_w = bar_w
        th_h = 24
        tr_h = 24

        # 表头
        cv2.rectangle(canvas, (tbl_x1, tbl_y1), (tbl_x1 + tbl_w, tbl_y1 + th_h), (34, 38, 48), -1)
        cv2.rectangle(canvas, (tbl_x1, tbl_y1), (tbl_x1 + tbl_w, tbl_y1 + th_h), (55, 62, 78), 1)

        c_round_w = 60
        c_item_w = 320
        c_before_w = 110
        c_after_w = 110
        c_delta_w = tbl_w - c_round_w - c_item_w - c_before_w - c_after_w

        tx0 = tbl_x1 + 8
        tx1 = tbl_x1 + c_round_w
        tx2 = tx1 + c_item_w
        tx3 = tx2 + c_before_w
        tx4 = tx3 + c_after_w

        put_text(canvas, "轮次", (tx0, tbl_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 205, 215), 1, cv2.LINE_AA)
        put_text(canvas, "淘汰坏样本 (图像 / Tag / 离差)", (tx1 + 6, tbl_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 205, 215), 1, cv2.LINE_AA)
        put_text(canvas, "平差前残差", (tx2 + 6, tbl_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 205, 215), 1, cv2.LINE_AA)
        put_text(canvas, "平差后残差", (tx3 + 6, tbl_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 205, 215), 1, cv2.LINE_AA)
        put_text(canvas, "进步幅度", (tx4 + 6, tbl_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 205, 215), 1, cv2.LINE_AA)

        # 动态行组合: 历史完成轮次 + 当前正在求解轮次
        display_rows = []
        for h_item in history:
            rnd = f"#{h_item.get('round', 1)}"
            pruned_info = ", ".join([f"{bn} #{tid} ({err:.1f}px)" for bn, tid, err, _ in h_item.get("pruned", [])])
            b_rmse = f"{h_item.get('rmse_before', 0.0):.2f}px"
            a_rmse = f"{h_item.get('rmse_after', 0.0):.2f}px"
            d_rmse = h_item.get('delta_rmse', 0.0)
            d_txt = f"↓{d_rmse:+.2f}px" if d_rmse >= 0 else f"{d_rmse:+.2f}px"
            display_rows.append((rnd, pruned_info, b_rmse, a_rmse, d_txt, False))

        # 当前正在求解的进行中行
        curr_target = getattr(app.ba_runner, "current_pruning_target", "")
        if curr_target:
            curr_row = (f"#{r}", curr_target, f"{curr_rmse:.2f}px", "--", "求解中...", True)
            display_rows.append(curr_row)

        # 仅展示最新的 5 行，保证排版整洁
        rows_to_show = display_rows[-5:] if len(display_rows) > 5 else display_rows
        max_rows = 5
        curr_row_y = tbl_y1 + th_h

        if not rows_to_show:
            cv2.rectangle(canvas, (tbl_x1, curr_row_y), (tbl_x1 + tbl_w, curr_row_y + tr_h * 2), (20, 22, 28), -1)
            cv2.rectangle(canvas, (tbl_x1, curr_row_y), (tbl_x1 + tbl_w, curr_row_y + tr_h * 2), (45, 50, 62), 1)
            put_text(canvas, "正在执行首轮共视拓扑分析与基准残差排查，即将生成实时对比明细...",
                        (tbl_x1 + 18, curr_row_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 150, 165), 1, cv2.LINE_AA)
            curr_row_y += tr_h * 2
        else:
            for idx, (rnd_s, item_s, b_s, a_s, d_s, is_active) in enumerate(rows_to_show):
                ry1 = curr_row_y + idx * tr_h
                ry2 = ry1 + tr_h
                row_bg = (32, 28, 20) if is_active else ((24, 27, 34) if idx % 2 == 0 else (20, 22, 28))
                row_border = (255, 140, 0) if is_active else (40, 44, 54)

                cv2.rectangle(canvas, (tbl_x1, ry1), (tbl_x1 + tbl_w, ry2), row_bg, -1)
                cv2.rectangle(canvas, (tbl_x1, ry1), (tbl_x1 + tbl_w, ry2), row_border, 1)

                col_txt = (255, 180, 0) if is_active else (210, 215, 225)
                delta_col = (0, 240, 120) if not is_active else (255, 200, 80)

                # 限制文字长度避免溢出
                s_item = item_s if len(item_s) <= 38 else item_s[:35] + "..."

                put_text(canvas, rnd_s, (tx0, ry1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col_txt, 1, cv2.LINE_AA)
                put_text(canvas, s_item, (tx1 + 6, ry1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col_txt, 1, cv2.LINE_AA)
                put_text(canvas, b_s, (tx2 + 6, ry1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 170, 185), 1, cv2.LINE_AA)
                put_text(canvas, a_s, (tx3 + 6, ry1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col_txt, 1, cv2.LINE_AA)
                put_text(canvas, d_s, (tx4 + 6, ry1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, delta_col, 1 if is_active else 2, cv2.LINE_AA)

        # 6. 底栏停机说明
        tip_txt = "收敛准则: 单轮改善 < 0.010 px 触发边际最优收敛 | 共视拓扑守门确保几何不退化 | 随时按 Space 急停"
        put_text(canvas, tip_txt, (cx1 + 20, cy1 + card_h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (130, 140, 155), 1, cv2.LINE_AA)

    def render_prune_settlement_card(self, app: Any, canvas: np.ndarray, w: int, h: int):
        """居中展示工序 5-Auto: 智能剪枝平差结算单对比卡片 (支持一键采纳或无损撤销)"""
        s_data = getattr(app.ba_runner, "prune_settlement_data", None)
        if not s_data:
            return

        card_w, card_h = 700, 340
        cx1, cy1 = (w - card_w) // 2, (h - card_h) // 2
        overlay = canvas.copy()
        cv2.rectangle(overlay, (cx1, cy1), (cx1 + card_w, cy1 + card_h), (18, 22, 28), -1)
        cv2.addWeighted(overlay, 0.95, canvas, 0.05, 0, canvas)
        cv2.rectangle(canvas, (cx1, cy1), (cx1 + card_w, cy1 + card_h), (0, 230, 100), 2)

        # 1. 顶部标题与收敛徽章
        put_text(canvas, "智能残差剪枝平差结算单 (Auto-Prune Settlement)", (cx1 + 22, cy1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

        reason = s_data.get("stop_reason", "最优收敛")
        (rw, _), _ = measure_text(f"[{reason}]", cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        put_text(canvas, f"[{reason}]", (cx1 + card_w - 22 - rw, cy1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 180), 1, cv2.LINE_AA)

        cv2.line(canvas, (cx1 + 22, cy1 + 44), (cx1 + card_w - 22, cy1 + 44), (50, 58, 72), 1)

        # 2. 关键前后指标对比
        init_rmse = s_data.get("initial_rmse", 0.0)
        final_rmse = s_data.get("final_rmse", 0.0)
        init_mm = s_data.get("initial_mm", 0.0)
        final_mm = s_data.get("final_mm", 0.0)
        rounds = s_data.get("rounds_executed", 0)
        pruned_cnt = s_data.get("total_pruned_count", 0)

        drop_pct = ((init_rmse - final_rmse) / max(0.001, init_rmse)) * 100.0 if init_rmse > 0 else 0.0

        y_c = cy1 + 72
        # RMSE 对比
        put_text(canvas, "全局像面 RMSE:", (cx1 + 24, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 205, 215), 1, cv2.LINE_AA)
        rmse_str = f"{init_rmse:.3f} px  ->  {final_rmse:.3f} px"
        put_text(canvas, rmse_str, (cx1 + 175, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 240, 100), 2, cv2.LINE_AA)
        put_text(canvas, f"(误差显著降低 {drop_pct:.1f}%)", (cx1 + 445, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1, cv2.LINE_AA)

        y_c += 28
        # 物理毫米对比
        put_text(canvas, "空间物理偏差 (中位):", (cx1 + 24, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 205, 215), 1, cv2.LINE_AA)
        mm_str = f"{init_mm:.2f} mm  ->  {final_mm:.2f} mm"
        put_text(canvas, mm_str, (cx1 + 175, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 240, 100), 2, cv2.LINE_AA)

        y_c += 28
        # 轮次与剔除汇总
        put_text(canvas, f"迭代执行: {rounds} 轮", (cx1 + 24, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 185, 195), 1, cv2.LINE_AA)
        put_text(canvas, f"累计淘汰外点: {pruned_cnt} 个 (已受共视拓扑严格保护)", (cx1 + 175, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 180, 0), 1, cv2.LINE_AA)

        y_c += 16
        cv2.line(canvas, (cx1 + 22, y_c), (cx1 + card_w - 22, y_c), (45, 52, 65), 1)
        y_c += 20

        # 3. 逐轮剔除明细 (最多展示最近 3 轮)
        put_text(canvas, "各轮剪枝与收敛明细:", (cx1 + 24, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 165, 175), 1, cv2.LINE_AA)
        y_c += 18
        hist = s_data.get("history", [])
        show_hist = hist[-3:] if len(hist) > 3 else hist
        for h_item in show_hist:
            r_num = h_item.get("round", 1)
            p_list = h_item.get("pruned", [])
            p_str = ", ".join([f"{bname}的#{tid}({err:.2f}px)" for bname, tid, err, _ in p_list])
            d_rmse = h_item.get("delta_rmse", 0.0)
            a_rmse = h_item.get("rmse_after", 0.0)
            log_line = f"轮次 #{r_num}: 淘汰 [{p_str}] -> RMSE降至 {a_rmse:.3f}px (改善: {d_rmse:.3f}px)"
            put_text(canvas, log_line, (cx1 + 32, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 210, 220), 1, cv2.LINE_AA)
            y_c += 20

        # 4. 底部决策操作与提示
        card_hint = "左侧列表已自动展开逐帧多轮残差演进矩阵大表 (按 X 键可自由收放)"
        put_text(canvas, card_hint, (cx1 + 24, cy1 + card_h - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 220, 255), 1, cv2.LINE_AA)

        btn_y1 = cy1 + card_h - 44
        btn_y2 = btn_y1 + 32

        # 采纳按钮 (Enter)
        b1_w = 200
        b1_x1 = cx1 + 40
        b1_x2 = b1_x1 + b1_w
        draw_styled_button(canvas, (b1_x1, btn_y1, b1_x2, btn_y2), "采纳成果 (Enter)",
                           mouse_pos=app.mouse_pos, btn_type="success")
        app.gui_buttons.append(("ACCEPT_PRUNE", (b1_x1, btn_y1, b1_x2, btn_y2), "ACCEPT_PRUNE"))

        # 导出报告按钮 (R)
        b2_w = 190
        b2_x1 = b1_x2 + 25
        b2_x2 = b2_x1 + b2_w
        draw_styled_button(canvas, (b2_x1, btn_y1, b2_x2, btn_y2), "导出质检单 (R)",
                           mouse_pos=app.mouse_pos, btn_type="primary")
        app.gui_buttons.append(("EXPORT_REPORT", (b2_x1, btn_y1, b2_x2, btn_y2), "EXPORT_REPORT"))

        # 撤销还原按钮 (Esc)
        b3_w = 170
        b3_x1 = b2_x2 + 25
        b3_x2 = b3_x1 + b3_w
        draw_styled_button(canvas, (b3_x1, btn_y1, b3_x2, btn_y2), "撤销还原 (Esc)",
                           mouse_pos=app.mouse_pos, btn_type="danger")
        app.gui_buttons.append(("UNDO_PRUNE", (b3_x1, btn_y1, b3_x2, btn_y2), "UNDO_PRUNE"))
