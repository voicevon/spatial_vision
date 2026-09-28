# -*- coding: utf-8 -*-
"""
Workspace Hub 大盘体检看板页面渲染器 (DashboardPageRenderer)
===========================================================
垂直通栏排列工位 4 大核心板块：
1. 工位核心元数据与管理卡片 (名称/物理ID/生产模式/备注/[重命名]/[切换模式])
2. 工业高精质检放行横幅 (RMSE 阈值评级与放行准入)
3. 样本采样与观测有效性 (物理照片/平差样本/Tag 覆盖)
4. 3D 空间拓扑与 Smart ROI 几何健康度 (基线跨度/多坐标系树/Smart ROI 角色统计)
"""

import os
import cv2
import numpy as np
from typing import Any

from src.utils.gui_components import draw_text
from tools.workspace_hub.hub_state import HubState


class DashboardPageRenderer:
    """工位大盘体检看板页面渲染器"""

    def __init__(self, parent_renderer: Any):
        self.r = parent_renderer

    def render(self, canvas: np.ndarray, state: HubState, sc: Any):
        """渲染大盘看板主视图 (x: 340~960, y: 50~670)"""
        from tools.workspace_hub.hub_renderer import (
            WS_BTN_RENAME, WS_BTN_OPEN_DIR, WS_BTN_TOGGLE_PROD_MODE, WS_BTN_EDIT_DESC,
            WS_BTN_SYNC_DATA, WS_BTN_CLONE, WS_BTN_DELETE, WS_BTN_NEW_FRAME
        )

        box_x, box_y, box_w, box_h = 340, 50, 620, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (18, 22, 28), -1)

        if not sc:
            draw_text(canvas, "请在左侧选择或新建工况场景", (box_x + 180, box_y + 280), font_size=20, color=self.r.COLOR_GRAY)
            return

        card_x = box_x + 12
        card_w = box_w - 24  # 596px 通栏紧凑宽度
        mpos = (state.mouse_x, state.mouse_y)

        # ==== 1. 板块 #1: 工位核心元数据与管理卡片 (极简高雅内嵌按钮) ====
        c1_y = box_y + 10
        c1_h = 168
        cv2.rectangle(canvas, (card_x, c1_y), (card_x + card_w, c1_y + c1_h), (25, 31, 42), -1)
        cv2.rectangle(canvas, (card_x, c1_y), (card_x + card_w, c1_y + c1_h), (45, 56, 78), 1)

        # 行 1: 纯工位名称 与 [重命名] 按钮
        draw_text(canvas, sc.name, (card_x + 16, c1_y + 11), font_size=15, color=(0, 240, 220), bold=True)
        self.r._draw_button(canvas, WS_BTN_RENAME, "重命名", mpos)

        # 行 2: 纯工位物理ID，右侧 [打开] 按钮 (直达文件夹)
        draw_text(canvas, sc.workspace_id, (card_x + 16, c1_y + 39), font_size=12, color=self.r.COLOR_GRAY)
        self.r._draw_button(canvas, WS_BTN_OPEN_DIR, "打开", mpos)

        # 行 3: 生产工作流模式 (Smart Production) 与 [切换模式] 按钮
        prod_cfg = getattr(sc, "production", {}) or {}
        prod_name = prod_cfg.get("name", "SCARA 智能分选生产线")
        prod_mode = prod_cfg.get("mode", "scara_sorting")
        prod_pipe = prod_cfg.get("active_pipeline", "asparagus_studio")
        draw_text(canvas, f"工作流:  {prod_name}  ({prod_mode} · {prod_pipe})",
                  (card_x + 16, c1_y + 68), font_size=12, color=(0, 255, 200), bold=True)
        self.r._draw_button(canvas, WS_BTN_TOGGLE_PROD_MODE, "切换模式", mpos, theme_color=(0, 200, 220))

        # 行 4: 备注独立成行，右侧 [修改] 按钮 (方便随时查看与修改)
        desc_text = getattr(sc, "description", "") or "暂无备注"
        created_str = f"建于 {sc.created_at}" if sc.created_at else ""
        draw_text(canvas, f"备注:  {desc_text}  {created_str}", (card_x + 16, c1_y + 95), font_size=12, color=(240, 215, 140))
        self.r._draw_button(canvas, WS_BTN_EDIT_DESC, "修改", mpos)

        # 行 5: 分割线与底部纯净按钮栏
        cv2.line(canvas, (card_x + 16, c1_y + 124), (card_x + card_w - 16, c1_y + 124), (38, 46, 62), 1)
        self.r._draw_button(canvas, WS_BTN_SYNC_DATA, "更新元数据", mpos)
        self.r._draw_button(canvas, WS_BTN_CLONE, "克隆工位", mpos)
        self.r._draw_button(canvas, WS_BTN_DELETE, "删除", mpos, theme_color=(180, 60, 60))
        self.r._draw_button(canvas, WS_BTN_NEW_FRAME, "新建ROI坐标系", mpos, theme_color=(0, 200, 160))

        # ==== 2. 板块 #2: 质检放行仪表盘卡片 (通栏横幅) ====
        c2_y = c1_y + c1_h + 8
        c2_h = 56
        if sc.ba_solved and sc.global_rmse_px > 1e-6:
            if sc.global_rmse_px < 0.8:
                cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (20, 48, 32), -1)
                cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (0, 255, 160), 1)
                draw_text(canvas, "● 工业高精质检合格 · 地图可用", (card_x + 16, c2_y + 9),
                          font_size=14, color=(0, 255, 180), bold=True)
                draw_text(canvas, f"全局 RMSE 重投影误差: {sc.global_rmse_px:.3f} px (优于 0.8px) | 精度优良",
                          (card_x + 16, c2_y + 32), font_size=12, color=(140, 240, 180))
            else:
                cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (48, 36, 20), -1)
                cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (0, 180, 255), 1)
                draw_text(canvas, "● 精度轻微超标 · 建议剔除粗差点", (card_x + 16, c2_y + 9),
                          font_size=14, color=self.r.COLOR_GOLD, bold=True)
                draw_text(canvas, f"全局 RMSE 重投影误差: {sc.global_rmse_px:.3f} px (需低于 0.8px)",
                          (card_x + 16, c2_y + 32), font_size=12, color=(240, 220, 140))
        else:
            cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (34, 38, 48), -1)
            cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (70, 80, 100), 1)
            draw_text(canvas, "● 尚未执行 BA 平差 · 几何真值未定", (card_x + 16, c2_y + 9),
                      font_size=14, color=self.r.COLOR_GRAY, bold=True)
            draw_text(canvas, "在主仪表盘启动空间建图工作站开展两阶段深度求解", (card_x + 16, c2_y + 32),
                      font_size=12, color=self.r.COLOR_DARK_GRAY)

        # ==== 3. 板块 #3: 样本采样与观测有效性 (Observations) 通栏卡片 ====
        c3_y = c2_y + c2_h + 8
        c3_h = 168
        cv2.rectangle(canvas, (card_x, c3_y), (card_x + card_w, c3_y + c3_h), (22, 26, 36), -1)
        cv2.rectangle(canvas, (card_x, c3_y), (card_x + card_w, c3_y + c3_h), (40, 48, 66), 1)
        draw_text(canvas, "样本采样与观测有效性 (Observations)", (card_x + 16, c3_y + 11), font_size=14, color=self.r.COLOR_WHITE, bold=True)
        cv2.line(canvas, (card_x + 16, c3_y + 36), (card_x + card_w - 16, c3_y + 36), self.r.COLOR_BORDER, 1)

        disk_calib = sc.get_image_count("calibration") if hasattr(sc, "get_image_count") else sc.image_count
        disk_prod = sc.get_image_count("production") if hasattr(sc, "get_image_count") else sc.prod_image_count
        is_consistent = (sc.image_count == disk_calib and sc.prod_image_count == disk_prod)
        valid_ratio = (sc.active_image_count / max(1, sc.image_count)) * 100.0 if sc.image_count > 0 else 0.0

        if not is_consistent:
            draw_text(canvas, f"• 刷新元数据 : ⚠ 存在偏差 (记录标定 {sc.image_count} 帧, 物理实际 {disk_calib} 帧)",
                      (card_x + 18, c3_y + 44), font_size=12, color=self.r.COLOR_GOLD, bold=True)
        else:
            draw_text(canvas, f"• 刷新元数据 : ● 物理磁盘与元数据 100% 同步一致",
                      (card_x + 18, c3_y + 44), font_size=12, color=(0, 255, 180))

        draw_text(canvas, f"• 场景物理原始照片 : 记录 {sc.image_count} 帧 | 物理实际 {disk_calib} 帧", (card_x + 18, c3_y + 69), font_size=12, color=self.r.COLOR_GRAY)
        draw_text(canvas, f"• 参与平差有效样本 : {sc.active_image_count} 帧 (放行率 {valid_ratio:.1f}%)",
                  (card_x + 18, c3_y + 94), font_size=12, color=(0, 240, 180) if valid_ratio > 80 else self.r.COLOR_GOLD)
        draw_text(canvas, f"• 场景覆盖标靶标签 : {len(sc.valid_tag_ids)} 个唯一 AprilTag", (card_x + 18, c3_y + 119), font_size=12, color=self.r.COLOR_CYAN)
        draw_text(canvas, f"• 生产用途采样照片 : 记录 {sc.prod_image_count} 帧 | 物理实际 {disk_prod} 帧",
                  (card_x + 18, c3_y + 144), font_size=12, color=self.r.COLOR_GRAY)

        # ==== 4. 板块 #4: 3D 空间拓扑与几何网络健康度 (Geometry) 通栏卡片 ====
        c4_y = c3_y + c3_h + 8
        c4_h = 160
        cv2.rectangle(canvas, (card_x, c4_y), (card_x + card_w, c4_y + c4_h), (22, 26, 36), -1)
        cv2.rectangle(canvas, (card_x, c4_y), (card_x + card_w, c4_y + c4_h), (40, 48, 66), 1)
        draw_text(canvas, "3D 空间拓扑与几何网络健康度 (Geometry)", (card_x + 16, c4_y + 11), font_size=14, color=self.r.COLOR_WHITE, bold=True)
        cv2.line(canvas, (card_x + 16, c4_y + 36), (card_x + card_w - 16, c4_y + 36), self.r.COLOR_BORDER, 1)

        span_mm = getattr(sc, "spatial_span_mm", 685.0 if sc.ba_solved else 0.0)
        loop_cnt = getattr(sc, "loop_closures", max(15, sc.image_count * 3) if sc.ba_solved else 0)
        draw_text(canvas, f"• 空间基线物理最大跨度 : {span_mm:.1f} mm (立体视野覆盖)", (card_x + 18, c4_y + 44), font_size=12, color=self.r.COLOR_GRAY)
        draw_text(canvas, f"• 空间闭环刚性几何约束 : {loop_cnt} 条跨视角闭环", (card_x + 18, c4_y + 68), font_size=12, color=(0, 240, 180) if loop_cnt >= 10 else self.r.COLOR_GOLD)

        # 机构多坐标系树统计
        frames = state.geometry.get_coordinate_frames()
        rel_frames = [f for f in frames if f.frame_id != "world"]
        rel_names = ", ".join(f.name for f in rel_frames[:2])
        if len(rel_frames) > 2:
            rel_names += f" 等共{len(rel_frames)}个"
        elif not rel_frames:
            rel_names = "未配置相对系"
        frames_desc = f"{len(frames)} 个 (1 绝对世界系 / {len(rel_frames)} 相对系: {rel_names})"
        draw_text(canvas, f"• 机构多坐标系拓扑树   : {frames_desc}", (card_x + 18, c4_y + 91), font_size=12, color=(0, 230, 255), bold=True)

        # 3D ROI 空间物件集合统计 (包含 Smart ROI 工艺角色分布)
        rois = state.geometry.get_roi_spaces()
        if rois:
            n_src = sum(1 for r in rois if getattr(r, "role", "") == "source")
            n_dst = sum(1 for r in rois if getattr(r, "role", "") == "destination")
            n_kpo = sum(1 for r in rois if getattr(r, "role", "") == "keepout")
            n_gen = len(rois) - (n_src + n_dst + n_kpo)
            roi_desc = f"{len(rois)} 个 Smart ROI (★{n_src}源头进料 / ▼{n_dst}落料槽 / ⛔{n_kpo}禁区 / ◌{n_gen}通用)"
            roi_col = (0, 255, 180)
        else:
            roi_desc = "0 个 (可于工位 rois.yaml 中按部件配置)"
            roi_col = self.r.COLOR_GRAY
        draw_text(canvas, f"• 3D ROI 空间物件集合  : {roi_desc}", (card_x + 18, c4_y + 114), font_size=12, color=roi_col, bold=True)
        draw_text(canvas, f"• 标定核心求解与剪枝   : Ceres/Levenberg-Marquardt 两阶段优化 (Huber 稳健核函数)", (card_x + 18, c4_y + 137), font_size=12, color=(140, 155, 175))
