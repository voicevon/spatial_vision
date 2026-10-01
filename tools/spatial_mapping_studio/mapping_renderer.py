"""
空间建图工作站 - UI 界面渲染器 (MappingRenderer, 核心调度类)
================================================================================
负责工作站现代深色全景三栏界面的整体排版调度, 并直接承担以下分区绘制：
1. 顶栏 (Top Navigation Bar: LOGO + 全局快捷动作按钮, Dashboard 同源风格)
2. 底栏 (Bottom Status Bar: 放行门限 / 拓扑连通度 / 采图与精度状态指标)
3. 浮层 (科技感居中异步 BA 双轨进度卡片、Toast 提示、置顶下拉菜单、Hover 气泡)

拆分结构 (上帝文件拆分重构, 对外行为与拆分前完全一致)：
- tools/spatial_mapping_studio/mapping_ui_common.py   共享常量 (下拉选项) 与模块级按钮绘制函数
- tools/spatial_mapping_studio/mapping_frame_list.py  MappingFrameListMixin   左栏帧列表/矩阵宽表
- tools/spatial_mapping_studio/mapping_center_view.py MappingCenterViewMixin  中栏视口与 3D 棱柱叠加
- tools/spatial_mapping_studio/mapping_inspector.py   MappingInspectorMixin   右栏检视与剪枝卡片
本类通过多继承组合三个 Mixin, 跨分区方法调用由 MRO 解析；
外部模块仍通过 self.ui_renderer.<method> 唯一通道访问, 并可从本模块
re-import 共享常量与绘制函数 (保持既有导入路径兼容)。
"""

import time
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

from src.ui.text_rendering import measure_text, put_text

# 共享常量与模块级绘制函数已迁移至 mapping_ui_common, 此处 re-import 保持
# 既有外部导入路径 (from tools.spatial_mapping_studio.mapping_renderer import ...) 兼容可用。
from src.ui.gui_theme import GuiTheme
from src.ui.gui_components import (
    draw_dashboard_button,
    draw_dropdown_button,
    render_dropdown_popup as common_render_dropdown_popup,
    render_floating_tooltip,
)
from tools.spatial_mapping_studio.mapping_ui_common import (
    VIEW_MODE_OPTIONS,
    FILTER_MODE_OPTIONS,
    SORT_MODE_OPTIONS,
    BA_VIEW_OPTIONS,
    OBS_VIEW_OPTIONS,
)
from tools.spatial_mapping_studio.mapping_frame_list import MappingFrameListMixin
from tools.spatial_mapping_studio.mapping_center_view import MappingCenterViewMixin
from tools.spatial_mapping_studio.mapping_inspector import MappingInspectorMixin


# ============================================================
# Hover Tooltip 定义 (按钮 id -> 帮助文字)
# ============================================================
HOVER_TOOLTIPS: Dict[str, List[str]] = {
    "RUN_BA": [
        "【阶段一：纯视觉自由平差 (B)】",
        "",
        "纯基于像面重投影残差优化相对刚体几何构型：",
        "  • 100% 独立，不依赖任何世界锚点真值",
        "  • 自动消除镜头畸变与粗差，解算各 Tag 相对间距",
        "  • 产出并固化相对底图 tags_map_raw.yaml",
        "快捷键: [B]",
    ],
    "ALIGN_WORLD_DATUM": [
        "【阶段二：校准世界坐标系 (C)】",
        "",
        "将阶段一已有的相对底图校准对齐至机械臂/现场世界系：",
        "  • 读取白名单/anchor_tags 中的已知世界坐标",
        "  • 运行 3D 刚体形变校验，防止录入错误污染全图",
        "  • 采用 Umeyama 3D 最优相似变换，毫秒级生效",
        "  • 修改锚点后随时单独点击，无需重跑平差！",
        "快捷键: [C]",
    ],
    "RUN_AUTO_PRUNE_BA": [
        "【智能残差剪枝平差 (A)】",
        "",
        "运行 BA 全局平差 + 自动剔除残差最大的离群观测",
        "保留 RMSE 最优的 Tag 组合，迭代收敛。",
        "",
        "剪枝依据: 单帧残差 / 单 Tag 离群 / 共视拓扑连通性",
        "快捷键: [A]",
    ],
}


class MappingRenderer(MappingFrameListMixin, MappingCenterViewMixin, MappingInspectorMixin):
    """空间建图工作站 UI 渲染器"""

    def __init__(self):
        pass

    def render(self, app: Any, canvas: np.ndarray):
        """完整渲染空间建图工作站 顶栏、左栏、中栏、右栏与底栏"""
        w, h = app.win_w, app.win_h
        top_h = app.viewport.top_bar_h
        bot_h = app.viewport.bottom_bar_h
        app.gui_buttons.clear()

        # 异步 BA 结果轮询
        ba_res = app.ba_runner.poll_result()
        if ba_res is not None:
            succ, msg = ba_res
            app.set_toast(msg)
            if succ:
                app.data_mgr.refresh_all_frame_metrics()

        # 异步全量超精提取结果轮询
        if hasattr(app, "poll_super_extract_result"):
            ext_res = app.poll_super_extract_result()
            if ext_res is not None:
                succ, msg = ext_res
                app.set_toast(msg)
                if succ:
                    app.data_mgr.refresh_all_frame_metrics()

        # 1. 顶栏
        self.render_top_bar(app, canvas, w, top_h)

        # 2. 工作区尺寸与左栏自适应宽度
        app.left_bar_w = getattr(app, "dynamic_left_bar_w", app.left_bar_w)
        content_y1 = top_h
        content_y2 = h - bot_h
        content_h = content_y2 - content_y1

        # 左栏：紧凑帧序列列表或逐帧多轮残差演进矩阵宽表
        self.render_left_frame_list(app, canvas, 0, content_y1, app.left_bar_w, content_h)

        # 右栏：180px 瘦身属性与单帧诊断面板
        self.render_right_inspector(app, canvas, w - app.right_bar_w, content_y1, app.right_bar_w, content_h)

        # 中栏：视口
        mid_x1 = app.left_bar_w
        mid_w = w - app.left_bar_w - app.right_bar_w
        self.render_center_viewport(app, canvas, mid_x1, content_y1, mid_w, content_h)

        # 4. 居中展示浮层卡片 (优先级: 结算对比卡片 > 智能剪枝进度卡片 > BA进度卡片 > 超精提取进度卡片)
        if getattr(app.ba_runner, "prune_settlement_data", None) is not None:
            self.render_prune_settlement_card(app, canvas, w, h)
        elif getattr(app.ba_runner, "is_auto_pruning", False):
            self.render_prune_ba_card(app, canvas, w, h)
        elif app.ba_runner.is_ba_running:
            self.render_ba_loading_card(app, canvas, w, h)
        elif getattr(app, "is_extracting_all", False):
            self.render_extract_loading_card(app, canvas, w, h)

        # 5. Toast 浮层 (持久显示, 用户点击 ❌ 才关闭)
        if getattr(app, 'toast_sticky', False) and app.status_toast:
            self.render_toast(app, canvas, w, h, bot_h)

        # 5b. 世界坐标系对齐质检单浮层 (像报告一样的 Toast 卡片)
        if getattr(app, 'alignment_report', None):
            self.render_alignment_report_toast(app, canvas, w, h, bot_h)

        # 6. 置顶悬浮下拉列表 (ROI_LIST_PANEL / COORD_FRAME_PANEL 走 checkbox 专用渲染，其余走通用单选渲染)
        if app.active_dropdown and app.active_dropdown in app.dropdown_boxes:
            if app.active_dropdown == "ROI_LIST_PANEL":
                self.render_roi_list_panel(app, canvas)
            elif app.active_dropdown == "COORD_FRAME_PANEL":
                self.render_coord_frame_panel(app, canvas)
            else:
                dd_info = app.dropdown_boxes[app.active_dropdown]
                self.render_dropdown_popup(app, canvas, app.active_dropdown,
                                          dd_info["rect"], dd_info["options"], dd_info["active_key"])

        # 7. Hover 帮助气泡 (最后绘制, 覆盖在所有面板之上, 不自动关闭)
        mx, my = app.mouse_pos
        self._render_hover_tooltip(canvas, app, mx, my, w, h)

    def render_top_bar(self, app: Any, canvas: np.ndarray, w: int, top_h: int):
        """顶栏：LOGO + 紧随其后的全局快捷动作按钮 (Dashboard 同源风格)"""
        cv2.rectangle(canvas, (0, 0), (w, top_h), (24, 26, 32), -1)
        cv2.line(canvas, (0, top_h), (w, top_h), (55, 60, 72), 1)

        # 1. LOGO
        put_text(canvas, "SPATIAL MAPPING STUDIO", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2, cv2.LINE_AA)
        (logo_w, _), _ = measure_text("SPATIAL MAPPING STUDIO", cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)

        # 2. LOGO 右侧紧邻的全局快捷动作按钮
        mx, my = app.mouse_pos
        btn_y_top, btn_y_bot = 7, top_h - 7
        bx = 16 + logo_w + 20

        # 0. 选择工位下拉框 (首要核心位置)
        sc_w = 145
        cur_sc_label = getattr(app, "current_workspace_name", "默认工位")
        is_sc_open = (app.active_dropdown == "WORKSPACE_DROPDOWN")
        draw_dropdown_button(canvas, (bx, btn_y_top, bx + sc_w, btn_y_bot), f"工位: {cur_sc_label}",
                             is_open=is_sc_open, mouse_pos=(mx, my),
                             theme_color=(0, 255, 180))
        app.dropdown_boxes["WORKSPACE_DROPDOWN"] = {
            "rect": (bx, btn_y_top, bx + sc_w, btn_y_bot),
            "options": getattr(app, "workspace_options", []),
            "active_key": getattr(app, "current_workspace_id", "")
        }
        app.gui_buttons.append(("TOGGLE_WORKSPACE_DROPDOWN", (bx, btn_y_top, bx + sc_w, btn_y_bot), "WORKSPACE_DROPDOWN"))
        bx += sc_w + 5

        # 1. 全局全量超精提取 (清空旧角点并从头重提取)
        ext_w = 85
        is_ext = getattr(app, "is_extracting_all", False)
        draw_dashboard_button(canvas, (bx, btn_y_top, bx + ext_w, btn_y_bot),
                              "提取中..." if is_ext else "超精提取",
                              mouse_pos=(mx, my), is_running=is_ext)
        app.gui_buttons.append(("SUPER_EXTRACT_ALL", (bx, btn_y_top, bx + ext_w, btn_y_bot), "SUPER_EXTRACT_ALL"))
        bx += ext_w + 5

        # 2. [B] 阶段一：纯视觉自由平差
        ba_w = 98
        draw_dashboard_button(canvas, (bx, btn_y_top, bx + ba_w, btn_y_bot),
                              "平差中..." if app.ba_runner.is_ba_running else "自由平差 (B)",
                              mouse_pos=(mx, my), is_running=app.ba_runner.is_ba_running)
        app.gui_buttons.append(("RUN_BA", (bx, btn_y_top, bx + ba_w, btn_y_bot), "RUN_BA"))
        bx += ba_w + 5

        # 3. [C] 阶段二：独立校准世界系 (Umeyama 3D 刚体对齐, 毫秒级)
        align_w = 112
        draw_dashboard_button(canvas, (bx, btn_y_top, bx + align_w, btn_y_bot), "校准世界系 (C)",
                              mouse_pos=(mx, my), accent=(210, 110, 255))
        app.gui_buttons.append(("ALIGN_WORLD_DATUM", (bx, btn_y_top, bx + align_w, btn_y_bot), "ALIGN_WORLD_DATUM"))
        bx += align_w + 5

        # 4. [A] 智能残差剪枝平差
        prune_w = 98
        is_prune = getattr(app.ba_runner, "is_auto_pruning", False)
        draw_dashboard_button(canvas, (bx, btn_y_top, bx + prune_w, btn_y_bot),
                              "剪枝中..." if is_prune else "剪枝平差 (A)",
                              mouse_pos=(mx, my), is_running=is_prune)
        app.gui_buttons.append(("RUN_AUTO_PRUNE_BA", (bx, btn_y_top, bx + prune_w, btn_y_bot), "RUN_AUTO_PRUNE_BA"))


        # 9. 右侧 [Q] 退出工作台 (最右侧退出不动)
        exit_w = 85
        exit_x1 = w - exit_w - 14
        draw_dashboard_button(canvas, (exit_x1, btn_y_top, exit_x1 + exit_w, btn_y_bot), "退出 (Q)",
                              mouse_pos=(mx, my), accent=(70, 60, 210))
        app.gui_buttons.append(("EXIT", (exit_x1, btn_y_top, exit_x1 + exit_w, btn_y_bot), "EXIT"))

    def _render_hover_tooltip(self, canvas: np.ndarray, app: Any, mx: int, my: int,
                               cw: int, ch: int):
        """检测鼠标是否悬停在有帮助文字的按钮上, 若有则渲染统一悬浮气泡面板"""
        if mx < 0 or my < 0:
            return
        for btn_id, (bx1, by1, bx2, by2), _ in getattr(app, "gui_buttons", []):
            if btn_id not in HOVER_TOOLTIPS:
                continue
            if bx1 <= mx <= bx2 and by1 <= my <= by2:
                raw_lines = HOVER_TOOLTIPS[btn_id]
                # 解析标题与正文
                if raw_lines and raw_lines[0].startswith("【") and raw_lines[0].endswith("】"):
                    title = raw_lines[0].strip("【】")
                    lines = raw_lines[1:]
                else:
                    title = "帮助说明"
                    lines = raw_lines
                render_floating_tooltip(canvas, title, lines, (bx1, by1))
                return


    def render_ba_loading_card(self, app: Any, canvas: np.ndarray, w: int, h: int):
        """居中展示异步 BA 全局平差双轨进度卡片 (大阶段主进度条 + 求解器子进度条与实时收敛指标)"""
        card_w, card_h = 640, 162
        cx1, cy1 = (w - card_w) // 2, (h - card_h) // 2
        overlay = canvas.copy()
        cv2.rectangle(overlay, (cx1, cy1), (cx1 + card_w, cy1 + card_h), (18, 20, 26), -1)
        cv2.addWeighted(overlay, 0.94, canvas, 0.06, 0, canvas)
        cv2.rectangle(canvas, (cx1, cy1), (cx1 + card_w, cy1 + card_h), (0, 220, 255), 2)

        # 1. 主阶段总流程进度条
        pct = max(0.0, min(1.0, app.ba_runner.ba_progress))
        pct_int = int(round(pct * 100))

        put_text(canvas, "[BA] 全局平差整体收敛进度", (cx1 + 22, cy1 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        pct_str = f"{pct_int}%"
        (pw, _), _ = measure_text(pct_str, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        put_text(canvas, pct_str, (cx1 + card_w - 22 - pw, cy1 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 235, 255), 2, cv2.LINE_AA)

        bar_x1 = cx1 + 22
        bar_y1 = cy1 + 34
        bar_x2 = cx1 + card_w - 22
        bar_y2 = bar_y1 + 12
        bar_w = bar_x2 - bar_x1

        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (30, 34, 44), -1)
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (55, 62, 78), 1)

        fill_w = int(bar_w * pct)
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), (0, 210, 255), -1)
            cv2.line(canvas, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y1), (180, 245, 255), 1)

        stage_txt = app.ba_runner.ba_stage_text or "准备进入优化平差管线..."
        put_text(canvas, stage_txt, (cx1 + 22, cy1 + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 215, 240), 1, cv2.LINE_AA)

        cv2.line(canvas, (cx1 + 22, cy1 + 74), (cx1 + card_w - 22, cy1 + 74), (45, 48, 60), 1)

        # 2. 求解器内部子阶段与迭代进度条
        sub_pct = max(0.0, min(1.0, app.ba_runner.ba_sub_progress))
        sub_pct_int = int(round(sub_pct * 100))

        put_text(canvas, "优化器实时迭代收敛监控 (Sub-Iteration)", (cx1 + 22, cy1 + 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 205, 215), 1, cv2.LINE_AA)
        sub_pct_str = f"{sub_pct_int}%"
        (spw, _), _ = measure_text(sub_pct_str, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 2)
        put_text(canvas, sub_pct_str, (cx1 + card_w - 22 - spw, cy1 + 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 180, 255), 2, cv2.LINE_AA)

        sbar_y1 = cy1 + 102
        sbar_y2 = sbar_y1 + 10
        cv2.rectangle(canvas, (bar_x1, sbar_y1), (bar_x2, sbar_y2), (25, 28, 36), -1)
        cv2.rectangle(canvas, (bar_x1, sbar_y1), (bar_x2, sbar_y2), (48, 54, 68), 1)

        sub_fill_w = int(bar_w * sub_pct)
        if sub_fill_w > 0:
            cv2.rectangle(canvas, (bar_x1, sbar_y1), (bar_x1 + sub_fill_w, sbar_y2), (0, 160, 255), -1)
            cv2.line(canvas, (bar_x1, sbar_y1), (bar_x1 + sub_fill_w, sbar_y1), (120, 220, 255), 1)

        sub_txt = app.ba_runner.ba_sub_text or "等待当前阶段迭代步进推进..."
        put_text(canvas, sub_txt, (cx1 + 22, cy1 + 132),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 230, 255), 1, cv2.LINE_AA)

    def render_extract_loading_card(self, app: Any, canvas: np.ndarray, w: int, h: int):
        """工序 3 全量超精提取进行中：居中磨砂半透明高科技进度卡片"""
        card_w, card_h = 520, 110
        cx1 = (w - card_w) // 2
        cy1 = (h - card_h) // 2
        cx2 = cx1 + card_w
        cy2 = cy1 + card_h

        overlay = canvas.copy()
        cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), (20, 24, 32), -1)
        cv2.addWeighted(overlay, 0.94, canvas, 0.06, 0, canvas)
        cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (0, 200, 255), 2)

        pct = max(0.0, min(1.0, getattr(app, "extract_progress", 0.0)))
        pct_int = int(round(pct * 100))

        put_text(canvas, "工序 3: 全局全量图像超精重提取", (cx1 + 22, cy1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        pct_str = f"{pct_int}%"
        (pw, _), _ = measure_text(pct_str, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        put_text(canvas, pct_str, (cx1 + card_w - 22 - pw, cy1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 235, 255), 2, cv2.LINE_AA)

        bar_x1 = cx1 + 22
        bar_y1 = cy1 + 38
        bar_x2 = cx1 + card_w - 22
        bar_y2 = bar_y1 + 14
        bar_w = bar_x2 - bar_x1

        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (30, 34, 44), -1)
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (55, 62, 78), 1)

        fill_w = int(bar_w * pct)
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), (0, 210, 255), -1)
            cv2.line(canvas, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y1), (180, 245, 255), 1)

        stage_txt = getattr(app, "extract_stage_text", "") or "正在全量调用多尺度增强与正交亚像素精修..."
        put_text(canvas, stage_txt, (cx1 + 22, cy1 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 215, 240), 1, cv2.LINE_AA)

    def render_toast(self, app: Any, canvas: np.ndarray, w: int, h: int, bot_h: int):
        close_label = "X"
        copy_label = "COPY"
        (cw, _), _ = measure_text(close_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
        close_btn_w = cw + 20  # ❌ 按钮宽度 (左右内边距 10px)
        (cpw, _), _ = measure_text(copy_label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        copy_btn_w = cpw + 20  # 复制按钮宽度

        (tw, _), _ = measure_text(app.status_toast, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
        total_w = tw + 32 + copy_btn_w + close_btn_w  # 文本 padding + 复制按钮 + 关闭按钮
        tx1 = (w - total_w) // 2
        ty1 = h - bot_h - 46
        tx2 = tx1 + total_w
        ty2 = ty1 + 32

        # 背景板
        cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2), (120, 30, 100), -1)
        cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2), (220, 60, 180), 1)
        # 消息文本
        put_text(canvas, app.status_toast, (tx1 + 16, ty1 + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

        # 📋 复制按钮 (右侧, 紧邻关闭按钮左侧)
        copy_x1 = tx2 - close_btn_w - copy_btn_w
        copy_y1 = ty1
        copy_x2 = copy_x1 + copy_btn_w
        copy_y2 = ty2
        # 分隔线
        cv2.line(canvas, (copy_x1, copy_y1 + 4), (copy_x1, copy_y2 - 4), (180, 80, 150), 1)
        # 复制文本居中
        put_text(canvas, copy_label, (copy_x1 + 10, ty1 + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 220, 255), 1, cv2.LINE_AA)
        # 注册点击区域
        app.gui_buttons.append(("COPY_TOAST", (copy_x1, copy_y1, copy_x2, copy_y2), None))

        # ❌ 关闭按钮 (最右侧)
        close_x1 = tx2 - close_btn_w
        close_y1 = ty1
        close_x2 = tx2
        close_y2 = ty2
        # 分隔线
        cv2.line(canvas, (close_x1, close_y1 + 4), (close_x1, close_y2 - 4), (180, 80, 150), 1)
        # ❌ 文本居中
        put_text(canvas, close_label, (close_x1 + 10, ty1 + 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 2, cv2.LINE_AA)
        # 注册点击区域
        app.gui_buttons.append(("DISMISS_TOAST", (close_x1, close_y1, close_x2, close_y2), None))

    def render_alignment_report_toast(self, app: Any, canvas: np.ndarray, w: int, h: int, bot_h: int):
        """像报告一样的世界坐标系对齐质检单浮层 (列举所有锚点物理残差，支持按 ID/残差排序与异常黄色高亮)"""
        rep = getattr(app, "alignment_report", None)
        if not rep or not isinstance(rep, dict):
            return

        rows = list(rep.get("rows", []))
        if not rows:
            return

        sort_mode = getattr(app, "alignment_report_sort", "id")
        if sort_mode == "err_desc":
            rows = sorted(rows, key=lambda r: r.get("dist_3d_mm", 0.0), reverse=True)
        else:
            rows = sorted(rows, key=lambda r: r.get("tag_id", 0))

        has_warn = rep.get("has_warn", False)
        card_w = 780
        row_h = 24
        header_h = 36
        sum_h = 24
        thead_h = 24
        card_h = header_h + sum_h + thead_h + len(rows) * row_h + 12

        cx1 = (w - card_w) // 2
        cx2 = cx1 + card_w
        cy2 = h - bot_h - 16
        cy1 = cy2 - card_h
        if cy1 < 50:
            cy1 = max(40, (h - card_h) // 2)
            cy2 = cy1 + card_h

        # 半透明深色质感卡片背景
        overlay = canvas.copy()
        cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), (20, 24, 32), -1)
        cv2.addWeighted(overlay, 0.94, canvas, 0.06, 0, canvas)

        # 外边框 (异常时醒目黄色，正常时典雅青绿色)
        border_color = (0, 190, 255) if has_warn else (0, 210, 150)
        cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), border_color, 2)

        # 1. 顶部操作栏
        conflict_pairs = rep.get("conflict_pairs", [])
        if conflict_pairs:
            title_txt = f"⚠️ 世界坐标系对齐质检单 (检出 {len(conflict_pairs)} 组锚点距离严重冲突!)"
            title_col = (0, 165, 255)
        elif has_warn:
            title_txt = "⚠️ 世界坐标系对齐质检单 (注意：检测到标靶物理残差偏大)"
            title_col = (0, 215, 255)
        else:
            title_txt = "✅ 世界坐标系对齐质检单 (各已知锚点高精吻合)"
            title_col = (0, 255, 170)
        put_text(canvas, title_txt, (cx1 + 16, cy1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, title_col, 2, cv2.LINE_AA)

        # 右侧操作按钮: [关闭 X]
        close_w = 28
        close_x1 = cx2 - close_w - 10
        close_x2 = cx2 - 10
        btn_y1 = cy1 + 7
        btn_y2 = cy1 + 29
        cv2.rectangle(canvas, (close_x1, btn_y1), (close_x2, btn_y2), (45, 50, 62), -1)
        cv2.rectangle(canvas, (close_x1, btn_y1), (close_x2, btn_y2), (70, 78, 95), 1)
        put_text(canvas, "X", (close_x1 + 8, btn_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 210, 225), 2, cv2.LINE_AA)
        app.gui_buttons.append(("CLOSE_ALIGN_REPORT", (close_x1, btn_y1, close_x2, btn_y2), None))

        # [📋 复制报告] 按钮
        copy_w = 78
        copy_x1 = close_x1 - copy_w - 8
        copy_x2 = copy_x1 + copy_w
        cv2.rectangle(canvas, (copy_x1, btn_y1), (copy_x2, btn_y2), (35, 45, 60), -1)
        cv2.rectangle(canvas, (copy_x1, btn_y1), (copy_x2, btn_y2), (60, 80, 110), 1)
        put_text(canvas, "复制报告", (copy_x1 + 10, btn_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 255), 1, cv2.LINE_AA)
        app.gui_buttons.append(("COPY_ALIGN_REPORT", (copy_x1, btn_y1, copy_x2, btn_y2), None))

        # [排序: ...] 切换按钮
        sort_label = "排序: 残差降序 ▾" if sort_mode == "err_desc" else "排序: Tag ID ▾"
        sort_w = 110
        sort_x1 = copy_x1 - sort_w - 8
        sort_x2 = sort_x1 + sort_w
        cv2.rectangle(canvas, (sort_x1, btn_y1), (sort_x2, btn_y2), (32, 40, 52), -1)
        cv2.rectangle(canvas, (sort_x1, btn_y1), (sort_x2, btn_y2), (55, 70, 92), 1)
        put_text(canvas, sort_label, (sort_x1 + 8, btn_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 230, 255), 1, cv2.LINE_AA)
        app.gui_buttons.append(("TOGGLE_ALIGN_REPORT_SORT", (sort_x1, btn_y1, sort_x2, btn_y2), None))

        # 2. 摘要信息行
        solver = rep.get("solver_type", "Umeyama 3D")
        mean_mm = rep.get("mean_mm", 0.0)
        max_mm = rep.get("max_mm", 0.0)
        warn_th = rep.get("warn_threshold_mm", 3.0)
        conflict_str = f"  |  ⚠️ 几何超差冲突: {len(conflict_pairs)} 对" if conflict_pairs else ""
        sum_str = f"解算算法: {solver}  |  锚点总数: {len(rows)} 枚  |  均值残差: {mean_mm:.2f} mm  |  最大残差: {max_mm:.2f} mm{conflict_str}  |  告警阈值: > {warn_th:.1f} mm"
        put_text(canvas, sum_str, (cx1 + 16, cy1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (165, 178, 195), 1, cv2.LINE_AA)

        # 3. 表头栏
        th_y1 = cy1 + 58
        th_y2 = th_y1 + thead_h
        cv2.rectangle(canvas, (cx1 + 10, th_y1), (cx2 - 10, th_y2), (28, 34, 46), -1)
        put_text(canvas, "标靶", (cx1 + 16, th_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 155, 175), 1, cv2.LINE_AA)
        put_text(canvas, "设定世界坐标 (X, Y, Z) mm", (cx1 + 85, th_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 155, 175), 1, cv2.LINE_AA)
        put_text(canvas, "实测对齐坐标 (X, Y, Z) mm", (cx1 + 265, th_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 155, 175), 1, cv2.LINE_AA)
        put_text(canvas, "分轴偏差 (dx, dy, dz) mm", (cx1 + 445, th_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 155, 175), 1, cv2.LINE_AA)
        put_text(canvas, "3D 物理残差", (cx1 + 600, th_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 155, 175), 1, cv2.LINE_AA)
        put_text(canvas, "质检状态", (cx1 + 695, th_y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 155, 175), 1, cv2.LINE_AA)

        # 4. 数据行渲染
        curr_y = th_y2 + 2
        for i, row in enumerate(rows):
            is_warn = row.get("is_warn", False)
            ry1 = curr_y
            ry2 = curr_y + row_h
            if is_warn:
                # 异常行: 黄色告警条纹背景 + 左侧黄色强调线
                cv2.rectangle(canvas, (cx1 + 10, ry1), (cx2 - 10, ry2), (28, 48, 68), -1)
                cv2.rectangle(canvas, (cx1 + 10, ry1), (cx1 + 13, ry2), (0, 200, 255), -1)
                t_color = (0, 225, 255)
            else:
                bg_col = (22, 26, 35) if i % 2 == 0 else (17, 21, 29)
                cv2.rectangle(canvas, (cx1 + 10, ry1), (cx2 - 10, ry2), bg_col, -1)
                t_color = (220, 230, 240)

            # 各列文字
            tid_str = f"⚠️ Tag #{row['tag_id']}" if is_warn else f"Tag #{row['tag_id']}"
            t_vals = [f"{v:.1f}" if v is not None else "--" for v in row.get("target_xyz", [])]
            tgt_str = f"({', '.join(t_vals)})"
            f_vals = [f"{v:.1f}" if v is not None else "--" for v in row.get("fitted_xyz", [])]
            fit_str = f"({', '.join(f_vals)})"
            d_vals = [str(v) if v is not None else "--" for v in row.get("delta_xyz", [])]
            delta_str = f"({', '.join(d_vals)})"
            dist_str = f"{row.get('dist_3d_mm', 0.0):.2f} mm"
            stat_str = "⚠️ 偏差过大" if is_warn else "🟢 吻合"
            stat_col = (0, 215, 255) if is_warn else (0, 255, 170)

            put_text(canvas, tid_str, (cx1 + 16, ry1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, t_color, 1, cv2.LINE_AA)
            put_text(canvas, tgt_str, (cx1 + 85, ry1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, t_color, 1, cv2.LINE_AA)
            put_text(canvas, fit_str, (cx1 + 265, ry1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, t_color, 1, cv2.LINE_AA)
            put_text(canvas, delta_str, (cx1 + 445, ry1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, t_color, 1, cv2.LINE_AA)
            put_text(canvas, dist_str, (cx1 + 600, ry1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, t_color, 1, cv2.LINE_AA)
            put_text(canvas, stat_str, (cx1 + 695, ry1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, stat_col, 1, cv2.LINE_AA)

            curr_y += row_h


    def render_dropdown_popup(
        self,
        app: Any,
        canvas: np.ndarray,
        pop_name: str,
        rect: Tuple[int, int, int, int],
        options: List[Tuple[str, str]],
        active_key: str
    ):
        """置顶悬浮下拉列表浮层 (统一委托给公共 gui_components)"""
        reg_btns = common_render_dropdown_popup(
            canvas,
            anchor_rect=rect,
            options=options,
            active_key=active_key,
            btn_prefix=f"DD_SELECT_{pop_name}_",
            item_h=32,
            min_width=190,
        )
        for _, item_rect, opt_key in reg_btns:
            btn_id = f"DD_SELECT_{pop_name}_{opt_key}"
            app.gui_buttons.append((btn_id, item_rect, (pop_name, opt_key)))

    def render_roi_list_panel(
        self,
        app: Any,
        canvas: np.ndarray,
    ):
        """ROI 物件 CheckList 浮层 — 从 ROI_LIST_PANEL anchor rect 向上弹出。

        每行: [☑/☐ checkbox] + ROI 名称
        勾选/取消直接切换 RoiDefinition.enabled，并立即写盘到 spatial_scene.yaml。
        """
        from src.ui.text_rendering import put_text as _put_text
        from src.ui.gui_theme import GuiTheme

        roi_mgr = getattr(app, "roi_mgr", None)
        rois = roi_mgr.list_rois() if roi_mgr else []

        anchor = app.dropdown_boxes.get("ROI_LIST_PANEL", {}).get("rect", (0, 0, 110, 22))
        ax1, ay1, ax2, ay2 = anchor

        item_h = 28
        panel_w = max(180, ax2 - ax1)
        header_h = 24
        empty_h = 32

        if not rois:
            panel_h = header_h + empty_h + 4
        else:
            panel_h = header_h + len(rois) * item_h + 6

        # 向下弹出（紧贴按钮底边）
        pop_x1 = ax1
        pop_y1 = ay2 + 2
        pop_y2 = pop_y1 + panel_h
        pop_x2 = pop_x1 + panel_w

        # 边界裁剪 (防溢出画布)
        ch, cw = canvas.shape[:2]
        if pop_x2 > cw - 4:
            pop_x1 = max(4, cw - panel_w - 4)
            pop_x2 = pop_x1 + panel_w
        if pop_y2 > ch - 4:
            pop_y2 = ch - 4
            pop_y1 = max(4, pop_y2 - panel_h)

        # 背景磨砂遮罩
        overlay = canvas.copy()
        cv2.rectangle(overlay, (pop_x1, pop_y1), (pop_x2, pop_y2), (22, 26, 34), -1)
        cv2.addWeighted(overlay, 0.96, canvas, 0.04, 0, canvas)
        cv2.rectangle(canvas, (pop_x1, pop_y1), (pop_x2, pop_y2), (0, 215, 90), 1)

        # 标题行
        _put_text(canvas, "ROI 物件显示",
                  (pop_x1 + 10, pop_y1 + 16),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 215, 90), 1, cv2.LINE_AA)
        cv2.line(canvas, (pop_x1 + 6, pop_y1 + header_h),
                 (pop_x2 - 6, pop_y1 + header_h), (50, 56, 70), 1)

        if not rois:
            _put_text(canvas, "暂无 ROI 物件",
                      (pop_x1 + 10, pop_y1 + header_h + 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 130, 150), 1, cv2.LINE_AA)
            return

        mx, my = app.mouse_pos
        for i, roi in enumerate(rois):
            iy1 = pop_y1 + header_h + 3 + i * item_h
            iy2 = iy1 + item_h
            if iy2 > pop_y2 - 2:
                break

            item_rect = (pop_x1 + 2, iy1, pop_x2 - 2, iy2)
            is_hover = (item_rect[0] <= mx <= item_rect[2] and item_rect[1] <= my <= item_rect[3])

            # Hover 高亮背景
            if is_hover:
                cv2.rectangle(canvas, (item_rect[0], iy1), (item_rect[2], iy2),
                              (38, 44, 56), -1)

            # Checkbox 方块 (12x12)
            cb_x = pop_x1 + 10
            cb_y = iy1 + (item_h - 12) // 2
            cb_color = (0, 215, 90) if roi.enabled else (70, 80, 100)
            cv2.rectangle(canvas, (cb_x, cb_y), (cb_x + 12, cb_y + 12), cb_color, -1 if roi.enabled else 1)
            if roi.enabled:
                # 勾号
                cv2.line(canvas, (cb_x + 2, cb_y + 6), (cb_x + 5, cb_y + 9), (10, 20, 10), 2)
                cv2.line(canvas, (cb_x + 5, cb_y + 9), (cb_x + 10, cb_y + 3), (10, 20, 10), 2)

            # ROI 名称
            name_lbl = roi.name or roi.roi_id
            txt_col = (220, 230, 245) if roi.enabled else (100, 110, 130)
            _put_text(canvas, name_lbl,
                      (cb_x + 18, iy1 + (item_h - 14) // 2 + 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.38, txt_col, 1, cv2.LINE_AA)

            # 注册可点击按钮区域 (整行)
            app.gui_buttons.append((f"TOGGLE_ROI_ENABLED_{roi.roi_id}", item_rect, roi.roi_id))

    def render_coord_frame_panel(
        self,
        app: Any,
        canvas: np.ndarray,
    ):
        """坐标系 CheckList 浮层 —— 从 COORD_FRAME_PANEL anchor rect 向下弹出。

        每行: [☑/☐ checkbox] + 坐标系名称 + frame_id
        勾选/取消直接切换 coord_frame_visibility[frame_id]。
        """
        from src.ui.text_rendering import put_text as _put_text

        coord_mgr = getattr(app, "coord_mgr", None)
        frames = coord_mgr.list_frames() if coord_mgr else []
        visibility = getattr(app, "coord_frame_visibility", {})

        anchor = app.dropdown_boxes.get("COORD_FRAME_PANEL", {}).get("rect", (0, 0, 110, 22))
        ax1, ay1, ax2, ay2 = anchor

        item_h = 28
        panel_w = max(200, ax2 - ax1)
        header_h = 24
        empty_h = 32

        if not frames:
            panel_h = header_h + empty_h + 4
        else:
            panel_h = header_h + len(frames) * item_h + 6

        # 向下弹出（紧贴按钮底边）
        pop_x1 = ax1
        pop_y1 = ay2 + 2
        pop_y2 = pop_y1 + panel_h
        pop_x2 = pop_x1 + panel_w

        # 边界裁剪
        ch, cw = canvas.shape[:2]
        if pop_x2 > cw - 4:
            pop_x1 = max(4, cw - panel_w - 4)
            pop_x2 = pop_x1 + panel_w
        if pop_y2 > ch - 4:
            pop_y2 = ch - 4
            pop_y1 = max(4, pop_y2 - panel_h)

        # 背景磨砂遮罩
        overlay = canvas.copy()
        cv2.rectangle(overlay, (pop_x1, pop_y1), (pop_x2, pop_y2), (22, 26, 34), -1)
        cv2.addWeighted(overlay, 0.96, canvas, 0.04, 0, canvas)
        cv2.rectangle(canvas, (pop_x1, pop_y1), (pop_x2, pop_y2), (0, 220, 255), 1)

        # 标题行
        _put_text(canvas, "坐标系显示",
                  (pop_x1 + 10, pop_y1 + 16),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (pop_x1 + 6, pop_y1 + header_h),
                 (pop_x2 - 6, pop_y1 + header_h), (50, 56, 70), 1)

        if not frames:
            _put_text(canvas, "暂无坐标系",
                      (pop_x1 + 10, pop_y1 + header_h + 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 130, 150), 1, cv2.LINE_AA)
            return

        mx, my = app.mouse_pos
        for i, frame in enumerate(frames):
            iy1 = pop_y1 + header_h + 3 + i * item_h
            iy2 = iy1 + item_h
            if iy2 > pop_y2 - 2:
                break

            is_visible = visibility.get(frame.frame_id, False)
            item_rect = (pop_x1 + 2, iy1, pop_x2 - 2, iy2)
            is_hover = (item_rect[0] <= mx <= item_rect[2] and item_rect[1] <= my <= item_rect[3])

            if is_hover:
                cv2.rectangle(canvas, (item_rect[0], iy1), (item_rect[2], iy2),
                              (38, 44, 56), -1)

            # Checkbox 方块 (12x12)
            cb_x = pop_x1 + 10
            cb_y = iy1 + (item_h - 12) // 2
            cb_color = (0, 220, 255) if is_visible else (70, 80, 100)
            cv2.rectangle(canvas, (cb_x, cb_y), (cb_x + 12, cb_y + 12),
                          cb_color, -1 if is_visible else 1)
            if is_visible:
                cv2.line(canvas, (cb_x + 2, cb_y + 6), (cb_x + 5, cb_y + 9), (10, 20, 30), 2)
                cv2.line(canvas, (cb_x + 5, cb_y + 9), (cb_x + 10, cb_y + 3), (10, 20, 30), 2)

            # 坐标系名称 + frame_id
            name_lbl = (frame.name or frame.frame_id)
            # 已解算状态指示
            T_w_f = None
            try:
                T_w_f, is_resolved = coord_mgr.get_frame_to_world(frame.frame_id)
            except Exception:
                is_resolved = False
            status_dot = "●" if is_resolved else "○"
            dot_col = (60, 200, 80) if is_resolved else (100, 100, 130)
            txt_col = (220, 230, 245) if is_visible else (100, 110, 130)

            _put_text(canvas, status_dot,
                      (cb_x + 18, iy1 + (item_h - 14) // 2 + 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.34, dot_col, 1, cv2.LINE_AA)
            _put_text(canvas, name_lbl,
                      (cb_x + 30, iy1 + (item_h - 14) // 2 + 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.38, txt_col, 1, cv2.LINE_AA)

            # 注册可点击按钮区域 (整行)
            app.gui_buttons.append((f"TOGGLE_COORD_FRAME_VIS_{frame.frame_id}", item_rect, frame.frame_id))
