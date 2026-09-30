#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
空间建图工作站 - UI 渲染数据视图选项常量 (mapping_ui_common)
============================================================
本模块仅保留空间建图工作站特定数据管理与过滤排序常量。
通用 UI 交互控件及 3D 棱柱/位姿视觉语言已全面提升至公共基础层：
- 控件：src.ui.gui_components (draw_dropdown_button, render_dropdown_popup, draw_dashboard_button)
- 主题与 3D 视觉语言：src.ui.gui_theme (GuiTheme)
"""

from src.ui.gui_theme import GuiTheme
from src.ui.gui_components import (
    draw_dropdown_button,
    draw_dashboard_button,
    render_dropdown_popup,
)

# 3D 视觉比对与双四棱柱选项 (取自全局单源 GuiTheme)
VIEW_MODE_OPTIONS = GuiTheme.VIEW_MODE_OPTIONS
BA_VIEW_OPTIONS = GuiTheme.BA_VIEW_OPTIONS
OBS_VIEW_OPTIONS = GuiTheme.OBS_VIEW_OPTIONS

# 空间建图工作站专属帧列表筛选模式
FILTER_MODE_OPTIONS = [
    ("all", "全部帧"),
    ("warning", "高残差 (>0.5px)"),
    ("excluded", "已剔除帧")
]

# 空间建图工作站专属帧列表排序模式
SORT_MODE_OPTIONS = [
    ("name_asc", "文件名升序"),
    ("tag_err_desc", "Tag残差降序 (最差优先 ↓)"),
    ("err_desc", "残差降序 (帧均最差 ↓)"),
    ("err_asc", "残差升序 (最优优先 ↑)"),
    ("tags_desc", "标靶数量降序")
]
