#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
芦笋位姿工作室 (Asparagus Pose Studio)
======================================
从原始离线验证脚本重构而来的模块化工具包。
提供芦笋 3D/2D 抓取位姿解算、算法流水线对比、多阶段特征图视口缩放与 G-code 导出。
"""

from src.workspace.workspace_manager import WorkspaceManager
from tools.asparagus_pose_studio.data_io import (
    APP_ID,
    BASE_H,
    BASE_W,
    CALIB_LABELS,
    CONFIG_PATH,
    DEFAULT_DIR,
    REPORT_DIR,
    GUI_SETTINGS_FILE,
    WINDOW_KEY,
    export_gcode_file,
    find_depth_pair,
    load_system_config,
    load_studio_settings,
    save_studio_settings,
    scan_samples,
)
from tools.asparagus_pose_studio.renderer import AsparagusPoseStudioRenderer
from tools.asparagus_pose_studio.app import AsparagusPoseStudioApp, main

__all__ = [
    "AsparagusPoseStudioApp",
    "AsparagusPoseStudioRenderer",
    "find_depth_pair",
    "scan_samples",
    "load_system_config",
    "load_studio_settings",
    "save_studio_settings",
    "export_gcode_file",
    "WorkspaceManager",
    "main",
    "APP_ID",
    "WINDOW_KEY",
    "BASE_W",
    "BASE_H",
    "DEFAULT_DIR",
    "REPORT_DIR",
    "GUI_SETTINGS_FILE",
    "CONFIG_PATH",
    "CALIB_LABELS",
]
