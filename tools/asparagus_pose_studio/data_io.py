#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
芦笋位姿工作室 (Asparagus Pose Studio) - 数据与配置模块
=====================================================
负责配置加载、生产样本扫描以及原始深度对齐配对。
"""

import os
import glob
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from src.utils.logger import get_logger

log = get_logger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
DEFAULT_DIR = os.path.join(PROJECT_ROOT, "data", "snapshots")
GUI_SETTINGS_FILE = os.path.join(PROJECT_ROOT, "config", "gui_settings.json")

WINDOW_KEY = "AsparagusPoseStudio"   # cv2 窗口内部 key (纯 ASCII, 标题经 win32 API 动态设置)
APP_ID = "asparagus_pose_studio"
BASE_W, BASE_H = 1280, 800          # 基准逻辑画布尺寸

CALIB_LABELS = {
    "tag_online": "AprilTag 在线定位",
    "tag_cached": "AprilTag 缓存外参",
    "hand_eye": "手工 SVD 标定",
    "2d_preview": "2D 预览 (无深度)",
    "uncalibrated": "未标定 - 防撞保护",
}


def load_studio_settings(settings_file: Optional[str] = None) -> Dict[str, Any]:
    """
    从 config/gui_settings.json 读取芦笋位姿工作室的持久化偏好 (算法路线、样本选中)
    """
    path = settings_file or GUI_SETTINGS_FILE
    if not os.path.exists(path):
        return {}
    try:
        import json
        with open(path, "r", encoding="utf-8") as f:
            root = json.load(f)
        if not isinstance(root, dict):
            return {}
        data = root.get(APP_ID, {})
        return data.get("studio_state") or {}
    except Exception as exc:
        log.warning("读取 gui_settings.json 失败: %s", exc)
        return {}


def save_studio_settings(state: Dict[str, Any], settings_file: Optional[str] = None):
    """
    将芦笋位姿工作室的偏好状态安全持久化至 config/gui_settings.json
    """
    path = settings_file or GUI_SETTINGS_FILE
    try:
        import json
        root = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    root = json.load(f)
                if not isinstance(root, dict):
                    root = {}
            except Exception:
                root = {}
        node = root.setdefault(APP_ID, {})
        studio_state = node.setdefault("studio_state", {})
        studio_state.update(state)
        node["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(root, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        log.warning("保存 gui_settings.json 失败: %s", exc)



def load_system_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    读取 config.yaml: 相机内参/标定矩阵/机械臂安全参数 (与生产链路同源)
    """
    cfg_file = config_path or CONFIG_PATH
    cfg = {
        "intrinsics": None,          # (fx, fy, cx, cy, width, height)
        "t_cam_to_scara": None,
        "tags_map_path": "",
        "safe_z": 80.0,
        "drop_x": 220.0,
        "drop_y": 0.0,
    }
    if not os.path.exists(cfg_file):
        return cfg

    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        cam = raw.get("camera", {}).get("color", {})
        if all(k in cam for k in ("fx", "fy", "cx", "cy")):
            cfg["intrinsics"] = (
                float(cam["fx"]), float(cam["fy"]),
                float(cam["cx"]), float(cam["cy"]),
                int(cam.get("width", 0)), int(cam.get("height", 0))
            )
        calib = raw.get("calibration", {})
        if calib.get("t_cam_to_scara"):
            cfg["t_cam_to_scara"] = np.array(calib["t_cam_to_scara"], dtype=float)
        cfg["tags_map_path"] = calib.get("tags_map_path", "") or ""
        robot = raw.get("robot", {})
        cfg["safe_z"] = float(robot.get("safe_z_mm", 80.0))
        cfg["drop_x"] = float(robot.get("drop_x_mm", 220.0))
        cfg["drop_y"] = float(robot.get("drop_y_mm", 0.0))
    except Exception as exc:
        log.warning("config.yaml 加载异常 (使用缺省值): %s", exc)
    return cfg


def find_depth_pair(png_path: str) -> Optional[str]:
    """
    彩色 png 配对原始深度 npy:
    支持两种命名规则:
      1. 同茎同名: img.png <-> img.npy
      2. d435_viewer 抓拍规则: color_TS.png <-> depth_raw_TS.npy
    """
    folder = os.path.dirname(png_path)
    stem = os.path.splitext(os.path.basename(png_path))[0]
    candidates = [stem + ".npy"]
    if stem.startswith("color_"):
        candidates.append("depth_raw_" + stem[len("color_"):] + ".npy")
    for name in candidates:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return None


def scan_samples(sample_dir: str) -> List[Dict[str, Any]]:
    """
    扫描样本目录: 返回 [{'png': path, 'depth': path_or_none, 'name': basename}]
    过滤可视化派生图 (depth_vis_*, height_vis_*)，按修改时间倒序排列 (最新在前)。
    """
    samples = []
    if not os.path.isdir(sample_dir):
        return samples
    patterns = [os.path.join(sample_dir, "*.png"), os.path.join(sample_dir, "*.jpg")]
    all_imgs = []
    for pat in patterns:
        all_imgs.extend(glob.glob(pat))

    for png in all_imgs:
        base = os.path.basename(png)
        if base.startswith(("depth_vis_", "height_vis_")):
            continue  # 跳过中间派生可视化图，只保留原始采集帧
        samples.append({
            "png": png,
            "depth": find_depth_pair(png),
            "name": base
        })
    samples.sort(key=lambda s: os.path.getmtime(s["png"]), reverse=True)
    return samples


