#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单根芦笋测量工作室 - 数据、工位 ROI 投影与偏好持久化模块
========================================================
"""

import glob
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from src.utils.logger import get_logger
from src.workspace.coordinate_manager import CoordinateTreeManager
from src.workspace.roi_manager import RoiSpaceManager

log = get_logger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
DEFAULT_DIR = os.path.join(PROJECT_ROOT, "data", "snapshots")
GUI_SETTINGS_FILE = os.path.join(PROJECT_ROOT, "config", "gui_settings.json")

WINDOW_KEY = "AsparagusMeasureStudio"
APP_ID = "asparagus_measure_studio"
BASE_W, BASE_H = 1280, 800


def load_measure_settings(settings_file: Optional[str] = None) -> Dict[str, Any]:
    """从 gui_settings.json 读取单根芦笋测量工作室的持久化状态"""
    path = settings_file or GUI_SETTINGS_FILE
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            root = json.load(f)
        if not isinstance(root, dict):
            return {}
        node = root.get(APP_ID, {})
        return node.get("measure_state") or {}
    except Exception as exc:
        log.warning("读取测量工作室偏好失败: %s", exc)
        return {}


def save_measure_settings(state: Dict[str, Any], settings_file: Optional[str] = None):
    """安全持久化单根芦笋测量工作室偏好至 gui_settings.json"""
    path = settings_file or GUI_SETTINGS_FILE
    try:
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
        m_state = node.setdefault("measure_state", {})
        m_state.update(state)
        node["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(root, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        log.warning("保存测量工作室偏好失败: %s", exc)


def load_camera_intrinsics_cfg(config_path: Optional[str] = None) -> Optional[Tuple[float, float, float, float, int, int]]:
    """读取 config.yaml 中的彩色相机内参 (fx, fy, cx, cy, width, height)"""
    cfg_file = config_path or CONFIG_PATH
    if not os.path.exists(cfg_file):
        return None
    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        cam = raw.get("camera", {}).get("color", {})
        if all(k in cam for k in ("fx", "fy", "cx", "cy")):
            return (
                float(cam["fx"]), float(cam["fy"]),
                float(cam["cx"]), float(cam["cy"]),
                int(cam.get("width", 0)), int(cam.get("height", 0)),
            )
    except Exception as exc:
        log.warning("读取 config.yaml 相机内参失败: %s", exc)
    return None


def scan_samples(sample_dir: str) -> List[Dict[str, Any]]:
    """扫描样本目录，返回按时间倒序排列的样本列表（自动配对可选对齐深度 .npy）"""
    if not sample_dir or not os.path.isdir(sample_dir):
        return []
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    paths: List[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(sample_dir, ext)))
    paths = sorted(set(paths), key=lambda p: os.path.getmtime(p), reverse=True)

    samples: List[Dict[str, Any]] = []
    for p in paths:
        base = os.path.splitext(os.path.basename(p))[0]
        if base.endswith("_depth") or base.endswith("_vis"):
            continue
        depth_candidates = [
            os.path.join(sample_dir, f"{base}_depth.npy"),
            os.path.join(sample_dir, f"{base}.npy"),
            os.path.join(sample_dir, f"{base.replace('_color', '_depth')}.npy"),
        ]
        depth_path = next((dp for dp in depth_candidates if os.path.exists(dp)), None)
        samples.append({
            "name": os.path.basename(p),
            "color_path": p,
            "depth_path": depth_path,
            "has_depth": depth_path is not None,
        })
    return samples


def project_workspace_roi_to_2d(
    roi_id: str,
    roi_mgr: Optional[RoiSpaceManager],
    coord_mgr: Optional[CoordinateTreeManager],
    t_cam_to_world: Optional[np.ndarray],
    intrinsics: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> Optional[Tuple[int, int, int, int]]:
    """
    将工位 3D ROI 的 8 个世界顶点投影到当前相机 2D 像素外接矩形 (x, y, w, h)。
    若缺少外参或 ROI 不可见返回 None。
    """
    if not roi_id or roi_mgr is None or coord_mgr is None or t_cam_to_world is None:
        return None
    obb = roi_mgr.get_roi_world_obb(roi_id, coord_mgr)
    if not obb or not obb.get("is_resolved"):
        return None

    corners_w = obb["corners_8x3"]  # (8, 3)
    try:
        t_world_to_cam = np.linalg.inv(np.asarray(t_cam_to_world, dtype=np.float64))
    except np.linalg.LinAlgError:
        return None

    homo_w = np.hstack([corners_w, np.ones((8, 1), dtype=np.float64)])
    corners_c = (t_world_to_cam @ homo_w.T).T[:, :3]
    valid = corners_c[:, 2] > 20.0
    if not np.any(valid):
        return None

    fx, fy, cx, cy = [float(v) for v in intrinsics]
    pts_c = corners_c[valid]
    us = fx * (pts_c[:, 0] / pts_c[:, 2]) + cx
    vs = fy * (pts_c[:, 1] / pts_c[:, 2]) + cy

    x1 = max(0, min(img_w - 1, int(np.floor(us.min()))))
    y1 = max(0, min(img_h - 1, int(np.floor(vs.min()))))
    x2 = max(0, min(img_w, int(np.ceil(us.max()))))
    y2 = max(0, min(img_h, int(np.ceil(vs.max()))))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return (x1, y1, x2 - x1, y2 - y1)
