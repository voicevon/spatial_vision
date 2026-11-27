#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单根芦笋测量工作室 - 数据、工位 ROI 投影与偏好持久化模块
========================================================
"""

import csv
import glob
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml

from src.utils.image_io import imread_unicode
from src.utils.logger import get_logger
from src.vision.single_asparagus_measurer import MeasureConfig
from src.workspace.coordinate_manager import CoordinateTreeManager
from src.workspace.roi_manager import RoiSpaceManager

log = get_logger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
SORTER_STATION_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "sorter_station.yaml")
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


# ------------------------------ 项目 1 产线配置同步与批量真值评测 ------------------------------


@dataclass
class BatchEvalSummary:
    """离线图集批量测量与卡尺真值对照汇总"""
    total: int = 0
    valid_count: int = 0
    gt_matched_count: int = 0
    len_mae_mm: Optional[float] = None
    len_max_err_mm: Optional[float] = None
    diam_mae_mm: Optional[float] = None
    diam_max_err_mm: Optional[float] = None
    grade_counts: Dict[int, int] = field(default_factory=dict)
    per_sample: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def load_sorter_station_yaml(config_path: Optional[str] = None) -> Dict[str, Any]:
    """读取项目 1 分拣工位单一配置文件 (sorter_station.yaml)"""
    path = config_path or SORTER_STATION_CONFIG_PATH
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        log.warning("读取 sorter_station.yaml 失败: %s", exc)
        return {}


def save_measure_config_to_station_yaml(
    cfg: MeasureConfig,
    config_path: Optional[str] = None,
) -> bool:
    """将 Studio 调好的测量参数原子写回 sorter_station.yaml 的 measure 节（保留其余配置）"""
    path = config_path or SORTER_STATION_CONFIG_PATH
    try:
        data = load_sorter_station_yaml(path)
        measure_sec = dict(data.get("measure") or {})
        measure_sec.update({
            "orientation": str(cfg.orientation),
            "diameter_offset_mm": round(float(cfg.diameter_offset_mm), 2),
            "green_h_low": int(cfg.green_h_low),
            "green_h_high": int(cfg.green_h_high),
            "green_s_min": int(cfg.green_s_min),
            "green_v_min": int(cfg.green_v_min),
            "white_s_max": int(cfg.white_s_max),
            "white_v_min": int(cfg.white_v_min),
            "green_ratio_threshold": round(float(cfg.green_ratio_threshold), 3),
            "min_white_root_mm": round(float(cfg.min_white_root_mm), 2),
            "min_area_px": int(cfg.min_area_px),
        })
        data["measure"] = measure_sec

        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        os.replace(tmp_path, path)
        return True
    except Exception as exc:
        log.warning("导出测量参数至 sorter_station.yaml 失败: %s", exc)
        return False


def load_ground_truth_csv(sample_dir: str) -> Dict[str, Tuple[float, float]]:
    """
    从样本目录加载可选的卡尺真值表 `ground_truth.csv`。
    支持列名: filename, true_len_mm, true_diam_mm（容忍 UTF-8 BOM 与大小写）。
    返回: {basename: (true_len_mm, true_diam_mm)}
    """
    if not sample_dir or not os.path.isdir(sample_dir):
        return {}
    csv_path = os.path.join(sample_dir, "ground_truth.csv")
    if not os.path.exists(csv_path):
        return {}

    gt: Dict[str, Tuple[float, float]] = {}
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return {}
            norm_map = {name.strip().lower(): name for name in reader.fieldnames if name}
            fn_col = norm_map.get("filename") or norm_map.get("file") or norm_map.get("name")
            len_col = norm_map.get("true_len_mm") or norm_map.get("length_mm") or norm_map.get("len_mm")
            dia_col = norm_map.get("true_diam_mm") or norm_map.get("diameter_mm") or norm_map.get("diam_mm")
            if not (fn_col and len_col and dia_col):
                return {}
            for row in reader:
                fname = (row.get(fn_col) or "").strip()
                if not fname:
                    continue
                try:
                    t_len = float(row[len_col])
                    t_dia = float(row[dia_col])
                    gt[os.path.basename(fname)] = (t_len, t_dia)
                except (TypeError, ValueError):
                    continue
    except Exception as exc:
        log.warning("读取 ground_truth.csv 失败: %s", exc)
    return gt


def run_batch_evaluation(
    samples: List[Dict[str, Any]],
    measure_fn: Callable[[np.ndarray, Optional[np.ndarray]], Tuple[Any, Optional[Any]]],
    ground_truth: Optional[Dict[str, Tuple[float, float]]] = None,
) -> BatchEvalSummary:
    """
    对离线样本集执行批量测量与可选真值误差统计。
    :param measure_fn: callable(color_bgr, depth_mm) -> (SingleAsparagusMeasurement, Optional[GradeDecision])
    """
    gt = ground_truth or {}
    summary = BatchEvalSummary(total=len(samples))
    len_errs: List[float] = []
    diam_errs: List[float] = []

    for smp in samples:
        name = smp["name"]
        color = imread_unicode(smp["color_path"])
        if color is None:
            summary.per_sample[name] = {"valid": False, "reason": "读取失败"}
            continue
        depth = None
        if smp.get("depth_path") and os.path.exists(smp["depth_path"]):
            try:
                depth = np.load(smp["depth_path"]).astype(np.float32)
            except Exception:
                depth = None

        meas, grade_dec = measure_fn(color, depth)
        entry: Dict[str, Any] = {
            "valid": bool(meas and meas.valid),
            "length_mm": float(meas.effective_length_mm) if (meas and meas.valid) else 0.0,
            "diameter_mm": float(meas.diameter_mm) if (meas and meas.valid) else 0.0,
            "grade": int(grade_dec.grade) if grade_dec is not None else None,
        }
        if meas and meas.valid:
            summary.valid_count += 1
            if grade_dec is not None:
                summary.grade_counts[grade_dec.grade] = summary.grade_counts.get(grade_dec.grade, 0) + 1
            if name in gt:
                true_l, true_d = gt[name]
                dl = abs(meas.effective_length_mm - true_l)
                dd = abs(meas.diameter_mm - true_d)
                len_errs.append(dl)
                diam_errs.append(dd)
                entry["len_err_mm"] = round(meas.effective_length_mm - true_l, 2)
                entry["diam_err_mm"] = round(meas.diameter_mm - true_d, 2)
        summary.per_sample[name] = entry

    summary.gt_matched_count = len(len_errs)
    if len_errs:
        summary.len_mae_mm = round(float(np.mean(len_errs)), 2)
        summary.len_max_err_mm = round(float(np.max(len_errs)), 2)
        summary.diam_mae_mm = round(float(np.mean(diam_errs)), 2)
        summary.diam_max_err_mm = round(float(np.max(diam_errs)), 2)
    return summary
