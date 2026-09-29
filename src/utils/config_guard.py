#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统配置安全守卫与动态防呆校验器 (ConfigGuard)
- 解决相机内参与输入图像分辨率脱节、导致空间反投影畸变的核心隐患
- 提供多分辨率自适应线性仿射缩放 (Auto-scaling)
- 跨配置项交叉物理校验 (机器人行程安全、深度门限合理性、单位阵回退拦截)
"""

import os
import yaml
import numpy as np
from typing import Dict, Tuple, Optional, Any

from src.utils.logger import get_logger

log = get_logger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SYSTEM_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")

_WARNED_SCALES = set()


def load_raw_config(config_path: str = SYSTEM_CONFIG_PATH) -> Dict[str, Any]:
    """安全读取 YAML 配置文件"""
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        log.warning(f"[WARN] 读取配置文件 '{config_path}' 失败: {e}")
        return {}


def _parse_anchor_entry(entry: Any) -> Optional[Dict[str, Any]]:
    """解析单条锚点: {"xyz_mm": [x,y,z], "known": [b,b,b]}，无效/全未知条目返回 None"""
    if not isinstance(entry, dict):
        return None
    xyz = entry.get("xyz_mm")
    if xyz is None or len(xyz) != 3:
        return None
    try:
        xyz_f = [float(v) for v in xyz]
    except (TypeError, ValueError):
        return None
    known_raw = entry.get("known", [True, True, True])
    if not isinstance(known_raw, (list, tuple)) or len(known_raw) != 3:
        known_raw = [True, True, True]
    known = [bool(v) for v in known_raw]
    res = {"xyz_mm": xyz_f, "known": known}
    if entry.get("frame_id"):
        res["frame_id"] = str(entry["frame_id"])
    return res


def parse_anchor_mapping(raw: Any) -> Dict[int, Dict[str, Any]]:
    """
    解析 anchor_tags 映射 (全局 config.yaml 与工位 anchor_tags.yaml 共用):
    {tag_id: {"xyz_mm": [f3], "known": [b3]}}，无效键/条目丢弃，全未知条目丢弃
    """
    anchors: Dict[int, Dict[str, Any]] = {}
    if not isinstance(raw, dict) or not raw:
        return anchors
    for k, v in raw.items():
        try:
            tid = int(k)
        except (TypeError, ValueError):
            continue
        entry = _parse_anchor_entry(v)
        if entry is not None:
            anchors[tid] = entry
    return anchors


def load_anchor_tags(config_path: str = SYSTEM_CONFIG_PATH) -> Dict[int, Dict[str, Any]]:
    """
    读取全局 config.yaml 的世界坐标锚点表 (旧版数据源, 工位未建 anchor_tags.yaml 时的兜底)。
    当前推荐数据源为每工位独立文件 (load_workspace_anchor_tags)。
    - 新格式: {tag_id: {"xyz_mm": [x,y,z], "known": [b,b,b]}}，known 缺省视为三轴全知
    - 兼容迁移: 若 anchor_tags 缺失，自动从旧 world_anchor (origin/align 两枚全知锚点) 转换
    :return: {int tag_id: {"xyz_mm": [float x3], "known": [bool x3]}}，无有效锚点返回空字典
    """
    cfg = load_raw_config(config_path)
    calib = cfg.get("calibration", {})

    anchors = parse_anchor_mapping(calib.get("anchor_tags"))
    if anchors:
        return anchors

    # 兼容迁移: 旧 world_anchor 双锚点格式 (origin/align 均视为三轴全知)
    wa = calib.get("world_anchor")
    if isinstance(wa, dict):
        for tid_key, xyz_key in (("origin_tag_id", "origin_xyz_mm"), ("align_tag_id", "align_xyz_mm")):
            if wa.get(tid_key) is not None and wa.get(xyz_key):
                entry = _parse_anchor_entry({"xyz_mm": wa[xyz_key], "known": [True, True, True]})
                if entry is not None:
                    anchors[int(wa[tid_key])] = entry
        if anchors:
            log.info(f"[GUARD] 已从旧 world_anchor 迁移 {len(anchors)} 枚世界锚点 (建议改用 calibration.anchor_tags)")
    return anchors


def resolve_camera_intrinsics(
    config_path: str = SYSTEM_CONFIG_PATH,
    actual_image_shape: Optional[Tuple[int, int]] = None,
    stream_profile: Optional[Any] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    自适应获取并校验相机内参 (具有防呆与分辨率自适应缩放能力)
    
    :param config_path: config.yaml 路径
    :param actual_image_shape: 实际图像尺寸 (height, width)，用于校验与自适应缩放
    :param stream_profile: pyrealsense2 的 video_stream_profile 活体句柄 (若可用)
    :return: (camera_matrix 3x3, dist_coeffs 5x1, metadata_dict)
    """
    metadata = {
        "source": "default",
        "base_resolution": (1080, 1920),
        "actual_resolution": None,
        "scaled": False,
        "scale_factor": 1.0
    }
    
    # 默认 D435 1080P 出厂实测内参
    fx, fy = 1363.68, 1361.19
    cx, cy = 971.19, 566.26
    dist = np.zeros((5, 1), dtype=np.float64)
    ref_w, ref_h = 1920, 1080

    # 1. 硬件活体在线直读 (最高优先级)
    if stream_profile is not None:
        try:
            intr = stream_profile.get_intrinsics()
            fx, fy = float(intr.fx), float(intr.fy)
            cx, cy = float(intr.ppx), float(intr.ppy)
            if hasattr(intr, 'coeffs'):
                dist = np.array(intr.coeffs[:5], dtype=np.float64).reshape((5, 1))
            ref_w, ref_h = int(intr.width), int(intr.height)
            metadata["source"] = "realsense_hardware_profile"
            metadata["base_resolution"] = (ref_h, ref_w)
        except Exception as e:
            log.warning(f"[WARN] 从硬件 profile 读取内参失败: {e}，回退至配置文件")

    # 2. 从 config.yaml 读取
    if metadata["source"] == "default":
        cfg = load_raw_config(config_path)
        cam_col = cfg.get("camera", {}).get("color", {})
        if cam_col:
            ref_w = int(cam_col.get("width", 1920))
            ref_h = int(cam_col.get("height", 1080))
            fx = float(cam_col.get("fx", fx))
            fy = float(cam_col.get("fy", fy))
            cx = float(cam_col.get("cx", cx))
            cy = float(cam_col.get("cy", cy))
            metadata["source"] = "config_yaml"
            metadata["base_resolution"] = (ref_h, ref_w)

    # 3. 图像实际尺寸比对与自适应等比缩放
    if actual_image_shape is not None:
        act_h, act_w = actual_image_shape[:2]
        metadata["actual_resolution"] = (act_h, act_w)

        if (act_w != ref_w) or (act_h != ref_h):
            scale_x = float(act_w) / float(ref_w)
            scale_y = float(act_h) / float(ref_h)
            
            # 执行仿射等比缩放
            fx_scaled = fx * scale_x
            fy_scaled = fy * scale_y
            cx_scaled = cx * scale_x
            cy_scaled = cy * scale_y
            
            scale_key = (act_w, act_h, ref_w, ref_h)
            if scale_key not in _WARNED_SCALES:
                _WARNED_SCALES.add(scale_key)
                log.warning(f"[GUARD] 检测到图像分辨率 ({act_w}x{act_h}) 与内参基准 ({ref_w}x{ref_h}) 不一致，已等比自适应缩放内参。")
            else:
                log.debug(f"[GUARD] 持续应用分辨率内参缩放: ({act_w}x{act_h}) <- ({ref_w}x{ref_h})")
            
            fx, fy, cx, cy = fx_scaled, fy_scaled, cx_scaled, cy_scaled
            metadata["scaled"] = True
            metadata["scale_factor"] = (scale_x + scale_y) / 2.0
            ref_w, ref_h = act_w, act_h

        # 4. 主点中心度合理性断言 (Sanity Check)
        if not (0.30 * act_w <= cx <= 0.70 * act_w) or not (0.30 * act_h <= cy <= 0.70 * act_h):
            err_msg = (f"[CRITICAL] 相机主点 ({cx:.1f}, {cy:.1f}) 严重偏离图像几何中心 ({act_w//2}, {act_h//2})！"
                       f"存在严重的内参或分辨率错配风险！")
            log.error(err_msg)

    K = np.array([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    return K, dist, metadata


def validate_system_config(config_path: str = SYSTEM_CONFIG_PATH) -> Dict[str, Any]:
    """
    对 config.yaml 执行全方位的系统级健康诊断与安全隐患排查
    """
    cfg = load_raw_config(config_path)
    issues = []
    warnings = []
    
    if not cfg:
        return {"ok": False, "fatal_errors": ["配置文件不存在或内容为空！"], "warnings": []}

    # 1. 机器人安全高度校验
    r_cfg = cfg.get("robot", {})
    safe_z = float(r_cfg.get("safe_z_mm", 80.0))
    drop_z = float(r_cfg.get("drop_z_mm", 50.0))
    if safe_z <= drop_z:
        issues.append(f"机械臂平移安全高度 safe_z_mm ({safe_z}mm) 必须严格大于落料高度 drop_z_mm ({drop_z}mm)！存在严重撞机风险！")
    elif safe_z < drop_z + 15.0:
        warnings.append(f"安全高度 safe_z_mm ({safe_z}mm) 与 drop_z_mm ({drop_z}mm) 余量小于 15mm，高速转移时有刮蹭风险。")

    # 2. 深度门限合理性校验
    cam_cfg = cfg.get("camera", {})
    thresh_cfg = cam_cfg.get("filters", {}).get("threshold", {})
    if thresh_cfg.get("enabled", False):
        min_d = float(thresh_cfg.get("min_distance", 0.40))
        max_d = float(thresh_cfg.get("max_distance", 0.70))
        if min_d >= max_d:
            issues.append(f"深度截断门限异常: min_distance ({min_d}m) >= max_distance ({max_d}m)！将导致所有深度点被完全抹除！")
        elif max_d < 0.65:
            warnings.append(f"最大深度门限 max_distance ({max_d}m) 较小，可能将工作台底板切掉，导致 RANSAC 平面拟合失败。")

    # 3. 手眼标定矩阵回退校验
    calib = cfg.get("calibration", {})
    t_mat = np.array(calib.get("t_cam_to_scara", np.eye(4)), dtype=float)
    if np.allclose(t_mat, np.eye(4)):
        warnings.append("手工手眼标定矩阵 t_cam_to_scara 仍为单位矩阵！若 AprilTag 地图失效，回退动作可能导致机械臂 Z 轴大行程撞击！")

    # 4. 白名单校验
    valid_tags = calib.get("valid_tag_ids", [])
    if not valid_tags:
        warnings.append("未配置标靶白名单 valid_tag_ids，系统将对所有 AprilTag 码放行，容易受到视野外未知反光噪点干扰。")

    return {
        "ok": len(issues) == 0,
        "fatal_errors": issues,
        "warnings": warnings
    }
