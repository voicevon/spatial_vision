# -*- coding: utf-8 -*-
"""
单目相机内参标定求解器 (CameraIntrinsicsCalibrator)
=================================================
基于标准棋盘格标定板（Chessboard），从全局相机档案 (`data/cameras/<camera_id>/<WxH>/raw_images/`)
提取角点并亚像素精化，求解相机物理内参矩阵 (K)、径向/切向畸变系数 (D)、
全局重投影均方根误差 (RMSE) 及逐帧重投影误差，持久化至 `intrinsics.yaml`。
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.cameras.camera_entity import CameraProfile, IntrinsicsRecord
from src.utils.logger import get_logger
from src.utils.image_io import imread_unicode

log = get_logger(__name__)


@dataclass
class IntrinsicsCalibrationResult:
    """相机内参标定计算结果数据结构"""
    success: bool
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    dist_coeffs: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    image_size: Tuple[int, int] = (0, 0)  # (width, height)
    rmse: float = 0.0                     # 全局重投影均方根误差 (pixels)
    valid_images_count: int = 0           # 成功检测到棋盘格的有效图片数量
    total_images_count: int = 0           # 输入待解算的总图片数量
    per_image_errors: List[Dict[str, Any]] = field(default_factory=list)
    message: str = ""


def detect_chessboard_corners(
    frame_bgr: np.ndarray,
    pattern_size: Tuple[int, int] = (9, 6),
    refine_subpix: bool = False,
    max_detect_width: int = 640,
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    单帧实时棋盘格内角点检测（供 Camera Hub 实时预览引导与抓拍叠加）。
    为保证预览流畅，支持在缩小灰度图上做快速初检再按比例映射回原图坐标。
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return False, None
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    h, w = gray.shape[:2]
    scale = 1.0
    work_gray = gray
    if max_detect_width > 0 and w > max_detect_width:
        scale = float(max_detect_width) / float(w)
        work_gray = cv2.resize(gray, (max_detect_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ret, corners = cv2.findChessboardCorners(work_gray, pattern_size, flags)
    if not ret or corners is None:
        return False, None

    if scale != 1.0:
        corners = corners / scale

    if refine_subpix:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners.astype(np.float32), (11, 11), (-1, -1), criteria)

    return True, corners


def calibrate_camera_from_images(
    image_paths: List[str],
    pattern_size: Tuple[int, int] = (9, 6),
    square_size_mm: float = 25.0,
) -> IntrinsicsCalibrationResult:
    """
    基于棋盘格图片集解算相机内参与畸变系数，并输出每帧重投影误差。

    :param image_paths: 棋盘格标定图片路径列表
    :param pattern_size: 棋盘格内部角点规格 (cols, rows)，默认 (9, 6)
    :param square_size_mm: 每个黑白方格物理边长 (毫米)，默认 25.0 mm
    :return: IntrinsicsCalibrationResult
    """
    total_count = len(image_paths)
    if total_count < 3:
        return IntrinsicsCalibrationResult(
            success=False,
            total_images_count=total_count,
            message=f"内参标定至少需要 3 张不同角度的标定板照片，当前仅有 {total_count} 张",
        )

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= float(square_size_mm)

    obj_points: List[np.ndarray] = []
    img_points: List[np.ndarray] = []
    valid_indices: List[int] = []
    per_image_errors: List[Dict[str, Any]] = []
    img_size: Optional[Tuple[int, int]] = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for img_path in image_paths:
        entry = {
            "path": img_path,
            "filename": os.path.basename(img_path),
            "detected": False,
            "error_px": None,
        }
        per_image_errors.append(entry)

        if not os.path.exists(img_path):
            continue

        img = imread_unicode(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        if img_size is None:
            img_size = (w, h)
        elif img_size != (w, h):
            log.warning(f"跳过分辨率不匹配图片: {img_path} ({w}x{h} vs 期望 {img_size[0]}x{img_size[1]})")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

        if ret and corners is not None:
            refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(refined_corners)
            entry["detected"] = True
            valid_indices.append(len(per_image_errors) - 1)

    valid_count = len(obj_points)
    if valid_count < 3:
        return IntrinsicsCalibrationResult(
            success=False,
            valid_images_count=valid_count,
            total_images_count=total_count,
            image_size=img_size or (0, 0),
            per_image_errors=per_image_errors,
            message=f"棋盘格角点有效检出张数不足 ({valid_count}/{total_count})，至少需要 3 张，请检查标定板内角点规格或反光遮挡",
        )

    try:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, img_size, None, None
        )
    except Exception as e:
        log.error(f"cv2.calibrateCamera 失败: {e}")
        return IntrinsicsCalibrationResult(
            success=False,
            valid_images_count=valid_count,
            total_images_count=total_count,
            image_size=img_size or (0, 0),
            per_image_errors=per_image_errors,
            message=f"标定求解异常: {e}",
        )

    # 计算每帧重投影均方根误差 (RMSE px)
    for k, entry_idx in enumerate(valid_indices):
        proj, _ = cv2.projectPoints(obj_points[k], rvecs[k], tvecs[k], mtx, dist)
        diff = img_points[k].reshape(-1, 2) - proj.reshape(-1, 2)
        err_px = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
        per_image_errors[entry_idx]["error_px"] = round(err_px, 4)

    fx = float(mtx[0, 0])
    fy = float(mtx[1, 1])
    cx = float(mtx[0, 2])
    cy = float(mtx[1, 2])
    dist_list = [float(x) for x in dist.ravel()[:5]]
    while len(dist_list) < 5:
        dist_list.append(0.0)

    return IntrinsicsCalibrationResult(
        success=True,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        dist_coeffs=dist_list,
        image_size=img_size,
        rmse=float(ret),
        valid_images_count=valid_count,
        total_images_count=total_count,
        per_image_errors=per_image_errors,
        message=f"标定成功: 有效图片 {valid_count}/{total_count} 张, 重投影误差 RMSE={ret:.3f} px",
    )


def calibrate_camera_profile(
    camera: CameraProfile,
    resolution: str,
    pattern_size: Tuple[int, int] = (9, 6),
    square_size_mm: float = 25.0,
    save_to_profile: bool = True,
) -> IntrinsicsCalibrationResult:
    """
    从指定相机档案的分辨率图集 (`data/cameras/<camera_id>/<resolution>/raw_images/`) 加载图片并标定。
    解算成功且 save_to_profile 为 True 时，持久化写入 `intrinsics.yaml`。
    """
    images = camera.list_images(resolution)
    if not images:
        return IntrinsicsCalibrationResult(
            success=False,
            message=f"相机 {camera.camera_id} ({resolution}) 尚无棋盘格标定照片",
        )

    res = calibrate_camera_from_images(
        image_paths=images,
        pattern_size=pattern_size,
        square_size_mm=square_size_mm,
    )

    if res.success and save_to_profile:
        record = IntrinsicsRecord(
            width=int(res.image_size[0]),
            height=int(res.image_size[1]),
            fx=res.fx,
            fy=res.fy,
            cx=res.cx,
            cy=res.cy,
            dist_coeffs=res.dist_coeffs,
            rmse=res.rmse,
            valid_images_count=res.valid_images_count,
            total_images_count=res.total_images_count,
            pattern_size=[int(pattern_size[0]), int(pattern_size[1])],
            square_size_mm=float(square_size_mm),
            per_image_errors=[
                {
                    "filename": e["filename"],
                    "detected": e["detected"],
                    "error_px": e["error_px"],
                }
                for e in res.per_image_errors
            ],
            calibrated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        ok = camera.save_intrinsics(resolution, record)
        if ok:
            log.info(f"已成功将标定内参写入相机档案: {camera.camera_id} ({resolution})")
        else:
            log.error(f"保存内参到相机档案失败: {camera.camera_id} ({resolution})")

    return res
