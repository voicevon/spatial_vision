# -*- coding: utf-8 -*-
"""
单目相机内参标定求解器 (CameraIntrinsicsCalibrator)
=================================================
基于标准棋盘格标定板（Chessboard），从工位内参图集 (`intrinsics/raw_images/`)
提取角点并亚像素精化，求解相机物理内参矩阵 (K)、径向/切向畸变系数 (D) 以及重投影均方根误差 (RMSE)。
解算成功后直接持久化至工位沙盒配置 (`calibration/camera_intrinsics.yaml`)。
"""

import os
import glob
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import cv2
import numpy as np

from src.utils.logger import get_logger
from src.workspace.workspace_manager import Workspace

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
    rmse: float = 0.0                     # 重投影均方根误差 (pixels)
    valid_images_count: int = 0           # 成功检测到棋盘格的有效图片数量
    total_images_count: int = 0           # 输入待解算的总图片数量
    message: str = ""


def calibrate_camera_from_images(
    image_paths: List[str],
    pattern_size: Tuple[int, int] = (9, 6),
    square_size_mm: float = 25.0,
) -> IntrinsicsCalibrationResult:
    """
    基于棋盘格图片集解算相机内参与畸变系数。

    :param image_paths: 棋盘格标定图片路径列表
    :param pattern_size: 棋盘格内部角点规格 (cols, rows)，默认 (9, 6) 即 9 列 6 行内角点
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

    # 1. 准备棋盘格 3D 物理空间坐标 (Z=0 平面)
    # objp 形状: (pattern_size[0] * pattern_size[1], 3)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    obj_points: List[np.ndarray] = []  # 3D 物理点集合
    img_points: List[np.ndarray] = []  # 2D 图像角点集合
    img_size: Optional[Tuple[int, int]] = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for img_path in image_paths:
        if not os.path.exists(img_path):
            continue

        try:
            data = np.fromfile(img_path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception as e:
            log.warning(f"读取图片失败: {img_path}, err: {e}")
            continue

        if img is None:
            continue

        h, w = img.shape[:2]
        if img_size is None:
            img_size = (w, h)
        elif img_size != (w, h):
            log.warning(f"跳过分辨率不匹配图片: {img_path} ({w}x{h} vs 期望 {img_size[0]}x{img_size[1]})")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 检测棋盘格内角点
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

        if ret and corners is not None:
            # 亚像素精化
            refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(refined_corners)

    valid_count = len(obj_points)
    if valid_count < 3:
        return IntrinsicsCalibrationResult(
            success=False,
            valid_images_count=valid_count,
            total_images_count=total_count,
            image_size=img_size or (0, 0),
            message=f"棋盘格角点有效检出张数不足 ({valid_count}/{total_count})，至少需要 3 张，请检查标定板内角点规格或反光遮挡",
        )

    # 2. 运行 OpenCV 相机标定
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
            message=f"标定求解异常: {e}",
        )

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
        message=f"标定成功: 有效图片 {valid_count}/{total_count} 张, 重投影误差 RMSE={ret:.3f} px",
    )


def calibrate_workspace_intrinsics(
    ws: Workspace,
    pattern_size: Tuple[int, int] = (9, 6),
    square_size_mm: float = 25.0,
    save_to_workspace: bool = True,
) -> IntrinsicsCalibrationResult:
    """
    自动从指定工位的内参图集目录 (`ws.intrinsics_raw_images_dir`) 加载图片并标定。
    如果解算成功且 save_to_workspace 为 True，自动将内参写入当前工位沙盒中。

    :param ws: 工位 Workspace 实例
    :param pattern_size: 标定板内角点规格 (列, 行)
    :param square_size_mm: 方格边长 (mm)
    :param save_to_workspace: 是否将求解出的内参持久化到工位
    :return: IntrinsicsCalibrationResult
    """
    raw_dir = ws.intrinsics_raw_images_dir
    if not os.path.exists(raw_dir):
        return IntrinsicsCalibrationResult(
            success=False,
            message=f"工位内参图集目录不存在: {raw_dir}",
        )

    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    images: List[str] = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(raw_dir, ext)))
    images.sort()

    res = calibrate_camera_from_images(
        image_paths=images,
        pattern_size=pattern_size,
        square_size_mm=square_size_mm,
    )

    if res.success and save_to_workspace:
        ok = ws.save_camera_intrinsics(
            fx=res.fx,
            fy=res.fy,
            cx=res.cx,
            cy=res.cy,
            dist_coeffs=res.dist_coeffs,
            image_size=res.image_size,
            rmse=res.rmse,
        )
        if ok:
            log.info(f"已成功将标定内参写入工位沙盒: {ws.workspace_id}")
        else:
            log.error(f"保存内参到工位沙盒失败: {ws.workspace_id}")

    return res
