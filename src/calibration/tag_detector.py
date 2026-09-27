#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一 AprilTag 双路互补检测器 (TagDetector)
==========================================
单一职责：统一 AprilTag 16h5 检测逻辑，提供唯一的双路互补检测实现。

双路互补策略 (彻底解决反光、黑度不纯、倾斜与远景漏检)：
  - 路 1 (高光路): 原图灰度 + 较严二值门限 C=5.5, minOtsu=0.55，
    精准捕获高光、灯光直射区域标靶；
  - 路 2 (暗部路): 动态直方图拉伸 (percentile 2~98) + 宽松门限 C=2.5, minOtsu=0.45，
    攻克发灰、低反差、暗部标靶；
  - 白名单机制硬锁保底，两路结果快速取并集，杜绝误检与漏检。
"""

from typing import Dict, Iterable, Optional

import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)

# 两路检测器的固定超参数组合: (adaptiveThreshConstant, minOtsuStdDev)
_BRIGHT_PARAMS = (5.5, 0.55)   # 抗反光/高亮清晰路
_DARK_PARAMS = (2.5, 0.45)     # 低反差/黑度不纯路

# 基础参数模版其余固定项 (兼顾高精度亚像素角点与大透视/低反差/发灰墨色)
_MAX_MARKER_PERIMETER_RATE = 4.0
_POLYGON_ACCURACY_RATE = 0.09
_PIXEL_PER_CELL = 10
_ERROR_CORRECTION_RATE = 0.50
_IGNORED_MARGIN_PER_CELL = 0.15
_MAX_ERRONEOUS_BITS_RATE = 0.30

# 亚像素精修常量: 窗口 = 平均边长 × _SUBPIX_WIN_RATE (夹紧在 _SUBPIX_WIN_MIN~MAX 像素),
# 任一角点漂移超过 _SUBPIX_DRIFT_MAX_PX 视为异常跳变, 安全回退原角点
_SUBPIX_WIN_RATE = 0.06
_SUBPIX_WIN_MIN = 3
_SUBPIX_WIN_MAX = 9
_SUBPIX_MAX_ITER = 40
_SUBPIX_EPS_PX = 0.001
_SUBPIX_DRIFT_MAX_PX = 2.5


class TagDetector:
    """AprilTag 16h5 双路互补检测器 (高光路 + 暗部动态拉伸路 + 亚像素精修)"""

    def __init__(
        self,
        valid_tag_ids: Optional[Iterable[int]] = None,
        min_perimeter_rate: float = 0.008,
        enable_auto_stretch: bool = True,
    ):
        """
        :param valid_tag_ids: 有效 Tag ID 白名单 (空/None 表示不限制)
        :param min_perimeter_rate: 最小周长占比门限 (内部夹紧下限 0.008)
        :param enable_auto_stretch: fast 模式下是否仍执行暗部动态拉伸路
        """
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
        self.valid_tag_ids = [int(x) for x in valid_tag_ids] if valid_tag_ids else []
        self.min_perimeter_rate = float(min_perimeter_rate)
        self.enable_auto_stretch = bool(enable_auto_stretch)
        self._rebuild()

    def set_params(
        self,
        valid_tag_ids: Optional[Iterable[int]] = None,
        min_perimeter_rate: Optional[float] = None,
        enable_auto_stretch: Optional[bool] = None,
    ) -> None:
        """运行时更新检测参数并重建两路检测器 (仅重建算法对象，与硬件无关)"""
        if valid_tag_ids is not None:
            self.valid_tag_ids = [int(x) for x in valid_tag_ids]
        if min_perimeter_rate is not None:
            self.min_perimeter_rate = float(min_perimeter_rate)
        if enable_auto_stretch is not None:
            self.enable_auto_stretch = bool(enable_auto_stretch)
        self._rebuild()

    def _make_params(self, thresh_c: float, min_otsu: float) -> cv2.aruco.DetectorParameters:
        """构建单路 DetectorParameters (三处历史实现统一后的超参数模版)"""
        p = cv2.aruco.DetectorParameters()
        p.adaptiveThreshWinSizeMin = 3
        p.adaptiveThreshWinSizeMax = 43
        p.adaptiveThreshWinSizeStep = 8
        p.adaptiveThreshConstant = thresh_c
        p.minOtsuStdDev = min_otsu
        p.minMarkerPerimeterRate = max(0.008, self.min_perimeter_rate)
        p.maxMarkerPerimeterRate = _MAX_MARKER_PERIMETER_RATE
        p.polygonalApproxAccuracyRate = _POLYGON_ACCURACY_RATE
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        p.perspectiveRemovePixelPerCell = _PIXEL_PER_CELL
        p.errorCorrectionRate = _ERROR_CORRECTION_RATE
        p.perspectiveRemoveIgnoredMarginPerCell = _IGNORED_MARGIN_PER_CELL
        p.maxErroneousBitsInBorderRate = _MAX_ERRONEOUS_BITS_RATE
        return p

    def _rebuild(self) -> None:
        """构建两路互补 ArucoDetector: 高光路 (路 1) + 暗部动态拉伸路 (路 2)"""
        self.detector_bright = cv2.aruco.ArucoDetector(
            self.dictionary, self._make_params(*_BRIGHT_PARAMS))
        self.detector_dark = cv2.aruco.ArucoDetector(
            self.dictionary, self._make_params(*_DARK_PARAMS))

    def detect_tags(
        self,
        image: np.ndarray,
        fast: bool = False,
        refine: bool = True,
    ) -> Dict[int, np.ndarray]:
        """
        双路互补融合检测标靶角点
        :param image: BGR 或灰度图
        :param fast: 极速模式 (采图向导实时取流用)：高光路检出 >= 2 个白名单标靶
                     且未开启 auto_stretch 时短路返回，消除拖影
        :param refine: 是否对全部角点执行亚像素二次精修
        :return: {tag_id: corners_4x2 (float64)}
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        results: Dict[int, np.ndarray] = {}

        # 路 1: 高光/清晰/反光区域
        c1, ids1, _ = self.detector_bright.detectMarkers(gray)
        if ids1 is not None and len(ids1) > 0:
            for idx, tag_id in enumerate(ids1.flatten()):
                tid = int(tag_id)
                if self.valid_tag_ids and tid not in self.valid_tag_ids:
                    continue
                results[tid] = c1[idx].reshape((4, 2))

        # 快速短路：高光路已满足共视条件且未要求强制拉伸时跳过耗时二次检测
        if fast and len(results) >= 2 and not self.enable_auto_stretch:
            return results

        # 路 2: 暗部/低反差/打印黑度不够纯区域 (动态拉伸 + 宽松门限)
        p_low, p_high = np.percentile(gray[::4, ::4], (2, 98))
        if p_high > p_low + 10:
            gray_stretch = np.clip(
                (gray.astype(np.float32) - p_low) * (255.0 / (p_high - p_low)),
                0, 255).astype(np.uint8)
        else:
            gray_stretch = gray

        c2, ids2, _ = self.detector_dark.detectMarkers(gray_stretch)
        if ids2 is not None and len(ids2) > 0:
            for idx, tag_id in enumerate(ids2.flatten()):
                tid = int(tag_id)
                if self.valid_tag_ids and tid not in self.valid_tag_ids:
                    continue
                if tid not in results:
                    results[tid] = c2[idx].reshape((4, 2))

        if refine:
            for tid in list(results.keys()):
                results[tid] = self.refine_corners_subpix(gray, results[tid])

        return results

    def refine_corners_subpix(self, gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        基于梯度自相关矩阵对 AprilTag 4 个角点进行高阶亚像素二次精修：
        - 窗口大小基于标靶平均边长自适应计算 (标靶尺度的 6%，夹紧在 3~9 像素之间)
        - 零区域 (-1, -1) 规避自相关矩阵退化
        - 严格迭代终止准则：40 次迭代或精度达到 0.001 像素
        - 包含异常漂移防呆：若精修漂移超过 2.5 像素，安全回退到原角点
        """
        try:
            pts = corners.reshape((4, 2)).astype(np.float32)
            avg_side = (np.linalg.norm(pts[0] - pts[1])
                        + np.linalg.norm(pts[1] - pts[2])) / 2.0
            half_win = int(np.clip(avg_side * _SUBPIX_WIN_RATE, _SUBPIX_WIN_MIN, _SUBPIX_WIN_MAX))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        _SUBPIX_MAX_ITER, _SUBPIX_EPS_PX)
            refined = cv2.cornerSubPix(gray, pts.copy(), (half_win, half_win), (-1, -1), criteria)
            if np.max(np.linalg.norm(refined - pts, axis=1)) > _SUBPIX_DRIFT_MAX_PX:
                # 局部边缘或反光导致非正常跳变，安全保底
                return pts.astype(np.float64)
            return refined.astype(np.float64)
        except Exception as e:
            log.warning(f"[TagDetector] 亚像素精修失败 (已回退原角点): {e}")
            return corners.astype(np.float64)
