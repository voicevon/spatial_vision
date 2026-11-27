#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机架固定标靶平面单应性拉正器 (PlanarRectifier)
==============================================
面向固定工位 2D 视觉精密测量场景，解决“每台产线相机位姿不同”与“现场相机被碰歪”问题：
  1. 在测量台面刚性固定 4+ 枚已知物理 CAD 坐标的 AprilTag 16h5 标靶；
  2. 复用 TagDetector 提取亚像素角点，直接求解从原始图像到标准鸟瞰物理 ROI 的单应矩阵 H；
  3. warp_roi() 输出严格等比 (1 / px_per_mm) 的无透视畸变鸟瞰图，消除近大远小误差；
  4. 支持空闲/周期巡检标靶漂移，发现相机被碰歪时自动重算并热更新 H，异常时熔断告警。
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger
from src.vision.tag_detector import TagDetector

log = get_logger(__name__)


@dataclass
class PlanarRigConfig:
    """机架固定标靶与物理 ROI 配置"""
    tag_size_mm: float = 30.0
    fixed_tags_mm: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    roi_mm: Tuple[float, float, float, float] = (20.0, 15.0, 300.0, 105.0)  # (x_min, y_min, x_max, y_max)
    px_per_mm: float = 4.0
    max_reproj_err_px: float = 1.5
    shift_threshold_px: float = 1.0
    use_tag_corners: bool = False  # False=按标靶中心求解(抗粘贴旋转,需>=4靶); True=按4角点求解(需统一朝向,支持3靶抗遮挡)

    def output_size_px(self) -> Tuple[int, int]:
        x_min, y_min, x_max, y_max = self.roi_mm
        w = max(16, int(round((x_max - x_min) * self.px_per_mm)))
        h = max(16, int(round((y_max - y_min) * self.px_per_mm)))
        return w, h


@dataclass
class RectifyStatus:
    """单应性锁定与健康状态快照"""
    locked: bool = False
    visible_tag_ids: List[int] = field(default_factory=list)
    reproj_err_px: float = 0.0
    max_shift_px: float = 0.0
    shifted_recovered: bool = False
    reason: str = "未初始化"
    last_update_ts: float = 0.0


class PlanarRectifier:
    """基于机架固定 AprilTag 的自动透视拉正与抗碰歪自愈引擎"""

    def __init__(self, config: PlanarRigConfig):
        self.config = config
        valid_ids = [int(k) for k in config.fixed_tags_mm.keys()]
        self._detector = TagDetector(valid_tag_ids=valid_ids)
        self._homography: Optional[np.ndarray] = None
        self._cached_centers: Dict[int, np.ndarray] = {}
        self.status = RectifyStatus()

    @property
    def mm_per_px(self) -> float:
        return 1.0 / float(self.config.px_per_mm)

    @property
    def is_locked(self) -> bool:
        return self._homography is not None and self.status.locked

    @property
    def homography(self) -> Optional[np.ndarray]:
        return None if self._homography is None else self._homography.copy()

    def _build_correspondences(
        self, detected: Dict[int, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """构建图像点 -> 鸟瞰 ROI 像素坐标点对应集"""
        x_min, y_min, _, _ = self.config.roi_mm
        scale = float(self.config.px_per_mm)
        half_s = float(self.config.tag_size_mm) / 2.0

        src_pts: List[List[float]] = []
        dst_pts: List[List[float]] = []
        used_ids: List[int] = []

        for tid in sorted(detected.keys()):
            if tid not in self.config.fixed_tags_mm:
                continue
            corners = np.asarray(detected[tid], dtype=np.float64).reshape(4, 2)
            cx_mm, cy_mm = self.config.fixed_tags_mm[tid]
            used_ids.append(tid)

            if self.config.use_tag_corners:
                # 标靶 4 角点顺时针顺序 (左上、右上、右下、左下，Y 轴向下与图像一致)
                local_mm = (
                    (cx_mm - half_s, cy_mm - half_s),
                    (cx_mm + half_s, cy_mm - half_s),
                    (cx_mm + half_s, cy_mm + half_s),
                    (cx_mm - half_s, cy_mm + half_s),
                )
                for pt_img, (px_mm, py_mm) in zip(corners, local_mm):
                    src_pts.append([float(pt_img[0]), float(pt_img[1])])
                    dst_pts.append([(px_mm - x_min) * scale, (py_mm - y_min) * scale])
            else:
                center_img = corners.mean(axis=0)
                src_pts.append([float(center_img[0]), float(center_img[1])])
                dst_pts.append([(cx_mm - x_min) * scale, (cy_mm - y_min) * scale])

        return (
            np.asarray(src_pts, dtype=np.float64),
            np.asarray(dst_pts, dtype=np.float64),
            used_ids,
        )

    def _measure_max_shift(self, detected: Dict[int, np.ndarray]) -> float:
        """计算当前帧标靶中心相对上次锁定位置的像素最大偏移"""
        if not self._cached_centers:
            return 0.0
        shifts = []
        for tid, prev_c in self._cached_centers.items():
            if tid in detected:
                cur_c = np.asarray(detected[tid], dtype=np.float64).reshape(4, 2).mean(axis=0)
                shifts.append(float(np.linalg.norm(cur_c - prev_c)))
        return max(shifts) if shifts else 0.0

    def update(self, image_bgr: np.ndarray, force_recompute: bool = False) -> RectifyStatus:
        """
        检测机架固定标靶并在需要时求解/热更新单应矩阵 H。
        - 首次调用或 force_recompute=True 时强制求解；
        - 已锁定状态下若标靶中心偏移 <= shift_threshold_px，则保留缓存 H 避免微抖；
        - 若标靶偏移 > shift_threshold_px，自动重算 H 并标记 shifted_recovered=True；
        - 若可见标靶不足或重投影误差超限，熔断解锁 (locked=False)。
        """
        now = time.time()
        if image_bgr is None or image_bgr.size == 0:
            self._homography = None
            self.status = RectifyStatus(locked=False, reason="输入图像为空", last_update_ts=now)
            return self.status

        detected = self._detector.detect_tags(image_bgr, fast=False, refine=True)
        src_pts, dst_pts, used_ids = self._build_correspondences(detected)

        min_tags = 3 if self.config.use_tag_corners else 4
        if len(used_ids) < min_tags or len(src_pts) < 4:
            self._homography = None
            self.status = RectifyStatus(
                locked=False,
                visible_tag_ids=used_ids,
                reason=f"可见机架标靶不足 ({len(used_ids)}/{min_tags})",
                last_update_ts=now,
            )
            return self.status

        max_shift = self._measure_max_shift(detected)
        if self.is_locked and not force_recompute and max_shift <= self.config.shift_threshold_px:
            self.status = RectifyStatus(
                locked=True,
                visible_tag_ids=used_ids,
                reproj_err_px=self.status.reproj_err_px,
                max_shift_px=round(max_shift, 3),
                shifted_recovered=False,
                reason="OK",
                last_update_ts=now,
            )
            return self.status

        method = cv2.RANSAC if len(src_pts) > 4 else 0
        H, _ = cv2.findHomography(src_pts, dst_pts, method, 2.0)
        if H is None or not np.all(np.isfinite(H)):
            self._homography = None
            self.status = RectifyStatus(
                locked=False,
                visible_tag_ids=used_ids,
                reason="单应矩阵求解失败",
                last_update_ts=now,
            )
            return self.status

        # 计算重投影残差 (像素)
        proj = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        reproj_err = float(np.mean(np.linalg.norm(proj - dst_pts, axis=1)))
        if reproj_err > self.config.max_reproj_err_px:
            self._homography = None
            self.status = RectifyStatus(
                locked=False,
                visible_tag_ids=used_ids,
                reproj_err_px=round(reproj_err, 3),
                max_shift_px=round(max_shift, 3),
                reason=f"重投影误差超限 ({reproj_err:.2f}px > {self.config.max_reproj_err_px:.2f}px)",
                last_update_ts=now,
            )
            return self.status

        was_locked = self.is_locked
        shifted_recovered = was_locked and (max_shift > self.config.shift_threshold_px)
        self._homography = H
        self._cached_centers = {
            tid: np.asarray(detected[tid], dtype=np.float64).reshape(4, 2).mean(axis=0)
            for tid in used_ids
        }
        if shifted_recovered:
            log.warning(
                f"[PlanarRectifier] 检测到相机位移 ({max_shift:.2f}px)，已自动重算单应矩阵恢复精度 "
                f"(reproj={reproj_err:.3f}px, tags={used_ids})"
            )

        self.status = RectifyStatus(
            locked=True,
            visible_tag_ids=used_ids,
            reproj_err_px=round(reproj_err, 3),
            max_shift_px=round(max_shift, 3),
            shifted_recovered=shifted_recovered,
            reason="CAMERA_SHIFT_AUTO_RECOVERED" if shifted_recovered else "OK",
            last_update_ts=now,
        )
        return self.status

    def warp_roi(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """使用已锁定的单应矩阵输出严格等比鸟瞰 ROI 图像"""
        if not self.is_locked or image_bgr is None or image_bgr.size == 0:
            return None
        out_w, out_h = self.config.output_size_px()
        return cv2.warpPerspective(
            image_bgr,
            self._homography,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
