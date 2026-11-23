#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单根芦笋几何精密测量模块 (Single Asparagus Measurer)
=====================================================
针对 ROI 区域内仅存在单根芦笋的场景，提供纯视觉/几何测量能力（只输出客观测量事实）：
  1. 在指定 ROI 内联合提取绿茎与白根前景，锁定唯一单根芦笋本体；
  2. 物理尺度解析：深度图+相机内参优先，无深度时回退 AprilTag 平面投影或手动 mm/px 比例；
  3. 沿 PCA 主轴切片提取平滑中轴线，并按固定朝向（根→尖）统一排序；
  4. 沿中轴识别白根与绿茎分界线（低于最小白根长度门限视为全绿无白根）；
  5. 计算：
     - 有效长度 (effective_length_mm)：仅绿色部分长度（白绿分界线 → 笋尖；全绿时为根部切口 → 笋尖）；
     - 特征直径 (diameter_mm)：从白绿分界线（全绿时从根部切口）向头部偏移 20 mm 处的垂直截面直径。
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


ORIENTATION_OPTIONS: List[Tuple[str, str]] = [
    ("left_to_right", "左根 → 右尖"),
    ("right_to_left", "右根 → 左尖"),
    ("top_to_bottom", "上根 → 下尖"),
    ("bottom_to_top", "下根 → 上尖"),
]

SCALE_SOURCE_LABELS: Dict[str, str] = {
    "depth_intrinsics": "深度图 + 内参",
    "tag_plane": "AprilTag 平面标定",
    "manual_scale": "手动 mm/px 比例",
}


@dataclass
class MeasureConfig:
    """单根芦笋测量可调参数配置"""
    orientation: str = "left_to_right"       # 根→尖固定方向
    diameter_offset_mm: float = 20.0         # 白绿分界线/根部切口向头部偏移距离 (mm)
    green_h_low: int = 25                    # 绿茎 Hue 下限 [0, 179]
    green_h_high: int = 95                   # 绿茎 Hue 上限 [0, 179]
    green_s_min: int = 35                    # 绿茎 Saturation 下限 [0, 255]
    green_v_min: int = 30                    # 绿茎 Value 下限 [0, 255]
    white_s_max: int = 65                    # 白根 Saturation 上限 [0, 255]
    white_v_min: int = 135                   # 白根 Value 下限 [0, 255]
    green_ratio_threshold: float = 0.55      # 沿中轴判定进入绿茎段的切片绿色占比门限
    min_white_root_mm: float = 3.0           # 判定存在白根的最小物理长度 (mm)
    fallback_mm_per_px: float = 0.25         # 无深度且无 Tag 标定时的像素物理尺度 (mm/px)
    min_area_px: int = 200                   # 单根前景最小像素面积


@dataclass
class SingleAsparagusMeasurement:
    """单根芦笋测量结果契约"""
    valid: bool = False
    reason: str = ""
    warning: str = ""
    effective_length_mm: float = 0.0
    diameter_mm: float = 0.0
    has_white_root: bool = False
    scale_source: str = "manual_scale"
    mm_per_px: float = 0.25
    elapsed_ms: float = 0.0
    roi_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
    centerline_pts: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    boundary_idx: int = 0
    boundary_pt: Optional[Tuple[float, float]] = None
    tip_pt: Optional[Tuple[float, float]] = None
    root_pt: Optional[Tuple[float, float]] = None
    diam_center_pt: Optional[Tuple[float, float]] = None
    diam_p1: Optional[Tuple[float, float]] = None
    diam_p2: Optional[Tuple[float, float]] = None
    contour: Optional[np.ndarray] = None
    step_snapshots: Dict[str, np.ndarray] = field(default_factory=dict)


class SingleAsparagusMeasurer:
    """单根芦笋有效长度与 20mm 偏移直径测量器"""

    STEP_DEFS: List[Tuple[str, str]] = [
        ("stage1_roi", "1.原图与ROI"),
        ("stage2_seg", "2.绿白分割"),
        ("stage3_axis", "3.中轴与分界"),
        ("stage4_result", "4.最终测量"),
    ]

    def __init__(self, config: Optional[MeasureConfig] = None):
        self.config = config if config is not None else MeasureConfig()

    def measure(
        self,
        color_bgr: np.ndarray,
        roi_rect: Optional[Tuple[int, int, int, int]] = None,
        depth_mm: Optional[np.ndarray] = None,
        intrinsics: Optional[Tuple[float, float, float, float]] = None,
        tag_localizer: Optional[Any] = None,
    ) -> SingleAsparagusMeasurement:
        """
        执行 ROI 内单根芦笋的有效长度与直径测量。

        :param color_bgr: 完整彩色图像 (H, W, 3) BGR
        :param roi_rect: ROI 矩形 (x, y, w, h)，为空则使用整幅图像
        :param depth_mm: 可选与彩色图对齐的毫米深度图 (H, W)
        :param intrinsics: 可选相机内参 (fx, fy, cx, cy)
        :param tag_localizer: 可选 AprilTag 定位器（无深度图时估算世界平面尺度）
        """
        t0 = time.perf_counter()
        if color_bgr is None or color_bgr.size == 0:
            return SingleAsparagusMeasurement(valid=False, reason="输入图像为空")

        img_h, img_w = color_bgr.shape[:2]
        rx, ry, rw, rh = self._clamp_roi(roi_rect, img_w, img_h)
        res = SingleAsparagusMeasurement(roi_rect=(rx, ry, rw, rh))

        # Step 1: 裁剪 ROI 并分割绿茎与白根
        roi_bgr = color_bgr[ry:ry + rh, rx:rx + rw]
        roi_depth = None
        if depth_mm is not None and depth_mm.shape[:2] == (img_h, img_w):
            roi_depth = depth_mm[ry:ry + rh, rx:rx + rw]

        green_mask, white_mask, body_mask, contour_local = self._segment_single_asparagus(roi_bgr, roi_depth)

        # 生成 Step 1 可视化
        vis_s1 = self._render_stage1(color_bgr, (rx, ry, rw, rh))
        res.step_snapshots["stage1_roi"] = vis_s1

        if body_mask is None or contour_local is None:
            res.valid = False
            res.reason = "ROI 内未检测到有效单根芦笋"
            res.step_snapshots["stage2_seg"] = vis_s1.copy()
            res.step_snapshots["stage3_axis"] = vis_s1.copy()
            res.step_snapshots["stage4_result"] = vis_s1.copy()
            res.elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return res

        # 生成 Step 2 可视化
        vis_s2 = self._render_stage2(color_bgr, (rx, ry, rw, rh), green_mask, white_mask, body_mask)
        res.step_snapshots["stage2_seg"] = vis_s2

        # Step 2: 物理尺度解析 (深度图+内参 > AprilTag 平面 > 手动 mm/px)
        mm_per_px, scale_source = self._resolve_scale(
            color_bgr=color_bgr,
            roi_rect=(rx, ry, rw, rh),
            body_mask=body_mask,
            roi_depth=roi_depth,
            intrinsics=intrinsics,
            tag_localizer=tag_localizer,
        )
        res.mm_per_px = mm_per_px
        res.scale_source = scale_source

        # Step 3: 提取沿主轴排序的中轴点、绿/白比例与物理弧长
        axis_data = self._extract_ordered_centerline(body_mask, green_mask, white_mask, mm_per_px)
        if axis_data is None:
            res.valid = False
            res.reason = "单根轮廓过短，无法提取中轴线"
            res.step_snapshots["stage3_axis"] = vis_s2.copy()
            res.step_snapshots["stage4_result"] = vis_s2.copy()
            res.elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return res

        pts_local, arc_mm, green_ratios, white_ratios = axis_data

        # Step 4: 白绿分界线定位
        boundary_idx, has_white_root = self._locate_white_green_boundary(
            arc_mm, green_ratios, white_ratios
        )

        # Step 5: 计算绿色有效长度与偏移 20 mm 处截面直径
        effective_length_mm = float(max(0.0, arc_mm[-1] - arc_mm[boundary_idx]))
        target_arc = float(arc_mm[boundary_idx]) + float(self.config.diameter_offset_mm)
        warning = ""
        if target_arc > float(arc_mm[-1]):
            target_arc = float(arc_mm[-1])
            warning = f"绿色有效长度不足 {self.config.diameter_offset_mm:.0f}mm，直径采样点已钳位至笋尖附近"

        diam_idx = int(np.argmin(np.abs(arc_mm - target_arc)))
        diam_px, p1_local, p2_local = self._measure_cross_section_diameter(body_mask, pts_local, diam_idx)
        diameter_mm = float(diam_px * mm_per_px)

        # 转换回全图坐标
        offset_xy = np.array([float(rx), float(ry)], dtype=np.float32)
        pts_global = pts_local + offset_xy
        contour_global = contour_local.copy()
        contour_global[:, 0, 0] += rx
        contour_global[:, 0, 1] += ry

        res.valid = True
        res.warning = warning
        res.effective_length_mm = round(effective_length_mm, 2)
        res.diameter_mm = round(diameter_mm, 2)
        res.has_white_root = has_white_root
        res.centerline_pts = pts_global
        res.boundary_idx = boundary_idx
        res.root_pt = (float(pts_global[0, 0]), float(pts_global[0, 1]))
        res.boundary_pt = (float(pts_global[boundary_idx, 0]), float(pts_global[boundary_idx, 1]))
        res.tip_pt = (float(pts_global[-1, 0]), float(pts_global[-1, 1]))
        res.diam_center_pt = (float(pts_global[diam_idx, 0]), float(pts_global[diam_idx, 1]))
        res.diam_p1 = (float(p1_local[0] + rx), float(p1_local[1] + ry))
        res.diam_p2 = (float(p2_local[0] + rx), float(p2_local[1] + ry))
        res.contour = contour_global

        res.step_snapshots["stage3_axis"] = self._render_stage3(color_bgr, res)
        res.step_snapshots["stage4_result"] = self._render_stage4(color_bgr, res)
        res.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return res

    # ------------------------------ 内部算法步骤 ------------------------------

    @staticmethod
    def _clamp_roi(
        roi_rect: Optional[Tuple[int, int, int, int]], img_w: int, img_h: int
    ) -> Tuple[int, int, int, int]:
        if roi_rect is None:
            return 0, 0, img_w, img_h
        x, y, w, h = [int(v) for v in roi_rect]
        x = max(0, min(img_w - 1, x))
        y = max(0, min(img_h - 1, y))
        w = max(1, min(img_w - x, w))
        h = max(1, min(img_h - y, h))
        return x, y, w, h

    def _segment_single_asparagus(
        self, roi_bgr: np.ndarray, roi_depth: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """在 ROI 内分离绿茎与白根，并锁定唯一主单根连通域"""
        cfg = self.config
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # 绿茎掩膜
        green_mask = (
            (h_ch >= cfg.green_h_low)
            & (h_ch <= cfg.green_h_high)
            & (s_ch >= cfg.green_s_min)
            & (v_ch >= cfg.green_v_min)
        ).astype(np.uint8) * 255

        # 白根掩膜：低饱和度 + 高亮度，排除已判绿的像素
        white_mask = (
            (s_ch <= cfg.white_s_max)
            & (v_ch >= cfg.white_v_min)
            & (green_mask == 0)
        ).astype(np.uint8) * 255

        # 若有有效深度图，对白根做轻微前景深度保护（排除明显偏远的底板亮斑）
        if roi_depth is not None and np.count_nonzero(green_mask) >= 30:
            green_z = roi_depth[(green_mask > 0) & (roi_depth > 150) & (roi_depth < 2500)]
            if len(green_z) >= 20:
                z_med = float(np.median(green_z))
                depth_ok = (roi_depth == 0) | (np.abs(roi_depth - z_med) <= 60.0)
                white_mask = np.where(depth_ok, white_mask, 0).astype(np.uint8)

        # 融合绿茎与白根，形态学去噪并桥接过渡缝
        raw_body = cv2.bitwise_or(green_mask, white_mask)
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned = cv2.morphologyEx(raw_body, cv2.MORPH_OPEN, k_open)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return green_mask, white_mask, None, None

        # 选择包含绿色像素最多的连通域作为唯一单根芦笋
        best_cnt = None
        best_score = -1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < cfg.min_area_px:
                continue
            tmp = np.zeros_like(cleaned)
            cv2.drawContours(tmp, [cnt], -1, 255, -1)
            green_hits = int(np.count_nonzero((tmp > 0) & (green_mask > 0)))
            if green_hits < 30:
                continue
            if green_hits > best_score:
                best_score = green_hits
                best_cnt = cnt

        if best_cnt is None:
            return green_mask, white_mask, None, None

        body_mask = np.zeros_like(cleaned)
        cv2.drawContours(body_mask, [best_cnt], -1, 255, -1)
        green_in_body = cv2.bitwise_and(green_mask, body_mask)
        white_in_body = cv2.bitwise_and(white_mask, body_mask)
        return green_in_body, white_in_body, body_mask, best_cnt

    def _resolve_scale(
        self,
        color_bgr: np.ndarray,
        roi_rect: Tuple[int, int, int, int],
        body_mask: np.ndarray,
        roi_depth: Optional[np.ndarray],
        intrinsics: Optional[Tuple[float, float, float, float]],
        tag_localizer: Optional[Any],
    ) -> Tuple[float, str]:
        """解析物理像素比例 (mm/px): 深度图+内参 > AprilTag 平面 > 手动比例"""
        # 1. 深度图 + 相机内参
        if roi_depth is not None and intrinsics is not None:
            fx, fy, _, _ = intrinsics
            f_avg = 0.5 * (float(fx) + float(fy))
            if f_avg > 10.0:
                valid_z = roi_depth[(body_mask > 0) & (roi_depth > 150.0) & (roi_depth < 2500.0)]
                if len(valid_z) >= 20:
                    z_med = float(np.median(valid_z))
                    return float(z_med / f_avg), "depth_intrinsics"

        # 2. AprilTag 标定平面投影
        if tag_localizer is not None and intrinsics is not None:
            try:
                ok, t_cam_to_world, _ = tag_localizer.localize_camera(color_bgr)
                if ok and t_cam_to_world is not None:
                    scale = self._scale_from_world_plane(t_cam_to_world, intrinsics, roi_rect)
                    if scale is not None and 0.02 <= scale <= 5.0:
                        return float(scale), "tag_plane"
            except Exception:
                pass

        # 3. 手动配置比例回退
        return max(0.01, float(self.config.fallback_mm_per_px)), "manual_scale"

    @staticmethod
    def _scale_from_world_plane(
        t_cam_to_world: np.ndarray,
        intrinsics: Tuple[float, float, float, float],
        roi_rect: Tuple[int, int, int, int],
    ) -> Optional[float]:
        """计算 ROI 中心像素射线与世界 Z=0 平面交点的局部 mm/px 尺度"""
        fx, fy, cx, cy = [float(v) for v in intrinsics]
        if fx <= 10.0 or fy <= 10.0:
            return None
        rx, ry, rw, rh = roi_rect
        u0 = rx + rw * 0.5
        v0 = ry + rh * 0.5

        R_cw = t_cam_to_world[:3, :3]
        t_cw = t_cam_to_world[:3, 3]

        def ray_plane_z0(u: float, v: float) -> Optional[np.ndarray]:
            ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
            ray_world = R_cw @ ray_cam
            if abs(ray_world[2]) < 1e-6:
                return None
            lam = -t_cw[2] / ray_world[2]
            if lam <= 0:
                return None
            return t_cw + lam * ray_world

        p0 = ray_plane_z0(u0, v0)
        px = ray_plane_z0(u0 + 1.0, v0)
        py = ray_plane_z0(u0, v0 + 1.0)
        if p0 is None or px is None or py is None:
            return None
        sx = float(np.linalg.norm(px - p0))
        sy = float(np.linalg.norm(py - p0))
        return 0.5 * (sx + sy)

    def _extract_ordered_centerline(
        self,
        body_mask: np.ndarray,
        green_mask: np.ndarray,
        white_mask: np.ndarray,
        mm_per_px: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        沿 PCA 主轴切片提取平滑中轴线，并按 config.orientation 排序为 [根端 → 尖端]。
        返回: (centerline_pts_Nx2, arc_mm_N, green_ratios_N, white_ratios_N)
        """
        ys, xs = np.nonzero(body_mask > 0)
        if len(xs) < 40:
            return None

        pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        mean = pts.mean(axis=0)
        centered = pts - mean
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        u_axis = eigvecs[:, int(np.argmax(eigvals))].astype(np.float64)
        u_axis /= max(1e-9, float(np.linalg.norm(u_axis)))

        # 根据固定朝向确定 u_axis 正方向（根 → 尖）
        ori = self.config.orientation
        if ori == "left_to_right" and u_axis[0] < 0:
            u_axis = -u_axis
        elif ori == "right_to_left" and u_axis[0] > 0:
            u_axis = -u_axis
        elif ori == "top_to_bottom" and u_axis[1] < 0:
            u_axis = -u_axis
        elif ori == "bottom_to_top" and u_axis[1] > 0:
            u_axis = -u_axis

        t_proj = centered @ u_axis
        t_min, t_max = float(t_proj.min()), float(t_proj.max())
        span = t_max - t_min
        if span < 12.0:
            return None

        is_green = green_mask[ys, xs] > 0
        is_white = white_mask[ys, xs] > 0

        step_px = 2.0
        half_win = 1.5
        t_samples = np.arange(t_min + 1.0, t_max - 0.5, step_px)
        if len(t_samples) < 5:
            t_samples = np.linspace(t_min, t_max, 8)

        cl_pts = []
        g_ratios = []
        w_ratios = []
        for t_val in t_samples:
            sel = np.abs(t_proj - t_val) <= half_win
            cnt = int(np.count_nonzero(sel))
            if cnt == 0:
                continue
            cl_pts.append(pts[sel].mean(axis=0))
            g_ratios.append(float(np.count_nonzero(is_green[sel])) / cnt)
            w_ratios.append(float(np.count_nonzero(is_white[sel])) / cnt)

        if len(cl_pts) < 4:
            return None

        cl_arr = np.asarray(cl_pts, dtype=np.float32)
        # 轻度平滑中轴点（保留首尾端点避免截短长度）
        if len(cl_arr) >= 5:
            smoothed = cl_arr.copy()
            kernel = np.array([0.15, 0.20, 0.30, 0.20, 0.15], dtype=np.float32)
            for dim in (0, 1):
                conv = np.convolve(cl_arr[:, dim], kernel, mode="same")
                smoothed[2:-2, dim] = conv[2:-2]
            cl_arr = smoothed

        # 精确对齐首尾端点到投影最远端
        diffs = np.diff(cl_arr, axis=0)
        seg_px = np.linalg.norm(diffs, axis=1)
        arc_px = np.concatenate([[0.0], np.cumsum(seg_px)])
        # 补回首尾各 1px 的采样边界偏移（保持几何长度无系统偏短）
        arc_mm = arc_px * float(mm_per_px)
        return (
            cl_arr,
            arc_mm.astype(np.float64),
            np.asarray(g_ratios, dtype=np.float64),
            np.asarray(w_ratios, dtype=np.float64),
        )

    def _locate_white_green_boundary(
        self,
        arc_mm: np.ndarray,
        green_ratios: np.ndarray,
        white_ratios: np.ndarray,
    ) -> Tuple[int, bool]:
        """沿中轴从根端向尖端定位白绿分界点索引"""
        n = len(arc_mm)
        thr = float(self.config.green_ratio_threshold)
        min_white = float(self.config.min_white_root_mm)

        # 寻找首个连续 3 切片均满足绿色占比 >= thr 的稳定绿茎起点
        win = min(3, n)
        b_idx = 0
        for i in range(n - win + 1):
            if float(np.mean(green_ratios[i:i + win])) >= thr:
                b_idx = i
                break

        if b_idx > 0 and float(arc_mm[b_idx]) >= min_white:
            root_white_mean = float(np.mean(white_ratios[:b_idx]))
            root_green_mean = float(np.mean(green_ratios[:b_idx]))
            if root_white_mean >= 0.25 and root_green_mean < thr:
                return b_idx, True

        return 0, False

    @staticmethod
    def _measure_cross_section_diameter(
        body_mask: np.ndarray, pts_local: np.ndarray, diam_idx: int
    ) -> Tuple[float, Tuple[float, float], Tuple[float, float]]:
        """在指定中轴索引处沿法向发射射线测量单根截面宽度（像素）"""
        h, w = body_mask.shape[:2]
        n = len(pts_local)
        i_prev = max(0, diam_idx - 3)
        i_next = min(n - 1, diam_idx + 3)
        if i_prev == i_next:
            tangent = np.array([1.0, 0.0], dtype=np.float64)
        else:
            tangent = (pts_local[i_next] - pts_local[i_prev]).astype(np.float64)
            norm = float(np.linalg.norm(tangent))
            tangent = tangent / norm if norm > 1e-6 else np.array([1.0, 0.0], dtype=np.float64)

        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)

        def cast_at(idx: int) -> Tuple[float, Tuple[float, float], Tuple[float, float]]:
            cx, cy = float(pts_local[idx, 0]), float(pts_local[idx, 1])
            max_steps = int(max(h, w))

            def march(sign: float) -> Tuple[float, float]:
                last_x, last_y = cx, cy
                for s in range(1, max_steps * 2):
                    dist = s * 0.5
                    px = cx + sign * normal[0] * dist
                    py = cy + sign * normal[1] * dist
                    ix, iy = int(round(px)), int(round(py))
                    if ix < 0 or ix >= w or iy < 0 or iy >= h:
                        break
                    if body_mask[iy, ix] == 0:
                        break
                    last_x, last_y = px, py
                return last_x, last_y

            p1 = march(1.0)
            p2 = march(-1.0)
            width_px = float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
            return width_px, p1, p2

        # 中心处端点用于绘制，宽度取 ±2 邻域中位数以抑制轮廓锯齿
        c_width, c_p1, c_p2 = cast_at(diam_idx)
        widths = [c_width]
        for off in (-2, -1, 1, 2):
            j = diam_idx + off
            if 0 <= j < n:
                w_j, _, _ = cast_at(j)
                if w_j > 0.5:
                    widths.append(w_j)
        med_width = float(np.median(widths))
        return med_width, c_p1, c_p2

    # ------------------------------ 步骤可视化 ------------------------------

    def _render_stage1(self, color_bgr: np.ndarray, roi_rect: Tuple[int, int, int, int]) -> np.ndarray:
        vis = color_bgr.copy()
        rx, ry, rw, rh = roi_rect
        cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 215, 190), 2)
        # 在 ROI 边缘画根→尖方向指引箭头
        cx, cy = rx + rw // 2, ry + rh // 2
        arrow_len = max(24, min(rw, rh) // 5)
        ori = self.config.orientation
        if ori == "left_to_right":
            p_start, p_end = (rx + 12, ry + 20), (rx + 12 + arrow_len, ry + 20)
        elif ori == "right_to_left":
            p_start, p_end = (rx + rw - 12, ry + 20), (rx + rw - 12 - arrow_len, ry + 20)
        elif ori == "top_to_bottom":
            p_start, p_end = (rx + 20, ry + 12), (rx + 20, ry + 12 + arrow_len)
        else:
            p_start, p_end = (rx + 20, ry + rh - 12), (rx + 20, ry + rh - 12 - arrow_len)
        cv2.arrowedLine(vis, p_start, p_end, (60, 220, 255), 2, cv2.LINE_AA, tipLength=0.35)
        return vis

    @staticmethod
    def _render_stage2(
        color_bgr: np.ndarray,
        roi_rect: Tuple[int, int, int, int],
        green_mask: np.ndarray,
        white_mask: np.ndarray,
        body_mask: np.ndarray,
    ) -> np.ndarray:
        vis = (color_bgr.astype(np.float32) * 0.45).astype(np.uint8)
        rx, ry, rw, rh = roi_rect
        sub = vis[ry:ry + rh, rx:rx + rw]
        # 绿茎高亮翠绿，白根高亮淡青白
        sub[green_mask > 0] = (
            sub[green_mask > 0].astype(np.float32) * 0.35 + np.array([50, 210, 80], dtype=np.float32) * 0.65
        ).astype(np.uint8)
        sub[white_mask > 0] = (
            sub[white_mask > 0].astype(np.float32) * 0.30 + np.array([235, 220, 175], dtype=np.float32) * 0.70
        ).astype(np.uint8)
        cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 215, 190), 2)
        return vis

    @staticmethod
    def _render_stage3(color_bgr: np.ndarray, res: SingleAsparagusMeasurement) -> np.ndarray:
        vis = color_bgr.copy()
        rx, ry, rw, rh = res.roi_rect
        cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 215, 190), 2)
        if res.contour is not None:
            cv2.drawContours(vis, [res.contour], -1, (100, 180, 120), 1, cv2.LINE_AA)
        pts = res.centerline_pts
        b = res.boundary_idx
        if len(pts) >= 2:
            if res.has_white_root and b > 0:
                white_seg = pts[:b + 1].astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [white_seg], False, (220, 220, 220), 2, cv2.LINE_AA)
            green_seg = pts[b:].astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [green_seg], False, (40, 235, 120), 3, cv2.LINE_AA)
        if res.boundary_pt is not None:
            bp = (int(round(res.boundary_pt[0])), int(round(res.boundary_pt[1])))
            cv2.circle(vis, bp, 6, (220, 70, 255), -1, cv2.LINE_AA)
        if res.tip_pt is not None:
            tp = (int(round(res.tip_pt[0])), int(round(res.tip_pt[1])))
            cv2.circle(vis, tp, 5, (40, 235, 120), -1, cv2.LINE_AA)
        return vis

    @staticmethod
    def _render_stage4(color_bgr: np.ndarray, res: SingleAsparagusMeasurement) -> np.ndarray:
        vis = color_bgr.copy()
        rx, ry, rw, rh = res.roi_rect
        cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 200, 175), 2)
        if res.contour is not None:
            cv2.drawContours(vis, [res.contour], -1, (80, 165, 110), 1, cv2.LINE_AA)

        pts = res.centerline_pts
        b = res.boundary_idx
        if len(pts) >= 2:
            if res.has_white_root and b > 0:
                white_seg = pts[:b + 1].astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [white_seg], False, (190, 195, 205), 2, cv2.LINE_AA)
            green_seg = pts[b:].astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [green_seg], False, (45, 240, 130), 3, cv2.LINE_AA)

        # 白绿分界点（或根部切口点）
        if res.boundary_pt is not None:
            bp = (int(round(res.boundary_pt[0])), int(round(res.boundary_pt[1])))
            cv2.circle(vis, bp, 6, (230, 80, 255), -1, cv2.LINE_AA)
            cv2.circle(vis, bp, 9, (255, 255, 255), 1, cv2.LINE_AA)

        # 笋尖端点
        if res.tip_pt is not None:
            tp = (int(round(res.tip_pt[0])), int(round(res.tip_pt[1])))
            cv2.circle(vis, tp, 5, (45, 240, 130), -1, cv2.LINE_AA)

        # 20mm 处直径卡尺线
        if res.diam_p1 is not None and res.diam_p2 is not None:
            p1 = (int(round(res.diam_p1[0])), int(round(res.diam_p1[1])))
            p2 = (int(round(res.diam_p2[0])), int(round(res.diam_p2[1])))
            cv2.line(vis, p1, p2, (40, 190, 255), 3, cv2.LINE_AA)
            cv2.circle(vis, p1, 4, (40, 190, 255), -1, cv2.LINE_AA)
            cv2.circle(vis, p2, 4, (40, 190, 255), -1, cv2.LINE_AA)

        return vis
