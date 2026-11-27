# -*- coding: utf-8 -*-
"""
3D 透视引导式棋盘格内参采图几何引擎 (ChessboardGuide)
====================================================
移植并适配自 flux_android/CalibrationGuideGeometry.kt：
- 提供覆盖全像面（中央、四角、边缘）、多尺度（近/中/远）与丰富 3D 倾角（俯仰/偏航/滚转/复合）的 24 个标准引导姿态
- 将 3D 棋盘格目标轮廓与内部透视网格投影至归一化 [0, 1] 视口坐标，兼容横屏 16:9 / 4:3 安全边界
- 对实时检测到的棋盘格内角点执行旋转/反转不变的对准度评估与中文方向引导
- 内置相邻帧角点位移防运动模糊稳定判定，确保自动抓拍画面清晰
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


VIRTUAL_FOCAL = 1.9
VIRTUAL_Z0 = 2.6


@dataclass(frozen=True)
class GuidePose:
    """单个 3D 引导目标姿态定义（归一化视口坐标）"""
    index: int
    title: str
    instruction: str
    tilt_hint: str
    center_x: float
    center_y: float
    scale: float
    pitch_deg: float
    yaw_deg: float
    roll_deg: float


@dataclass
class ProjectedGuideMesh:
    """投影至归一化 [0, 1] 视口坐标的目标引导网格"""
    outer_quad: List[Tuple[float, float]]                    # [TL, TR, BR, BL]
    corner_depths: List[float]                               # 各角点相对深度 z (越小越近)
    grid_lines: List[Tuple[Tuple[float, float], Tuple[float, float]]]


@dataclass
class PoseAlignmentFeedback:
    """当前帧棋盘格与目标姿态的实时对准与稳定反馈"""
    detected: bool
    aligned: bool
    stable: bool
    match_score: float
    guidance_text: str
    detected_quad_norm: List[Tuple[float, float]] = field(default_factory=list)
    detected_corners_norm: List[Tuple[float, float]] = field(default_factory=list)


def default_guide_poses() -> List[GuidePose]:
    """24 姿态标准引导队列（5 阶段覆盖全画幅与 3D 倾角）"""
    return [
        # --- Phase 1: 平正全画幅覆盖 (5 姿态) ---
        GuidePose(0, "中央正面（中距）", "将标定板水平正对画面中央", "平正无倾角", 0.50, 0.50, 0.50, 0.0, 0.0, 0.0),
        GuidePose(1, "左上角正面", "将标定板移至画面左上区域", "平正无倾角", 0.30, 0.32, 0.40, 0.0, 0.0, 0.0),
        GuidePose(2, "右上角正面", "将标定板移至画面右上区域", "平正无倾角", 0.70, 0.32, 0.40, 0.0, 0.0, 0.0),
        GuidePose(3, "右下角正面", "将标定板移至画面右下区域", "平正无倾角", 0.70, 0.68, 0.40, 0.0, 0.0, 0.0),
        GuidePose(4, "左下角正面", "将标定板移至画面左下区域", "平正无倾角", 0.30, 0.68, 0.40, 0.0, 0.0, 0.0),

        # --- Phase 2: 尺度与边缘覆盖 (4 姿态) ---
        GuidePose(5, "中央近景（放大）", "靠近标定板使其占满中央大部分区域", "平正近距", 0.50, 0.50, 0.66, 0.0, 0.0, 0.0),
        GuidePose(6, "中央远景（缩小）", "拉远距离使标定板位于中央偏小范围", "平正远距", 0.50, 0.50, 0.32, 0.0, 0.0, 0.0),
        GuidePose(7, "顶部中央", "将标定板移至画面上方中央", "轻微前倾 12°", 0.50, 0.28, 0.42, 12.0, 0.0, 0.0),
        GuidePose(8, "底部中央", "将标定板移至画面下方中央", "轻微后仰 12°", 0.50, 0.72, 0.42, -12.0, 0.0, 0.0),

        # --- Phase 3: 纯俯仰与偏航倾角 (6 姿态) ---
        GuidePose(9, "顶部远离（前倾）", "标定板顶部向远离相机方向倾斜", "俯仰前倾 +28°", 0.50, 0.48, 0.50, 28.0, 0.0, 0.0),
        GuidePose(10, "底部远离（后仰）", "标定板底部向远离相机方向倾斜", "俯仰后仰 -28°", 0.50, 0.52, 0.50, -28.0, 0.0, 0.0),
        GuidePose(11, "右侧远离（右偏）", "标定板右侧向远离相机方向转动", "水平右偏 +30°", 0.50, 0.50, 0.50, 0.0, 30.0, 0.0),
        GuidePose(12, "左侧远离（左偏）", "标定板左侧向远离相机方向转动", "水平左偏 -30°", 0.50, 0.50, 0.50, 0.0, -30.0, 0.0),
        GuidePose(13, "左侧边缘 + 右偏", "移至画面左侧并让右侧远离相机", "偏航 +24°", 0.30, 0.50, 0.44, 0.0, 24.0, 0.0),
        GuidePose(14, "右侧边缘 + 左偏", "移至画面右侧并让左侧远离相机", "偏航 -24°", 0.70, 0.50, 0.44, 0.0, -24.0, 0.0),

        # --- Phase 4: 四角 + 复合 3D 倾角 (6 姿态) ---
        GuidePose(15, "左上角 + 复合倾角", "移至左上角，顶部和右侧略微远离", "前倾 +20° / 右偏 +20°", 0.32, 0.34, 0.40, 20.0, 20.0, -8.0),
        GuidePose(16, "右上角 + 复合倾角", "移至右上角，顶部和左侧略微远离", "前倾 +20° / 左偏 -20°", 0.68, 0.34, 0.40, 20.0, -20.0, 8.0),
        GuidePose(17, "右下角 + 复合倾角", "移至右下角，底部和左侧略微远离", "后仰 -20° / 左偏 -20°", 0.68, 0.66, 0.40, -20.0, -20.0, -8.0),
        GuidePose(18, "左下角 + 复合倾角", "移至左下角，底部和右侧略微远离", "后仰 -20° / 右偏 +20°", 0.32, 0.66, 0.40, -20.0, 20.0, 8.0),
        GuidePose(19, "顺时针滚转 + 前倾", "顺时针旋转约 20° 同时顶部远离", "滚转 +20° / 前倾 +22°", 0.50, 0.50, 0.46, 22.0, 0.0, 20.0),
        GuidePose(20, "逆时针滚转 + 后仰", "逆时针旋转约 20° 同时底部远离", "滚转 -20° / 后仰 -22°", 0.50, 0.50, 0.46, -22.0, 0.0, -20.0),

        # --- Phase 5: 大角度透视与边缘强化 (3 姿态) ---
        GuidePose(21, "顶部 + 右偏大倾角", "移至上方并明显向右偏转", "前倾 +18° / 右偏 +28°", 0.48, 0.32, 0.42, 18.0, 28.0, 0.0),
        GuidePose(22, "底部 + 左偏大倾角", "移至下方并明显向左偏转", "后仰 -18° / 左偏 -28°", 0.52, 0.68, 0.42, -18.0, -28.0, 0.0),
        GuidePose(23, "中央大角度对角倾斜", "在中央摆出明显的对角透视倾角", "前倾 +26° / 左偏 -26°", 0.50, 0.50, 0.50, 26.0, -26.0, 12.0),
    ]


def project_guide_mesh(
    pose: GuidePose,
    pattern_size: Tuple[int, int] = (9, 6),
    viewport_aspect: float = 16.0 / 9.0,
    margin: float = 0.04,
) -> ProjectedGuideMesh:
    """
    将 3D 目标棋盘格（内角点外框 + 内部透视网格）投影至归一化 [0, 1] 视口坐标。
    自动适配横屏/竖屏宽高比并在越界时按中心安全微缩，保证 24 个姿态均完整落于画面安全区内。
    """
    safe_aspect = max(0.3, min(3.0, float(viewport_aspect)))
    cols = max(2, int(pattern_size[0]))
    rows = max(2, int(pattern_size[1]))
    board_aspect = float(cols - 1) / float(rows - 1)

    flat_scale_factor = VIRTUAL_Z0 / VIRTUAL_FOCAL
    half_w = (pose.scale * 0.5) * flat_scale_factor
    half_h = half_w / board_aspect

    rx = math.radians(pose.pitch_deg)
    ry = math.radians(pose.yaw_deg)
    rz = math.radians(pose.roll_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    def _project_raw(lx: float, ly: float, shrink: float = 1.0) -> Tuple[Tuple[float, float], float]:
        slx = lx * shrink
        sly = ly * shrink
        # 1. Pitch around X: positive pitch tilts top (ly < 0) away (+z)
        x1 = slx
        y1 = sly * cx
        z1 = -sly * sx
        # 2. Yaw around Y: positive yaw tilts right (lx > 0) away (+z)
        x2 = x1 * cy + z1 * sy
        y2 = y1
        z2 = x1 * sy + z1 * cy
        # 3. Roll around Z: clockwise in screen coords (x right, y down)
        x3 = x2 * cz - y2 * sz
        y3 = x2 * sz + y2 * cz
        z3 = z2 + VIRTUAL_Z0

        px = VIRTUAL_FOCAL * (x3 / z3)
        py = VIRTUAL_FOCAL * (y3 / z3)
        nx = pose.center_x + px
        ny = pose.center_y + py * safe_aspect
        return (nx, ny), z3

    corner_locals = [
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    ]
    raw_corners = [_project_raw(lx, ly, 1.0) for lx, ly in corner_locals]

    # 横屏/边缘安全微缩：确保所有角点严格位于 [margin, 1 - margin] 内
    shrink = 1.0
    lo, hi = margin, 1.0 - margin
    for (nx, ny), _ in raw_corners:
        if nx < lo and (pose.center_x - nx) > 1e-6:
            shrink = min(shrink, (pose.center_x - lo) / (pose.center_x - nx))
        elif nx > hi and (nx - pose.center_x) > 1e-6:
            shrink = min(shrink, (hi - pose.center_x) / (nx - pose.center_x))
        if ny < lo and (pose.center_y - ny) > 1e-6:
            shrink = min(shrink, (pose.center_y - lo) / (pose.center_y - ny))
        elif ny > hi and (ny - pose.center_y) > 1e-6:
            shrink = min(shrink, (hi - pose.center_y) / (ny - pose.center_y))
    shrink = max(0.35, min(1.0, shrink))

    proj_corners = [_project_raw(lx, ly, shrink) for lx, ly in corner_locals]
    outer_quad = [p[0] for p in proj_corners]
    depths = [p[1] for p in proj_corners]

    # 内部透视网格线
    grid_lines: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for c in range(1, cols - 1):
        t = float(c) / float(cols - 1)
        lx = -half_w + 2.0 * half_w * t
        grid_lines.append((_project_raw(lx, -half_h, shrink)[0], _project_raw(lx, half_h, shrink)[0]))
    for r in range(1, rows - 1):
        t = float(r) / float(rows - 1)
        ly = -half_h + 2.0 * half_h * t
        grid_lines.append((_project_raw(-half_w, ly, shrink)[0], _project_raw(half_w, ly, shrink)[0]))

    return ProjectedGuideMesh(
        outer_quad=outer_quad,
        corner_depths=depths,
        grid_lines=grid_lines,
    )


def extract_detected_quad_norm(
    corners: np.ndarray,
    pattern_size: Tuple[int, int],
    img_w: int,
    img_h: int,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """从 cv2.findChessboardCorners 输出提取归一化 4 外角点 [TL, TR, BR, BL] 与全部内角点"""
    cols, rows = int(pattern_size[0]), int(pattern_size[1])
    pts = corners.reshape(-1, 2)
    if len(pts) != cols * rows or img_w <= 0 or img_h <= 0:
        return [], []

    inv_w = 1.0 / float(img_w)
    inv_h = 1.0 / float(img_h)
    all_norm = [(float(p[0]) * inv_w, float(p[1]) * inv_h) for p in pts]
    tl = all_norm[0]
    tr = all_norm[cols - 1]
    br = all_norm[rows * cols - 1]
    bl = all_norm[(rows - 1) * cols]
    return [tl, tr, br, bl], all_norm


def centroid(pts: List[Tuple[float, float]]) -> Tuple[float, float]:
    n = float(len(pts))
    return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)


def quad_area(pts: List[Tuple[float, float]]) -> float:
    area = 0.0
    n = len(pts)
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        area += a[0] * b[1] - b[0] * a[1]
    return abs(area) * 0.5


def min_matched_corner_rms(
    detected: List[Tuple[float, float]],
    target: List[Tuple[float, float]],
) -> float:
    """计算旋转与绕向反转不变的最小四角点 RMS 归一化距离"""
    best = float("inf")
    for cand in (detected, list(reversed(detected))):
        for shift in range(4):
            sum_sq = 0.0
            for i in range(4):
                a = cand[(i + shift) % 4]
                b = target[i]
                ex = a[0] - b[0]
                ey = a[1] - b[1]
                sum_sq += ex * ex + ey * ey
            rms = math.sqrt(sum_sq / 4.0)
            if rms < best:
                best = rms
    return best


def evaluate_alignment(
    detected_quad_norm: List[Tuple[float, float]],
    target_quad_norm: List[Tuple[float, float]],
    detected_corners_norm: Optional[List[Tuple[float, float]]] = None,
    prev_quad_norm: Optional[List[Tuple[float, float]]] = None,
    stability_thresh_norm: float = 0.015,
) -> PoseAlignmentFeedback:
    """
    评估实时检测到的棋盘格四边形与目标 3D 引导框的对准度与静止稳定性。
    """
    corners_norm = list(detected_corners_norm or [])
    if len(detected_quad_norm) != 4 or len(target_quad_norm) != 4:
        return PoseAlignmentFeedback(
            detected=False,
            aligned=False,
            stable=False,
            match_score=0.0,
            guidance_text="请将棋盘格完整置于画面内",
        )

    det_c = centroid(detected_quad_norm)
    tgt_c = centroid(target_quad_norm)
    dx = tgt_c[0] - det_c[0]
    dy = tgt_c[1] - det_c[1]
    center_dist = math.hypot(dx, dy)

    det_area = max(1e-6, quad_area(detected_quad_norm))
    tgt_area = max(1e-6, quad_area(target_quad_norm))
    scale_ratio = math.sqrt(det_area / tgt_area)

    corner_rms = min_matched_corner_rms(detected_quad_norm, target_quad_norm)
    score = max(0.0, min(1.0, 1.0 - (corner_rms / 0.18)))

    aligned = (
        center_dist < 0.085
        and 0.72 <= scale_ratio <= 1.35
        and corner_rms < 0.095
    )

    # 防运动模糊：检查与上一检测帧的四角 RMS 位移
    stable = True
    if prev_quad_norm is not None and len(prev_quad_norm) == 4:
        motion_rms = min_matched_corner_rms(detected_quad_norm, prev_quad_norm)
        stable = (motion_rms <= stability_thresh_norm)

    if aligned and stable:
        guidance = "已对准目标轮廓，保持稳定..."
    elif aligned and not stable:
        guidance = "检测到画面晃动，请稳住标定板..."
    elif center_dist >= 0.085:
        if abs(dx) >= abs(dy):
            guidance = "请将标定板向右移动" if dx > 0.0 else "请将标定板向左移动"
        else:
            guidance = "请将标定板向下移动" if dy > 0.0 else "请将标定板向上移动"
    elif scale_ratio < 0.72:
        guidance = "标定板偏小，请靠近一些"
    elif scale_ratio > 1.35:
        guidance = "标定板偏大，请远离一些"
    else:
        guidance = "位置接近，请微调倾角以贴合透视轮廓"

    return PoseAlignmentFeedback(
        detected=True,
        aligned=aligned,
        stable=stable,
        match_score=score,
        guidance_text=guidance,
        detected_quad_norm=detected_quad_norm,
        detected_corners_norm=corners_norm,
    )
