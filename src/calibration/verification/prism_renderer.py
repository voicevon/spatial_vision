#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一 3D 四棱柱渲染器 (PrismRenderer)
====================================
收编四处重复的棱柱投影绘制实现 (verification_visualizer 双棱柱对比、
tracker/renderer Studio 同款棱柱、capture_wizard / tag_map_builder 建图可视化棱柱)。

纯函数式设计：输入相机内参与位姿，输出 OpenCV 绘制调用；
颜色 / 尺寸 / 透明度全部参数化，坐标轴绘制可选。
"""

from typing import Dict, Optional

import cv2
import numpy as np

from src.ui.text_rendering import put_text

# Studio 双棱柱配色 (VerificationVisualizer 历史样式)
COLORS_THEORY = dict(  # BA 理论 (翡翠绿)
    side=(0, 185, 60), cap=(50, 240, 100), edge=(0, 255, 100),
    top_edge=(120, 255, 160), dot=(0, 255, 120))
COLORS_OBSERVED = dict(  # 实测 (科技天蓝)
    side=(235, 125, 20), cap=(255, 175, 50), edge=(255, 195, 70),
    top_edge=(255, 255, 255), dot=(255, 200, 60))

# 建图可视化配色 (wizard/builder 实心柱体样式)
COLORS_MAPPING = dict(
    side=(240, 160, 30), cap=(255, 220, 90), edge=(180, 80, 0),
    top_edge=(255, 240, 120), dot=(255, 255, 255))

# 坐标轴配色 (X 红 / Y 绿 / Z 金黄文字)
AXIS_X_COLOR = (0, 0, 240)
AXIS_Y_COLOR = (0, 220, 0)
AXIS_Z_LABEL_COLOR = (255, 230, 80)


def draw_prism(
    img: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    half_w: float = 15.0,
    height: float = 75.0,
    colors: Optional[dict] = None,
    alpha: float = 0.35,
    draw_axes: bool = False,
    axis_len: float = 25.0,
    shape: str = "prism",
) -> Optional[Dict[str, np.ndarray]]:
    """
    在图像上绘制单根半透明实心 3D 几何标靶 (四棱柱 prism 或金字塔锥 pyramid)。
    :param colors: dict(side, cap, edge, top_edge, dot) BGR 配色, None=Studio 理论绿
    :param alpha: 半透明填充强度 (0~1)
    :param draw_axes: 绘制 X/Y 轴 (自标靶中心伸出 axis_len) 与 Z 顶面标注
    :param shape: 几何形态 'prism' (四棱柱) 或 'pyramid' (金字塔锥)
    :return: {"bottom": 4x2, "top": 4x2, "top_center": (2,), "origin": (2,)} 像素投影点
    """
    c = dict(COLORS_THEORY if colors is None else colors)

    if shape == "pyramid":
        pts_3d = [
            [-half_w, -half_w, 0.0], [half_w, -half_w, 0.0],
            [half_w, half_w, 0.0], [-half_w, half_w, 0.0],        # 底面 4 点 (0:3)
            [0.0, 0.0, height],                                    # 锥顶 Apex (4)
        ]
        if draw_axes:
            pts_3d += [[axis_len, 0.0, 0.0], [0.0, axis_len, 0.0], [0.0, 0.0, 0.0]]  # X/Y 端点与原点 (5:7)

        proj, _ = cv2.projectPoints(
            np.array(pts_3d, dtype=np.float64), rvec, tvec,
            np.asarray(camera_matrix, dtype=np.float64),
            np.asarray(dist_coeffs, dtype=np.float64))
        proj = proj.reshape((-1, 2)).astype(int)
        b, apex = proj[0:4], tuple(proj[4])

        # 1. 半透明填充 4 个三角形侧面与底面
        overlay = img.copy()
        for i in range(4):
            j = (i + 1) % 4
            cv2.fillPoly(overlay, [np.array([b[i], b[j], apex], dtype=np.int32)], c["side"])
        cv2.fillPoly(overlay, [b], c["cap"])
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

        # 2. 8 条棱线描边 (底圈 4 条 + 4 条斜棱)
        cv2.polylines(img, [b], True, c["edge"], 2, cv2.LINE_AA)
        for i in range(4):
            cv2.line(img, tuple(b[i]), apex, c.get("top_edge", c["edge"]), 2, cv2.LINE_AA)

        # 3. 锥顶标注点
        if c.get("dot") is not None:
            cv2.circle(img, apex, 4 if not draw_axes else 3, c["dot"], -1, cv2.LINE_AA)

        # 4. 坐标轴 (X 红 / Y 绿 / Z 金黄文字)
        if draw_axes:
            p_x, p_y, p_orig = tuple(proj[5]), tuple(proj[6]), tuple(proj[7])
            cv2.line(img, p_orig, p_x, AXIS_X_COLOR, 2, cv2.LINE_AA)
            put_text(img, "X", p_x, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.line(img, p_orig, p_y, AXIS_Y_COLOR, 2, cv2.LINE_AA)
            put_text(img, "Y", p_y, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            put_text(img, "Z", (apex[0] + 5, apex[1] - 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, AXIS_Z_LABEL_COLOR, 2, cv2.LINE_AA)

        return {"bottom": b, "top": np.array([apex, apex, apex, apex]), "top_center": np.array(apex), "origin": proj[7] if draw_axes else None}

    pts_3d = [
        [-half_w, -half_w, 0.0], [half_w, -half_w, 0.0],
        [half_w, half_w, 0.0], [-half_w, half_w, 0.0],        # 底面 4 点 (0:3)
        [-half_w, -half_w, height], [half_w, -half_w, height],
        [half_w, half_w, height], [-half_w, half_w, height],  # 顶面 4 点 (4:7)
        [0.0, 0.0, height],                                    # 顶面中心 (8)
    ]
    if draw_axes:
        pts_3d += [[axis_len, 0.0, 0.0], [0.0, axis_len, 0.0], [0.0, 0.0, 0.0]]  # X/Y 端点与原点 (9:11)

    proj, _ = cv2.projectPoints(
        np.array(pts_3d, dtype=np.float64), rvec, tvec,
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(dist_coeffs, dtype=np.float64))
    proj = proj.reshape((-1, 2)).astype(int)
    b, t, tc = proj[0:4], proj[4:8], tuple(proj[8])

    # 1. 半透明填充 4 个侧面与顶面
    overlay = img.copy()
    for i in range(4):
        j = (i + 1) % 4
        cv2.fillPoly(overlay, [np.array([b[i], b[j], t[j], t[i]], dtype=np.int32)], c["side"])
    cv2.fillPoly(overlay, [t], c["cap"])
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)

    # 2. 12 条棱线描边 (底圈 / 立柱 / 顶圈)
    cv2.polylines(img, [b], True, c["edge"], 2, cv2.LINE_AA)
    for i in range(4):
        cv2.line(img, tuple(b[i]), tuple(t[i]), c["edge"], 2, cv2.LINE_AA)
    cv2.polylines(img, [t], True, c.get("top_edge", c["edge"]), 2, cv2.LINE_AA)

    # 3. 顶面中心标注点
    if c.get("dot") is not None:
        cv2.circle(img, tc, 4 if not draw_axes else 3, c["dot"], -1, cv2.LINE_AA)

    # 4. 坐标轴 (X 红 / Y 绿 / Z 金黄文字)
    if draw_axes:
        p_x, p_y, p_orig = tuple(proj[9]), tuple(proj[10]), tuple(proj[11])
        cv2.line(img, p_orig, p_x, AXIS_X_COLOR, 2, cv2.LINE_AA)
        put_text(img, "X", p_x, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(img, p_orig, p_y, AXIS_Y_COLOR, 2, cv2.LINE_AA)
        put_text(img, "Y", p_y, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        put_text(img, "Z", (tc[0] + 5, tc[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, AXIS_Z_LABEL_COLOR, 2, cv2.LINE_AA)

    return {"bottom": b, "top": t, "top_center": np.array(tc), "origin": proj[11] if draw_axes else None}
