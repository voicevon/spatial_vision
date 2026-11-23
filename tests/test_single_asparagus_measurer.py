#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单根芦笋测量核心算法单元测试 (tests/test_single_asparagus_measurer.py)
"""

import unittest
import cv2
import numpy as np

from src.vision.single_asparagus_measurer import MeasureConfig, SingleAsparagusMeasurer


def _make_synthetic_asparagus(
    img_w: int = 640,
    img_h: int = 360,
    orientation: str = "left_to_right",
    white_px: int = 80,
    green_px: int = 320,
    diam_at_20mm_px: int = 28,
    diam_far_px: int = 16,
    mm_per_px: float = 0.5,
) -> np.ndarray:
    """
    合成深色背景下的单根芦笋 BGR 测试图：
    - 白根段长度 white_px，绿茎段长度 green_px
    - 绿茎前段（靠近分界线 20mm 处）宽度为 diam_at_20mm_px，后半段逐渐变细为 diam_far_px
    """
    img = np.full((img_h, img_w, 3), (20, 22, 24), dtype=np.uint8)
    green_bgr = (45, 165, 55)
    white_bgr = (225, 228, 222)

    total_px = white_px + green_px
    offset_20mm_px = int(round(20.0 / mm_per_px))

    if orientation in ("left_to_right", "right_to_left"):
        cy = img_h // 2
        x0 = (img_w - total_px) // 2
        for i in range(total_px):
            x = (x0 + i) if orientation == "left_to_right" else (x0 + total_px - 1 - i)
            is_white = i < white_px
            if is_white:
                half_w = diam_at_20mm_px // 2
                col = white_bgr
            else:
                g_pos = i - white_px
                # 在分界线向头部 0 ~ 1.8*offset_20mm 区间保持稳定直径 diam_at_20mm_px，远端收窄
                if g_pos <= int(offset_20mm_px * 1.8):
                    half_w = diam_at_20mm_px // 2
                else:
                    half_w = diam_far_px // 2
                col = green_bgr
            cv2.line(img, (x, cy - half_w), (x, cy + half_w), col, 1)
    else:
        cx = img_w // 2
        y0 = (img_h - total_px) // 2
        for i in range(total_px):
            y = (y0 + i) if orientation == "top_to_bottom" else (y0 + total_px - 1 - i)
            is_white = i < white_px
            if is_white:
                half_w = diam_at_20mm_px // 2
                col = white_bgr
            else:
                g_pos = i - white_px
                if g_pos <= int(offset_20mm_px * 1.8):
                    half_w = diam_at_20mm_px // 2
                else:
                    half_w = diam_far_px // 2
                col = green_bgr
            cv2.line(img, (cx - half_w, y), (cx + half_w, y), col, 1)

    return img


class TestSingleAsparagusMeasurer(unittest.TestCase):

    def test_with_white_root_left_to_right(self):
        """带白根单根芦笋：剔除白根计算绿色有效长度，并测量白绿分界线向头部偏移 20mm 处直径"""
        mm_per_px = 0.5
        img = _make_synthetic_asparagus(
            orientation="left_to_right", white_px=80, green_px=300,
            diam_at_20mm_px=28, diam_far_px=14, mm_per_px=mm_per_px,
        )
        cfg = MeasureConfig(orientation="left_to_right", diameter_offset_mm=20.0, fallback_mm_per_px=mm_per_px)
        res = SingleAsparagusMeasurer(cfg).measure(img, roi_rect=(80, 100, 480, 160))

        self.assertTrue(res.valid, res.reason)
        self.assertTrue(res.has_white_root)
        self.assertEqual(res.scale_source, "manual_scale")
        # 绿色有效长度理论值 = 300 * 0.5 = 150 mm
        self.assertAlmostEqual(res.effective_length_mm, 150.0, delta=6.0)
        # 偏移 20mm 处直径理论值 = 28 * 0.5 = 14 mm
        self.assertAlmostEqual(res.diameter_mm, 14.0, delta=1.5)
        self.assertIn("stage4_result", res.step_snapshots)

    def test_full_green_without_white_root(self):
        """无白根全绿芦笋：has_white_root=False，有效长度等于全长，从根切口偏移 20mm 测直径"""
        mm_per_px = 0.5
        img = _make_synthetic_asparagus(
            orientation="left_to_right", white_px=0, green_px=320,
            diam_at_20mm_px=30, diam_far_px=14, mm_per_px=mm_per_px,
        )
        cfg = MeasureConfig(orientation="left_to_right", diameter_offset_mm=20.0, fallback_mm_per_px=mm_per_px)
        res = SingleAsparagusMeasurer(cfg).measure(img)

        self.assertTrue(res.valid, res.reason)
        self.assertFalse(res.has_white_root)
        self.assertAlmostEqual(res.effective_length_mm, 160.0, delta=6.0)
        self.assertAlmostEqual(res.diameter_mm, 15.0, delta=1.5)

    def test_all_four_fixed_orientations(self):
        """验证四个固定朝向（左→右、右→左、上→下、下→上）均可正确定位分界与直径"""
        mm_per_px = 0.5
        for ori in ("left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"):
            img_w, img_h = (640, 360) if "left" in ori else (360, 640)
            img = _make_synthetic_asparagus(
                img_w=img_w, img_h=img_h, orientation=ori,
                white_px=60, green_px=260, diam_at_20mm_px=26, diam_far_px=14, mm_per_px=mm_per_px,
            )
            cfg = MeasureConfig(orientation=ori, diameter_offset_mm=20.0, fallback_mm_per_px=mm_per_px)
            res = SingleAsparagusMeasurer(cfg).measure(img)
            self.assertTrue(res.valid, f"orientation={ori}: {res.reason}")
            self.assertTrue(res.has_white_root, f"orientation={ori}")
            self.assertAlmostEqual(res.effective_length_mm, 130.0, delta=6.0)
            self.assertAlmostEqual(res.diameter_mm, 13.0, delta=1.5)

    def test_depth_intrinsics_scale_priority(self):
        """存在对齐深度图与相机内参时，优先使用 depth_intrinsics 换算物理尺寸"""
        # Z = 500mm, f = 1000px => mm_per_px = 0.5
        img = _make_synthetic_asparagus(
            orientation="left_to_right", white_px=60, green_px=280,
            diam_at_20mm_px=24, diam_far_px=12, mm_per_px=0.5,
        )
        depth_mm = np.full((img.shape[0], img.shape[1]), 500.0, dtype=np.float32)
        cfg = MeasureConfig(orientation="left_to_right", fallback_mm_per_px=9.99)
        res = SingleAsparagusMeasurer(cfg).measure(
            img, depth_mm=depth_mm, intrinsics=(1000.0, 1000.0, 320.0, 180.0),
        )
        self.assertTrue(res.valid)
        self.assertEqual(res.scale_source, "depth_intrinsics")
        self.assertAlmostEqual(res.mm_per_px, 0.5, delta=1e-3)
        self.assertAlmostEqual(res.effective_length_mm, 140.0, delta=6.0)
        self.assertAlmostEqual(res.diameter_mm, 12.0, delta=1.5)

    def test_roi_isolation_ignores_outside_distractor(self):
        """ROI 外部存在更大的绿色干扰块时，不应干扰 ROI 内单根测量"""
        img = _make_synthetic_asparagus(
            orientation="left_to_right", white_px=0, green_px=200,
            diam_at_20mm_px=20, diam_far_px=20, mm_per_px=0.5,
        )
        # 在 ROI 上方画一块巨大的绿色干扰区
        cv2.rectangle(img, (10, 5), (600, 90), (45, 165, 55), -1)
        cfg = MeasureConfig(orientation="left_to_right", fallback_mm_per_px=0.5)
        res = SingleAsparagusMeasurer(cfg).measure(img, roi_rect=(150, 120, 340, 120))
        self.assertTrue(res.valid)
        self.assertAlmostEqual(res.effective_length_mm, 100.0, delta=6.0)
        self.assertAlmostEqual(res.diameter_mm, 10.0, delta=1.5)


if __name__ == "__main__":
    unittest.main()
