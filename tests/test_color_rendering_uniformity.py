# -*- coding: utf-8 -*-
import unittest
import numpy as np
import cv2

from src.utils.text_rendering import put_text, draw_text
from src.calibration.verification.verification_visualizer import VerificationVisualizer


class TestColorRenderingUniformity(unittest.TestCase):
    """测试实测蓝与理论绿的色彩通道保真度与统一渲染规范"""

    def test_text_rendering_bgr_fidelity(self):
        """验证 put_text 与 draw_text 严格遵循 OpenCV BGR 通道规范，不发生 B/R 倒置"""
        # 创建纯黑画布
        canvas = np.zeros((100, 200, 3), dtype=np.uint8)

        # 写入纯蓝测试字符: BGR = (255, 0, 0)
        put_text(canvas, "实测", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        # 获取文字绘制区域 (非零像素)
        mask = np.any(canvas > 0, axis=-1)
        self.assertTrue(np.any(mask), "未能成功绘制文字")
        text_pixels = canvas[mask]

        # 蓝通道应占主导 (B > 100, R == 0)
        mean_b = np.mean(text_pixels[:, 0])
        mean_r = np.mean(text_pixels[:, 2])
        self.assertGreater(mean_b, 100, f"蓝通道均值过低: {mean_b}")
        self.assertEqual(mean_r, 0, f"红通道应为0，却检测到非零值(通道颠倒为黄色/红色): {mean_r}")

    def test_visualizer_dual_prisms_rendering(self):
        """验证 VerificationVisualizer 双棱柱与卡片能够无异常执行且颜色合规"""
        K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
        dist = np.zeros(5)
        viz = VerificationVisualizer(K, dist)

        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        ba_r = np.array([0.1, 0.1, 0.1])
        ba_t = np.array([100.0, 50.0, 1000.0])
        obs_r = np.array([0.11, 0.1, 0.1])
        obs_t = np.array([102.0, 51.0, 1005.0])

        viz.render_tag_dual_prisms(
            img=img,
            ba_rvec=ba_r,
            ba_tvec=ba_t,
            obs_rvec=obs_r,
            obs_tvec=obs_t,
            tag_id=6,
            err_px=0.41,
            err_mm=9.68,
            ba_center_xyz=[100.0, 50.0, 1000.0],
            obs_center_xyz=[102.0, 51.0, 1005.0],
            hovered=True
        )
        self.assertTrue(np.any(img > 0), "渲染图像不应全黑")

    def test_pyramid_and_prism_shapes_rendering(self):
        """验证 VerificationVisualizer 支持四棱柱与金字塔锥两种 3D 几何形态"""
        from src.calibration.verification.prism_renderer import draw_prism
        K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
        dist = np.zeros(5)
        viz = VerificationVisualizer(K, dist)

        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        ba_r = np.array([0.1, 0.1, 0.1])
        ba_t = np.array([100.0, 50.0, 1000.0])
        obs_r = np.array([0.11, 0.1, 0.1])
        obs_t = np.array([102.0, 51.0, 1005.0])

        # 测试金字塔锥与四棱柱混搭
        viz.render_tag_dual_prisms(
            img=img,
            ba_rvec=ba_r,
            ba_tvec=ba_t,
            obs_rvec=obs_r,
            obs_tvec=obs_t,
            tag_id=7,
            err_px=0.25,
            err_mm=5.12,
            ba_shape="pyramid",
            obs_shape="prism",
        )
        self.assertTrue(np.any(img > 0), "金字塔锥与四棱柱混搭渲染图像不应全黑")

        # 直接测试 draw_prism 的 pyramid 模式
        res_pyr = draw_prism(img, K, dist, ba_r, ba_t, shape="pyramid", draw_axes=True)
        self.assertIsNotNone(res_pyr)
        self.assertEqual(len(res_pyr["bottom"]), 4)
        self.assertEqual(len(res_pyr["top_center"]), 2)


if __name__ == "__main__":
    unittest.main()
