#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""回归测试: Tag 高精度二次识别 (ROI 放大重检) — 合成图对比粗检 vs 高精度角点误差
(源自 temp/test_hp_detect.py, P2 收编为标准单测)"""
import unittest

import cv2
import numpy as np

from src.calibration.tag_detector import TagDetector
from tools.tracker.app import RobotOnlineTracker


class TestHighPrecisionRedetect(unittest.TestCase):

    """高精度二次识别角点误差应不劣于全景粗检"""

    @classmethod
    def setUpClass(cls):
        cls.K = np.array([[1363.68, 0, 971.19], [0, 1361.19, 566.26], [0, 0, 1.0]])
        cls.detector = TagDetector(valid_tag_ids=None)
        # 合成 1280x720 图, Tag 2 (16h5) 以带透视的四边形贴入 + 模糊 + 噪声
        marker = cv2.aruco.generateImageMarker(cls.detector.dictionary, 2, 200)
        marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        img = np.full((720, 1280, 3), 210, dtype=np.uint8)
        src = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], dtype=np.float32)
        cls.gt = np.array([[420, 300], [580, 282], [596, 442], [412, 462]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, cls.gt)
        warp = cv2.warpPerspective(marker, M, (1280, 720), borderValue=(210, 210, 210))
        mask = np.full((720, 1280), 255, np.uint8)
        cv2.fillPoly(mask, [cls.gt.astype(np.int32)], 0)
        img[mask == 0] = warp[mask == 0]
        img = cv2.GaussianBlur(img, (3, 3), 0.8)
        noise = np.random.default_rng(7).normal(0, 5, img.shape).astype(np.float32)
        cls.img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def test_hp_not_worse_than_coarse(self):
        det = self.detector.detect_tags(self.img)
        self.assertIn(2, det, f"全景未检出 Tag 2, 实际: {list(det.keys())}")
        coarse = det[2].reshape(4, 2)
        err_coarse = float(np.mean(np.linalg.norm(coarse - self.gt, axis=1)))

        app = RobotOnlineTracker.__new__(RobotOnlineTracker)
        app.target_tag_id = 2
        app.tag_detector = self.detector
        det_hp = app._detect_high_precision(self.img, dict(det))
        self.assertIn(2, det_hp, "高精度流程丢失 Tag 2")
        hp = det_hp[2].reshape(4, 2)
        err_hp = float(np.mean(np.linalg.norm(hp - self.gt, axis=1)))
        self.assertLessEqual(err_hp, err_coarse * 1.05, "高精度结果未达粗检水平")


if __name__ == "__main__":
    unittest.main()
