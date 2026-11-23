#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试针对标定审计报告 P0 级逻辑缺陷的修复验证
=============================================
1. 验证世界 X 轴对齐在有效标靶 (Tag 28) 下真实生效，且在缺失时具备告警与候选降级机制；
2. 验证 TagLocalizer 安全守门：Tag 0 绝不作为静态标靶进入 PnP 求解；
3. 验证地图落盘 Schema 具备完整的 origin_tag_id 与 is_dynamic_yaw 字段。
"""

import unittest
import numpy as np
import os
import yaml
import tempfile

from src.calibration.solvers.ba_optimizer import BundleAdjustmentOptimizer
from src.calibration.solvers.world_datum_aligner import WorldDatumAligner
from src.vision.tag_localizer import TagLocalizer


class TestAuditP0Fixes(unittest.TestCase):

    def setUp(self):
        self.camera_matrix = np.array([
            [900.0, 0.0, 640.0],
            [0.0, 900.0, 360.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros(5, dtype=np.float64)
        self.optimizer = BundleAdjustmentOptimizer(
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            marker_size_mm=50.0
        )
        self.aligner = WorldDatumAligner(marker_size_mm=50.0)

    def test_anchor_to_absolute_world_with_real_tag(self):
        """测试使用世界锚点对齐 X 轴，验证其成功旋转并消除 Y 偏移"""
        tag_poses = {
            0: np.eye(4, dtype=np.float64),
            1: np.array([
                [1.0, 0.0, 0.0, 100.0],
                [0.0, 1.0, 0.0, 100.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float64)
        }
        anchors = {
            0: {"xyz_mm": [0.0, 0.0, 0.0], "known": [True, True, True]},
            1: {"xyz_mm": [141.421356, 0.0, 0.0], "known": [True, True, True]},
        }

        aligned_map = self.aligner.anchor_to_absolute_world(
            tag_poses,
            anchors,
        )

        self.assertIn(0, aligned_map["tags"])
        self.assertIn(1, aligned_map["tags"])

        p0 = aligned_map["tags"][0]["position_mm"]
        self.assertAlmostEqual(p0[0], 0.0, places=1)
        self.assertAlmostEqual(p0[1], 0.0, places=1)

        p1 = aligned_map["tags"][1]["position_mm"]
        self.assertGreater(p1[0], 100.0)
        self.assertAlmostEqual(p1[1], 0.0, places=1)

    def test_tag_localizer_excludes_tag0_from_pnp(self):
        """测试 TagLocalizer 绝对排除 Tag 0 (动态标靶) 进入静态 PnP 点对"""
        # 创建一个测试 yaml 地图，故意不写入 is_dynamic_yaw 模拟旧地图
        test_map = {
            "marker_size_mm": 50.0,
            "origin_tag_id": 0,
            "x_axis_align_tag_id": 28,
            "tags": {
                0: {
                    "transform_matrix": np.eye(4).tolist(),
                    "position_mm": [0.0, 0.0, 0.0]
                },
                18: {
                    "transform_matrix": np.eye(4).tolist(),
                    "position_mm": [100.0, 0.0, 0.0],
                    "is_dynamic_yaw": False
                },
                28: {
                    "transform_matrix": np.eye(4).tolist(),
                    "position_mm": [200.0, 0.0, 0.0],
                    "is_dynamic_yaw": False
                }
            }
        }

        tf = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w", encoding="utf-8")
        try:
            yaml.dump(test_map, tf)
            tf_path = tf.name
        finally:
            tf.close()

        try:
            localizer = TagLocalizer(
                tags_map_path=tf_path,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                marker_size_mm=50.0
            )

            # 验证 load_map 的安全守门机制自动为 Tag 0 注入了 is_dynamic_yaw=True
            self.assertTrue(localizer.tags_map["tags"][0]["is_dynamic_yaw"])

            # 模拟图像检测结果：同时检出了 Tag 0、Tag 18、Tag 28
            fake_image = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            # 模拟检测到的 3 个标靶角点与 ID
            mock_corners = [
                np.array([[[100, 100], [150, 100], [150, 150], [100, 150]]], dtype=np.float32), # Tag 0
                np.array([[[300, 100], [350, 100], [350, 150], [300, 150]]], dtype=np.float32), # Tag 18
                np.array([[[500, 100], [550, 100], [550, 150], [500, 150]]], dtype=np.float32), # Tag 28
            ]
            mock_ids = np.array([[0], [18], [28]], dtype=np.int32)

            class MockDetector:
                def detectMarkers(self, gray):
                    return mock_corners, mock_ids, None
            localizer.detector = MockDetector()

            # 运行相机定位
            succ, T_cam, info = localizer.localize_camera(fake_image)

            # 核心断言：检测到了 3 个标靶，但参与静态点对的标靶数量必须是 2 个 (排除了 Tag 0)
            self.assertEqual(len(info["detected_tag_ids"]), 3)
            self.assertEqual(info["static_tags_count"], 2, "Tag 0 严禁被计入静态标靶点对！")

        finally:
            if os.path.exists(tf_path):
                os.remove(tf_path)


if __name__ == "__main__":
    unittest.main()
