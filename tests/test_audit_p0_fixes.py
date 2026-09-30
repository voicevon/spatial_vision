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

    def test_align_to_scara_world_with_real_tag(self):
        """测试使用真实存在的 Tag 28 对齐 X 轴，验证其成功旋转并消除 Y 偏移"""
        # 构造包含 Tag 0 (原点) 与 Tag 28 (位于 (100, 100, 0)) 的位姿字典
        tag_poses = {
            0: np.eye(4, dtype=np.float64),
            28: np.array([
                [1.0, 0.0, 0.0, 100.0],
                [0.0, 1.0, 0.0, 100.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float64)
        }

        aligned_map = self.aligner.align_to_scara_world(
            tag_poses,
            origin_tag_id=0,
            x_align_tag_id=28
        )

        self.assertIn(0, aligned_map["tags"])
        self.assertIn(28, aligned_map["tags"])
        self.assertEqual(aligned_map["origin_tag_id"], 0)
        self.assertEqual(aligned_map["x_axis_align_tag_id"], 28)

        # 原点位置必须为 (0, 0, 0)
        p0 = aligned_map["tags"][0]["position_mm"]
        self.assertAlmostEqual(p0[0], 0.0, places=1)
        self.assertAlmostEqual(p0[1], 0.0, places=1)

        # 对齐后的 Tag 28 其 Y 坐标必须被旋转校正为接近 0.0 (严密处于 +X 轴)
        p28 = aligned_map["tags"][28]["position_mm"]
        self.assertGreater(p28[0], 100.0, "旋转到 +X 轴后，X 坐标应大于等于 100mm")
        self.assertAlmostEqual(p28[1], 0.0, places=1, msg="X 轴对齐后 Tag 28 的 Y 偏差必须严格趋近于 0")

        # 检查标记
        self.assertTrue(aligned_map["tags"][0]["is_origin"])
        self.assertTrue(aligned_map["tags"][0]["is_dynamic_yaw"])
        self.assertFalse(aligned_map["tags"][28]["is_origin"])
        self.assertFalse(aligned_map["tags"][28]["is_dynamic_yaw"])

    def test_align_to_scara_world_fallback_when_x_tag_missing(self):
        """测试当指定的 X 轴标靶缺失时，系统自动降级使用最远端刚体标靶而非静默跳过"""
        # 构造包含 Tag 0、Tag 18 (较远) 的位姿，但未包含 Tag 1
        tag_poses = {
            0: np.eye(4, dtype=np.float64),
            18: np.array([
                [1.0, 0.0, 0.0, 200.0],
                [0.0, 1.0, 0.0, 200.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float64)
        }

        # 传入现场不存在的 Tag 1
        aligned_map = self.aligner.align_to_scara_world(
            tag_poses,
            origin_tag_id=0,
            x_align_tag_id=1
        )

        # 验证自适应降级至候选远端标靶 Tag 18
        self.assertEqual(aligned_map["x_axis_align_tag_id"], 18)
        p18 = aligned_map["tags"][18]["position_mm"]
        self.assertAlmostEqual(p18[1], 0.0, places=1, msg="自适应降级后 Tag 18 的 Y 坐标亦应被校正为 0")

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

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w", encoding="utf-8") as tf:
            yaml.dump(test_map, tf)
            tf_path = tf.name

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
