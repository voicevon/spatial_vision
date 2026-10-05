# -*- coding: utf-8 -*-
import os
import shutil
import tempfile
import unittest
import numpy as np
import yaml

from src.utils.config_guard import resolve_camera_intrinsics
from src.workspace.workspace_manager import Workspace


class DummyHardwareIntrinsics:
    def __init__(self):
        self.width = 1920
        self.height = 1080
        self.fx = 1234.56
        self.fy = 1230.45
        self.ppx = 965.12
        self.ppy = 542.31
        self.coeffs = [0.01, -0.02, 0.001, 0.002, 0.0]


class DummyStreamProfile:
    def get_intrinsics(self):
        return DummyHardwareIntrinsics()


class TestCameraIntrinsicsResolver(unittest.TestCase):
    """测试相机内参自适应解析器与工位沙盒专属机制"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test_cam_intr_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_hardware_profile_direct_read(self):
        """测试硬件 active profile 在线直读最高优先级"""
        dummy_prof = DummyStreamProfile()
        K, dist, meta = resolve_camera_intrinsics(stream_profile=dummy_prof)
        self.assertEqual(meta["source"], "realsense_hardware_profile")
        self.assertAlmostEqual(K[0, 0], 1234.56, places=2)
        self.assertAlmostEqual(K[1, 1], 1230.45, places=2)
        self.assertAlmostEqual(K[0, 2], 965.12, places=2)
        self.assertAlmostEqual(dist[0, 0], 0.01, places=3)

    def test_resolution_adaptive_scaling(self):
        """测试 1080P 内参在遇到 720P 图像时的自动等比缩放"""
        # 基准 1920x1080 -> 实际图像 1280x720 (缩放比 2/3)
        dummy_prof = DummyStreamProfile()
        K, dist, meta = resolve_camera_intrinsics(
            stream_profile=dummy_prof,
            actual_image_shape=(720, 1280)
        )
        self.assertTrue(meta["scaled"])
        expected_scale = 1280.0 / 1920.0
        self.assertAlmostEqual(K[0, 0], 1234.56 * expected_scale, places=2)
        self.assertAlmostEqual(K[0, 2], 965.12 * expected_scale, places=2)

    def test_workspace_custom_intrinsics_priority(self):
        """测试工位沙盒 intrinsics/camera_intrinsics.yaml 专属内参优先加载"""
        intr_dir = os.path.join(self.tmp_dir, "intrinsics")
        os.makedirs(intr_dir, exist_ok=True)
        intr_file = os.path.join(intr_dir, "camera_intrinsics.yaml")
        
        ws_intr_data = {
            "width": 1920,
            "height": 1080,
            "fx": 1225.0,
            "fy": 1222.0,
            "cx": 960.0,
            "cy": 540.0,
            "dist_coeffs": [-0.05, 0.08, 0.0, 0.0, -0.02]
        }
        with open(intr_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(ws_intr_data, f)

        # 传入工位目录，即便有硬件 profile 也必须以工位高精校准内参优先！
        dummy_prof = DummyStreamProfile()
        K, dist, meta = resolve_camera_intrinsics(
            stream_profile=dummy_prof,
            workspace_dir=self.tmp_dir
        )
        self.assertEqual(meta["source"], "workspace_intrinsics")
        self.assertAlmostEqual(K[0, 0], 1225.0, places=1)
        self.assertAlmostEqual(K[1, 1], 1222.0, places=1)
        self.assertAlmostEqual(dist[0, 0], -0.05, places=3)
        self.assertAlmostEqual(dist[4, 0], -0.02, places=3)

    def test_workspace_class_property(self):
        """测试 Workspace 对象的 camera_intrinsics_path 属性"""
        ws = Workspace(workspace_id="test_ws", name="Test Workspace", workspace_dir=self.tmp_dir)
        expected_path = os.path.join(self.tmp_dir, "intrinsics", "camera_intrinsics.yaml")
        self.assertEqual(ws.camera_intrinsics_path, expected_path)


if __name__ == "__main__":
    unittest.main()
