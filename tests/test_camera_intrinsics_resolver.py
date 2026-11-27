# -*- coding: utf-8 -*-
import os
import shutil
import tempfile
import unittest
import numpy as np

from src.cameras.camera_entity import IntrinsicsRecord
from src.cameras.camera_registry import CameraRegistry
from src.utils.intrinsics_resolver import resolve_camera_intrinsics
from src.workspace.workspace_manager import WorkspaceManager


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
    """测试相机内参自适应解析器与全局相机库优先机制"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test_cam_intr_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_hardware_profile_direct_read(self):
        """测试无全局相机绑定时硬件 active profile 在线直读优先级"""
        dummy_prof = DummyStreamProfile()
        K, dist, meta = resolve_camera_intrinsics(stream_profile=dummy_prof)
        self.assertEqual(meta["source"], "realsense_hardware_profile")
        self.assertAlmostEqual(K[0, 0], 1234.56, places=2)
        self.assertAlmostEqual(K[1, 1], 1230.45, places=2)
        self.assertAlmostEqual(K[0, 2], 965.12, places=2)
        self.assertAlmostEqual(dist[0, 0], 0.01, places=3)

    def test_resolution_adaptive_scaling(self):
        """测试 1080P 内参在遇到 720P 图像时的自动等比缩放"""
        dummy_prof = DummyStreamProfile()
        K, dist, meta = resolve_camera_intrinsics(
            stream_profile=dummy_prof,
            actual_image_shape=(720, 1280)
        )
        self.assertTrue(meta["scaled"])
        expected_scale = 1280.0 / 1920.0
        self.assertAlmostEqual(K[0, 0], 1234.56 * expected_scale, places=2)
        self.assertAlmostEqual(K[0, 2], 965.12 * expected_scale, places=2)

    def test_workspace_bound_camera_registry_priority(self):
        """测试工位绑定全局 camera_id 时优先从全局相机库加载高精内参"""
        cams_root = os.path.join(self.tmp_dir, "cameras")
        ws_root = os.path.join(self.tmp_dir, "workspaces")
        reg = CameraRegistry(cameras_root=cams_root)
        cam = reg.register_camera("工位主摄D435", camera_type="realsense", hardware_serial="SN_PRIO")
        rec = IntrinsicsRecord(
            width=1920, height=1080,
            fx=1225.0, fy=1222.0, cx=960.0, cy=540.0,
            dist_coeffs=[-0.05, 0.08, 0.0, 0.0, -0.02],
            rmse=0.25, valid_images_count=10, total_images_count=10,
        )
        cam.save_intrinsics("1920x1080", rec)

        ws_mgr = WorkspaceManager(workspaces_dir=ws_root)
        ws = ws_mgr.create_workspace("ws_prio", camera_id=cam.camera_id)

        # 即便有硬件 profile 也必须以全局相机库高精标定内参优先
        dummy_prof = DummyStreamProfile()
        K, dist, meta = resolve_camera_intrinsics(
            stream_profile=dummy_prof,
            workspace_dir=ws.workspace_dir,
            cameras_root=cams_root,
        )
        self.assertEqual(meta["source"], "camera_registry")
        self.assertEqual(meta["camera_id"], cam.camera_id)
        self.assertAlmostEqual(K[0, 0], 1225.0, places=1)
        self.assertAlmostEqual(K[1, 1], 1222.0, places=1)
        self.assertAlmostEqual(dist[0, 0], -0.05, places=3)
        self.assertAlmostEqual(dist[4, 0], -0.02, places=3)


if __name__ == "__main__":
    unittest.main()
