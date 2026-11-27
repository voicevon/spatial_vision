# -*- coding: utf-8 -*-
"""全局多相机资产注册表与两层目录契约单元测试"""

import os
import shutil
import tempfile
import unittest
import cv2
import numpy as np

from src.cameras.camera_entity import IntrinsicsRecord
from src.cameras.camera_registry import CameraRegistry
from src.utils.intrinsics_resolver import resolve_camera_intrinsics


class TestCameraRegistry(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_cam_reg_")
        self.registry = CameraRegistry(cameras_root=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_register_realsense_and_usb_ids(self):
        rs = self.registry.register_camera("产线1号D435", camera_type="realsense", hardware_serial="141722070123")
        self.assertEqual(rs.camera_id, "rs_141722070123")
        # 重复登记同序列号应返回既有档案
        rs_again = self.registry.register_camera("重复", camera_type="realsense", hardware_serial="141722070123")
        self.assertEqual(rs_again.camera_id, rs.camera_id)

        usb = self.registry.register_camera("工位USB", camera_type="usb", usb_index=0)
        self.assertTrue(usb.camera_id.startswith("usb_"))
        self.assertEqual(len(self.registry.list_cameras()), 2)

        # 重命名与删除
        self.assertTrue(self.registry.rename_camera(usb.camera_id, "工位USB-改名"))
        self.assertEqual(self.registry.get_camera(usb.camera_id).name, "工位USB-改名")
        self.assertTrue(self.registry.delete_camera(usb.camera_id))
        self.assertIsNone(self.registry.get_camera(usb.camera_id))

    def test_two_level_resolution_and_non_overwrite_numbering(self):
        cam = self.registry.register_camera("D435-A", camera_type="realsense", hardware_serial="SN999")
        p1 = cam.next_image_path("1920x1080")
        self.assertTrue(p1.endswith("intr_0001.png"))
        cv2.imwrite(p1, np.zeros((1080, 1920, 3), dtype=np.uint8))

        p2 = cam.next_image_path("1920x1080")
        self.assertTrue(p2.endswith("intr_0002.png"))
        cv2.imwrite(p2, np.zeros((1080, 1920, 3), dtype=np.uint8))

        # 删除第 1 张后下一张应为 0003，禁止覆盖
        self.assertTrue(cam.delete_image("1920x1080", p1))
        p3 = cam.next_image_path("1920x1080")
        self.assertTrue(p3.endswith("intr_0003.png"))

    def test_intrinsics_save_and_resolver_lookup(self):
        cam = self.registry.register_camera("D435-B", camera_type="realsense", hardware_serial="SN888")
        rec = IntrinsicsRecord(
            width=1920, height=1080,
            fx=1380.5, fy=1379.2, cx=960.0, cy=540.0,
            dist_coeffs=[0.01, -0.02, 0.0, 0.0, 0.0],
            rmse=0.23, valid_images_count=12, total_images_count=12,
        )
        self.assertTrue(cam.save_intrinsics("1920x1080", rec))

        K, dist, meta = resolve_camera_intrinsics(
            camera_id=cam.camera_id,
            actual_image_shape=(1080, 1920),
            cameras_root=self.temp_dir,
        )
        self.assertEqual(meta["source"], "camera_registry")
        self.assertEqual(meta["camera_id"], cam.camera_id)
        self.assertAlmostEqual(float(K[0, 0]), 1380.5, places=2)
        self.assertAlmostEqual(float(dist[0, 0]), 0.01, places=4)


if __name__ == "__main__":
    unittest.main()
