# -*- coding: utf-8 -*-
"""Camera Hub UI 与棋盘格内参标定闭环单元测试 (严格隔离 gui_settings.json)"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np

import tools.camera_hub.app as camera_hub_mod
from tools.camera_hub.app import CameraHubApp
from tools.camera_hub.state import CameraHubState


def _make_synthetic_chessboard_views(out_dir: str, count: int = 5, pattern=(9, 6), img_size=(640, 480)):
    """生成多视角透视变换棋盘格合成图用于无硬件单测"""
    os.makedirs(out_dir, exist_ok=True)
    cols, rows = pattern
    sq = 32
    board_w, board_h = (cols + 1) * sq, (rows + 1) * sq
    board = np.full((board_h + 60, board_w + 60), 240, dtype=np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y0 = 30 + r * sq
                x0 = 30 + c * sq
                board[y0:y0 + sq, x0:x0 + sq] = 20

    bh, bw = board.shape[:2]
    src_pts = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
    w, h = img_size
    offsets = [
        (0, 0, 0, 0),
        (25, 10, -15, 20),
        (-20, 15, 20, -10),
        (15, -18, 12, 18),
        (-12, -10, -20, 12),
    ]
    paths = []
    for i in range(count):
        dx1, dy1, dx2, dy2 = offsets[i % len(offsets)]
        dst_pts = np.float32([
            [110 + dx1, 70 + dy1],
            [w - 110 + dx2, 75 - dy1],
            [w - 95 - dx1, h - 70 + dy2],
            [95 - dx2, h - 75 - dy2],
        ])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(board, M, (w, h), borderValue=220)
        bgr = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        p = os.path.join(out_dir, f"intr_{i + 1:04d}.png")
        cv2.imwrite(p, bgr)
        paths.append(p)
    return paths


class TestCameraHubApp(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_cam_hub_")
        self.cameras_root = os.path.join(self.temp_dir, "cameras")
        self.tmp_settings = os.path.join(self.temp_dir, "gui_settings.json")
        p = patch.object(camera_hub_mod, "GUI_SETTINGS_FILE", self.tmp_settings)
        p.start()
        self.addCleanup(p.stop)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_smoke_render_and_probe(self):
        app = CameraHubApp(cameras_root=self.cameras_root, settings_file=self.tmp_settings)
        canvas = app.render()
        self.assertIsNotNone(canvas)
        self.assertEqual(canvas.shape[2], 3)

        fake_devs = {
            "realsense": [{"name": "Intel RealSense D435", "serial": "99887766"}],
            "usb": [{"name": "USB Camera #0", "index": 0}],
        }
        with patch("tools.camera_hub.app.CameraService.probe_available_devices", return_value=fake_devs):
            n = app.probe_local_cameras()
        self.assertEqual(n, 2)
        self.assertEqual(len(app.state.cameras), 2)

        app.state.active_mode = CameraHubState.MODE_GALLERY
        canvas2 = app.render()
        self.assertIsNotNone(canvas2)

    def test_calibration_and_prune_flow(self):
        app = CameraHubApp(cameras_root=self.cameras_root, settings_file=self.tmp_settings)
        cam = app.registry.register_camera("测试D435", camera_type="realsense", hardware_serial="TEST01")
        app.state.refresh_cameras(select_id=cam.camera_id)
        app.state.select_resolution("640x480")

        raw_dir = cam.raw_images_dir("640x480")
        _make_synthetic_chessboard_views(raw_dir, count=5, pattern=(9, 6), img_size=(640, 480))
        # 塞入一张全黑无效图测试剔除
        bad_path = os.path.join(raw_dir, "intr_0099.png")
        cv2.imwrite(bad_path, np.zeros((480, 640, 3), dtype=np.uint8))

        app.state.refresh_profile_data()
        res = app.run_calibration()
        self.assertTrue(res.success, res.message)
        self.assertEqual(res.valid_images_count, 5)
        self.assertEqual(res.total_images_count, 6)
        self.assertIsNotNone(cam.load_intrinsics("640x480"))

        removed = app.prune_bad_images(max_error_px=1.0)
        self.assertGreaterEqual(removed, 1)
        self.assertFalse(os.path.exists(bad_path))


if __name__ == "__main__":
    unittest.main()
