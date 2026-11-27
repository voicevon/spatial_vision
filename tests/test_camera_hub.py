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

    def test_auto_capture_hold_stability_skip_undo_and_overlay_render(self):
        """测试 3D 引导自动抓拍 0.7s 稳定触发、晃动重置、跳过/撤销与 3D 叠加层渲染"""
        from src.calibration.solvers.chessboard_guide import project_guide_mesh

        app = CameraHubApp(cameras_root=self.cameras_root, settings_file=self.tmp_settings)
        app.sound_enabled = False  # 单测静音
        cam = app.registry.register_camera("自动抓拍D435", camera_type="realsense", hardware_serial="AUTO01")
        app.state.refresh_cameras(select_id=cam.camera_id)
        app.state.select_resolution("640x480")
        app.state.pipeline_running = True

        dummy_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        pose0 = app.state.get_current_pose()
        mesh0 = project_guide_mesh(pose0, app.state.pattern_size, viewport_aspect=640.0 / 480.0)
        # 构造与目标外框精确对齐的 9x6 合成角点
        tl, tr, br, bl = mesh0.outer_quad
        cols, rows = app.state.pattern_size
        pts = []
        for r in range(rows):
            v = r / float(rows - 1)
            for c in range(cols):
                u = c / float(cols - 1)
                x = (1 - u) * (1 - v) * tl[0] + u * (1 - v) * tr[0] + u * v * br[0] + (1 - u) * v * bl[0]
                y = (1 - u) * (1 - v) * tl[1] + u * (1 - v) * tr[1] + u * v * br[1] + (1 - u) * v * bl[1]
                pts.append([[x * 640.0, y * 480.0]])
        aligned_corners = np.array(pts, dtype=np.float32)

        with patch("tools.camera_hub.app.detect_chessboard_corners", return_value=(True, aligned_corners)):
            # t=10.0 首次对准 -> 启动计时，未抓拍
            app.process_live_frame_for_auto_capture(dummy_frame, now=10.0)
            self.assertEqual(app.state.current_pose_idx, 0)
            self.assertEqual(len(app.state.images), 0)

            # t=10.35 保持中 -> 进度约 0.5，渲染 3D 引导叠加层不报错
            app.process_live_frame_for_auto_capture(dummy_frame, now=10.35)
            self.assertAlmostEqual(app.state.hold_progress, 0.5, places=1)
            canvas = app.renderer.render(1280, 760, (0, 0))
            self.assertEqual(canvas.shape, (760, 1280, 3))

            # t=10.75 满 0.7s -> 自动抓拍保存并推进至姿态 1
            app.process_live_frame_for_auto_capture(dummy_frame, now=10.75)
            self.assertEqual(len(app.state.images), 1)
            self.assertEqual(app.state.current_pose_idx, 1)

        # 撤销上一张 -> 删除照片并回退到姿态 0
        self.assertTrue(app.undo_last_capture())
        self.assertEqual(len(app.state.images), 0)
        self.assertEqual(app.state.current_pose_idx, 0)

        # 跳过当前姿态 -> 前进到姿态 1
        app.state.skip_pose()
        self.assertEqual(app.state.current_pose_idx, 1)

        # 关闭自动抓拍开关 -> 即使持续对准超过 1s 也不自动触发
        app.state.current_pose_idx = 0
        app.state.auto_capture_enabled = False
        app.state.capture_cooldown_until = 0.0
        with patch("tools.camera_hub.app.detect_chessboard_corners", return_value=(True, aligned_corners)):
            app.process_live_frame_for_auto_capture(dummy_frame, now=20.0)
            app.process_live_frame_for_auto_capture(dummy_frame, now=21.5)
            self.assertEqual(len(app.state.images), 0)
            self.assertEqual(app.state.hold_progress, 0.0)

    def test_front_facing_mirror_preview_and_unflipped_save(self):
        """测试前置/后置持久化、左右镜像预览对齐及落盘保持传感器真实原图不翻转"""
        from src.calibration.solvers.chessboard_guide import project_guide_mesh

        app = CameraHubApp(cameras_root=self.cameras_root, settings_file=self.tmp_settings)
        app.sound_enabled = False
        cam = app.registry.register_camera("笔记本前摄", camera_type="usb", usb_index=0)
        app.state.refresh_cameras(select_id=cam.camera_id)
        app.state.select_resolution("640x480")

        # 默认后置 -> 切换前置并验证落盘持久化
        self.assertFalse(app.state.is_mirror_preview)
        self.assertEqual(app.state.toggle_camera_facing(), "front")
        self.assertTrue(app.state.is_mirror_preview)
        reloaded = app.registry.get_camera(cam.camera_id)
        self.assertTrue(reloaded.is_front_facing)

        # 构造左半边全白(255)、右半边全黑(0)的传感器原图
        raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        raw_frame[:, :320] = 255
        app.state.pipeline_running = True
        app.state.live_frame = raw_frame

        # 选中姿态 1（左上角正面, center_x=0.30）：在未翻转原图中对应右侧 x = 1 - target_x
        app.state.current_pose_idx = 1
        mesh1 = project_guide_mesh(app.state.get_current_pose(), app.state.pattern_size, viewport_aspect=640.0 / 480.0)
        tl, tr, br, bl = mesh1.outer_quad
        cols, rows = app.state.pattern_size
        raw_pts = []
        for r in range(rows):
            v = r / float(rows - 1)
            for c in range(cols):
                u = c / float(cols - 1)
                mx = (1 - u) * (1 - v) * tl[0] + u * (1 - v) * tr[0] + u * v * br[0] + (1 - u) * v * bl[0]
                my = (1 - u) * (1 - v) * tl[1] + u * (1 - v) * tr[1] + u * v * br[1] + (1 - u) * v * bl[1]
                # 传感器原图上的 x 坐标是镜像前的 (1 - mx)
                raw_pts.append([[(1.0 - mx) * 640.0, my * 480.0]])
        raw_corners = np.array(raw_pts, dtype=np.float32)

        with patch("tools.camera_hub.app.detect_chessboard_corners", return_value=(True, raw_corners)):
            fb = app.process_live_frame_for_auto_capture(raw_frame, now=1.0)
            self.assertTrue(fb.aligned)

        # 抓拍落盘并校验原图未翻转（左白右黑保持不变）
        saved_path = app.capture_current_frame(auto_triggered=False)
        saved_img = cv2.imread(saved_path)
        self.assertIsNotNone(saved_img)
        self.assertGreater(int(saved_img[240, 50, 0]), 200)
        self.assertLess(int(saved_img[240, 590, 0]), 50)

        # 渲染预览画布冒烟通过
        canvas = app.renderer.render(1280, 760, (0, 0))
        self.assertEqual(canvas.shape, (760, 1280, 3))


if __name__ == "__main__":
    unittest.main()
