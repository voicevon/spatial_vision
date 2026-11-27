# -*- coding: utf-8 -*-
"""
相机内参标定求解器与两大图集架构单元测试
========================================
1. 单目棋盘格内参标定求解 (OpenCV calibrateCamera + RMSE + 逐帧误差)
2. 全局相机档案内参持久化与读取 (calibrate_camera_profile)
3. Workspace Hub 状态机三大 Tab 顺序与两大业务图集 (calibration / production)
"""

import os
import shutil
import tempfile
import unittest
import cv2
import numpy as np

from src.cameras.camera_registry import CameraRegistry
from src.workspace.workspace_manager import WorkspaceManager
from src.calibration.solvers.camera_intrinsics_calibrator import (
    calibrate_camera_from_images,
    calibrate_camera_profile,
)
from tools.workspace_hub.hub_state import HubState


def generate_synthetic_chessboard_images(
    output_dir: str,
    num_images: int = 5,
    pattern_size: tuple[int, int] = (9, 6),
    square_size_mm: float = 25.0,
    img_size: tuple[int, int] = (640, 480),
) -> list[str]:
    """生成具有微小透视与旋转变化的棋盘格图像用于内参标定测试"""
    os.makedirs(output_dir, exist_ok=True)
    w_out, h_out = img_size
    cols, rows = pattern_size
    square_px = 32
    board_w = (cols + 1) * square_px
    board_h = (rows + 1) * square_px

    margin = 50
    canvas_w = board_w + margin * 2
    canvas_h = board_h + margin * 2
    base_img = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 1:
                x1 = margin + c * square_px
                y1 = margin + r * square_px
                base_img[y1:y1 + square_px, x1:x1 + square_px] = (0, 0, 0)

    src_pts = np.float32([
        [0, 0],
        [canvas_w, 0],
        [canvas_w, canvas_h],
        [0, canvas_h]
    ])

    image_paths = []
    for i in range(num_images):
        dx = (i - num_images // 2) * 12
        dy = (i % 2 * 2 - 1) * 8
        scale = 0.85 + (i * 0.04)

        cx, cy = w_out / 2.0 + dx, h_out / 2.0 + dy
        half_w = (canvas_w * scale) / 2.0
        half_h = (canvas_h * scale) / 2.0

        skew = (i - 2) * 6
        dst_pts = np.float32([
            [cx - half_w + skew, cy - half_h],
            [cx + half_w, cy - half_h + skew],
            [cx + half_w - skew, cy + half_h],
            [cx - half_w, cy + half_h - skew]
        ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(base_img, M, (w_out, h_out), borderValue=(255, 255, 255))

        filepath = os.path.join(output_dir, f"intr_{i + 1:04d}.png")
        cv2.imwrite(filepath, warped)
        image_paths.append(filepath)

    return image_paths


class TestIntrinsicsCalibratorAndGallery(unittest.TestCase):
    """测试相机内参标定求解与两大业务图集管理"""

    def test_camera_intrinsics_solver(self):
        """测试单目相机棋盘格内参标定求解器及逐帧误差输出"""
        temp_dir = tempfile.mkdtemp(prefix="test_intr_solver_")
        try:
            paths = generate_synthetic_chessboard_images(temp_dir, num_images=5, pattern_size=(9, 6), square_size_mm=25.0)
            self.assertEqual(len(paths), 5)

            res = calibrate_camera_from_images(paths, pattern_size=(9, 6), square_size_mm=25.0)
            self.assertTrue(res.success, f"标定失败: {res.message}")
            self.assertGreaterEqual(res.valid_images_count, 3)
            self.assertEqual(len(res.per_image_errors), 5)
            self.assertTrue(all(e["detected"] and e["error_px"] is not None for e in res.per_image_errors))
            self.assertGreater(res.fx, 200.0)
            self.assertGreater(res.fy, 200.0)
            self.assertLess(res.rmse, 2.0)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_camera_profile_intrinsics_save_and_load(self):
        """测试全局相机档案棋盘格标定与持久化读取"""
        temp_root = tempfile.mkdtemp(prefix="test_cam_prof_intr_")
        try:
            reg = CameraRegistry(cameras_root=temp_root)
            cam = reg.register_camera("测试D435", camera_type="realsense", hardware_serial="SN_INTR_01")

            self.assertIsNone(cam.load_intrinsics("640x480"))
            generate_synthetic_chessboard_images(cam.raw_images_dir("640x480"), num_images=4)
            self.assertEqual(len(cam.list_images("640x480")), 4)

            calib_res = calibrate_camera_profile(cam, "640x480", pattern_size=(9, 6), square_size_mm=25.0)
            self.assertTrue(calib_res.success, f"标定失败: {calib_res.message}")

            saved = cam.load_intrinsics("640x480")
            self.assertIsNotNone(saved)
            self.assertAlmostEqual(saved.rmse, calib_res.rmse, places=4)
            self.assertAlmostEqual(saved.fx, calib_res.fx, places=4)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_hub_state_three_tabs_and_two_galleries(self):
        """测试 HubState 树形组织：大盘看板 + 外参/生产两大独立图集 (不含相机内参页签)"""
        temp_root = tempfile.mkdtemp(prefix="test_hub_tabs_")
        try:
            mgr = WorkspaceManager(workspaces_dir=temp_root)
            mgr.create_workspace("ws_alpha", "测试工位Alpha")

            hub_state = HubState(workspace_mgr=mgr)
            self.assertEqual(hub_state.WS_TAB_ORDER, (
                HubState.TAB_REPORT,
                HubState.TAB_CALIB_IMAGES,
                HubState.TAB_PROD_IMAGES,
            ))

            tab_keys = [t.key for t in hub_state.tab_bar.items]
            self.assertEqual(tab_keys, [HubState.TAB_REPORT, HubState.TAB_CALIB_IMAGES, HubState.TAB_PROD_IMAGES])

            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            hub_state.gallery.save_capture_frame(dummy_img, purpose="calibration")
            hub_state.gallery.save_capture_frame(dummy_img, purpose="production")

            self.assertEqual(len(hub_state.gallery.calib_images), 1)
            self.assertEqual(len(hub_state.gallery.prod_images), 1)

            hub_state.set_tab(HubState.TAB_PROD_IMAGES)
            imgs, idx, name = hub_state.gallery.get_active_images_and_index()
            self.assertEqual(name, "生产图集")
            self.assertEqual(len(imgs), 1)

            del_ok = hub_state.gallery.delete_selected_image()
            self.assertTrue(del_ok)
            self.assertEqual(len(hub_state.gallery.prod_images), 0)
            self.assertEqual(len(hub_state.gallery.calib_images), 1)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
