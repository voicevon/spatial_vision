# -*- coding: utf-8 -*-
"""
相机内参标定求解器与三大图集架构单元测试
======================================
基于标准库 unittest 验证：
1. 单目棋盘格内参标定求解 (OpenCV calibrateCamera + RMSE)
2. 工位沙盒内参持久化与加载 (workspace.save_camera_intrinsics / load_camera_intrinsics)
3. Workspace Hub 状态机四大 Tab 顺序与三大独立图集 (intrinsics / calibration / production)
4. 相册抓拍分类归档与命中测试
"""

import os
import shutil
import tempfile
import unittest
import cv2
import numpy as np

from src.workspace.workspace_manager import WorkspaceManager, Workspace
from src.calibration.solvers.camera_intrinsics_calibrator import (
    calibrate_camera_from_images,
    calibrate_workspace_intrinsics,
)
from tools.workspace_hub.hub_state import HubState
from tools.workspace_hub.hub_renderer import HubRenderer, grid_hit_test
from tools.workspace_hub.hub_hit_tester import HubHitTester


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

        filepath = os.path.join(output_dir, f"intr_{i:04d}.png")
        cv2.imwrite(filepath, warped)
        image_paths.append(filepath)

    return image_paths


class TestIntrinsicsCalibratorAndGallery(unittest.TestCase):
    """测试相机内参标定求解与三大图集管理"""

    def test_camera_intrinsics_solver(self):
        """测试单目相机棋盘格内参标定求解器"""
        temp_dir = tempfile.mkdtemp(prefix="test_intr_solver_")
        try:
            paths = generate_synthetic_chessboard_images(temp_dir, num_images=5, pattern_size=(9, 6), square_size_mm=25.0)
            self.assertEqual(len(paths), 5)

            # 运行求解
            res = calibrate_camera_from_images(paths, pattern_size=(9, 6), square_size_mm=25.0)
            self.assertTrue(res.success, f"标定失败: {res.message}")
            self.assertGreaterEqual(res.valid_images_count, 3)
            self.assertGreater(res.fx, 200.0)
            self.assertGreater(res.fy, 200.0)
            self.assertGreater(res.cx, 100.0)
            self.assertGreater(res.cy, 100.0)
            self.assertLess(res.rmse, 2.0)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_workspace_intrinsics_save_and_load(self):
        """测试工位沙盒内参的持久化与读取"""
        temp_root = tempfile.mkdtemp(prefix="test_ws_intr_")
        try:
            mgr = WorkspaceManager(workspaces_dir=temp_root)
            ws = mgr.create_workspace("ws_intr_test", "内参测试工位")

            # 验证三图集与内参路径
            self.assertTrue(os.path.isdir(ws.intrinsics_raw_images_dir))
            self.assertTrue(os.path.isdir(ws.calib_raw_images_dir))
            self.assertTrue(os.path.isdir(ws.prod_raw_images_dir))
            self.assertTrue(ws.camera_intrinsics_path.endswith("camera_intrinsics.yaml"))

            # 初始状态未标定
            initial_intr = ws.load_camera_intrinsics()
            self.assertIsNone(initial_intr)

            # 生成棋盘格照片并进行工位标定
            generate_synthetic_chessboard_images(ws.intrinsics_raw_images_dir, num_images=4)
            ws.refresh_stats()
            self.assertEqual(ws.intrinsics_image_count, 4)

            # 自动标定并写穿至工位沙盒
            calib_res = calibrate_workspace_intrinsics(ws, pattern_size=(9, 6), square_size_mm=25.0, save_to_workspace=True)
            self.assertTrue(calib_res.success, f"标定失败: {calib_res.message}")

            # 验证工位内参持久化文件已创建并加载正确
            self.assertTrue(os.path.exists(ws.camera_intrinsics_path))
            saved_intr = ws.load_camera_intrinsics()
            self.assertIsNotNone(saved_intr)
            self.assertTrue(saved_intr["calibrated"])
            self.assertEqual(saved_intr["rmse"], calib_res.rmse)
            self.assertAlmostEqual(saved_intr["fx"], calib_res.fx, places=4)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_hub_state_four_tabs_and_three_galleries(self):
        """测试 HubState 树形组织：工位 -> 内参 -> 三大独立图集"""
        temp_root = tempfile.mkdtemp(prefix="test_hub_tabs_")
        try:
            mgr = WorkspaceManager(workspaces_dir=temp_root)
            ws = mgr.create_workspace("ws_alpha", "测试工位Alpha")

            hub_state = HubState(workspace_mgr=mgr)
            self.assertEqual(hub_state.WS_TAB_ORDER, (
                HubState.TAB_REPORT,
                HubState.TAB_INTRINSICS_IMAGES,
                HubState.TAB_CALIB_IMAGES,
                HubState.TAB_PROD_IMAGES,
            ))

            # 验证 Tab 标签
            tab_keys = [t.key for t in hub_state.tab_bar.items]
            self.assertIn(HubState.TAB_INTRINSICS_IMAGES, tab_keys)
            self.assertIn(HubState.TAB_CALIB_IMAGES, tab_keys)
            self.assertIn(HubState.TAB_PROD_IMAGES, tab_keys)

            # 向三个图集各写入一张图片
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            hub_state.gallery.save_capture_frame(dummy_img, purpose="intrinsics")
            hub_state.gallery.save_capture_frame(dummy_img, purpose="calibration")
            hub_state.gallery.save_capture_frame(dummy_img, purpose="production")

            self.assertEqual(len(hub_state.gallery.intrinsics_images), 1)
            self.assertEqual(len(hub_state.gallery.current_images), 1)
            self.assertEqual(len(hub_state.gallery.prod_images), 1)

            # 切换到相机内参页签
            hub_state.set_tab(HubState.TAB_INTRINSICS_IMAGES)
            imgs, idx, name = hub_state.gallery.get_active_images_and_index()
            self.assertEqual(name, "内参图集")
            self.assertEqual(len(imgs), 1)

            # 切换到生产相册页签
            hub_state.set_tab(HubState.TAB_PROD_IMAGES)
            imgs, idx, name = hub_state.gallery.get_active_images_and_index()
            self.assertEqual(name, "生产图集")
            self.assertEqual(len(imgs), 1)

            # 删除内参图片
            hub_state.set_tab(HubState.TAB_INTRINSICS_IMAGES)
            del_ok = hub_state.gallery.delete_selected_image()
            self.assertTrue(del_ok)
            self.assertEqual(len(hub_state.gallery.intrinsics_images), 0)
            # 外参和生产图集不受任何影响
            self.assertEqual(len(hub_state.gallery.current_images), 1)
            self.assertEqual(len(hub_state.gallery.prod_images), 1)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_hub_hit_tester_and_intrinsics_calib_button(self):
        """测试命中检测：内参标定按钮与带偏移量的卡片点击"""
        temp_root = tempfile.mkdtemp(prefix="test_hit_")
        try:
            mgr = WorkspaceManager(workspaces_dir=temp_root)
            mgr.create_workspace("ws_hit", "命中测试工位")

            hub_state = HubState(workspace_mgr=mgr)
            hub_state.set_tab(HubState.TAB_INTRINSICS_IMAGES)

            renderer = HubRenderer()
            hit_tester = HubHitTester(renderer)

            # 1. 点击 [标定内参] 按钮 (x: 830~950, y: 74~110)
            action = hit_tester.hit_test(850, 85, hub_state)
            self.assertEqual(action, "run_intrinsics_calib")

            # 2. grid_hit_test 带 y_offset 验证
            self.assertEqual(grid_hit_test(360, 90, y_offset=0), 0)
            # 当 y_offset=74 时，y=90 落在顶部看板区，不在卡片网格内
            self.assertIsNone(grid_hit_test(360, 90, y_offset=74))
            # 当 y_offset=74 时，y=170 准确命中第 0 个卡片
            self.assertEqual(grid_hit_test(360, 170, y_offset=74), 0)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
