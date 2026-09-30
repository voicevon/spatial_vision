"""
测试通用视口管理器 (ViewportManager) 的缩放计算、坐标映射、动态 Resize 监听与渲染保真逻辑
"""
import unittest
import numpy as np
import cv2
import sys
from pathlib import Path

# 添加项目路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.ui.viewport_manager import ViewportManager, get_safe_screen_size


class TestViewportManager(unittest.TestCase):
    def test_safe_screen_size(self):
        """测试安全屏幕尺寸获取与兜底逻辑"""
        w, h = get_safe_screen_size()
        self.assertGreater(w, 600)
        self.assertGreater(h, 400)
        self.assertLessEqual(w, 3840)
        self.assertLessEqual(h, 2160)

    def test_viewport_geometry_calculation(self):
        """测试视口几何尺寸与缩放因子的正确性"""
        vp = ViewportManager(
            win_w=1280,
            win_h=720,
            top_bar_h=40,
            bottom_bar_h=60
        )
        vp.calculate_transform(1920, 1080)

        self.assertEqual(vp.win_w, 1280)
        self.assertEqual(vp.win_h, 720)
        self.assertEqual(vp.view_h, 620)
        self.assertAlmostEqual(vp.scale, 620.0 / 1080.0, places=4)
        
        # 缩放后图像高度应精确等于 620
        self.assertEqual(vp.fitted_h, 620)
        # 缩放后图像宽度应精确等于 int(round(1920 * (620/1080))) = 1102
        self.assertEqual(vp.fitted_w, int(round(1920 * (620.0 / 1080.0))))
        
        # 居中偏移验证
        self.assertEqual(vp.pad_y, 0) # 上下贴满视口
        self.assertEqual(vp.pad_x, (1280 - vp.fitted_w) // 2)

    def test_dynamic_resize_and_relayout(self):
        """测试用户拉伸或最大化窗口时的动态重排与缩放因子自适应"""
        vp = ViewportManager(
            win_w=1280,
            win_h=720,
            top_bar_h=50,
            bottom_bar_h=56
        )
        vp.calculate_transform(1920, 1080)
        initial_scale = vp.scale
        
        # 模拟用户点击“最大化”按钮，窗口放大到 1920x1080
        vp.update_window_size(1920, 1080)
        self.assertEqual(vp.win_w, 1920)
        self.assertEqual(vp.win_h, 1080)
        self.assertEqual(vp.view_h, 1080 - 50 - 56) # 974
        
        # 缩放比例必须随窗口变大而显著提升
        self.assertGreater(vp.scale, initial_scale)
        # 底部栏对应的 Y 坐标起点应变到 1080 - 56 = 1024
        bottom_bar_y = vp.win_h - vp.bottom_bar_h
        self.assertEqual(bottom_bar_y, 1024)

    def test_sync_window_size_with_opencv(self):
        """测试通过真实 OpenCV 窗口进行尺寸探测与自适应同步"""
        win_name = "test_viewport_resize_unit"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 900, 650)
        
        vp = ViewportManager(win_w=1280, win_h=720)
        # 首次同步应检测到窗口实际尺寸为 900x650
        changed = vp.sync_window_size(win_name)
        self.assertTrue(changed)
        self.assertEqual(vp.win_w, 900)
        self.assertEqual(vp.win_h, 650)
        
        # 再次同步无变化时应返回 False
        changed2 = vp.sync_window_size(win_name)
        self.assertFalse(changed2)
        
        cv2.destroyWindow(win_name)

    def test_coordinate_mapping_bidirectional(self):
        """测试窗口坐标与原图坐标的双向精准映射"""
        vp = ViewportManager(
            win_w=1280,
            win_h=720,
            top_bar_h=40,
            bottom_bar_h=60
        )
        vp.calculate_transform(1920, 1080)

        # 测试原图中心点 (960, 540)
        orig_cx, orig_cy = 960, 540
        win_cx, win_cy = vp.img_to_win_coords(orig_cx, orig_cy)

        # 窗口中心点 Y 坐标应为 top_bar_h + pad_y + 540 * scale = 40 + 0 + 310 = 350
        self.assertEqual(win_cy, 350)
        self.assertAlmostEqual(win_cx, 640, delta=1)

        # 反向映射回去应回到 (960, 540)
        mapped_orig_x, mapped_orig_y = vp.win_to_img_coords(win_cx, win_cy)
        self.assertIsNotNone(mapped_orig_x)
        self.assertIsNotNone(mapped_orig_y)
        self.assertAlmostEqual(mapped_orig_x, orig_cx, delta=1)
        self.assertAlmostEqual(mapped_orig_y, orig_cy, delta=1)

    def test_out_of_bounds_protection(self):
        """测试在视口外部（如点击顶部/底部 UI 栏或左右黑边）时的边界判定"""
        vp = ViewportManager(
            win_w=1280,
            win_h=720,
            top_bar_h=50,
            bottom_bar_h=50
        )
        vp.calculate_transform(1920, 1080)

        # 点击顶部栏 (100, 10) 处
        self.assertTrue(vp.is_in_top_bar(100, 10))
        self.assertFalse(vp.is_in_bottom_bar(100, 10))
        img_x, img_y = vp.win_to_img_coords(100, 10)
        self.assertIsNone(img_x)
        self.assertIsNone(img_y)

        # 点击底部栏 (100, 700) 处
        self.assertTrue(vp.is_in_bottom_bar(100, 700))
        self.assertFalse(vp.is_in_top_bar(100, 700))
        img_x, img_y = vp.win_to_img_coords(100, 700)
        self.assertIsNone(img_x)
        self.assertIsNone(img_y)

        # 点击视口左右黑边处 (pad_x 为大约 (1280-1102)//2 = 89，所以在 x=10 处是黑边)
        self.assertLess(10, vp.pad_x)
        img_x, img_y = vp.win_to_img_coords(10, 300)
        self.assertIsNone(img_x)
        self.assertIsNone(img_y)

    def test_render_viewport(self):
        """测试视口渲染输出与画布尺寸对齐"""
        vp = ViewportManager(
            win_w=1200,
            win_h=700,
            top_bar_h=40,
            bottom_bar_h=50
        )

        test_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # 画一个白色矩形在中心
        test_img[400:600, 800:1200] = 255

        canvas = np.zeros((700, 1200, 3), dtype=np.uint8)
        rendered = vp.render_viewport(canvas, test_img)
        self.assertEqual(rendered.shape, (700, 1200, 3))
        
        # 检查中心区域确实被成功贴入非零像素
        center_color = rendered[350, 600]
        self.assertTrue(np.all(center_color == 255))
        
        # 检查视口外的顶部工具栏区域仍然保持黑色 (0, 0, 0)
        top_bar_sample = rendered[20, 600]
        self.assertTrue(np.all(top_bar_sample == 0))

    def test_mouse_wheel_zoom_and_anchor(self):
        """测试鼠标滚轮缩放与以鼠标所在像素为锚点的坐标稳定性"""
        vp = ViewportManager(win_w=1280, win_h=720, top_bar_h=50, bottom_bar_h=50)
        vp.calculate_transform(1920, 1080)

        mouse_win_x, mouse_win_y = 640, 360
        # 初始 1.0x 状态下的映射坐标
        orig_img_x, orig_img_y = vp.win_to_img_coords(mouse_win_x, mouse_win_y)
        self.assertIsNotNone(orig_img_x)
        self.assertIsNotNone(orig_img_y)

        # 模拟滚轮向前滚动放大 (flags > 0)
        changed = vp.handle_mouse_wheel(mouse_win_x, mouse_win_y, flags=7864320)
        self.assertTrue(changed)
        self.assertGreater(vp.user_zoom, 1.0)

        # 验证缩放后，同一鼠标屏幕位置下的原图像素坐标必须保持一致 (误差 < 0.1px)
        new_img_x, new_img_y = vp.win_to_img_coords(mouse_win_x, mouse_win_y)
        self.assertAlmostEqual(orig_img_x, new_img_x, delta=0.1)
        self.assertAlmostEqual(orig_img_y, new_img_y, delta=0.1)

        # 多次向上滚动至最大倍率上限 8.0x
        for _ in range(25):
            vp.handle_mouse_wheel(mouse_win_x, mouse_win_y, flags=7864320)
        self.assertAlmostEqual(vp.user_zoom, 8.0, places=2)

        # 再次滚动不能突破 8.0
        vp.handle_mouse_wheel(mouse_win_x, mouse_win_y, flags=7864320)
        self.assertLessEqual(vp.user_zoom, 8.0)

        # 连续向下滚动缩小至 1.0x
        for _ in range(30):
            vp.handle_mouse_wheel(mouse_win_x, mouse_win_y, flags=-7864320)
        self.assertAlmostEqual(vp.user_zoom, 1.0, places=2)
        self.assertEqual(vp.pan_x, 0.0)
        self.assertEqual(vp.pan_y, 0.0)

    def test_middle_button_pan_and_reset(self):
        """测试中键拖拽平移漫游与一键复位"""
        vp = ViewportManager(win_w=1280, win_h=720, top_bar_h=50, bottom_bar_h=50)
        vp.calculate_transform(1920, 1080)

        # 放大到 2.0x
        vp.user_zoom = 2.0
        vp._constrain_pan()

        # 开始拖拽
        vp.start_pan(400, 300)
        self.assertTrue(vp.is_panning)

        # 鼠标移动 50px, -30px
        moved = vp.update_pan(450, 270)
        self.assertTrue(moved)
        self.assertEqual(vp.pan_x, 50.0)
        self.assertEqual(vp.pan_y, -30.0)

        # 结束拖拽
        vp.end_pan()
        self.assertFalse(vp.is_panning)

        # 执行视口复位
        reset_ok = vp.reset_zoom()
        self.assertTrue(reset_ok)
        self.assertEqual(vp.user_zoom, 1.0)
        self.assertEqual(vp.pan_x, 0.0)
        self.assertEqual(vp.pan_y, 0.0)


if __name__ == '__main__':
    unittest.main()

