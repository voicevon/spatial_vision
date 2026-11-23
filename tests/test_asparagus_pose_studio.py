#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
芦笋位姿工作室 (Asparagus Pose Studio) 单元测试
===============================================
验证数据IO配对与扫描、G-code 导出、UI 渲染器度量与主控制器行为。
"""

import os
import sys
import time
import tempfile
import unittest

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.asparagus_pose_studio.data_io import (
    find_depth_pair,
    scan_samples,
    load_system_config,
)
from tools.asparagus_pose_studio.renderer import AsparagusPoseStudioRenderer
from tools.asparagus_pose_studio.app import AsparagusPoseStudioApp


def _create_dummy_samples(tmpdir):
    """构造测试样本集：一组成对快照、一张独立彩色照、一张派生可视化图"""
    color1 = os.path.join(tmpdir, "color_20260920_120000.png")
    depth1 = os.path.join(tmpdir, "depth_raw_20260920_120000.npy")
    photo2 = os.path.join(tmpdir, "sample_test.png")
    vis = os.path.join(tmpdir, "depth_vis_preview.png")

    cv2.imwrite(color1, np.zeros((48, 64, 3), dtype=np.uint8))
    cv2.imwrite(photo2, np.zeros((48, 64, 3), dtype=np.uint8))
    cv2.imwrite(vis, np.zeros((48, 64, 3), dtype=np.uint8))
    np.save(depth1, np.zeros((48, 64), dtype=np.uint16))

    old = time.time() - 50
    os.utime(color1, (old, old))
    os.utime(depth1, (old, old))
    return color1, depth1, photo2, vis


class TestDataIO(unittest.TestCase):

    def test_find_depth_pair_and_scan(self):
        """测试成对深度查找与样本过滤扫描"""
        with tempfile.TemporaryDirectory() as tmpdir:
            color1, depth1, photo2, vis = _create_dummy_samples(tmpdir)

            # 配对规则测试
            self.assertEqual(find_depth_pair(color1), depth1)
            self.assertIsNone(find_depth_pair(photo2))

            # 扫描测试 (跳过 depth_vis_，最新在前)
            samples = scan_samples(tmpdir)
            self.assertEqual(len(samples), 2)
            names = [s["name"] for s in samples]
            self.assertIn("sample_test.png", names)
            self.assertIn("color_20260920_120000.png", names)
            self.assertNotIn("depth_vis_preview.png", names)
            # photo2 较新，应排第一
            self.assertEqual(samples[0]["name"], "sample_test.png")


    def test_load_system_config_fallback(self):
        """测试加载不存在配置文件的缺省降级"""
        cfg = load_system_config("non_existent_config.yaml")
        self.assertIn("safe_z", cfg)
        self.assertEqual(cfg["safe_z"], 80.0)


class TestRenderer(unittest.TestCase):

    def test_compute_metrics_and_panels(self):
        """测试度量计算与三栏面板物理分割"""
        m = AsparagusPoseStudioRenderer.compute_metrics(1280)
        self.assertEqual(m["s"], 1.0)
        self.assertGreater(m["list_w"], 0)
        self.assertGreater(m["right_w"], 0)

        list_p, img_p, right_p = AsparagusPoseStudioRenderer.compute_panels(1280, 800, m)
        self.assertEqual(list_p[0], m["L"])
        self.assertLess(list_p[2], img_p[0])
        self.assertLess(img_p[2], right_p[0])

    def test_topbar_layout_and_no_duplicate_dropdowns(self):
        """测试第一排工具栏严格顺序排布且不存在重复下拉框"""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_dummy_samples(tmpdir)
            cfg_file = os.path.join(tmpdir, "settings.json")
            app = AsparagusPoseStudioApp(sample_dir=tmpdir, settings_file=cfg_file)
            canvas = np.zeros((800, 1280, 3), dtype=np.uint8)
            res = AsparagusPoseStudioRenderer.render_scene(canvas, app)
            buttons = res[0]

            ws_dd_buttons = [b for b in buttons if b[1] == ("toggle_dd", "WORKSPACE_DROPDOWN")]
            pipe_dd_buttons = [b for b in buttons if b[1] == ("toggle_dd", "PIPELINE_DROPDOWN")]
            exit_buttons = [b for b in buttons if b[1] == ("btn", "退出 [X]")]

            self.assertEqual(len(ws_dd_buttons), 1, "工位地图下拉按钮必须且仅有1个")
            self.assertEqual(len(pipe_dd_buttons), 1, "算法路线下拉按钮必须且仅有1个")
            self.assertEqual(len(exit_buttons), 1, "退出按钮必须且仅有1个")

            # 验证横向位置顺序: 地图下拉.x < 算法下拉.x < 退出.x
            ws_rect = ws_dd_buttons[0][0]
            pipe_rect = pipe_dd_buttons[0][0]
            exit_rect = exit_buttons[0][0]
            self.assertLess(ws_rect[0], pipe_rect[0], "地图下拉应在算法下拉左侧")
            self.assertLess(pipe_rect[2], exit_rect[0], "算法下拉应在退出按钮左侧")


class TestApp(unittest.TestCase):

    def test_app_lifecycle_and_render(self):
        """测试主应用初始化、样本切换与画布渲染"""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_dummy_samples(tmpdir)
            cfg_file = os.path.join(tmpdir, "settings.json")
            app = AsparagusPoseStudioApp(sample_dir=tmpdir, settings_file=cfg_file)
            self.assertEqual(len(app.samples), 2)
            self.assertEqual(app.sel_idx, 0)
            self.assertEqual(app.mode, "2d")

            # 首次渲染
            canvas = app.render()
            self.assertIsInstance(canvas, np.ndarray)
            self.assertEqual(canvas.shape[0], app.win_mgr.canvas_h)
            self.assertEqual(canvas.shape[1], app.win_mgr.canvas_w)

            # 切换到成对深度快照
            app._select_sample(1)
            self.assertEqual(app.sel_idx, 1)
            self.assertEqual(app.mode, "3d")

            # 切换算法流水线
            app.switch_pipeline("ridge_tracing")
            self.assertEqual(app.pipeline_key, "ridge_tracing")

            # 快捷键退出测试
            app.on_key(ord("x"))
            self.assertFalse(app._running)

    def test_pipeline_and_sample_persistence(self):
        """测试算法路线与选定样本的跨次持久化保存与自动恢复"""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_dir = os.path.join(tmpdir, "samples")
            os.makedirs(sample_dir, exist_ok=True)
            _create_dummy_samples(sample_dir)
            settings_path = os.path.join(tmpdir, "gui_settings.json")

            # 第一次启动: 默认选中首项 (sample_test.png)
            app1 = AsparagusPoseStudioApp(sample_dir=sample_dir, settings_file=settings_path)
            self.assertEqual(app1.samples[app1.sel_idx]["name"], "sample_test.png")

            # 切换到第二个样本 (color_20260920_120000.png)
            app1._select_sample(1)
            self.assertEqual(app1.samples[app1.sel_idx]["name"], "color_20260920_120000.png")

            # 切换算法路线
            available_pipes = [k for k, _ in app1.pipeline_options]
            target_pipe = available_pipes[-1] if len(available_pipes) > 1 else "ridge_tracing"
            app1.switch_pipeline(target_pipe)

            # 第二次启动: 读取同一 settings_file
            app2 = AsparagusPoseStudioApp(sample_dir=sample_dir, settings_file=settings_path)
            # 验证算法自动恢复
            self.assertEqual(app2.pipeline_key, target_pipe)
            # 验证自动定位并选中了上次选中的样本
            self.assertEqual(app2.sel_idx, 1)
            self.assertEqual(app2.samples[app2.sel_idx]["name"], "color_20260920_120000.png")

    def test_auto_analysis_and_step_pill_switch(self):
        """测试加载样本和切换算法时自动触发分析，以及步骤药丸切换"""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_dir = os.path.join(tmpdir, "samples")
            os.makedirs(sample_dir, exist_ok=True)
            _create_dummy_samples(sample_dir)
            settings_path = os.path.join(tmpdir, "gui_settings.json")

            app = AsparagusPoseStudioApp(sample_dir=sample_dir, settings_file=settings_path)
            # 初始启动已自动触发分析
            self.assertIsNotNone(app.pipeline_result)
            self.assertIsNotNone(app.pipeline_result.step_snapshots)
            self.assertIn("stage3_poses", app.pipeline_result.step_snapshots)

            # 切换算法路线，自动重新分析
            app.switch_pipeline("ridge_tracing")
            self.assertIsNotNone(app.pipeline_result)
            self.assertIsNotNone(app.pipeline_result.step_snapshots)
            self.assertIn("stage3_poses", app.pipeline_result.step_snapshots)

            # 步骤切换
            app._select_step("stage1_fg")
            self.assertEqual(app.active_step_key, "stage1_fg")

            # 切换至算法 C1 (骨架细化法)
            app.switch_pipeline("skeleton_thinning")
            self.assertEqual(app.pipeline_key, "skeleton_thinning")
            self.assertIsNotNone(app.pipeline_result)
            self.assertIn("stage3_skeleton", app.pipeline_result.step_snapshots)

            # 切换至算法 C2 (Frangi管状滤波法)
            app.switch_pipeline("frangi_vesselness")
            self.assertEqual(app.pipeline_key, "frangi_vesselness")
            self.assertIsNotNone(app.pipeline_result)
            self.assertIn("stage3_vesselness", app.pipeline_result.step_snapshots)

            # 模拟鼠标 Hover 在第一枚药丸上，验证 Tooltip 气泡提示框渲染无异常
            app.mouse_pos = (app.viewport.win_w // 3, 58)
            canvas_hover = app.render()
            self.assertIsNotNone(canvas_hover)

            # 验证顶部双排布局 header_h 为 86
            self.assertEqual(app.viewport.top_bar_h, 86)


if __name__ == "__main__":
    unittest.main()

