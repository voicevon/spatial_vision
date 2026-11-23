# -*- coding: utf-8 -*-
"""
SceneConfigGuard 单元测试与并发防竞态验证
===========================================
验证内容：
1. 空间场景初始加载与缺失兜底；
2. update_frames 原子更新坐标系树并保全 rois 空间物件；
3. update_rois 原子更新 3D ROI 空间物件并保全 frames 坐标系树；
4. 多线程并发交替写入 frames 与 rois，验证原子性与数据无损。
"""

import os
import shutil
import tempfile
import threading
import unittest

from src.workspace.scene_config_guard import SceneConfigGuard


class TestSceneConfigGuard(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.scene_path = os.path.join(self.test_dir, "spatial_scene.yaml")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_load_non_existent_file(self):
        """测试加载不存在的文件返回标准默认骨架"""
        data = SceneConfigGuard.load_scene(self.scene_path)
        self.assertEqual(data["version"], "2.0")
        self.assertEqual(data["active_frame_id"], "world")
        self.assertEqual(data["frames"], [])
        self.assertEqual(data["rois"], [])

    def test_isolated_updates_preserve_counterparts(self):
        """测试 update_frames 与 update_rois 相互保全对方节数据"""
        # 1. 写入 frames
        frames = [
            {"frame_id": "world", "name": "World", "type": "world", "status": "calibrated"},
            {"frame_id": "conveyor", "name": "Belt", "type": "fixed_transform", "status": "calibrated"},
        ]
        ok = SceneConfigGuard.update_frames(self.scene_path, frames, active_frame_id="conveyor", workspace_id="ws_01")
        self.assertTrue(ok)

        # 检查仅 frames 写入，rois 仍为空
        scene = SceneConfigGuard.load_scene(self.scene_path)
        self.assertEqual(len(scene["frames"]), 2)
        self.assertEqual(scene["active_frame_id"], "conveyor")
        self.assertEqual(scene["rois"], [])

        # 2. 写入 rois
        rois = [
            {"roi_id": "r1", "name": "Inlet", "frame_id": "conveyor", "category": "feeder", "role": "source"},
        ]
        ok = SceneConfigGuard.update_rois(self.scene_path, rois, workspace_id="ws_01")
        self.assertTrue(ok)

        # 检查 frames 被完全保留，未被覆盖！
        scene_after = SceneConfigGuard.load_scene(self.scene_path)
        self.assertEqual(len(scene_after["frames"]), 2)
        self.assertEqual(scene_after["active_frame_id"], "conveyor")
        self.assertEqual(len(scene_after["rois"]), 1)
        self.assertEqual(scene_after["rois"][0]["roi_id"], "r1")

    def test_concurrent_interleaved_updates(self):
        """测试多线程交替并发保存 frames 和 rois，验证无数据丢失与竞态安全"""
        errors = []

        def worker_frames():
            for i in range(20):
                f_list = [
                    {"frame_id": "world", "type": "world"},
                    {"frame_id": f"sub_{i}", "type": "fixed_transform"},
                ]
                ok = SceneConfigGuard.update_frames(self.scene_path, f_list, active_frame_id=f"sub_{i}")
                if not ok:
                    errors.append(f"worker_frames failed at {i}")

        def worker_rois():
            for i in range(20):
                r_list = [
                    {"roi_id": f"roi_{i}", "frame_id": "world", "category": "feeder"},
                ]
                ok = SceneConfigGuard.update_rois(self.scene_path, r_list)
                if not ok:
                    errors.append(f"worker_rois failed at {i}")

        t1 = threading.Thread(target=worker_frames)
        t2 = threading.Thread(target=worker_rois)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(errors, [])
        final_scene = SceneConfigGuard.load_scene(self.scene_path)
        self.assertEqual(len(final_scene["frames"]), 2)
        self.assertEqual(len(final_scene["rois"]), 1)


if __name__ == "__main__":
    unittest.main()
