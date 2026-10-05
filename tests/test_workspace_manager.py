"""
单元测试：工位工作空间 (Workspace) 管理与双业务沙盒系统 (tests/test_workspace_manager.py)
================================================================================
验证：
1. 工位创建、目录生成 (calibration/ 与 production/)、白名单与元数据生成
2. 顶层核心资产 (tags_map.yaml, tag_whitelist.yaml) 隔离与读取
3. 工位当前选择切换 (运行时内存态, 无落盘标记)
4. 工位克隆与沙盒独立性 (标定/生产图片双克隆)
5. 工位发布至全局生产环境 (config/tags_map.yaml 与 config.yaml)
6. 工位物理删除
7. 双用途采图路径路由测试 (calibration vs production)
"""

import os
import sys
import shutil
import tempfile
import unittest
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.workspace.workspace_manager import WorkspaceManager, Workspace


class TestWorkspaceManager(unittest.TestCase):

    def setUp(self):
        # 当前工位为类级运行时状态, 用例间显式重置防串扰
        WorkspaceManager._current_ws_id = None
        self.temp_dir = tempfile.mkdtemp()
        self.workspaces_dir = os.path.join(self.temp_dir, "workspaces")
        self.config_path = os.path.join(self.temp_dir, "config.yaml")

        # 写入初始空 config.yaml
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump({"calibration": {}}, f)

        self.mgr = WorkspaceManager(
            workspaces_dir=self.workspaces_dir,
            config_path=self.config_path
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_and_list_workspace(self):
        """测试工位创建、目录层级与列表检索"""
        ws1 = self.mgr.create_workspace(alias="bench_a", description="工位A测试")
        self.assertTrue(os.path.isdir(ws1.workspace_dir))
        self.assertTrue(os.path.isdir(ws1.calibration_dir))
        self.assertTrue(os.path.isdir(ws1.calib_raw_images_dir))
        self.assertTrue(os.path.isdir(ws1.production_dir))
        self.assertTrue(os.path.isdir(ws1.prod_raw_images_dir))
        self.assertTrue(os.path.exists(ws1.meta_path))
        self.assertTrue(os.path.exists(ws1.whitelist_path))
        self.assertIn("bench_a", ws1.workspace_id)

        # 验证自动获取工况工位
        self.assertEqual(self.mgr.get_current_workspace_id(), ws1.workspace_id)

        # 创建第二个工位
        ws2 = self.mgr.create_workspace(alias="bench_b", description="工位B测试")
        workspaces = self.mgr.list_workspaces()
        self.assertEqual(len(workspaces), 2)
        ws_ids = [w.workspace_id for w in workspaces]
        self.assertIn(ws1.workspace_id, ws_ids)
        self.assertIn(ws2.workspace_id, ws_ids)

    def test_switch_current_workspace(self):
        """测试当前工位切换 (运行时内存态)"""
        ws1 = self.mgr.create_workspace(alias="w1")
        ws2 = self.mgr.create_workspace(alias="w2")

        self.assertEqual(self.mgr.get_current_workspace_id(), ws2.workspace_id)

        # 切换回 ws1
        ok = self.mgr.set_current_workspace(ws1.workspace_id)
        self.assertTrue(ok)
        self.assertEqual(self.mgr.get_current_workspace_id(), ws1.workspace_id)
        self.assertEqual(self.mgr.get_current_workspace().workspace_id, ws1.workspace_id)

        # 新建工位不隐式翻转当前选择 (显式选择保持不变)
        self.mgr.create_workspace(alias="w3")
        self.assertEqual(self.mgr.get_current_workspace_id(), ws1.workspace_id)

    def test_clone_workspace_independence(self):
        """测试工位克隆与标定/生产沙盒独立性"""
        ws1 = self.mgr.create_workspace(alias="orig")

        # 写入一张标定模拟图片与一张生产模拟图片
        calib_img = os.path.join(ws1.calib_raw_images_dir, "view_0001.png")
        prod_img = os.path.join(ws1.prod_raw_images_dir, "view_0001.png")
        with open(calib_img, "wb") as f:
            f.write(b"calib_png")
        with open(prod_img, "wb") as f:
            f.write(b"prod_png")

        # 克隆至新工位
        cloned = self.mgr.clone_workspace(ws1.workspace_id, new_alias="cloned")
        self.assertIsNotNone(cloned)
        self.assertTrue(os.path.exists(os.path.join(cloned.calib_raw_images_dir, "view_0001.png")))
        self.assertTrue(os.path.exists(os.path.join(cloned.prod_raw_images_dir, "view_0001.png")))

        # 修改原始工位的图片，不应影响克隆工位
        os.remove(calib_img)
        self.assertFalse(os.path.exists(calib_img))
        self.assertTrue(os.path.exists(os.path.join(cloned.calib_raw_images_dir, "view_0001.png")))

    def test_workspace_map_sandbox(self):
        """测试工位沙盒地图自包含存储与独立性 (无全局发布)"""
        ws = self.mgr.create_workspace(alias="sandbox_test")

        # 写入工位自身顶层地图 (tags_map.yaml)
        map_content = {
            "marker_size_mm": 50.0,
            "rmse_reprojection_px": 0.158,
            "tags": {0: {"position_mm": [0, 0, 0]}}
        }
        with open(ws.map_path, "w", encoding="utf-8") as f:
            yaml.dump(map_content, f)

        # 验证工位自身地图存在并可读取
        self.assertTrue(os.path.exists(ws.map_path))
        with open(ws.map_path, "r", encoding="utf-8") as f:
            loaded_m = yaml.safe_load(f)
        self.assertEqual(loaded_m["rmse_reprojection_px"], 0.158)

        # 验证刷新状态后工位自感知 ba_solved
        ws.refresh_stats()
        self.assertTrue(ws.ba_solved)

    def test_delete_workspace(self):
        """测试工位自由物理删除"""
        ws1 = self.mgr.create_workspace(alias="del_target")
        self.assertTrue(os.path.exists(ws1.workspace_dir))

        ok, msg = self.mgr.delete_workspace(ws1.workspace_id)
        self.assertTrue(ok)
        self.assertFalse(os.path.exists(ws1.workspace_dir))

    def test_dual_purpose_routing(self):
        """测试双用途 (calibration vs production) 采图路径与图片统计解耦"""
        ws = self.mgr.create_workspace(alias="dual_test")
        self.assertEqual(ws.get_image_count("calibration"), 0)
        self.assertEqual(ws.get_image_count("production"), 0)

        # 模拟写入 2 张标定图片和 1 张生产图片
        with open(os.path.join(ws.calib_raw_images_dir, "view_0001.png"), "wb") as f:
            f.write(b"c1")
        with open(os.path.join(ws.calib_raw_images_dir, "view_0002.png"), "wb") as f:
            f.write(b"c2")
        with open(os.path.join(ws.prod_raw_images_dir, "view_0001.png"), "wb") as f:
            f.write(b"p1")

        self.assertEqual(ws.get_image_count("calibration"), 2)
        self.assertEqual(ws.get_image_count("production"), 1)

        # 刷新统计
        ws.refresh_stats()
        self.assertEqual(ws.extrinsic_calib_image_count, 2)
        self.assertEqual(ws.prod_image_count, 1)


if __name__ == "__main__":
    unittest.main()
