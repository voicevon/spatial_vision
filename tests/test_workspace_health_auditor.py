# -*- coding: utf-8 -*-
"""
工位空间场景与标靶全量健康体检服务单元测试 (TestWorkspaceHealthAuditor)
"""

import os
import shutil
import tempfile
import unittest
import yaml

from src.workspace.workspace_manager import (
    WorkspaceManager,
    save_workspace_tag_config,
    load_workspace_tag_config,
)
from src.workspace.health_auditor import audit_workspace, audit_all_workspaces
from tools.workspace_hub.hub_hit_tester import HubHitTester


class TestWorkspaceHealthAuditor(unittest.TestCase):
    """测试健康体检、幽灵标靶修剪及报告生成"""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_audit_")
        self.wm = WorkspaceManager(workspaces_dir=self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_prune_ghost_tags_in_calibrated_workspace(self):
        """测试已建图标定工位的幽灵标靶与悬空锚点自愈修剪"""
        ws = self.wm.create_workspace(alias="测试自愈工位")

        # 模拟 tags_map.yaml (真实建图只有 0, 1, 3 号)
        map_content = {
            "tags": {
                "0": {"position_mm": [0, 0, 0]},
                "1": {"position_mm": [0, 500, 0]},
                "3": {"position_mm": [100, 100, 0]},
            }
        }
        with open(ws.map_path, "w", encoding="utf-8") as f:
            yaml.dump(map_content, f)

        # 模拟 tag_whitelist.yaml 混入了幽灵标靶 20, 28, 29 和全空锚点
        dirty_allowed = [0, 1, 3, 20, 28, 29]
        dirty_anchors = {
            0: {"xyz_mm": [0.0, 0.0, None]},
            1: {"xyz_mm": [0.0, 500.0, None]},
            20: {"xyz_mm": [0.0, 0.0, None]},   # 幽灵锚点
            28: {"xyz_mm": [None, None, 0.0]},  # 幽灵锚点
            3: {"xyz_mm": [None, None, None]},  # 全空无效锚点
        }
        save_workspace_tag_config(ws, allowed_ids=dirty_allowed, anchor_tags=dirty_anchors)

        # 执行体检与自愈
        res = audit_workspace(ws, auto_fix=True)

        self.assertIn(20, res["tags"]["ghost_allowed"])
        self.assertIn(28, res["tags"]["ghost_allowed"])
        self.assertIn(29, res["tags"]["ghost_allowed"])
        self.assertTrue(len(res["actions_taken"]) > 0)

        # 核验证盘文件已完成自愈
        clean_cfg = load_workspace_tag_config(ws.workspace_dir)
        self.assertEqual(clean_cfg["allowed_ids"], [0, 1, 3])
        self.assertEqual(sorted(list(clean_cfg["anchor_tags"].keys())), [0, 1])

    def test_protect_uncalibrated_workspace_whitelist(self):
        """测试未建图工位（准备期）不会被误删预设白名单"""
        ws = self.wm.create_workspace(alias="未建图准备工位")
        preset_allowed = [5, 6, 7, 8, 11]
        save_workspace_tag_config(ws, allowed_ids=preset_allowed)

        res = audit_workspace(ws, auto_fix=True)

        # 没有 tags_map.yaml，不触发幽灵剔除
        self.assertEqual(res["tags"]["ghost_allowed"], [])
        clean_cfg = load_workspace_tag_config(ws.workspace_dir)
        self.assertEqual(clean_cfg["allowed_ids"], preset_allowed)

    def test_audit_all_workspaces_and_report_generation(self):
        """测试全工位体检并生成 temp/ 目录 Markdown 报告"""
        ws1 = self.wm.create_workspace(alias="工位A")
        ws2 = self.wm.create_workspace(alias="工位B")

        # 模拟工位A有 tags_map，且有多余幽灵标靶
        with open(ws1.map_path, "w", encoding="utf-8") as f:
            yaml.dump({"tags": {"0": {}, "1": {}}}, f)
        save_workspace_tag_config(ws1, allowed_ids=[0, 1, 99])

        res_all = audit_all_workspaces(self.wm, auto_fix=True)

        self.assertEqual(res_all["total_workspaces"], 2)
        self.assertGreaterEqual(res_all["healed_workspaces"], 1)
        self.assertIn(res_all["total_ghost_pruned"], [1, 99])
        self.assertTrue(os.path.isfile(res_all["report_path"]))

        # 检查报告内容格式
        with open(res_all["report_path"], "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("工位空间场景与标靶全量健康体检报告", content)
        self.assertIn("工位A", content)
        self.assertIn("工位B", content)

    def test_prune_illegal_output_tags_from_tags_map(self):
        """测试平差输出端非法垃圾标靶自愈清除: 地图中收录了未放行的 Tag，自动从 tags_map.yaml 中彻底清除"""
        ws = self.wm.create_workspace(alias="测试输出端非法标靶清除工位")

        # 白名单仅放行 5, 6, 7, 8
        save_workspace_tag_config(ws, allowed_ids=[5, 6, 7, 8])

        # tags_map.yaml 却混入了未放行的非法标靶 27 与 99
        map_content = {
            "tags": {
                "5": {"position_mm": [0, 0, 0]},
                "6": {"position_mm": [10, 0, 0]},
                "7": {"position_mm": [20, 0, 0]},
                "8": {"position_mm": [30, 0, 0]},
                "27": {"position_mm": [100, 200, 0]},  # 未放行非法标靶
                "99": {"position_mm": [50, 50, 0]},    # 未放行非法标靶
            },
            "raw_relative_poses": {
                "5": {}, "6": {}, "7": {}, "8": {}, "27": {}, "99": {}
            }
        }
        with open(ws.map_path, "w", encoding="utf-8") as f:
            yaml.dump(map_content, f)

        # 执行体检与自愈
        res = audit_workspace(ws, auto_fix=True)

        self.assertIn(27, res["tags"]["illegal_map_tags"])
        self.assertIn(99, res["tags"]["illegal_map_tags"])
        self.assertTrue(any("彻底清除未放行非法标靶" in act for act in res["actions_taken"]))

        # 验证 tags_map.yaml 已完成物理清洗，27 与 99 已不复存在
        with open(ws.map_path, "r", encoding="utf-8") as f:
            clean_map = yaml.safe_load(f)
        self.assertEqual(sorted([int(k) for k in clean_map["tags"].keys()]), [5, 6, 7, 8])
        self.assertEqual(sorted([int(k) for k in clean_map["raw_relative_poses"].keys()]), [5, 6, 7, 8])

    def test_hub_hit_test_audit_button(self):
        """测试 Workspace Hub 左侧上下结构按钮命中测试 (预留56px底边距防截断)"""
        from tools.workspace_hub.hub_state import HubState
        from tools.workspace_hub.hub_renderer import HubRenderer
        renderer = HubRenderer()
        tester = HubHitTester(renderer)
        state = HubState(workspace_mgr=self.wm, force_mock=True)

        # 点击全工位体检按钮区域 (x: 10~330, y: 628~664)
        action = tester.hit_test(150, 646, state=state)
        self.assertEqual(action, "btn_audit_all_workspaces")

        # 点击新建工位按钮区域 (x: 10~330, y: 584~620)
        action_new = tester.hit_test(150, 602, state=state)
        self.assertEqual(action_new, "btn_new_workspace")


if __name__ == "__main__":
    unittest.main()
