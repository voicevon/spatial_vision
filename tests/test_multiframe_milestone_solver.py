#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多坐标系分层里程碑解算器单元测试
"""

import unittest
import numpy as np

from src.workspace.coordinate_manager import CoordinateTreeManager, FrameDefinition
from src.calibration.solvers.multiframe_milestone_solver import (
    MultiFrameMilestoneSolver,
    check_sub_frame_local_rigidity,
    extract_tag_positions_from_map,
)


class TestMultiFrameMilestoneSolver(unittest.TestCase):

    def setUp(self):
        # 构造基准世界坐标系和两个子坐标系
        self.coord_mgr = CoordinateTreeManager(workspace_id="test_ws")

        # 子系 1: 传送带 (frame_sub_1), 标称 Tag 11 与 Tag 14 相距 490 mm
        self.frame_sub_1 = FrameDefinition(
            frame_id="frame_sub_1",
            name="传送带机构",
            parent_frame_id="world",
            type="fixed_transform",
            calibration_spec={
                "method": "anchor_tags_registration",
                "reference_tag_ids": [11, 14, 19]
            }
        )
        self.coord_mgr.add_frame(self.frame_sub_1)

        # 子系 2: 抓取滑台 (frame_sub_2), 标称 Tag 30 与 Tag 31 相距 300 mm
        self.frame_sub_2 = FrameDefinition(
            frame_id="frame_sub_2",
            name="抓取滑台",
            parent_frame_id="world",
            type="fixed_transform",
            calibration_spec={
                "method": "anchor_tags_registration",
                "reference_tag_ids": [30, 31, 32]
            }
        )
        self.coord_mgr.add_frame(self.frame_sub_2)

    def test_extract_tag_positions(self):
        """测试从各类 tags_map 数据结构提取 3D 位置"""
        tags_map = {
            "tags": {
                "1": {"position_mm": [10.0, 20.0, 30.0]},
                "2": {"transform_matrix": np.eye(4).tolist()},
                "3": np.eye(4),
            }
        }
        res = extract_tag_positions_from_map(tags_map)
        self.assertEqual(len(res), 3)
        np.testing.assert_allclose(res[1], [10.0, 20.0, 30.0])
        np.testing.assert_allclose(res[2], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(res[3], [0.0, 0.0, 0.0])

    def test_sub_frame_local_rigidity_pass_and_fail(self):
        """测试子坐标系局部刚体一致性自检通过与超差拦截"""
        # 1. 正常情况: 实测间距与标称 490 mm 仅相差 0.5 mm
        rel_tags_good = {
            11: np.array([100.0, 200.0, 300.0]),
            14: np.array([589.5, 200.0, 300.0]),  # 间距 489.5 mm (误差 -0.5 mm)
            19: np.array([345.0, 250.2, 300.1]),
        }
        anchor_sub_1 = {
            11: {"xyz_mm": [0.0, 0.0, 0.0], "frame_id": "frame_sub_1"},
            14: {"xyz_mm": [490.0, 0.0, 0.0], "frame_id": "frame_sub_1"},
            19: {"xyz_mm": [245.0, 50.0, 0.0], "frame_id": "frame_sub_1"},
        }
        rep_good = check_sub_frame_local_rigidity(
            frame=self.frame_sub_1,
            relative_tags=rel_tags_good,
            dist_tol_mm=3.0,
            fallback_anchor_tags=anchor_sub_1,
        )
        self.assertTrue(rep_good.local_rigidity_passed)
        self.assertEqual(len(rep_good.local_conflict_pairs), 0)
        self.assertLess(rep_good.local_max_err_mm, 2.0)

        # 2. 超差情况: 实测间距 510 mm (误差 20 mm，贴标严重错误)
        rel_tags_bad = {
            11: np.array([100.0, 200.0, 300.0]),
            14: np.array([610.0, 200.0, 300.0]),  # 间距 510.0 mm
            19: np.array([345.0, 250.0, 300.0]),
        }
        rep_bad = check_sub_frame_local_rigidity(
            frame=self.frame_sub_1,
            relative_tags=rel_tags_bad,
            dist_tol_mm=3.0,
            fallback_anchor_tags=anchor_sub_1,
        )
        self.assertFalse(rep_bad.local_rigidity_passed)
        self.assertGreater(len(rep_bad.local_conflict_pairs), 0)
        self.assertGreater(rep_bad.local_max_err_mm, 15.0)

    def test_milestone_solver_all_passed(self):
        """测试 M1 + M2 + 动态子系全流程成功"""
        # 构造 BA 相对底图 (含底座 Tag 0, 1, 2 与 子系 Tag 11, 14, 19)
        # 假定相对系与世界系完全重合以便验证
        rel_map = {
            "final_rmse": 0.25,
            "tags": {
                "0": {"position_mm": [0.0, 0.0, 0.0]},
                "1": {"position_mm": [0.0, 520.0, 0.0]},
                "2": {"position_mm": [300.0, 0.0, 0.0]},
                # 子系 1
                "11": {"position_mm": [50.0, 100.0, 0.0]},
                "14": {"position_mm": [540.0, 100.0, 0.0]},
                "19": {"position_mm": [295.0, 150.0, 0.0]},
            }
        }
        # 工位锚点: 0, 1, 2 为 world，11, 14, 19 为 frame_sub_1
        anchor_tags = {
            0: {"xyz_mm": [0.0, 0.0, 0.0], "frame_id": "world"},
            1: {"xyz_mm": [0.0, 520.0, 0.0], "frame_id": "world"},
            2: {"xyz_mm": [300.0, 0.0, 0.0], "frame_id": "world"},
            11: {"xyz_mm": [0.0, 0.0, 0.0], "frame_id": "frame_sub_1"},
            14: {"xyz_mm": [490.0, 0.0, 0.0], "frame_id": "frame_sub_1"},
            19: {"xyz_mm": [245.0, 50.0, 0.0], "frame_id": "frame_sub_1"},
        }

        # 仅保留 frame_sub_1
        coord_mgr_single = CoordinateTreeManager(workspace_id="test_single")
        coord_mgr_single.add_frame(self.frame_sub_1)

        solver = MultiFrameMilestoneSolver(marker_size_mm=80.0)
        succ, rep, world_map = solver.solve(
            relative_map=rel_map,
            anchor_tags=anchor_tags,
            coord_mgr=coord_mgr_single,
        )

        self.assertTrue(succ, f"M2 msg: {rep.m2_msg}, Sub1: {rep.sub_frames.get('frame_sub_1')}")
        self.assertTrue(rep.m1_free_ba_passed)
        self.assertTrue(rep.m2_world_datum_passed)
        self.assertIn("frame_sub_1", rep.sub_frames)

        sub_1_rep = rep.sub_frames["frame_sub_1"]
        self.assertTrue(sub_1_rep.local_rigidity_passed)
        self.assertTrue(sub_1_rep.extrinsic_solved)
        self.assertFalse(sub_1_rep.is_isolated)

        # 验证外参已写回 coord_mgr
        updated_frame = coord_mgr_single.get_frame("frame_sub_1")
        self.assertEqual(updated_frame.status, "calibrated")
        np.testing.assert_allclose(updated_frame.translation_xyz_mm, [50.0, 100.0, 0.0], atol=1e-1)

        # 验证弹窗与 Markdown 格式化
        diag_text = solver.format_diagnostic_dialog_text(rep)
        self.assertIn("【建图里程碑多坐标系质检诊断】", diag_text)
        self.assertIn("[frame_sub_1 (传送带机构)]", diag_text)

        md_text = solver.format_markdown_report(rep)
        self.assertIn("# 工位多坐标系分层里程碑质检报告", md_text)
        self.assertIn("Tag #11 ⇋ #14", md_text)

    def test_milestone_solver_sub_frame_isolation(self):
        """测试子系超差时独立熔断隔离，不影响世界系和其他合格子系"""
        # 子系 1 尺寸正常，子系 2 尺寸严重变形超差 (标称 300 mm，实测 380 mm)
        rel_map = {
            "final_rmse": 0.22,
            "tags": {
                "0": {"position_mm": [0.0, 0.0, 0.0]},
                "1": {"position_mm": [0.0, 520.0, 0.0]},
                "2": {"position_mm": [300.0, 0.0, 0.0]},
                # 子系 1: 正常
                "11": {"position_mm": [50.0, 100.0, 0.0]},
                "14": {"position_mm": [540.0, 100.0, 0.0]},
                "19": {"position_mm": [295.0, 150.0, 0.0]},
                # 子系 2: 严重超差
                "30": {"position_mm": [0.0, 800.0, 0.0]},
                "31": {"position_mm": [380.0, 800.0, 0.0]},  # 标称 300，实测 380 -> 偏差 80 mm!
                "32": {"position_mm": [0.0, 1000.0, 0.0]},
            }
        }
        anchor_tags = {
            0: {"xyz_mm": [0.0, 0.0, 0.0], "frame_id": "world"},
            1: {"xyz_mm": [0.0, 520.0, 0.0], "frame_id": "world"},
            2: {"xyz_mm": [300.0, 0.0, 0.0], "frame_id": "world"},
            11: {"xyz_mm": [50.0, 100.0, 0.0], "frame_id": "frame_sub_1"},
            14: {"xyz_mm": [540.0, 100.0, 0.0], "frame_id": "frame_sub_1"},
            19: {"xyz_mm": [295.0, 150.0, 0.0], "frame_id": "frame_sub_1"},
            30: {"xyz_mm": [0.0, 800.0, 0.0], "frame_id": "frame_sub_2"},
            31: {"xyz_mm": [300.0, 800.0, 0.0], "frame_id": "frame_sub_2"},
            32: {"xyz_mm": [0.0, 1000.0, 0.0], "frame_id": "frame_sub_2"},
        }

        solver = MultiFrameMilestoneSolver(marker_size_mm=80.0)
        succ, rep, world_map = solver.solve(
            relative_map=rel_map,
            anchor_tags=anchor_tags,
            coord_mgr=self.coord_mgr,
        )

        # M1 和 M2 均通过
        self.assertTrue(rep.m1_free_ba_passed)
        self.assertTrue(rep.m2_world_datum_passed)

        # 子系 1 绿灯
        sub_1 = rep.sub_frames["frame_sub_1"]
        self.assertTrue(sub_1.local_rigidity_passed)
        self.assertTrue(sub_1.extrinsic_solved)

        # 子系 2 熔断隔离
        sub_2 = rep.sub_frames["frame_sub_2"]
        self.assertFalse(sub_2.local_rigidity_passed)
        self.assertTrue(sub_2.is_isolated)
        self.assertFalse(sub_2.extrinsic_solved)
        self.assertIn("已触发外参熔断隔离", sub_2.extrinsic_status_msg)

        # 全局总判定应为 False (因为存在超差子系)
        self.assertFalse(succ)
        self.assertFalse(rep.all_milestones_passed)

        # 检查诊断文本中给出隔离与排查建议
        diag_text = solver.format_diagnostic_dialog_text(rep)
        self.assertIn("frame_sub_2", diag_text)
        self.assertIn("熔断隔离", diag_text)

    def test_milestone_solver_pure_tag_range_mapping(self):
        """测试纯净 anchor_tags (无 frame_id 字段) 依据系统 ID 区间天然映射分组 (0~9 world, 10~19 sub1)"""
        rel_map = {
            "final_rmse": 0.25,
            "tags": {
                "0": {"position_mm": [0.0, 0.0, 0.0]},
                "1": {"position_mm": [0.0, 520.0, 0.0]},
                "2": {"position_mm": [300.0, 0.0, 0.0]},
                "11": {"position_mm": [50.0, 100.0, 0.0]},
                "14": {"position_mm": [540.0, 100.0, 0.0]},
                "19": {"position_mm": [295.0, 150.0, 0.0]},
            }
        }
        # 纯净 anchor_tags: 绝无 frame_id 补丁！
        pure_anchor_tags = {
            0: {"xyz_mm": [0.0, 0.0, 0.0], "known": [True, True, True]},
            1: {"xyz_mm": [0.0, 520.0, 0.0], "known": [True, True, True]},
            2: {"xyz_mm": [300.0, 0.0, 0.0], "known": [True, True, True]},
            11: {"xyz_mm": [0.0, 0.0, 0.0], "known": [True, True, True]},
            14: {"xyz_mm": [490.0, 0.0, 0.0], "known": [True, True, True]},
            19: {"xyz_mm": [245.0, 50.0, 0.0], "known": [True, True, True]},
        }

        coord_mgr = CoordinateTreeManager(workspace_id="test_pure_mapping")
        coord_mgr.add_frame(self.frame_sub_1)

        solver = MultiFrameMilestoneSolver(marker_size_mm=80.0)
        succ, rep, world_map = solver.solve(
            relative_map=rel_map,
            anchor_tags=pure_anchor_tags,
            coord_mgr=coord_mgr,
        )

        self.assertTrue(succ, f"M2: {rep.m2_msg}, Sub1: {rep.sub_frames.get('frame_sub_1')}")
        self.assertTrue(rep.m1_free_ba_passed)
        self.assertTrue(rep.m2_world_datum_passed)
        self.assertIn("frame_sub_1", rep.sub_frames)

        sub_1_rep = rep.sub_frames["frame_sub_1"]
        self.assertTrue(sub_1_rep.local_rigidity_passed)
        self.assertTrue(sub_1_rep.extrinsic_solved)
        self.assertFalse(sub_1_rep.is_isolated)

        # 验证外参准确反推并写回 coord_mgr
        updated_frame = coord_mgr.get_frame("frame_sub_1")
        self.assertEqual(updated_frame.status, "calibrated")
        np.testing.assert_allclose(updated_frame.translation_xyz_mm, [50.0, 100.0, 0.0], atol=1e-1)

    def test_sub_frame_scale_normalization(self):
        """测试当底图尺度与真实物理尺度不一致 (scale != 1.0) 时，阶段 A 自动按真实物理尺度核对，彻底消除假超差"""
        # 真实物理世界中: 标靶 0, 1 间距为 500mm; 子系标靶 11, 14 间距为 490mm
        # 模拟相机自由平差因为名义标靶尺寸偏大，底图坐标被整体放大了 1.2 倍 (scale_factor 应为 1 / 1.2 = 0.83333)
        scale_err = 1.2
        rel_map = {
            "final_rmse": 0.3,
            "tags": {
                "0": {"position_mm": [0.0, 0.0, 0.0]},
                "1": {"position_mm": [0.0, 500.0 * scale_err, 0.0]},
                "2": {"position_mm": [300.0 * scale_err, 0.0, 0.0]},
                # 子系 1: 未缩放前间距为 490 * 1.2 = 588mm
                "11": {"position_mm": [50.0 * scale_err, 100.0 * scale_err, 0.0]},
                "14": {"position_mm": [540.0 * scale_err, 100.0 * scale_err, 0.0]},
            }
        }

        # 世界锚点真值与子系名义真值 (严格物理毫米)
        anchor_tags = {
            0: {"xyz_mm": [0.0, 0.0, 0.0], "frame_id": "world"},
            1: {"xyz_mm": [0.0, 500.0, 0.0], "frame_id": "world"},
            2: {"xyz_mm": [300.0, 0.0, 0.0], "frame_id": "world"},
            11: {"xyz_mm": [0.0, 0.0, 0.0], "frame_id": "frame_sub_1"},
            14: {"xyz_mm": [490.0, 0.0, 0.0], "frame_id": "frame_sub_1"},
        }

        coord_mgr = CoordinateTreeManager(workspace_id="test_scale_norm")
        coord_mgr.add_frame(self.frame_sub_1)

        solver = MultiFrameMilestoneSolver(marker_size_mm=80.0)
        succ, rep, world_map = solver.solve(
            relative_map=rel_map,
            anchor_tags=anchor_tags,
            coord_mgr=coord_mgr,
        )

        self.assertTrue(succ, f"M2 msg: {rep.m2_msg}")
        self.assertTrue(rep.m2_world_datum_passed)
        sub_rep = rep.sub_frames["frame_sub_1"]
        # 核心断言: 阶段 A 成功通过，未因底图放大了 1.2 倍而误报超差 98mm
        self.assertTrue(sub_rep.local_rigidity_passed, f"阶段 A 应通过，但实际消息: {sub_rep.local_status_msg}")
        self.assertLess(sub_rep.local_max_err_mm, 1.0)
        self.assertTrue(sub_rep.extrinsic_solved)
        self.assertFalse(sub_rep.is_isolated)


if __name__ == "__main__":
    unittest.main()
