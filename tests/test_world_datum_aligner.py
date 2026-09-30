#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WorldDatumAligner 独立解耦单元测试套件
验证:
1. 锚点配置归一化与 DoF 自由度记账
2. Umeyama 3D 闭式解析相似变换精度
3. 相对底图到世界生产地图的独立对齐
4. 锚点几何形变校验与对齐质检单 (正常吻合 vs 错录黄色告警)
"""

import math
import unittest
from unittest.mock import patch
import numpy as np

from src.calibration.world_datum_aligner import WorldDatumAligner


class TestWorldDatumAligner(unittest.TestCase):

    def setUp(self):
        self.aligner = WorldDatumAligner(marker_size_mm=50.0)

    def test_normalize_anchor_tags(self):
        # 1. 统一新格式输入
        raw_new = {
            0: {"xyz_mm": [0.0, 0.0, 0.0], "known": [True, True, True]},
            "1": {"xyz_mm": [100.0, 200.0, 0.0]},  # known 缺失默认为 True
        }
        res_new = WorldDatumAligner.normalize_anchor_tags(raw_new)
        self.assertIsNotNone(res_new)
        self.assertIn(0, res_new)
        self.assertIn(1, res_new)
        self.assertEqual(res_new[1]["known"], [True, True, True])

        # 2. 支持 null 轴输入并自动推导 known 掩码
        raw_null = {
            5: {"xyz_mm": [0.0, 0.0, None]},
            6: {"xyz_mm": [300.0, None, 0.0]}
        }
        res_null = WorldDatumAligner.normalize_anchor_tags(raw_null)
        self.assertIsNotNone(res_null)
        self.assertIn(5, res_null)
        self.assertIn(6, res_null)
        self.assertEqual(res_null[5]["known"], [True, True, False])
        self.assertEqual(res_null[6]["known"], [True, False, True])

    def test_evaluate_anchor_dof(self):
        # 3个全知点 -> full (5/5 DoF)
        anchors_full = {
            0: {"xyz_mm": [0.0, 0.0, 0.0], "known": [True, True, True]},
            1: {"xyz_mm": [300.0, 0.0, 0.0], "known": [True, True, True]},
            2: {"xyz_mm": [0.0, 400.0, 0.0], "known": [True, True, True]},
        }
        dof_full = WorldDatumAligner.evaluate_anchor_dof(anchors_full)
        self.assertEqual(dof_full["mode"], "full")
        self.assertEqual(dof_full["dof_solved"], 5)

        # 仅已知 XY, Z 悬空 -> partial (4/5 DoF)
        anchors_partial = {
            0: {"xyz_mm": [0.0, 0.0, 0.0], "known": [True, True, False]},
            1: {"xyz_mm": [300.0, 0.0, 0.0], "known": [True, True, False]},
        }
        dof_partial = WorldDatumAligner.evaluate_anchor_dof(anchors_partial)
        self.assertEqual(dof_partial["mode"], "partial")
        self.assertEqual(dof_partial["dof_solved"], 4)

    def test_umeyama_alignment_synthetic(self):
        # 构造基准 3D 点云
        src = np.array([
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [0.0, 150.0, 0.0],
            [100.0, 150.0, 20.0],
        ], dtype=np.float64)

        # 真值相似变换: s=1.05, 旋转 30° 绕 Z, 平移 [50, -30, 10]
        angle = math.radians(30.0)
        c, s = math.cos(angle), math.sin(angle)
        R_gt = np.array([
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        t_gt = np.array([50.0, -30.0, 10.0], dtype=np.float64)
        s_gt = 1.05

        dst = s_gt * (src @ R_gt.T) + t_gt

        # 解算
        scale, R, t = WorldDatumAligner.umeyama_alignment(src, dst)

        self.assertAlmostEqual(scale, s_gt, places=4)
        np.testing.assert_allclose(R, R_gt, atol=1e-4)
        np.testing.assert_allclose(t, t_gt, atol=1e-4)

    def test_align_relative_map_to_world_and_report(self):
        # 1. 构造一个相对底图 (假定相机重构出的相对位置)
        relative_map = {
            "origin_tag_id": 0,
            "marker_size_mm": 50.0,
            "final_rmse": 0.12,
            "raw_relative_poses": {
                0: np.eye(4).tolist(),
                1: [
                    [1.0, 0.0, 0.0, 300.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ],
                2: [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 400.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ]
            }
        }

        # 2. 正常锚点配置
        normal_anchors = {
            0: {"xyz_mm": [0.0, 0.0, 0.0], "known": [True, True, True]},
            1: {"xyz_mm": [300.0, 0.0, 0.0], "known": [True, True, True]},
            2: {"xyz_mm": [0.0, 400.0, 0.0], "known": [True, True, True]}
        }

        world_map = self.aligner.align_relative_map_to_world(
            relative_map=relative_map,
            anchor_tags=normal_anchors,
            origin_tag_id=0,
            x_align_tag_id=1,
            strict=True
        )

        self.assertEqual(world_map["anchor_mode"], "full")
        self.assertIn("world_anchor", world_map)
        rep = world_map["world_anchor"]["alignment_report"]
        self.assertFalse(rep["has_warn"], "正常坐标不应触发警告")
        self.assertLess(rep["mean_mm"], 0.01)

        # 验证 Markdown 格式化输出
        md_text = WorldDatumAligner.format_alignment_report_markdown(rep)
        self.assertIn("# 世界坐标系对齐质检单", md_text)
        self.assertIn("Tag #0", md_text)
        self.assertIn("Tag #1", md_text)
        self.assertIn("Tag #2", md_text)

        # 3. 错录锚点坐标 (例如 Tag 2 坐标录错为 [400, 0, 0])
        bad_anchors = {
            0: {"xyz_mm": [0.0, 0.0, 0.0], "known": [True, True, True]},
            1: {"xyz_mm": [300.0, 0.0, 0.0], "known": [True, True, True]},
            2: {"xyz_mm": [400.0, 0.0, 0.0], "known": [True, True, True]}  # 错录
        }

        bad_map = self.aligner.align_relative_map_to_world(
            relative_map=relative_map,
            anchor_tags=bad_anchors,
            origin_tag_id=0,
            x_align_tag_id=1,
            strict=False
        )

        bad_rep = bad_map["world_anchor"]["alignment_report"]
        self.assertTrue(bad_rep["has_warn"], "错录锚点应触发黄色告警")
        row_tag2 = next(r for r in bad_rep["rows"] if r["tag_id"] == 2)
        self.assertTrue(row_tag2["is_warn"])
        self.assertGreater(row_tag2["dist_3d_mm"], 10.0)

    def test_format_conflict_pairs_report(self):
        """测试几何冲突报告格式化输出与智能纠错线索分析"""
        from src.calibration.world_datum_aligner import format_conflict_pairs_report
        conflicts = [
            {"pair": (0, 1), "world_dist_mm": 520.0, "measured_dist_mm": 960.1, "diff_mm": 440.1, "rel_error": 0.458},
            {"pair": (1, 11), "world_dist_mm": 520.0, "measured_dist_mm": 344.2, "diff_mm": 175.8, "rel_error": 0.511},
            {"pair": (1, 14), "world_dist_mm": 30.0, "measured_dist_mm": 470.1, "diff_mm": 440.1, "rel_error": 0.936},
        ]
        text = format_conflict_pairs_report(conflicts)
        self.assertIn("锚点几何严重冲突", text)
        self.assertIn("Tag #0 ⇋ Tag #1", text)
        self.assertIn("520.0 mm", text)
        self.assertIn("960.1 mm", text)
        self.assertIn("智能纠错线索分析", text)
        self.assertIn("Tag #1", text)

    @patch.object(WorldDatumAligner, "solve_similarity_from_anchors")
    def test_conflict_pairs_in_exception(self, mock_solve):
        """测试当存在真实几何冲突导致偏航不可解退化时，ValueError 携带冲突明细与排查线索"""
        mock_solve.return_value = ("none", {
            "reason": "无任何共同已知 XY 的锚点对, 偏航不可解",
            "conflict_pairs": [
                {"pair": (0, 1), "world_dist_mm": 520.0, "measured_dist_mm": 960.1, "diff_mm": 440.1, "rel_error": 0.458}
            ]
        })
        tag_poses = {0: np.eye(4), 1: np.eye(4)}
        conflict_anchors = {
            0: {"xyz_mm": [0.0, 0.0, 0.0], "known": [True, True, True]},
            1: {"xyz_mm": [0.0, 520.0, 0.0], "known": [True, True, True]}
        }
        with self.assertRaises(ValueError) as ctx:
            self.aligner.anchor_to_absolute_world(tag_poses, conflict_anchors, strict=True)

        err_msg = str(ctx.exception)
        self.assertIn("锚点求解退化", err_msg)
        self.assertIn("锚点几何严重冲突", err_msg)
        self.assertIn("智能纠错线索分析", err_msg)

    def test_sub_frame_anchors_isolated(self):
        """测试子坐标系标靶 (frame_id != 'world') 自动隔离，不混入世界坐标系对齐"""
        mixed_anchors = {
            0: {"xyz_mm": [0.0, 0.0, 0.0], "known": [True, True, True], "frame_id": "world"},
            1: {"xyz_mm": [300.0, 0.0, 0.0], "known": [True, True, True], "frame_id": "world"},
            2: {"xyz_mm": [0.0, 400.0, 0.0], "known": [True, True, True], "frame_id": "world"},
            11: {"xyz_mm": [0.0, 0.0, 0.0], "known": [True, True, True], "frame_id": "frame_sub_1"},
            14: {"xyz_mm": [0.0, 490.0, 0.0], "known": [True, True, True], "frame_id": "frame_sub_1"},
        }
        tag_poses = {
            0: np.eye(4),
            1: np.array([[1, 0, 0, 300], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64),
            2: np.array([[1, 0, 0, 0], [0, 1, 0, 400], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64),
            11: np.array([[1, 0, 0, 1000], [0, 1, 0, 1000], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64),
            14: np.array([[1, 0, 0, 1000], [0, 1, 0, 1490], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64),
        }
        res = self.aligner.anchor_to_absolute_world(tag_poses, mixed_anchors, strict=True)
        # 验证世界对齐质检单中只有 world 标靶 (0, 1, 2)，没有子坐标系标靶 (11, 14)
        rep = res["world_anchor"]["alignment_report"]
        report_tids = [r["tag_id"] for r in rep.get("rows", [])]
        self.assertIn(0, report_tids)
        self.assertIn(1, report_tids)
        self.assertIn(2, report_tids)
        self.assertNotIn(11, report_tids)
        self.assertNotIn(14, report_tids)


if __name__ == "__main__":
    unittest.main()
