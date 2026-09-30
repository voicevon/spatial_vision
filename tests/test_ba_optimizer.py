#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BundleAdjustmentOptimizer 单元测试 (test_ba_optimizer.py)
=========================================================
覆盖核心特性：
  1. 旋转平移向量与 4x4 齐次矩阵双向精确可逆转换；
  2. 双标靶基线尺度修正算法 (Metric Baseline Gauge)；
  3. SCARA 世界坐标系对齐闭环 (Origin 锚定与 X 轴水平对齐)；
  4. 3D 不确定度协方差提取；
  5. 约束积累式世界锚定 (全知/部分已知锚点, full/partial/none 三级模式)。
"""

import os
import sys
import math
import tempfile
import unittest
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.calibration.solvers.ba_optimizer import BundleAdjustmentOptimizer
from src.utils.config_guard import load_anchor_tags


class TestBundleAdjustmentOptimizer(unittest.TestCase):
    def setUp(self):
        K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]])
        dist = np.zeros(5)
        self.optimizer = BundleAdjustmentOptimizer(
            camera_matrix=K,
            dist_coeffs=dist,
            marker_size_mm=50.0
        )

    def test_matrix_rvec_tvec_roundtrip(self):
        """测试 4x4 矩阵与 rvec/tvec 的无损双向转换"""
        rvec_in = np.array([0.1, -0.2, 0.3], dtype=np.float64)
        tvec_in = np.array([100.0, 200.0, 300.0], dtype=np.float64)

        T = self.optimizer.rvec_tvec_to_matrix(rvec_in, tvec_in)
        rvec_out, tvec_out = self.optimizer.matrix_to_rvec_tvec(T)

        np.testing.assert_allclose(rvec_out.flatten(), rvec_in.flatten(), atol=1e-7)
        np.testing.assert_allclose(tvec_out.flatten(), tvec_in.flatten(), atol=1e-7)

    def test_apply_baseline_scale(self):
        """测试 Metric Baseline Gauge 尺度校准"""
        # 初始名义坐标：Tag 0 在 (0, 0, 0), Tag 1 在 (500, 0, 0)，名义距离 500mm
        tag_poses = {
            0: np.eye(4),
            1: np.array([[1, 0, 0, 500.0],
                         [0, 1, 0, 0.0],
                         [0, 0, 1, 0.0],
                         [0, 0, 0, 1.0]], dtype=np.float64)
        }
        # 实际物理测量距离 525.0mm (比例尺 1.05)
        scaled_poses, scale, real_marker_size = self.optimizer.apply_baseline_scale(
            tag_poses, tag_id_a=0, tag_id_b=1, real_distance_mm=525.0
        )
        self.assertAlmostEqual(scale, 1.05, places=5)
        self.assertAlmostEqual(real_marker_size, 52.5, places=2)
        np.testing.assert_allclose(scaled_poses[1][:3, 3], [525.0, 0.0, 0.0], atol=1e-5)

    def test_align_to_scara_world(self):
        """测试将 Tag 0 绑定为原点，Tag 1 对齐至 +X 轴"""
        # 构造未对齐状态：Tag 0 在 (100, 100, 0)，Tag 1 在 (100+300*cos(45°), 100+300*sin(45°), 0)
        p0 = np.array([100.0, 100.0, 0.0])
        p1 = np.array([100.0 + 300.0 * np.cos(np.pi / 4), 100.0 + 300.0 * np.sin(np.pi / 4), 0.0])
        T0 = np.eye(4)
        T0[:3, 3] = p0
        T1 = np.eye(4)
        T1[:3, 3] = p1

        tag_poses = {0: T0, 1: T1}
        aligned_map = self.optimizer.align_to_scara_world(tag_poses, origin_tag_id=0, x_align_tag_id=1)

        pos0 = aligned_map["tags"][0]["position_mm"]
        pos1 = aligned_map["tags"][1]["position_mm"]

        # 校验 Tag 0 归零
        np.testing.assert_allclose(pos0, [0.0, 0.0, 0.0], atol=1e-2)
        # 校验 Tag 1 落在 +X 轴上 (Y=0, X 约为 300.0)
        self.assertAlmostEqual(pos1[1], 0.0, places=1)
        self.assertAlmostEqual(pos1[0], 300.0, places=1)

    def test_optimize_with_callback(self):
        """测试 BA optimize 求解过程中 callback 回调机制正常被触发且接收到指标"""
        # 构造合成理想观测: 2 帧视角，共视 Tag 0 与 Tag 1
        # 相机内参
        K = self.optimizer.camera_matrix
        dist = self.optimizer.dist_coeffs

        # Tag 0 在 (0, 0, 0)，Tag 1 在 (300, 0, 0)
        T_w_t0 = np.eye(4)
        T_w_t1 = np.eye(4)
        T_w_t1[:3, 3] = [300.0, 0.0, 0.0]

        # 两个相机机位
        # 相机 1 在 (150, 0, -800) 正看
        rvec_c1 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        tvec_c1 = np.array([-150.0, 0.0, 800.0], dtype=np.float64)

        # 相机 2 略微偏斜机位
        rvec_c2 = np.array([0.05, -0.08, 0.0], dtype=np.float64)
        tvec_c2 = np.array([-120.0, 20.0, 850.0], dtype=np.float64)

        # 生成投影角点
        def project(T_w_t, rvec_c, tvec_c):
            R_c, _ = cv2.Rodrigues(rvec_c)
            T_c_w = np.eye(4)
            T_c_w[:3, :3] = R_c
            T_c_w[:3, 3] = tvec_c
            T_c_t = T_c_w @ T_w_t
            rv, tv = self.optimizer.matrix_to_rvec_tvec(T_c_t)
            pts2d, _ = cv2.projectPoints(self.optimizer.obj_points, rv, tv, K, dist)
            return pts2d.reshape(4, 2)

        frame0_tags = {0: project(T_w_t0, rvec_c1, tvec_c1), 1: project(T_w_t1, rvec_c1, tvec_c1)}
        frame1_tags = {0: project(T_w_t0, rvec_c2, tvec_c2), 1: project(T_w_t1, rvec_c2, tvec_c2)}
        detections = [frame0_tags, frame1_tags]

        callback_events = []
        def test_callback(info):
            callback_events.append(info)

        res = self.optimizer.optimize(
            detections,
            active_frame_names=["test_f0", "test_f1"],
            origin_tag_id=0,
            x_align_tag_id=1,
            anchor_tags={0: {"xyz_mm": [0.0, 0.0, 0.0], "known": [True, True, True]},
                         1: {"xyz_mm": [500.0, 0.0, 0.0], "known": [True, True, True]}},
            callback=test_callback
        )

        self.assertIsNotNone(res)
        self.assertIn("final_rmse", res)
        self.assertLess(res["final_rmse"], 0.2, "理想几何下的平差 RMSE 应小于 0.2px")
        self.assertGreater(len(callback_events), 0, "callback 应被多次触发")

        # 检查事件字典字段
        first_event = callback_events[0]
        self.assertIn("stage", first_event)
        self.assertIn("iter", first_event)
        self.assertIn("rmse", first_event)
        self.assertIn("sub_progress", first_event)


def _make_ba_poses(world_pts: dict, scale: float, yaw: float, t) -> dict:
    """按已知相似变换 p_w = s·Rz(yaw)·p_ba + t 反推 BA 系标靶平移, 构造 4x4 位姿字典"""
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t = np.asarray(t, dtype=np.float64)
    poses = {}
    for tid, p_w in world_pts.items():
        p_ba = (R.T @ (np.asarray(p_w, dtype=np.float64) - t)) / scale
        T = np.eye(4)
        T[:3, 3] = p_ba
        poses[tid] = T
    return poses


class TestConstraintAnchor(unittest.TestCase):
    """约束积累式世界锚定: normalize / DoF 记账 / 分阶段闭式解 / 三级模式"""

    def setUp(self):
        K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]])
        self.optimizer = BundleAdjustmentOptimizer(camera_matrix=K, dist_coeffs=np.zeros(5), marker_size_mm=50.0)

    def test_normalize_anchor_tags(self):
        """测试锚点配置格式归一化 (支持完整坐标与包含 null 轴的已知/未知掩码自动推导)"""
        norm_fmt = {0: {"xyz_mm": [0.0, 0.0, 405.0]},
                    1: {"xyz_mm": [0.0, 520.0, 196.0]}}
        out = BundleAdjustmentOptimizer.normalize_anchor_tags(norm_fmt)
        self.assertEqual(set(out.keys()), {0, 1})
        self.assertEqual(out[0]["known"], [True, True, True])
        self.assertEqual(out[1]["xyz_mm"], [0.0, 520.0, 196.0])

        new_fmt = {0: {"xyz_mm": [1.0, 2.0, 3.0], "known": [True, False, True]}}
        out = BundleAdjustmentOptimizer.normalize_anchor_tags(new_fmt)
        self.assertEqual(out[0]["known"], [True, False, True])

        # known 缺省视为三轴全知; 全未知条目被剔除; 无效输入返回 None
        out = BundleAdjustmentOptimizer.normalize_anchor_tags({2: {"xyz_mm": [1, 2, 3]}})
        self.assertEqual(out[2]["known"], [True, True, True])
        out = BundleAdjustmentOptimizer.normalize_anchor_tags({2: {"xyz_mm": [1, 2, 3], "known": [False] * 3}})
        self.assertIsNone(out)
        self.assertIsNone(BundleAdjustmentOptimizer.normalize_anchor_tags(None))
        self.assertIsNone(BundleAdjustmentOptimizer.normalize_anchor_tags({}))

    def test_evaluate_anchor_dof(self):
        """测试配置级 DoF 记账: full / partial / none 及退化原因"""
        two_full = {0: {"xyz_mm": [0, 0, 405], "known": [True] * 3},
                    1: {"xyz_mm": [0, 520, 196], "known": [True] * 3}}
        self.assertEqual(BundleAdjustmentOptimizer.evaluate_anchor_dof(two_full)["mode"], "full")

        # 0 枚全知: 全部只知 XY (网格板场景) → partial (4/5 DoF)
        xy_only = {0: {"xyz_mm": [0, 0, 0], "known": [True, True, False]},
                   1: {"xyz_mm": [300, 0, 0], "known": [True, True, False]},
                   2: {"xyz_mm": [0, 400, 0], "known": [True, True, False]}}
        dof = BundleAdjustmentOptimizer.evaluate_anchor_dof(xy_only)
        self.assertEqual(dof["mode"], "partial")
        self.assertEqual(dof["dof_solved"], 4)

        # 单枚锚点: 无尺度对 → none
        single = {0: {"xyz_mm": [10, 20, 30], "known": [True] * 3}}
        dof = BundleAdjustmentOptimizer.evaluate_anchor_dof(single)
        self.assertEqual(dof["mode"], "none")
        self.assertIn("尺度", dof["reason"])

        # 无共同已知轴 (A 只知 X, B 只知 Y) → none
        disjoint = {0: {"xyz_mm": [0, 0, 0], "known": [True, False, False]},
                    1: {"xyz_mm": [0, 500, 0], "known": [False, True, False]}}
        self.assertEqual(BundleAdjustmentOptimizer.evaluate_anchor_dof(disjoint)["mode"], "none")

        # 仅 Z 已知对: 有尺度但无偏航 → none
        z_only = {0: {"xyz_mm": [0, 0, 100], "known": [False, False, True]},
                  1: {"xyz_mm": [0, 0, 300], "known": [False, False, True]}}
        dof = BundleAdjustmentOptimizer.evaluate_anchor_dof(z_only)
        self.assertEqual(dof["mode"], "none")
        self.assertIn("偏航", dof["reason"])

    def test_anchor_full_regression_two_full_tags(self):
        """回归: 双全知锚点 (Tag0/Tag1 旧配置) 走新求解器, 数学结果与旧闭式解一致"""
        scale_gt, yaw_gt = 1.25, 0.4
        t_gt = np.array([100.0, 50.0, 300.0])
        world_pts = {0: [0.0, 0.0, 405.0], 1: [0.0, 520.0, 196.0]}
        poses = _make_ba_poses(world_pts, scale_gt, yaw_gt, t_gt)

        anchor_tags = {tid: {"xyz_mm": list(p), "known": [True] * 3} for tid, p in world_pts.items()}
        result = self.optimizer.anchor_to_absolute_world(poses, anchor_tags, origin_tag_id=0, x_align_tag_id=1)

        self.assertEqual(result["anchor_mode"], "full")
        self.assertAlmostEqual(result["world_anchor"]["scale_factor"], scale_gt, places=5)
        # 锚点精确落位 (与旧闭式解 "origin 精确落在绝对坐标" 行为一致)
        np.testing.assert_allclose(result["tags"][0]["position_mm"], [0.0, 0.0, 405.0], atol=0.01)
        np.testing.assert_allclose(result["tags"][1]["position_mm"], [0.0, 520.0, 196.0], atol=0.01)
        # 锚点残差 ≈ 0
        self.assertLess(result["world_anchor"]["anchor_residual_mm"]["max_mm"], 0.01)
        # 尺度同步作用于边长模型
        self.assertAlmostEqual(self.optimizer.marker_size_mm, 50.0 * scale_gt, places=2)
        # 验证单真理源 anchor_tags 字典格式
        self.optimizer.marker_size_mm = 50.0
        anchors_dict = {0: {"xyz_mm": world_pts[0]},
                        1: {"xyz_mm": world_pts[1]}}
        result2 = self.optimizer.anchor_to_absolute_world(poses, anchors_dict)
        self.assertEqual(result2["anchor_mode"], "full")
        np.testing.assert_allclose(result2["tags"][1]["position_mm"], [0.0, 520.0, 196.0], atol=0.01)

    def test_anchor_partial_xy_only(self):
        """测试 0 枚全知锚点 (仅 XY 已知) → partial 模式: XY 绝对锚定, Z 保持 BA 尺度相对坐标"""
        scale_gt, yaw_gt = 1.1, -0.3
        t_gt = np.array([50.0, 60.0, 123.0])
        world_pts = {0: [0.0, 0.0, 0.0], 1: [300.0, 0.0, 0.0], 2: [0.0, 400.0, 0.0]}
        poses = _make_ba_poses(world_pts, scale_gt, yaw_gt, t_gt)
        z_ba = {tid: float(poses[tid][:3, 3][2]) for tid in poses}

        anchor_tags = {tid: {"xyz_mm": [p[0], p[1], 0.0], "known": [True, True, False]}
                       for tid, p in world_pts.items()}
        result = self.optimizer.anchor_to_absolute_world(poses, anchor_tags)

        self.assertEqual(result["anchor_mode"], "partial")
        # XY 精确落位
        for tid, p_w in world_pts.items():
            pos = result["tags"][tid]["position_mm"]
            self.assertAlmostEqual(pos[0], p_w[0], places=1)
            self.assertAlmostEqual(pos[1], p_w[1], places=1)
            # Z = s·z_ba (t_z 悬空保持 0)
            self.assertAlmostEqual(pos[2], scale_gt * z_ba[tid], places=1)
        self.assertAlmostEqual(result["world_anchor"]["scale_factor"], scale_gt, places=5)

    def test_anchor_none_fallback(self):
        """测试约束不足 → none 模式退化相对对齐 (含配置锚点未被检出的情形)"""
        # 单枚全知锚点: 无尺度来源 → 相对对齐 + none 标记
        poses = {0: np.eye(4), 1: np.eye(4)}
        poses[1][:3, 3] = [400.0, 0.0, 0.0]
        anchor_tags = {0: {"xyz_mm": [10.0, 20.0, 30.0], "known": [True] * 3}}
        result = self.optimizer.anchor_to_absolute_world(poses, anchor_tags)
        self.assertEqual(result["anchor_mode"], "none")
        self.assertTrue(result.get("anchor_skip_reason"))
        # 相对对齐行为: Tag 0 归零
        np.testing.assert_allclose(result["tags"][0]["position_mm"], [0.0, 0.0, 0.0], atol=1e-6)

        # 配置双锚点但 Tag 5 未被检出 → 可用约束只剩 1 枚 → none
        anchor_tags = {0: {"xyz_mm": [0.0, 0.0, 405.0], "known": [True] * 3},
                       5: {"xyz_mm": [0.0, 520.0, 196.0], "known": [True] * 3}}
        result = self.optimizer.anchor_to_absolute_world(poses, anchor_tags)
        self.assertEqual(result["anchor_mode"], "none")
        self.assertTrue(result.get("anchor_skip_reason"))

    def test_load_anchor_tags_migration(self):
        """【无向后兼容原则】旧 world_anchor 不再支持，仅加载标准的 calibration.anchor_tags"""
        with tempfile.TemporaryDirectory() as td:
            # 旧 world_anchor 格式不再被解析
            p_old = os.path.join(td, "old.yaml")
            with open(p_old, "w", encoding="utf-8") as f:
                f.write("calibration:\n"
                        "  world_anchor:\n"
                        "    origin_tag_id: 0\n"
                        "    origin_xyz_mm: [0.0, 0.0, 405.0]\n"
                        "    align_tag_id: 1\n"
                        "    align_xyz_mm: [0.0, 520.0, 196.0]\n")
            anchors_old = load_anchor_tags(p_old)
            self.assertEqual(anchors_old, {}, "旧格式不再解析为有效锚点")

            # 新 anchor_tags 格式 (含部分已知与 null 轴)
            p_new = os.path.join(td, "new.yaml")
            with open(p_new, "w", encoding="utf-8") as f:
                f.write("calibration:\n"
                        "  anchor_tags:\n"
                        "    0:\n"
                        "      xyz_mm: [0.0, 0.0, 405.0]\n"
                        "    5:\n"
                        "      xyz_mm: [100.0, 200.0, null]\n")
            anchors = load_anchor_tags(p_new)
            self.assertEqual(set(anchors.keys()), {0, 5})
            self.assertEqual(anchors[5]["known"], [True, True, False])

    def test_anchor_umeyama_3d_with_roll_pitch(self):
        """测试 >=3 枚全知锚点时自动触发 Umeyama 3D 最优相似变换，消除标靶自身倾角对世界系的绑架"""
        # 构造带有 3D 旋转 (包含 roll 和 pitch) 的世界系与 BA 系点云
        world_pts = {
            5: [0.0, 0.0, 0.0],
            6: [348.0, 0.0, 0.0],
            7: [0.0, 470.0, 0.0],
            8: [348.0, 470.0, 0.0]
        }
        # 绕任意 3D 轴旋转 R_gt
        rx, ry, rz = np.radians(3.5), np.radians(-2.1), np.radians(25.0)
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        R_gt = Rz @ Ry @ Rx
        scale_gt = 0.885
        t_gt = np.array([45.0, -30.0, 12.0])

        # 反推 BA 系标靶位置: p_w = s * R @ p_ba + t  =>  p_ba = (1/s) * R^T @ (p_w - t)
        poses = {}
        for tid, pw in world_pts.items():
            p_ba = (1.0 / scale_gt) * (R_gt.T @ (np.array(pw) - t_gt))
            T = np.eye(4)
            T[:3, 3] = p_ba
            poses[tid] = T

        anchor_tags = {tid: {"xyz_mm": list(pw), "known": [True, True, True]} for tid, pw in world_pts.items()}
        result = self.optimizer.anchor_to_absolute_world(poses, anchor_tags, origin_tag_id=5, x_align_tag_id=6)

        self.assertEqual(result["anchor_mode"], "full")
        self.assertEqual(result["world_anchor"]["solver_type"], "umeyama_3d")
        self.assertAlmostEqual(result["world_anchor"]["scale_factor"], scale_gt, places=4)

        # 检查各标靶对齐到世界坐标系后的残差 (应 < 0.01mm)
        for tid, pw in world_pts.items():
            pos = result["tags"][tid]["position_mm"]
            np.testing.assert_allclose(pos, pw, atol=0.05)

        self.assertLess(result["world_anchor"]["anchor_residual_mm"]["max_mm"], 0.05)

    def test_anchor_conflict_detection(self):
        """测试锚点世界坐标输入冲突检测: 几何形变过大时应记录 conflict_pairs 警示"""
        poses = {
            5: np.eye(4),
            6: np.eye(4),
            7: np.eye(4)
        }
        poses[5][:3, 3] = [0.0, 0.0, 0.0]
        poses[6][:3, 3] = [350.0, 0.0, 0.0]
        poses[7][:3, 3] = [0.0, 500.0, 0.0]

        # 故意给 Tag 7 录入冲突的世界坐标 (误录在 X 轴上)
        anchor_tags = {
            5: {"xyz_mm": [0.0, 0.0, 0.0], "known": [True, True, True]},
            6: {"xyz_mm": [350.0, 0.0, 0.0], "known": [True, True, True]},
            7: {"xyz_mm": [450.0, 0.0, 0.0], "known": [True, True, True]}
        }
        mode, info = self.optimizer.solve_similarity_from_anchors(poses, anchor_tags)
        self.assertTrue(len(info["conflict_pairs"]) > 0)
        # 应检测出 (6, 7) 之间的几何距离冲突
        conflicted_tags = [c["pair"] for c in info["conflict_pairs"]]
        self.assertIn((6, 7), conflicted_tags)

    def test_free_ba_without_anchors(self):
        """阶段一验证: 无任何锚点配置时，自由平差依然 100% 收敛且输出相对底图"""
        T_w_t0 = np.eye(4)
        T_w_t1 = np.eye(4)
        T_w_t1[0, 3] = 300.0  # Tag 1 在 X=300mm 处

        rvec_c1 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        tvec_c1 = np.array([0.0, 0.0, 1000.0], dtype=np.float64)
        rvec_c2 = np.array([0.0, 0.1, 0.0], dtype=np.float64)
        tvec_c2 = np.array([50.0, 0.0, 980.0], dtype=np.float64)

        def project(T_w_t, rv, tv):
            R_c, _ = cv2.Rodrigues(rv)
            T_c_w = np.eye(4)
            T_c_w[:3, :3] = R_c
            T_c_w[:3, 3] = tv
            T_c_t = T_c_w @ T_w_t
            r_t, _ = cv2.Rodrigues(T_c_t[:3, :3])
            t_t = T_c_t[:3, 3]
            pts2d, _ = cv2.projectPoints(self.optimizer.obj_points, r_t, t_t, self.optimizer.camera_matrix, self.optimizer.dist_coeffs)
            return pts2d.reshape(4, 2)

        detections = [
            {0: project(T_w_t0, rvec_c1, tvec_c1), 1: project(T_w_t1, rvec_c1, tvec_c1)},
            {0: project(T_w_t0, rvec_c2, tvec_c2), 1: project(T_w_t1, rvec_c2, tvec_c2)}
        ]

        # 阶段一: 完全不传 anchor_tags
        rel_map = self.optimizer.optimize(
            detections,
            active_frame_names=["f0", "f1"],
            origin_tag_id=0,
            x_align_tag_id=1,
            anchor_tags=None
        )

        self.assertIsNotNone(rel_map)
        self.assertEqual(rel_map["anchor_mode"], "unaligned")
        self.assertIn("raw_relative_poses", rel_map)
        self.assertLess(rel_map["final_rmse"], 0.2)
        # 标靶 0 作为相对原点
        np.testing.assert_allclose(rel_map["tags"][0]["position_mm"], [0.0, 0.0, 0.0], atol=0.01)

    def test_independent_world_alignment(self):
        """阶段二验证: 基于阶段一相对底图，独立调用 align_relative_map_to_world 完成世界坐标系校准"""
        # 模拟阶段一的相对底图
        raw_map = {
            "origin_tag_id": 5,
            "x_axis_align_tag_id": 6,
            "anchor_mode": "unaligned",
            "marker_size_mm": 50.0,
            "final_rmse": 0.15,
            "raw_relative_poses": {
                5: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                6: [[1, 0, 0, 200], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                7: [[1, 0, 0, 0], [0, 1, 0, 300], [0, 0, 1, 0], [0, 0, 0, 1]]
            }
        }
        # 用户在白名单配置的已知世界坐标
        anchor_tags = {
            5: {"xyz_mm": [100.0, 100.0, 0.0], "known": [True, True, True]},
            6: {"xyz_mm": [300.0, 100.0, 0.0], "known": [True, True, True]},
            7: {"xyz_mm": [100.0, 400.0, 0.0], "known": [True, True, True]}
        }

        world_map = self.optimizer.align_relative_map_to_world(
            relative_map=raw_map,
            anchor_tags=anchor_tags,
            origin_tag_id=5,
            x_align_tag_id=6,
            strict=True
        )

        self.assertEqual(world_map["anchor_mode"], "full")
        self.assertEqual(world_map["world_anchor"]["solver_type"], "umeyama_3d")
        np.testing.assert_allclose(world_map["tags"][5]["position_mm"], [100.0, 100.0, 0.0], atol=0.01)
        np.testing.assert_allclose(world_map["tags"][6]["position_mm"], [300.0, 100.0, 0.0], atol=0.01)
        np.testing.assert_allclose(world_map["tags"][7]["position_mm"], [100.0, 400.0, 0.0], atol=0.01)


if __name__ == "__main__":
    unittest.main()
