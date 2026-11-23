#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
子坐标系外参反推与 unknown 状态机单元测试
========================================
验证:
1. 默认 unknown 状态下子坐标系及其下游 ROI 的 is_resolved 为 False；
2. FrameExtrinsicSolver 多标靶 3D 刚体配准反推外参与 RMSE 精度；
3. 双标靶定轴与单标靶偏移反推；
4. 多级父子坐标系树的拓扑级联求解与状态写穿；
5. YAML 序列化与反序列化状态持久化。
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.workspace.coordinate_manager import (
    CoordinateTreeManager,
    FrameDefinition,
    make_transform_matrix,
    rot_mat_to_rpy_deg,
    rpy_deg_to_rot_mat,
)
from src.workspace.roi_manager import (
    RoiSpaceManager,
    RoiDefinition,
)
from src.calibration.solvers.frame_extrinsic_solver import (
    FrameExtrinsicSolver,
    rigid_transform_3d,
)


class TestFrameExtrinsicSolver(unittest.TestCase):

    def setUp(self):
        self.coord_mgr = CoordinateTreeManager(workspace_id="test_ws")

    def test_unknown_fixed_transform_blocks_resolution(self):
        """测试未知外参的子坐标系默认不可解，阻断下游坐标系与 ROI 空间"""
        # 创建默认 unknown 子坐标系
        unknown_frame = FrameDefinition(
            frame_id="frame_conveyor",
            name="主输送机",
            parent_frame_id="world",
            type="fixed_transform",
            status="unknown",
            translation_xyz_mm=None,
            rotation_rpy_deg=None,
        )
        self.assertEqual(unknown_frame.status, "unknown")
        self.coord_mgr.add_frame(unknown_frame)

        # 1. 相对父级与绝对到世界变换必须返回 is_resolved = False
        _, rel_valid = self.coord_mgr.get_relative_transform_to_parent("frame_conveyor")
        self.assertFalse(rel_valid, "未标定子坐标系的相对变换必须为 False")

        _, world_valid = self.coord_mgr.get_frame_to_world("frame_conveyor")
        self.assertFalse(world_valid, "未标定子坐标系到世界系的变换必须为 False")

        # 2. 级联测试：挂载在未知子系下的二级子系也必须不可解
        sub_frame = FrameDefinition(
            frame_id="frame_bracket",
            name="支架",
            parent_frame_id="frame_conveyor",
            type="fixed_transform",
            translation_xyz_mm=[10.0, 0.0, 0.0],
            rotation_rpy_deg=[0.0, 0.0, 0.0],
        )
        self.coord_mgr.add_frame(sub_frame)
        _, sub_world_valid = self.coord_mgr.get_frame_to_world("frame_bracket")
        self.assertFalse(sub_world_valid, "父坐标系未知时，二级子坐标系也必须继承不可解状态")

        # 3. 挂载在未知子系下的 ROI 物件必须守门，is_resolved 必须为 False
        roi_mgr = RoiSpaceManager()
        test_roi = RoiDefinition(
            roi_id="roi_test",
            name="测试工作面",
            frame_id="frame_conveyor",
            center_xyz_mm=[0.0, 0.0, 0.0],
            size_xyz_mm=[100.0, 100.0, 10.0],
        )
        roi_mgr.add_roi(test_roi)
        obb = roi_mgr.get_roi_world_obb("roi_test", self.coord_mgr)
        self.assertIsNotNone(obb)
        self.assertFalse(obb["is_resolved"], "未标定坐标系下的 ROI 物件必须处于未解算待定状态")

    def test_multi_tag_registration_extrinsic_solver(self):
        """测试多标靶 3D 刚体配准精确反推外参矩阵与残差"""
        # 设定真值刚体变换: T_world_from_conveyor
        R_gt = rpy_deg_to_rot_mat([5.0, -10.0, 30.0])
        t_gt = np.array([250.0, -120.0, 80.0])
        T_gt = make_transform_matrix(R_gt, t_gt.tolist())

        # 在输送机局部坐标系下设定 4 枚标靶名义坐标
        local_tags = {
            10: [0.0, 0.0, 0.0],
            11: [200.0, 0.0, 0.0],
            12: [200.0, 150.0, 0.0],
            13: [0.0, 150.0, 0.0],
        }

        # 投影计算世界系下的标靶实测坐标 (加入少量合成测量噪声 0.02mm)
        rng = np.random.default_rng(123)
        tags_map = {}
        for tid, p_loc in local_tags.items():
            p_w = (T_gt @ np.array([p_loc[0], p_loc[1], p_loc[2], 1.0]))[:3]
            p_w_noise = p_w + rng.normal(0, 0.02, 3)
            T_w_t = make_transform_matrix(R_gt, p_w_noise.tolist())
            tags_map[tid] = T_w_t

        whitelist_anchors = {tid: {"xyz_mm": pos} for tid, pos in local_tags.items()}
        solver = FrameExtrinsicSolver(tags_map=tags_map, whitelist_anchors=whitelist_anchors)

        # 定义待标定子坐标系
        conveyor_frame = FrameDefinition(
            frame_id="frame_conveyor",
            name="主输送机",
            parent_frame_id="world",
            type="fixed_transform",
            status="unknown",
            calibration_spec={
                "method": "anchor_tags_registration",
                "reference_tag_ids": list(local_tags.keys()),
            }
        )

        succ, t_est, rpy_est, rmse, msg = solver.solve_frame_extrinsic(conveyor_frame, self.coord_mgr)
        self.assertTrue(succ, f"反推应成功: {msg}")
        self.assertIsNotNone(t_est)
        self.assertIsNotNone(rpy_est)

        # 校验外参平移与旋转恢复精度
        np.testing.assert_allclose(t_est, t_gt, atol=0.1, err_msg="平移外参反推误差应在 0.1mm 以内")
        np.testing.assert_allclose(rpy_est, [5.0, -10.0, 30.0], atol=0.1, err_msg="欧拉角外参反推误差应在 0.1° 以内")
        self.assertLess(rmse, 0.1, f"拟合 RMSE 应极小: {rmse}")

    def test_solve_by_axis_align(self):
        """测试基于原点标靶与 X 轴标靶的定轴反推"""
        # Tag 0 在 [100, 200, 50], Tag 1 在 [400, 200, 50] (沿世界 X 轴正向延伸 300mm)
        tags_map = {
            0: make_transform_matrix(np.eye(3), [100.0, 200.0, 50.0]),
            1: make_transform_matrix(np.eye(3), [400.0, 200.0, 50.0]),
        }
        solver = FrameExtrinsicSolver(tags_map=tags_map)

        axis_frame = FrameDefinition(
            frame_id="frame_axis",
            name="双靶定轴系",
            parent_frame_id="world",
            type="fixed_transform",
            status="unknown",
            calibration_spec={
                "method": "axis_align",
                "origin_tag_id": 0,
                "x_axis_tag_id": 1,
                "origin_local_xyz_mm": [0.0, 0.0, 0.0],
                "x_axis_local_xyz_mm": [300.0, 0.0, 0.0],
            }
        )

        succ, t_est, rpy_est, rmse, msg = solver.solve_frame_extrinsic(axis_frame, self.coord_mgr)
        self.assertTrue(succ)
        np.testing.assert_allclose(t_est, [100.0, 200.0, 50.0], atol=1e-3)
        # 朝向应与世界 X 轴平行，RPY 全 0
        np.testing.assert_allclose(rpy_est, [0.0, 0.0, 0.0], atol=1e-2)
        self.assertAlmostEqual(rmse, 0.0, places=2)

    def test_solve_by_two_tags_arbitrary_direction(self):
        """测试任意非 X 轴走向的双标靶刚体对齐 (如沿局部 Y 轴或任意斜向)"""
        # 设子坐标系在世界系中的真值外参: Yaw = 45°, t = [200.0, 300.0, 50.0]
        yaw_gt_deg = 45.0
        yaw_rad = np.radians(yaw_gt_deg)
        R_gt = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0.0],
            [np.sin(yaw_rad),  np.cos(yaw_rad), 0.0],
            [0.0,              0.0,             1.0]
        ])
        t_gt = np.array([200.0, 300.0, 50.0])

        # 两标靶在局部系中非纯 X 轴，例如类似皮带机走向: Tag 10 在 [0, 0, 0], Tag 11 在 [-30.0, 400.0, 0.0]
        p_child_10 = np.array([0.0, 0.0, 0.0])
        p_child_11 = np.array([-30.0, 400.0, 0.0])

        # 投影至世界系
        p_world_10 = R_gt @ p_child_10 + t_gt
        p_world_11 = R_gt @ p_child_11 + t_gt

        tags_map = {
            10: make_transform_matrix(R_gt, p_world_10.tolist()),
            11: make_transform_matrix(R_gt, p_world_11.tolist()),
        }
        whitelist_anchors = {
            10: {"xyz_mm": p_child_10.tolist()},
            11: {"xyz_mm": p_child_11.tolist()},
        }
        solver = FrameExtrinsicSolver(tags_map=tags_map, whitelist_anchors=whitelist_anchors)

        sub_frame = FrameDefinition(
            frame_id="frame_tilted",
            name="斜向标靶子系",
            parent_frame_id="world",
            type="fixed_transform",
            status="unknown",
            calibration_spec={
                "method": "anchor_tags_registration",
                "reference_tag_ids": [10, 11],
            }
        )

        succ, t_est, rpy_est, rmse, msg = solver.solve_frame_extrinsic(sub_frame, self.coord_mgr)
        self.assertTrue(succ, f"反推失败: {msg}")
        self.assertLess(rmse, 1e-4, f"残差应极小: {rmse}")
        np.testing.assert_allclose(t_est, t_gt, atol=1e-3, err_msg="平移应精准恢复")
        np.testing.assert_allclose(rpy_est, [0.0, 0.0, 45.0], atol=1e-3, err_msg="偏航角应精准恢复为 45°")

    def test_solve_all_unknown_frames_pipeline(self):
        """测试整树批量反推及状态写穿与持久化流程"""
        # 构建两级待反推坐标系:
        # world -> frame_conveyor (由 Tag 5, 6, 7 反推) -> frame_slider (局部平移 [0, 50, 0])
        tags_map = {
            5: make_transform_matrix(np.eye(3), [100.0, 0.0, 0.0]),
            6: make_transform_matrix(np.eye(3), [300.0, 0.0, 0.0]),
            7: make_transform_matrix(np.eye(3), [100.0, 200.0, 0.0]),
        }

        frame_conv = FrameDefinition(
            frame_id="frame_conveyor",
            name="输送机",
            parent_frame_id="world",
            type="fixed_transform",
            status="unknown",
            calibration_spec={
                "method": "anchor_tags_registration",
                "reference_tag_ids": [5, 6, 7],
            }
        )
        self.coord_mgr.add_frame(frame_conv)

        # 初始未标定
        self.assertEqual(frame_conv.status, "unknown")
        _, valid_before = self.coord_mgr.get_frame_to_world("frame_conveyor")
        self.assertFalse(valid_before)

        # 执行求解
        whitelist_anchors = {
            5: {"xyz_mm": [0.0, 0.0, 0.0]},
            6: {"xyz_mm": [200.0, 0.0, 0.0]},
            7: {"xyz_mm": [0.0, 200.0, 0.0]},
        }
        solver = FrameExtrinsicSolver(tags_map=tags_map, whitelist_anchors=whitelist_anchors)
        res = solver.solve_all_unknown_frames(self.coord_mgr)

        self.assertIn("frame_conveyor", res)
        self.assertTrue(res["frame_conveyor"]["success"])

        # 校验求解后状态已变为 calibrated 且具备有效外参
        updated_frame = self.coord_mgr.get_frame("frame_conveyor")
        self.assertEqual(updated_frame.status, "calibrated")
        self.assertIsNotNone(updated_frame.translation_xyz_mm)
        self.assertIsNotNone(updated_frame.calibration_metrics)

        # 校验可解性恢复
        T_w_c, valid_after = self.coord_mgr.get_frame_to_world("frame_conveyor")
        self.assertTrue(valid_after)
        np.testing.assert_allclose(T_w_c[:3, 3], [100.0, 0.0, 0.0], atol=1e-2)

        # 校验序列化与反序列化
        d = updated_frame.to_dict()
        self.assertEqual(d["status"], "calibrated")
        self.assertIn("calibration_metrics", d)
        restored = FrameDefinition.from_dict(d)
        self.assertEqual(restored.status, "calibrated")
        np.testing.assert_allclose(restored.translation_xyz_mm, updated_frame.translation_xyz_mm)

    def test_solve_by_two_tags_with_pitch_roll_eliminates_z_error(self):
        """测试存在 3D 空间倾角(包含 Pitch/Roll)的双标靶对齐，局部 Z 轴误差严格归零"""
        # 设子坐标系在世界系中存在 Pitch 和 Roll 倾角 (类似真实物理工装微小倾斜)
        rpy_gt = [3.5, -2.8, 85.0]
        R_gt = rpy_deg_to_rot_mat(rpy_gt)
        t_gt = np.array([500.0, 200.0, -100.0])
        T_gt = make_transform_matrix(R_gt, t_gt.tolist())

        # 子坐标系中名义 Z 均为 0.0
        p_child_a = np.array([0.0, 0.0, 0.0])
        p_child_b = np.array([-33.0, 490.0, 0.0])

        # 在世界坐标系中生成两标靶的实测坐标 (由于倾角，两标靶在世界系下 Z 轴有显著高度差)
        p_world_a = (T_gt @ np.append(p_child_a, 1.0))[:3]
        p_world_b = (T_gt @ np.append(p_child_b, 1.0))[:3]

        tags_map = {
            20: make_transform_matrix(R_gt, p_world_a.tolist()),
            21: make_transform_matrix(R_gt, p_world_b.tolist()),
        }
        whitelist_anchors = {
            20: {"xyz_mm": p_child_a.tolist()},
            21: {"xyz_mm": p_child_b.tolist()},
        }
        solver = FrameExtrinsicSolver(tags_map=tags_map, whitelist_anchors=whitelist_anchors)

        sub_frame = FrameDefinition(
            frame_id="frame_conveyor_3d",
            name="空间倾斜输送机",
            parent_frame_id="world",
            type="fixed_transform",
            status="unknown",
            calibration_spec={
                "method": "anchor_tags_registration",
                "reference_tag_ids": [20, 21],
            }
        )

        succ, t_est, rpy_est, rmse, msg = solver.solve_frame_extrinsic(sub_frame, self.coord_mgr)
        self.assertTrue(succ, f"反推应成功: {msg}")
        self.assertLess(rmse, 1e-4, f"残差应极小: {rmse}")

        # 校验恢复的外参姿态能完全吸收 Pitch 和 Roll
        np.testing.assert_allclose(t_est, t_gt, atol=1e-3)
        np.testing.assert_allclose(rpy_est, rpy_gt, atol=1e-2)

        # 核心断言：将世界实测坐标反投影回子坐标系后，两已知标靶的局部 Z 坐标必须严格为 0.00 mm
        T_est = make_transform_matrix(rpy_deg_to_rot_mat(rpy_est), t_est)
        T_inv = np.linalg.inv(T_est)
        p_loc_a = (T_inv @ np.append(p_world_a, 1.0))[:3]
        p_loc_b = (T_inv @ np.append(p_world_b, 1.0))[:3]

        self.assertAlmostEqual(p_loc_a[2], 0.0, places=4, msg="标靶 A 在子系下的 Z 坐标必须严格为 0.0 mm")
        self.assertAlmostEqual(p_loc_b[2], 0.0, places=4, msg="标靶 B 在子系下的 Z 坐标必须严格为 0.0 mm")


if __name__ == "__main__":
    unittest.main()

