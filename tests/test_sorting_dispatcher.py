#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
业务分选调度器单元测试 (TestSortingDispatcher)
==============================================
验证 Phase 2 调度核心能力：
1. Smart ROI 解析与 Source/Destination 槽位建立
2. 智能候选物料挑选 (顶层优先、凸起优先)
3. 品质分级 (A/B/C) 槽位路由
4. 满溢上限防护与备用槽切换
5. 端到端联动: 感知目标 -> 调度决策 -> 生成针对动态槽位的 G-code 指令
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.calibration.roi_manager import RoiDefinition
from src.vision.asparagus_analyzer import AsparagusTarget
from src.control.sorting_dispatcher import SortingDispatcher, DestinationSlot
from src.control.scara_motion_planner import ScaraMotionPlanner


class TestSortingDispatcher(unittest.TestCase):
    """分选调度器测试套件"""

    def setUp(self):
        # 模拟工业场景：1 条进料皮带 (Source) + 3 个落料槽位 (Slot 1=A级, Slot 2=B级, Slot 3=通用备用)
        self.rois = [
            RoiDefinition(
                roi_id="roi_belt",
                name="进料皮带",
                role="source",
                target_intent="pose_pick",
                category="belt",
                center_xyz_mm=[0.0, 300.0, 50.0],
            ),
            RoiDefinition(
                roi_id="roi_slot_A",
                name="A级料槽",
                role="destination",
                target_intent="piece_count",
                category="tray",
                center_xyz_mm=[200.0, -100.0, 10.0],
                binding={"slot_index": 0, "capacity_max": 2, "grades": ["A"]},
            ),
            RoiDefinition(
                roi_id="roi_slot_B",
                name="B级料槽",
                role="destination",
                target_intent="piece_count",
                category="tray",
                center_xyz_mm=[200.0, 0.0, 10.0],
                binding={"slot_index": 1, "capacity_max": 2, "grades": ["B"]},
            ),
            RoiDefinition(
                roi_id="roi_slot_overflow",
                name="综合备用槽",
                role="destination",
                target_intent="piece_count",
                category="tray",
                center_xyz_mm=[200.0, 100.0, 10.0],
                binding={"slot_index": 2, "capacity_max": 5, "grades": ["A", "B", "C"]},
            ),
        ]
        self.dispatcher = SortingDispatcher(rois=self.rois)

    def _make_dummy_target(self, tid: int, grade: str = "A", is_topmost: bool = True,
                           rel_h: float = 20.0, robot_xy=(100.0, 200.0)) -> AsparagusTarget:
        return AsparagusTarget(
            id=tid,
            center_px=(300.0, 300.0),
            length_px=150.0,
            diam_px=15.0,
            yaw_deg=20.0,
            axis_vector=(1.0, 0.0),
            box_corners=np.zeros((4, 2)),
            contour=np.zeros((10, 1, 2)),
            length_mm=180.0,
            diam_mm=14.0,
            grip_x=0.0,
            grip_y=0.0,
            grip_z=500.0,
            z_top=480.0,
            rel_height_mm=rel_h,
            robot_x=robot_xy[0],
            robot_y=robot_xy[1],
            robot_z=15.0,
            robot_r=20.0,
            is_topmost=is_topmost,
            calibration_source="tag_online",
            grade=grade,
            confidence=0.9,
        )

    def test_rois_loading(self):
        """测试 Smart ROI 装载与槽位初始化"""
        self.assertEqual(len(self.dispatcher.source_rois), 1)
        self.assertEqual(len(self.dispatcher.destination_slots), 3)

        slot0 = self.dispatcher.destination_slots[0]
        self.assertEqual(slot0.name, "A级料槽")
        self.assertEqual(slot0.target_grades, ["A"])
        self.assertEqual(slot0.capacity_max, 2)
        self.assertEqual(slot0.current_count, 0)
        self.assertFalse(slot0.is_full)

    def test_select_best_target_topmost_priority(self):
        """测试最优目标挑选策略：顶层且凸起最高者优先"""
        t_bottom = self._make_dummy_target(1, is_topmost=False, rel_h=5.0)
        t_top_lower = self._make_dummy_target(2, is_topmost=True, rel_h=10.0)
        t_top_highest = self._make_dummy_target(3, is_topmost=True, rel_h=25.0)

        best = self.dispatcher.select_best_target([t_bottom, t_top_lower, t_top_highest])
        self.assertIsNotNone(best)
        self.assertEqual(best.id, 3)

    def test_grade_routing_and_overflow(self):
        """测试品质分级路由与槽位满溢防护"""
        t_a1 = self._make_dummy_target(1, grade="A")
        t_a2 = self._make_dummy_target(2, grade="A")
        t_a3 = self._make_dummy_target(3, grade="A")

        # 1. 抓取第 1 根 A 级，应该分配至 0 号槽 (A级料槽)
        _, slot, task = self.dispatcher.dispatch_cycle([t_a1])
        self.assertIsNotNone(slot)
        self.assertEqual(slot.slot_index, 0)
        self.assertEqual(task.drop_x, 200.0)
        self.assertEqual(task.drop_y, -100.0)

        # 模拟执行完成
        self.dispatcher.confirm_placed(0)
        self.assertEqual(self.dispatcher.destination_slots[0].current_count, 1)

        # 2. 抓取第 2 根 A 级，仍进入 0 号槽
        _, slot, _ = self.dispatcher.dispatch_cycle([t_a2])
        self.assertEqual(slot.slot_index, 0)
        self.dispatcher.confirm_placed(0)
        self.assertEqual(self.dispatcher.destination_slots[0].current_count, 2)
        self.assertTrue(self.dispatcher.destination_slots[0].is_full)

        # 3. 抓取第 3 根 A 级，0 号槽已满，自动溢出至 2 号槽 (综合备用槽)
        _, slot, task = self.dispatcher.dispatch_cycle([t_a3])
        self.assertEqual(slot.slot_index, 2)
        self.assertEqual(slot.name, "综合备用槽")
        self.assertEqual(task.drop_y, 100.0)
        self.dispatcher.confirm_placed(2)

        # 4. 人工换料，复位 0 号槽
        self.dispatcher.reset_slot(0)
        self.assertEqual(self.dispatcher.destination_slots[0].current_count, 0)
        self.assertFalse(self.dispatcher.destination_slots[0].is_full)

        # 5. 再来一根 A 级，重新优先进入 0 号槽
        t_a4 = self._make_dummy_target(4, grade="A")
        _, slot, _ = self.dispatcher.dispatch_cycle([t_a4])
        self.assertEqual(slot.slot_index, 0)

    def test_end_to_end_planner_integration(self):
        """测试端到端联动: 感知目标 -> 调度器决策 -> 轨迹规划器生成动态 G-code"""
        planner = ScaraMotionPlanner(safe_z=80.0, feedrate_xy=4000)
        target_b = self._make_dummy_target(10, grade="B", robot_xy=(150.0, 80.0))

        # 调度决策
        target, slot, task = self.dispatcher.dispatch_cycle([target_b])
        self.assertIsNotNone(task)
        self.assertEqual(slot.slot_index, 1)  # B 级槽

        # 轨迹规划器直接消费调度器产出的 task
        gcode = planner.plan(task)

        # 验证生成的 G-code 包含从 Pick 坐标 (150, 80) 到 B 槽 Drop 坐标 (200, 0) 的精准指令
        self.assertIn("G0 X150.00 Y80.00", gcode)
        self.assertIn("G0 X200.00 Y0.00", gcode)
        self.assertIn("槽位#2[B级料槽]", gcode)


if __name__ == "__main__":
    unittest.main()
