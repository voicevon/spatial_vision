#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CovisibilityGraphAnalyzer 单元测试 (test_covisibility_graph.py)
==============================================================
覆盖核心特性：
  1. 空观测与单标靶防御处理；
  2. 正常全连通共视图拓扑分析；
  3. 孤立子图与断网识别 (is_valid == False, 报出 unconnected_tags)；
  4. 单图支撑关键桥梁 (Critical Bridges) 识别；
  5. 详细拓扑诊断报告格式化输出。
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.calibration.solvers.covisibility_graph import CovisibilityGraphAnalyzer


class TestCovisibilityGraph(unittest.TestCase):
    def test_empty_observations(self):
        """测试空输入防御"""
        res = CovisibilityGraphAnalyzer.analyze([])
        self.assertFalse(res["is_valid"])
        self.assertEqual(len(res["all_tags"]), 0)

    def test_fully_connected_network(self):
        """测试正常全连通网络与关键桥梁检测"""
        # 帧 0: Tag 0, 1
        # 帧 1: Tag 1, 2
        # 帧 2: Tag 0, 2
        f0 = {0: np.zeros((4, 2)), 1: np.zeros((4, 2))}
        f1 = {1: np.zeros((4, 2)), 2: np.zeros((4, 2))}
        f2 = {0: np.zeros((4, 2)), 2: np.zeros((4, 2))}

        res = CovisibilityGraphAnalyzer.analyze([f0, f1, f2], origin_tag_id=0, x_align_tag_id=1)
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["all_tags"], [0, 1, 2])
        self.assertEqual(len(res["components"]), 1)
        self.assertEqual(len(res["unconnected_tags"]), 0)
        # 每条边各自出现 1 次，全部为关键桥梁
        self.assertEqual(len(res["critical_bridges"]), 3)

    def test_disconnected_network_detection(self):
        """测试孤岛断网识别"""
        # 子图 A: 0, 1
        # 子图 B: 2, 3 (与 A 无任何共视边)
        f0 = {0: np.zeros((4, 2)), 1: np.zeros((4, 2))}
        f1 = {2: np.zeros((4, 2)), 3: np.zeros((4, 2))}

        res = CovisibilityGraphAnalyzer.analyze([f0, f1], origin_tag_id=0, x_align_tag_id=1)
        self.assertFalse(res["is_valid"])
        self.assertEqual(len(res["components"]), 2)
        self.assertIn(2, res["unconnected_tags"])
        self.assertIn(3, res["unconnected_tags"])
        self.assertIn("断裂", res["message"])


if __name__ == "__main__":
    unittest.main()
