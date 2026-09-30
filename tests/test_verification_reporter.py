#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VerificationReporter 单元测试 (test_verification_reporter.py)
============================================================
覆盖核心特性：
  1. 空结果安全边界处理；
  2. 多 Tag / 多 Frame 统计中位数与均值聚合；
  3. MAD 稳健统计与异常回审帧标记；
  4. 系统性偏差标靶标记 (中位误差 > 2倍全局中位数)；
  5. Markdown 报告完整渲染与文件导出。
"""

import os
import sys
import unittest
import tempfile
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.calibration.verification.verification_reporter import VerificationReporter


class TestVerificationReporter(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_empty_results_handling(self):
        """测试空结果输入时的防御与兜底"""
        stats = VerificationReporter.aggregate_statistics([])
        self.assertFalse(stats["pass"])
        self.assertEqual(stats["total_tests"], 0)
        self.assertEqual(stats["global_median_px"], 0.0)

    def test_aggregate_statistics_grading_and_flags(self):
        """测试正常样本与异常样本的分级判定"""
        # 1. 达标 PASS 样本
        good_results = [
            {"image": "frame_01.png", "blind_tag_id": 18, "err_px": 0.8, "err_mm": 0.3},
            {"image": "frame_01.png", "blind_tag_id": 19, "err_px": 0.9, "err_mm": 0.4},
            {"image": "frame_02.png", "blind_tag_id": 18, "err_px": 0.7, "err_mm": 0.3},
            {"image": "frame_02.png", "blind_tag_id": 19, "err_px": 1.0, "err_mm": 0.5},
        ]
        stats_good = VerificationReporter.aggregate_statistics(good_results)
        self.assertTrue(stats_good["pass"])
        self.assertEqual(stats_good["grade"], "PASS")
        self.assertEqual(len(stats_good["flagged_tags"]), 0)
        self.assertEqual(len(stats_good["flagged_frames"]), 0)

        # 2. 注入系统性偏差 Tag (Tag #99 误差明显过大)
        biased_results = good_results + [
            {"image": "frame_01.png", "blind_tag_id": 99, "err_px": 4.5, "err_mm": 3.0},
            {"image": "frame_02.png", "blind_tag_id": 99, "err_px": 5.0, "err_mm": 3.2},
        ]
        stats_biased = VerificationReporter.aggregate_statistics(biased_results)
        self.assertIn(99, stats_biased["flagged_tags"], "Tag 99 应被标记为系统性偏差嫌疑标靶")

    def test_export_markdown_report_generation(self):
        """测试 Markdown 报告文件写盘与核心区块内容"""
        results = [
            {"image": "view_01.png", "blind_tag_id": 18, "err_px": 1.1, "err_mm": 0.5},
            {"image": "view_01.png", "blind_tag_id": 19, "err_px": 1.2, "err_mm": 0.6},
        ]
        stats = VerificationReporter.aggregate_statistics(results)
        out_file = os.path.join(self.tmp_dir, "test_report.md")

        saved_path = VerificationReporter.export_markdown_report(
            stats=stats,
            all_results=results,
            map_path="dummy_map.yaml",
            image_dir="dummy_img_dir",
            actual_source="manifest",
            vis_dir=self.tmp_dir,
            out_path=out_file
        )

        self.assertTrue(os.path.exists(saved_path), "报告文件必须成功生成")
        with open(saved_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("离线标定精度体检与 Leave-One-Out 盲测批量验证报告", content)
        self.assertIn("LOO 盲测中位像元误差", content)
        self.assertIn("Tag #18", content)
        self.assertIn("view_01.png", content)


if __name__ == "__main__":
    unittest.main()
