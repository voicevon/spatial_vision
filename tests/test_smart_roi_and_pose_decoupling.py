#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart ROI 与位姿工作室解耦单元测试
==================================
验证 Phase 1 的所有核心改进：
1. Smart ROI 属性完备性与序列化/反序列化 (role, target_intent, binding...)
2. Workspace 元数据 production 配置节点的正确持久化
3. AsparagusTarget 纯 DTO 属性与纯净化验证 (无 generate_gcode)
4. ScaraMotionPlanner 控制层独立轨迹规划器逻辑
5. AsparagusPoseStudio 渲染与状态无 G-code 耦合验证
"""

import os
import sys
import tempfile
import shutil
import unittest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.calibration.roi_manager import RoiDefinition, RoiSpaceManager
from src.calibration.workspace_manager import WorkspaceManager, Workspace
from src.vision.asparagus_analyzer import AsparagusTarget
from src.control.scara_motion_planner import ScaraMotionPlanner, ScaraPickTask
from tools.asparagus_pose_studio.renderer import AsparagusPoseStudioRenderer


class TestSmartRoiAndPoseDecoupling(unittest.TestCase):
    """Smart ROI 与位姿工作室解耦测试套件"""

    def test_smart_roi_definition_defaults_and_serialization(self):
        """测试 Smart ROI 默认属性及序列化/反序列化"""
        # 1. 默认属性
        roi = RoiDefinition(roi_id="roi_belt", name="进料皮带")
        self.assertEqual(roi.role, "general")
        self.assertEqual(roi.target_intent, "general")
        self.assertEqual(roi.binding, {})
        self.assertEqual(roi.pipeline_override, "")
        self.assertEqual(roi.min_confidence, 0.5)

        # 2. 赋值为 Smart ROI
        roi.role = "source"
        roi.target_intent = "pose_pick"
        roi.pipeline_override = "feng_green_axis_v2"
        roi.min_confidence = 0.8
        roi.binding = {"channel": "conveyor_main", "speed_mm_s": 50.0}

        d = roi.to_dict()
        self.assertEqual(d["role"], "source")
        self.assertEqual(d["target_intent"], "pose_pick")
        self.assertEqual(d["pipeline_override"], "feng_green_axis_v2")
        self.assertEqual(d["binding"]["channel"], "conveyor_main")
        self.assertAlmostEqual(d["min_confidence"], 0.8)

        # 3. 反序列化
        roi_restored = RoiDefinition.from_dict(d)
        self.assertEqual(roi_restored.role, "source")
        self.assertEqual(roi_restored.target_intent, "pose_pick")
        self.assertEqual(roi_restored.pipeline_override, "feng_green_axis_v2")
        self.assertEqual(roi_restored.binding["channel"], "conveyor_main")
        self.assertAlmostEqual(roi_restored.min_confidence, 0.8)

    def test_workspace_production_metadata_persistence(self):
        """测试 Workspace 的 production 配置持久化与读取"""
        tmp_dir = tempfile.mkdtemp(prefix="test_ws_prod_")
        try:
            mgr = WorkspaceManager(workspaces_dir=tmp_dir)
            ws = mgr.create_workspace(alias="测试工位")

            # 初始状态默认具备生产工作流配置
            self.assertEqual(ws.production.get("mode"), "scara_sorting")

            # 写入全局生产工作流规范
            ws.production = {
                "mode": "isolate_wheels_sorting",
                "active_gui": "isolate_wheels_production",
                "cycle_interval_ms": 100,
                "hardware": {
                    "mqtt_topic": "flux/loader/device1",
                    "broker": "voicevon.vicp.io"
                }
            }
            ws.save_meta()

            # 重新加载验证
            ws_loaded = Workspace.load(ws.workspace_dir)
            self.assertIsNotNone(ws_loaded)
            self.assertEqual(ws_loaded.production["mode"], "isolate_wheels_sorting")
            self.assertEqual(ws_loaded.production["active_gui"], "isolate_wheels_production")
            self.assertEqual(ws_loaded.production["hardware"]["broker"], "voicevon.vicp.io")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_asparagus_target_pure_dto(self):
        """验证 AsparagusTarget 纯 DTO 属性与解耦"""
        t = AsparagusTarget(
            id=1,
            center_px=(320.0, 240.0),
            length_px=200.0,
            diam_px=20.0,
            yaw_deg=30.0,
            axis_vector=(1.0, 0.0),
            box_corners=np.zeros((4, 2)),
            contour=np.zeros((10, 1, 2)),
            length_mm=180.0,
            diam_mm=12.5,
            grip_x=0.0,
            grip_y=0.0,
            grip_z=500.0,
            z_top=480.0,
            rel_height_mm=20.0,
            robot_x=150.0,
            robot_y=50.0,
            robot_z=15.0,
            robot_r=30.0,
            is_topmost=True,
            calibration_source="tag_online",
            straightness_ratio=0.98,
            grade="A",
            confidence=0.95,
        )
        self.assertEqual(t.grade, "A")
        self.assertAlmostEqual(t.straightness_ratio, 0.98)
        self.assertAlmostEqual(t.confidence, 0.95)

        # 核心断言: 目标类中彻底不存在 generate_gcode 方法
        self.assertFalse(hasattr(t, "generate_gcode"))

    def test_scara_motion_planner_independent(self):
        """测试独立的 SCARA 运动轨迹规划器"""
        planner = ScaraMotionPlanner(safe_z=85.0, feedrate_xy=4500, feedrate_z=1200)

        # 拾取与指定槽位落料 (例如 drop 位于槽位 3)
        gcode = planner.plan_from_target(
            target_id=2,
            pick_x=120.5,
            pick_y=60.0,
            pick_z=15.0,
            pick_r=45.0,
            drop_x=300.0,
            drop_y=-50.0,
            slot_index=2,
            tag_info="AprilTag 在线解算"
        )

        self.assertIn("G90", gcode)
        self.assertIn("G0 Z85.0 F4500", gcode)
        self.assertIn("G0 X120.50 Y60.00 R45.00 F4500", gcode)
        self.assertIn("M3", gcode)
        self.assertIn("G1 Z15.00 F1200", gcode)
        self.assertIn("M4", gcode)
        self.assertIn("G0 X300.00 Y-50.00 R0.00 F4500", gcode)
        self.assertIn("slot_index=2", gcode)

    def test_pose_studio_renderer_no_gcode_artifacts(self):
        """验证 Pose Studio 渲染器中不存在 G-code 导出按钮与脏文本框"""
        dummy_target = AsparagusTarget(
            id=1,
            center_px=(100.0, 100.0),
            length_px=100.0,
            diam_px=10.0,
            yaw_deg=15.0,
            axis_vector=(1.0, 0.0),
            box_corners=np.zeros((4, 2)),
            contour=np.zeros((10, 1, 2)),
            length_mm=150.0,
            diam_mm=12.0,
            grip_x=0.0,
            grip_y=0.0,
            grip_z=500.0,
            z_top=480.0,
            rel_height_mm=20.0,
            grade="A",
            robot_x=100.0,
            robot_y=50.0,
            robot_z=10.0,
            robot_r=15.0,
            is_topmost=True,
            calibration_source="tag_online"
        )

        class DummyState:
            mouse_pos = (0, 0)
            pipeline = None
            pipeline_result = None
            current_workspace_name = "test"
            current_pipeline_name = "测试流水线"
            pipeline_key = "test"
            active_step_key = "orig"
            samples = []
            sel_idx = -1
            targets = [dummy_target]
            sel_target = 0
            mode = "3d"
            error = ""
            _toast_msg = ""
            _toast_until = 0.0

        canvas = np.zeros((800, 1200, 3), dtype=np.uint8)
        state = DummyState()

        # 调用完整渲染
        state.active_dropdown = None
        state.scroll_off = 0
        state.viewport = None
        state.vis_img = None
        state._workspace_rect = None
        state._pipeline_rect = None
        buttons, sample_rows, result_rows, dd_items, slider_bars = AsparagusPoseStudioRenderer.render_scene(
            canvas, state
        )

        # 验证按钮中绝不包含 "导出G-code"
        button_labels = [b[1][1] for b in buttons if b[1][0] == "btn"]
        self.assertNotIn("导出G-code [E]", button_labels)
        self.assertIn("退出 [X]", button_labels)

        # 验证面板方法中已不再有 _draw_gcode_box
        self.assertFalse(hasattr(AsparagusPoseStudioRenderer, "_draw_gcode_box"))
        self.assertTrue(hasattr(AsparagusPoseStudioRenderer, "_draw_pose_detail_box"))


if __name__ == "__main__":
    unittest.main()
