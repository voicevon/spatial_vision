#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目 1 极简闭环单元测试:
  - PlanarRectifier: 透视倾斜场景 4 标靶单应性拉正 + 相机碰歪自动恢复 + 失锁熔断
  - GradingEngine: 多级规则匹配与剔除兜底
  - SorterSerialLink: TRIG / RES / PING 协议编解码与仿真触发
  - SorterPipelineController: 歪斜图像 -> 拉正 -> 实测毫米尺寸 -> 分级 -> 串口回传完整闭环
"""

import unittest
from typing import Tuple

import cv2
import numpy as np

from src.control.grading_engine import GradeRule, GradingEngine
from src.devices.sorter_serial import (
    SorterSerialLink,
    format_pong_line,
    format_result_line,
    parse_mcu_line,
)
from src.vision.planar_rectifier import PlanarRectifier, PlanarRigConfig
from src.vision.single_asparagus_measurer import MeasureConfig, SingleAsparagusMeasurer
from tools.sorter_runtime import SorterPipelineController


def _make_synthetic_rig_and_camera_frame(
    *,
    asparagus_len_mm: float = 220.0,
    asparagus_diam_mm: float = 14.0,
    shift_px: Tuple[float, float] = (0.0, 0.0),
    tilt_strength: float = 1.0,
) -> tuple:
    """
    在标准鸟瞰物理平面上绘制 4 枚真实 AprilTag 16h5 与一根已知物理尺寸的绿芦笋，
    再施加显著透视倾斜投影，模拟现场歪斜安装的 2D 相机画面。
    """
    rig_cfg = PlanarRigConfig(
        tag_size_mm=30.0,
        fixed_tags_mm={
            0: (25.0, 25.0),
            1: (335.0, 25.0),
            2: (335.0, 155.0),
            3: (25.0, 155.0),
        },
        roi_mm=(45.0, 45.0, 315.0, 135.0),
        px_per_mm=4.0,
        max_reproj_err_px=1.5,
        shift_threshold_px=1.0,
        use_tag_corners=False,
    )

    # 先构建完整台面物理画布 (360mm x 180mm @ 4 px/mm = 1440 x 720)
    scale = 4.0
    board_w, board_h = int(360 * scale), int(180 * scale)
    board = np.full((board_h, board_w, 3), (35, 35, 35), dtype=np.uint8)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    tag_px = int(round(rig_cfg.tag_size_mm * scale))
    pad_px = int(round(6.0 * scale))  # 标靶白色静区

    for tid, (cx_mm, cy_mm) in rig_cfg.fixed_tags_mm.items():
        marker = cv2.aruco.generateImageMarker(aruco_dict, tid, tag_px)
        cx_px = int(round(cx_mm * scale))
        cy_px = int(round(cy_mm * scale))
        x0, y0 = cx_px - tag_px // 2, cy_px - tag_px // 2
        board[y0 - pad_px : y0 + tag_px + pad_px, x0 - pad_px : x0 + tag_px + pad_px] = (245, 245, 245)
        board[y0 : y0 + tag_px, x0 : x0 + tag_px] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

    # 在 ROI 内部绘制一根水平放置、已知物理长度与直径的纯绿芦笋
    if asparagus_len_mm > 0 and asparagus_diam_mm > 0:
        x_start = int(round(65.0 * scale))
        x_end = int(round((65.0 + asparagus_len_mm) * scale))
        y_mid = int(round(90.0 * scale))
        half_d = int(round((asparagus_diam_mm / 2.0) * scale))
        # BGR 鲜绿色 (HSV H≈60, S≈190, V≈170)
        cv2.rectangle(board, (x_start, y_mid - half_d), (x_end, y_mid + half_d), (45, 175, 55), -1)

    # 将台面透视投影到 1280x720 模拟相机画面 (制造明显近大远小 + 歪斜 + 可选碰歪位移)
    src_quad = np.array(
        [[0, 0], [board_w - 1, 0], [board_w - 1, board_h - 1], [0, board_h - 1]],
        dtype=np.float32,
    )
    dx, dy = shift_px
    t = float(tilt_strength)
    dst_quad = np.array(
        [
            [110.0 + 35.0 * t + dx, 80.0 + 25.0 * t + dy],
            [1160.0 - 20.0 * t + dx, 65.0 + dy],
            [1210.0 + dx, 650.0 - 15.0 * t + dy],
            [70.0 + dx, 630.0 + 20.0 * t + dy],
        ],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src_quad, dst_quad)
    cam_frame = cv2.warpPerspective(board, M, (1280, 720), borderValue=(20, 20, 20))
    return rig_cfg, cam_frame


class TestGradingEngine(unittest.TestCase):
    def setUp(self):
        self.engine = GradingEngine(
            rules=[
                GradeRule(grade=1, name="特级粗", min_len_mm=220.0, min_diam_mm=14.0),
                GradeRule(grade=2, name="一级中", min_len_mm=200.0, min_diam_mm=10.0),
                GradeRule(grade=3, name="无白根精品", min_len_mm=180.0, min_diam_mm=8.0, allow_white_root=False),
            ],
            default_reject_grade=0,
        )

    def test_rule_priority_and_reject(self):
        d1 = self.engine.evaluate(valid=True, effective_length_mm=235.0, diameter_mm=15.2)
        self.assertEqual(d1.grade, 1)
        self.assertTrue(d1.matched)

        d2 = self.engine.evaluate(valid=True, effective_length_mm=210.0, diameter_mm=11.0)
        self.assertEqual(d2.grade, 2)

        d3_reject = self.engine.evaluate(valid=True, effective_length_mm=190.0, diameter_mm=9.0, has_white_root=True)
        self.assertEqual(d3_reject.grade, 0)
        self.assertEqual(d3_reject.reason, "OUT_OF_SPEC")

        d_invalid = self.engine.evaluate(valid=False, effective_length_mm=250.0, diameter_mm=15.0)
        self.assertEqual(d_invalid.grade, 0)
        self.assertEqual(d_invalid.reason, "INVALID_MEASUREMENT")


class TestSorterSerialCodec(unittest.TestCase):
    def test_parse_and_format(self):
        self.assertEqual(parse_mcu_line("TRIG,1042\r\n"), ("TRIG", 1042))
        self.assertEqual(parse_mcu_line("PING\n"), ("PING", None))
        self.assertEqual(parse_mcu_line("TRIG,abc\n"), ("UNKNOWN", None))
        self.assertEqual(parse_mcu_line("   "), ("EMPTY", None))
        self.assertEqual(
            format_result_line(1042, 2, 215.44, 11.82, "OK"),
            "RES,1042,2,215.4,11.8,OK\n",
        )
        self.assertEqual(format_pong_line(True), "PONG,1\n")
        self.assertEqual(format_pong_line(False), "PONG,0\n")

    def test_simulate_trigger_and_ping(self):
        received = []
        link = SorterSerialLink(simulate=True, on_trigger=lambda seq: received.append(seq), is_ready_fn=lambda: True)
        link.start()
        link.inject_trigger(7)
        self.assertEqual(received, [7])
        link._dispatch_line("PING\n")
        self.assertEqual(link.last_tx_line, "PONG,1\n")
        link.stop()


class TestPlanarRectifierAndClosedLoop(unittest.TestCase):
    def test_tilted_camera_rectification_and_bump_recovery(self):
        rig_cfg, frame_initial = _make_synthetic_rig_and_camera_frame(
            asparagus_len_mm=220.0, asparagus_diam_mm=14.0, shift_px=(0.0, 0.0), tilt_strength=1.0
        )
        rectifier = PlanarRectifier(rig_cfg)
        st1 = rectifier.update(frame_initial)
        self.assertTrue(st1.locked, f"预期锁定成功，实际失败: {st1.reason}")
        self.assertEqual(st1.visible_tag_ids, [0, 1, 2, 3])
        self.assertLess(st1.reproj_err_px, 1.0)
        self.assertFalse(st1.shifted_recovered)

        # 验证拉正后单根测量物理尺寸精度 (抗透视畸变)
        warped = rectifier.warp_roi(frame_initial)
        self.assertIsNotNone(warped)
        measurer = SingleAsparagusMeasurer(MeasureConfig(fallback_mm_per_px=rectifier.mm_per_px))
        m = measurer.measure(warped, roi_rect=None)
        self.assertTrue(m.valid, m.reason)
        self.assertAlmostEqual(m.effective_length_mm, 220.0, delta=3.0)
        self.assertAlmostEqual(m.diameter_mm, 14.0, delta=1.5)

        # 模拟客户把相机碰歪 (平移 + 倾角变化) -> 自动检测位移并热更新恢复
        _, frame_bumped = _make_synthetic_rig_and_camera_frame(
            asparagus_len_mm=220.0, asparagus_diam_mm=14.0, shift_px=(18.0, -12.0), tilt_strength=1.8
        )
        st2 = rectifier.update(frame_bumped)
        self.assertTrue(st2.locked)
        self.assertTrue(st2.shifted_recovered)
        self.assertGreater(st2.max_shift_px, 5.0)

        warped2 = rectifier.warp_roi(frame_bumped)
        m2 = measurer.measure(warped2, roi_rect=None)
        self.assertTrue(m2.valid)
        self.assertAlmostEqual(m2.effective_length_mm, 220.0, delta=3.0)
        self.assertAlmostEqual(m2.diameter_mm, 14.0, delta=1.5)

    def test_end_to_end_controller_closed_loop_and_calib_lost_failsafe(self):
        rig_cfg, frame = _make_synthetic_rig_and_camera_frame(
            asparagus_len_mm=225.0, asparagus_diam_mm=14.5
        )
        rectifier = PlanarRectifier(rig_cfg)
        measurer = SingleAsparagusMeasurer(MeasureConfig(fallback_mm_per_px=rectifier.mm_per_px))
        grading = GradingEngine(
            rules=[
                GradeRule(grade=1, name="特级粗", min_len_mm=215.0, min_diam_mm=13.0),
                GradeRule(grade=2, name="一级中", min_len_mm=190.0, min_diam_mm=9.0),
            ],
            default_reject_grade=0,
        )
        link = SorterSerialLink(simulate=True)
        link.start()
        controller = SorterPipelineController(rectifier, measurer, grading, link)

        # 正常触发闭环
        controller.update_latest_frame(frame)
        link.inject_trigger(101)
        self.assertIsNotNone(controller.last_outcome)
        self.assertEqual(controller.last_outcome.seq, 101)
        self.assertEqual(controller.last_outcome.grade, 1)
        self.assertEqual(controller.last_outcome.status, "OK")
        self.assertTrue(link.last_tx_line.startswith("RES,101,1,"))

        # 标靶完全遮挡失锁 -> 熔断输出 grade=0, CALIB_LOST
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        rectifier.update(blank)
        self.assertFalse(rectifier.is_locked)
        controller.update_latest_frame(blank)
        link.inject_trigger(102)
        self.assertEqual(controller.last_outcome.grade, 0)
        self.assertEqual(controller.last_outcome.status, "CALIB_LOST")
        self.assertEqual(link.last_tx_line, "RES,102,0,0.0,0.0,CALIB_LOST\n")
        link.stop()


if __name__ == "__main__":
    unittest.main()
