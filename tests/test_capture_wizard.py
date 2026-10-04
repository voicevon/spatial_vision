#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：多视角采图向导 (纯预览 + 保存 瘦身版)
覆盖 渲染器工具栏命中表 / 画布合成冒烟 / 工具栏状态持久化 /
     相机开关状态机 / 保存快照
"""
import os
import sys
import json
import tempfile
import unittest

import numpy as np

# 添加工程根目录到 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tools.capture.capture_wizard as wizard_mod
from tools.capture.capture_wizard import CaptureWizard, APP_ID
from tools.capture.renderer import TOOLBAR_H


def make_wizard():
    """构造向导实例 (输出目录与 settings 文件均指向临时路径, 避免污染真实数据)。
    GUI_SETTINGS_FILE 重定向后保持生效, 使测试内的 _save_viewer_state 也写入临时文件。"""
    tmp_dir = tempfile.mkdtemp(prefix="wizard_test_")
    wizard_mod.GUI_SETTINGS_FILE = os.path.join(tmp_dir, "gui_settings.json")
    return CaptureWizard(output_dir=tmp_dir)


class TestCaptureWizard(unittest.TestCase):

    def test_toolbar_buttons_and_hit_test(self):
        """工具栏命中表: 常态按钮 id 存在且可命中; 包含工位硬件与分辨率一体化锁定卡片"""
        wiz = make_wizard()
        canvas = wiz.renderer.make_canvas()
        wiz.renderer.draw_toolbar(canvas)

        ids = [btn_id for btn_id, _, _ in wiz.renderer.buttons]
        for expected in ("TOGGLE_WS_DD", "TOGGLE_PURPOSE_DD", "LOCKED_HW_CLICK", "TOGGLE_CAMERA", "CAPTURE", "QUIT"):
            self.assertIn(expected, ids)

        # 命中检测: 第1组工作空间(x=8..203)、用途(x=209..334)、第2组硬件规格卡片(x=340..605)、第3组拍照 与第4组退出
        hit_ws = wiz.renderer.hit_test(20, TOOLBAR_H // 2)
        self.assertIsNotNone(hit_ws)
        self.assertEqual(hit_ws[0], "TOGGLE_WS_DD")

        hit_purpose = wiz.renderer.hit_test(250, TOOLBAR_H // 2)
        self.assertIsNotNone(hit_purpose)
        self.assertEqual(hit_purpose[0], "TOGGLE_PURPOSE_DD")

        hit_hw = wiz.renderer.hit_test(450, TOOLBAR_H // 2)
        self.assertIsNotNone(hit_hw)
        self.assertEqual(hit_hw[0], "LOCKED_HW_CLICK")

        hit_cap = wiz.renderer.hit_test(850, TOOLBAR_H // 2)
        self.assertIsNotNone(hit_cap)
        self.assertEqual(hit_cap[0], "CAPTURE")

        tw = canvas.shape[1]
        hit_quit = wiz.renderer.hit_test(tw - 40, TOOLBAR_H // 2)
        self.assertIsNotNone(hit_quit)
        self.assertEqual(hit_quit[0], "QUIT")

        # 展开用途下拉后: 出现 DD_PURPOSE_ 选项按钮, 点击动作可分发
        wiz.active_dropdown = "PURPOSE_DROPDOWN"
        wiz.renderer.draw_toolbar(canvas)
        pur_ids = [btn_id for btn_id, _, _ in wiz.renderer.buttons if btn_id.startswith("DD_PURPOSE_")]
        self.assertEqual(len(pur_ids), len(wiz.purpose_options))

        # 点击用途选项 -> 切换用途为 production
        wiz._handle_action(pur_ids[1], "production")
        self.assertEqual(wiz.purpose, "production")
        self.assertIsNone(wiz.active_dropdown)

    def test_canvas_compose_smoke(self):
        """画布合成冒烟: make_canvas / compose_canvas / draw_toast 不抛异常且尺寸正确"""
        wiz = make_wizard()
        canvas = wiz.renderer.make_canvas()
        cw, ch = wiz.win_mgr.canvas_w, wiz.win_mgr.canvas_h
        self.assertEqual(canvas.shape, (ch, cw, 3))

        frame = np.full((1080, 1920, 3), 80, dtype=np.uint8)  # 1080P 帧
        composed = wiz.renderer.compose_canvas(frame)
        self.assertEqual(composed.shape, (ch, cw, 3))

        wiz.set_toast("测试 Toast")
        wiz.renderer.draw_toolbar(composed)
        wiz.renderer.draw_toast(composed)  # 不抛异常即可

    def test_viewer_state_persistence(self):
        """工具栏状态持久化: _save_viewer_state -> _load_viewer_state 往返一致 (仅保留采集用途)"""
        wiz = make_wizard()  # make_wizard 已将 GUI_SETTINGS_FILE 重定向至临时路径
        wiz.purpose = "production"
        wiz._save_viewer_state()

        settings_file = wizard_mod.GUI_SETTINGS_FILE
        self.assertTrue(os.path.exists(settings_file))
        with open(settings_file, "r", encoding="utf-8") as f:
            root = json.load(f)
        self.assertEqual(root[APP_ID]["viewer_state"]["purpose"], "production")

        # 模拟重启: 恢复采集用途设置
        wiz.purpose = "intrinsics"
        wiz._load_viewer_state()
        self.assertEqual(wiz.purpose, "production")

    def test_camera_toggle_state_machine(self):
        """相机开关状态机: 未开启 get_frame 返回 None; toggle 翻转取流状态;
        开启失败时报 Toast 不静默, 开启成功后可正常关回"""
        wiz = make_wizard()
        self.assertFalse(wiz.pipeline_running)
        self.assertIsNone(wiz.get_frame(0))  # 未开启: 不取帧

        wiz._toggle_camera()  # 有物理相机则成功, 无则失败 Toast (与硬件环境无关)
        started = wiz.pipeline_running
        if started:
            self.assertIsNotNone(wiz.get_frame(0))
        else:
            self.assertNotEqual(wiz.status_toast, "")

        wiz._toggle_camera()  # 关闭
        self.assertFalse(wiz.pipeline_running)
        self.assertIsNone(wiz.get_frame(0))

    def test_workspace_hardware_locking(self):
        """工位锁定硬件规范: 相机类型与分辨率强绑定工位，点击锁定牌提示前往 Workspace Hub 配置"""
        wiz = make_wizard()

        # 点击硬件规格卡片触发锁定提示
        wiz._handle_action("LOCKED_HW_CLICK", None)
        self.assertIn("已锁定硬件", wiz.status_toast)
        self.assertIn("Workspace Hub", wiz.status_toast)

    def test_save_image_snapshot(self):
        """保存快照: 符合工位锁定规格成功保存；不符合规格防呆拒绝"""
        wiz = make_wizard()
        # 1. 规格匹配 (1080P: 1920x1080) -> 保存成功
        frame_valid = np.full((1080, 1920, 3), 60, dtype=np.uint8)
        path = wiz.save_image(frame_valid)

        self.assertEqual(wiz.image_count, 1)
        self.assertTrue(os.path.exists(path))
        self.assertTrue(os.path.basename(path).startswith("view_0001"))

        # 2. 规格失配 (720P: 1280x720) 试图写入 1080P 工位 -> 防呆拒绝保存
        frame_mismatch = np.full((720, 1280, 3), 60, dtype=np.uint8)
        rejected_path = wiz.save_image(frame_mismatch)
        self.assertEqual(rejected_path, "")
        self.assertEqual(wiz.image_count, 1)  # 计数不递增
        self.assertIn("严重失配", wiz.status_toast)

    def test_capture_action_handling(self):
        """测试点击拍照按钮分发逻辑：无流时 Toast 提示，有帧时成功保存"""
        wiz = make_wizard()
        # 1. 相机未开启时点击拍照
        wiz._handle_action("CAPTURE", None)
        self.assertIn("相机未开启", wiz.status_toast)

        # 2. 模拟相机运行并拥有最新帧
        wiz.pipeline_running = True
        wiz.last_raw_frame = np.full((1080, 1920, 3), 100, dtype=np.uint8)
        wiz._handle_action("CAPTURE", None)
        self.assertEqual(wiz.image_count, 1)



    def test_switch_purpose_three_datasets(self):
        """测试三大图集用途切换 (intrinsics / calibration / production)"""
        wiz = CaptureWizard(output_dir=None)
        # 切换到相机内参
        wiz.switch_purpose("intrinsics")
        self.assertEqual(wiz.purpose, "intrinsics")
        self.assertIn("intrinsics", wiz.output_dir.replace("\\", "/"))

        # 切换到生产采样
        wiz.switch_purpose("production")
        self.assertEqual(wiz.purpose, "production")
        self.assertIn("production", wiz.output_dir.replace("\\", "/"))

        # 切换回外参建图
        wiz.switch_purpose("calibration")
        self.assertEqual(wiz.purpose, "calibration")
        self.assertIn("calibration", wiz.output_dir.replace("\\", "/"))


if __name__ == "__main__":
    unittest.main()
