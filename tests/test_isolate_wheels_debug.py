"""
Isolator WHEELS 调试应用单元测试
================================
无网络冒烟测试：MQTT 消息路由、节拍命令稳妥模式、v1.1 motor 单电机调试、
托架步进器交互与渲染画布
"""

import os
import sys
import json
import tempfile
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.isolate_wheels_debug.app import (
    IsolateWheelsDebuggerApp, LOGIC_W, LOGIC_H,
    DEV_CHIP_X0, DEV_CHIP_Y, DEV_CHIP_W, DEV_CHIP_H, DEV_CHIP_STEP,
    BTN_CLEAR_COUNTS, BTN_SEND_LOAD, BTN_QUIT,
    BTN_DIR_FWD, BTN_DIR_REV,
    TAB_SINGLE, TAB_MULTI, BTN_SEND_MOTOR, MULTI_ANGLE_OPTS, POPUP_ROW_H,
)


class _FakeMsg:
    """模拟 paho-mqtt 上行消息对象"""

    def __init__(self, topic: str, payload: bytes):
        self.topic = topic
        self.payload = payload


class _FakeClient:
    """捕获 publish 调用的假 MQTT 客户端"""

    def __init__(self):
        self.published = []

    def publish(self, topic, payload, qos=0):
        self.published.append((topic, payload))
        return True


class TestIsolateWheelsDebug(unittest.TestCase):

    def setUp(self):
        self.app = IsolateWheelsDebuggerApp(settings_file=os.path.join(
            tempfile.mkdtemp(prefix="test_wheels_"), "gui_settings.json"))

    # ---------- MQTT 消息路由 ----------
    def test_on_message_state_routing(self):
        """state 保留消息初始化设备状态"""
        self.app._on_message(None, None, _FakeMsg("flux/loader/F8EC/state", b'{"state":"idle"}'))
        self.assertEqual(self.app._device_state("F8EC"), "idle")

    def test_on_message_offline_will_log(self):
        """offline 遗嘱记日志且状态更新"""
        self.app.devices["F8EC"] = "running"
        self.app._on_message(None, None, _FakeMsg("flux/loader/F8EC/state", b'{"state":"offline"}'))
        self.assertEqual(self.app._device_state("F8EC"), "offline")
        self.assertTrue(any("遗嘱" in line for line in self.app.log_lines))

    def test_on_message_done_and_log(self):
        """done 计数与 log 行追加"""
        self.app._on_message(None, None, _FakeMsg("flux/loader/F8EC/done", b'{"event":"done"}'))
        self.app._on_message(None, None, _FakeMsg("flux/loader/F8EC/log", "[INFO] 节拍运动步数".encode("utf-8")))
        self.assertEqual(self.app.done_count, 1)
        self.assertTrue(any("[F8EC]" in line for line in self.app.log_lines))

    def test_on_message_done_carries_cmd(self):
        """v1.1: done 回带 cmd 类型并记录"""
        self.app._on_message(None, None, _FakeMsg("flux/loader/F8EC/done", b'{"event":"done","cmd":"motor"}'))
        self.app._on_message(None, None, _FakeMsg("flux/loader/F8EC/done", b'{"event":"done","cmd":"load"}'))
        self.assertEqual(self.app.done_count, 2)
        self.assertEqual(self.app.last_done_cmd, "load")

    def test_on_message_ignores_foreign_and_malformed(self):
        """非本系统主题与非法载荷安全忽略"""
        self.app._on_message(None, None, _FakeMsg("flux/other/F8EC/state", b'{"state":"idle"}'))
        self.app._on_message(None, None, _FakeMsg("flux/loader/F8EC/state", b"not json"))
        self.assertEqual(self.app.devices, {})
        self.assertEqual(self.app.done_count, 0)

    # ---------- 节拍命令下发 ----------
    def test_send_load_requires_connection(self):
        """未连接 Broker 时拒绝下发 (稳妥模式)"""
        self.assertFalse(self.app.send_load())

    def test_send_load_requires_idle(self):
        """设备非 idle 时拒绝下发"""
        self.app._connected = True
        self.app.devices["F8EC"] = "running"
        self.assertFalse(self.app.send_load())

    def test_send_load_publishes_json(self):
        """连接且 idle 时下发 JSON 载荷到 cmd 主题"""
        self.app._connected = True
        self.app.devices["F8EC"] = "idle"
        self.app.counts = [1, 0, 2, 0, 0, 0, 0, 3]
        fake = _FakeClient()
        self.app._client = fake
        self.assertTrue(self.app.send_load())
        self.assertEqual(len(fake.published), 1)
        topic, payload = fake.published[0]
        self.assertEqual(topic, "flux/loader/F8EC/cmd")
        self.assertEqual(json.loads(payload), {"cmd": "load", "counts": [1, 0, 2, 0, 0, 0, 0, 3]})
        self.assertEqual(self.app.last_cmd_json, payload)

    # ---------- v1.1 motor 单电机调试 ----------
    def test_send_motor_requires_connection_and_idle(self):
        """未连接或非 idle 时拒绝下发 motor 命令"""
        self.assertFalse(self.app.send_motor())
        self.app._connected = True
        self.app.devices["F8EC"] = "running"
        self.assertFalse(self.app.send_motor())

    def test_send_motor_publishes_json(self):
        """连接且 idle 时下发 motor JSON 载荷"""
        self.app._connected = True
        self.app.devices["F8EC"] = "idle"
        self.app.motor_sel = 5
        self.app.motor_dir = 0
        self.app.motor_angle = 22.5
        fake = _FakeClient()
        self.app._client = fake
        self.assertTrue(self.app.send_motor())
        topic, payload = fake.published[0]
        self.assertEqual(topic, "flux/loader/F8EC/cmd")
        self.assertEqual(json.loads(payload), {"cmd": "motor", "motor": 5, "dir": 0, "angle": 22.5})
        self.assertEqual(self.app.last_cmd_json, payload)

    def test_motor_panel_clicks(self):
        """通道 Header/方向/角度步进与下拉交互"""
        # 点击 5 号电机 (idx=4, col=4) Header
        col = 4
        hdr = self.app._col_header_rect(col)
        self.app.on_click(hdr[0] + hdr[2] // 2, hdr[1] + hdr[3] // 2)
        self.assertEqual(self.app.motor_sel, 5)
        # 方向切换
        self.app.motor_mode = "single"
        self.app.on_click(BTN_DIR_REV[0] + 2, BTN_DIR_REV[1] + 2)
        self.assertEqual(self.app.motor_dir, 0)
        self.app.on_click(BTN_DIR_FWD[0] + 2, BTN_DIR_FWD[1] + 2)
        self.assertEqual(self.app.motor_dir, 1)
        # 角度步进微调
        aminus = self.app._col_angle_minus(col)
        aplus = self.app._col_angle_plus(col)
        self.app.motor_angle = 90.0
        self.app.on_click(aminus[0] + 2, aminus[1] + 2)
        self.assertEqual(self.app.motor_angle, 67.5)
        self.app.motor_angle = 360.0
        self.app.on_click(aplus[0] + 2, aplus[1] + 2)
        self.assertEqual(self.app.motor_angle, 360.0)

    # ---------- v1.2 multi 多电机调试 ----------
    def test_send_multi_requires_connection_and_idle(self):
        """未连接或非 idle 时拒绝下发 multi 命令"""
        self.assertFalse(self.app.send_multi())
        self.app._connected = True
        self.app.devices["F8EC"] = "running"
        self.assertFalse(self.app.send_multi())

    def test_send_multi_publishes_json(self):
        """连接且 idle 时下发 multi JSON 载荷 (0=不动, 负值=反转)"""
        self.app._connected = True
        self.app.devices["F8EC"] = "idle"
        self.app.multi_angles = [90.0, -45.0, 0.0, 0.0, 0.0, 22.5, 0.0, 360.0]
        fake = _FakeClient()
        self.app._client = fake
        self.assertTrue(self.app.send_multi())
        topic, payload = fake.published[0]
        self.assertEqual(topic, "flux/loader/F8EC/cmd")
        self.assertEqual(json.loads(payload),
                         {"cmd": "multi", "angles": [90, -45, 0, 0, 0, 22.5, 0, 360]})

    def test_multi_mode_tabs(self):
        """电机面板模式页签切换"""
        self.app.on_click(TAB_MULTI[0] + 2, TAB_MULTI[1] + 2)
        self.assertEqual(self.app.motor_mode, "multi")
        self.app.on_click(TAB_SINGLE[0] + 2, TAB_SINGLE[1] + 2)
        self.assertEqual(self.app.motor_mode, "single")

    def test_multi_dropdown_interaction(self):
        """多电机下拉框: 展开/选择角度/点击外部收起"""
        self.app.motor_mode = "multi"
        col = 2  # 3 号电机 (idx=2)
        box = self.app._col_angle_box(col)
        self.app.on_click(box[0] + 2, box[1] + 2)
        self.assertEqual(self.app.popup_col, col)
        # 选择 90° (选项第 4 行)
        px, py, pw, ph = self.app._popup_rect()
        self.app.on_click(px + 2, py + 4 + 3 * POPUP_ROW_H + 2)
        self.assertEqual(self.app.multi_angles[2], 90.0)
        self.assertEqual(self.app.popup_col, -1)
        # 再展开后点击弹层外仅收起
        self.app.on_click(box[0] + 2, box[1] + 2)
        self.app.on_click(20, 20)
        self.assertEqual(self.app.popup_col, -1)
        self.assertEqual(self.app.multi_angles[2], 90.0)
        # 发送按钮点击 (未连接, 无副作用)
        self.app.on_click(BTN_SEND_MOTOR[0] + 5, BTN_SEND_MOTOR[1] + 5)
        self.assertEqual(self.app.last_cmd_json, "")

    def test_multi_options_in_range(self):
        """下拉角度选项均在协议范围 [-360, 360] 内"""
        for opt in MULTI_ANGLE_OPTS:
            self.assertGreaterEqual(opt, -360.0)
            self.assertLessEqual(opt, 360.0)

    # ---------- 托架步进器交互 ----------
    def test_stepper_click_and_clamp(self):
        """[+]/[-] 点击改变托架数量且 0~9 边界钳制"""
        col = 2  # 3 号电机 (idx=2)
        minus = self.app._col_count_minus(col)
        plus = self.app._col_count_plus(col)
        self.app.on_click(plus[0] + plus[2] // 2, plus[1] + plus[3] // 2)
        self.assertEqual(self.app.counts[2], 1)
        self.app.on_click(minus[0] + minus[2] // 2, minus[1] + minus[3] // 2)
        self.assertEqual(self.app.counts[2], 0)
        self.app.on_click(minus[0] + minus[2] // 2, minus[1] + minus[3] // 2)
        self.assertEqual(self.app.counts[2], 0)

    def test_clear_button_zeroes_all(self):
        """全部清零按钮重置 8 托架"""
        self.app.counts = [3] * 8
        self.app.on_click(BTN_CLEAR_COUNTS[0] + BTN_CLEAR_COUNTS[2] // 2, BTN_CLEAR_COUNTS[1] + BTN_CLEAR_COUNTS[3] // 2)
        self.assertEqual(self.app.counts, [0] * 8)

    def test_send_button_blocked_when_disconnected(self):
        """点击发送按钮在未连接时无副作用"""
        self.app.on_click(BTN_SEND_LOAD[0] + BTN_SEND_LOAD[2] // 2, BTN_SEND_LOAD[1] + BTN_SEND_LOAD[3] // 2)
        self.assertEqual(self.app.last_cmd_json, "")

    # ---------- 设备芯片选择 ----------
    def test_device_chip_selection(self):
        """点击设备芯片切换目标设备"""
        self.app.devices = {"AAA1": "idle", "BBB2": "running"}
        cx = DEV_CHIP_X0 + DEV_CHIP_STEP + DEV_CHIP_W // 2   # 第 2 枚芯片中心
        cy = DEV_CHIP_Y + DEV_CHIP_H // 2
        self.app.on_click(cx, cy)
        self.assertEqual(self.app.selected_devid, "BBB2")

    def test_quit_button(self):
        """顶栏退出按钮终止主循环"""
        self.app.on_click(BTN_QUIT[0] + BTN_QUIT[2] // 2, BTN_QUIT[1] + BTN_QUIT[3] // 2)
        self.assertFalse(self.app._running)

    # ---------- 渲染冒烟 ----------
    def test_render_canvas_shape(self):
        """渲染输出 1280x760 三通道逻辑画布"""
        canvas = self.app.render()
        self.assertEqual(canvas.shape, (LOGIC_H, LOGIC_W, 3))


if __name__ == "__main__":
    unittest.main()
