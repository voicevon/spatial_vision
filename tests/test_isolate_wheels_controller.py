#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IsolateWheelsController 单元测试
=================================
验证分离轮中枢控制器的网络生命周期、命令打包、安全互锁与参数校验机制。
"""

import json
import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.control.isolate_wheels_controller import (
    IsolateWheelsController,
    WheelState,
    LOAD_SPEED_OPTS,
    TOPIC_PREFIX,
    DEFAULT_DEVID,
)


class _FakeMqttClient:
    def __init__(self):
        self.published = []
        self.subscribed = []

    def publish(self, topic: str, payload: str, qos: int = 0):
        self.published.append((topic, payload))

    def subscribe(self, topic: str, qos: int = 0):
        self.subscribed.append((topic, qos))

    def loop_start(self):
        pass

    def loop_stop(self):
        pass

    def disconnect(self):
        pass


class _FakeMsg:
    def __init__(self, topic: str, payload: bytes):
        self.topic = topic
        self.payload = payload


class TestIsolateWheelsController(unittest.TestCase):

    def setUp(self):
        self.controller = IsolateWheelsController()

    def tearDown(self):
        self.controller.disconnect()

    def test_initial_state(self):
        """测试初始状态"""
        self.assertFalse(self.controller.is_connected)
        self.assertEqual(self.controller.selected_devid, DEFAULT_DEVID)
        self.assertEqual(len(self.controller.devices), 0)
        self.assertEqual(self.controller.done_count, 0)

    def test_interlock_blocks_when_disconnected(self):
        """未连接时拦截命令"""
        ok, msg, _ = self.controller.send_load([0] * 8)
        self.assertFalse(ok)
        self.assertIn("尚未连接", msg)

        ok, msg, _ = self.controller.send_motor(1, 1, 90)
        self.assertFalse(ok)

        ok, msg, _ = self.controller.send_multi([0] * 8)
        self.assertFalse(ok)

    def test_interlock_blocks_when_not_idle(self):
        """非 idle 状态时安全互锁拦截"""
        self.controller._connected = True
        fake = _FakeMqttClient()
        self.controller._client = fake
        self.controller.devices[DEFAULT_DEVID] = "running"

        ok, msg, _ = self.controller.send_load([0] * 8)
        self.assertFalse(ok)
        self.assertIn("非 idle", msg)

        # force=True 允许强制绕过
        ok, _, payload = self.controller.send_load([0] * 8, force=True)
        self.assertTrue(ok)
        self.assertEqual(len(fake.published), 1)

    def test_send_load_validation_and_payload(self):
        """验证 send_load 的参数校验与 JSON 构造"""
        self.controller._connected = True
        fake = _FakeMqttClient()
        self.controller._client = fake
        self.controller.devices[DEFAULT_DEVID] = "idle"

        # 错误 counts 长度
        ok, msg, _ = self.controller.send_load([1, 2, 3])
        self.assertFalse(ok)
        self.assertIn("长度必须为 8", msg)

        # 错误 counts 数值
        ok, msg, _ = self.controller.send_load([0, 0, 0, -1, 0, 0, 0, 0])
        self.assertFalse(ok)
        self.assertIn("非法", msg)

        # 正常下发带速度倍率
        counts = [1, 0, 2, 0, 0, 0, 0, 3]
        ok, msg, payload = self.controller.send_load(counts, speed=0.5)
        self.assertTrue(ok)
        data = json.loads(payload)
        self.assertEqual(data["cmd"], "load")
        self.assertEqual(data["counts"], counts)
        self.assertEqual(data["speed"], 0.5)

    def test_send_motor_validation(self):
        """验证 send_motor 的参数校验与 JSON 构造"""
        self.controller._connected = True
        fake = _FakeMqttClient()
        self.controller._client = fake
        self.controller.devices[DEFAULT_DEVID] = "idle"

        # 电机编号越界
        ok, _, _ = self.controller.send_motor(9, 1, 90)
        self.assertFalse(ok)

        # 方向非法
        ok, _, _ = self.controller.send_motor(1, 2, 90)
        self.assertFalse(ok)

        # 角度非法
        ok, _, _ = self.controller.send_motor(1, 1, 0)
        self.assertFalse(ok)
        ok, _, _ = self.controller.send_motor(1, 1, 400)
        self.assertFalse(ok)

        # 成功下发
        ok, _, payload = self.controller.send_motor(3, 0, 45.0)
        self.assertTrue(ok)
        data = json.loads(payload)
        self.assertEqual(data, {"cmd": "motor", "motor": 3, "dir": 0, "angle": 45})

    def test_send_multi_validation(self):
        """验证 send_multi 的参数校验与 JSON 构造"""
        self.controller._connected = True
        fake = _FakeMqttClient()
        self.controller._client = fake
        self.controller.devices[DEFAULT_DEVID] = "idle"

        # 长度错误
        ok, _, _ = self.controller.send_multi([90] * 7)
        self.assertFalse(ok)

        # 越界角度
        ok, _, _ = self.controller.send_multi([400] + [0] * 7)
        self.assertFalse(ok)

        # 成功下发
        angles = [90, -45.5, 0, 0, 0, 22.5, 0, 360]
        ok, _, payload = self.controller.send_multi(angles)
        self.assertTrue(ok)
        data = json.loads(payload)
        self.assertEqual(data["cmd"], "multi")
        self.assertEqual(data["angles"], [90, -45.5, 0, 0, 0, 22.5, 0, 360])

    def test_message_dispatching(self):
        """验证下行消息处理 (state, done, log)"""
        states_received = []
        dones_received = []
        logs_received = []

        self.controller.add_on_state_listener(lambda d, s: states_received.append((d, s)))
        self.controller.add_on_done_listener(lambda d, c: dones_received.append((d, c)))
        self.controller.add_on_log_listener(lambda d, l: logs_received.append((d, l)))

        # 1. 模拟收到 state 消息
        msg1 = _FakeMsg(f"{TOPIC_PREFIX}/F8EC/state", b"idle")
        self.controller._on_message(None, None, msg1)
        self.assertEqual(self.controller.devices["F8EC"], "idle")
        self.assertEqual(states_received, [("F8EC", "idle")])

        # 2. 模拟收到 done 消息
        msg2 = _FakeMsg(f"{TOPIC_PREFIX}/F8EC/done", b'{"cmd":"load"}')
        self.controller._on_message(None, None, msg2)
        self.assertEqual(self.controller.done_count, 1)
        self.assertEqual(self.controller.last_done_cmd, "load")
        self.assertEqual(dones_received, [("F8EC", "load")])

        # 3. 模拟收到 log 消息
        msg3 = _FakeMsg(f"{TOPIC_PREFIX}/F8EC/log", b"Beat finished")
        self.controller._on_message(None, None, msg3)
        self.assertIn("[F8EC] Beat finished", self.controller.log_lines)
        self.assertEqual(logs_received, [("F8EC", "Beat finished")])


if __name__ == "__main__":
    unittest.main()
