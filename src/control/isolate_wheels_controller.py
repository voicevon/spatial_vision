#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分离轮步进电机与节拍控制中枢 (IsolateWheelsController)
=====================================================
负责向各类应用工台（如 isolate_wheels_debug 调试视窗、isolate_wheels_production 生产分选看板）
提供统一、高内聚、线程安全的分离轮硬件控制与节拍调度接口：

1. 通信与生命周期: 基于 paho-mqtt 维护与 ESP32 分离轮控制器的长连接、心跳与异常感知;
2. 设备拓扑与多机管理: 自动监听 flux/loader/+/state 与 log，动态维护在线设备字典与保留状态;
3. 节拍与调度命令 (v1.3):
   - load: 8 托架物料数量与速度倍率 (speed: 0.1x~2.0x) 下发;
   - motor: 单电机精准方向与角度调试;
   - multi: 8 轴独立角度并发调试;
4. 安全互锁与参数校验: 仅当目标设备处于 IDLE 状态时受理生产指令，严格防止时序冲突;
5. 观测与诊断缓冲: 统一收集固件上行日志、done 完成应答计数与最近下发记录。
"""

import json
import os
import threading
import time
from collections import deque
from enum import Enum
from typing import Callable, Deque, Dict, List, Optional, Tuple

import paho.mqtt.client as mqtt

from src.utils.logger import get_logger

log = get_logger(__name__)

# ==================== 协议与网络配置常量 ====================
BROKER_HOST = "voicevon.vicp.io"
BROKER_PORT = 1883
BROKER_USER = "von"
BROKER_PASS = "von123456"
TOPIC_PREFIX = "flux/loader"
DEFAULT_DEVID = "F8EC"
KEEPALIVE_S = 15

# 预设速度倍率档位 (协议 v1.3)
LOAD_SPEED_OPTS = (0.1, 0.2, 0.5, 1.0, 1.5, 2.0)


class WheelState(str, Enum):
    """分离轮设备运行状态"""
    IDLE = "idle"
    RUNNING = "running"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class IsolateWheelsController:
    """分离轮核心控制器 (线程安全)"""

    def __init__(
        self,
        broker_host: str = BROKER_HOST,
        broker_port: int = BROKER_PORT,
        broker_user: str = BROKER_USER,
        broker_pass: str = BROKER_PASS,
        default_devid: str = DEFAULT_DEVID,
        max_log_lines: int = 300,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.broker_user = broker_user
        self.broker_pass = broker_pass
        self.selected_devid = default_devid

        # 线程安全锁
        self._lock = threading.Lock()
        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._connecting = False

        # 设备状态字典: devid -> state (idle / running / offline)
        self.devices: Dict[str, str] = {}

        # 运行统计与记录
        self.done_count: int = 0
        self.last_done_time: str = ""
        self.last_done_cmd: str = ""
        self.last_cmd_json: str = ""
        self.last_publish_msg: str = ""

        # 日志缓冲区
        self.log_lines: Deque[str] = deque(maxlen=max_log_lines)

        # 外部回调列表
        self._on_connect_cbs: List[Callable[[], None]] = []
        self._on_disconnect_cbs: List[Callable[[], None]] = []
        self._on_state_cbs: List[Callable[[str, str], None]] = []
        self._on_log_cbs: List[Callable[[str, str], None]] = []
        self._on_done_cbs: List[Callable[[str, str], None]] = []

    # ==================== 生命周期与连接管理 ====================
    @property
    def is_connected(self) -> bool:
        """MQTT Broker 连接状态"""
        with self._lock:
            return self._connected

    def connect(self) -> bool:
        """异步连接 MQTT Broker"""
        with self._lock:
            if self._connected or self._connecting:
                return True
            self._connecting = True

        try:
            client_id = f"flux_wheels_ctl_{os.getpid()}_{int(time.time() * 1000) % 10000}"
            client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=client_id,
                protocol=mqtt.MQTTv311,
            )
            client.username_pw_set(self.broker_user, self.broker_pass)
            client.reconnect_delay_set(min_delay=2, max_delay=10)
            client.on_connect = self._on_connect
            client.on_disconnect = self._on_disconnect
            client.on_message = self._on_message

            client.connect_async(self.broker_host, self.broker_port, keepalive=KEEPALIVE_S)
            client.loop_start()
            self._client = client
            log.info(f"[WheelsController] 开始连接 Broker: {self.broker_host}:{self.broker_port}")
            return True
        except Exception as e:
            with self._lock:
                self._connecting = False
            log.error(f"[WheelsController] 连接 Broker 异常: {e}")
            return False

    def disconnect(self):
        """断开连接并停止客户端"""
        with self._lock:
            client = self._client
            self._client = None
            self._connected = False
            self._connecting = False
        if client:
            try:
                client.loop_stop()
                client.disconnect()
            except Exception as e:
                log.warning(f"[WheelsController] 断开连接异常: {e}")

    # ==================== 状态查询与监听 ====================
    def get_device_state(self, devid: Optional[str] = None) -> str:
        """获取指定设备当前运行状态 (idle / running / offline / '')"""
        target = devid or self.selected_devid
        with self._lock:
            return self.devices.get(target, "")

    def is_device_idle(self, devid: Optional[str] = None) -> bool:
        """目标设备是否处于可受理新节拍命令的 IDLE 状态"""
        return self.get_device_state(devid) == WheelState.IDLE.value

    def add_on_state_listener(self, cb: Callable[[str, str], None]):
        """注册设备状态更新回调 cb(devid, state)"""
        self._on_state_cbs.append(cb)

    def add_on_log_listener(self, cb: Callable[[str, str], None]):
        """注册实时日志回调 cb(devid, log_text)"""
        self._on_log_cbs.append(cb)

    def add_on_done_listener(self, cb: Callable[[str, str], None]):
        """注册节拍完成回调 cb(devid, cmd_type)"""
        self._on_done_cbs.append(cb)

    # ==================== 核心命令发布 API (带安全互锁) ====================
    def send_load(
        self,
        counts: List[int],
        speed: float = 1.0,
        devid: Optional[str] = None,
        force: bool = False,
    ) -> Tuple[bool, str, str]:
        """
        下发生产节拍命令 (load)

        :param counts: 8 个托架物料数量 (1~8号轮，元素为 0~255 整数)
        :param speed: 节拍速度倍率 (如 0.1, 0.2, 0.5, 1.0, 1.5, 2.0，默认 1.0)
        :param devid: 目标设备 ID (缺省时使用当前选中的设备)
        :param force: 是否绕过 idle 互锁检查强制下发 (仅特殊调试场景使用)
        :return: (成功标志, 说明信息/错误原因, 实际下发的 JSON 载荷)
        """
        target_devid = devid or self.selected_devid
        if not self.is_connected:
            return False, "尚未连接 Broker，无法下发命令。", ""

        if not force and not self.is_device_idle(target_devid):
            cur_st = self.get_device_state(target_devid)
            return False, f"设备 {target_devid} 当前状态为 [{cur_st or '未知'}]，非 idle 状态，已被安全互锁拦截。", ""

        # 校验 counts
        if not isinstance(counts, (list, tuple)) or len(counts) != 8:
            return False, f"counts 长度必须为 8，当前为 {len(counts) if hasattr(counts, '__len__') else '无效'}", ""
        for i, c in enumerate(counts):
            if not isinstance(c, (int, float)) or c < 0 or c > 255:
                return False, f"托架 {i + 1} 数量非法 ({c})，须为 0~255 整数", ""

        # 校验与规整 speed
        try:
            spd = round(float(speed), 2)
            if spd <= 0.0 or spd > 5.0:
                spd = 1.0
        except Exception:
            spd = 1.0

        int_counts = [int(x) for x in counts]
        payload_dict = {"cmd": "load", "counts": int_counts, "speed": spd}
        payload = json.dumps(payload_dict, ensure_ascii=False)
        topic = f"{TOPIC_PREFIX}/{target_devid}/cmd"

        return self._publish(topic, payload, f"load ({spd:.1f}x)")

    def send_motor(
        self,
        motor: int,
        dir: int,
        angle: float,
        devid: Optional[str] = None,
        force: bool = False,
    ) -> Tuple[bool, str, str]:
        """
        下发单电机调试命令 (motor)

        :param motor: 电机编号 (1~8)
        :param dir: 方向 (1=正转, 0=反转)
        :param angle: 旋转角度 (0, 360]
        :param devid: 目标设备 ID
        :param force: 是否绕过 idle 互锁
        :return: (成功标志, 说明信息, JSON 载荷)
        """
        target_devid = devid or self.selected_devid
        if not self.is_connected:
            return False, "尚未连接 Broker，无法下发命令。", ""

        if not force and not self.is_device_idle(target_devid):
            cur_st = self.get_device_state(target_devid)
            return False, f"设备 {target_devid} 状态为 [{cur_st or '未知'}] 非 idle，已拦截。", ""

        if motor < 1 or motor > 8:
            return False, f"电机编号 {motor} 越界 (1~8)", ""
        if dir not in (0, 1):
            return False, f"方向代码 {dir} 非法 (0=反转, 1=正转)", ""
        ang = round(float(angle), 1)
        if not (0 < ang <= 360):
            return False, f"角度 {ang}° 越界 (0, 360]", ""

        ang_val = int(ang) if ang == int(ang) else ang
        payload_dict = {"cmd": "motor", "motor": int(motor), "dir": int(dir), "angle": ang_val}
        payload = json.dumps(payload_dict, ensure_ascii=False)
        topic = f"{TOPIC_PREFIX}/{target_devid}/cmd"

        return self._publish(topic, payload, f"motor #{motor} {ang}°")

    def send_multi(
        self,
        angles: List[float],
        devid: Optional[str] = None,
        force: bool = False,
    ) -> Tuple[bool, str, str]:
        """
        下发 8 轴并发调试命令 (multi)

        :param angles: 8 个电机的目标角度，0=不动作，正值=正转，负值=反转
        :param devid: 目标设备 ID
        :param force: 是否绕过 idle 互锁
        :return: (成功标志, 说明信息, JSON 载荷)
        """
        target_devid = devid or self.selected_devid
        if not self.is_connected:
            return False, "尚未连接 Broker，无法下发命令。", ""

        if not force and not self.is_device_idle(target_devid):
            cur_st = self.get_device_state(target_devid)
            return False, f"设备 {target_devid} 状态为 [{cur_st or '未知'}] 非 idle，已拦截。", ""

        if not isinstance(angles, (list, tuple)) or len(angles) != 8:
            return False, f"angles 长度必须为 8，当前为 {len(angles) if hasattr(angles, '__len__') else '无效'}", ""

        clean_angles = []
        for i, a in enumerate(angles):
            r = round(float(a), 1)
            if not (-360.0 <= r <= 360.0):
                return False, f"第 {i + 1} 号电机角度 {r}° 越界 [-360, 360]", ""
            clean_angles.append(int(r) if r == int(r) else r)

        payload_dict = {"cmd": "multi", "angles": clean_angles}
        payload = json.dumps(payload_dict, ensure_ascii=False)
        topic = f"{TOPIC_PREFIX}/{target_devid}/cmd"

        return self._publish(topic, payload, "multi 8轴并发")

    # ==================== 内部网络事件 ====================
    def _publish(self, topic: str, payload: str, desc: str) -> Tuple[bool, str, str]:
        with self._lock:
            client = self._client
        if not client:
            return False, "客户端未初始化", ""

        try:
            client.publish(topic, payload, qos=0)
            self.last_cmd_json = payload
            self.last_publish_msg = f"{time.strftime('%H:%M:%S')} 已下发 -> {desc}"
            log.info(f"[WheelsController] {self.last_publish_msg}: {payload}")
            return True, f"命令已成功下发: {desc}", payload
        except Exception as e:
            msg = f"MQTT 发布异常: {e}"
            log.error(f"[WheelsController] {msg}")
            return False, msg, payload

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            with self._lock:
                self._connected = True
                self._connecting = False
            sub_topic = f"{TOPIC_PREFIX}/+/+"
            client.subscribe(sub_topic, qos=0)
            log.info(f"[WheelsController] MQTT 连接就绪, 订阅主题: {sub_topic}")
            for cb in self._on_connect_cbs:
                try:
                    cb()
                except Exception as e:
                    log.warning(f"[WheelsController] on_connect 回调异常: {e}")
        else:
            with self._lock:
                self._connected = False
                self._connecting = False
            log.warning(f"[WheelsController] 连接 Broker 失败，返回码: {rc}")

    def _on_disconnect(self, client, userdata, flags_or_rc, rc_or_none=None, properties=None):
        with self._lock:
            self._connected = False
            self._connecting = False
        log.warning("[WheelsController] MQTT 连接断开")
        for cb in self._on_disconnect_cbs:
            try:
                cb()
            except Exception as e:
                log.warning(f"[WheelsController] on_disconnect 回调异常: {e}")

    def _on_message(self, client, userdata, msg):
        topic = msg.topic
        try:
            payload = msg.payload.decode("utf-8", errors="replace").strip()
        except Exception:
            return

        parts = topic.split("/")
        if len(parts) < 4 or parts[0] != "flux" or parts[1] != "loader":
            return
        devid, kind = parts[2], parts[3]

        parsed_state = ""
        with self._lock:
            if kind == "state":
                try:
                    data = json.loads(payload)
                    if isinstance(data, dict):
                        parsed_state = str(data.get("state", "")).strip()
                except Exception:
                    if payload in ("idle", "running", "offline", "fault", "homing", "pause"):
                        parsed_state = payload
                    else:
                        return

                if not parsed_state:
                    return

                self.devices[devid] = parsed_state
                if parsed_state == "offline":
                    self.log_lines.append(f"[{devid}] 设备已离线 (遗嘱消息)")
            elif kind == "done":
                if devid not in self.devices:
                    self.devices[devid] = ""
                self.done_count += 1
                self.last_done_time = time.strftime("%H:%M:%S")
                cmd_type = ""
                try:
                    data = json.loads(payload)
                    cmd_type = str(data.get("cmd", ""))
                except Exception:
                    cmd_type = payload
                self.last_done_cmd = cmd_type
            elif kind == "log":
                if devid not in self.devices:
                    self.devices[devid] = ""
                self.log_lines.append(f"[{devid}] {payload}")

        # 触发监听回调 (锁外安全执行)
        if kind == "state" and parsed_state:
            for cb in self._on_state_cbs:
                try:
                    cb(devid, parsed_state)
                except Exception as e:
                    log.warning(f"state 回调异常: {e}")
        elif kind == "done":
            for cb in self._on_done_cbs:
                try:
                    cb(devid, self.last_done_cmd)
                except Exception as e:
                    log.warning(f"done 回调异常: {e}")
        elif kind == "log":
            for cb in self._on_log_cbs:
                try:
                    cb(devid, payload)
                except Exception as e:
                    log.warning(f"log 回调异常: {e}")
