# -*- coding: utf-8 -*-
"""
NetCamera 调试视窗 (android_as_camera 手机网络摄像机 MQTT + RTSP 调试 GUI)
==========================================================================
协议依据: android_as_camera/需求文档.md + 技术设计文档.md
  - 连接: MQTT voicevon.vicp.io:1883 (von), keepalive 15s
  - 订阅: camera/+/state | camera/+/event (通配符设备发现, state 周期上报)
  - 下发: camera/{devid}/cmd  {"action": "...", ...}
        start_stream / stop_stream / switch_camera(camera=front|back) /
        zoom(ratio) / capture / start_record / stop_record / motion(enabled,sensitivity)
  - 视频: rtsp://{state.ip}:8554/live  (TCP 交错 RTP, OpenCV FFMPEG 拉流)
纯 cv2 矢量 GUI (全鼠标化), ESC/红叉退出。
"""

import os
import sys
import json
import time
import threading
from collections import deque
from typing import Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# RTSP 仅 TCP 交错传输 (技术设计 D2), 须在建流前声明, OpenCV FFMPEG 才会走 TCP
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import paho.mqtt.client as mqtt  # noqa: E402

from src.utils.logger import get_logger  # noqa: E402
from src.ui.base_cv_app import BaseCvApp  # noqa: E402
from src.ui.text_rendering import draw_text  # noqa: E402

log = get_logger(__name__)

# ==================== 协议常量 (android_as_camera 文档) ====================
BROKER_HOST = "voicevon.vicp.io"
BROKER_PORT = 1883
BROKER_USER = "von"
BROKER_PASS = "von123456"
TOPIC_PREFIX = "camera"
RTSP_PORT = 8554
RTSP_PATH = "/live"
KEEPALIVE_S = 15

# ==================== 逻辑画布与几何常量 (渲染与命中测试单源) ====================
LOGIC_W, LOGIC_H = 1100, 720

MQTT_PANEL = (728, 12, 300, 46)
BTN_CONNECT = (852, 20, 78, 30)
BTN_DISCONNECT = (936, 20, 86, 30)
BTN_QUIT = (1032, 18, 48, 34)

DEV_CHIP_X0, DEV_CHIP_Y, DEV_CHIP_W, DEV_CHIP_H, DEV_CHIP_STEP = 110, 66, 130, 30, 136
DEV_CHIP_MAX = 8

STATE_CARD = (20, 106, 340, 116)
CTRL_CARD = (20, 226, 340, 318)
PROTO_CARD = (20, 572, 340, 110)

CTRL_BTN_W, CTRL_BTN_H = 152, 34
CTRL_COL1_X, CTRL_COL2_X = 32, 196
CTRL_ROW_Y0, CTRL_ROW_STEP = 262, 44
CTRL_FULL_W = CTRL_COL2_X + CTRL_BTN_W - CTRL_COL1_X
# 行布局: [推流开关(整行,状态化)] [前摄|后摄] [变焦-|变焦+] [拍照|录像] [侦测开关(整行)] [灵敏度-|灵敏度+]
BTN_STREAM = (CTRL_COL1_X, CTRL_ROW_Y0, CTRL_FULL_W, CTRL_BTN_H)
BTN_CAM_FRONT = (CTRL_COL1_X, CTRL_ROW_Y0 + CTRL_ROW_STEP, CTRL_BTN_W, CTRL_BTN_H)
BTN_CAM_BACK = (CTRL_COL2_X, CTRL_ROW_Y0 + CTRL_ROW_STEP, CTRL_BTN_W, CTRL_BTN_H)
BTN_ZOOM_DEC = (CTRL_COL1_X, CTRL_ROW_Y0 + 2 * CTRL_ROW_STEP, CTRL_BTN_W, CTRL_BTN_H)
BTN_ZOOM_INC = (CTRL_COL2_X, CTRL_ROW_Y0 + 2 * CTRL_ROW_STEP, CTRL_BTN_W, CTRL_BTN_H)
BTN_CAPTURE = (CTRL_COL1_X, CTRL_ROW_Y0 + 3 * CTRL_ROW_STEP, CTRL_BTN_W, CTRL_BTN_H)
BTN_RECORD = (CTRL_COL2_X, CTRL_ROW_Y0 + 3 * CTRL_ROW_STEP, CTRL_BTN_W, CTRL_BTN_H)
BTN_MOTION = (CTRL_COL1_X, CTRL_ROW_Y0 + 4 * CTRL_ROW_STEP, CTRL_FULL_W, CTRL_BTN_H)
BTN_SENS_DEC = (CTRL_COL1_X, CTRL_ROW_Y0 + 5 * CTRL_ROW_STEP, CTRL_BTN_W, CTRL_BTN_H)
BTN_SENS_INC = (CTRL_COL2_X, CTRL_ROW_Y0 + 5 * CTRL_ROW_STEP, CTRL_BTN_W, CTRL_BTN_H)

PREVIEW_CARD = (380, 106, 700, 400)
PREVIEW_VIDEO = (392, 142, 676, 352)
BTN_PV_START = (884, 112, 92, 26)
BTN_PV_STOP = (982, 112, 92, 26)

LOG_CARD = (380, 514, 700, 168)
LOG_HEADER_H = 30
LOG_LINE_H = 22

COLOR_BG = (18, 20, 24)
COLOR_PANEL = (26, 30, 38)
COLOR_BORDER = (52, 60, 74)
COLOR_ACCENT = (0, 200, 240)
COLOR_GREEN = (0, 220, 140)
COLOR_WARN = (0, 170, 255)
COLOR_RED = (80, 80, 255)
COLOR_TEXT = (230, 236, 244)
COLOR_SUB = (160, 172, 188)
COLOR_MUTED = (110, 120, 136)

ZOOM_MIN, ZOOM_MAX, ZOOM_STEP = 1.0, 10.0, 0.5
SENS_MIN, SENS_MAX, SENS_STEP = 0, 100, 10
# 判活: state 为周期上报, 超过此时长无上报即判定离线
# (broker 中 retained 的旧 state 会让离开的设备"永远在线", 只能靠客户端超时判活)
OFFLINE_TIMEOUT_S = 90


class NetCameraDebugApp(BaseCvApp):
    """NetCamera 调试主应用: 摄像机发现/状态监视/远程命令/RTSP 预览/事件日志 (基于 BaseCvApp)"""

    def __init__(self, settings_file: Optional[str] = None):
        super().__init__(
            app_id="net_camera_debug",
            base_w=LOGIC_W,
            base_h=LOGIC_H,
            window_name="flux_vision_3d | net_camera",
            window_title="flux_vision_3d | NetCamera 调试",
            settings_file=settings_file,
            enable_keyboard_zoom=False,
        )

        # MQTT 运行态 (回调来自网络线程, 共享读写加锁)
        self._lock = threading.Lock()
        self._client: Optional[mqtt.Client] = None
        self._connected = False
        self._connecting = False
        self.devices: dict = {}                 # devid -> state dict (streaming/camera/resolution/ip/battery)
        self.device_last_seen: dict = {}        # devid -> 最后一次 state 上报时间 (超时判活)
        self.selected_devid: str = ""
        self.log_lines: deque = deque(maxlen=300)

        # 远程控制本地态
        self.zoom_ratio = 1.0
        self.sensitivity = 50
        self.recording = False
        self.motion_enabled = False

        # RTSP 预览线程态
        self._frame: Optional[np.ndarray] = None
        self._preview_thread: Optional[threading.Thread] = None
        self._preview_stop = threading.Event()
        self._preview_running = False
        self._preview_url = ""
        self._preview_error = ""

    # ==================== 生命周期钩子 ====================
    def setup(self):
        """主循环启动时自动连接 Broker"""
        self.connect_broker()

    def cleanup(self):
        """退出前清理网络和预览流"""
        self.disconnect_broker()

    # ==================== MQTT 层 ====================
    def connect_broker(self):
        """异步连接 Broker 并订阅通配主题 (loop_start 内部线程自动重连)"""
        if self._connected or self._connecting:
            return
        self._connecting = True
        try:
            client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=f"flux_vision_netcam_{os.getpid()}",
                protocol=mqtt.MQTTv311,
            )
            client.username_pw_set(BROKER_USER, BROKER_PASS)
            client.reconnect_delay_set(min_delay=2, max_delay=10)
            client.on_connect = self._on_connect
            client.on_disconnect = self._on_disconnect
            client.on_message = self._on_message
            client.connect_async(BROKER_HOST, BROKER_PORT, keepalive=KEEPALIVE_S)
            client.loop_start()
            self._client = client
            self.set_toast(f"正在连接 Broker {BROKER_HOST}:{BROKER_PORT} ...")
            log.info(f"[NETCAM] 连接 Broker: {BROKER_HOST}:{BROKER_PORT}")
        except Exception as e:
            self._connecting = False
            self.set_toast(f"MQTT 连接失败: {e}")
            log.warning(f"[NETCAM] MQTT 连接失败: {e}")

    def disconnect_broker(self):
        """主动断开并停掉网络线程"""
        self.stop_preview()
        if self._client is not None:
            try:
                self._client.disconnect()
                self._client.loop_stop()
            except Exception:
                pass
        self._client = None
        self._connected = False
        self._connecting = False

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        ok = (not reason_code.is_failure) if hasattr(reason_code, "is_failure") else (int(reason_code) == 0)
        self._connected = ok
        self._connecting = False
        if ok:
            client.subscribe([
                (f"{TOPIC_PREFIX}/+/state", 0),
                (f"{TOPIC_PREFIX}/+/event", 0),
            ])
            self.set_toast("Broker 已连接, 订阅 camera/+/state|event (等待摄像机上报...)")
            log.info("[NETCAM] Broker 已连接并完成订阅")
        else:
            self.set_toast(f"Broker 连接被拒: {reason_code}")

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        self._connected = False
        self.set_toast(f"Broker 连接断开: {reason_code}")

    def _on_message(self, client, userdata, msg):
        """路由 camera/{devid}/{state|event} 上行消息 (网络线程)"""
        parts = msg.topic.split("/")
        if len(parts) != 3 or parts[0] != TOPIC_PREFIX:
            return
        devid, kind = parts[1], parts[2]
        payload = msg.payload.decode("utf-8", errors="replace")

        if kind == "state":
            try:
                st = json.loads(payload)
            except Exception:
                st = {"raw": payload}
            with self._lock:
                is_new = devid not in self.devices
                self.devices[devid] = st
                self.device_last_seen[devid] = time.time()
                if not self.selected_devid:
                    self.selected_devid = devid
            if is_new:
                self._log_line(f"[SYS] 发现摄像机 {devid} (ip={st.get('ip', '?')})")
        elif kind == "event":
            self._log_line(f"[{devid}] {payload}")
            try:
                ev = json.loads(payload)
                if ev.get("type") == "motion":
                    self.set_toast(f"!! 移动侦测告警: {devid} !!")
            except Exception:
                pass

    # ==================== 命令下发 ====================
    def _send_cmd(self, action: str, **params) -> bool:
        """向选中摄像机下发命令 JSON {"action": action, ...}"""
        if not self._connected:
            self.set_toast("尚未连接 Broker, 无法下发命令。")
            return False
        if not self.selected_devid:
            self.set_toast("尚未发现任何摄像机 (等待 camera/+/state 上报)。")
            return False
        offline = not self._device_online(self.selected_devid)
        if offline:
            stale = self._device_stale_s(self.selected_devid)
            self._log_line(f"[SYS] 警告: {self.selected_devid} 已离线 {stale}s (retained 残留), 命令可能不被受理")
        payload = json.dumps({"action": action, **params}, ensure_ascii=False)
        topic = f"{TOPIC_PREFIX}/{self.selected_devid}/cmd"
        try:
            self._client.publish(topic, payload, qos=0)
        except Exception as e:
            self.set_toast(f"发送失败: {e}")
            return False
        self._log_line(f"[TX] {topic} {payload}")
        self.set_toast(f"已下发 -> {self.selected_devid}: {payload}" + ("  [设备离线,可能无效]" if offline else ""))
        log.info(f"[NETCAM] TX {topic} {payload}")
        return True

    def _log_line(self, line: str):
        with self._lock:
            self.log_lines.append(f"{time.strftime('%H:%M:%S')} {line}")

    # ==================== RTSP 预览 ====================
    def start_preview(self):
        """按选中摄像机 state 上报的 IP 拉取 RTSP 实时画面"""
        if self._preview_running:
            return
        with self._lock:
            st = self.devices.get(self.selected_devid) or {}
        ip = st.get("ip", "")
        if not ip:
            self.set_toast("未知摄像机 IP (等待 state 上报), 无法拉流。")
            return
        if not self._device_online(self.selected_devid):
            self.set_toast(f"设备已离线, IP {ip} 为过期数据, 拉流大概率失败。")
            return
        url = f"rtsp://{ip}:{RTSP_PORT}{RTSP_PATH}"
        self._preview_stop.clear()
        self._preview_url = url
        self._preview_error = ""
        self._preview_running = True
        self._preview_thread = threading.Thread(target=self._preview_loop, args=(url,), daemon=True)
        self._preview_thread.start()
        self.set_toast(f"正在拉流 {url} ...")
        log.info(f"[NETCAM] 拉流开始: {url}")

    def stop_preview(self):
        self._preview_stop.set()
        if self._preview_thread is not None:
            self._preview_thread.join(timeout=1.5)
            self._preview_thread = None
        self._preview_running = False
        with self._lock:
            self._frame = None

    def _preview_loop(self, url: str):
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            with self._lock:
                self._preview_error = "无法打开 RTSP 流 (未推流/网络不通/与本机不同网段)"
            self._preview_running = False
            self.set_toast(self._preview_error)
            log.warning(f"[NETCAM] {self._preview_error}: {url}")
            return
        while not self._preview_stop.is_set():
            ok, frame = cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
                    self._preview_error = ""
            else:
                with self._lock:
                    self._preview_error = "流中断, 重试中..."
                time.sleep(0.3)
        cap.release()
        self._preview_running = False
        log.info("[NETCAM] 拉流停止")


    def _device_streaming(self) -> bool:
        """选中摄像机当前是否推流 (依 state 上报, 未知视为否)"""
        with self._lock:
            st = self.devices.get(self.selected_devid, {})
        return bool(st.get("streaming", False))

    def _device_online(self, devid: str) -> bool:
        """设备是否在线: OFFLINE_TIMEOUT_S 内有 state 上报才算在线
        (retained 旧消息会让已离开设备残留在列表里, 必须超时判活)"""
        with self._lock:
            last = self.device_last_seen.get(devid)
        return last is not None and (time.time() - last) < OFFLINE_TIMEOUT_S

    def _device_stale_s(self, devid: str) -> int:
        """距最后一次 state 上报的秒数 (未知返回 -1)"""
        with self._lock:
            last = self.device_last_seen.get(devid)
        return -1 if last is None else int(time.time() - last)

    def _sorted_dev_ids(self) -> list:
        """设备列表: 在线优先, 同组内按 id 排序"""
        with self._lock:
            ids = list(self.devices.keys())
        return sorted(ids, key=lambda d: (not self._device_online(d), d))


    def on_click(self, x: int, y: int):
        """逻辑坐标左键点击分发 (覆盖基类 on_click)"""
        # 1. 顶栏按钮
        if self.pt_in(x, y, BTN_CONNECT):
            self.connect_broker()
            return
        if self.pt_in(x, y, BTN_DISCONNECT):
            self.disconnect_broker()
            self.set_toast("已断开 Broker 连接。")
            return
        if self.pt_in(x, y, BTN_QUIT):
            self._running = False
            return

        # 2. 设备芯片选择
        if DEV_CHIP_Y <= y <= DEV_CHIP_Y + DEV_CHIP_H:
            dev_ids = self._sorted_dev_ids()
            for i in range(min(len(dev_ids), DEV_CHIP_MAX)):
                if self.pt_in(x, y, self._device_chip_rect(i)):
                    self.selected_devid = dev_ids[i]
                    with self._lock:
                        self._frame = None  # 切换目标后清掉旧画面, 避免误判
                    return

        # 3. 远程控制按钮
        if self.pt_in(x, y, BTN_STREAM):
            # 状态化开关: 依设备 state.streaming 决定动作 (未知状态视为未推流)
            self._send_cmd("stop_stream" if self._device_streaming() else "start_stream")
            return
        if self.pt_in(x, y, BTN_CAM_FRONT):
            self._send_cmd("switch_camera", camera="front")
            return
        if self.pt_in(x, y, BTN_CAM_BACK):
            self._send_cmd("switch_camera", camera="back")
            return
        if self.pt_in(x, y, BTN_ZOOM_DEC):
            self.zoom_ratio = max(ZOOM_MIN, round(self.zoom_ratio - ZOOM_STEP, 1))
            self._send_cmd("zoom", ratio=self.zoom_ratio)
            return
        if self.pt_in(x, y, BTN_ZOOM_INC):
            self.zoom_ratio = min(ZOOM_MAX, round(self.zoom_ratio + ZOOM_STEP, 1))
            self._send_cmd("zoom", ratio=self.zoom_ratio)
            return
        if self.pt_in(x, y, BTN_CAPTURE):
            self._send_cmd("capture")
            return
        if self.pt_in(x, y, BTN_RECORD):
            self.recording = not self.recording
            self._send_cmd("stop_record" if not self.recording else "start_record")
            return
        if self.pt_in(x, y, BTN_MOTION):
            self.motion_enabled = not self.motion_enabled
            self._send_cmd("motion", enabled=self.motion_enabled, sensitivity=self.sensitivity)
            return
        if self.pt_in(x, y, BTN_SENS_DEC):
            self.sensitivity = max(SENS_MIN, self.sensitivity - SENS_STEP)
            self.set_toast(f"移动侦测灵敏度: {self.sensitivity}")
            return
        if self.pt_in(x, y, BTN_SENS_INC):
            self.sensitivity = min(SENS_MAX, self.sensitivity + SENS_STEP)
            self.set_toast(f"移动侦测灵敏度: {self.sensitivity}")
            return

        # 4. RTSP 预览按钮
        if self.pt_in(x, y, BTN_PV_START):
            self.start_preview()
            return
        if self.pt_in(x, y, BTN_PV_STOP):
            self.stop_preview()
            self.set_toast("已停止预览。")
            return

    @staticmethod
    def _device_chip_rect(idx: int) -> Tuple[int, int, int, int]:
        return (DEV_CHIP_X0 + idx * DEV_CHIP_STEP, DEV_CHIP_Y, DEV_CHIP_W, DEV_CHIP_H)

    # ==================== 渲染 ====================
    def render(self) -> np.ndarray:
        """渲染 1100x720 逻辑画布"""
        canvas = np.full((LOGIC_H, LOGIC_W, 3), COLOR_BG, dtype=np.uint8)
        mpos = (self.mouse_x, self.mouse_y)
        self._draw_topbar(canvas, mpos)
        self._draw_device_chips(canvas, mpos)
        self._draw_state_card(canvas)
        self._draw_ctrl_card(canvas, mpos)
        self._draw_proto_card(canvas)
        self._draw_preview_card(canvas, mpos)
        self._draw_log_panel(canvas)
        self.draw_toast(canvas)
        return canvas

    def _draw_topbar(self, canvas, mpos):
        draw_text(canvas, "NetCamera 调试", (24, 24), font_size=17, color=COLOR_TEXT, bold=True)
        draw_text(canvas, f"MQTT {BROKER_HOST}:{BROKER_PORT}  (android_as_camera 手机网络摄像机)",
                  (240, 28), font_size=12, color=COLOR_SUB)
        # MQTT 连接分组面板: 标签 + 状态灯 + 连接/断开按钮
        px, py, pw, ph = MQTT_PANEL
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (20, 24, 30), -1)
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (44, 52, 66), 1)
        draw_text(canvas, "MQTT", (px + 12, py + 17), font_size=11, color=COLOR_MUTED, bold=True)
        if self._connected:
            dot_col, dot_text = COLOR_GREEN, "已连接"
        elif self._connecting:
            dot_col, dot_text = COLOR_WARN, "连接中"
        else:
            dot_col, dot_text = (90, 100, 115), "未连接"
        cv2.circle(canvas, (px + 56, py + ph // 2), 6, dot_col, -1)
        draw_text(canvas, dot_text, (px + 68, py + 16), font_size=12, color=dot_col, bold=True)
        self.draw_btn(canvas, BTN_CONNECT, "连接", mpos, theme_color=(0, 160, 200), enabled=not self._connected)
        self.draw_btn(canvas, BTN_DISCONNECT, "断开", mpos, theme_color=(140, 90, 60), enabled=self._connected)
        self.draw_btn(canvas, BTN_QUIT, "退出", mpos, theme_color=(120, 60, 60))

    def _draw_device_chips(self, canvas, mpos):
        draw_text(canvas, "摄像机 (在线/离线):", (24, 72), font_size=12, color=COLOR_SUB)
        dev_ids = self._sorted_dev_ids()
        if not dev_ids:
            draw_text(canvas, "(等待 camera/+/state 上报...)", (170, 72), font_size=12, color=COLOR_MUTED)
            return
        for i, devid in enumerate(dev_ids[:DEV_CHIP_MAX]):
            rx, ry, rw, rh = self._device_chip_rect(i)
            sel = (devid == self.selected_devid)
            online = self._device_online(devid)
            hov = self.pt_in(mpos[0], mpos[1], (rx, ry, rw, rh))
            bg = (24, 44, 52) if sel else ((34, 42, 54) if hov else COLOR_PANEL)
            border = COLOR_ACCENT if sel else COLOR_BORDER
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), bg, -1)
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), border, 2 if sel else 1)
            # 状态灯: 绿=90s 内有上报; 灰=超时离线 (retained 旧 state, 不可信)
            dot_col = COLOR_GREEN if online else (90, 100, 115)
            cv2.circle(canvas, (rx + 14, ry + rh // 2), 4, dot_col, -1)
            if online:
                label, tcol = devid[:10], (COLOR_TEXT if sel else COLOR_SUB)
            else:
                label, tcol = f"{devid[:6]}..离线", COLOR_MUTED
            draw_text(canvas, label, (rx + 26, ry + 8), font_size=12, color=tcol, bold=sel)

    def _draw_state_card(self, canvas):
        x, y, w, h = STATE_CARD
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_PANEL, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_BORDER, 1)
        dev = self.selected_devid or "(未选择)"
        online = self._device_online(self.selected_devid) if self.selected_devid else False
        tag = "在线" if online else "离线"
        tag_col = COLOR_GREEN if online else (200, 110, 90)
        draw_text(canvas, f"设备状态  {dev}", (x + 12, y + 8), font_size=12, color=COLOR_SUB)
        draw_text(canvas, tag, (x + w - 44, y + 8), font_size=12, color=tag_col, bold=True)

        with self._lock:
            st = dict(self.devices.get(self.selected_devid, {}))
        if not st:
            draw_text(canvas, "暂无 state 上报", (x + 12, y + 40), font_size=12, color=COLOR_MUTED)
            return
        if not online:
            stale = self._device_stale_s(self.selected_devid)
            draw_text(canvas, f"已离线 {stale}s 无上报, 以下为过期数据",
                      (x + 12, y + 24), font_size=11, color=(200, 110, 90))
        streaming = bool(st.get("streaming", False))
        cam = st.get("camera", "?")
        res = st.get("resolution", "?")
        ip = st.get("ip", "?")
        batt = st.get("battery", "?")
        rows = [
            ("镜头", str(cam), COLOR_TEXT if online else COLOR_MUTED),
            ("分辨率", f"{res}  (推流{'中' if streaming else '停'})", COLOR_TEXT if online else COLOR_MUTED),
            ("IP", f"{ip}:{RTSP_PORT}", COLOR_ACCENT if (ip != "?" and online) else COLOR_MUTED),
            ("电量", f"{batt}%" if isinstance(batt, int) else str(batt), COLOR_TEXT if online else COLOR_MUTED),
        ]
        for i, (label, val, col) in enumerate(rows):
            ry = y + 44 + i * 18
            draw_text(canvas, label, (x + 12, ry), font_size=11, color=COLOR_MUTED)
            draw_text(canvas, val, (x + 90, ry), font_size=11, color=col, bold=(i == 0))

    def _draw_ctrl_card(self, canvas, mpos):
        x, y, w, h = CTRL_CARD
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_PANEL, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_BORDER, 1)
        draw_text(canvas, "远程控制", (x + 12, y + 8), font_size=12, color=COLOR_SUB)
        draw_text(canvas, f"zoom {self.zoom_ratio:.1f}x   sens {self.sensitivity}",
                  (x + 190, y + 8), font_size=11, color=COLOR_ACCENT)

        # 推流状态化开关: 标签即状态 (推流中→点击停止; 未推流→点击开始)
        streaming = self._device_streaming()
        self.draw_btn(canvas, BTN_STREAM,
                       "推流中... 点击停止" if streaming else "开始推流",
                       mpos, theme_color=COLOR_GREEN if streaming else (0, 160, 120),
                       active=streaming)
        self.draw_btn(canvas, BTN_CAM_FRONT, "切换前摄", mpos)
        self.draw_btn(canvas, BTN_CAM_BACK, "切换后摄", mpos)
        self.draw_btn(canvas, BTN_ZOOM_DEC, "变焦 -", mpos)
        self.draw_btn(canvas, BTN_ZOOM_INC, "变焦 +", mpos)
        self.draw_btn(canvas, BTN_CAPTURE, "拍 照", mpos, theme_color=(0, 160, 200))
        self.draw_btn(canvas, BTN_RECORD, "停止录像" if self.recording else "开始录像",
                       mpos, theme_color=COLOR_RED, active=self.recording)
        self.draw_btn(canvas, BTN_MOTION,
                       f"移动侦测: {'开' if self.motion_enabled else '关'}", mpos,
                       theme_color=(0, 160, 120), active=self.motion_enabled)
        self.draw_btn(canvas, BTN_SENS_DEC, "灵敏度 -", mpos)
        self.draw_btn(canvas, BTN_SENS_INC, "灵敏度 +", mpos)

    def _draw_proto_card(self, canvas):
        x, y, w, h = PROTO_CARD
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (20, 24, 30), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (40, 48, 60), 1)
        lines = [
            "订阅  camera/+/state | camera/+/event",
            "下发  camera/{id}/cmd  {\"action\": ...}",
            "视频  rtsp://{ip}:8554/live  (TCP interleaved)",
            "命令  start/stop_stream, switch_camera,",
            "      zoom, capture, start/stop_record, motion",
        ]
        for i, ln in enumerate(lines):
            draw_text(canvas, ln, (x + 12, y + 10 + i * 19), font_size=10, color=(140, 152, 168))

    def _draw_preview_card(self, canvas, mpos):
        x, y, w, h = PREVIEW_CARD
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_PANEL, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), COLOR_BORDER, 1)
        draw_text(canvas, "RTSP 预览", (x + 12, y + 8), font_size=12, color=COLOR_SUB)
        if self._preview_url:
            draw_text(canvas, self._preview_url, (x + 120, y + 10), font_size=10, color=COLOR_MUTED)
        self.draw_btn(canvas, BTN_PV_START, "拉流预览", mpos,
                       theme_color=(0, 160, 200), enabled=not self._preview_running)
        self.draw_btn(canvas, BTN_PV_STOP, "停止预览", mpos,
                       theme_color=(140, 90, 60), enabled=self._preview_running)

        vx, vy, vw, vh = PREVIEW_VIDEO
        cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), (10, 12, 16), -1)
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            err = self._preview_error
        if frame is not None:
            fh, fw = frame.shape[:2]
            scale = min(vw / float(fw), vh / float(fh))
            tw, th = max(1, int(fw * scale)), max(1, int(fh * scale))
            scaled = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
            ox = vx + (vw - tw) // 2
            oy = vy + (vh - th) // 2
            canvas[oy:oy + th, ox:ox + tw] = scaled
            cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), (40, 48, 60), 1)
        else:
            msg = err or ("拉流中..." if self._preview_running else "点击 [拉流预览] 查看实时画面")
            est_w = 8 * len(msg) if msg.isascii() else 14 * len(msg)
            draw_text(canvas, msg, (vx + max(8, (vw - est_w) // 2), vy + vh // 2 - 8),
                      font_size=12, color=COLOR_WARN if err else COLOR_MUTED)

    def _draw_log_panel(self, canvas):
        x, y, w, h = LOG_CARD
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (18, 22, 28), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (40, 48, 60), 1)
        draw_text(canvas, f"事件日志 ({TOPIC_PREFIX}/+/event 与命令回显)", (x + 14, y + 8),
                  font_size=12, color=COLOR_SUB)
        area_y = y + LOG_HEADER_H + 4
        max_lines = (h - LOG_HEADER_H - 12) // LOG_LINE_H
        with self._lock:
            lines = list(self.log_lines)
        for i, ln in enumerate(lines[-max_lines:]):
            ly = area_y + i * LOG_LINE_H
            col = (255, 170, 120) if "!! " in ln else (185, 196, 208)
            draw_text(canvas, ln[:78], (x + 14, ly), font_size=11, color=col)


if __name__ == "__main__":
    app = NetCameraDebugApp()
    app.run()
