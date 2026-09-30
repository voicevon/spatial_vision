# -*- coding: utf-8 -*-
"""
SCARA 机械臂调试终端 (Flux Loader GUI)
========================================
基于 loader_core 业务核心的图形化调试界面，功能与 CLI 调试器一比一：
  - 串口连接管理 (自动枚举 / 手动输入 / 仿真 MOCK 模式)
  - 限位诊断 M119、一键回零 G28、设零 G92、坐标刷新 M114、释放电机 M84
  - 笛卡尔与关节角点动 (W/S/A/D/U/J/Q/E + O/L/I/K)、三档步长
  - Z 轴快捷升降与指定高度、双/单夹爪舵机控制
  - 直达目标坐标、预设工位跳转 (与 CLI 共享 ~/.flux_loader/presets.json)
  - 芦笋搬运节拍宏 (N 次循环)、原生 G-code 透传
业务逻辑全部委托 loader_core，本文件仅负责 GUI 事件循环与动作调度。
"""

import os
import re
import sys
import json
import time
import argparse
from collections import deque
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_SCARA_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCARA_DIR not in sys.path:
    sys.path.insert(0, _SCARA_DIR)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from loader_core import (  # noqa: E402
    JogController,
    LoaderConfig,
    MarlinProtocolHandler,
    PickAndPlaceWorkflow,
    Pose,
    ScaraRobot,
    SerialTransceiver,
    MockTransceiver,
)
from src.utils.logger import get_logger  # noqa: E402
from src.ui.base_cv_app import BaseCvApp  # noqa: E402
from tools.scara_debug.renderer import ScaraDebugRenderer, LOGIC_W, LOGIC_H  # noqa: E402

log = get_logger(__name__)


from src.ui.dialog_utils import prompt_input_text


class PresetManager:
    """预设特征点位管理器，只读加载 (优先 ~/.flux_loader/presets.json, 否则用内置硬编码默认工位)"""

    _DEFAULT_PRESETS: Dict[str, Pose] = {
        "机械零位 (Home Pose)":         Pose(x=0.0,    y=600.0, z=80.0, r=90.0),
        "安全待机位 (Standby)":          Pose(x=0.0,    y=300.0, z=80.0, r=90.0),
        "标准抓取位 (Pick Pose)":        Pose(x=250.0,  y=250.0, z=20.0, r=90.0),
        "落料入料口 (Dealer Drop Pose)": Pose(x=-250.0, y=350.0, z=80.0, r=90.0),
    }

    def __init__(self, filepath: str = "~/.flux_loader/presets.json") -> None:
        self._path = Path(filepath).expanduser()
        self._presets: Dict[str, Pose] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._presets = {name: Pose(**data) for name, data in raw.items()}
                return
            except Exception as exc:
                log.warning("读取预设文件失败 (%s)，使用内置默认值。", exc)
        self._presets = dict(self._DEFAULT_PRESETS)

    def list_presets(self) -> Dict[str, Pose]:
        return dict(self._presets)


class ScaraDebugApp(BaseCvApp):
    """SCARA 调试终端 GUI 主应用：事件循环、动作调度与日志缓冲 (基于 BaseCvApp)"""

    def __init__(self, port: Optional[str] = None, mock: bool = False):
        super().__init__(
            app_id="scara_debug",
            base_w=LOGIC_W,
            base_h=LOGIC_H,
            window_name="flux_vision_3d | scara_debug",
            window_title="flux_vision_3d | SCARA 调试",
            enable_keyboard_zoom=False,
        )
        self.mock_mode = mock
        self._config = LoaderConfig()
        tx = MockTransceiver() if mock else SerialTransceiver()
        self._handler = MarlinProtocolHandler(tx)
        self.robot = ScaraRobot(self._handler, self._config)
        self._handler.on_send = lambda cmd: self.add_log(f">> {cmd}")
        self.jog = JogController(self.robot)
        self._workflow = PickAndPlaceWorkflow(self.robot)
        self.presets = PresetManager(self._config.presets_file)

        self.port_list = []
        self.selected_port = port or ""
        self.current_port = ""
        self.macro_cycles = 1
        self.limit_lines = []
        self.log_lines = deque(maxlen=200)

        # 下拉框与自动刷新状态
        self.dd_serial_open = False
        self.dd_z_open = False
        self.dd_step_open = False
        self.auto_refresh = False
        self.connecting = False
        self._last_auto_refresh = 0.0

        self.renderer = ScaraDebugRenderer()

        self.refresh_ports()
        self.add_log("[就绪] SCARA 调试已启动，请选择串口并点击 [连接]。")

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------
    def add_log(self, line: str) -> None:
        self.log_lines.append(line)

    # ------------------------------------------------------------------
    # 串口连接
    # ------------------------------------------------------------------
    def refresh_ports(self) -> None:
        self.port_list = SerialTransceiver.list_ports() if not self.mock_mode else ["MOCK"]
        if not self.selected_port and self.port_list:
            # 优先推荐 MKS 控制板常用端口
            self.selected_port = next(
                (p for p in self.port_list if "COM11" in p.upper()), self.port_list[0])
        self.add_log(f"[串口] 已枚举 {len(self.port_list)} 个端口: {', '.join(self.port_list) or '无'}")

    def select_serial_port(self, port: str) -> None:
        """下拉框选择端口：若已连接则先断开旧连接 (切换串口)"""
        if port == self.selected_port:
            return
        self.selected_port = port
        if self.robot.is_connected():
            self.disconnect()
            self.add_log(f"[切换] 已断开旧连接，选中新串口 {port}，请点击下拉框 [连接]。")
        else:
            self.add_log(f"[串口] 已选择 {port}")

    def connect(self) -> None:
        port = self.selected_port
        if not port:
            port = prompt_input_text("手动连接", "请输入串口名 (如 COM11):")
            if not port:
                return
        self.add_log(f"[连接] 正在连接 {port} @ {self._config.default_baudrate}...")
        try:
            ok = self.robot.connect(port)
        except Exception as exc:
            self.add_log(f"[ERR] 连接异常: {exc}")
            return
        if ok:
            self.current_port = port
            self.robot.refresh_state()
            self.add_log(f"[OK] 已连接 {port}，基准坐标已同步。")
        else:
            self.add_log(f"[ERR] 连接 {port} 失败，请检查线缆与端口占用。")

    def disconnect(self) -> None:
        self.robot.disconnect()
        self.current_port = ""
        self.add_log("[OK] 已断开串口连接。")

    # ------------------------------------------------------------------
    # 状态与原点
    # ------------------------------------------------------------------
    def refresh_pos(self) -> None:
        if not self.robot.is_connected():
            self.add_log("[提示] 未连接，无法刷新坐标。")
            return
        self.robot.refresh_state()
        p = self.robot.current_pose
        self.add_log(f"[M114] X:{p.x:.1f} Y:{p.y:.1f} Z:{p.z:.1f} R:{p.r:.1f}")

    def do_m119(self) -> None:
        if not self._require_conn():
            return
        status = self.robot.get_limit_status()
        self.limit_lines = [f"{k}: {v}" for k, v in status.raw.items()]
        self.add_log(f"[M119] 限位诊断完成 ({len(self.limit_lines)} 项)。")

    def do_home(self) -> None:
        if not self._require_conn():
            return
        self.add_log("[G28] 正在三轴回零，请等待机械臂动作完成...")
        ok = self.robot.home()
        self.add_log("[OK] 回零完成。" if ok else "[ERR] 回零失败，请查看串口应答。")

    def do_set_origin(self) -> None:
        if not self._require_conn():
            return
        # G92 设机械零点 (定值, 不再弹框): X0 Y600 Z80 E90 (R 轴 → Marlin E 轴)
        hp = self._config.home_pose
        self.robot.set_coordinate_origin(x=hp.x, y=hp.y, z=hp.z, r=hp.r)
        self.add_log(f"[G92] 已设为机械零点 ({hp.x}, {hp.y}, {hp.z}, {hp.r})。")

    def do_m84(self) -> None:
        if not self._require_conn():
            return
        self.robot.disable_steppers()
        self.add_log("[安全] M84 已发送，现在可手动轻推关节。")

    # ------------------------------------------------------------------
    # 点动与末端工具
    # ------------------------------------------------------------------
    def do_jog(self, key: str) -> None:
        if not self._require_conn():
            return
        self.jog.handle_key(key)
        self.robot.refresh_state()

    def do_step(self, key: str) -> None:
        self.jog.set_step_profile(key)
        self.add_log(f"[步长] 已切换: {self.jog.step_info}")

    def do_z(self, z_mm: Optional[float] = None) -> None:
        if not self._require_conn():
            return
        if z_mm is None:
            raw = prompt_input_text("指定 Z 高度", "请输入目标 Z 轴高度 (0~100 mm):")
            if not raw:
                return
            try:
                z_mm = float(raw)
            except ValueError:
                self.add_log("[ERR] 请输入有效数字！")
                return
        self.robot.set_z_height(z_mm)
        self.robot.refresh_state()
        self.add_log(f"[Z 轴] 已移动至 {self.robot.current_pose.z:.1f} mm。")

    def do_gripper(self, action: str) -> None:
        if not self._require_conn():
            return
        g = self.robot.gripper
        cfg = self._config
        mapping = {
            "grip_close":   lambda: g.set_both_grippers(open_state=False),
            "grip_open":    lambda: g.set_both_grippers(open_state=True),
            "grip1_close":  lambda: g.set_gripper(cfg.gripper1_id, open_state=False),
            "grip1_open":   lambda: g.set_gripper(cfg.gripper1_id, open_state=True),
            "grip2_close":  lambda: g.set_gripper(cfg.gripper2_id, open_state=False),
            "grip2_open":   lambda: g.set_gripper(cfg.gripper2_id, open_state=True),
        }
        label = {
            "grip_close": "双夹爪闭合", "grip_open": "双夹爪打开",
            "grip1_close": "夹爪1闭合", "grip1_open": "夹爪1打开",
            "grip2_close": "夹爪2闭合", "grip2_open": "夹爪2打开",
        }[action]
        mapping[action]()
        self.add_log(f"[舵机] {label} 已执行 (开 S{cfg.gripper_open_angle} / 闭 S{cfg.gripper_close_angle})。")

    # ------------------------------------------------------------------
    # 直达坐标 / 工位 / 宏 / 透传
    # ------------------------------------------------------------------
    def do_goto(self) -> None:
        if not self._require_conn():
            return
        raw = prompt_input_text(
            "直达目标坐标",
            "输入目标 (如 X100 Y300 Z50 R0 F9000)\n省略的轴保持当前值:")
        if not raw:
            return
        p = self.robot.current_pose
        x = y = z = r = None
        f = self._config.default_feedrate
        for axis, holder in (("x", "x"), ("y", "y"), ("z", "z"), ("r", "r")):
            m = re.search(rf"[{axis}{axis.upper()}]([-+]?\d*\.?\d+)", raw)
            if m:
                val = float(m.group(1))
                if axis == "x": x = val
                elif axis == "y": y = val
                elif axis == "z": z = val
                else: r = val
        m = re.search(r"[Ff](\d+)", raw)
        if m:
            f = float(m.group(1))
        target = Pose(x=x if x is not None else p.x, y=y if y is not None else p.y,
                      z=z if z is not None else p.z, r=r if r is not None else p.r, f=f)
        self.add_log(f"[G1] 移动至 X:{target.x:.1f} Y:{target.y:.1f} Z:{target.z:.1f} R:{target.r:.1f} F{f:.0f}...")
        ok = self.robot.move_to_pose(target)
        self.robot.refresh_state()
        self.add_log("[OK] 已到达目标坐标。" if ok else "[ERR] 移动失败 (可能超出可达范围)。")

    def do_preset_jump(self, idx: int) -> None:
        if not self._require_conn():
            return
        presets = self.presets.list_presets()
        keys = list(presets.keys())
        if not (0 <= idx < len(keys)):
            return
        name = keys[idx]
        target = presets[name]
        self.add_log(f"[跳转] 前往 [{name}] (X:{target.x:.0f} Y:{target.y:.0f} Z:{target.z:.0f} R:{target.r:.0f})...")
        safe_z = self._config.home_pose.z
        self.robot.set_z_height(safe_z)
        self.robot.move_to_pose(Pose(x=target.x, y=target.y, z=safe_z, r=target.r))
        self.robot.set_z_height(target.z)
        self.robot.refresh_state()
        self.add_log(f"[OK] 已精准到达: {name}")

    def do_macro(self) -> None:
        if not self._require_conn():
            return
        cycles = max(1, self.macro_cycles)
        self.add_log(f"[宏] 开始芦笋搬运闭环测试 (共 {cycles} 轮)...")
        executed = 0
        for i in range(1, cycles + 1):
            self.add_log(f"[宏] === 第 {i}/{cycles} 轮 ===")
            ok = self._workflow.run()
            executed = i
            if not ok:
                self.add_log(f"[警告] 第 {i} 轮循环被中断或失败！")
                break
            if i < cycles:
                time.sleep(0.5)
        self.add_log(f"[完成] 搬运宏结束 (完成 {executed}/{cycles} 轮)。")

    def do_gcode(self) -> None:
        if not self._require_conn():
            return
        cmd = prompt_input_text("G-code 透传", "输入透传指令 (如 M119 / G28 / M114):")
        if not cmd:
            return
        self.add_log(f"> {cmd}")
        lines = self.robot.send_raw(cmd)
        for ln in lines:
            self.add_log(f"< {ln}")

    # ------------------------------------------------------------------
    # 事件处理
    # ------------------------------------------------------------------
    def _present_frame(self) -> None:
        """立即渲染并呈现一帧 (用于连接等阻塞动作前的界面反馈)"""
        canvas = self.renderer.render(self)
        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(30)

    def _require_conn(self) -> bool:
        if not self.robot.is_connected():
            self.add_log("[提示] 请先连接串口！")
            return False
        return True

    def _on_button(self, bid: str) -> None:
        if bid == "quit":
            self._running = False
        elif bid == "auto_refresh":
            self.auto_refresh = not self.auto_refresh
            self.add_log(f"[自动刷新] {'已开启 (0.3s 周期 M114)' if self.auto_refresh else '已关闭'}。")
        elif bid.startswith("dd_serial:"):
            arg = bid.split(":", 1)[1]
            if arg == "__refresh__":
                self.refresh_ports()
            else:
                self.select_serial_port(arg)
        elif bid == "conn_toggle":
            self.dd_serial_open = False
            if self.robot.is_connected():
                self.disconnect()
            else:
                self.connecting = True
                self._present_frame()  # 先显示"正在连接..."再阻塞连接
                try:
                    self.connect()
                finally:
                    self.connecting = False
        elif bid == "refresh_ports":
            self.refresh_ports()
        elif bid == "connect":
            self.connect()
        elif bid == "disconnect":
            self.disconnect()
        elif bid == "refresh_pos":
            self.refresh_pos()
        elif bid == "m119":
            self.do_m119()
        elif bid == "home":
            self.do_home()
        elif bid == "g92":
            self.do_set_origin()
        elif bid == "m84":
            self.do_m84()
        elif bid == "goto":
            self.do_goto()
        elif bid == "z_up":
            self.do_z(100.0)
        elif bid == "z_down":
            self.do_z(20.0)
        elif bid.startswith("grip_"):
            self.do_gripper(bid)
        elif bid.startswith("jog:"):
            self.do_jog(bid.split(":", 1)[1])
        elif bid.startswith("dd_step:"):
            self.do_step(bid.split(":", 1)[1])
        elif bid.startswith("preset:"):
            self.do_preset_jump(int(bid.split(":", 1)[1]))
        elif bid == "macro_minus":
            self.macro_cycles = max(1, self.macro_cycles - 1)
        elif bid == "macro_plus":
            self.macro_cycles = min(99, self.macro_cycles + 1)
        elif bid == "macro_run":
            self.do_macro()
        elif bid == "gcode_input":
            self.do_gcode()

    # ==================== BaseCvApp 钩子实现 ====================
    def on_tick(self):
        """自动刷新坐标 (checkbox 开启后每 0.3s 静默 M114, 不写日志)"""
        if self.auto_refresh and self.robot.is_connected() \
                and time.time() - self._last_auto_refresh >= 0.3:
            self._last_auto_refresh = time.time()
            try:
                self.robot.refresh_state()
            except Exception:
                pass

    def render(self) -> np.ndarray:
        """核心渲染: 同步鼠标坐标并调用 ScaraDebugRenderer 渲染"""
        self.renderer.mouse_x = self.mouse_x
        self.renderer.mouse_y = self.mouse_y
        return self.renderer.render(self)

    def on_click(self, x: int, y: int):
        """逻辑坐标点击分发"""
        bid = self.renderer.hit_test(x, y)
        # 下拉框开合优先处理
        if bid == "dd_open:serial":
            self.dd_serial_open = not self.dd_serial_open
            self.dd_z_open = False
            self.dd_step_open = False
            return
        if bid == "dd_open:z":
            self.dd_z_open = not self.dd_z_open and self.robot.is_connected()
            self.dd_serial_open = False
            self.dd_step_open = False
            return
        if bid == "dd_open:step":
            self.dd_step_open = not self.dd_step_open
            self.dd_serial_open = False
            self.dd_z_open = False
            return
        # 任一浮层展开时: 点击浮层项执行动作, 点击其他区域仅收起
        any_open = self.dd_serial_open or self.dd_z_open or self.dd_step_open
        if any_open:
            if self.dd_serial_open and bid.startswith("dd_serial:"):
                try:
                    self._on_button(bid)
                except Exception as exc:
                    log.exception("下拉框动作异常: %s", exc)
                    self.add_log(f"[ERR] 动作异常: {exc}")
            elif self.dd_z_open and bid.startswith("dd_z:"):
                self.do_z(float(bid.split(":", 1)[1]))
            elif self.dd_step_open and bid.startswith("dd_step:"):
                self.do_step(bid.split(":", 1)[1])
            self.dd_serial_open = False
            self.dd_z_open = False
            self.dd_step_open = False
            return
        if bid:
            try:
                self._on_button(bid)
            except Exception as exc:
                log.exception("按钮动作异常: %s", exc)
                self.add_log(f"[ERR] 动作异常: {exc}")

    def on_key(self, raw_key: int) -> bool:
        """按键事件响应"""
        fb_changed, _fb_toast = self.win_mgr.handle_keyboard_fallback(raw_key)
        if fb_changed:
            return True
        self._handle_key(raw_key)
        return True

    def _handle_key(self, raw_key: int) -> None:
        key = raw_key & 0xFF
        if raw_key in (27,):
            self._running = False
            return
        jog_keys = "wsadueqolik"
        ch = chr(key).lower() if 0 <= key < 256 else ""
        if ch in jog_keys and self.robot.is_connected():
            self.do_jog(ch)
        elif ch in ("1", "2", "3"):
            self.do_step(ch)
        elif raw_key == 32:  # 空格刷新坐标
            self.refresh_pos()

    def cleanup(self) -> None:
        """应用退出前清理"""
        self.add_log("[退出] 正在安全退出...")
        try:
            if self.robot.is_connected():
                self.robot.disconnect()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="SCARA 机械臂调试终端 (Flux Loader GUI)")
    parser.add_argument("--port", "-p", type=str, default=None, help="串口号 (如 COM11)")
    parser.add_argument("--mock", action="store_true", help="仿真模式 (不依赖真实硬件)")
    args = parser.parse_args()

    app = ScaraDebugApp(port=args.port, mock=args.mock)
    app.run()


if __name__ == "__main__":
    main()
