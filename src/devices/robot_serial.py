# -*- coding: utf-8 -*-
"""
机械臂串口控制 (RobotSerial)
============================
MKS Base V1.6 (Marlin 2.0+) G-code 串口通信封装 (FR-7.1 / FR-12.3)：
  - G-code 指令发送与 `ok` 应答闭环侦听
  - M114 末端位姿回读 (X/Y/Z 解析)
  - "抬起 -> 平移 -> 下探" 三段式安全移动路径 (M400 等待到位)
端口、波特率与安全参数默认读取 config.yaml 的 robot: 配置节。
"""

import os
import re
import time
import threading

try:
    import serial
except ImportError:
    serial = None

from src.utils.config_guard import load_raw_config
from src.utils.logger import get_logger

log = get_logger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")


def load_robot_config() -> dict:
    """读取 config.yaml 的 robot 配置节 (port / baudrate / safe_z_mm / feedrate)"""
    cfg = {
        "port": "COM3",
        "baudrate": 115200,
        "safe_z_mm": 80.0,
        "feedrate_travel": 4000,
        "feedrate_grip": 1500,
    }
    robot_cfg = load_raw_config(CONFIG_PATH).get("robot") or {}
    for key in cfg:
        if robot_cfg.get(key) is not None:
            cfg[key] = robot_cfg[key]
    return cfg


class RobotSerial:
    """机械臂串口控制器 (线程安全 / Marlin G-code 协议 / ok 应答闭环)"""

    def __init__(self, port: str = "", baudrate: int = 0):
        cfg = load_robot_config()
        self.port = port or str(cfg["port"])
        self.baudrate = int(baudrate) if baudrate else int(cfg["baudrate"])
        self.safe_z_mm = float(cfg.get("safe_z_mm", 80.0))
        self.feedrate_travel = int(cfg.get("feedrate_travel", 4000))
        self.feedrate_grip = int(cfg.get("feedrate_grip", 1500))
        self.ser = None
        self._lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self.ser is not None

    def connect(self) -> bool:
        """打开串口并初始化绝对坐标模式 (G21/G90)，等待控制器就绪"""
        if serial is None:
            raise RuntimeError("pyserial 未安装, 请执行: pip install 'pyserial>=3.5'")
        if self.ser is not None:
            return True
        ser = serial.Serial(self.port, self.baudrate, timeout=2.0)
        time.sleep(2.0)  # Marlin 上电初始化等待
        ser.reset_input_buffer()  # 丢弃启动横幅
        self.ser = ser
        self.send_gcode("G21")  # 单位 mm
        self.send_gcode("G90")  # 绝对坐标模式
        return True

    def close(self, timeout: float = 2.0):
        """关闭串口连接 (锁获取限时, 杜绝 UI 卡死):
        后台运动任务可能长时间持有 _lock (G1+M400 最长约 2 分钟/段), 无限等待会冻结调用方线程。
        超时未取得锁则强制标记断开 (ser=None), 后续所有操作立即快速失败, 保证界面始终可退出。
        """
        acquired = self._lock.acquire(timeout=timeout)
        try:
            ser = self.ser
            self.ser = None
            if acquired and ser is not None:
                try:
                    ser.close()
                except Exception:
                    pass
        finally:
            if acquired:
                self._lock.release()

    def _mark_dead(self):
        """串口致命异常 (设备拔出/拒绝访问/句柄失效): 立即标记断开, 防止僵尸连接反复报错。
        调用方必须已持有 _lock (或处于异常收尾路径), 本方法不再获取锁。"""
        try:
            if self.ser is not None:
                self.ser.close()
        except Exception:
            pass
        self.ser = None
        log.error("[RobotSerial] 串口发生致命 I/O 异常, 已强制断开 (请重新连接)")

    def send_gcode(self, cmd: str, expect_ok: bool = True,
                   timeout: float = 2.0, wait_done: bool = False) -> bool:
        """
        发送单条 G-code 并侦听 `ok` 应答闭环
        :param timeout: ok 应答等待上限 (秒), 运动指令需覆盖运动时长
        :param wait_done: True 时追加 M400 等待运动队列完全排空 (阻塞到位)
        """
        if self.ser is None:
            return False
        with self._lock:
            if not self._send_locked(cmd, expect_ok, timeout):
                return False
            if wait_done:
                return self._send_locked("M400", True, timeout)
            return True

    def _send_locked(self, cmd: str, expect_ok: bool, timeout: float) -> bool:
        """串口锁内发送与应答侦听 (调用方必须已持有 _lock)"""
        try:
            self.ser.reset_input_buffer()
            self.ser.write((cmd.strip() + "\n").encode("ascii"))
            if not expect_ok:
                return True
            deadline = time.time() + timeout
            while time.time() < deadline:
                line = self.ser.readline().decode("ascii", errors="ignore").strip()
                if line.lower().startswith("ok"):
                    return True
                if line.lower().startswith("error"):
                    log.error(f"[RobotSerial] 控制器报错: {line} (cmd: {cmd})")
                    return False
            log.warning(f"[RobotSerial] 应答超时: {cmd}")
            return False
        except Exception as e:
            log.error(f"[RobotSerial] 串口异常: {e}")
            self._mark_dead()   # 端口级 I/O 异常 (拒绝访问/拔出): 标记断开, 避免僵尸连接
            return False

    def get_position(self):
        """
        M114 回读当前末端坐标
        :return: (x, y, z) mm 元组, 失败返回 None
        """
        if self.ser is None:
            return None
        with self._lock:
            try:
                self.ser.reset_input_buffer()
                self.ser.write(b"M114\n")
                deadline = time.time() + 2.0
                while time.time() < deadline:
                    line = self.ser.readline().decode("ascii", errors="ignore").strip()
                    m = re.search(r"X:\s*([-\d.]+)\s+Y:\s*([-\d.]+)\s+Z:\s*([-\d.]+)(?:\s+E:\s*([-\d.]+))?", line)
                    if m:
                        x, y, z = float(m.group(1)), float(m.group(2)), float(m.group(3))
                        if m.group(4) is not None:
                            return (x, y, z, float(m.group(4)))
                        return (x, y, z)
                    if line.lower().startswith("ok"):
                        break
            except Exception as e:
                log.warning(f"[RobotSerial] M114 读取失败: {e}")
                self._mark_dead()   # 端口级 I/O 异常: 标记断开, 避免僵尸连接
        return None

    def move_to(self, x: float, y: float, z: float, r: float = None,
                feed: int = 0, safe_lift_mm: float = 0.0,
                step_cb=None, stage_pause_s: float = 0.0) -> bool:
        """
        三段式安全移动: 抬起 -> 平移 -> 下探, 每段 M400 等待到位
        :param r: 可选末端 R 轴旋转角度 (度, 映射为 Marlin E 轴)
        :param feed: XY 平移进给 (mm/min), 0 则用 config feedrate_travel
        :param safe_lift_mm: 抬起相对高度 (mm), 0 则用 config safe_z_mm
        :param step_cb: 可选回调 step_cb(cmd), 每段 G-code 发送前上报 (GUI 调试面板用)
        :param stage_pause_s: 段间停顿秒数 (抬起→平移、平移→下探之间各停一次, 调试观察用)
        """
        cur = self.get_position()
        if cur is None:
            log.warning("[RobotSerial] 无法读取当前位姿, 取消移动")
            return False
        lift = float(safe_lift_mm) if safe_lift_mm > 0 else self.safe_z_mm
        travel_feed = int(feed) if feed > 0 else self.feedrate_travel
        e_cmd = f" E{r:.2f}" if r is not None else ""
        steps = [
            (f"G1 Z{cur[2] + lift:.2f} F{self.feedrate_grip}", 60.0),   # 抬起
            (f"G1 X{x:.2f} Y{y:.2f}{e_cmd} F{travel_feed}", 60.0),      # 平移 (+可选旋转)
            (f"G1 Z{z:.2f} F{self.feedrate_grip}", 60.0),               # 下探
        ]
        for i, (cmd, tmo) in enumerate(steps):
            if step_cb is not None:
                try:
                    step_cb(cmd)
                except Exception:
                    pass
            if not self.send_gcode(cmd, timeout=tmo, wait_done=True):
                return False
            if stage_pause_s > 0 and i < len(steps) - 1:
                time.sleep(stage_pause_s)
        return True

    def set_gripper(self, close: bool = False, timeout: float = 2.0) -> bool:
        """
        控制夹爪开闭 (兼容气动 M3/M4 与舵机 M280 P1/P2 双轨道)
        :param close: True 为夹紧/闭合, False 为张开/释放
        """
        if close:
            # 闭合夹爪: 气动闭合 M4 + 舵机 Servo 1/2 旋转至 0° 抓紧
            cmds = ["M4", "M280 P1 S0", "M280 P2 S0"]
        else:
            # 打开夹爪: 气动开启 M3 + 舵机 Servo 1/2 旋转至 30° 张开
            cmds = ["M3", "M280 P1 S30", "M280 P2 S30"]

        ok = True
        for cmd in cmds:
            if not self.send_gcode(cmd, timeout=timeout):
                ok = False
        return ok

    def set_z_height(self, z_mm: float, timeout: float = 3.0) -> bool:
        """
        设置 Z 轴物理高度 (0 ~ 100 mm):
        SCARA 机械臂物理 Z 轴由 Servo 0 (板载 A11/D65 舵机) 驱动:
          - M280 P0 S<angle>: 物理舵机角度驱动, 映射关系: Z=0mm -> 270°, Z=100mm -> 0°
          - G92 Z<val>: 同步 Marlin 内部坐标状态
        """
        z_clamped = max(0.0, min(100.0, float(z_mm)))
        servo_angle = 270.0 - 2.7 * z_clamped
        ok1 = self.send_gcode(f"M280 P0 S{servo_angle:.0f}", timeout=timeout)
        ok2 = self.send_gcode(f"G92 Z{z_clamped:.2f}", timeout=timeout)
        return ok1 and ok2
