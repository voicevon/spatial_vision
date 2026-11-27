#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分拣机下位机串口通信链路 (SorterSerialLink)
===========================================
面向项目 1 成熟嵌入式控制器 (MCU / PLC) 的 USB 串口 / RS485 文本行协议链路：
  - MCU -> PC: `TRIG,<seq>\n` (光电/编码器确认单根芦笋到位，请求测量)
  - PC -> MCU: `RES,<seq>,<grade>,<len_mm>,<diam_mm>,<status>\n`
  - 心跳探测: `PING\n` <-> `PONG,<ready:0|1>\n`

编解码逻辑独立为纯函数，若后续下位机改为二进制/Modbus 帧，仅需替换编解码函数即可。
支持 simulate=True 模式，无需物理串口即可通过 inject_trigger() 跑通全链路闭环。
"""

import threading
import time
from typing import Callable, Optional, Tuple

try:
    import serial
except ImportError:
    serial = None

from src.utils.logger import get_logger

log = get_logger(__name__)


def parse_mcu_line(raw_line: str) -> Tuple[str, Optional[int]]:
    """
    解析下位机发来的单行文本指令。
    返回:
      - ("TRIG", seq_int)
      - ("PING", None)
      - ("UNKNOWN", None)
    """
    line = raw_line.strip()
    if not line:
        return "EMPTY", None
    parts = [p.strip() for p in line.split(",")]
    cmd = parts[0].upper()
    if cmd == "TRIG" and len(parts) >= 2:
        try:
            return "TRIG", int(parts[1])
        except ValueError:
            return "UNKNOWN", None
    if cmd == "PING":
        return "PING", None
    return "UNKNOWN", None


def format_result_line(
    seq: int,
    grade: int,
    length_mm: float,
    diameter_mm: float,
    status: str = "OK",
) -> str:
    """格式化回传给下位机的测量分级结果行"""
    return f"RES,{int(seq)},{int(grade)},{float(length_mm):.1f},{float(diameter_mm):.1f},{status}\n"


def format_pong_line(ready: bool) -> str:
    """格式化心跳应答行"""
    return f"PONG,{1 if ready else 0}\n"


class SorterSerialLink:
    """线程安全的分拣下位机串口通信中枢"""

    def __init__(
        self,
        port: str = "COM3",
        baudrate: int = 115200,
        timeout_ms: int = 50,
        simulate: bool = False,
        on_trigger: Optional[Callable[[int], None]] = None,
        is_ready_fn: Optional[Callable[[], bool]] = None,
    ):
        self.port = port
        self.baudrate = int(baudrate)
        self.timeout_s = max(0.01, float(timeout_ms) / 1000.0)
        self.simulate = bool(simulate)
        self.on_trigger = on_trigger
        self.is_ready_fn = is_ready_fn or (lambda: True)

        self._ser = None
        self._tx_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._rx_thread: Optional[threading.Thread] = None
        self.last_tx_line: str = ""

    @property
    def is_connected(self) -> bool:
        return self.simulate or (self._ser is not None and bool(getattr(self._ser, "is_open", False)))

    def start(self) -> bool:
        """打开串口并启动后台接收线程（simulate 模式下直接就绪）"""
        self._stop_event.clear()
        if self.simulate:
            log.info("[SorterSerial] 仿真模式已启动 (无物理串口)")
            return True

        if serial is None:
            raise RuntimeError("pyserial 未安装，请执行: pip install 'pyserial>=3.5'")

        self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout_s)
        self._ser.reset_input_buffer()
        self._rx_thread = threading.Thread(
            target=self._rx_loop, name="SorterSerialRX", daemon=True
        )
        self._rx_thread.start()
        log.info(f"[SorterSerial] 已连接 {self.port} @ {self.baudrate}bps")
        return True

    def stop(self) -> None:
        """停止接收线程并安全关闭串口"""
        self._stop_event.set()
        if self._rx_thread is not None and self._rx_thread.is_alive():
            self._rx_thread.join(timeout=1.0)
        self._rx_thread = None
        with self._tx_lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None

    def send_result(
        self,
        seq: int,
        grade: int,
        length_mm: float,
        diameter_mm: float,
        status: str = "OK",
    ) -> str:
        """发送测量与分级结果到下位机，返回实际发送的文本行"""
        line = format_result_line(seq, grade, length_mm, diameter_mm, status)
        self._write_line(line)
        return line

    def send_pong(self, ready: bool) -> str:
        """发送心跳就绪状态到下位机"""
        line = format_pong_line(ready)
        self._write_line(line)
        return line

    def inject_trigger(self, seq: int) -> None:
        """手动或仿真注入一次下位机触发事件"""
        self._dispatch_line(f"TRIG,{int(seq)}\n")

    def _write_line(self, line: str) -> None:
        self.last_tx_line = line
        if self.simulate:
            log.debug(f"[SorterSerial:SIM TX] {line.strip()}")
            return
        with self._tx_lock:
            if self._ser is None:
                log.warning(f"[SorterSerial] 串口未连接，丢弃发送: {line.strip()}")
                return
            try:
                self._ser.write(line.encode("ascii"))
                self._ser.flush()
            except Exception as e:
                log.error(f"[SorterSerial] 串口写入异常: {e}")

    def _dispatch_line(self, raw_line: str) -> None:
        cmd, seq = parse_mcu_line(raw_line)
        if cmd == "TRIG" and seq is not None:
            if self.on_trigger is not None:
                self.on_trigger(seq)
        elif cmd == "PING":
            self.send_pong(bool(self.is_ready_fn()))
        elif cmd == "UNKNOWN":
            log.warning(f"[SorterSerial] 忽略未识别下位机指令: {raw_line.strip()!r}")

    def _rx_loop(self) -> None:
        while not self._stop_event.is_set():
            ser = self._ser
            if ser is None:
                break
            try:
                raw = ser.readline()
                if not raw:
                    continue
                text = raw.decode("ascii", errors="ignore")
                self._dispatch_line(text)
            except Exception as e:
                if not self._stop_event.is_set():
                    log.error(f"[SorterSerial] 串口读取异常: {e}")
                    time.sleep(0.1)
