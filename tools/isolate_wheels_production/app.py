#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flux_vision_3d | isolate_wheels_production - 分离轮自动生产工作台
==============================================================
主生产运行调度程序：
- 依赖公共核心控制器 IsolateWheelsController
- 工业看板架构：8 通道生产状态、速度倍率缩放、PPM 生产效率监控
- 闭环调度：支持自动连续循环生产与单拍步进出料
"""

import os
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.control.isolate_wheels_controller import IsolateWheelsController
from src.utils.base_cv_app import BaseCvApp
from src.utils.logger import get_logger
from tools.isolate_wheels_production.renderer import (
    COLOR_BG,
    LOGIC_H,
    LOGIC_W,
    ProductionRenderer,
)

log = get_logger("flux_vision.tools.isolate_wheels_production.app")


class IsolateWheelsProductionApp(BaseCvApp):
    """分离轮 8 通道自动生产工作台主应用"""

    def __init__(self, settings_file: Optional[str] = None):
        super().__init__(
            app_id="isolate_wheels_production",
            base_w=LOGIC_W,
            base_h=LOGIC_H,
            window_name="flux_vision_3d | isolate_wheels_production",
            window_title="flux_vision_3d | 分离轮 8 通道自动生产工作台",
            settings_file=settings_file,
            enable_keyboard_zoom=False,
        )

        # 核心控制中枢与渲染器
        self.controller = IsolateWheelsController()
        self.renderer = ProductionRenderer()

        # 生产控制与运行状态
        self.auto_running: bool = False
        self.paused: bool = False
        self.cycle_delay: float = 1.0
        self.load_speed: float = 1.0
        self.popup_speed: bool = False

        # 生产配方与数据统计
        self.recipe_counts: List[int] = [1] * 8
        self.discharged_counts: List[int] = [0] * 8
        self.total_cycles: int = 0
        self.cycle_start_time: float = 0.0
        self.last_beat_time: float = 0.0
        self.cycle_durations: List[float] = []

        # 注册中枢回调
        self.controller.add_on_done_listener(self._on_done_event)
        self.controller.add_on_state_listener(self._on_state_event)

    @property
    def selected_devid(self) -> str:
        return self.controller.selected_devid

    @property
    def avg_cycle_time(self) -> float:
        if not self.cycle_durations:
            return 0.0
        return sum(self.cycle_durations[-10:]) / len(self.cycle_durations[-10:])

    def calculate_ppm(self) -> float:
        """计算当前每分钟生产工件数 (PPM)"""
        ct = self.avg_cycle_time
        if ct <= 0.001:
            return 0.0
        pcs_per_cycle = sum(self.recipe_counts)
        return (60.0 / ct) * pcs_per_cycle

    # ==================== 生命周期钩子 ====================
    def setup(self):
        log.info("[Production] 正在连接 MQTT 控制中枢 ...")
        self.controller.connect()
        self.set_toast("正在连接 MQTT Broker ...")

    def cleanup(self):
        self.auto_running = False
        log.info("[Production] 断开控制中枢连接并退出 ...")
        self.controller.disconnect()

    # ==================== 生产调度与中枢交互 ====================
    def send_step_beat(self) -> bool:
        """执行单步生产节拍"""
        if not any(c > 0 for c in self.recipe_counts):
            self.set_toast("当前配方落料数为 0，无法生产。")
            return False

        ok, msg, payload = self.controller.send_load(self.recipe_counts, speed=self.load_speed)
        if ok:
            self.cycle_start_time = time.time()
            self.set_toast(f"生产节拍已下发至 {self.selected_devid} (速度 {self.load_speed:.1f}x)")
            log.info(f"[Production] 节拍下发: {payload}")
        else:
            self.set_toast(f"下发失败: {msg}")
            if self.auto_running:
                log.warning(f"[Production] 自动生产被拦截: {msg}")
        return ok

    def start_auto_production(self):
        """启动连续自动生产循环"""
        if not self.controller.is_connected:
            self.set_toast("尚未连接 Broker，无法启动自动生产。")
            return
        self.auto_running = True
        self.paused = False
        self.last_beat_time = 0.0
        self.set_toast("已启动连续自动生产循环 (按 SPACE 可暂停)")
        log.info("[Production] 连续自动生产已启动。")
        self.send_step_beat()

    def pause_auto_production(self):
        """暂停自动生产"""
        self.paused = True
        self.auto_running = False
        self.set_toast("生产已暂停。")
        log.info("[Production] 生产已暂停。")

    def reset_batch_statistics(self):
        """清空重置批次统计"""
        self.total_cycles = 0
        self.discharged_counts = [0] * 8
        self.cycle_durations.clear()
        self.set_toast("批次生产统计已全部重置。")
        log.info("[Production] 批次统计已重置。")

    def _on_done_event(self, devid: str, cmd_type: str):
        """设备完成节拍事件回调"""
        if devid == self.selected_devid and cmd_type == "load":
            now = time.time()
            if self.cycle_start_time > 0:
                duration = max(0.1, now - self.cycle_start_time)
                self.cycle_durations.append(duration)

            self.total_cycles += 1
            for i in range(8):
                self.discharged_counts[i] += self.recipe_counts[i]

            self.last_beat_time = now
            log.info(f"[Production] 节拍完成 (第 {self.total_cycles} 拍)")

    def _on_state_event(self, devid: str, state: str):
        """设备状态变动回调 (安全停机)"""
        if devid == self.selected_devid and state in ("offline", "fault"):
            if self.auto_running:
                self.auto_running = False
                self.set_toast(f"设备处于异常状态 [{state}]，已紧急停止自动生产！")
                log.error(f"[Production] 设备异常 [{state}]，生产自动中止！")

    # ==================== 主循环与事件调度 ====================
    def update(self):
        # 连续自动生产状态机调度
        if self.auto_running and not self.paused:
            now = time.time()
            if self.controller.is_device_idle(self.selected_devid):
                if (now - self.last_beat_time) >= self.cycle_delay:
                    self.send_step_beat()

    def render(self) -> np.ndarray:
        canvas = np.full((LOGIC_H, LOGIC_W, 3), COLOR_BG, dtype=np.uint8)
        mpos = (self.mouse_x, self.mouse_y)
        self.renderer.render(self, canvas, mpos)
        self.draw_toast(canvas)
        return canvas

    # ==================== 鼠标与快捷键交互 ====================
    def handle_mouse(self, event: int, x: int, y: int, flags: int, param):
        super().handle_mouse(event, x, y, flags, param)
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # 1. 速度倍率下拉弹窗处理
        if self.popup_speed:
            speeds = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
            for idx, spd in enumerate(speeds):
                iy = 44 + 4 + idx * 28
                if ProductionRenderer.pt_in(x, y, (1120 + 4, iy, 132, 24)):
                    self.load_speed = spd
                    self.popup_speed = False
                    self.set_toast(f"生产速度倍率已切换至: {spd:.1f}x")
                    return
            self.popup_speed = False
            return

        # 2. 点击速度下拉按钮
        if ProductionRenderer.pt_in(x, y, (1120, 14, 140, 28)):
            self.popup_speed = not self.popup_speed
            return

        # 3. 8 通道卡片步进按钮处理
        card_w, gap, start_x, card_y = 118, 14, 32, 112
        for i in range(8):
            cx = start_x + i * (card_w + gap)
            btn_minus = (cx + 10, card_y + 106, 44, 26)
            btn_plus = (cx + card_w - 54, card_y + 106, 44, 26)
            if ProductionRenderer.pt_in(x, y, btn_minus):
                if self.recipe_counts[i] > 0:
                    self.recipe_counts[i] -= 1
                return
            if ProductionRenderer.pt_in(x, y, btn_plus):
                if self.recipe_counts[i] < 9:
                    self.recipe_counts[i] += 1
                return

        # 4. 生产控制区按钮
        py = 335
        # 启动 / 暂停
        if ProductionRenderer.pt_in(x, y, (32, py + 45, 210, 44)):
            if self.auto_running:
                self.pause_auto_production()
            else:
                self.start_auto_production()
            return

        # 单拍出料
        if ProductionRenderer.pt_in(x, y, (255, py + 45, 180, 44)):
            if not self.auto_running:
                self.send_step_beat()
            return

        # 停止生产
        if ProductionRenderer.pt_in(x, y, (450, py + 45, 140, 44)):
            self.pause_auto_production()
            return

        # 重置批次统计
        if ProductionRenderer.pt_in(x, y, (605, py + 45, 135, 44)):
            self.reset_batch_statistics()
            return

        # 配方快捷预设
        if ProductionRenderer.pt_in(x, y, (120, py + 105, 110, 32)):
            self.recipe_counts = [1] * 8
            self.set_toast("配方已快速设为全部 1 件")
            return
        if ProductionRenderer.pt_in(x, y, (240, py + 105, 110, 32)):
            self.recipe_counts = [2] * 8
            self.set_toast("配方已快速设为全部 2 件")
            return
        if ProductionRenderer.pt_in(x, y, (360, py + 105, 110, 32)):
            self.recipe_counts = [0] * 8
            self.set_toast("配方已清空")
            return

        # 节拍间隔
        delays = [0.5, 1.0, 2.0, 3.0]
        for idx, d in enumerate(delays):
            btn_d = (120 + idx * 75, py + 152, 65, 30)
            if ProductionRenderer.pt_in(x, y, btn_d):
                self.cycle_delay = d
                self.set_toast(f"生产循环间隔设为: {d}s")
                return

    def handle_key(self, key: int) -> bool:
        if key in (ord(' '),):
            if self.auto_running:
                self.pause_auto_production()
            else:
                self.start_auto_production()
            return True
        elif key in (ord('s'), ord('S')):
            if not self.auto_running:
                self.send_step_beat()
            return True
        elif key in (ord('c'), ord('C')):
            self.reset_batch_statistics()
            return True
        return super().handle_key(key)


def main():
    app = IsolateWheelsProductionApp()
    app.run()


if __name__ == "__main__":
    main()
