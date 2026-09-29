#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flux_vision_3d | isolate_wheels_production - 分离轮自动生产工作台
==============================================================
主生产运行调度程序：
- 步骤一（工位选择）：支持动态选择与加载工位 Workspace（含 rois.yaml 标靶与区域配置）
- 步骤二（相机与分辨率）：选择 RealSense/USB/Mock 并切换分辨率，启动/停止实时流
- 步骤三（视觉自动分离）：8 通道 ROI 视觉检测自动数出各轮物料数量，数字只读显示，无需人工干预
- 步骤四（生产流水线模式）：
  * 单次生产模式：人工触发节拍下发 (S)
  * 连续生产流水线模式：收到下位机 MQTT done 信号时，以最新视觉检出物料数自动下发下一个节拍 (SPACE)
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

from src.calibration.workspace_manager import Workspace, WorkspaceManager
from src.control.isolate_wheels_controller import IsolateWheelsController
from src.hardware.camera_service import CameraService
from src.utils.base_cv_app import BaseCvApp
from src.utils.logger import get_logger
from tools.isolate_wheels_production.renderer import (
    COLOR_BG,
    LOGIC_H,
    LOGIC_W,
    ProductionRenderer,
)
from tools.isolate_wheels_production.vision_detector import WheelVisionDetector

log = get_logger("flux_vision.tools.isolate_wheels_production.app")


class IsolateWheelsProductionApp(BaseCvApp):
    """分离轮 8 通道自动化视觉生产工作台主应用"""

    def __init__(self, settings_file: Optional[str] = None):
        super().__init__(
            app_id="isolate_wheels_production",
            base_w=LOGIC_W,
            base_h=LOGIC_H,
            window_name="flux_vision_3d | isolate_wheels_production",
            window_title="flux_vision_3d | 分离轮 8 通道视觉生产工作台",
            settings_file=settings_file,
            enable_keyboard_zoom=False,
        )

        # 核心控制中枢、工位管理器与视觉检测器
        self.controller = IsolateWheelsController()
        self.workspace_mgr = WorkspaceManager()
        self.camera_service = CameraService()
        self.detector = WheelVisionDetector()
        self.renderer = ProductionRenderer()

        # 1. 工位状态
        self.current_workspace_id: str = ""
        self._init_workspace()

        # 2. 相机配置与状态
        self.camera_types: List[Tuple[str, str]] = [
            ("realsense", "RealSense"),
            ("usb:0", "USB 相机 0"),
            ("mock", "Mock 模拟"),
        ]
        self.camera_type: str = "realsense"
        self.resolutions: List[Tuple[int, int]] = [(1280, 720), (640, 480)]
        self.camera_w: int = 1280
        self.camera_h: int = 720
        self.is_camera_running: bool = False
        self.camera_fps: float = 0.0
        self.current_frame: Optional[np.ndarray] = None
        self._frame_count: int = 0
        self._last_fps_time: float = time.time()

        # 3. 视觉自动计数与出料跟踪 (严禁人工干预加减，只由视觉更新)
        self.detected_counts: List[int] = [1] * 8
        self._last_sent_counts: List[int] = [0] * 8
        self.discharged_counts: List[int] = [0] * 8

        # 4. 生产流水线状态机
        self.auto_pipeline: bool = False
        self.paused: bool = False
        self.waiting_done: bool = False
        self.load_speed: float = 1.0
        self.total_cycles: int = 0
        self.cycle_start_time: float = 0.0
        self.last_beat_time: float = 0.0
        self.cycle_durations: List[float] = []

        # 5. UI 下拉弹窗展开标记
        self.popup_ws: bool = False
        self.popup_cam: bool = False
        self.popup_res: bool = False
        self.popup_speed: bool = False

        # 注册 MQTT 中枢事件回调
        self.controller.add_on_done_listener(self._on_done_event)
        self.controller.add_on_state_listener(self._on_state_event)

    # ==================== 兼容性属性映射 ====================
    @property
    def recipe_counts(self) -> List[int]:
        """兼容旧测试与外部调用: 配方数量直接映射至视觉检出数量"""
        return self.detected_counts

    @recipe_counts.setter
    def recipe_counts(self, val: List[int]):
        self.detected_counts = list(val)

    @property
    def auto_running(self) -> bool:
        """兼容旧测试属性 auto_running"""
        return self.auto_pipeline

    @auto_running.setter
    def auto_running(self, val: bool):
        self.auto_pipeline = val

    @property
    def selected_devid(self) -> str:
        return self.controller.selected_devid

    @property
    def workspace_list(self) -> List[Workspace]:
        return self.workspace_mgr.list_workspaces()

    @property
    def current_workspace_name(self) -> str:
        ws = self.workspace_mgr.get_workspace_by_id(self.current_workspace_id)
        return ws.name if ws else (self.current_workspace_id or "未选择工位")

    @property
    def current_camera_desc(self) -> str:
        for cid, cname in self.camera_types:
            if cid == self.camera_type:
                return cname
        return self.camera_type

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
        # 统计最近 10 次下发物料平均数，若无则使用当前检出数
        avg_pcs = max(1, sum(self.detected_counts))
        return (60.0 / ct) * avg_pcs

    # ==================== 工位与标靶管理 ====================
    def _init_workspace(self):
        cur_ws = self.workspace_mgr.get_current_workspace()
        if cur_ws:
            self.current_workspace_id = cur_ws.workspace_id
            self._load_workspace_rois(cur_ws)
        else:
            all_ws = self.workspace_list
            if all_ws:
                self.current_workspace_id = all_ws[0].workspace_id
                self._load_workspace_rois(all_ws[0])

    def switch_workspace(self, ws_id: str):
        """切换当前工位场景"""
        if ws_id == self.current_workspace_id:
            return
        ws = self.workspace_mgr.get_workspace_by_id(ws_id)
        if not ws:
            self.set_toast(f"工位 {ws_id} 不存在")
            return
        self.current_workspace_id = ws_id
        self.workspace_mgr.set_current_workspace(ws_id)
        self._load_workspace_rois(ws)
        self.set_toast(f"已切换至工位: {ws.name}")
        log.info(f"[Production] 切换工位 -> {ws.name} ({ws_id})")

    def _load_workspace_rois(self, ws: Workspace):
        """加载工位下的 rois.yaml 配置"""
        rois_file = os.path.join(ws.workspace_dir, "rois.yaml")
        if os.path.exists(rois_file):
            self.detector.load_rois(rois_file)
            log.info(f"[Production] 已加载工位 ROI 配置: {rois_file}")
        else:
            self.detector.generate_default_rois(self.camera_w, self.camera_h)
            log.info(f"[Production] 工位无专用 rois.yaml，采用等距 8 分区标准布局")

    # ==================== 相机启停与分辨率切换 ====================
    def open_camera(self) -> bool:
        """打开相机流"""
        if self.is_camera_running:
            return True
        log.info(f"[Production] 正在启动相机: {self.camera_type} ({self.camera_w}x{self.camera_h})")
        ok = False
        try:
            if self.camera_type == "realsense":
                ok = self.camera_service.start_realsense(self.camera_w, self.camera_h, fps=30, mock_fallback=True)
            elif self.camera_type.startswith("usb"):
                ok = self.camera_service.start_usb(width=self.camera_w, height=self.camera_h)
            else:
                ok = self.camera_service.enter_mock()
        except Exception as e:
            log.warning(f"[Production] 启动物理相机异常 ({e})，降级为 Mock 模式")
            ok = self.camera_service.enter_mock()

        if ok:
            self.is_camera_running = True
            self._frame_count = 0
            self._last_fps_time = time.time()
            self.set_toast(f"相机启动成功 [{self.current_camera_desc}]")
            log.info(f"[Production] 相机已启动: {self.current_camera_desc}")
        else:
            self.set_toast(f"相机启动失败: {self.current_camera_desc}")
            log.error(f"[Production] 相机启动失败: {self.current_camera_desc}")
        return ok

    def close_camera(self):
        """关闭相机流"""
        if not self.is_camera_running:
            return
        self.camera_service.stop()
        self.is_camera_running = False
        self.current_frame = None
        self.camera_fps = 0.0
        self.set_toast("相机已关闭")
        log.info("[Production] 相机已关闭。")

    def switch_camera_type(self, ctype: str):
        """切换相机类型"""
        if ctype == self.camera_type:
            return
        reopen = self.is_camera_running
        self.close_camera()
        self.camera_type = ctype
        if reopen:
            self.open_camera()

    def switch_resolution(self, width: int, height: int):
        """切换相机分辨率"""
        if width == self.camera_w and height == self.camera_h:
            return
        reopen = self.is_camera_running
        self.close_camera()
        self.camera_w = width
        self.camera_h = height
        # 重新生成默认 ROI 缩放适配
        self.detector.generate_default_rois(width, height)
        if reopen:
            self.open_camera()

    # ==================== 生命周期钩子 ====================
    def setup(self):
        log.info("[Production] 正在连接 MQTT 控制中枢 ...")
        self.controller.connect()
        self.set_toast("正在连接 MQTT Broker ...")
        # 默认尝试打开相机
        self.open_camera()

    def cleanup(self):
        self.auto_pipeline = False
        self.close_camera()
        log.info("[Production] 断开控制中枢连接并退出 ...")
        self.controller.disconnect()

    # ==================== 生产调度与流水线中枢交互 ====================
    def send_step_beat(self) -> bool:
        """
        执行单步生产节拍：
        读取视觉自动数出的 8 轮数量，下发至 MQTT load 节拍
        """
        if self.waiting_done:
            self.set_toast("当前节拍正在由下位机执行中，请等待完成...")
            return False

        # 读取最新视觉检测物料数
        counts_to_send = list(self.detected_counts)
        self._last_sent_counts = list(counts_to_send)

        ok, msg, payload = self.controller.send_load(counts_to_send, speed=self.load_speed)
        if ok:
            self.waiting_done = True
            self.cycle_start_time = time.time()
            self.set_toast(f"节拍已下发至 {self.selected_devid} | 视觉数出: {counts_to_send}")
            log.info(f"[Production] 节拍下发成功: {payload}")
        else:
            self.set_toast(f"节拍下发失败: {msg}")
            log.warning(f"[Production] 节拍下发拦截: {msg}")
            if self.auto_pipeline:
                self.auto_pipeline = False
        return ok

    def start_pipeline(self):
        """启动连续生产流水线模式"""
        if not self.controller.is_connected:
            self.set_toast("未连接 MQTT Broker，无法启动流水线")
            return
        self.auto_pipeline = True
        self.paused = False
        self.set_toast("已启动连续流水线模式 (收到 done 自动下发下一拍)")
        log.info("[Production] 连续流水线启动。")
        if not self.waiting_done:
            self.send_step_beat()

    def pause_pipeline(self):
        """暂停连续生产流水线"""
        self.auto_pipeline = False
        self.paused = True
        self.set_toast("流水线已暂停。")
        log.info("[Production] 流水线已暂停。")

    def emergency_stop(self):
        """急停生产"""
        self.auto_pipeline = False
        self.paused = False
        self.waiting_done = False
        self.controller.send_stop()
        self.set_toast("已下发急停指令！")
        log.warning("[Production] 生产急停。")

    def reset_batch_statistics(self):
        """清空重置批次统计"""
        self.total_cycles = 0
        self.discharged_counts = [0] * 8
        self.cycle_durations.clear()
        self.set_toast("批次生产统计已全部清空重置。")
        log.info("[Production] 批次统计已重置。")

    # ==================== MQTT 中枢事件响应 ====================
    def _on_done_event(self, devid: str, cmd_type: str):
        """
        下位机完成节拍事件回调：
        收到 MQTT done 时，累加累计出料，并在连续模式下自动以最新视觉检测数量下发下一拍
        """
        if devid == self.selected_devid and cmd_type == "load":
            now = time.time()
            if self.cycle_start_time > 0:
                duration = max(0.05, now - self.cycle_start_time)
                self.cycle_durations.append(duration)

            self.total_cycles += 1
            for i in range(8):
                self.discharged_counts[i] += self._last_sent_counts[i]

            self.waiting_done = False
            self.last_beat_time = now
            log.info(f"[Production] 节拍完成 (第 {self.total_cycles} 拍) -> 累计出料: {sum(self.discharged_counts)}")

            # 核心业务逻辑：连续流水线模式下，收到 done 立即自动触发下一拍
            if self.auto_pipeline and not self.paused:
                log.info("[Production] 流水线闭环: 收到 done，正在自动读取视觉物料并启动下一节拍...")
                self.send_step_beat()

    def _on_state_event(self, devid: str, state: str):
        """设备状态变动回调 (安全停机)"""
        if devid == self.selected_devid and state in ("offline", "fault"):
            if self.auto_pipeline:
                self.auto_pipeline = False
                self.waiting_done = False
                self.set_toast(f"设备处于异常状态 [{state}]，已安全中止自动流水线！")
                log.error(f"[Production] 设备异常 [{state}]，流水线中止！")

    # ==================== 主循环与事件调度 ====================
    def update(self):
        # 1. 相机取流与视觉检测自动计数
        if self.is_camera_running:
            frame = self.camera_service.read_frame()
            if frame is not None:
                self.current_frame = frame
                self._frame_count += 1
                now = time.time()
                elapsed = now - self._last_fps_time
                if elapsed >= 1.0:
                    self.camera_fps = self._frame_count / elapsed
                    self._frame_count = 0
                    self._last_fps_time = now

                # 步骤三：从 8 个 ROI 自动检出 8 轮数量，数字实时更新至 detected_counts
                self.detected_counts = self.detector.update_frame(frame)

        # 2. 流水线守护检查 (若未处于 waiting_done 且流水线激活，可自动步进)
        if self.auto_pipeline and not self.paused and not self.waiting_done:
            now = time.time()
            if (now - self.last_beat_time) >= 1.5:
                # 处于空闲时自动唤醒节拍
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

        # ---------------- 优先处理弹窗点击 ----------------
        if self.popup_ws:
            ws_list = self.workspace_list
            for idx, ws in enumerate(ws_list):
                iy = 48 + 4 + idx * 28
                if ProductionRenderer.pt_in(x, y, (434, iy, 192, 24)):
                    self.switch_workspace(ws.workspace_id)
                    self.popup_ws = False
                    return
            self.popup_ws = False
            return

        if self.popup_cam:
            cams = self.camera_types
            for idx, (cid, _) in enumerate(cams):
                iy = 48 + 4 + idx * 28
                if ProductionRenderer.pt_in(x, y, (604, iy, 132, 24)):
                    self.switch_camera_type(cid)
                    self.popup_cam = False
                    return
            self.popup_cam = False
            return

        if self.popup_res:
            res_list = self.resolutions
            for idx, (rw, rh) in enumerate(res_list):
                iy = 48 + 4 + idx * 28
                if ProductionRenderer.pt_in(x, y, (744, iy, 112, 24)):
                    self.switch_resolution(rw, rh)
                    self.popup_res = False
                    return
            self.popup_res = False
            return

        if self.popup_speed:
            speeds = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
            for idx, spd in enumerate(speeds):
                iy = 48 + 4 + idx * 28
                if ProductionRenderer.pt_in(x, y, (964, iy, 102, 24)):
                    self.load_speed = spd
                    self.popup_speed = False
                    self.set_toast(f"生产速度倍率已切换至: {spd:.1f}x")
                    return
            self.popup_speed = False
            return

        # ---------------- 顶部控制栏按钮 ----------------
        # 步骤 1：工位选择下拉
        if ProductionRenderer.pt_in(x, y, (430, 14, 160, 32)):
            self.popup_ws = not self.popup_ws
            return

        # 步骤 2：相机设备与分辨率选择
        if ProductionRenderer.pt_in(x, y, (600, 14, 130, 32)):
            self.popup_cam = not self.popup_cam
            return
        if ProductionRenderer.pt_in(x, y, (740, 14, 110, 32)):
            self.popup_res = not self.popup_res
            return

        # 相机打开 / 关闭按钮
        if ProductionRenderer.pt_in(x, y, (860, 14, 90, 32)):
            if self.is_camera_running:
                self.close_camera()
            else:
                self.open_camera()
            return

        # 速度倍率下拉按钮
        if ProductionRenderer.pt_in(x, y, (960, 14, 100, 32)):
            self.popup_speed = not self.popup_speed
            return

        # ---------------- 底部流水线控制按钮 ----------------
        by = 560
        # 启动 / 暂停连续流水线
        if ProductionRenderer.pt_in(x, y, (32, by + 40, 220, 46)):
            if self.auto_pipeline:
                self.pause_pipeline()
            else:
                self.start_pipeline()
            return

        # 单次执行节拍 (S)
        if ProductionRenderer.pt_in(x, y, (264, by + 40, 180, 46)):
            if not self.auto_pipeline:
                self.send_step_beat()
            return

        # 急停 (ESC)
        if ProductionRenderer.pt_in(x, y, (454, by + 40, 120, 46)):
            self.emergency_stop()
            return

        # 重置批次统计 (C)
        if ProductionRenderer.pt_in(x, y, (584, by + 40, 130, 46)):
            self.reset_batch_statistics()
            return

    def handle_key(self, key: int) -> bool:
        if key in (ord(' '),):
            if self.auto_pipeline:
                self.pause_pipeline()
            else:
                self.start_pipeline()
            return True
        elif key in (ord('s'), ord('S')):
            if not self.auto_pipeline:
                self.send_step_beat()
            return True
        elif key in (27,):  # ESC
            self.emergency_stop()
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
