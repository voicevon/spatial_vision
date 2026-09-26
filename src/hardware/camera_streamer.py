#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机实时取流器 (Camera Streamer)
================================
统一取流服务 CameraService 的流式外观 (Facade)，保持简洁 API：
- 自动检测并优先连接 Intel RealSense D435 物理硬件 (1080P/720P 回退链)
- 未插相机或硬件异常时自动优雅切入高保真 AprilTag 仿真视频流 (Mock 模式)
- 保持低延迟、稳定帧率与 FPS 统计，适配 GUI 实时主循环与连拍归档
"""

import time

import numpy as np

from src.hardware.camera_service import CameraService
from src.utils.logger import get_logger

log = get_logger(__name__)

try:
    import pyrealsense2  # noqa: F401  # 可用性探测
    HAVE_REALSENSE = True
except ImportError:
    HAVE_REALSENSE = False


class CameraStreamer:
    """统一相机取流服务外观，支持硬件真实流与数学仿真流无缝切换"""

    def __init__(self, force_mock: bool = False, req_width: int = 1280, req_height: int = 720, req_fps: int = 30):
        self._srv = CameraService()
        self.force_mock = force_mock
        self.req_width = req_width
        self.req_height = req_height
        self.req_fps = req_fps
        self.frame_count = 0

        # FPS 统计
        self.fps = 0.0
        self._fps_last_time = time.time()
        self._fps_frame_count = 0

    # ------------------------------ 服务状态委托 ------------------------------
    @property
    def is_running(self) -> bool:
        return self._srv.is_running

    @property
    def is_mock(self) -> bool:
        return self._srv.is_mock

    @property
    def stream_desc(self) -> str:
        return self._srv.stream_desc

    @property
    def last_valid_frame(self):
        return self._srv.last_valid_frame

    # ------------------------------ 启停与读取 ------------------------------
    def start(self) -> bool:
        """启动相机流 (硬件优先, Mock 优雅回退)"""
        if self.is_running:
            return True

        if not self.force_mock and HAVE_REALSENSE:
            self._srv.start_realsense(
                self.req_width, self.req_height, fps=self.req_fps,
                fallbacks=((640, 480, 30),),
                mock_fallback=True)
            if not self._srv.is_mock:
                return True
        else:
            self._srv.enter_mock()
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        """非阻塞或微等待读取一帧 (BGR)"""
        frame = self._srv.read_frame(timeout_ms=100)
        if not self.is_running:
            return False, None

        self.frame_count += 1
        now = time.time()
        self._fps_frame_count += 1
        if now - self._fps_last_time >= 0.5:
            self.fps = self._fps_frame_count / (now - self._fps_last_time)
            self._fps_frame_count = 0
            self._fps_last_time = now

        if frame is not None:
            return True, frame
        # 硬件瞬时失败且无历史帧: Mock 模式必出帧, 此处兜底仿真帧
        return True, self._srv.make_mock_frame(self.frame_count)

    def stop(self):
        """停止取流并释放相机资源"""
        self._srv.stop()
