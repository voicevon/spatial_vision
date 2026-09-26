# -*- coding: utf-8 -*-
"""
硬件抽象层 (Hardware Abstraction Layer - HAL)
==============================================
提供物理传感器、执行器与相机等外设的统一硬件管理、容错回退与仿真驱动。
"""

from src.hardware.camera_service import CameraService
from src.hardware.camera_streamer import CameraStreamer

__all__ = ["CameraService", "CameraStreamer"]
