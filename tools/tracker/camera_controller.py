#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robot 在线跟踪 - 相机硬件控制器:
类型/分辨率 GUI 状态机 (下拉选项/持久化字段), 硬件启停/帧读取/内参刷新全部委托
统一取流服务 src/hardware/camera_service.py 的 CameraService。
任务互斥、Toast 提示、世界系联动等业务编排仍在主控制器 RobotOnlineTracker。
"""

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

from src.devices.camera_service import CameraService


class CameraController:
    """相机硬件管理: 只负责取流启停 / 内参刷新 / 帧读取, 不含业务编排"""

    def __init__(self, pnp_solver):
        self.pnp_solver = pnp_solver    # 引用主控制器的几何 PnP 求解器, 内参按实际分辨率刷新
        self.camera_type = "realsense"
        self.camera_options = [
            ("realsense", "RealSense D435"),
            ("usb",       "USB 普通摄像头"),
        ]
        self.resolution = "1280x720"
        self.resolution_options = [
            ("1280x720",  "1280 × 720  (推荐)"),
            ("1920x1080", "1920 × 1080"),
            ("848x480",   "848 × 480"),
            ("640x480",   "640 × 480"),
        ]
        self.pipeline_running = False
        _w, _h = self.resolution.split("x")
        self.frame_w, self.frame_h = int(_w), int(_h)
        # 统一取流服务: 内参经回调刷新引擎 (RealSense=标定内参 / USB=近似针孔)
        self._srv = CameraService(intrinsics_callback=self._apply_intrinsics)

    # ------------------------------ 状态切换 ------------------------------
    def set_resolution_key(self, res_key):
        """更新分辨率选择 (仅状态与帧尺寸, 不启停硬件)"""
        self.resolution = res_key
        _w, _h = res_key.split("x")
        self.frame_w, self.frame_h = int(_w), int(_h)

    # ------------------------------ 取流启停 ------------------------------
    def start(self):
        """按当前类型/分辨率启动取流并刷新引擎内参, 失败抛异常 (pipeline_running 不变)"""
        w, h = self.frame_w, self.frame_h
        if self.camera_type == "realsense":
            # 逐级尝试 30/15/8 fps (与建图内参一致, 按实际分辨率自适应)
            self._srv.start_realsense(
                w, h, fps=30 if w <= 1280 else 8,
                fallbacks=((w, h, 15), (w, h, 8)),
                mock_fallback=False)
        else:
            self._srv.start_usb(w, h)
        self.pipeline_running = True

    def stop(self):
        """幂等关闭取流"""
        self._srv.stop()
        self.pipeline_running = False

    def read_frame(self):
        """从当前后端读取一帧彩色图, 失败/未运行返回 None"""
        return self._srv.read_frame(timeout_ms=4000)

    # ------------------------------ 内参回调 ------------------------------
    def _apply_intrinsics(self, K, dist):
        """经 CameraService 推送的标定/近似内参刷新几何求解器"""
        if self.pnp_solver is not None:
            self.pnp_solver.camera_matrix = K
            self.pnp_solver.dist_coeffs = dist
