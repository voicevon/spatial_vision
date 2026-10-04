#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一相机取流服务 (CameraService)
================================
收编重复的相机硬件管理 (tools/tracker/camera_controller、src/devices/camera_streamer、
tools/capture/capture_wizard)：
  - RealSense D435 / USB 摄像头 双物理后端统一启停与帧读取；
  - 分级回退链 (帧率/分辨率逐级降级)；
  - 内参解析推送: RealSense 走 config_guard 标定内参 (按实际分辨率自适应)，
    USB 用近似针孔模型，经 intrinsics_callback 推送给宿主 (如 tracker 刷新引擎)。
工具层只保留 GUI 状态与业务编排，硬件操作全部委托本服务。
"""

import os
import math

import numpy as np
import cv2

from src.utils.logger import get_logger

log = get_logger(__name__)

try:
    import pyrealsense2 as rs
    HAVE_REALSENSE = True
except ImportError:
    HAVE_REALSENSE = False

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")

try:
    from src.utils.config_guard import resolve_camera_intrinsics
except ImportError:
    resolve_camera_intrinsics = None


class CameraService:
    """统一相机取流服务: 硬件启停 / 帧读取 / 曝光控制"""

    def __init__(self, intrinsics_callback=None):
        """
        :param intrinsics_callback: callable(K, dist) — 后端启动成功后推送内参
                                    (RealSense 为标定内参, USB 为近似针孔模型)
        """
        self.intrinsics_callback = intrinsics_callback

        self.pipeline = None          # rs.pipeline (RealSense 后端)
        self.usb_capture = None       # cv2.VideoCapture (USB 后端)
        self.is_running = False
        self.stream_desc = "未初始化"
        self.color_sensor = None      # rs.sensor 彩色传感器句柄 (曝光调控)
        self.last_valid_frame = None  # 最近一帧有效图像 (瞬时失败回退用)

    @staticmethod
    def probe_available_devices() -> dict:
        """探测当前主机连接的相机硬件列表 (RealSense 设备序列号、USB 摄像头等)"""
        devices = {"realsense": [], "usb": []}
        if HAVE_REALSENSE:
            try:
                ctx = rs.context()
                for dev in ctx.query_devices():
                    name = dev.get_info(rs.camera_info.name) if dev.supports(rs.camera_info.name) else "RealSense Device"
                    serial = dev.get_info(rs.camera_info.serial_number) if dev.supports(rs.camera_info.serial_number) else ""
                    devices["realsense"].append({"name": name, "serial": serial})
            except Exception as e:
                log.debug(f"[CameraService] 探测 RealSense 异常: {e}")
        for idx in range(3):
            try:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    devices["usb"].append({"name": f"USB Video Camera #{idx}", "index": idx})
                    cap.release()
            except Exception:
                pass
        return devices

    # ------------------------------ 启动 ------------------------------
    def start_realsense(self, width, height, fps=30, serial=None, fallbacks=()) -> bool:
        """
        启动 RealSense 彩色流。
        :param serial: 指定 RealSense 硬件序列号 (为 None 或空则使用首台可用设备)
        :param fallbacks: ((w, h, fps), ...) 逐级降级链, 主档失败后依次尝试
        """
        if not HAVE_REALSENSE:
            raise RuntimeError("pyrealsense2 未安装, 请先安装 RealSense SDK")
        try:
            ctx = rs.context()
            devices = list(ctx.query_devices())
            if not devices:
                raise RuntimeError("未检测到 RealSense 设备, 请检查 USB 连接")

            last_err = None
            cur = (width, height, fps)
            for w, h, f in ((width, height, fps),) + tuple(fallbacks):
                try:
                    pipeline = rs.pipeline()
                    cfg = rs.config()
                    if serial and str(serial).strip():
                        cfg.enable_device(str(serial).strip())
                    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)
                    pipeline.start(cfg)
                    self.pipeline = pipeline
                    cur = (w, h, f)
                    dev_sn = f" (S/N: {serial})" if serial else ""
                    self.stream_desc = f"{w}x{h} @ {f}fps{dev_sn}"
                    log.info(f"[Camera] RealSense 彩色流: {w}x{h} @ {f}fps{dev_sn}")
                    break
                except Exception as e:
                    last_err = e
                    log.info(f"[Camera] {w}x{h} @ {f}fps 请求未满足: {e}")
            else:
                raise RuntimeError(f"RealSense {width}x{height} 启动失败: {last_err}")

            self._warmup_realsense()
            self._probe_color_sensor()
            self.is_running = True
            self._push_realsense_intrinsics(cur[0], cur[1])
            return True
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"连接物理相机失败: {e}")

    def start_usb(self, width, height, fps=30, device_index=0) -> bool:
        """启动普通 USB 摄像头 (cv2.VideoCapture)，使用近似针孔内参 (未标定)"""
        dev_idx = int(device_index) if str(device_index).isdigit() else 0
        cap = cv2.VideoCapture(dev_idx)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开 USB 摄像头 (index={dev_idx})")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        self.usb_capture = cap
        self.is_running = True
        self.stream_desc = f"USB[#{dev_idx}] {width}x{height}"
        self._push_usb_intrinsics(width, height)
        log.warning(f"[Camera] USB 摄像头已开启 (Dev #{dev_idx}) {width}x{height} (未标定内参, 世界坐标仅供流程验证)")
        return True

    def stop(self):
        """幂等关闭取流并释放资源"""
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
        if self.usb_capture is not None:
            try:
                self.usb_capture.release()
            except Exception:
                pass
            self.usb_capture = None
        self.color_sensor = None
        self.is_running = False

    # ------------------------------ 帧读取 ------------------------------
    def read_frame(self, timeout_ms: int = 1000):
        """读取一帧 BGR 图像; 瞬时失败回退最近有效帧; 未运行返回 None"""
        if not self.is_running:
            return None
        try:
            if self.pipeline is not None:
                frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
                color = frames.get_color_frame()
                if color:
                    bgr = np.asanyarray(color.get_data())
                    self.last_valid_frame = bgr
                    return bgr
            elif self.usb_capture is not None:
                ok, frame = self.usb_capture.read()
                if ok:
                    self.last_valid_frame = frame
                    return frame
        except Exception:
            pass
        return self.last_valid_frame

    def toggle_auto_exposure(self):
        """乒乓切换自动曝光; 返回描述文本, 无彩色传感器或不支持返回 None"""
        if self.color_sensor is None:
            return None
        try:
            if not self.color_sensor.supports(rs.option.enable_auto_exposure):
                return None
            cur = self.color_sensor.get_option(rs.option.enable_auto_exposure)
            new_state = 0 if cur > 0.5 else 1
            self.color_sensor.set_option(rs.option.enable_auto_exposure, new_state)
            return "已开启【自动曝光 Auto】" if new_state == 1 else "已关闭【手动曝光模式】"
        except Exception as e:
            log.warning(f"[Camera] 切换自动曝光失败: {e}")
            return None

    # ------------------------------ 内部辅助 ------------------------------

    def _warmup_realsense(self, warmup_frames: int = 5, timeout_ms: int = 2000):
        """预热抛弃前 N 帧, 让感光元件自动曝光稳定"""
        for _ in range(warmup_frames):
            try:
                self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
            except Exception:
                break

    def _probe_color_sensor(self):
        """获取物理彩色传感器句柄 (支持实时快捷调控硬件曝光与增益)"""
        try:
            prof = self.pipeline.get_active_profile()
            for s in prof.get_device().query_sensors():
                if s.is_color_sensor():
                    self.color_sensor = s
                    break
        except Exception as e:
            log.warning(f"[Camera] 彩色传感器探测失败: {e}")

    def _push_realsense_intrinsics(self, w, h):
        """RealSense: 结合当前硬件实时 Profile 与配置文件，并按实际分辨率自适应缩放"""
        stream_prof = None
        try:
            if self.pipeline:
                prof = self.pipeline.get_active_profile()
                if prof:
                    stream_prof = prof.get_stream(rs.stream.color).as_video_stream_profile()
        except Exception as e:
            log.debug(f"[Camera] 获取 active stream profile 异常 (可忽略): {e}")

        if resolve_camera_intrinsics is not None:
            K, dist, meta = resolve_camera_intrinsics(
                config_path=CONFIG_PATH,
                actual_image_shape=(h, w),
                stream_profile=stream_prof
            )
            log.info(f"[Camera] 相机内参: {meta.get('source')} | {w}x{h}")
        else:
            K = np.array([[1363.68, 0, 971.19], [0, 1361.19, 566.26], [0, 0, 1]], dtype=np.float64)
            dist = np.zeros((5, 1), dtype=np.float64)
        if self.intrinsics_callback is not None:
            self.intrinsics_callback(np.array(K, dtype=np.float64), np.array(dist, dtype=np.float64))

    def _push_usb_intrinsics(self, w, h):
        """USB: 近似针孔模型内参 (世界坐标解算精度受限)"""
        K = np.array([
            [0.8 * max(w, h), 0.0, w / 2.0],
            [0.0, 0.8 * max(w, h), h / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        if self.intrinsics_callback is not None:
            self.intrinsics_callback(K, np.zeros((5, 1), dtype=np.float64))
