#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一相机取流服务 (CameraService)
================================
收编重复的相机硬件管理 (tools/tracker/camera_controller、src/hardware/camera_streamer、
tools/capture/capture_wizard)：
  - RealSense D435 / USB 摄像头 / Mock 仿真 三后端统一启停与帧读取；
  - 分级回退链 (帧率/分辨率逐级降级) 与 Mock 优雅降级；
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
    """统一相机取流服务: 硬件启停 / 帧读取 / 曝光控制 / Mock 仿真帧"""

    def __init__(self, intrinsics_callback=None):
        """
        :param intrinsics_callback: callable(K, dist) — 后端启动成功后推送内参
                                    (RealSense 为标定内参, USB 为近似针孔模型)
        """
        self.intrinsics_callback = intrinsics_callback

        self.pipeline = None          # rs.pipeline (RealSense 后端)
        self.usb_capture = None       # cv2.VideoCapture (USB 后端)
        self.is_running = False
        self.is_mock = False
        self.stream_desc = "未初始化"
        self.color_sensor = None      # rs.sensor 彩色传感器句柄 (曝光调控)
        self.last_valid_frame = None  # 最近一帧有效图像 (瞬时失败回退用)
        self._mock_frame_idx = 0

    # ------------------------------ 启动 ------------------------------
    def start_realsense(self, width, height, fps=30, fallbacks=(), mock_fallback=True) -> bool:
        """
        启动 RealSense 彩色流。
        :param fallbacks: ((w, h, fps), ...) 逐级降级链, 主档失败后依次尝试
        :param mock_fallback: 全链失败时 True=优雅切 Mock 返回 True, False=抛 RuntimeError
        """
        if not HAVE_REALSENSE:
            return self._enter_mock_or_raise("pyrealsense2 未安装, 请先安装 RealSense SDK", mock_fallback)
        try:
            ctx = rs.context()
            devices = list(ctx.query_devices())
            if not devices:
                return self._enter_mock_or_raise("未检测到 RealSense 设备, 请检查 USB 连接", mock_fallback)

            last_err = None
            cur = (width, height, fps)
            for w, h, f in ((width, height, fps),) + tuple(fallbacks):
                try:
                    pipeline = rs.pipeline()
                    cfg = rs.config()
                    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)
                    pipeline.start(cfg)
                    self.pipeline = pipeline
                    cur = (w, h, f)
                    self.stream_desc = f"{w}x{h} @ {f}fps"
                    log.info(f"[Camera] RealSense 彩色流: {w}x{h} @ {f}fps")
                    break
                except Exception as e:
                    last_err = e
                    log.info(f"[Camera] {w}x{h} @ {f}fps 请求未满足: {e}")
            else:
                return self._enter_mock_or_raise(f"RealSense {width}x{height} 启动失败: {last_err}", mock_fallback)

            self._warmup_realsense()
            self._probe_color_sensor()
            self.is_mock = False
            self.is_running = True
            self._push_realsense_intrinsics(cur[0], cur[1])
            return True
        except RuntimeError:
            raise
        except Exception as e:
            return self._enter_mock_or_raise(f"连接物理相机失败: {e}", mock_fallback)

    def start_usb(self, width, height, fps=30) -> bool:
        """启动普通 USB 摄像头 (cv2.VideoCapture)，使用近似针孔内参 (未标定)"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("无法打开 USB 摄像头 (index=0)")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        self.usb_capture = cap
        self.is_mock = False
        self.is_running = True
        self.stream_desc = f"USB {width}x{height}"
        self._push_usb_intrinsics(width, height)
        log.warning(f"[Camera] USB 摄像头已开启 {width}x{height} (未标定内参, 世界坐标仅供流程验证)")
        return True

    def enter_mock(self) -> bool:
        """切入 Mock 仿真流"""
        self.is_mock = True
        self.is_running = True
        self.stream_desc = "仿真模拟相机 (Mock)"
        log.warning("[Camera] 已切入 Mock 仿真模式")
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
        self.is_mock = False

    # ------------------------------ 帧读取 ------------------------------
    def read_frame(self, timeout_ms: int = 1000):
        """读取一帧 BGR 图像; Mock 模式生成仿真帧; 瞬时失败回退最近有效帧; 未运行返回 None"""
        if not self.is_running:
            return None
        if self.is_mock:
            self._mock_frame_idx += 1
            frame = self.make_mock_frame(self._mock_frame_idx)
            self.last_valid_frame = frame
            return frame
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

    # ------------------------------ Mock 仿真帧 ------------------------------
    def make_mock_frame(self, frame_idx: int) -> np.ndarray:
        """生成带动态运动 AprilTag 的高保真仿真视频帧 (720p)"""
        w, h = 1280, 720
        frame = np.full((h, w, 3), 32, dtype=np.uint8)
        for x in range(0, w, 80):
            cv2.line(frame, (x, 0), (x, h), (44, 44, 44), 1)
        for y in range(0, h, 80):
            cv2.line(frame, (0, y), (w, y), (44, 44, 44), 1)

        dx = int(12 * math.sin(frame_idx * 0.08))
        dy = int(8 * math.cos(frame_idx * 0.06))
        tag_configs = [
            (0, 320 + dx, 220 + dy, 110),
            (1, 640 + dx, 200 - dy, 110),
            (2, 960 - dx, 220 + dy, 110),
            (18, 400 - dx, 480 + dy, 100),
            (28, 880 + dx, 480 - dy, 100),
            (4, 640 + dx, 500 + dy, 90),
        ]
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
        for tag_id, cx, cy, sz in tag_configs:
            hs = sz // 2
            x1, y1 = max(0, cx - hs), max(0, cy - hs)
            x2, y2 = min(w, cx + hs), min(h, cy + hs)
            tag_img = cv2.aruco.generateImageMarker(dictionary, tag_id, sz)
            tag_bgr = cv2.cvtColor(tag_img, cv2.COLOR_GRAY2BGR)
            h_sub, w_sub = y2 - y1, x2 - x1
            if h_sub > 0 and w_sub > 0:
                frame[y1:y2, x1:x2] = tag_bgr[:h_sub, :w_sub]
        return frame

    # ------------------------------ 内部辅助 ------------------------------
    def _enter_mock_or_raise(self, msg: str, mock_fallback: bool) -> bool:
        if mock_fallback:
            log.warning(f"[Camera] {msg}, 切至 Mock 仿真模式")
            return self.enter_mock()
        raise RuntimeError(msg)

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
        """RealSense: 使用 config.yaml 标定内参 (与建图一致) 并按实际分辨率自适应缩放"""
        if resolve_camera_intrinsics is not None:
            K, dist, meta = resolve_camera_intrinsics(CONFIG_PATH, actual_image_shape=(h, w))
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
