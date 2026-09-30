#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多视角采图向导 (Capture Wizard)
===================================================
单一职责：多视角采集高质量图像 (纯预览 + 保存)。
工位与双用途体系：
  1. 顶部工具栏：
     - [工位: XXX ▼] 选择目标工作空间 (Workspace)
     - [用途: 标定/生产 ▼] 切换采集用途 (标定 calibration / 生产 production)
     - [相机类型 ▼] RealSense D435 / USB
     - [分辨率 ▼] 1920x1080 / 1280x720 等
     - [开启/关闭] 乒乓开关
     - [退出 X]
  2. 按 [空格] 键一键拍摄保存高清原始帧 (view_XXXX.png) 至对应用途的 raw_images 目录；
  3. 提供拍照快门白闪视觉反馈与该用途下的实时采样计数；
  4. 曝光调节 [ / ] 与自动曝光切换 [E] (RealSense 物理感光控制)。
"""

import os
import sys
import json
import glob
import time
import argparse
import numpy as np
import cv2

# Windows 终端中文色彩
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        pass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.workspace.workspace_manager import WorkspaceManager, Workspace
from src.devices.camera_service import CameraService
from src.ui.text_rendering import draw_text
from src.utils.logger import get_logger
from src.ui.base_cv_app import BaseCvApp
from tools.capture.renderer import (
    CaptureRenderer, COLOR_ACCENT, COLOR_TEXT_SUB, COL_YELLOW)

GUI_SETTINGS_FILE = os.path.join(PROJECT_ROOT, "config", "gui_settings.json")
APP_ID = "capture_wizard"

try:
    from tools.window_helper import force_window_focus
except ImportError:
    force_window_focus = None

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

log = get_logger(__name__)


class CaptureWizard(BaseCvApp):
    def __init__(self, output_dir: str = None, workspace_id: str = None, purpose: str = "calibration"):
        super().__init__(
            app_id=APP_ID,
            base_w=1280,
            base_h=720,
            window_name="capture_wizard",
            window_title="图像采集 (工作空间与双用途) | flux_vision_3d",
            responsive=True,
        )
        self.ws_mgr = WorkspaceManager()
        self.workspaces = self.ws_mgr.list_workspaces()

        # 采集用途: calibration (标定) | production (生产)
        self.purpose = purpose if purpose in ("calibration", "production") else "calibration"
        self.purpose_options = [
            ("calibration", "标定 (Calib)"),
            ("production",  "生产 (Prod)"),
        ]

        # 确定初始归档工位
        ws = None
        if workspace_id:
            ws = self.ws_mgr.get_workspace_by_id(workspace_id)
        if not ws:
            ws = self.ws_mgr.get_current_workspace()

        self.current_workspace = ws
        self.current_workspace_id = ws.workspace_id if ws else ""

        # 自定义 output_dir 或工位自动路由
        self._custom_output_dir = output_dir
        self.output_dir = ""
        self.image_count = 0
        self._update_output_dir()

        # 硬件与运行时状态
        self.is_running = False
        self.flash_timer = 0.0
        self.actual_stream_desc = "相机未开启"
        self._cam_srv = CameraService()
        self._frame_idx = 0

        # GUI 状态
        self.renderer = CaptureRenderer(self)
        self.active_dropdown = None        # None / WS_DROPDOWN / PURPOSE_DROPDOWN / CAMERA_TYPE_DROPDOWN / RES_DROPDOWN
        self.pipeline_running = False
        self.camera_type = "realsense"
        self.camera_options = [
            ("realsense", "RealSense D435"),
            ("usb",       "USB 普通摄像头"),
        ]
        self.resolution = "1920x1080"
        self.resolution_options = [
            ("1920x1080", "1920 × 1080  (推荐)"),
            ("1280x720",  "1280 × 720"),
            ("848x480",   "848 × 480"),
            ("640x480",   "640 × 480"),
        ]
        _w, _h = self.resolution.split("x")
        self.frame_w, self.frame_h = int(_w), int(_h)
        self._load_viewer_state()

        # 运行时状态
        self.status_toast = ""
        self.status_toast_time = 0.0
        self.color_sensor = None
        self.last_raw_frame = None

    def _update_output_dir(self):
        """根据当前工位与用途，重新计算并确保采图存储路径"""
        if self._custom_output_dir:
            self.output_dir = self._custom_output_dir
        elif self.current_workspace:
            self.output_dir = self.current_workspace.get_raw_images_dir(self.purpose)
        else:
            self.output_dir = os.path.join(PROJECT_ROOT, "data", "workspaces", "default", self.purpose, "raw_images")

        os.makedirs(self.output_dir, exist_ok=True)
        existing = glob.glob(os.path.join(self.output_dir, "view_*.png"))
        self.image_count = len(existing)

    @property
    def workspace_options(self):
        """动态读取所有可用工位供下拉菜单展示"""
        self.workspaces = self.ws_mgr.list_workspaces()
        opts = []
        for w in self.workspaces:
            cnt = w.image_count if self.purpose == "calibration" else w.prod_image_count
            opts.append((w.workspace_id, f"{w.name} ({cnt}帧)"))
        return opts

    @property
    def current_workspace_name(self):
        return self.current_workspace.name if self.current_workspace else "默认工位"

    @property
    def current_purpose_label(self):
        return dict(self.purpose_options).get(self.purpose, self.purpose)

    def switch_workspace(self, ws_id: str):
        """实时切换采图目标工位并持久化"""
        ws = self.ws_mgr.get_workspace_by_id(ws_id)
        if not ws:
            return
        self.current_workspace = ws
        self.current_workspace_id = ws.workspace_id
        self.ws_mgr.set_current_workspace(ws.workspace_id)
        self._update_output_dir()
        self.set_toast(f"已切换归档工位: 【{ws.name}】/【{self.current_purpose_label}】(当前 {self.image_count} 帧)")
        log.info(f"采图向导已切换归档工位: {ws.name} ({ws.workspace_id}) [{self.purpose}] -> {self.output_dir}")

    def switch_purpose(self, purpose_key: str):
        """实时切换采集用途 (标定 calibration / 生产 production)"""
        if purpose_key not in ("calibration", "production"):
            return
        if purpose_key == self.purpose:
            return
        self.purpose = purpose_key
        self._save_viewer_state()
        self._update_output_dir()
        self.set_toast(f"已切换采集用途: 【{self.current_purpose_label}】(当前 {self.image_count} 帧)")
        log.info(f"采图向导已切换采集用途: {self.purpose} -> {self.output_dir}")

    def set_toast(self, msg: str):
        self.status_toast = msg
        self.status_toast_time = time.time()

    # ------------------------------ 状态持久化 ------------------------------
    def _load_viewer_state(self):
        """从 config/gui_settings.json 恢复下拉选择"""
        try:
            if not os.path.exists(GUI_SETTINGS_FILE):
                return
            with open(GUI_SETTINGS_FILE, "r", encoding="utf-8") as f:
                root = json.load(f)
            state = (root.get(APP_ID) or {}).get("viewer_state") or {}
            if state.get("camera_type") in ("realsense", "usb"):
                self.camera_type = state["camera_type"]
            if any(k == state.get("resolution") for k, _ in self.resolution_options):
                self.resolution = state["resolution"]
                _w, _h = self.resolution.split("x")
                self.frame_w, self.frame_h = int(_w), int(_h)
            if state.get("purpose") in ("calibration", "production"):
                self.purpose = state["purpose"]
                self._update_output_dir()
        except Exception as e:
            log.warning(f"恢复采图向导状态失败，使用默认配置: {e}")

    def _save_viewer_state(self):
        """保存下拉选择到 config/gui_settings.json"""
        try:
            root = {}
            if os.path.exists(GUI_SETTINGS_FILE):
                try:
                    with open(GUI_SETTINGS_FILE, "r", encoding="utf-8") as f:
                        root = json.load(f)
                    if not isinstance(root, dict):
                        root = {}
                except Exception:
                    root = {}
            node = root.setdefault(APP_ID, {})
            node["viewer_state"] = {
                "camera_type": self.camera_type,
                "resolution": self.resolution,
                "purpose": self.purpose,
            }
            node["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            os.makedirs(os.path.dirname(GUI_SETTINGS_FILE), exist_ok=True)
            with open(GUI_SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(root, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.warning(f"保存采图向导状态失败: {e}")

    def _select_camera_type(self, cam_key):
        if cam_key == self.camera_type:
            return
        if self.pipeline_running:
            self._stop_camera()
        self.camera_type = cam_key
        self._save_viewer_state()
        log.info(f"相机类型已切换为: {dict(self.camera_options).get(cam_key, cam_key)}")

    def _change_resolution(self, res_key):
        if res_key == self.resolution:
            return
        was_running = self.pipeline_running
        if was_running:
            self._stop_camera()
        self.resolution = res_key
        _w, _h = res_key.split("x")
        self.frame_w, self.frame_h = int(_w), int(_h)
        self._save_viewer_state()
        if was_running:
            self._start_camera()
        log.info(f"分辨率已切换: {res_key}")

    def _toggle_camera(self):
        if self.pipeline_running:
            self.set_toast("正在关闭相机...")
            self._stop_camera()
            self.set_toast("相机已关闭")
            log.info("相机已关闭")
        else:
            self.set_toast("正在开启相机...")
            self._start_camera()

    def _start_camera(self):
        w, h = self.frame_w, self.frame_h
        try:
            if self.camera_type == "realsense":
                self._cam_srv.start_realsense(
                    w, h, fps=8 if w > 1280 else 15,
                    fallbacks=((w, h, 8),),
                    mock_fallback=False)
            else:
                self._cam_srv.start_usb(w, h)
        except Exception as e:
            log.warning(f"相机开启失败: {e}")
            self.set_toast(f"相机开启失败: {e}")
            return
        self.pipeline_running = True
        self.color_sensor = self._cam_srv.color_sensor
        self.actual_stream_desc = self._cam_srv.stream_desc
        cam_desc = dict(self.camera_options).get(self.camera_type, self.camera_type)
        self.set_toast(f"相机已开启: {cam_desc} @ {self.resolution}")
        log.info(f"[OK] 相机已开启: {self.camera_type} @ {self.resolution} ({self.actual_stream_desc})")

    def _stop_camera(self):
        self._cam_srv.stop()
        self.pipeline_running = False
        self.color_sensor = None

    def get_frame(self, frame_idx: int):
        if self.pipeline_running:
            return self._cam_srv.read_frame(timeout_ms=2500)
        return None

    def save_image(self, raw_frame: np.ndarray) -> str:
        """保存采图快照：无标注原始帧存至当前用途对应目录 view_XXXX.png"""
        self.image_count += 1
        raw_filename = f"view_{self.image_count:04d}.png"
        raw_filepath = os.path.join(self.output_dir, raw_filename)
        cv2.imwrite(raw_filepath, raw_frame)
        log.info(f"[CAPTURE] [{self.purpose}] 快照 #{self.image_count} 拍摄成功: {raw_filepath}")

        # 同步更新工位元数据
        try:
            if self.current_workspace:
                self.current_workspace.refresh_stats()
                self.current_workspace.save_meta()
        except Exception as e:
            log.warning(f"工位元数据刷新失败 (非致命): {e}")

        self.flash_timer = time.time()
        return raw_filepath

    def adjust_hardware_exposure(self, delta_us: float):
        if self.color_sensor is None:
            self.set_toast("当前未检测到 RealSense 物理彩色传感器")
            return
        try:
            if self.color_sensor.supports(rs.option.enable_auto_exposure):
                is_auto = self.color_sensor.get_option(rs.option.enable_auto_exposure)
                if is_auto > 0.5:
                    self.color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            
            if self.color_sensor.supports(rs.option.exposure):
                cur_exp = self.color_sensor.get_option(rs.option.exposure)
                new_exp = max(10.0, min(1000.0, cur_exp + delta_us))
                self.color_sensor.set_option(rs.option.exposure, new_exp)
                self.set_toast(f"硬件手动曝光: {int(new_exp)} (按 [ 压暗 / ] 提亮)")
        except Exception as e:
            self.set_toast(f"调曝光失败: {e}")

    def toggle_auto_exposure(self):
        if self.color_sensor is None:
            self.set_toast("当前非物理相机")
            return
        try:
            if self.color_sensor.supports(rs.option.enable_auto_exposure):
                cur = self.color_sensor.get_option(rs.option.enable_auto_exposure)
                new_state = 0 if cur > 0.5 else 1
                self.color_sensor.set_option(rs.option.enable_auto_exposure, new_state)
                desc = "已开启【自动曝光 Auto】" if new_state == 1 else "已关闭【手动曝光模式】"
                self.set_toast(desc)
        except Exception as e:
            self.set_toast(f"切换自动曝光失败: {e}")

    # ==================== BaseCvApp 钩子实现 ====================
    def set_toast(self, msg: str, duration: float = 3.5):
        """同步设置 BaseCvApp 与 CaptureRenderer 的 Toast"""
        super().set_toast(msg, duration)
        self.status_toast = msg
        self.status_toast_time = time.time()

    def setup(self):
        log.info(f"图像采集已启动，工作空间: {self.current_workspace_name}，用途: {self.purpose}，存储目录: {self.output_dir}")

    def on_mouse_move(self, x: int, y: int):
        self.renderer.on_mouse_move(x, y)

    def _handle_action(self, btn_id, payload):
        if btn_id == "TOGGLE_WS_DD":
            self.active_dropdown = None if self.active_dropdown == "WS_DROPDOWN" else "WS_DROPDOWN"
        elif btn_id.startswith("DD_WS_"):
            self.active_dropdown = None
            self.switch_workspace(payload)
        elif btn_id == "TOGGLE_PURPOSE_DD":
            self.active_dropdown = None if self.active_dropdown == "PURPOSE_DROPDOWN" else "PURPOSE_DROPDOWN"
        elif btn_id.startswith("DD_PURPOSE_"):
            self.active_dropdown = None
            self.switch_purpose(payload)
        elif btn_id == "TOGGLE_CAM_DD":
            self.active_dropdown = None if self.active_dropdown == "CAMERA_TYPE_DROPDOWN" else "CAMERA_TYPE_DROPDOWN"
        elif btn_id == "TOGGLE_RES_DD":
            self.active_dropdown = None if self.active_dropdown == "RES_DROPDOWN" else "RES_DROPDOWN"
        elif btn_id.startswith("DD_CAM_"):
            self.active_dropdown = None
            self._select_camera_type(payload)
        elif btn_id.startswith("DD_RES_"):
            self.active_dropdown = None
            self._change_resolution(payload)
        elif btn_id == "TOGGLE_CAMERA":
            self.active_dropdown = None
            self._toggle_camera()
        elif btn_id == "CAPTURE":
            self.active_dropdown = None
            if self.pipeline_running and self.last_raw_frame is not None:
                self.save_image(self.last_raw_frame)
            elif not self.pipeline_running:
                self.set_toast("相机未开启，请先点击 [开启] 取流！")
            else:
                self.set_toast("正在等待有效画面帧...")
        elif btn_id == "QUIT":
            self.stop()

    def on_click(self, x: int, y: int):
        hit = self.renderer.hit_test(x, y)
        if hit is not None:
            self._handle_action(*hit)
            return
        if self.active_dropdown is not None:
            self.active_dropdown = None

    def on_key(self, key: int) -> bool:
        k = chr(key & 0xFF).lower() if (key & 0xFF) < 128 else ""
        if k in ("q", "x"):
            self._running = False
            return True
        elif k == " ":
            if self.last_raw_frame is not None:
                self.save_image(self.last_raw_frame)
            return True
        elif k == "[":
            self.adjust_hardware_exposure(-50.0)
            return True
        elif k == "]":
            self.adjust_hardware_exposure(50.0)
            return True
        elif k == "e":
            self.toggle_auto_exposure()
            return True
        return False

    def render(self) -> np.ndarray:
        raw_frame = None
        if self.pipeline_running:
            raw_frame = self.get_frame(self._frame_idx)
            self._frame_idx += 1
            if raw_frame is not None:
                self.last_raw_frame = raw_frame
        else:
            self.last_raw_frame = None

        if raw_frame is not None:
            if time.time() - self.flash_timer < 0.12:
                disp = cv2.addWeighted(raw_frame, 0.4, np.full_like(raw_frame, 255), 0.6, 0)
            else:
                disp = raw_frame
            canvas = self.renderer.compose_canvas(disp)
        elif self.pipeline_running:
            canvas = self.renderer.make_canvas()
            cw, ch = self.base_w, self.base_h
            draw_text(canvas, "取流中...", (cw // 2 - 60, ch // 2), 22, COL_YELLOW, True)
        else:
            canvas = self.renderer.make_canvas()
            cw, ch = self.base_w, self.base_h
            draw_text(canvas, "相机未开启",
                      (cw // 2 - 120, ch // 2 - 50), 32, COLOR_ACCENT, True)
            draw_text(canvas, f"归档: 【{self.current_workspace_name}】/【{self.current_purpose_label}】",
                      (cw // 2 - 150, ch // 2 + 10), 20, COL_YELLOW)
            draw_text(canvas, "点击 [开启] 预览画面，按 [空格] 或点击 [拍照] 保存无标注原始帧",
                      (cw // 2 - 270, ch // 2 + 50), 16, COLOR_TEXT_SUB)

        self.renderer.draw_toolbar(canvas)

        # 底部状态栏: 目录路径与照片计数
        ch, cw = canvas.shape[:2]
        bar_y = ch - 28
        cv2.rectangle(canvas, (0, bar_y), (cw, ch), (16, 18, 22), -1)
        cv2.line(canvas, (0, bar_y), (cw, bar_y), (48, 56, 70), 1)
        status = f"目录: {self.output_dir}  |  已采集 {self.image_count} 张"
        draw_text(canvas, status, (12, bar_y + 6), 12, COLOR_TEXT_SUB)

        self.renderer.draw_toast(canvas)
        return canvas

    def cleanup(self):
        self._stop_camera()
        log.info(f"[OK] 采图向导已退出 (当前工位【{self.purpose}】共计 {self.image_count} 帧)")


def main():
    parser = argparse.ArgumentParser(description="多视角采图向导 (纯预览 + 保存)")
    parser.add_argument("--workspace", dest="workspace", type=str, default="", help="指定初始归档工位 ID")
    parser.add_argument("--purpose", type=str, default="calibration", choices=["calibration", "production"], help="采集用途: calibration 标定 / production 生产")
    parser.add_argument("--dir", "--output-dir", "--output_dir", dest="dir", type=str, default=None, help="自定义保存目录路径")
    args = parser.parse_args()

    wizard = CaptureWizard(output_dir=args.dir, workspace_id=args.workspace, purpose=args.purpose)
    wizard.run()


if __name__ == "__main__":
    main()
