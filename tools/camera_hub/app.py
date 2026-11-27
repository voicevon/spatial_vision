# -*- coding: utf-8 -*-
"""
Camera Hub 主控制器 (CameraHubApp)
==================================
全局多相机资产与棋盘格内参标定工作台：
- 继承 BaseCvApp 统一窗口生命周期与矢量缩放
- 硬件操作完全委托 CameraService (支持 RealSense + USB)
- 渲染完全委托只读 CameraHubRenderer
"""

import argparse
import os
import sys
import time
from typing import Optional
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.calibration.solvers.camera_intrinsics_calibrator import (
    calibrate_camera_profile,
    detect_chessboard_corners,
)
from src.cameras.camera_entity import parse_resolution
from src.cameras.camera_registry import CameraRegistry
from src.devices.camera_service import CameraService
from src.ui.base_cv_app import BaseCvApp
from src.ui.dialog_utils import show_confirm_dialog, show_text_input_dialog
from src.utils.image_io import imwrite_unicode
from src.utils.logger import get_logger
from tools.camera_hub.renderer import CameraHubRenderer
from tools.camera_hub.state import CameraHubState

log = get_logger(__name__)

GUI_SETTINGS_FILE = os.path.join(PROJECT_ROOT, "config", "gui_settings.json")
APP_ID = "camera_hub"


class CameraHubApp(BaseCvApp):
    """Camera Hub 交互式主应用"""

    def __init__(
        self,
        cameras_root: Optional[str] = None,
        camera_id: Optional[str] = None,
        settings_file: Optional[str] = None,
    ):
        super().__init__(
            app_id=APP_ID,
            base_w=1280,
            base_h=760,
            window_name="camera_hub",
            window_title="Camera Hub | 多相机资产与内参标定中枢",
            settings_file=settings_file or GUI_SETTINGS_FILE,
            responsive=True,
        )
        self.registry = CameraRegistry(cameras_root=cameras_root)
        self.state = CameraHubState(registry=self.registry)
        if camera_id:
            self.state.select_camera(camera_id)
        self.renderer = CameraHubRenderer(self.state)
        self._cam_srv = CameraService()
        self._last_detect_ts: float = 0.0
        self._detect_interval_s: float = 0.22  # ~4.5 Hz 节流，保持预览流畅

    # ==================== 业务动作 ====================
    def probe_local_cameras(self) -> int:
        """探测本机直连 RealSense 与 USB 相机并自动登记至全局相机库"""
        devs = CameraService.probe_available_devices()
        online_ids: set[str] = set()
        new_count = 0

        for rs_dev in devs.get("realsense", []):
            serial = str(rs_dev.get("serial") or "").strip()
            name = str(rs_dev.get("name") or "RealSense D435")
            existing = self.registry.find_by_hardware("realsense", hardware_serial=serial) if serial else None
            if existing is None:
                prof = self.registry.register_camera(
                    name=f"{name} ({serial[-4:]})" if serial else name,
                    camera_type="realsense",
                    hardware_serial=serial,
                )
                new_count += 1
            else:
                prof = existing
            online_ids.add(prof.camera_id)

        for usb_dev in devs.get("usb", []):
            idx = int(usb_dev.get("index", 0))
            name = str(usb_dev.get("name") or f"USB Camera #{idx}")
            existing = self.registry.find_by_hardware("usb", usb_index=idx)
            if existing is None:
                prof = self.registry.register_camera(
                    name=name,
                    camera_type="usb",
                    usb_index=idx,
                )
                new_count += 1
            else:
                prof = existing
            online_ids.add(prof.camera_id)

        self.state.online_camera_ids = online_ids
        self.state.refresh_cameras()
        self.set_toast(f"本机探测完成: 在线 {len(online_ids)} 台，新登记 {new_count} 台相机")
        return len(online_ids)

    def _action_add_manual_usb(self):
        name = show_text_input_dialog("手动登记相机", "请输入相机别名 (例如: 工位2号USB相机):", initial="USB 相机")
        if not name or not name.strip():
            return
        prof = self.registry.register_camera(name=name.strip(), camera_type="usb", usb_index=0)
        self.state.refresh_cameras(select_id=prof.camera_id)
        self.set_toast(f"已手动登记相机: {prof.name} ({prof.camera_id})")

    def _action_rename_camera(self):
        cam = self.state.get_selected_camera()
        if cam is None:
            return
        new_name = show_text_input_dialog("重命名相机", f"为 {cam.camera_id} 输入新别名:", initial=cam.name)
        if not new_name or not new_name.strip():
            return
        if self.registry.rename_camera(cam.camera_id, new_name.strip()):
            self.state.refresh_cameras(select_id=cam.camera_id)
            self.set_toast(f"相机已重命名为: {new_name.strip()}")

    def _action_delete_camera(self):
        cam = self.state.get_selected_camera()
        if cam is None:
            return
        if not show_confirm_dialog("删除相机档案", f"确定永久删除相机 [{cam.name}] ({cam.camera_id}) 及其全部内参原图？"):
            return
        self._stop_camera()
        if self.registry.delete_camera(cam.camera_id):
            self.state.refresh_cameras()
            self.set_toast(f"已删除相机档案: {cam.camera_id}")

    def _toggle_camera(self):
        if self.state.pipeline_running:
            self._stop_camera()
            self.set_toast("相机预览已关闭")
        else:
            self._start_camera()

    def _start_camera(self):
        cam = self.state.get_selected_camera()
        if cam is None:
            self.set_toast("请先选择或登记一台相机")
            return
        parsed = parse_resolution(self.state.selected_resolution)
        w, h = parsed if parsed else (1920, 1080)
        try:
            if cam.camera_type == "realsense":
                self._cam_srv.start_realsense(
                    w, h, fps=8 if w > 1280 else 15,
                    serial=cam.hardware_serial or None,
                    fallbacks=((w, h, 8),),
                )
            else:
                dev_idx = cam.usb_index if cam.usb_index is not None else 0
                self._cam_srv.start_usb(w, h, device_index=dev_idx)
        except Exception as e:
            self.set_toast(f"开启相机失败: {e}")
            log.warning(f"[CameraHub] 开启相机失败 ({cam.camera_id}): {e}")
            return

        self.state.pipeline_running = True
        self.state.stream_desc = self._cam_srv.stream_desc
        self.set_toast(f"相机已开启: {cam.name} @ {self.state.selected_resolution}")

    def _stop_camera(self):
        self._cam_srv.stop()
        self.state.pipeline_running = False
        self.state.live_frame = None
        self.state.live_corners_found = False
        self.state.live_corners = None
        self.state.stream_desc = "相机未开启"

    def capture_current_frame(self) -> str:
        cam = self.state.get_selected_camera()
        if cam is None:
            self.set_toast("请先选择相机")
            return ""
        if not self.state.pipeline_running or self.state.live_frame is None:
            self.set_toast("请先开启相机预览再抓拍")
            return ""

        out_path = cam.next_image_path(self.state.selected_resolution)
        if not imwrite_unicode(out_path, self.state.live_frame):
            self.set_toast("保存棋盘格原图失败")
            return ""
        self.state.flash_timer = time.time() + 0.12
        self.state.refresh_profile_data()
        self.set_toast(f"已抓拍: {os.path.basename(out_path)} (累计 {len(self.state.images)} 帧)")
        return out_path

    def run_calibration(self):
        cam = self.state.get_selected_camera()
        if cam is None:
            self.set_toast("请先选择相机")
            return
        res = calibrate_camera_profile(
            camera=cam,
            resolution=self.state.selected_resolution,
            pattern_size=self.state.pattern_size,
            square_size_mm=self.state.square_size_mm,
            save_to_profile=True,
        )
        self.state.refresh_profile_data()
        if res.success:
            self.state.active_mode = CameraHubState.MODE_GALLERY
            self.set_toast(f"标定成功! RMSE={res.rmse:.3f}px ({res.valid_images_count}/{res.total_images_count} 帧)")
        else:
            self.set_toast(f"标定失败: {res.message}")
        return res

    def delete_selected_image(self) -> bool:
        cam = self.state.get_selected_camera()
        if cam is None or not self.state.images:
            return False
        idx = self.state.selected_image_idx
        if not (0 <= idx < len(self.state.images)):
            return False
        target = self.state.images[idx]
        if cam.delete_image(self.state.selected_resolution, target):
            self.state.invalidate_thumb(target)
            self.state.refresh_profile_data()
            self.set_toast(f"已删除照片: {os.path.basename(target)}")
            return True
        return False

    def prune_bad_images(self, max_error_px: float = 1.0) -> int:
        """一键剔除未检出棋盘格或重投影误差高于阈值的废片"""
        cam = self.state.get_selected_camera()
        if cam is None or not self.state.per_image_detected:
            self.set_toast("请先执行一次棋盘格标定以获取逐帧误差")
            return 0
        removed = 0
        for img_path in list(self.state.images):
            fn = os.path.basename(img_path)
            if fn not in self.state.per_image_detected:
                continue
            detected = self.state.per_image_detected[fn]
            err = self.state.per_image_errors.get(fn)
            if (not detected) or (err is not None and err > max_error_px):
                if cam.delete_image(self.state.selected_resolution, img_path):
                    self.state.invalidate_thumb(img_path)
                    removed += 1
        self.state.refresh_profile_data()
        self.set_toast(f"已剔除 {removed} 张无效/高误差帧 (> {max_error_px:.1f}px)" if removed else "当前所有帧误差均达标，无需剔除")
        return removed

    # ==================== BaseCvApp 生命周期 ====================
    def render(self):
        if self.state.pipeline_running:
            frame = self._cam_srv.read_frame(timeout_ms=500)
            if frame is not None:
                self.state.live_frame = frame
                now = time.time()
                if now - self._last_detect_ts >= self._detect_interval_s:
                    self._last_detect_ts = now
                    found, corners = detect_chessboard_corners(frame, self.state.pattern_size)
                    self.state.live_corners_found = found
                    self.state.live_corners = corners

        cw = self.win_mgr.canvas_w
        ch = self.win_mgr.canvas_h
        toast = self._toast if time.time() < self._toast_until else ""
        return self.renderer.render(cw, ch, (self.mouse_x, self.mouse_y), toast_msg=toast)

    def on_click(self, x: int, y: int):
        for btn_id, (x1, y1, x2, y2), payload in reversed(self.renderer.buttons):
            if x1 <= x <= x2 and y1 <= y <= y2:
                self._dispatch_button(btn_id, payload)
                return

    def _dispatch_button(self, btn_id: str, payload):
        if btn_id == "btn_exit":
            self.stop()
        elif btn_id == "mode_live":
            self.state.active_mode = CameraHubState.MODE_LIVE
        elif btn_id == "mode_gallery":
            self.state.active_mode = CameraHubState.MODE_GALLERY
            self.state.refresh_profile_data()
        elif btn_id == "btn_probe":
            self.probe_local_cameras()
        elif btn_id == "btn_add_usb":
            self._action_add_manual_usb()
        elif btn_id == "select_cam":
            if self.state.pipeline_running:
                self._stop_camera()
            self.state.select_camera(str(payload))
        elif btn_id == "select_res":
            if self.state.pipeline_running:
                self._stop_camera()
            self.state.select_resolution(str(payload))
        elif btn_id == "btn_rename_cam":
            self._action_rename_camera()
        elif btn_id == "btn_delete_cam":
            self._action_delete_camera()
        elif btn_id == "pat_col_dec":
            self.state.adjust_pattern(d_cols=-1)
        elif btn_id == "pat_col_inc":
            self.state.adjust_pattern(d_cols=1)
        elif btn_id == "pat_row_dec":
            self.state.adjust_pattern(d_rows=-1)
        elif btn_id == "pat_row_inc":
            self.state.adjust_pattern(d_rows=1)
        elif btn_id == "pat_sq_dec":
            self.state.adjust_pattern(d_square_mm=-1.0)
        elif btn_id == "pat_sq_inc":
            self.state.adjust_pattern(d_square_mm=1.0)
        elif btn_id == "btn_toggle_cam":
            self._toggle_camera()
        elif btn_id == "btn_capture":
            self.capture_current_frame()
        elif btn_id == "btn_solve":
            self.run_calibration()
        elif btn_id == "btn_prune_bad":
            self.prune_bad_images()
        elif btn_id == "btn_del_img":
            self.delete_selected_image()
        elif btn_id == "select_img":
            self.state.selected_image_idx = int(payload)

    def on_key(self, key: int):
        if key in (27, ord("q"), ord("Q")):
            self.stop()
        elif key == 32 and self.state.active_mode == CameraHubState.MODE_LIVE:
            self.capture_current_frame()
        elif key in (ord("b"), ord("B")):
            self.run_calibration()
        elif key in (127, 8, 3014656):  # Del / Backspace
            if self.state.active_mode == CameraHubState.MODE_GALLERY:
                self.delete_selected_image()

    def on_mouse_wheel(self, delta: int, flags: int):
        if self.state.active_mode == CameraHubState.MODE_GALLERY:
            self.state.scroll_grid(-1 if delta > 0 else 1)

    def cleanup(self):
        self._stop_camera()


def main():
    parser = argparse.ArgumentParser(description="Camera Hub | 多相机资产与棋盘格内参标定中枢")
    parser.add_argument("--camera", dest="camera", type=str, default="", help="指定初始选中的 camera_id")
    args = parser.parse_args()
    app = CameraHubApp(camera_id=args.camera or None)
    app.run()


if __name__ == "__main__":
    main()
