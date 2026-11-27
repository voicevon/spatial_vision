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
import threading
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
from src.calibration.solvers.chessboard_guide import (
    PoseAlignmentFeedback,
    evaluate_alignment,
    extract_detected_quad_norm,
    project_guide_mesh,
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
        self._detect_interval_s: float = 0.10  # ~10 Hz 保证自动抓拍进度环顺滑且防抖灵敏
        self.sound_enabled: bool = True

    def _play_beep_async(self, double_beep: bool = False):
        """异步播放 Windows 原生提示音（不阻塞 OpenCV 主循环，非 Windows 静默）"""
        if not self.sound_enabled or sys.platform != "win32":
            return

        def _worker():
            try:
                import winsound
                if double_beep:
                    winsound.Beep(1000, 90)
                    winsound.Beep(1500, 140)
                else:
                    winsound.Beep(1200, 85)
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()

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
        self.state.last_feedback = None
        self.state.prev_detected_quad_norm = None
        self.state.reset_hold_timer()
        self.state.stream_desc = "相机未开启"

    def capture_current_frame(self, auto_triggered: bool = False) -> str:
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
        self.state.flash_timer = time.time() + 0.14
        self.state.session_captured_paths.append(out_path)
        self.state.capture_cooldown_until = time.time() + 0.45
        self.state.refresh_profile_data()

        pose = self.state.get_current_pose()
        finished_all = self.state.advance_pose()
        prefix = "自动抓拍" if auto_triggered else "手动抓拍"

        if finished_all:
            self._play_beep_async(double_beep=True)
            self._stop_camera()
            self.set_toast(f"全部 {len(self.state.guide_poses)} 个引导姿态采图完成，正在自动求解内参...")
            self.run_calibration()
        else:
            self._play_beep_async(double_beep=False)
            next_pose = self.state.get_current_pose()
            self.set_toast(
                f"{prefix}成功 ({pose.title}): {os.path.basename(out_path)} | 下一姿态: {next_pose.title}"
            )
        return out_path

    def undo_last_capture(self) -> bool:
        """撤销本次引导会话最近抓拍的一张照片并回退一个姿态"""
        cam = self.state.get_selected_camera()
        if cam is None or not self.state.session_captured_paths:
            self.set_toast("本次会话暂无可撤销的抓拍记录")
            return False
        last_path = self.state.session_captured_paths.pop()
        cam.delete_image(self.state.selected_resolution, last_path)
        self.state.invalidate_thumb(last_path)
        self.state.current_pose_idx = max(0, self.state.current_pose_idx - 1)
        self.state.reset_hold_timer()
        self.state.prev_detected_quad_norm = None
        self.state.refresh_profile_data()
        self.set_toast(f"已撤销抓拍: {os.path.basename(last_path)}，回退至步骤 {self.state.current_pose_idx + 1}")
        return True

    def process_live_frame_for_auto_capture(self, frame, now: Optional[float] = None) -> PoseAlignmentFeedback:
        """对当前预览帧执行棋盘格检测、3D 引导框对准评估与 0.7s 稳定自动抓拍驱动"""
        now_ts = time.time() if now is None else float(now)
        self.state.live_frame = frame
        fh, fw = frame.shape[:2]
        viewport_aspect = float(fw) / float(max(1, fh))

        pose = self.state.get_current_pose()
        mesh = project_guide_mesh(pose, self.state.pattern_size, viewport_aspect=viewport_aspect)

        found, corners = detect_chessboard_corners(frame, self.state.pattern_size)
        self.state.live_corners_found = found
        self.state.live_corners = corners

        if not found or corners is None:
            fb = PoseAlignmentFeedback(
                detected=False, aligned=False, stable=False, match_score=0.0,
                guidance_text=f"{pose.instruction}（请将棋盘格完整置于画面内）",
            )
            self.state.last_feedback = fb
            self.state.prev_detected_quad_norm = None
            self.state.reset_hold_timer()
            return fb

        det_quad, det_corners = extract_detected_quad_norm(corners, self.state.pattern_size, fw, fh)
        # 前置摄像头镜像预览模式：将检测角点水平镜像至预览坐标系以对齐 3D 引导与方向提示
        # （落盘原图仍保持 self.state.live_frame 未翻转的传感器真实帧）
        if self.state.is_mirror_preview and det_quad:
            det_quad = [(1.0 - x, y) for x, y in det_quad]
            det_corners = [(1.0 - x, y) for x, y in det_corners]

        fb = evaluate_alignment(
            detected_quad_norm=det_quad,
            target_quad_norm=mesh.outer_quad,
            detected_corners_norm=det_corners,
            prev_quad_norm=self.state.prev_detected_quad_norm,
        )
        self.state.last_feedback = fb
        self.state.prev_detected_quad_norm = det_quad

        # 自动抓拍稳定保持计时器
        if (
            self.state.auto_capture_enabled
            and fb.aligned
            and fb.stable
            and now_ts >= self.state.capture_cooldown_until
        ):
            if self.state.aligned_since_ts <= 0.0:
                self.state.aligned_since_ts = now_ts
            elapsed = now_ts - self.state.aligned_since_ts
            self.state.hold_progress = min(1.0, max(0.0, elapsed / max(0.05, self.state.hold_duration_s)))
            if elapsed >= self.state.hold_duration_s:
                self.capture_current_frame(auto_triggered=True)
        else:
            self.state.reset_hold_timer()

        return fb

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
                    self.process_live_frame_for_auto_capture(frame, now=now)

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
        elif btn_id == "btn_toggle_auto":
            self.state.auto_capture_enabled = not self.state.auto_capture_enabled
            self.state.reset_hold_timer()
            self.set_toast(f"自动抓拍已{'开启' if self.state.auto_capture_enabled else '关闭'}")
        elif btn_id == "btn_toggle_facing":
            new_facing = self.state.toggle_camera_facing()
            desc = "前置摄像头（左右镜像预览，保存真实原图）" if new_facing == "front" else "后置摄像头（正常预览）"
            self.set_toast(f"已切换为: {desc}")
        elif btn_id == "btn_skip_pose":
            self.state.skip_pose()
            p = self.state.get_current_pose()
            self.set_toast(f"已跳过，当前引导步骤 {self.state.current_pose_idx + 1}/{len(self.state.guide_poses)}: {p.title}")
        elif btn_id == "btn_undo_capture":
            self.undo_last_capture()
        elif btn_id == "btn_reset_guide":
            self.state.reset_guide_session()
            self.set_toast("引导姿态进度已重置为第 1 步")
        elif btn_id == "btn_capture":
            self.capture_current_frame(auto_triggered=False)
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
            self.capture_current_frame(auto_triggered=False)
        elif key in (ord("n"), ord("N")) and self.state.active_mode == CameraHubState.MODE_LIVE:
            self.state.skip_pose()
            p = self.state.get_current_pose()
            self.set_toast(f"已跳过至步骤 {self.state.current_pose_idx + 1}/{len(self.state.guide_poses)}: {p.title}")
        elif key in (ord("z"), ord("Z")) and self.state.active_mode == CameraHubState.MODE_LIVE:
            self.undo_last_capture()
        elif key in (ord("m"), ord("M")) and self.state.active_mode == CameraHubState.MODE_LIVE:
            new_facing = self.state.toggle_camera_facing()
            desc = "前置（左右镜像预览）" if new_facing == "front" else "后置（正常预览）"
            self.set_toast(f"已切换为: {desc}")
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
