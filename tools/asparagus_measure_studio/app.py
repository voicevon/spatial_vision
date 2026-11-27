#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单根芦笋测量工作室 (Single Asparagus Measure Studio) - 主应用控制器
==================================================================
支持离线样本与实时相机双模式、工位 3D ROI 投影与手动画框 2D ROI 双源配置，
实时输出绿色有效长度 (mm) 与白绿分界线向头部偏移 20 mm 处直径 (mm)。
"""

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.control.grading_engine import GradeDecision, GradingEngine
from src.devices.camera_service import CameraService
from src.ui.base_cv_app import BaseCvApp
from src.ui.canvas_viewport_interactor import CanvasViewportInteractor
from src.utils.image_io import imread_unicode, imwrite_unicode
from src.utils.logger import get_logger
from src.vision.planar_rectifier import PlanarRectifier, PlanarRigConfig
from src.vision.single_asparagus_measurer import (
    MeasureConfig,
    ORIENTATION_OPTIONS,
    SingleAsparagusMeasurement,
    SingleAsparagusMeasurer,
)
from src.workspace.coordinate_manager import CoordinateTreeManager
from src.workspace.roi_manager import RoiSpaceManager
from src.workspace.workspace_manager import WorkspaceManager
from tools.asparagus_measure_studio.data_io import (
    APP_ID,
    BASE_H,
    BASE_W,
    BatchEvalSummary,
    DEFAULT_DIR,
    SORTER_STATION_CONFIG_PATH,
    WINDOW_KEY,
    load_camera_intrinsics_cfg,
    load_ground_truth_csv,
    load_measure_settings,
    load_sorter_station_yaml,
    project_workspace_roi_to_2d,
    run_batch_evaluation,
    save_measure_config_to_station_yaml,
    save_measure_settings,
    scan_samples,
)
from tools.asparagus_measure_studio.renderer import AsparagusMeasureStudioRenderer

log = get_logger(__name__)


class AsparagusMeasureStudioApp(BaseCvApp):
    """单根芦笋测量工作室 GUI 主应用 (基于 BaseCvApp)"""

    def __init__(
        self,
        sample_dir: Optional[str] = None,
        settings_file: Optional[str] = None,
        camera_service: Optional[CameraService] = None,
        station_config_path: Optional[str] = None,
    ):
        super().__init__(
            app_id=APP_ID,
            base_w=BASE_W,
            base_h=BASE_H,
            window_name=WINDOW_KEY,
            window_title="单根芦笋测量工作室 - Single Asparagus Measure Studio",
            settings_file=settings_file,
            min_w=900,
            min_h=600,
            enable_keyboard_zoom=True,
            responsive=True,
        )
        self.settings_file = settings_file
        self.intrinsics_cfg = load_camera_intrinsics_cfg()
        self.live_intrinsics: Optional[Tuple[float, float, float, float]] = None

        # 工位与空间资产管理
        self.workspace_mgr = WorkspaceManager()
        cur_ws = self.workspace_mgr.get_current_workspace()
        self.current_workspace_id = cur_ws.workspace_id if cur_ws else ""
        self.sample_dir = sample_dir if sample_dir else (cur_ws.prod_raw_images_dir if cur_ws else DEFAULT_DIR)

        self.tag_localizer = None
        self.coord_mgr: Optional[CoordinateTreeManager] = None
        self.roi_mgr: Optional[RoiSpaceManager] = None
        self._load_workspace_assets()

        # 持久化偏好加载
        self._persisted = load_measure_settings(self.settings_file)
        cfg = MeasureConfig(
            orientation=str(self._persisted.get("orientation", "left_to_right")),
            diameter_offset_mm=float(self._persisted.get("diameter_offset_mm", 20.0)),
            green_h_low=int(self._persisted.get("green_h_low", 25)),
            green_h_high=int(self._persisted.get("green_h_high", 95)),
            white_s_max=int(self._persisted.get("white_s_max", 65)),
            white_v_min=int(self._persisted.get("white_v_min", 135)),
            fallback_mm_per_px=float(self._persisted.get("fallback_mm_per_px", 0.25)),
        )
        self.measurer = SingleAsparagusMeasurer(config=cfg)

        # 相机服务（严格委托 CameraService，严禁在工具层直接调用 pyrealsense2）
        self.camera_service = camera_service if camera_service is not None else CameraService(
            intrinsics_callback=self._on_camera_intrinsics
        )

        # 运行模式与 ROI 状态
        self.station_config_path: str = station_config_path or SORTER_STATION_CONFIG_PATH
        self.use_planar_rectify: bool = bool(self._persisted.get("use_planar_rectify", False))
        self.rectifier: Optional[PlanarRectifier] = None
        self.grading_engine: Optional[GradingEngine] = None
        self.grade_decision: Optional[GradeDecision] = None
        self.ground_truth: Dict[str, Tuple[float, float]] = {}
        self.batch_eval: Optional[BatchEvalSummary] = None
        self._load_station_assets()

        self.source_mode: str = "offline"     # "offline" | "live"
        self.selected_roi_id: str = str(self._persisted.get("selected_roi_id", ""))
        saved_manual = self._persisted.get("manual_roi")
        self.manual_roi: Optional[Tuple[int, int, int, int]] = (
            tuple(int(v) for v in saved_manual) if (isinstance(saved_manual, list) and len(saved_manual) == 4) else None
        )
        self.is_drawing_roi: bool = False
        self.drag_roi_start: Optional[Tuple[int, int]] = None
        self.drag_roi_curr: Optional[Tuple[int, int]] = None

        # 图像与测量结果
        self.samples: List[Dict[str, Any]] = []
        self.sel_idx: int = -1
        self.scroll_off: int = 0
        self.current_color: Optional[np.ndarray] = None
        self.current_depth: Optional[np.ndarray] = None
        self.measurement: Optional[SingleAsparagusMeasurement] = None
        self.active_step_key: str = "stage4_result"
        self.error_msg: str = ""

        # 视口与交互状态
        self.viewport = CanvasViewportInteractor(top_bar_h=88, bottom_bar_h=34, win_w=BASE_W, win_h=BASE_H)
        self.active_dropdown: Optional[str] = None
        self.mouse_pos: Tuple[int, int] = (-1, -1)
        self._buttons: List[Tuple[str, Tuple[int, int, int, int], Any]] = []
        self._sample_rows: List[Tuple[int, Tuple[int, int, int, int]]] = []
        self._dd_items: List[Tuple[str, Tuple[int, int, int, int], Any]] = []
        self._slider_bars: List[Tuple[Tuple[int, int, int, int], Dict[str, Any]]] = []
        self._drag_slider: Optional[Tuple[Tuple[int, int, int, int], Dict[str, Any]]] = None

        self.rescan(auto_load=True)

    # ------------------------------ 属性与配置 ------------------------------

    @property
    def toast_text(self) -> str:
        return self._toast if time.time() < self._toast_until else ""

    @property
    def current_workspace_name(self) -> str:
        ws = self.workspace_mgr.get_workspace_by_id(self.current_workspace_id)
        return ws.name if ws else "默认工位"

    @property
    def workspace_options(self) -> List[Tuple[str, str]]:
        return [(s.workspace_id, s.name) for s in self.workspace_mgr.list_workspaces()]

    @property
    def roi_options(self) -> List[Tuple[str, str]]:
        opts: List[Tuple[str, str]] = [("", "自动/全图")]
        if self.roi_mgr:
            for r in self.roi_mgr.list_rois():
                if r.enabled:
                    opts.append((r.roi_id, f"{r.name or r.roi_id}"))
        return opts

    @property
    def current_roi_label(self) -> str:
        if self.manual_roi is not None:
            return "手动2D框"
        if self.selected_roi_id and self.roi_mgr:
            r = self.roi_mgr.get_roi(self.selected_roi_id)
            if r:
                return r.name or r.roi_id
        return "自动/全图"

    def _save_state(self):
        cfg = self.measurer.config
        save_measure_settings({
            "orientation": cfg.orientation,
            "diameter_offset_mm": cfg.diameter_offset_mm,
            "green_h_low": cfg.green_h_low,
            "green_h_high": cfg.green_h_high,
            "white_s_max": cfg.white_s_max,
            "white_v_min": cfg.white_v_min,
            "fallback_mm_per_px": cfg.fallback_mm_per_px,
            "selected_roi_id": self.selected_roi_id,
            "manual_roi": list(self.manual_roi) if self.manual_roi is not None else None,
            "use_planar_rectify": self.use_planar_rectify,
        }, settings_file=self.settings_file)

    def _load_station_assets(self):
        """从 sorter_station.yaml 加载机架 4 标靶单应性拉正器与分级规则引擎"""
        station_cfg = load_sorter_station_yaml(self.station_config_path)
        rig = station_cfg.get("rig") or {}
        raw_tags = rig.get("fixed_tags_mm") or {}
        if raw_tags:
            fixed_tags = {int(k): (float(v[0]), float(v[1])) for k, v in raw_tags.items()}
            roi = rig.get("roi_mm") or [20.0, 15.0, 300.0, 105.0]
            rig_cfg = PlanarRigConfig(
                tag_size_mm=float(rig.get("tag_size_mm", 30.0)),
                fixed_tags_mm=fixed_tags,
                roi_mm=(float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3])),
                px_per_mm=float(rig.get("px_per_mm", 4.0)),
                max_reproj_err_px=float(rig.get("max_reproj_err_px", 1.5)),
                shift_threshold_px=float(rig.get("shift_threshold_px", 1.0)),
                use_tag_corners=bool(rig.get("use_tag_corners", False)),
            )
            self.rectifier = PlanarRectifier(rig_cfg)
        else:
            self.rectifier = None

        grading_cfg = station_cfg.get("grading")
        self.grading_engine = GradingEngine.from_config(grading_cfg) if grading_cfg else None

    @property
    def current_sample_gt(self) -> Optional[Tuple[float, float]]:
        if self.source_mode != "offline" or not (0 <= self.sel_idx < len(self.samples)):
            return None
        return self.ground_truth.get(self.samples[self.sel_idx]["name"])

    def _on_camera_intrinsics(self, K: np.ndarray, dist: np.ndarray):
        try:
            self.live_intrinsics = (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
        except Exception:
            pass

    def _load_workspace_assets(self):
        """装载当前工位的 TagLocalizer、坐标系树与 3D ROI 集合"""
        ws = self.workspace_mgr.get_workspace_by_id(self.current_workspace_id)
        self.tag_localizer = None
        self.coord_mgr = None
        self.roi_mgr = None
        if not ws:
            return

        if os.path.exists(ws.spatial_scene_path):
            try:
                self.coord_mgr = CoordinateTreeManager(ws.workspace_id, ws.spatial_scene_path)
                self.roi_mgr = RoiSpaceManager(ws.workspace_id, ws.spatial_scene_path)
            except Exception as exc:
                log.warning("加载工位空间场景 ROI 失败: %s", exc)

        if ws.map_path and os.path.exists(ws.map_path) and os.path.getsize(ws.map_path) > 50:
            try:
                from src.vision.tag_localizer import TagLocalizer
                from src.workspace.workspace_manager import load_workspace_marker_size_mm
                marker_size_mm = load_workspace_marker_size_mm(ws.workspace_dir)
                if marker_size_mm is not None:
                    self.tag_localizer = TagLocalizer(
                        tags_map_path=ws.map_path,
                        marker_size_mm=marker_size_mm,
                    )
            except Exception as exc:
                log.warning("加载工位 AprilTag 定位器失败: %s", exc)

    def _get_intrinsics(self, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        if self.source_mode == "live" and self.live_intrinsics is not None:
            return self.live_intrinsics
        if self.intrinsics_cfg is not None:
            fx, fy, cx, cy, cw, ch = self.intrinsics_cfg
            if cw > 0 and ch > 0 and (cw != img_w or ch != img_h):
                return (fx * img_w / cw, fy * img_h / ch, cx * img_w / cw, cy * img_h / ch)
            return (fx, fy, cx, cy)
        return (909.12, 907.46, img_w / 2.0, img_h / 2.0)

    def _resolve_active_roi(self, color_bgr: np.ndarray, intrinsics: Tuple[float, float, float, float]) -> Optional[Tuple[int, int, int, int]]:
        """优先手动 2D ROI，其次工位 3D ROI 投影，否则全图"""
        if self.manual_roi is not None:
            return self.manual_roi
        if self.selected_roi_id and self.roi_mgr and self.coord_mgr and self.tag_localizer:
            try:
                ok, t_cam_to_world, _ = self.tag_localizer.localize_camera(color_bgr)
                if ok and t_cam_to_world is not None:
                    h, w = color_bgr.shape[:2]
                    proj = project_workspace_roi_to_2d(
                        self.selected_roi_id, self.roi_mgr, self.coord_mgr,
                        t_cam_to_world, intrinsics, w, h,
                    )
                    if proj is not None:
                        return proj
            except Exception:
                pass
        return None

    # ------------------------------ 数据源与测量 ------------------------------

    def switch_workspace(self, workspace_id: str):
        self.current_workspace_id = workspace_id
        self.workspace_mgr.set_current_workspace(workspace_id)
        ws = self.workspace_mgr.get_workspace_by_id(workspace_id)
        self.sample_dir = ws.prod_raw_images_dir if ws else DEFAULT_DIR
        self._load_workspace_assets()
        self.selected_roi_id = ""
        self.rescan(auto_load=(self.source_mode == "offline"))
        self.set_toast(f"已切换工位: {self.current_workspace_name}")

    def rescan(self, auto_load: bool = False):
        self.samples = scan_samples(self.sample_dir)
        self.ground_truth = load_ground_truth_csv(self.sample_dir)
        self.batch_eval = None
        self.sel_idx = -1
        self.scroll_off = 0
        if auto_load and self.samples:
            self._select_sample(0)
        elif self.source_mode == "offline" and not self.samples:
            self.current_color = None
            self.current_depth = None
            self.measurement = None
            self.grade_decision = None
            self.error_msg = f"样本目录为空: {self.sample_dir}"

    def _select_sample(self, idx: int):
        if not (0 <= idx < len(self.samples)):
            return
        self.sel_idx = idx
        smp = self.samples[idx]
        color = imread_unicode(smp["color_path"])
        if color is None:
            self.error_msg = f"无法读取图像: {smp['name']}"
            self.current_color = None
            self.measurement = None
            return
        depth = None
        if smp["depth_path"] and os.path.exists(smp["depth_path"]):
            try:
                depth = np.load(smp["depth_path"]).astype(np.float32)
            except Exception as exc:
                log.warning("读取深度图失败 (%s): %s", smp["depth_path"], exc)
        self.current_color = color
        self.current_depth = depth
        self.error_msg = ""
        self.run_measurement()

    def _measure_single_frame(
        self,
        color_bgr: np.ndarray,
        depth_mm: Optional[np.ndarray] = None,
    ) -> Tuple[SingleAsparagusMeasurement, Optional[GradeDecision]]:
        """执行单帧测量与分级（统一供当前帧展示与离线批量评测调用）"""
        if self.use_planar_rectify and self.rectifier is not None:
            st = self.rectifier.update(color_bgr, force_recompute=True)
            if not st.locked:
                meas = SingleAsparagusMeasurement(valid=False, reason=f"4标靶拉正失锁: {st.reason}")
                dec = (
                    self.grading_engine.evaluate(valid=False, effective_length_mm=0.0, diameter_mm=0.0)
                    if self.grading_engine is not None else None
                )
                return meas, dec
            warped = self.rectifier.warp_roi(color_bgr)
            if warped is None:
                meas = SingleAsparagusMeasurement(valid=False, reason="4标靶拉正输出为空")
                dec = (
                    self.grading_engine.evaluate(valid=False, effective_length_mm=0.0, diameter_mm=0.0)
                    if self.grading_engine is not None else None
                )
                return meas, dec
            saved_scale = self.measurer.config.fallback_mm_per_px
            try:
                self.measurer.config.fallback_mm_per_px = self.rectifier.mm_per_px
                meas = self.measurer.measure(
                    color_bgr=warped,
                    roi_rect=self.manual_roi,
                    depth_mm=None,
                    intrinsics=None,
                    tag_localizer=None,
                )
                if meas.valid:
                    meas.scale_source = "planar_homography"
            finally:
                self.measurer.config.fallback_mm_per_px = saved_scale
        else:
            h, w = color_bgr.shape[:2]
            intr = self._get_intrinsics(w, h)
            roi_rect = self._resolve_active_roi(color_bgr, intr)
            meas = self.measurer.measure(
                color_bgr=color_bgr,
                roi_rect=roi_rect,
                depth_mm=depth_mm,
                intrinsics=intr,
                tag_localizer=self.tag_localizer,
            )

        dec = None
        if self.grading_engine is not None:
            dec = self.grading_engine.evaluate(
                valid=meas.valid,
                effective_length_mm=meas.effective_length_mm,
                diameter_mm=meas.diameter_mm,
                has_white_root=meas.has_white_root,
            )
        return meas, dec

    def run_measurement(self):
        if self.current_color is None:
            self.measurement = None
            self.grade_decision = None
            return
        self.measurement, self.grade_decision = self._measure_single_frame(
            self.current_color, self.current_depth
        )

    def toggle_planar_rectify(self):
        if self.rectifier is None:
            self.set_toast("未找到 sorter_station.yaml 中的 rig.fixed_tags_mm 配置")
            return
        self.use_planar_rectify = not self.use_planar_rectify
        self._save_state()
        self.run_measurement()
        self.set_toast("已启用机架 4 标靶单应性拉正" if self.use_planar_rectify else "已关闭 4 标靶单应性拉正")

    def export_to_station_yaml(self) -> bool:
        ok = save_measure_config_to_station_yaml(self.measurer.config, self.station_config_path)
        if ok:
            self.set_toast(f"已导出测量参数至 {os.path.basename(self.station_config_path)}", duration=3.5)
        else:
            self.set_toast("导出产线 YAML 失败，请检查日志", duration=3.5)
        return ok

    def run_offline_batch_eval(self) -> Optional[BatchEvalSummary]:
        if not self.samples:
            self.set_toast("当前目录无离线样本可评测")
            return None
        self.ground_truth = load_ground_truth_csv(self.sample_dir)
        summary = run_batch_evaluation(
            samples=self.samples,
            measure_fn=self._measure_single_frame,
            ground_truth=self.ground_truth,
        )
        self.batch_eval = summary
        if summary.gt_matched_count > 0:
            self.set_toast(
                f"批量评测完成: 有效 {summary.valid_count}/{summary.total} | "
                f"真值({summary.gt_matched_count}) L-MAE={summary.len_mae_mm:.2f}mm D-MAE={summary.diam_mae_mm:.2f}mm",
                duration=5.0,
            )
        else:
            self.set_toast(
                f"批量评测完成: 有效 {summary.valid_count}/{summary.total} (未发现 ground_truth.csv)",
                duration=4.0,
            )
        return summary

    def toggle_source_mode(self):
        """切换离线样本 / 实时相机模式"""
        if self.source_mode == "offline":
            if self._start_live_camera():
                self.source_mode = "live"
                self.error_msg = ""
                self.set_toast(f"已进入实时相机模式 ({self.camera_service.stream_desc})")
        else:
            self.camera_service.stop()
            self.source_mode = "offline"
            self.set_toast("已切回离线样本模式")
            if self.samples:
                self._select_sample(max(0, self.sel_idx))
            else:
                self.rescan(auto_load=True)

    def _start_live_camera(self) -> bool:
        ws = self.workspace_mgr.get_workspace_by_id(self.current_workspace_id)
        cam_type = (getattr(ws, "camera_type", None) or "realsense").strip().lower()
        w, h = ws.resolution if ws else (1280, 720)
        serial = ws.camera_serial if ws else None
        try:
            if cam_type == "usb":
                self.camera_service.start_usb(w, h, fps=30, device_index=serial or 0)
            else:
                self.camera_service.start_realsense(
                    w, h, fps=30, serial=serial,
                    fallbacks=((1280, 720, 30), (640, 480, 30)),
                    enable_depth=True,
                )
            return True
        except Exception as cam_err:
            log.warning("实时相机启动失败 (%s): %s", cam_type, cam_err)
            self.error_msg = f"实时相机启动失败: {cam_err}"
            self.set_toast(self.error_msg, duration=4.0)
            return False

    def capture_live_snapshot(self):
        if self.current_color is None:
            self.set_toast("当前无实时画面可抓拍")
            return
        os.makedirs(self.sample_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        color_path = os.path.join(self.sample_dir, f"measure_{ts}.png")
        ok = imwrite_unicode(color_path, self.current_color)
        if ok and self.current_depth is not None:
            np.save(os.path.join(self.sample_dir, f"measure_{ts}_depth.npy"), self.current_depth)
        self.set_toast(f"已保存抓拍样本: measure_{ts}.png" if ok else "抓拍保存失败")

    def cycle_orientation(self):
        keys = [k for k, _ in ORIENTATION_OPTIONS]
        cur = self.measurer.config.orientation
        nxt = keys[(keys.index(cur) + 1) % len(keys)] if cur in keys else keys[0]
        self.measurer.config.orientation = nxt
        self._save_state()
        self.run_measurement()
        self.set_toast(f"朝向已切换: {dict(ORIENTATION_OPTIONS)[nxt]}")

    def get_active_display_image(self) -> Optional[np.ndarray]:
        if self.measurement is not None:
            snap = self.measurement.step_snapshots.get(self.active_step_key)
            if snap is not None:
                return snap
        return self.current_color

    # ------------------------------ 视口坐标逆映射 ------------------------------

    def _canvas_to_image_xy(self, cx: int, cy: int) -> Optional[Tuple[int, int]]:
        vis = self.get_active_display_image()
        if vis is None or vis.size == 0:
            return None
        fh, fw = vis.shape[:2]
        m = AsparagusMeasureStudioRenderer.compute_metrics(self.win_mgr.canvas_w)
        _, img_p, _ = AsparagusMeasureStudioRenderer.compute_panels(self.win_mgr.canvas_w, self.win_mgr.canvas_h, m)
        vx, vy, vw, vh = img_p[0], img_p[1], img_p[2] - img_p[0], img_p[3] - img_p[1]
        ix1, iy1, ix2, iy2, _ = self.viewport.compute_image_rect((vx, vy, vw, vh), fw, fh)
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        px = int(round((cx - ix1) * fw / float(ix2 - ix1)))
        py = int(round((cy - iy1) * fh / float(iy2 - iy1)))
        px = max(0, min(fw - 1, px))
        py = max(0, min(fh - 1, py))
        return (px, py)

    def _update_slider_from_x(self, bar: Tuple[int, int, int, int], spec: Dict[str, Any], mx: int):
        x1, _, x2, _ = bar
        ratio = max(0.0, min(1.0, (mx - x1) / max(1.0, float(x2 - x1))))
        raw = spec["min"] + ratio * (spec["max"] - spec["min"])
        step = spec["step"]
        val = round(raw / step) * step
        if isinstance(spec["min"], int) and isinstance(spec["step"], int):
            val = int(round(val))
        else:
            val = round(float(val), 3)
        setattr(self.measurer.config, spec["key"], val)
        self.run_measurement()

    # ------------------------------ 事件处理 ------------------------------

    def on_tick(self):
        if self.source_mode == "live" and self.camera_service.is_running:
            bgr, depth = self.camera_service.read_rgbd_frame(timeout_ms=200)
            if bgr is not None:
                self.current_color = bgr
                self.current_depth = depth
                self.run_measurement()

    def on_mouse_down(self, x: int, y: int, button: str):
        self.mouse_pos = (x, y)
        if button == "right":
            self.viewport.start_pan(x, y)
            return
        if button != "left":
            return

        # 1. 下拉菜单优先命中
        if self.active_dropdown:
            for act_id, rect, payload in self._dd_items:
                if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                    self.active_dropdown = None
                    if act_id == "SELECT_WS":
                        self.switch_workspace(payload)
                    elif act_id == "SELECT_ROI":
                        self.selected_roi_id = payload
                        self.manual_roi = None
                        self._save_state()
                        self.run_measurement()
                        self.set_toast(f"ROI 已切换: {self.current_roi_label}")
                    return
            self.active_dropdown = None

        # 2. 顶部与工具栏按钮
        for btn_id, rect, payload in self._buttons:
            if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                self._handle_button(btn_id, payload)
                return

        # 3. 左侧样本列表
        for idx, rect in self._sample_rows:
            if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                self._select_sample(idx)
                return

        # 4. 右侧参数滑条
        for bar, spec in self._slider_bars:
            if bar[0] - 6 <= x <= bar[2] + 6 and bar[1] - 4 <= y <= bar[3] + 4:
                self._drag_slider = (bar, spec)
                self._update_slider_from_x(bar, spec, x)
                return

        # 5. 中栏手动画框 ROI
        if self.is_drawing_roi:
            m = AsparagusMeasureStudioRenderer.compute_metrics(self.win_mgr.canvas_w)
            _, img_p, _ = AsparagusMeasureStudioRenderer.compute_panels(self.win_mgr.canvas_w, self.win_mgr.canvas_h, m)
            if img_p[0] <= x <= img_p[2] and img_p[1] <= y <= img_p[3]:
                self.drag_roi_start = (x, y)
                self.drag_roi_curr = (x, y)
                return

    def on_mouse_move(self, x: int, y: int):
        self.mouse_pos = (x, y)
        if self.viewport.is_panning:
            self.viewport.update_pan(x, y)
            return
        if self._drag_slider is not None:
            self._update_slider_from_x(self._drag_slider[0], self._drag_slider[1], x)
            return
        if self.is_drawing_roi and self.drag_roi_start is not None:
            self.drag_roi_curr = (x, y)

    def on_mouse_up(self, x: int, y: int, button: str):
        if button == "right":
            self.viewport.end_pan()
            return
        if self._drag_slider is not None:
            self._drag_slider = None
            self._save_state()
            return
        if self.is_drawing_roi and self.drag_roi_start is not None:
            p1 = self._canvas_to_image_xy(*self.drag_roi_start)
            p2 = self._canvas_to_image_xy(x, y)
            self.drag_roi_start = None
            self.drag_roi_curr = None
            self.is_drawing_roi = False
            if p1 is not None and p2 is not None:
                rx = min(p1[0], p2[0])
                ry = min(p1[1], p2[1])
                rw = abs(p2[0] - p1[0])
                rh = abs(p2[1] - p1[1])
                if rw >= 16 and rh >= 16:
                    self.manual_roi = (rx, ry, rw, rh)
                    self._save_state()
                    self.run_measurement()
                    self.set_toast(f"已设定手动 ROI: ({rx}, {ry}, {rw}x{rh})")
                else:
                    self.set_toast("框选区域过小，已忽略")

    def on_mouse_wheel(self, delta: int, flags: int):
        x, y = self.mouse_pos
        m = AsparagusMeasureStudioRenderer.compute_metrics(self.win_mgr.canvas_w)
        left_p, img_p, _ = AsparagusMeasureStudioRenderer.compute_panels(self.win_mgr.canvas_w, self.win_mgr.canvas_h, m)
        if left_p[0] <= x <= left_p[2] and left_p[1] <= y <= left_p[3] and self.source_mode == "offline":
            step = -1 if delta > 0 else 1
            self.scroll_off = max(0, min(max(0, len(self.samples) - 1), self.scroll_off + step))
        elif img_p[0] <= x <= img_p[2] and img_p[1] <= y <= img_p[3]:
            vp = (img_p[0], img_p[1], img_p[2] - img_p[0], img_p[3] - img_p[1])
            self.viewport.zoom_at(x, y, wheel_up=(delta > 0), viewport_rect=vp)

    def _handle_button(self, btn_id: str, payload: Any):
        if btn_id == "EXIT":
            self.stop()
        elif btn_id == "TOGGLE_WS_DD":
            self.active_dropdown = None if self.active_dropdown == "workspace" else "workspace"
        elif btn_id == "TOGGLE_ROI_DD":
            self.active_dropdown = None if self.active_dropdown == "roi" else "roi"
        elif btn_id == "TOGGLE_SOURCE_MODE":
            self.toggle_source_mode()
        elif btn_id == "TOGGLE_DRAW_ROI":
            self.is_drawing_roi = not self.is_drawing_roi
            self.drag_roi_start = None
            self.drag_roi_curr = None
            self.set_toast("请在主画面按住左键拖拽框选单根芦笋 ROI" if self.is_drawing_roi else "已取消手动画框")
        elif btn_id == "CLEAR_MANUAL_ROI":
            self.manual_roi = None
            self._save_state()
            self.run_measurement()
            self.set_toast("已清除手动 ROI")
        elif btn_id == "SELECT_STEP":
            self.active_step_key = str(payload)
        elif btn_id == "CYCLE_ORIENTATION":
            self.cycle_orientation()
        elif btn_id == "RESET_VIEW":
            self.viewport.reset()
        elif btn_id == "RESCAN":
            self.rescan(auto_load=True)
            self.set_toast(f"已刷新样本 ({len(self.samples)} 帧)")
        elif btn_id == "CAPTURE_LIVE":
            self.capture_live_snapshot()
        elif btn_id == "TOGGLE_RECTIFY":
            self.toggle_planar_rectify()
        elif btn_id == "RUN_BATCH_EVAL":
            self.run_offline_batch_eval()
        elif btn_id == "EXPORT_STATION_YAML":
            self.export_to_station_yaml()

    def on_key(self, raw_key: int) -> bool:
        if raw_key == 27 and self.is_drawing_roi:
            self.is_drawing_roi = False
            self.drag_roi_start = None
            self.drag_roi_curr = None
            return True
        if raw_key in (ord('m'), ord('M')):
            self.toggle_source_mode()
            return True
        if raw_key in (ord('d'), ord('D')):
            self._handle_button("TOGGLE_DRAW_ROI", None)
            return True
        if raw_key in (ord('o'), ord('O')):
            self.cycle_orientation()
            return True
        if raw_key in (ord('p'), ord('P')):
            self.toggle_planar_rectify()
            return True
        if raw_key in (ord('b'), ord('B')) and self.source_mode == "offline":
            self.run_offline_batch_eval()
            return True
        if raw_key in (ord('s'), ord('S')):
            self.export_to_station_yaml()
            return True
        if raw_key == ord(' ') and self.source_mode == "live":
            self.capture_live_snapshot()
            return True
        if raw_key in (2490368, 65362, 38) and self.source_mode == "offline":
            if self.sel_idx > 0:
                self._select_sample(self.sel_idx - 1)
            return True
        if raw_key in (2621440, 65364, 40) and self.source_mode == "offline":
            if self.sel_idx < len(self.samples) - 1:
                self._select_sample(self.sel_idx + 1)
            return True
        return False

    def setup(self):
        log.info("单根芦笋测量工作室 GUI 已启动: %s (%d 个样本)", self.sample_dir, len(self.samples))

    def cleanup(self):
        self.camera_service.stop()
        self._save_state()
        log.info("单根芦笋测量工作室 GUI 已安全退出")

    def render(self) -> np.ndarray:
        W, H = self.win_mgr.canvas_w, self.win_mgr.canvas_h
        canvas = np.full((H, W, 3), self.COLOR_BG, dtype=np.uint8)
        self._buttons, self._sample_rows, self._dd_items, self._slider_bars = (
            AsparagusMeasureStudioRenderer.render_scene(canvas, self)
        )
        return canvas


def main():
    parser = argparse.ArgumentParser(description="单根芦笋测量工作室 GUI (Single Asparagus Measure Studio)")
    parser.add_argument("--dir", type=str, default=None, help="离线样本目录 (默认当前工位 production/raw_images/)")
    args = parser.parse_args()
    app = AsparagusMeasureStudioApp(sample_dir=args.dir)
    app.run()


if __name__ == "__main__":
    main()
